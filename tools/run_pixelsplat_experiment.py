import argparse
import json
import math
import os
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("[pixelsplat] $", " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    tail: list[str] = []
    for line in p.stdout:
        text = line.rstrip("\n")
        tail.append(text)
        if len(tail) > 120:
            tail.pop(0)
        print(text)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(
            f"command failed (rc={rc}): {' '.join(cmd)}\n"
            f"----- stdout tail -----\n{os.linesep.join(tail)}"
        )
    print(f"[pixelsplat] [ok] {time.time() - t0:.1f}s")


def _find_latest_ckpt(train_hydra_dir: Path) -> Path:
    ckpts = sorted(
        train_hydra_dir.glob("checkpoints/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint found under {train_hydra_dir}/checkpoints")
    return ckpts[-1]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _decode_image_bytes(byte_tensor: torch.Tensor) -> np.ndarray:
    arr = byte_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
    img = Image.open(BytesIO(arr.tobytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_image_path(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _to_float01(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def _metric_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    return float(-10.0 * math.log10(max(mse, 1e-10)))


def _metric_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _metric_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        structural_similarity(
            gt,
            pred,
            data_range=1.0,
            channel_axis=2,
        )
    )


def _build_scene_map(dataset_root: Path, stage: str) -> tuple[dict[str, str], Path]:
    stage_dir = dataset_root / stage
    index = _load_json(stage_dir / "index.json")
    mapping = {str(k): str(v) for k, v in index.items()}
    return mapping, stage_dir


def _extract_scene_example(stage_dir: Path, scene_key: str, chunk_name: str) -> dict[str, Any]:
    chunk = torch.load(stage_dir / chunk_name, map_location="cpu")
    rows = [x for x in chunk if str(x.get("key", "")) == scene_key]
    if len(rows) != 1:
        raise RuntimeError(f"scene lookup mismatch: scene={scene_key}, found={len(rows)}")
    return rows[0]


def evaluate_outputs(
    dataset_root: Path,
    eval_index_path: Path,
    pred_root: Path,
    stage: str,
) -> dict[str, Any]:
    idx = _load_json(eval_index_path)
    scene_to_chunk, stage_dir = _build_scene_map(dataset_root, stage=stage)

    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    scene_cache: dict[str, dict[str, Any]] = {}
    missing_scene = 0
    missing_pred = 0
    skipped_none = 0
    compared = 0
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    l1_vals: list[float] = []

    for scene_key, entry in idx.items():
        if entry is None:
            skipped_none += 1
            continue
        if scene_key not in scene_to_chunk:
            missing_scene += 1
            continue

        chunk_name = scene_to_chunk[scene_key]
        if chunk_name not in chunk_cache:
            chunk_cache[chunk_name] = torch.load(stage_dir / chunk_name, map_location="cpu")
        if scene_key not in scene_cache:
            rows = [x for x in chunk_cache[chunk_name] if str(x.get("key", "")) == scene_key]
            if len(rows) != 1:
                missing_scene += 1
                continue
            scene_cache[scene_key] = rows[0]
        ex = scene_cache[scene_key]
        images = ex["images"]

        targets = [int(x) for x in entry.get("target", [])]
        for t in targets:
            if t < 0 or t >= len(images):
                continue
            pred_path = pred_root / scene_key / "color" / f"{t:06d}.png"
            if not pred_path.exists():
                missing_pred += 1
                continue

            gt_u8 = _decode_image_bytes(images[t])
            pred_u8 = _decode_image_path(pred_path)
            if pred_u8.shape[:2] != gt_u8.shape[:2]:
                pred_u8 = np.asarray(
                    Image.fromarray(pred_u8).resize(
                        (gt_u8.shape[1], gt_u8.shape[0]), Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )

            gt = _to_float01(gt_u8)
            pred = _to_float01(pred_u8)

            psnr_vals.append(_metric_psnr(pred, gt))
            ssim_vals.append(_metric_ssim(pred, gt))
            l1_vals.append(_metric_l1(pred, gt))
            compared += 1

    ok = compared > 0
    return {
        "ok": ok,
        "stage": stage,
        "eval_index_path": str(eval_index_path),
        "pred_root": str(pred_root),
        "compared_targets": compared,
        "missing_scene": missing_scene,
        "missing_pred": missing_pred,
        "skipped_none": skipped_none,
        "mean_psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "mean_ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "mean_l1": float(np.mean(l1_vals)) if l1_vals else 0.0,
    }


def _train(
    repo_root: Path,
    dataset_root: Path,
    run_root: Path,
    max_steps: int,
    val_check_interval: int,
    checkpoint_every: int,
    batch_size: int,
    num_workers_train: int,
    num_workers_val: int,
    num_context_views: int,
) -> Path:
    train_hydra_dir = run_root / "hydra_train"
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    cmd = [
        "python",
        "-m",
        "src.main",
        "+experiment=re10k",
        f"dataset.roots=[{str(dataset_root)}]",
        f"dataset.view_sampler.num_context_views={int(num_context_views)}",
        f"data_loader.train.batch_size={int(batch_size)}",
        f"data_loader.train.num_workers={int(num_workers_train)}",
        "data_loader.train.persistent_workers=false",
        "data_loader.val.batch_size=1",
        f"data_loader.val.num_workers={int(num_workers_val)}",
        "data_loader.val.persistent_workers=false",
        "data_loader.test.batch_size=1",
        "data_loader.test.num_workers=1",
        "data_loader.test.persistent_workers=false",
        f"trainer.max_steps={int(max_steps)}",
        f"trainer.val_check_interval={int(val_check_interval)}",
        f"checkpointing.every_n_train_steps={int(checkpoint_every)}",
        "checkpointing.save_top_k=1",
        "wandb.mode=disabled",
        "wandb.name=zju_phase2_train",
        f"hydra.run.dir={str(train_hydra_dir)}",
    ]
    _run_cmd(cmd, cwd=repo_root, env=env)
    return _find_latest_ckpt(train_hydra_dir)


def _test_once(
    repo_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    eval_index_path: Path,
    run_root: Path,
    name: str,
    num_workers_test: int,
    num_context_views: int,
) -> Path:
    hydra_dir = run_root / f"hydra_test_{name}"
    out_root = run_root / "test_outputs"
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    cmd = [
        "python",
        "-m",
        "src.main",
        "+experiment=re10k",
        "mode=test",
        f"dataset.roots=[{str(dataset_root)}]",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={str(eval_index_path)}",
        f"dataset.view_sampler.num_context_views={int(num_context_views)}",
        f"checkpointing.load={str(checkpoint_path)}",
        f"test.output_path={str(out_root)}",
        "wandb.mode=disabled",
        f"wandb.name={name}",
        "data_loader.test.batch_size=1",
        f"data_loader.test.num_workers={int(num_workers_test)}",
        "data_loader.test.persistent_workers=false",
        f"hydra.run.dir={str(hydra_dir)}",
    ]
    _run_cmd(cmd, cwd=repo_root, env=env)
    return out_root / name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/mnt/code/external/pixelsplat")
    ap.add_argument("--dataset_root", type=str, default="/mnt/out/pixelsplat_datasets/zju_phase2_300_6v")
    ap.add_argument("--eval_index_uniform", type=str, default="/mnt/out/pixelsplat_assets/evaluation_index_zju_uniform_6v.json")
    ap.add_argument("--eval_index_random", type=str, default="/mnt/out/pixelsplat_assets/evaluation_index_zju_random_6v.json")
    ap.add_argument("--output_root", type=str, default="/mnt/out/pixelsplat_runs")
    ap.add_argument("--run_tag", type=str, default="")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--val_check_interval", type=int, default=100)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers_train", type=int, default=4)
    ap.add_argument("--num_workers_val", type=int, default=1)
    ap.add_argument("--num_workers_test", type=int, default=2)
    ap.add_argument("--num_context_views", type=int, default=2)
    ap.add_argument("--stage", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--checkpoint_load", type=str, default="")
    ap.add_argument("--eval_mode", type=str, default="both", choices=["both", "uniform", "random"])
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    eval_uniform = Path(args.eval_index_uniform).resolve()
    eval_random = Path(args.eval_index_random).resolve()
    output_root = Path(args.output_root).resolve()
    tag = str(args.run_tag).strip() or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_root / tag
    run_root.mkdir(parents=True, exist_ok=True)

    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not eval_uniform.exists():
        raise FileNotFoundError(f"eval_index_uniform not found: {eval_uniform}")
    if not eval_random.exists():
        raise FileNotFoundError(f"eval_index_random not found: {eval_random}")

    if args.skip_train:
        if not str(args.checkpoint_load).strip():
            raise ValueError("--skip_train requires --checkpoint_load")
        ckpt = Path(args.checkpoint_load).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint_load not found: {ckpt}")
    else:
        ckpt = _train(
            repo_root=repo_root,
            dataset_root=dataset_root,
            run_root=run_root,
            max_steps=int(args.max_steps),
            val_check_interval=int(args.val_check_interval),
            checkpoint_every=int(args.checkpoint_every),
            batch_size=int(args.batch_size),
            num_workers_train=int(args.num_workers_train),
            num_workers_val=int(args.num_workers_val),
            num_context_views=int(args.num_context_views),
        )

    pred_uniform = ""
    pred_random = ""
    metrics_uniform: dict[str, Any] | None = None
    metrics_random: dict[str, Any] | None = None

    if args.eval_mode in ("both", "uniform"):
        pred_uniform_path = _test_once(
            repo_root=repo_root,
            dataset_root=dataset_root,
            checkpoint_path=ckpt,
            eval_index_path=eval_uniform,
            run_root=run_root,
            name="uniform",
            num_workers_test=int(args.num_workers_test),
            num_context_views=int(args.num_context_views),
        )
        pred_uniform = str(pred_uniform_path)
        metrics_uniform = evaluate_outputs(
            dataset_root=dataset_root,
            eval_index_path=eval_uniform,
            pred_root=pred_uniform_path,
            stage=str(args.stage),
        )

    if args.eval_mode in ("both", "random"):
        pred_random_path = _test_once(
            repo_root=repo_root,
            dataset_root=dataset_root,
            checkpoint_path=ckpt,
            eval_index_path=eval_random,
            run_root=run_root,
            name="random",
            num_workers_test=int(args.num_workers_test),
            num_context_views=int(args.num_context_views),
        )
        pred_random = str(pred_random_path)
        metrics_random = evaluate_outputs(
            dataset_root=dataset_root,
            eval_index_path=eval_random,
            pred_root=pred_random_path,
            stage=str(args.stage),
        )

    ok_uniform = bool(metrics_uniform and metrics_uniform.get("ok"))
    ok_random = bool(metrics_random and metrics_random.get("ok"))
    ok = ok_uniform if args.eval_mode == "uniform" else ok_random if args.eval_mode == "random" else (ok_uniform and ok_random)

    delta_psnr = 0.0
    delta_ssim = 0.0
    delta_l1 = 0.0
    if metrics_uniform is not None and metrics_random is not None:
        delta_psnr = float(metrics_uniform["mean_psnr"] - metrics_random["mean_psnr"])
        delta_ssim = float(metrics_uniform["mean_ssim"] - metrics_random["mean_ssim"])
        delta_l1 = float(metrics_uniform["mean_l1"] - metrics_random["mean_l1"])

    summary = {
        "ok": ok,
        "run_tag": tag,
        "eval_mode": str(args.eval_mode),
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "checkpoint": str(ckpt),
        "pred_uniform": pred_uniform,
        "pred_random": pred_random,
        "uniform": metrics_uniform,
        "random": metrics_random,
        "delta_uniform_minus_random": {
            "psnr": delta_psnr,
            "ssim": delta_ssim,
            "l1": delta_l1,
        },
    }

    summary_path = run_root / "compare_summary.json"
    _save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if str(args.save_json).strip():
        _save_json(Path(args.save_json), summary)


if __name__ == "__main__":
    main()
