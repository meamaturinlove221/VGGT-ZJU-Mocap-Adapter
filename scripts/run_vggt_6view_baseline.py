#!/usr/bin/env python
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_CAMERAS = [
    "Camera_B1",
    "Camera_B5",
    "Camera_B10",
    "Camera_B14",
    "Camera_B19",
    "Camera_B23",
]


def _parse_csv(raw: str) -> list[str]:
    s = (raw or "").replace(";", ",").replace(" ", ",")
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_frame_ids(raw: str) -> list[int]:
    out: list[int] = []
    for token in _parse_csv(raw):
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("frame_ids is empty")
    return uniq


def _tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
    }


def _resolve_device(device_arg: str) -> str:
    raw = (device_arg or "auto").strip().lower()
    if raw in ("", "auto", "none"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def _probe_usable_cuda() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        x = torch.randn(1, 3, 8, 8, device="cuda")
        m = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).cuda()
        _ = m(x)
        return True, "ok"
    except Exception as e:  # pragma: no cover - hardware dependent
        return False, f"{type(e).__name__}: {e}"


def _autocast_context(device: str):
    if device.startswith("cuda"):
        cap_major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
        return torch.cuda.amp.autocast(dtype=amp_dtype), str(amp_dtype)
    return contextlib.nullcontext(), "none"


def _find_frame_image(seq_dir: Path, cam_name: str, frame_id: int) -> Path:
    cam_dir = seq_dir / cam_name
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera dir not found: {cam_dir}")
    stem = f"{frame_id:06d}"
    candidates = [
        cam_dir / f"{stem}.jpg",
        cam_dir / f"{stem}.png",
        cam_dir / f"{stem}.jpeg",
        cam_dir / f"{stem}.JPG",
        cam_dir / f"{stem}.PNG",
        cam_dir / f"{stem}.JPEG",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"frame image not found for {cam_name} frame={stem} in {cam_dir}")


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_file():
            p.unlink()


def _stage_fixed_6views(
    zju_root: Path,
    seq_name: str,
    frame_id: int,
    cameras: list[str],
    scene_dir: Path,
) -> list[Path]:
    seq_dir = zju_root / seq_name
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"sequence dir not found: {seq_dir}")

    image_dir = scene_dir / "images"
    _ensure_clean_dir(image_dir)

    staged_paths: list[Path] = []
    for idx, cam in enumerate(cameras):
        src = _find_frame_image(seq_dir, cam, frame_id)
        dst = image_dir / f"{idx:02d}_{cam}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        staged_paths.append(dst)
    return staged_paths


def _run_tensor_baseline(model: VGGT, device: str, image_paths: list[Path]) -> dict:
    t0 = time.time()
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    image_load_sec = time.time() - t0

    amp_ctx, amp_dtype = _autocast_context(device)
    t1 = time.time()
    with torch.no_grad():
        with amp_ctx:
            predictions = model(images)
    forward_sec = time.time() - t1

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    out = {
        "device_used": device,
        "amp_dtype": amp_dtype,
        "input_shape": list(images.shape),
        "image_load_sec": round(image_load_sec, 4),
        "forward_sec": round(forward_sec, 4),
        "prediction_keys": sorted(list(predictions.keys())),
        "tensor_stats": {},
        "decoded_camera": {
            "extrinsic_shape": list(extrinsic.shape),
            "intrinsic_shape": list(intrinsic.shape),
            "intrinsic_0_0": intrinsic[0, 0].detach().cpu().tolist(),
        },
        "npy_payload": {
            "extrinsic": extrinsic.detach().cpu().numpy(),
            "intrinsic": intrinsic.detach().cpu().numpy(),
        },
    }
    for k, v in predictions.items():
        if torch.is_tensor(v):
            out["tensor_stats"][k] = _tensor_stats(v)
    return out


def _run_demo_colmap(scene_dir: Path, python_bin: str, log_path: Path) -> dict:
    cmd = [python_bin, "demo_colmap.py", f"--scene_dir={scene_dir}"]
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n",
        encoding="utf-8",
    )

    sparse_dir = scene_dir / "sparse"
    outputs = {
        "cameras_bin": (sparse_dir / "cameras.bin").exists(),
        "images_bin": (sparse_dir / "images.bin").exists(),
        "points3D_bin": (sparse_dir / "points3D.bin").exists(),
        "points_ply": (sparse_dir / "points.ply").exists(),
    }
    return {
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "elapsed_sec": round(elapsed, 4),
        "log_path": str(log_path),
        "outputs_exist": outputs,
    }


def main():
    ap = argparse.ArgumentParser(description="Run VGGT fixed 6-view baseline on ZJU frames.")
    ap.add_argument("--zju_root", type=str, default=r"F:\datasets\ZJU_MoCap\data\zju_mocap")
    ap.add_argument("--seq_name", type=str, default="CoreView_390")
    ap.add_argument("--frame_ids", type=str, default="0,120")
    ap.add_argument("--camera_names", type=str, default=",".join(DEFAULT_CAMERAS))
    ap.add_argument("--model_path", type=str, default="model.pt")
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    ap.add_argument("--output_root", type=str, default=r"infer_out\vggt_6view_baseline")
    ap.add_argument("--python_bin", type=str, default=sys.executable, help="Python used to run demo_colmap.py")
    ap.add_argument("--skip_colmap", action="store_true", help="Only run tensor baseline")
    args = ap.parse_args()

    zju_root = Path(args.zju_root).resolve()
    frame_ids = _parse_frame_ids(args.frame_ids)
    cameras = _parse_csv(args.camera_names)
    if len(cameras) != 6:
        raise ValueError(f"camera_names must contain 6 views, got {len(cameras)}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    device_req = _resolve_device(args.device)
    cuda_ok, cuda_probe_msg = _probe_usable_cuda()
    if device_req.startswith("cuda") and not cuda_ok:
        print(f"[warn] requested CUDA but probe failed, fallback to CPU: {cuda_probe_msg}")
        device = "cpu"
    elif device_req == "auto" and cuda_ok:
        device = "cuda"
    else:
        device = device_req if device_req != "auto" else "cpu"
    if device.startswith("cuda") and not cuda_ok:
        device = "cpu"

    print(f"[env] python={sys.executable}")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_probe={cuda_ok}")
    print(f"[env] selected_device={device}")

    t0 = time.time()
    model = VGGT().to(device)
    print(f"[model] loading weights from {model_path}")
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model_load_sec = time.time() - t0
    print(f"[model] loaded in {model_load_sec:.2f}s")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root).resolve() / args.seq_name / f"run_{run_stamp}"
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_stamp": run_stamp,
        "zju_root": str(zju_root),
        "seq_name": args.seq_name,
        "frame_ids": frame_ids,
        "camera_names": cameras,
        "model_path": str(model_path.resolve()),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_probe_ok": bool(cuda_ok),
        "cuda_probe_msg": cuda_probe_msg,
        "device_selected": device,
        "model_load_sec": round(model_load_sec, 4),
        "frames": [],
    }

    for frame_id in frame_ids:
        frame_tag = f"frame_{frame_id:06d}"
        frame_dir = root / frame_tag
        scene_dir = frame_dir / "scene"
        frame_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[frame] {frame_tag}")
        staged = _stage_fixed_6views(zju_root, args.seq_name, frame_id, cameras, scene_dir)
        print("[frame] staged images:")
        for p in staged:
            print(f"  - {p}")

        tensor_ret = _run_tensor_baseline(model=model, device=device, image_paths=staged)
        np.save(frame_dir / "extrinsic.npy", tensor_ret["npy_payload"]["extrinsic"])
        np.save(frame_dir / "intrinsic.npy", tensor_ret["npy_payload"]["intrinsic"])
        tensor_ret.pop("npy_payload", None)

        stats_json = frame_dir / "tensor_stats.json"
        stats_json.write_text(json.dumps(tensor_ret, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[frame] wrote tensor stats: {stats_json}")

        frame_summary = {
            "frame_id": frame_id,
            "scene_dir": str(scene_dir),
            "staged_images": [str(p) for p in staged],
            "tensor_stats_path": str(stats_json),
            "tensor": tensor_ret,
        }

        if not args.skip_colmap:
            colmap_log = frame_dir / "demo_colmap.log"
            colmap_ret = _run_demo_colmap(scene_dir=scene_dir, python_bin=args.python_bin, log_path=colmap_log)
            frame_summary["demo_colmap"] = colmap_ret
            print(
                "[frame] demo_colmap "
                f"exit={colmap_ret['exit_code']} "
                f"outputs={colmap_ret['outputs_exist']}"
            )

        summary["frames"].append(frame_summary)

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
