import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_index(stage_dir: Path) -> dict[str, str]:
    with (stage_dir / "index.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def _camera_centers_from_pose18(cameras: np.ndarray) -> np.ndarray:
    # cameras: [V, 18], where [6:] is row-major w2c 3x4
    mats = cameras[:, 6:].reshape(-1, 3, 4)
    r = mats[:, :, :3]
    t = mats[:, :, 3]
    # C = -R^T t
    c = -np.einsum("vji,vj->vi", r, t)
    return c


def _yaws_from_centers(centers: np.ndarray) -> np.ndarray:
    c0 = centers.mean(axis=0, keepdims=True)
    d = centers - c0
    x = d[:, 0]
    z = d[:, 2]
    yaw = np.arctan2(z, x)
    return yaw


def _uniform_context_indices(num_views: int, num_context: int, yaws: np.ndarray) -> list[int]:
    order = np.argsort(yaws)
    if num_context >= num_views:
        return order.tolist()
    ticks = np.linspace(0, num_views, num=num_context, endpoint=False)
    pick = [int(round(t)) % num_views for t in ticks]
    out = [int(order[p]) for p in pick]
    # Preserve order and uniqueness.
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    # Fill if rounding caused collisions.
    if len(uniq) < num_context:
        for x in order.tolist():
            if x not in seen:
                uniq.append(int(x))
                seen.add(int(x))
            if len(uniq) >= num_context:
                break
    return uniq[:num_context]


def _random_context_indices(
    num_views: int,
    num_context: int,
    scene_key: str,
    seed: int,
) -> list[int]:
    digest = hashlib.md5(f"{scene_key}|{seed}".encode("utf-8")).hexdigest()
    s = int(digest[:8], 16)
    rng = np.random.RandomState(s)
    idx = np.arange(num_views)
    rng.shuffle(idx)
    return idx[:num_context].tolist()


def _target_from_context(num_views: int, context: list[int]) -> list[int]:
    cset = set(int(x) for x in context)
    for i in range(num_views):
        if i not in cset:
            return [int(i)]
    return [0]


def _build_index(
    dataset_root: Path,
    stage: str,
    mode: str,
    num_context: int,
    seed: int,
    max_scenes: int,
) -> dict[str, Any]:
    stage_dir = dataset_root / stage
    key_to_chunk = _load_index(stage_dir)
    keys = sorted(key_to_chunk.keys())
    if max_scenes > 0:
        keys = keys[:max_scenes]

    index_obj: dict[str, Any] = {}
    chunk_cache: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        chunk_name = key_to_chunk[key]
        if chunk_name not in chunk_cache:
            chunk_cache[chunk_name] = torch.load(stage_dir / chunk_name, map_location="cpu")
        chunk = chunk_cache[chunk_name]
        row = [x for x in chunk if str(x.get("key", "")) == key]
        if len(row) != 1:
            continue
        ex = row[0]
        cams = ex["cameras"].detach().cpu().numpy()
        num_views = int(cams.shape[0])
        if num_views < 2:
            continue
        ctx_n = min(int(num_context), num_views - 1)
        centers = _camera_centers_from_pose18(cams)
        yaws = _yaws_from_centers(centers)
        if mode == "uniform":
            context = _uniform_context_indices(num_views, ctx_n, yaws)
        elif mode == "random":
            context = _random_context_indices(num_views, ctx_n, key, seed)
        else:
            raise ValueError(f"unknown mode: {mode}")
        target = _target_from_context(num_views, context)
        index_obj[key] = {"context": [int(x) for x in context], "target": [int(x) for x in target]}
    return index_obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--stage", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--out_uniform", type=str, required=True)
    ap.add_argument("--out_random", type=str, required=True)
    ap.add_argument("--num_context_views", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_scenes", type=int, default=0)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_uniform = Path(args.out_uniform).resolve()
    out_random = Path(args.out_random).resolve()
    out_uniform.parent.mkdir(parents=True, exist_ok=True)
    out_random.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    uniform_idx = _build_index(
        dataset_root=dataset_root,
        stage=str(args.stage),
        mode="uniform",
        num_context=int(args.num_context_views),
        seed=int(args.seed),
        max_scenes=int(args.max_scenes),
    )
    random_idx = _build_index(
        dataset_root=dataset_root,
        stage=str(args.stage),
        mode="random",
        num_context=int(args.num_context_views),
        seed=int(args.seed),
        max_scenes=int(args.max_scenes),
    )

    with out_uniform.open("w", encoding="utf-8") as f:
        json.dump(uniform_idx, f, ensure_ascii=False, indent=2)
    with out_random.open("w", encoding="utf-8") as f:
        json.dump(random_idx, f, ensure_ascii=False, indent=2)

    summary = {
        "ok": True,
        "dataset_root": str(dataset_root),
        "stage": str(args.stage),
        "num_context_views": int(args.num_context_views),
        "num_scenes_uniform": len(uniform_idx),
        "num_scenes_random": len(random_idx),
        "out_uniform": str(out_uniform),
        "out_random": str(out_random),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
