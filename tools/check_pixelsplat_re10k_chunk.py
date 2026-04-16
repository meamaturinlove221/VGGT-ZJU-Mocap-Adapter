import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_image(blob_tensor: torch.Tensor) -> tuple[int, int]:
    arr = blob_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
    img = Image.open(BytesIO(arr.tobytes()))
    img = img.convert("RGB")
    return img.size


def _check_stage(stage_dir: Path, max_examples: int) -> dict[str, Any]:
    index_path = stage_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")
    index = _read_json(index_path)
    keys = sorted(index.keys())
    if len(keys) == 0:
        return {
            "num_index_entries": 0,
            "checked_examples": 0,
            "unique_chunk_files": 0,
            "ok": True,
        }

    checked = 0
    bad: list[str] = []
    chunk_files = set()
    image_sizes = []
    num_views = []
    for k in keys:
        chunk_name = str(index[k])
        chunk_files.add(chunk_name)
        chunk_path = stage_dir / chunk_name
        if not chunk_path.exists():
            bad.append(f"{k}: missing chunk {chunk_name}")
            continue
        chunk = torch.load(chunk_path, map_location="cpu")
        hit = [x for x in chunk if str(x.get("key", "")) == k]
        if len(hit) != 1:
            bad.append(f"{k}: chunk lookup mismatch ({len(hit)})")
            continue
        ex = hit[0]
        cameras = ex.get("cameras", None)
        images = ex.get("images", None)
        if cameras is None or images is None:
            bad.append(f"{k}: missing cameras/images")
            continue
        if not torch.is_tensor(cameras):
            bad.append(f"{k}: cameras not tensor")
            continue
        if cameras.ndim != 2 or cameras.shape[1] != 18:
            bad.append(f"{k}: cameras shape {tuple(cameras.shape)}")
            continue
        if not isinstance(images, list) or len(images) != int(cameras.shape[0]):
            bad.append(
                f"{k}: images length mismatch ({len(images) if isinstance(images, list) else 'non-list'})"
            )
            continue
        if len(images) > 0 and torch.is_tensor(images[0]):
            try:
                size = _decode_image(images[0])
                image_sizes.append(size)
            except Exception as e:
                bad.append(f"{k}: decode failed ({e})")
                continue
        num_views.append(int(cameras.shape[0]))
        checked += 1
        if max_examples > 0 and checked >= max_examples:
            break

    return {
        "num_index_entries": len(keys),
        "checked_examples": checked,
        "unique_chunk_files": len(chunk_files),
        "min_views": min(num_views) if num_views else 0,
        "max_views": max(num_views) if num_views else 0,
        "sample_image_size": image_sizes[0] if image_sizes else None,
        "ok": len(bad) == 0,
        "errors": bad[:50],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--max_examples", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    train_stats = _check_stage(root / "train", max_examples=int(args.max_examples))
    test_stats = _check_stage(root / "test", max_examples=int(args.max_examples))
    summary = {
        "ok": bool(train_stats.get("ok", False) and test_stats.get("ok", False)),
        "dataset_root": str(root),
        "train": train_stats,
        "test": test_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
