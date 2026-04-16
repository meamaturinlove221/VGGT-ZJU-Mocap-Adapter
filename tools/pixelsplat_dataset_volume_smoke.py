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


def _check_stage(stage_dir: Path, max_examples: int) -> dict[str, Any]:
    idx_path = stage_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing index json: {idx_path}")
    idx = _read_json(idx_path)
    keys = sorted(idx.keys())
    chunk_names = sorted(set(str(v) for v in idx.values()))

    checked = 0
    min_views = None
    max_views = None
    image_size = None
    for key in keys:
        chunk_name = str(idx[key])
        chunk_path = stage_dir / chunk_name
        if not chunk_path.exists():
            raise FileNotFoundError(f"missing chunk: {chunk_path}")
        chunk = torch.load(chunk_path, map_location="cpu")
        rows = [x for x in chunk if str(x.get("key", "")) == key]
        if len(rows) != 1:
            raise RuntimeError(f"scene lookup mismatch for {key}: {len(rows)}")
        ex = rows[0]
        cams = ex["cameras"]
        imgs = ex["images"]
        v = int(cams.shape[0])
        min_views = v if min_views is None else min(min_views, v)
        max_views = v if max_views is None else max(max_views, v)
        if image_size is None and len(imgs) > 0:
            arr = imgs[0].detach().cpu().numpy().astype(np.uint8, copy=False)
            im = Image.open(BytesIO(arr.tobytes())).convert("RGB")
            image_size = [int(im.size[0]), int(im.size[1])]
        checked += 1
        if max_examples > 0 and checked >= max_examples:
            break

    return {
        "num_index_entries": len(keys),
        "num_unique_chunks": len(chunk_names),
        "checked_examples": checked,
        "min_views": int(min_views or 0),
        "max_views": int(max_views or 0),
        "sample_image_size": image_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="/mnt/out/pixelsplat_datasets/zju_phase2_300_6v")
    ap.add_argument("--max_examples", type=int, default=10)
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root missing in volume: {root}")

    summary = {
        "ok": True,
        "dataset_root": str(root),
        "train": _check_stage(root / "train", int(args.max_examples)),
        "test": _check_stage(root / "test", int(args.max_examples)),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if str(args.save_json).strip():
        out = Path(str(args.save_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
