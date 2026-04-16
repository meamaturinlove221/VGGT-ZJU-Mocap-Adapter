import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _resolve_image_path(
    raw_path: str,
    local_zju_root: Path,
    remote_zju_root: str,
) -> Path:
    s = str(raw_path).replace("\\", "/")
    p = Path(s)
    if p.exists():
        return p

    remote_root = remote_zju_root.rstrip("/")
    if s.startswith(remote_root + "/"):
        tail = s[len(remote_root) + 1 :]
        mapped = local_zju_root / tail
        if mapped.exists():
            return mapped

    if "/zju_mocap/" in s:
        tail2 = s.split("/zju_mocap/", 1)[1]
        mapped2 = local_zju_root / tail2
        if mapped2.exists():
            return mapped2

    raise FileNotFoundError(f"image path not found locally: {raw_path}")


def _to_pose18(view: dict[str, Any]) -> np.ndarray:
    w = float(view["width"])
    h = float(view["height"])
    fx = float(view["fx"]) / max(w, 1.0)
    fy = float(view["fy"]) / max(h, 1.0)
    cx = float(view["cx"]) / max(w, 1.0)
    cy = float(view["cy"]) / max(h, 1.0)

    w2c = np.asarray(view["w2c_opencv"], dtype=np.float32)
    if w2c.shape == (4, 4):
        w2c_3x4 = w2c[:3, :4]
    elif w2c.shape == (3, 4):
        w2c_3x4 = w2c
    else:
        raise ValueError(f"invalid w2c shape: {w2c.shape}")

    pose = np.zeros((18,), dtype=np.float32)
    pose[0] = fx
    pose[1] = fy
    pose[2] = cx
    pose[3] = cy
    pose[4] = 0.0
    pose[5] = 0.0
    pose[6:] = w2c_3x4.reshape(-1)
    return pose


def _collect_examples(
    manifest_path: Path,
    local_converted_root: Path | None,
    local_zju_root: Path,
    remote_zju_root: str,
    max_scenes: int,
) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    converted_root = local_converted_root if local_converted_root is not None else manifest_path.parent

    seq_block = manifest.get("sequences", {})
    examples: list[dict[str, Any]] = []
    for seq_name, seq_obj in seq_block.items():
        frames = seq_obj.get("frames", [])
        for frame in frames:
            frame_id = str(frame["frame_id"])
            local_frame_dir = converted_root / seq_name / f"frame_{frame_id}"
            cam_json = local_frame_dir / "cameras_opencv_c2w.json"
            if not cam_json.exists():
                raise FileNotFoundError(f"missing local camera json: {cam_json}")
            cam_data = _read_json(cam_json)
            views = cam_data.get("views", [])
            if len(views) < 2:
                continue

            camera_rows = []
            image_payloads = []
            for v in views:
                camera_rows.append(_to_pose18(v))
                img_path = _resolve_image_path(
                    raw_path=str(v["image_path"]),
                    local_zju_root=local_zju_root,
                    remote_zju_root=remote_zju_root,
                )
                raw_bytes = img_path.read_bytes()
                image_payloads.append(
                    torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy())
                )

            scene_key = f"{seq_name}_{frame_id}"
            ex = {
                "key": scene_key,
                "cameras": torch.from_numpy(np.stack(camera_rows, axis=0)),
                "images": image_payloads,
            }
            examples.append(ex)
            if max_scenes > 0 and len(examples) >= max_scenes:
                return examples
    return examples


def _split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(examples)
    if n == 0:
        return [], []
    cut = int(math.floor(n * float(train_ratio)))
    cut = max(1, min(n - 1, cut)) if n > 1 else 1
    train = examples[:cut]
    test = examples[cut:] if n > 1 else examples[:]
    if len(test) == 0:
        test = train[:1]
    return train, test


def _write_stage(
    stage_dir: Path,
    examples: list[dict[str, Any]],
    chunk_size: int,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    chunk_files: list[str] = []

    if len(examples) == 0:
        _write_json(stage_dir / "index.json", index)
        return {"num_examples": 0, "num_chunks": 0, "chunk_files": []}

    if chunk_size <= 0:
        chunk_size = len(examples)

    chunk_id = 0
    for start in range(0, len(examples), chunk_size):
        chunk = examples[start : start + chunk_size]
        chunk_name = f"chunk_{chunk_id:04d}.torch"
        chunk_path = stage_dir / chunk_name
        torch.save(chunk, chunk_path)
        chunk_files.append(chunk_name)
        for ex in chunk:
            index[str(ex["key"])] = chunk_name
        chunk_id += 1

    _write_json(stage_dir / "index.json", index)
    return {
        "num_examples": len(examples),
        "num_chunks": len(chunk_files),
        "chunk_files": chunk_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--local_zju_root", type=str, required=True)
    ap.add_argument("--local_converted_root", type=str, default="")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--remote_zju_root", type=str, default="/mnt/data/zju_mocap")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--max_scenes", type=int, default=0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_root = Path(args.out_root).resolve()
    local_zju_root = Path(args.local_zju_root).resolve()
    local_converted_root = None
    if str(args.local_converted_root).strip():
        local_converted_root = Path(args.local_converted_root).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not local_zju_root.exists():
        raise FileNotFoundError(f"local_zju_root not found: {local_zju_root}")
    if local_converted_root is not None and (not local_converted_root.exists()):
        raise FileNotFoundError(f"local_converted_root not found: {local_converted_root}")

    examples = _collect_examples(
        manifest_path=manifest_path,
        local_converted_root=local_converted_root,
        local_zju_root=local_zju_root,
        remote_zju_root=str(args.remote_zju_root),
        max_scenes=int(args.max_scenes),
    )
    if len(examples) == 0:
        raise RuntimeError("no examples collected from manifest")

    train_examples, test_examples = _split_examples(
        examples=examples,
        train_ratio=float(args.train_ratio),
    )
    train_stats = _write_stage(
        stage_dir=out_root / "train",
        examples=train_examples,
        chunk_size=int(args.chunk_size),
    )
    test_stats = _write_stage(
        stage_dir=out_root / "test",
        examples=test_examples,
        chunk_size=int(args.chunk_size),
    )

    summary = {
        "ok": True,
        "manifest": str(manifest_path),
        "local_converted_root": str(local_converted_root) if local_converted_root is not None else "",
        "local_zju_root": str(local_zju_root),
        "out_root": str(out_root),
        "num_examples_total": len(examples),
        "train": train_stats,
        "test": test_stats,
    }
    _write_json(out_root / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
