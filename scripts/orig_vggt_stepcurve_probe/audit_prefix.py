from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_vggt_pseudo import PseudoGeomDataset, Sample  # noqa: E402
from scripts.orig_vggt_stepcurve_probe.common import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    stable_hash,
    write_json,
)


@dataclass
class SampleWithMeta:
    dataset_index: int
    npz_path: str
    sample: Sample


class AuditDataset(Dataset):
    def __init__(self, base: PseudoGeomDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> SampleWithMeta:
        seq, npz_path = self.base.items[int(idx)]
        _ = seq
        sample = self.base[int(idx)]
        return SampleWithMeta(dataset_index=int(idx), npz_path=str(npz_path), sample=sample)


def _camera_names_from_img_paths(img_paths: list[str]) -> list[str]:
    names: list[str] = []
    for path in img_paths:
        p = Path(str(path))
        names.append(p.parent.name)
    return names


def _tensor_shape_meta(sample: Sample) -> dict:
    return {
        "depth_shape": list(np.asarray(sample.depth).shape),
        "depth_conf_shape": list(np.asarray(sample.depth_conf).shape),
        "pointmap_shape": list(np.asarray(sample.pointmap).shape),
        "extrinsic_shape": [] if sample.extrinsic is None else list(np.asarray(sample.extrinsic).shape),
        "intrinsic_shape": [] if sample.intrinsic is None else list(np.asarray(sample.intrinsic).shape),
    }


def _frame_stem(img_paths: list[str]) -> str:
    if not img_paths:
        return ""
    return Path(str(img_paths[0])).stem


def _step_record(item: SampleWithMeta, tgt_camera: str) -> dict:
    sample = item.sample
    img_paths = [str(x) for x in list(sample.img_paths)]
    camera_names = _camera_names_from_img_paths(img_paths)
    src_cameras = [name for name in camera_names if name != str(tgt_camera)]
    tensor_shape_meta = _tensor_shape_meta(sample)
    payload = {
        "dataset_index": int(item.dataset_index),
        "npz_path": str(item.npz_path),
        "img_paths": img_paths,
        "camera_names": camera_names,
        "src_cameras": src_cameras,
        "tgt_camera": str(tgt_camera),
        "frame_stem": _frame_stem(img_paths),
        "pointmap_source": str(sample.pointmap_source or ""),
        "pointmap_frame": str(sample.pointmap_frame or ""),
        "tensor_shape_meta": tensor_shape_meta,
    }
    payload["sample_hash"] = stable_hash(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser("audit_prefix")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--zju_root", required=True)
    ap.add_argument("--seq_name", required=True)
    ap.add_argument("--cam_names", required=True)
    ap.add_argument("--geom_subdir", required=True)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step_horizon", type=int, required=True)
    ap.add_argument("--tgt_camera", required=True)
    ap.add_argument("--strict_deterministic", default="on")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    base = PseudoGeomDataset(
        zju_root=str(args.zju_root),
        seq_names=[str(args.seq_name)],
        cam_names=[str(x).strip() for x in str(args.cam_names).split(",") if str(x).strip()],
        max_frames=int(args.max_frames),
        geom_subdir=str(args.geom_subdir),
    )
    audit_ds = AuditDataset(base)
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    dl = DataLoader(
        audit_ds,
        batch_size=1,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )

    steps: list[dict] = []
    for step_idx, item in enumerate(dl, start=1):
        rec = {"train_step": int(step_idx)}
        rec.update(_step_record(item, args.tgt_camera))
        steps.append(rec)
        if step_idx >= int(args.step_horizon):
            break

    payload = {
        "profile": str(args.profile),
        "step_horizon": int(args.step_horizon),
        "seed": int(args.seed),
        "strict_deterministic_requested": str(args.strict_deterministic).strip().lower() in {"1", "true", "yes", "on"},
        "dataset_len": int(len(base)),
        "geom_subdir": str(args.geom_subdir),
        "cam_names": [str(x).strip() for x in str(args.cam_names).split(",") if str(x).strip()],
        "steps": steps,
    }
    payload["prefix_hash"] = stable_hash(
        {
            "profile": payload["profile"],
            "step_horizon": payload["step_horizon"],
            "seed": payload["seed"],
            "steps": [
                {
                    "train_step": row["train_step"],
                    "sample_hash": row["sample_hash"],
                }
                for row in steps
            ],
        }
    )
    write_json(args.out_json, payload)
    print(json.dumps({"out_json": str(args.out_json), "step_horizon": int(args.step_horizon), "count": len(steps)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
