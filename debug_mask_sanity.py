# debug_mask_sanity.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, Optional, List

import torch

from mask_ops import (
    binarize,
    resize_mask_nearest,
    mask_dilation,
    cover_ratio,
    save_overlay_triplet,
)


def _to_chw(img: torch.Tensor) -> torch.Tensor:
    """
    Accept CHW or HWC, return CHW.
    """
    if img.dim() == 3 and img.shape[0] in (1, 3):
        return img
    if img.dim() == 3 and img.shape[-1] in (1, 3):
        return img.permute(2, 0, 1)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def inspect_sample(
    sample: Dict[str, Any],
    out_dir: str,
    prefix: str,
    expect_hw: Optional[tuple] = None,
    dilate_k: int = 1,
) -> None:
    """
    sample should include:
      - tgt_rgb: (3,H,W) float
      - fg_mask: (1,H,W) or (H,W)
      - valid_mask: (1,H,W) or (H,W) (optional)
    """
    tgt = _to_chw(sample["tgt_rgb"]).float()
    fg = sample["fg_mask"].float()
    valid = sample.get("valid_mask", None)

    if fg.dim() == 2:
        fg = fg[None, ...]
    fg = binarize(fg)

    if valid is not None:
        if valid.dim() == 2:
            valid = valid[None, ...]
        valid = binarize(valid)

    # force resize masks to tgt size with nearest if mismatched
    H, W = tgt.shape[-2], tgt.shape[-1]
    if fg.shape[-2:] != (H, W):
        fg = resize_mask_nearest(fg, (H, W))
        fg = binarize(fg)
    if valid is not None and valid.shape[-2:] != (H, W):
        valid = resize_mask_nearest(valid, (H, W))
        valid = binarize(valid)

    # optional dilation (for checking effect)
    if dilate_k > 1:
        fg = mask_dilation(fg, k=dilate_k)

    cover_fg = cover_ratio(fg)
    cover_valid = cover_ratio(valid) if valid is not None else 1.0
    cover_mask = cover_valid

    print(f"[{prefix}] cover_fg={cover_fg:.6f} cover_valid={cover_valid:.6f} cover_mask={cover_mask:.6f}")

    save_overlay_triplet(out_dir, prefix, tgt, fg, valid)


def load_pt_and_inspect(pt_path: str, out_dir: str, dilate_k: int = 1) -> None:
    """
    Expect a dict saved by torch.save containing keys:
      tgt_rgb, fg_mask, valid_mask(optional)
    """
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Saved .pt must be a dict")

    sample = {
        "tgt_rgb": obj["tgt_rgb"],
        "fg_mask": obj["fg_mask"],
    }
    if "valid_mask" in obj:
        sample["valid_mask"] = obj["valid_mask"]

    prefix = os.path.splitext(os.path.basename(pt_path))[0]
    inspect_sample(sample, out_dir=out_dir, prefix=prefix, dilate_k=dilate_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="mask_debug_out")
    ap.add_argument("--pt", type=str, default=None,
                    help="Path to a saved sample .pt")
    ap.add_argument("--dilate_k", type=int, default=1,
                    help="Dilation kernel for fg mask sanity test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.pt is None:
        raise RuntimeError(
            "Please provide --pt path, or adapt this script to import your dataset.")
    load_pt_and_inspect(args.pt, out_dir=args.out, dilate_k=args.dilate_k)


if __name__ == "__main__":
    main()
