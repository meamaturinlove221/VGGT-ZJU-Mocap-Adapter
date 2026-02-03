# debug_mask_sanity.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, Optional, List, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw

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


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _jet_colormap(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _save_heatmap(path: str, x: np.ndarray, vmin: float, vmax: float) -> None:
    x = np.array(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=vmax, posinf=vmax, neginf=vmin)
    x01 = (x - float(vmin)) / (float(vmax) - float(vmin) + 1e-8)
    img = _jet_colormap(x01)
    Image.fromarray(img).save(path)


def _plot_curve_png(path: str, xs: np.ndarray, ys: np.ndarray, title: str = "") -> None:
    W, H = 800, 480
    pad = 50
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if xs.size == 0 or ys.size == 0:
        draw.text((10, 10), "empty curve", fill=(0, 0, 0))
        img.save(path)
        return

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    # axes
    draw.line([(pad, H - pad), (W - pad, H - pad)], fill=(0, 0, 0), width=2)
    draw.line([(pad, pad), (pad, H - pad)], fill=(0, 0, 0), width=2)
    if title:
        draw.text((pad, 10), title, fill=(0, 0, 0))

    pts = []
    for x, y in zip(xs, ys):
        px = pad + (float(x) - x_min) / (x_max - x_min) * (W - 2 * pad)
        py = H - pad - (float(y) - y_min) / (y_max - y_min) * (H - 2 * pad)
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill=(255, 0, 0), width=2)
    img.save(path)


def _project_world_to_pixel(pointmap_hw3: np.ndarray, K: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W, _ = pointmap_hw3.shape
    pts = pointmap_hw3.reshape(-1, 3).T  # (3, N)
    R = T[:3, :3]
    t = T[:3, 3:4]
    cam = (R @ pts) + t  # (3, N)
    z = cam[2, :]
    x = cam[0, :] / (z + 1e-8)
    y = cam[1, :] / (z + 1e-8)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u.reshape(H, W), v.reshape(H, W), z.reshape(H, W)


def _geom_sanity_from_npz(npz_path: str, out_dir: str, view: str, err_clip: float,
                          min_pos_depth_ratio: float, min_inbound_ratio: float, max_nan_ratio: float,
                          do_assert: bool, curve_bins: int = 64) -> None:
    data = np.load(npz_path, allow_pickle=True)

    def _pick_view():
        if view.startswith("src"):
            idx = 0
            if ":" in view:
                try:
                    idx = int(view.split(":")[1])
                except Exception:
                    idx = 0
            src_imgs = data.get("src_imgs", None)
            src_depth = data.get("src_depth", None)
            src_pointmap = data.get("src_pointmap", None)
            src_K = data.get("src_K", None)
            src_T = data.get("src_T", None)
            if src_imgs is None or src_depth is None or src_pointmap is None or src_K is None or src_T is None:
                raise RuntimeError("npz missing src_* keys for geometry sanity")
            return (
                src_imgs[idx], src_depth[idx], src_pointmap[idx],
                src_K[idx], src_T[idx],
                f"src{idx}"
            )
        # default tgt
        return (
            data["tgt_img"], data["tgt_depth"], data["tgt_pointmap"],
            data["tgt_K"], data["tgt_T"],
            "tgt"
        )

    img, depth, pointmap, K, T, tag = _pick_view()

    # ensure shapes
    img = _to_numpy(img)
    depth = _to_numpy(depth)
    pointmap = _to_numpy(pointmap)
    K = _to_numpy(K)
    T = _to_numpy(T)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img_hwc = np.transpose(img, (1, 2, 0))
    else:
        img_hwc = img
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth_hw = depth[0]
    elif depth.ndim == 2:
        depth_hw = depth
    else:
        depth_hw = depth.squeeze()
    if pointmap.ndim == 3 and pointmap.shape[0] == 3:
        pointmap_hw3 = np.transpose(pointmap, (1, 2, 0))
    else:
        pointmap_hw3 = pointmap

    H, W = depth_hw.shape
    u, v, z_cam = _project_world_to_pixel(pointmap_hw3, K, T)
    grid_u, grid_v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(z_cam) & np.isfinite(u) & np.isfinite(v)
    pos_depth = depth_hw > 0
    valid = valid & pos_depth
    in_bounds = (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))

    reproj_err = np.sqrt((u - grid_u) ** 2 + (v - grid_v) ** 2)
    reproj_err[~valid] = np.nan
    depth_err = np.abs(depth_hw - z_cam)
    depth_err[~valid] = np.nan

    nan_ratio = 1.0 - float(np.isfinite(pointmap_hw3).all(axis=-1).mean())
    pos_depth_ratio = float(pos_depth.mean())
    inbound_ratio = float((in_bounds & valid).mean())

    print(f"[geom] view={tag} HxW={H}x{W}")
    print(f"[geom] pos_depth_ratio={pos_depth_ratio:.4f} inbound_ratio={inbound_ratio:.4f} nan_ratio={nan_ratio:.6f}")
    print(f"[geom] reproj_err: mean={np.nanmean(reproj_err):.4f} median={np.nanmedian(reproj_err):.4f} max={np.nanmax(reproj_err):.4f}")
    print(f"[geom] depth_vs_z: mean_abs={np.nanmean(depth_err):.4f} median_abs={np.nanmedian(depth_err):.4f} max_abs={np.nanmax(depth_err):.4f}")

    prefix = os.path.splitext(os.path.basename(npz_path))[0] + f"_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    if img_hwc is not None:
        img_u8 = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img_u8).save(os.path.join(out_dir, f"{prefix}_img.png"))

    _save_heatmap(os.path.join(out_dir, f"{prefix}_reproj_err.png"), reproj_err, 0.0, float(err_clip))
    _save_heatmap(os.path.join(out_dir, f"{prefix}_depth_err.png"), depth_err, 0.0, float(err_clip))

    # depth vs z consistency curve (binned)
    vals = depth_hw[valid]
    errs = depth_err[valid]
    if vals.size > 0:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        bins = np.linspace(vmin, vmax, int(curve_bins) + 1)
        xs = []
        ys = []
        for i in range(len(bins) - 1):
            mask = (vals >= bins[i]) & (vals < bins[i + 1])
            if mask.sum() == 0:
                continue
            xs.append(0.5 * (bins[i] + bins[i + 1]))
            ys.append(float(np.nanmean(errs[mask])))
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        if xs.size > 0:
            _plot_curve_png(os.path.join(out_dir, f"{prefix}_depth_vs_z_curve.png"), xs, ys,
                            title="depth vs pointmap.z (mean abs err)")

    if do_assert:
        if pos_depth_ratio < float(min_pos_depth_ratio):
            raise AssertionError(f"pos_depth_ratio {pos_depth_ratio:.4f} < {min_pos_depth_ratio}")
        if inbound_ratio < float(min_inbound_ratio):
            raise AssertionError(f"inbound_ratio {inbound_ratio:.4f} < {min_inbound_ratio}")
        if nan_ratio > float(max_nan_ratio):
            raise AssertionError(f"nan_ratio {nan_ratio:.6f} > {max_nan_ratio}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="mask_debug_out")
    ap.add_argument("--pt", type=str, default=None,
                    help="Path to a saved sample .pt")
    ap.add_argument("--npz", type=str, default=None,
                    help="Path to a dumped sample .npz (from zju_dataset_view.py --dump_one_batch)")
    ap.add_argument("--view", type=str, default="tgt",
                    help="Which view to inspect for npz: 'tgt' or 'src:0'")
    ap.add_argument("--err_clip", type=float, default=5.0,
                    help="Max error shown in heatmap (pixels or depth units)")
    ap.add_argument("--min_pos_depth_ratio", type=float, default=0.90)
    ap.add_argument("--min_inbound_ratio", type=float, default=0.85)
    ap.add_argument("--max_nan_ratio", type=float, default=1e-4)
    ap.add_argument("--no_assert", action="store_true", default=False)
    ap.add_argument("--curve_bins", type=int, default=64)
    ap.add_argument("--dilate_k", type=int, default=1,
                    help="Dilation kernel for fg mask sanity test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.npz:
        _geom_sanity_from_npz(
            args.npz,
            out_dir=args.out,
            view=str(args.view),
            err_clip=float(args.err_clip),
            min_pos_depth_ratio=float(args.min_pos_depth_ratio),
            min_inbound_ratio=float(args.min_inbound_ratio),
            max_nan_ratio=float(args.max_nan_ratio),
            do_assert=not bool(args.no_assert),
            curve_bins=int(args.curve_bins),
        )
        return
    if args.pt is None:
        raise RuntimeError(
            "Please provide --pt or --npz path, or adapt this script to import your dataset.")
    load_pt_and_inspect(args.pt, out_dir=args.out, dilate_k=args.dilate_k)


if __name__ == "__main__":
    main()
