from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_viewcount.common import (  # noqa: E402
    ALL_COREVIEW390_CAMERAS,
    DEFAULT_OUT_ROOT,
    DEFAULT_ZJU_ROOT,
    infer_mask_path,
    resolve_view_spec,
    write_json,
)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def _resolve_device(raw: str) -> str:
    device = str(raw or "auto").strip().lower()
    if device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _resolve_amp_dtype(device: str, raw: str) -> torch.dtype | None:
    if not str(device).startswith("cuda"):
        return None
    text = str(raw or "auto").strip().lower()
    if text in {"", "auto"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if text in {"off", "none", "false", "0"}:
        return None
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported amp dtype: {raw}")


def _load_model(ckpt_path: Path, device: str) -> tuple[VGGT, float]:
    t0 = time.perf_counter()
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(state)}")
    keys = [str(k) for k in state.keys()]
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
        keys = [str(k) for k in state.keys()]
    has_track = any(k.startswith("track_head.") for k in keys)
    model = VGGT(enable_track=has_track).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        "[render_raw_compare] load_state_dict strict=False "
        f"missing={len(missing)} unexpected={len(unexpected)} enable_track={has_track}",
        flush=True,
    )
    model.eval()
    return model, time.perf_counter() - t0


def _tensor_stats(x: torch.Tensor) -> dict:
    arr = x.detach().float().cpu()
    return {
        "shape": list(arr.shape),
        "dtype": str(x.dtype),
        "min": float(arr.min().item()),
        "max": float(arr.max().item()),
        "mean": float(arr.mean().item()),
        "std": float(arr.std(unbiased=False).item()),
    }


def _gaussian_kernel(ch: int, device: torch.device, dtype: torch.dtype, ksize: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return g2d.repeat(ch, 1, 1, 1)


def _metrics(pred01: np.ndarray, tgt01: np.ndarray) -> dict:
    pred = torch.from_numpy(pred01.transpose(2, 0, 1)).unsqueeze(0).float()
    tgt = torch.from_numpy(tgt01.transpose(2, 0, 1)).unsqueeze(0).float()
    mae = float(F.l1_loss(pred, tgt, reduction="mean").item())
    mse = F.mse_loss(pred, tgt, reduction="mean").clamp_min(1e-8)
    psnr = float((-10.0 * torch.log10(mse)).item())
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = _gaussian_kernel(3, pred.device, pred.dtype)

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=5, groups=3)

    mu_x = filt(pred)
    mu_y = filt(tgt)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(tgt * tgt) - mu_y2
    sigma_xy = filt(pred * tgt) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return {"mae": mae, "psnr": psnr, "ssim": float(ssim_map.mean().item())}


def _resize01(arr: np.ndarray, size_hw: tuple[int, int], *, mode: int) -> np.ndarray:
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        img = img.resize((size_hw[1], size_hw[0]), mode)
        return np.asarray(img, dtype=np.float32) / 255.0
    img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    img = img.resize((size_hw[1], size_hw[0]), mode)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_image01(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    img.save(str(path))


def _cat_h(*images: np.ndarray) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _normalize_mask01(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(m).all():
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(m.max()) if m.size > 0 else 0.0
    if maxv <= 1.5:
        pass
    elif maxv <= 255.0 + 1e-3:
        m = m / 255.0
    elif maxv > 0.0:
        m = m / maxv
    return np.clip(m, 0.0, 1.0)


def _weight_to_rgb(weight01_hw: np.ndarray) -> np.ndarray:
    w = np.clip(weight01_hw.astype(np.float32), 0.0, 1.0)
    out = np.zeros(w.shape + (3,), dtype=np.float32)
    out[..., 1] = w
    out[..., 2] = 0.2 * w
    return out


def _camera_centers(extrinsic_3x4: np.ndarray) -> np.ndarray:
    r = extrinsic_3x4[:, :3, :3]
    t = extrinsic_3x4[:, :3, 3]
    return -(np.transpose(r, (0, 2, 1)) @ t[..., None])[..., 0]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def _umeyama_similarity(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("umeyama expects Nx3 inputs with identical shapes")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst
    cov = (dst0.T @ src0) / float(src.shape[0])
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    var_src = float(np.sum(src0 * src0) / float(src.shape[0]))
    if not math.isfinite(var_src) or var_src <= 1e-12:
        raise RuntimeError("degenerate source variance for sim3")
    scale = float(np.trace(np.diag(s) @ d) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def _apply_sim3_points(points_xyz: np.ndarray, scale: float, r_3x3: np.ndarray, t_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    return scale * (pts @ r_3x3.T) + t_xyz


def _read_opencv_matrix(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing node in yaml: {key}")
    mat = node.mat()
    if mat is None:
        raise KeyError(f"node has no matrix data: {key}")
    return np.asarray(mat, dtype=np.float64)


def _load_zju_cameras(seq_dir: Path, camera_names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    intri_path = seq_dir / "intri.yml"
    extri_path = seq_dir / "extri.yml"
    if not intri_path.is_file():
        raise FileNotFoundError(f"intri file not found: {intri_path}")
    if not extri_path.is_file():
        raise FileNotFoundError(f"extri file not found: {extri_path}")
    intri_fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    extri_fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not intri_fs.isOpened() or not extri_fs.isOpened():
        raise RuntimeError("failed to open zju camera yaml files")
    out: dict[str, dict[str, np.ndarray]] = {}
    try:
        for cam in camera_names:
            k = _read_opencv_matrix(intri_fs, f"K_{cam}")
            rot = _read_opencv_matrix(extri_fs, f"Rot_{cam}")
            t = _read_opencv_matrix(extri_fs, f"T_{cam}").reshape(3)
            extrinsic = np.concatenate([rot, t[:, None]], axis=1)
            out[cam] = {"intrinsic": k, "extrinsic": extrinsic}
    finally:
        intri_fs.release()
        extri_fs.release()
    return out


def _resolve_frame_image_path(seq_dir: Path, camera_name: str, frame_id: int) -> Path:
    stem = f"{int(frame_id):06d}"
    for ext in (".jpg", ".png", ".jpeg"):
        path = seq_dir / camera_name / f"{stem}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"frame image not found for {camera_name} frame={frame_id}")


def _scale_intrinsic(k_3x3: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = np.asarray(k_3x3, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out


def _render_forward_splat(
    *,
    world_points_s_hw3: np.ndarray,
    world_conf_s_hw: np.ndarray,
    src_rgb_s_hw3: np.ndarray,
    tgt_extrinsic_3x4: np.ndarray,
    tgt_intrinsic_3x3: np.ndarray,
    out_hw: tuple[int, int],
    z_eps: float,
    min_conf: float,
    z_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = int(out_hw[0]), int(out_hw[1])
    pts = np.asarray(world_points_s_hw3, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(world_conf_s_hw, dtype=np.float64).reshape(-1)
    rgb = np.asarray(src_rgb_s_hw3, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(tgt_extrinsic_3x4[:3, :3], dtype=np.float64)
    t = np.asarray(tgt_extrinsic_3x4[:3, 3], dtype=np.float64)
    cam = pts @ r.T + t[None, :]
    z = cam[:, 2]
    x = cam[:, 0] / np.maximum(z, 1e-12)
    y = cam[:, 1] / np.maximum(z, 1e-12)
    fx = float(tgt_intrinsic_3x3[0, 0])
    fy = float(tgt_intrinsic_3x3[1, 1])
    cx = float(tgt_intrinsic_3x3[0, 2])
    cy = float(tgt_intrinsic_3x3[1, 2])
    u = fx * x + cx
    v = fy * y + cy
    xi = np.rint(u).astype(np.int64)
    yi = np.rint(v).astype(np.int64)
    valid = np.isfinite(pts).all(axis=1)
    valid &= np.isfinite(conf)
    valid &= np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    valid &= z > float(z_eps)
    valid &= conf >= float(min_conf)
    valid &= xi >= 0
    valid &= yi >= 0
    valid &= xi < w
    valid &= yi < h
    if not np.any(valid):
        pred = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        stats = {
            "coverage_ratio": 0.0,
            "valid_contrib": 0,
            "mean_conf": 0.0,
            "z_eps": float(z_eps),
            "min_conf": float(min_conf),
            "z_tolerance": float(z_tolerance),
        }
        return pred, weight, stats

    pix = yi[valid] * w + xi[valid]
    z_v = z[valid]
    conf_v = conf[valid]
    rgb_v = rgb[valid]
    order = np.lexsort((z_v, pix))
    pix_s = pix[order]
    z_s = z_v[order]
    conf_s = conf_v[order]
    rgb_s = rgb_v[order]
    _, start_idx = np.unique(pix_s, return_index=True)
    counts = np.diff(np.concatenate([start_idx, np.array([pix_s.size])]))
    z_min = z_s[start_idx]
    z_min_rep = np.repeat(z_min, counts)
    keep = z_s <= (z_min_rep + float(z_tolerance))
    pix_k = pix_s[keep]
    conf_k = conf_s[keep]
    rgb_k = rgb_s[keep]
    n_pix = int(h * w)
    sum_w = np.bincount(pix_k, weights=conf_k, minlength=n_pix).astype(np.float64)
    sum_r = np.bincount(pix_k, weights=conf_k * rgb_k[:, 0], minlength=n_pix).astype(np.float64)
    sum_g = np.bincount(pix_k, weights=conf_k * rgb_k[:, 1], minlength=n_pix).astype(np.float64)
    sum_b = np.bincount(pix_k, weights=conf_k * rgb_k[:, 2], minlength=n_pix).astype(np.float64)
    hit = sum_w > 0.0
    pred = np.zeros((n_pix, 3), dtype=np.float64)
    pred[hit, 0] = sum_r[hit] / np.maximum(sum_w[hit], 1e-12)
    pred[hit, 1] = sum_g[hit] / np.maximum(sum_w[hit], 1e-12)
    pred[hit, 2] = sum_b[hit] / np.maximum(sum_w[hit], 1e-12)
    pred = np.clip(pred.reshape(h, w, 3), 0.0, 1.0).astype(np.float32)
    weight_map = sum_w.reshape(h, w)
    if np.any(hit):
        p99 = float(np.percentile(weight_map[hit.reshape(h, w)], 99.0))
        denom = max(p99, 1e-8)
        weight01 = np.clip(weight_map / denom, 0.0, 1.0).astype(np.float32)
    else:
        weight01 = np.zeros((h, w), dtype=np.float32)
    stats = {
        "coverage_ratio": float(hit.mean()),
        "valid_contrib": int(conf_k.size),
        "mean_conf": float(conf_k.mean()) if conf_k.size > 0 else 0.0,
        "z_eps": float(z_eps),
        "min_conf": float(min_conf),
        "z_tolerance": float(z_tolerance),
    }
    return pred, weight01, stats


def _detect_anchor_best_camera(anchor_cat_path: Path, seq_dir: Path, frame_id: int, render_hw: tuple[int, int]) -> tuple[str, str, float, np.ndarray] | None:
    if not anchor_cat_path.is_file():
        return None
    anchor = np.asarray(Image.open(str(anchor_cat_path)).convert("RGB"), dtype=np.float32) / 255.0
    if anchor.ndim != 3 or anchor.shape[1] < 3:
        return None
    p = anchor.shape[1] // 3
    if p <= 0:
        return None
    tgt_panel = anchor[:, 2 * p : 3 * p]
    if tuple(tgt_panel.shape[:2]) != tuple(render_hw):
        tgt_panel = _resize01(tgt_panel, render_hw, mode=Image.Resampling.BILINEAR)
    best_cam = ""
    best_path = ""
    best_mae = float("inf")
    best_img = None
    for cam in ALL_COREVIEW390_CAMERAS:
        try:
            img_path = _resolve_frame_image_path(seq_dir, cam, frame_id)
        except FileNotFoundError:
            continue
        img = np.asarray(Image.open(str(img_path)).convert("RGB"), dtype=np.float32) / 255.0
        img = _resize01(img, render_hw, mode=Image.Resampling.BILINEAR)
        mae = float(np.abs(img - tgt_panel).mean())
        if mae < best_mae:
            best_mae = mae
            best_cam = cam
            best_path = str(img_path)
            best_img = img
    if best_img is None:
        return None
    return best_cam, best_path, best_mae, best_img


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("render_raw_compare")
    ap.add_argument("--seq_name", default=os.environ.get("VGGT_SEQ_NAMES", "CoreView_390").split(",")[0].strip() or "CoreView_390")
    ap.add_argument("--frame_id", type=int, default=1080)
    ap.add_argument("--tgt_camera", default="Camera_B5")
    ap.add_argument("--view_profile", default="6src_hist")
    ap.add_argument("--src_cameras", default="")
    ap.add_argument("--zju_root", default=os.environ.get("VGGT_ZJU_ROOT", str(DEFAULT_ZJU_ROOT)))
    ap.add_argument(
        "--ckpt",
        default=os.environ.get("VGGT_PRECOMPUTE_CKPT", os.environ.get("VGGT_CKPT", str(REPO_ROOT / "model.pt"))),
    )
    ap.add_argument("--out_dir", default=os.environ.get("VGGT_OUT_DIR", str(DEFAULT_OUT_ROOT)))
    ap.add_argument("--device", default=os.environ.get("VGGT_DEVICE", "auto"))
    ap.add_argument("--amp_dtype", default=os.environ.get("VGGT_AMP_DTYPE", "auto"))
    ap.add_argument("--render_size", nargs=2, type=int, default=[518, 518])
    ap.add_argument("--min_conf", type=float, default=1e-6)
    ap.add_argument("--z_eps", type=float, default=1e-6)
    ap.add_argument("--z_tolerance", type=float, default=0.02)
    ap.add_argument("--anchor_cat_path", default="")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    t_total0 = time.perf_counter()
    device = _resolve_device(args.device)
    amp_dtype = _resolve_amp_dtype(device, args.amp_dtype)
    render_hw = (int(args.render_size[0]), int(args.render_size[1]))
    view_spec = resolve_view_spec(
        view_profile=args.view_profile,
        tgt_camera=args.tgt_camera,
        src_cameras=args.src_cameras,
    )
    seq_dir = Path(args.zju_root) / args.seq_name
    ckpt_path = Path(args.ckpt)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    frame_dir = out_root / args.seq_name / f"frame_{int(args.frame_id):06d}_{args.tgt_camera}"
    run_dir = frame_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    src_image_paths = [_resolve_frame_image_path(seq_dir, cam, int(args.frame_id)) for cam in view_spec["src_cameras"]]
    tgt_image_path = _resolve_frame_image_path(seq_dir, args.tgt_camera, int(args.frame_id))
    images = load_and_preprocess_images(src_image_paths)
    input_shape = list(images.shape)
    images_np = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, model_load_sec = _load_model(ckpt_path, device=device)
    images_batched = images.unsqueeze(0).to(device)
    ctx = (
        torch.cuda.amp.autocast(dtype=amp_dtype)
        if (amp_dtype is not None and str(device).startswith("cuda"))
        else contextlib.nullcontext()
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        with ctx:
            predictions = model(images_batched)
    inference_sec = time.perf_counter() - t0

    pose_enc = predictions["pose_enc"]
    extrinsic_pred, intrinsic_pred = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    world_points = predictions["world_points"][0].detach().float().cpu().numpy().astype(np.float64)
    world_points_conf = predictions["world_points_conf"][0].detach().float().cpu().numpy().astype(np.float64)
    extrinsic_pred_np = extrinsic_pred[0].detach().float().cpu().numpy().astype(np.float64)
    intrinsic_pred_np = intrinsic_pred[0].detach().float().cpu().numpy().astype(np.float64)

    tensor_stats = {
        "input_shape": input_shape,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else "off",
        "model_load_sec": float(model_load_sec),
        "inference_sec": float(inference_sec),
        "prediction_keys": sorted([str(k) for k in predictions.keys()]),
        "tensors": {
            "pose_enc": _tensor_stats(predictions["pose_enc"]),
            "depth": _tensor_stats(predictions["depth"]),
            "depth_conf": _tensor_stats(predictions["depth_conf"]),
            "world_points": _tensor_stats(predictions["world_points"]),
            "world_points_conf": _tensor_stats(predictions["world_points_conf"]),
            "images": _tensor_stats(predictions["images"]),
        },
        "decoded_camera": {
            "extrinsic_shape": list(extrinsic_pred_np.shape),
            "intrinsic_shape": list(intrinsic_pred_np.shape),
        },
    }
    write_json(run_dir / "tensor_stats.json", tensor_stats)

    gt_cameras = _load_zju_cameras(seq_dir, view_spec["src_cameras"] + [args.tgt_camera])
    centers_pred = _camera_centers(extrinsic_pred_np)
    centers_gt = _camera_centers(
        np.stack([gt_cameras[cam]["extrinsic"] for cam in view_spec["src_cameras"]], axis=0)
    )
    rmse_before = _rmse(centers_pred, centers_gt)
    scale, sim_r, sim_t = _umeyama_similarity(centers_pred, centers_gt)
    centers_after = _apply_sim3_points(centers_pred, scale, sim_r, sim_t)
    rmse_after = _rmse(centers_after, centers_gt)
    if rmse_after > 0.15:
        raise RuntimeError(f"sim3 alignment rmse_after too high: {rmse_after:.6f}")
    world_points_aligned = _apply_sim3_points(world_points.reshape(-1, 3), scale, sim_r, sim_t).reshape(world_points.shape)

    tgt_img_full = np.asarray(Image.open(str(tgt_image_path)).convert("RGB"), dtype=np.float32) / 255.0
    tgt_img_native = _resize01(tgt_img_full, render_hw, mode=Image.Resampling.BILINEAR)
    tgt_intrinsic_native = _scale_intrinsic(
        gt_cameras[args.tgt_camera]["intrinsic"],
        src_hw=tgt_img_full.shape[:2],
        dst_hw=render_hw,
    )
    tgt_extrinsic = gt_cameras[args.tgt_camera]["extrinsic"]
    pred_native, weight_native, render_stats = _render_forward_splat(
        world_points_s_hw3=world_points_aligned,
        world_conf_s_hw=world_points_conf,
        src_rgb_s_hw3=images_np,
        tgt_extrinsic_3x4=tgt_extrinsic,
        tgt_intrinsic_3x3=tgt_intrinsic_native,
        out_hw=render_hw,
        z_eps=float(args.z_eps),
        min_conf=float(args.min_conf),
        z_tolerance=float(args.z_tolerance),
    )
    weight_rgb_native = _weight_to_rgb(weight_native)
    diff_native = np.repeat(
        np.abs(pred_native - tgt_img_native).mean(axis=2, keepdims=True),
        3,
        axis=2,
    ).astype(np.float32)
    up_hw = (1024, 1024)
    weight_rgb_1024 = _resize01(weight_rgb_native, up_hw, mode=Image.Resampling.BILINEAR)
    pred_1024 = _resize01(pred_native, up_hw, mode=Image.Resampling.BILINEAR)
    tgt_1024 = _resize01(tgt_img_native, up_hw, mode=Image.Resampling.BILINEAR)
    diff_1024 = _resize01(diff_native, up_hw, mode=Image.Resampling.BILINEAR)

    cat_native = _cat_h(weight_rgb_native, pred_native, tgt_img_native)
    cat_1024 = _cat_h(weight_rgb_1024, pred_1024, tgt_1024)
    p_w = cat_1024.shape[1] // 3
    p0 = cat_1024[:, :p_w]
    p1 = cat_1024[:, p_w : 2 * p_w]
    p2 = cat_1024[:, 2 * p_w : 3 * p_w]

    mask_path = infer_mask_path(tgt_image_path)
    if mask_path is not None and mask_path.is_file():
        mask_full = _normalize_mask01(np.asarray(Image.open(str(mask_path)).convert("L"), dtype=np.float32))
        mask_native = _resize01(mask_full, render_hw, mode=Image.Resampling.NEAREST)
    else:
        mask_native = np.zeros(render_hw, dtype=np.float32)
    mask_rgb_native = np.repeat(mask_native[..., None], 3, axis=2)
    ghost_triplet = _cat_h(mask_rgb_native, pred_native, tgt_img_native)
    pred_tgt_pair = _cat_h(pred_native, tgt_img_native)
    overlay = tgt_img_native.copy()
    overlay[..., 1] = np.clip(overlay[..., 1] * 0.6 + mask_native * 0.4, 0.0, 1.0)
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.85, 0.0, 1.0)
    overlay[..., 2] = np.clip(overlay[..., 2] * 0.85, 0.0, 1.0)

    _save_image01(run_dir / "weight_native.png", weight_rgb_native)
    _save_image01(run_dir / "pred_native.png", pred_native)
    _save_image01(run_dir / "tgt_native.png", tgt_img_native)
    _save_image01(run_dir / "cat_weight_pred_tgt_native.png", cat_native)
    _save_image01(run_dir / "cat_weight_pred_tgt.png", cat_1024)
    _save_image01(run_dir / "cat_weight_pred_tgt_p0.png", p0)
    _save_image01(run_dir / "cat_weight_pred_tgt_p1.png", p1)
    _save_image01(run_dir / "cat_weight_pred_tgt_p2.png", p2)
    _save_image01(run_dir / "pred_tgt_diff.png", diff_1024)
    _save_image01(run_dir / "cat_fg_mask_pred_tgt_step000000.png", ghost_triplet)
    _save_image01(run_dir / "cat_pred_tgt_step000000.png", pred_tgt_pair)
    _save_image01(run_dir / "gt_with_fg_overlay_step000000.png", overlay)

    anchor_info = None
    anchor_compare_rel = None
    anchor_path = Path(args.anchor_cat_path) if str(args.anchor_cat_path).strip() else None
    if anchor_path is not None and anchor_path.is_file():
        anchor_info = _detect_anchor_best_camera(anchor_path, seq_dir, int(args.frame_id), render_hw)
        if anchor_info is not None:
            best_img = anchor_info[3]
            _save_image01(run_dir / "anchor_pred_tgt_compare.png", _cat_h(best_img, tgt_img_native))
            anchor_compare_rel = str(run_dir / "anchor_pred_tgt_compare.png")

    metrics_native = _metrics(pred_native, tgt_img_native)
    metrics_1024 = _metrics(pred_1024, tgt_1024)
    cuda_peak_mem_mb = None
    if str(device).startswith("cuda"):
        cuda_peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "zju_root": str(Path(args.zju_root)),
        "seq_name": args.seq_name,
        "frame_id": int(args.frame_id),
        "view_profile": view_spec["view_profile"],
        "profile_kind": view_spec["profile_kind"],
        "num_total_cams": int(view_spec["num_total_cams"]),
        "num_src_views_actual": int(view_spec["num_src_views_actual"]),
        "src_cameras": view_spec["src_cameras"],
        "tgt_camera": args.tgt_camera,
        "src_image_paths": [str(p) for p in src_image_paths],
        "tgt_image_path": str(tgt_image_path),
        "tgt_mask_path": str(mask_path) if mask_path is not None else "",
        "python": sys.executable,
        "torch": torch.__version__,
        "device": device,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else "off",
        "ckpt": str(ckpt_path),
        "cuda_peak_mem_mb": cuda_peak_mem_mb,
    }
    if anchor_path is not None and anchor_path.is_file():
        meta["anchor_cat_path"] = str(anchor_path)
    if anchor_info is not None:
        meta["anchor_detected_best_camera"] = anchor_info[0]
        meta["anchor_detected_best_image_path"] = anchor_info[1]
        meta["anchor_detected_best_mae"] = float(anchor_info[2])

    report = {
        "meta": meta,
        "sim3": {
            "scale": float(scale),
            "R": sim_r.tolist(),
            "t": sim_t.tolist(),
            "src_center_rmse_before": float(rmse_before),
            "src_center_rmse_after": float(rmse_after),
        },
        "render": {"render_size": [int(render_hw[0]), int(render_hw[1])], **render_stats},
        "metrics": {"native": metrics_native, "upsampled_1024": metrics_1024},
        "paths": {
            "weight_native": str(run_dir / "weight_native.png"),
            "pred_native": str(run_dir / "pred_native.png"),
            "tgt_native": str(run_dir / "tgt_native.png"),
            "cat_weight_pred_tgt_native": str(run_dir / "cat_weight_pred_tgt_native.png"),
            "cat_weight_pred_tgt": str(run_dir / "cat_weight_pred_tgt.png"),
            "cat_weight_pred_tgt_p0": str(run_dir / "cat_weight_pred_tgt_p0.png"),
            "cat_weight_pred_tgt_p1": str(run_dir / "cat_weight_pred_tgt_p1.png"),
            "cat_weight_pred_tgt_p2": str(run_dir / "cat_weight_pred_tgt_p2.png"),
            "pred_tgt_diff": str(run_dir / "pred_tgt_diff.png"),
            "ghost_triplet": str(run_dir / "cat_fg_mask_pred_tgt_step000000.png"),
            "pred_tgt_pair": str(run_dir / "cat_pred_tgt_step000000.png"),
            "gt_with_fg_overlay": str(run_dir / "gt_with_fg_overlay_step000000.png"),
        },
        "elapsed_sec_total": float(time.perf_counter() - t_total0),
    }
    if anchor_compare_rel is not None:
        report["paths"]["anchor_pred_tgt_compare"] = anchor_compare_rel

    write_json(run_dir / "meta.json", meta)
    write_json(run_dir / "report.json", report)
    (run_dir / "sources.txt").write_text(
        "\n".join([str(p) for p in src_image_paths]) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "run_dir": str(run_dir),
        "report_json": str(run_dir / "report.json"),
        "view_profile": view_spec["view_profile"],
        "num_src_views_actual": int(view_spec["num_src_views_actual"]),
        "tgt_camera": args.tgt_camera,
        "frame_id": int(args.frame_id),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"RUN_DIR: {run_dir}")
    print(f"REPORT_JSON: {run_dir / 'report.json'}")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
