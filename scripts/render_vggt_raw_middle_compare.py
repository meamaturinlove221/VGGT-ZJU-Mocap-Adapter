#!/usr/bin/env python
import argparse
import contextlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_ZJU_ROOT = r"F:\datasets\ZJU_MoCap\data\zju_mocap"
DEFAULT_SEQ_NAME = "CoreView_390"
DEFAULT_FRAME_ID = 1080
DEFAULT_TGT_CAMERA = "Camera_B5"
DEFAULT_SRC_CAMERAS = "Camera_B1,Camera_B9,Camera_B10,Camera_B14,Camera_B19,Camera_B23"
DEFAULT_ANCHOR = (
    r"logs\vd_ablation_smoke\smoke_gatefloor_lcc\val"
    r"\val_e000_cat_gate_pred_tgt_step000017.png"
)


def _parse_csv(raw: str) -> List[str]:
    s = (raw or "").replace(";", ",").replace(" ", ",")
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_device(raw: str) -> str:
    v = (raw or "auto").strip().lower()
    if v in ("", "auto", "none"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return v


def _probe_cuda() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        x = torch.randn(1, 3, 8, 8, device="cuda")
        m = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).cuda()
        _ = m(x)
        return True, "ok"
    except Exception as e:  # pragma: no cover - hardware dependent
        return False, f"{type(e).__name__}: {e}"


def _autocast_ctx(device: str):
    if device.startswith("cuda"):
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype), str(dtype)
    return contextlib.nullcontext(), "none"


def _to_u8(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_u8(path: Path, arr_u8_hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8_hwc).save(path)


def _split_triptych_u8(cat_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = cat_u8.shape
    if w % 3 != 0:
        raise ValueError(f"triptych width must be divisible by 3, got {w}")
    w1 = w // 3
    return cat_u8[:, :w1], cat_u8[:, w1:2 * w1], cat_u8[:, 2 * w1:]


def _find_frame_image(seq_dir: Path, cam: str, frame_id: int) -> Path:
    cam_dir = seq_dir / cam
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera dir not found: {cam_dir}")
    stem = f"{int(frame_id):06d}"
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        p = cam_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"image not found for cam={cam} frame={stem} under {cam_dir}")


def _tensor_stats_torch(x: torch.Tensor) -> Dict:
    y = x.detach().float().cpu()
    return {
        "shape": list(y.shape),
        "dtype": str(x.dtype),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "std": float(y.std().item()),
    }


def _camera_center_from_extrinsic(extrinsic_3x4: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic_3x4, dtype=np.float64)
    r = e[:, :3]
    t = e[:, 3]
    return -r.T @ t


def _camera_centers_batch(extrinsics_3x4: np.ndarray) -> np.ndarray:
    out = []
    for i in range(extrinsics_3x4.shape[0]):
        out.append(_camera_center_from_extrinsic(extrinsics_3x4[i]))
    return np.asarray(out, dtype=np.float64)


def _umeyama_similarity(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solve y ~= s * R * x + t for row-wise points.
    x, y: (N,3)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"invalid shapes x={x.shape}, y={y.shape}")
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points for Sim(3)")

    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    xc = x - mx
    yc = y - my

    cov = (yc.T @ xc) / float(n)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[-1, -1] = -1.0
    r = u @ s_fix @ vt

    var_x = float((xc * xc).sum() / float(n))
    if var_x <= 1e-12:
        raise ValueError("degenerate source points for Sim(3)")
    scale = float(np.trace(np.diag(d) @ s_fix) / var_x)
    t = my - scale * (r @ mx)
    return scale, r, t


def _apply_sim3_points(points: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return scale * (p @ r.T) + t.reshape(1, 3)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=-1))))


def _read_gt_camera_params(seq_dir: Path, cameras: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    intri_path = seq_dir / "intri.yml"
    extri_path = seq_dir / "extri.yml"
    if not intri_path.exists():
        raise FileNotFoundError(f"intri file not found: {intri_path}")
    if not extri_path.exists():
        raise FileNotFoundError(f"extri file not found: {extri_path}")

    fs_i = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    fs_e = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not fs_i.isOpened():
        raise RuntimeError(f"failed to open intri.yml: {intri_path}")
    if not fs_e.isOpened():
        raise RuntimeError(f"failed to open extri.yml: {extri_path}")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    try:
        for cam in cameras:
            node_k = fs_i.getNode(f"K_{cam}")
            node_r = fs_e.getNode(f"Rot_{cam}")
            node_t = fs_e.getNode(f"T_{cam}")
            if node_k.empty():
                raise KeyError(f"missing K_{cam} in {intri_path}")
            if node_r.empty():
                raise KeyError(f"missing Rot_{cam} in {extri_path}")
            if node_t.empty():
                raise KeyError(f"missing T_{cam} in {extri_path}")
            k = node_k.mat().astype(np.float64)
            r = node_r.mat().astype(np.float64)
            t = node_t.mat().astype(np.float64).reshape(3, 1)
            ex = np.concatenate([r, t], axis=1)
            out[cam] = {"K": k, "R": r, "t": t.reshape(3), "extrinsic": ex}
    finally:
        fs_i.release()
        fs_e.release()
    return out


def _scale_intrinsic(k: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    h0, w0 = src_hw
    h1, w1 = dst_hw
    sx = float(w1) / float(w0)
    sy = float(h1) / float(h0)
    kk = np.asarray(k, dtype=np.float64).copy()
    kk[0, 0] *= sx
    kk[1, 1] *= sy
    kk[0, 2] *= sx
    kk[1, 2] *= sy
    return kk


def _detect_target_from_anchor(
    anchor_cat_path: Path,
    seq_dir: Path,
    frame_id: int,
    target_hw: Tuple[int, int],
) -> Dict:
    if not anchor_cat_path.exists():
        raise FileNotFoundError(f"anchor cat image not found: {anchor_cat_path}")
    cat = np.array(Image.open(anchor_cat_path).convert("RGB"), dtype=np.uint8)
    _, anchor_mid, anchor_tgt = _split_triptych_u8(cat)

    h, w = target_hw
    anchor_tgt_rs = np.array(Image.fromarray(anchor_tgt).resize((w, h), Image.BILINEAR), dtype=np.uint8)

    cams = sorted([p.name for p in seq_dir.iterdir() if p.is_dir() and p.name.startswith("Camera_")])
    best_mae = float("inf")
    best_cam = None
    best_path = None
    for cam in cams:
        try:
            p = _find_frame_image(seq_dir, cam, frame_id)
        except FileNotFoundError:
            continue
        img = np.array(Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8)
        mae = float(np.abs(img.astype(np.float32) - anchor_tgt_rs.astype(np.float32)).mean())
        if mae < best_mae:
            best_mae = mae
            best_cam = cam
            best_path = p
    if best_cam is None:
        raise RuntimeError("failed to match anchor target camera at given frame")

    return {
        "best_camera": best_cam,
        "best_image_path": str(best_path),
        "best_mae": best_mae,
        "anchor_mid": anchor_mid,
        "anchor_tgt": anchor_tgt,
    }


def _render_forward_splat(
    world_points_s_hw3: np.ndarray,
    world_conf_s_hw: np.ndarray,
    src_rgb_s_hw3: np.ndarray,
    tgt_extrinsic_3x4: np.ndarray,
    tgt_intrinsic_3x3: np.ndarray,
    out_hw: Tuple[int, int],
    z_eps: float,
    min_conf: float,
    z_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    h, w = out_hw
    pts = world_points_s_hw3.reshape(-1, 3).astype(np.float64)
    conf = world_conf_s_hw.reshape(-1).astype(np.float64)
    rgb = src_rgb_s_hw3.reshape(-1, 3).astype(np.float64)

    r = tgt_extrinsic_3x4[:, :3]
    t = tgt_extrinsic_3x4[:, 3]
    x_cam = pts @ r.T + t.reshape(1, 3)
    z = x_cam[:, 2]

    valid = np.isfinite(x_cam).all(axis=1) & np.isfinite(conf)
    valid &= (z > float(z_eps))
    valid &= (conf >= float(min_conf))
    if not np.any(valid):
        pred = np.zeros((h, w, 3), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)
        stats = {"coverage_ratio": 0.0, "valid_contrib": 0, "mean_conf": 0.0}
        return pred, weight, stats

    x = x_cam[:, 0] / (z + 1e-12)
    y = x_cam[:, 1] / (z + 1e-12)
    u = tgt_intrinsic_3x3[0, 0] * x + tgt_intrinsic_3x3[0, 2]
    v = tgt_intrinsic_3x3[1, 1] * y + tgt_intrinsic_3x3[1, 2]

    xi = np.rint(u).astype(np.int64)
    yi = np.rint(v).astype(np.int64)
    valid &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(valid):
        pred = np.zeros((h, w, 3), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)
        stats = {"coverage_ratio": 0.0, "valid_contrib": 0, "mean_conf": 0.0}
        return pred, weight, stats

    pix = yi[valid] * int(w) + xi[valid]
    z_v = z[valid]
    conf_v = conf[valid]
    rgb_v = rgb[valid]

    order = np.lexsort((z_v, pix))
    pix_s = pix[order]
    z_s = z_v[order]
    conf_s = conf_v[order]
    rgb_s = rgb_v[order]

    uniq_pix, start_idx, counts = np.unique(pix_s, return_index=True, return_counts=True)
    z_min = z_s[start_idx]
    z_min_rep = np.repeat(z_min, counts)
    keep = z_s <= (z_min_rep + float(z_tolerance))

    pix_k = pix_s[keep]
    conf_k = conf_s[keep]
    rgb_k = rgb_s[keep]

    total = int(h * w)
    sum_w = np.bincount(pix_k, weights=conf_k, minlength=total).astype(np.float64)
    sum_r = np.bincount(pix_k, weights=conf_k * rgb_k[:, 0], minlength=total).astype(np.float64)
    sum_g = np.bincount(pix_k, weights=conf_k * rgb_k[:, 1], minlength=total).astype(np.float64)
    sum_b = np.bincount(pix_k, weights=conf_k * rgb_k[:, 2], minlength=total).astype(np.float64)

    pred = np.zeros((total, 3), dtype=np.float64)
    hit = sum_w > 1e-12
    pred[hit, 0] = sum_r[hit] / sum_w[hit]
    pred[hit, 1] = sum_g[hit] / sum_w[hit]
    pred[hit, 2] = sum_b[hit] / sum_w[hit]
    pred = pred.reshape(h, w, 3)
    pred = np.clip(pred, 0.0, 1.0)

    w_map = sum_w.reshape(h, w)
    if np.any(hit):
        p99 = float(np.percentile(w_map[hit.reshape(h, w)], 99))
        denom = max(p99, 1e-8)
        w_norm = np.clip(w_map / denom, 0.0, 1.0)
    else:
        w_norm = np.zeros_like(w_map)

    stats = {
        "coverage_ratio": float(hit.mean()),
        "valid_contrib": int(conf_k.size),
        "mean_conf": float(conf_k.mean()) if conf_k.size > 0 else 0.0,
    }
    return pred, w_norm, stats


def _weight_to_rgb(weight01_hw: np.ndarray) -> np.ndarray:
    w = np.clip(weight01_hw, 0.0, 1.0)
    out = np.zeros((w.shape[0], w.shape[1], 3), dtype=np.float64)
    out[..., 1] = w
    out[..., 2] = w * 0.2
    return out


def _simple_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Global SSIM averaged over RGB channels.
    x, y: HxWx3 in [0,1]
    """
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    vals = []
    for c in range(3):
        a = x[..., c].astype(np.float64)
        b = y[..., c].astype(np.float64)
        mu_a = a.mean()
        mu_b = b.mean()
        var_a = a.var()
        var_b = b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        vals.append(float(num / max(den, 1e-12)))
    return float(np.mean(vals))


def _metrics(pred01: np.ndarray, tgt01: np.ndarray) -> Dict:
    diff = pred01 - tgt01
    mae = float(np.abs(diff).mean())
    mse = float((diff * diff).mean())
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    ssim = _simple_ssim(pred01, tgt01)
    return {"mae": mae, "psnr": psnr, "ssim": ssim}


def _resize01(img01: np.ndarray, size_hw: Tuple[int, int], mode: int = Image.BILINEAR) -> np.ndarray:
    h, w = size_hw
    u8 = _to_u8(img01)
    rs = np.array(Image.fromarray(u8).resize((w, h), mode), dtype=np.uint8)
    return rs.astype(np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser("Render VGGT raw geometric middle panel [weight,pred,tgt].")
    ap.add_argument("--zju_root", type=str, default=DEFAULT_ZJU_ROOT)
    ap.add_argument("--seq_name", type=str, default=DEFAULT_SEQ_NAME)
    ap.add_argument("--frame_id", type=int, default=DEFAULT_FRAME_ID)
    ap.add_argument("--tgt_camera", type=str, default=DEFAULT_TGT_CAMERA)
    ap.add_argument("--src_cameras", type=str, default=DEFAULT_SRC_CAMERAS)
    ap.add_argument("--model_path", type=str, default="model.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--anchor_cat_path", type=str, default=DEFAULT_ANCHOR)
    ap.add_argument("--out_dir", type=str, default=r"infer_out\vggt_raw_middle_compare")
    ap.add_argument("--render_size", type=int, default=518)
    ap.add_argument("--z_eps", type=float, default=1e-6)
    ap.add_argument("--min_conf", type=float, default=1e-6)
    ap.add_argument("--z_tolerance", type=float, default=0.02)
    args = ap.parse_args()

    t_begin = time.time()

    src_cameras = _parse_csv(args.src_cameras)
    if len(src_cameras) != 6:
        raise ValueError(f"src_cameras must contain 6 names, got {len(src_cameras)}: {src_cameras}")
    if args.tgt_camera in src_cameras:
        raise ValueError(f"tgt_camera must not be in src_cameras: {args.tgt_camera}")

    zju_root = Path(args.zju_root).resolve()
    seq_dir = zju_root / args.seq_name
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"sequence dir not found: {seq_dir}")

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.out_dir).resolve()
        / args.seq_name
        / f"frame_{int(args.frame_id):06d}_{args.tgt_camera}"
        / f"run_{run_stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    anchor_cat_path = Path(args.anchor_cat_path).resolve()
    # Use target image shape for anchor matching.
    tgt_path = _find_frame_image(seq_dir, args.tgt_camera, int(args.frame_id))
    tgt_img_full = np.array(Image.open(tgt_path).convert("RGB"), dtype=np.uint8)
    anchor_match = _detect_target_from_anchor(
        anchor_cat_path=anchor_cat_path,
        seq_dir=seq_dir,
        frame_id=int(args.frame_id),
        target_hw=tgt_img_full.shape[:2],
    )
    if anchor_match["best_camera"] != args.tgt_camera:
        raise RuntimeError(
            f"anchor match mismatch: expected {args.tgt_camera}, "
            f"but best camera is {anchor_match['best_camera']}, "
            f"mae={anchor_match['best_mae']:.6f}"
        )

    src_paths = [_find_frame_image(seq_dir, cam, int(args.frame_id)) for cam in src_cameras]
    (run_dir / "sources.txt").write_text(
        "\n".join(str(p) for p in src_paths) + "\n", encoding="utf-8"
    )

    device_req = _resolve_device(args.device)
    cuda_ok, cuda_msg = _probe_cuda()
    if device_req.startswith("cuda") and not cuda_ok:
        print(f"[warn] CUDA requested but unavailable; fallback to CPU. reason: {cuda_msg}")
        device = "cpu"
    elif device_req == "auto":
        device = "cuda" if cuda_ok else "cpu"
    else:
        device = device_req

    print(f"[env] device={device} torch={torch.__version__} cuda_probe_ok={cuda_ok}")

    t_load = time.time()
    model = VGGT().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model_load_sec = time.time() - t_load

    images = load_and_preprocess_images([str(p) for p in src_paths], mode="crop")
    h_native, w_native = int(images.shape[-2]), int(images.shape[-1])
    src_rgb = images.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float64)
    images_dev = images.to(device)

    t_inf = time.time()
    amp_ctx, amp_dtype = _autocast_ctx(device)
    with torch.no_grad():
        with amp_ctx:
            pred = model(images_dev)
    infer_sec = time.time() - t_inf

    extrinsic_pred_t, intrinsic_pred_t = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic_pred = extrinsic_pred_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
    intrinsic_pred = intrinsic_pred_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
    world_points = pred["world_points"].squeeze(0).detach().cpu().numpy().astype(np.float64)
    world_conf = pred["world_points_conf"].squeeze(0).detach().cpu().numpy().astype(np.float64)

    tensor_stats = {
        "input_shape": list(images.shape),
        "amp_dtype": amp_dtype,
        "model_load_sec": round(model_load_sec, 6),
        "inference_sec": round(infer_sec, 6),
        "prediction_keys": sorted(list(pred.keys())),
        "tensors": {},
        "decoded_camera": {
            "extrinsic_shape": list(extrinsic_pred.shape),
            "intrinsic_shape": list(intrinsic_pred.shape),
        },
    }
    for k, v in pred.items():
        if torch.is_tensor(v):
            tensor_stats["tensors"][k] = _tensor_stats_torch(v)
    (run_dir / "tensor_stats.json").write_text(json.dumps(tensor_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    gt_cam = _read_gt_camera_params(seq_dir, cameras=src_cameras + [args.tgt_camera])
    centers_pred = _camera_centers_batch(extrinsic_pred)
    centers_gt = np.asarray(
        [_camera_center_from_extrinsic(gt_cam[c]["extrinsic"]) for c in src_cameras],
        dtype=np.float64,
    )

    rmse_before = _rmse(centers_pred, centers_gt)
    scale, r_sim3, t_sim3 = _umeyama_similarity(centers_pred, centers_gt)
    centers_aligned = _apply_sim3_points(centers_pred, scale, r_sim3, t_sim3)
    rmse_after = _rmse(centers_aligned, centers_gt)
    if rmse_after > 0.15:
        raise RuntimeError(f"Sim(3) alignment failed: rmse_after={rmse_after:.6f} > 0.15")

    world_points_aligned = _apply_sim3_points(world_points.reshape(-1, 3), scale, r_sim3, t_sim3).reshape(world_points.shape)

    render_size = int(args.render_size)
    if render_size <= 0:
        render_size = h_native
    render_hw = (render_size, render_size)

    tgt_img_native = np.array(
        Image.open(tgt_path).convert("RGB").resize((render_hw[1], render_hw[0]), Image.BILINEAR),
        dtype=np.uint8,
    ).astype(np.float64) / 255.0

    tgt_k = gt_cam[args.tgt_camera]["K"]
    tgt_ex = gt_cam[args.tgt_camera]["extrinsic"]
    tgt_k_scaled = _scale_intrinsic(tgt_k, src_hw=tgt_img_full.shape[:2], dst_hw=render_hw)

    pred_native, weight_native, render_stats = _render_forward_splat(
        world_points_s_hw3=world_points_aligned,
        world_conf_s_hw=world_conf,
        src_rgb_s_hw3=src_rgb,
        tgt_extrinsic_3x4=tgt_ex,
        tgt_intrinsic_3x3=tgt_k_scaled,
        out_hw=render_hw,
        z_eps=float(args.z_eps),
        min_conf=float(args.min_conf),
        z_tolerance=float(args.z_tolerance),
    )

    weight_rgb_native = _weight_to_rgb(weight_native)
    diff_native = np.abs(pred_native - tgt_img_native).mean(axis=2, keepdims=True)
    diff_native = np.repeat(np.clip(diff_native, 0.0, 1.0), 3, axis=2)

    out_1024_hw = (1024, 1024)
    pred_1024 = _resize01(pred_native, out_1024_hw, mode=Image.BILINEAR)
    tgt_1024 = _resize01(tgt_img_native, out_1024_hw, mode=Image.BILINEAR)
    weight_rgb_1024 = _resize01(weight_rgb_native, out_1024_hw, mode=Image.BILINEAR)
    diff_1024 = _resize01(diff_native, out_1024_hw, mode=Image.BILINEAR)

    cat_native = np.concatenate([weight_rgb_native, pred_native, tgt_img_native], axis=1)
    cat_1024 = np.concatenate([weight_rgb_1024, pred_1024, tgt_1024], axis=1)
    p0, p1, p2 = _split_triptych_u8(_to_u8(cat_1024))

    # Save images
    paths = {}
    paths["weight_native"] = str(run_dir / "weight_native.png")
    paths["pred_native"] = str(run_dir / "pred_native.png")
    paths["tgt_native"] = str(run_dir / "tgt_native.png")
    paths["cat_weight_pred_tgt_native"] = str(run_dir / "cat_weight_pred_tgt_native.png")
    paths["cat_weight_pred_tgt"] = str(run_dir / "cat_weight_pred_tgt.png")
    paths["cat_weight_pred_tgt_p0"] = str(run_dir / "cat_weight_pred_tgt_p0.png")
    paths["cat_weight_pred_tgt_p1"] = str(run_dir / "cat_weight_pred_tgt_p1.png")
    paths["cat_weight_pred_tgt_p2"] = str(run_dir / "cat_weight_pred_tgt_p2.png")
    paths["pred_tgt_diff"] = str(run_dir / "pred_tgt_diff.png")

    _save_u8(Path(paths["weight_native"]), _to_u8(weight_rgb_native))
    _save_u8(Path(paths["pred_native"]), _to_u8(pred_native))
    _save_u8(Path(paths["tgt_native"]), _to_u8(tgt_img_native))
    _save_u8(Path(paths["cat_weight_pred_tgt_native"]), _to_u8(cat_native))
    _save_u8(Path(paths["cat_weight_pred_tgt"]), _to_u8(cat_1024))
    _save_u8(Path(paths["cat_weight_pred_tgt_p0"]), p0)
    _save_u8(Path(paths["cat_weight_pred_tgt_p1"]), p1)
    _save_u8(Path(paths["cat_weight_pred_tgt_p2"]), p2)
    _save_u8(Path(paths["pred_tgt_diff"]), _to_u8(diff_1024))

    # Optional anchor comparison collage: [anchor_mid, ours_pred, tgt]
    try:
        anchor_mid = anchor_match["anchor_mid"]
        anchor_mid_1024 = np.array(Image.fromarray(anchor_mid).resize((1024, 1024), Image.BILINEAR), dtype=np.uint8)
        ours_pred_1024_u8 = _to_u8(pred_1024)
        tgt_1024_u8 = _to_u8(tgt_1024)
        cmp_cat = np.concatenate([anchor_mid_1024, ours_pred_1024_u8, tgt_1024_u8], axis=1)
        cmp_path = run_dir / "anchor_pred_tgt_compare.png"
        _save_u8(cmp_path, cmp_cat)
        paths["anchor_pred_tgt_compare"] = str(cmp_path)
    except Exception:
        pass

    m_native = _metrics(pred_native, tgt_img_native)
    m_1024 = _metrics(pred_1024, tgt_1024)

    meta = {
        "time": datetime.now().isoformat(),
        "zju_root": str(zju_root),
        "seq_name": args.seq_name,
        "frame_id": int(args.frame_id),
        "src_cameras": src_cameras,
        "tgt_camera": args.tgt_camera,
        "src_image_paths": [str(p) for p in src_paths],
        "tgt_image_path": str(tgt_path),
        "anchor_cat_path": str(anchor_cat_path),
        "anchor_detected_best_camera": anchor_match["best_camera"],
        "anchor_detected_best_image_path": anchor_match["best_image_path"],
        "anchor_detected_best_mae": float(anchor_match["best_mae"]),
        "python": os.sys.executable,
        "torch": torch.__version__,
        "device": device,
        "amp_dtype": amp_dtype,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    sim3 = {
        "scale": float(scale),
        "R": r_sim3.tolist(),
        "t": t_sim3.tolist(),
        "src_center_rmse_before": float(rmse_before),
        "src_center_rmse_after": float(rmse_after),
    }
    render = {
        "render_size": [int(render_hw[0]), int(render_hw[1])],
        "coverage_ratio": float(render_stats["coverage_ratio"]),
        "valid_contrib": int(render_stats["valid_contrib"]),
        "mean_conf": float(render_stats["mean_conf"]),
        "z_eps": float(args.z_eps),
        "min_conf": float(args.min_conf),
        "z_tolerance": float(args.z_tolerance),
    }
    report = {
        "meta": meta,
        "sim3": sim3,
        "render": render,
        "metrics": {
            "native": m_native,
            "upsampled_1024": m_1024,
        },
        "paths": paths,
        "elapsed_sec_total": float(time.time() - t_begin),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] report: {report_path}")
    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "sim3_rmse_before": sim3["src_center_rmse_before"],
            "sim3_rmse_after": sim3["src_center_rmse_after"],
            "coverage_ratio": render["coverage_ratio"],
            "metrics_1024": m_1024,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
