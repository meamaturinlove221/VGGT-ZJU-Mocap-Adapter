import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError, cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export 4K4D human_prior sidecars aligned to geom-cache coordinates."
    )
    ap.add_argument("--seq-root", required=True, help="Local bridged sequence root, e.g. out_vis/bridge_4k4d_med96/0012_11")
    ap.add_argument("--geom-root", default="", help="Local geom cache dir. Default: latest vggt_geom* under seq-root")
    ap.add_argument("--annots-smc", default="", help="Optional annotations SMC path. Default: read from bridge_manifest.json")
    ap.add_argument("--manifest-path", default="", help="Optional bridge_manifest.json path")
    ap.add_argument("--output-subdir", default="human_prior", help="Output sidecar subdir under seq-root")
    ap.add_argument("--frames", nargs="*", type=int, default=None, help="Optional explicit frame ids")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap after sorting geom npz files")
    ap.add_argument("--conf-threshold", type=float, default=0.20, help="2D keypoint confidence threshold")
    ap.add_argument("--min-triangulated-views", type=int, default=4, help="Minimum views for multi-view triangulation")
    ap.add_argument("--max-triangulation-views", type=int, default=24, help="Use at most this many highest-confidence views per joint")
    ap.add_argument("--reproj-threshold-px", type=float, default=12.0, help="Remove worst triangulation view while max reproj error exceeds this")
    ap.add_argument("--min-align-joints", type=int, default=8, help="Minimum aligned joints required to estimate sim3")
    ap.add_argument("--mask-threshold", type=int, default=127, help="Binary threshold for bridged masks")
    ap.add_argument("--head-top-ratio", type=float, default=0.32, help="Top-of-body fallback ratio for head mask")
    ap.add_argument("--face-top-ratio", type=float, default=0.18, help="Top-of-head/body fallback ratio for face mask")
    ap.add_argument("--smplx-model-dir", default="", help="Optional SMPL-X model dir containing SMPLX_* files")
    ap.add_argument("--smplx-gender", default="neutral", choices=["neutral", "female", "male"], help="SMPL-X gender model")
    ap.add_argument("--smplx-ext", default="npz", choices=["npz", "pkl"], help="SMPL-X model file extension")
    ap.add_argument("--disable-smplx", action="store_true", help="Disable SMPL-X mesh export and fall back to sparse keypoint interpolation")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecars")
    return ap.parse_args()


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    pairs: int
    inliers: int
    source: str


@dataclass
class SmplxRuntime:
    model: Any
    model_dir: str
    gender: str
    ext: str


def _resolve_manifest_path(seq_root: Path, manifest_path: str) -> Path:
    if manifest_path:
        return Path(manifest_path).expanduser().resolve()
    return (seq_root / "bridge_manifest.json").resolve()


def _resolve_annots_smc(seq_root: Path, annots_smc: str, manifest_path: str) -> Path:
    if annots_smc:
        return Path(annots_smc).expanduser().resolve()
    manifest = json.loads(_resolve_manifest_path(seq_root, manifest_path).read_text(encoding="utf-8"))
    raw = manifest.get("annotations_smc", "")
    if not raw:
        raise RuntimeError(f"annotations_smc missing in manifest: {seq_root / 'bridge_manifest.json'}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"annotations SMC not found: {path}")
    return path.resolve()


def _resolve_geom_root(seq_root: Path, geom_root: str) -> Path:
    if geom_root:
        root = Path(geom_root).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"geom root not found: {root}")
        return root
    candidates = [p for p in seq_root.iterdir() if p.is_dir() and p.name.startswith("vggt_geom")]
    if not candidates:
        raise RuntimeError(f"no vggt_geom* directory found under {seq_root}")
    candidates.sort(key=lambda p: (len(list(p.glob("*.npz"))), p.stat().st_mtime), reverse=True)
    return candidates[0].resolve()


def _resolve_smplx_model_dir(raw: str, disable_smplx: bool) -> Optional[Path]:
    if disable_smplx:
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"SMPL-X model dir not found: {path}")
    return path


def _frame_id_from_stem(stem: str) -> int:
    token = stem.split("_")[-1]
    return int(token)


def _select_geom_npz_paths(geom_root: Path, frames: Optional[Sequence[int]], max_frames: int) -> list[Path]:
    files = sorted([p for p in geom_root.glob("frame_*.npz") if p.is_file()])
    if frames:
        wanted = {int(x) for x in frames}
        files = [p for p in files if _frame_id_from_stem(p.stem) in wanted]
    if max_frames > 0:
        files = files[: int(max_frames)]
    if not files:
        raise RuntimeError(f"no geom npz files selected under {geom_root}")
    return files


def _build_top_band_mask(mask01: np.ndarray, top_ratio: float, min_height_px: int) -> np.ndarray:
    fg = np.asarray(mask01, dtype=np.float32) > 0.5
    out = np.zeros_like(fg, dtype=np.float32)
    ys, xs = np.nonzero(fg)
    if ys.size <= 0:
        return out
    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())
    band_h = max(int(min_height_px), int(round((y1 - y0 + 1) * float(np.clip(top_ratio, 0.01, 1.0)))))
    band_h = min(band_h, y1 - y0 + 1)
    yb1 = min(mask01.shape[0] - 1, y0 + band_h - 1)
    out[y0 : yb1 + 1, x0 : x1 + 1] = 1.0
    return out


def _camera_name_to_id(name: str) -> str:
    token = str(name).split("_")[-1]
    return f"{int(token):02d}"


def _preprocess_xy_to_518(xy: np.ndarray, width: int, height: int, target_size: int = 518) -> np.ndarray:
    scale = float(target_size) / float(max(width, 1))
    new_h = int(round(float(height) * scale / 14.0) * 14.0)
    crop_top = max((new_h - int(target_size)) // 2, 0)
    out = np.asarray(xy, dtype=np.float32).copy()
    out[..., 0] *= float(scale)
    out[..., 1] = out[..., 1] * float(scale) - float(crop_top)
    return out


def _load_processed_mask(mask_path: Path, threshold: int, target_size: int = 518) -> np.ndarray:
    if not mask_path.is_file():
        return np.ones((target_size, target_size), dtype=np.float32)
    img = Image.open(mask_path).convert("L")
    width, height = img.size
    new_width = target_size
    new_height = int(round(height * (new_width / float(max(width, 1))) / 14.0) * 14.0)
    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        arr = arr[start_y : start_y + target_size]
    elif new_height < target_size:
        pad = target_size - new_height
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0)
    return (arr > int(threshold)).astype(np.float32)


def _bilinear_sample_pointmap(pointmap: np.ndarray, xy: np.ndarray) -> Optional[np.ndarray]:
    h, w = pointmap.shape[:2]
    x = float(xy[0])
    y = float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if x < 0.0 or y < 0.0 or x > (w - 1) or y > (h - 1):
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = x - float(x0)
    wy = y - float(y0)
    p00 = pointmap[y0, x0]
    p10 = pointmap[y0, x1]
    p01 = pointmap[y1, x0]
    p11 = pointmap[y1, x1]
    sample = (
        (1.0 - wx) * (1.0 - wy) * p00
        + wx * (1.0 - wy) * p10
        + (1.0 - wx) * wy * p01
        + wx * wy * p11
    )
    if not np.isfinite(sample).all():
        return None
    return np.asarray(sample, dtype=np.float32)


def _cam_to_world_points(points_cam: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    r = np.asarray(extrinsic_w2c[:3, :3], dtype=np.float32)
    t = np.asarray(extrinsic_w2c[:3, 3], dtype=np.float32)
    centered = np.asarray(points_cam, dtype=np.float32) - t[None, :]
    return centered @ r


def _self_reproj_err_px(point_world: np.ndarray, extrinsic_w2c: np.ndarray, intrinsic: np.ndarray) -> float:
    point_world = np.asarray(point_world, dtype=np.float32)
    extrinsic_w2c = np.asarray(extrinsic_w2c, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if point_world.ndim != 4 or point_world.shape[-1] != 3:
        return float("inf")
    h, w = point_world.shape[1:3]
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cam = np.einsum("vij,vhwj->vhwi", extrinsic_w2c[:, :3, :3], point_world) + extrinsic_w2c[:, None, None, :3, 3]
    z = cam[..., 2]
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    u = fx * (cam[..., 0] / np.maximum(z, 1e-8)) + cx
    v = fy * (cam[..., 1] / np.maximum(z, 1e-8)) + cy
    err = np.sqrt((u - xs[None]) ** 2 + (v - ys[None]) ** 2)
    valid = np.isfinite(err) & np.isfinite(z) & (z > 1e-6)
    if int(valid.sum()) <= 0:
        return float("inf")
    return float(err[valid].mean())


def _resolve_pointmap_world(pointmap: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, declared_frame: str) -> tuple[np.ndarray, str, Dict[str, float]]:
    mode = str(declared_frame or "").strip().lower()
    if mode == "world":
        return np.asarray(pointmap, dtype=np.float32), "world", {}
    if mode == "camera":
        cam_world = np.stack(
            [_cam_to_world_points(np.asarray(pointmap[v], dtype=np.float32).reshape(-1, 3), extrinsic[v]).reshape(pointmap[v].shape) for v in range(pointmap.shape[0])],
            axis=0,
        ).astype(np.float32)
        return cam_world, "camera", {}
    world = np.asarray(pointmap, dtype=np.float32)
    cam2world = np.stack([_cam_to_world_points(world[v].reshape(-1, 3), extrinsic[v]).reshape(world[v].shape) for v in range(world.shape[0])], axis=0)
    err_world = _self_reproj_err_px(world, extrinsic, intrinsic)
    err_cam = _self_reproj_err_px(cam2world, extrinsic, intrinsic)
    if err_cam < err_world:
        return cam2world.astype(np.float32, copy=False), "camera", {
            "point_frame_err_world": float(err_world),
            "point_frame_err_camera": float(err_cam),
        }
    return world.astype(np.float32, copy=False), "world", {
        "point_frame_err_world": float(err_world),
        "point_frame_err_camera": float(err_cam),
    }


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


def _apply_sim3(points_xyz: np.ndarray, sim3: Sim3) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    return (sim3.scale * (pts @ sim3.rotation.T) + sim3.translation[None, :]).astype(np.float32)


def _merge_duplicate_seeds(seed_xy: np.ndarray, seed_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(np.asarray(seed_xy, dtype=np.float64), 3)
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
    xyz = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    counts = np.zeros((uniq.shape[0], 1), dtype=np.float64)
    for idx in range(inv.shape[0]):
        xyz[inv[idx]] += np.asarray(seed_xyz[idx], dtype=np.float64)
        counts[inv[idx], 0] += 1.0
    xyz /= np.maximum(counts, 1.0)
    return uniq.astype(np.float32), xyz.astype(np.float32)


def _interpolate_pointmap(seed_xy: np.ndarray, seed_xyz: np.ndarray, body_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body = np.asarray(body_mask, dtype=np.float32) > 0.5
    h, w = body.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.float32)
    if int(body.sum()) <= 0 or seed_xy.shape[0] <= 0:
        return out, valid
    xy, xyz = _merge_duplicate_seeds(seed_xy=seed_xy, seed_xyz=seed_xyz)
    ys, xs = np.nonzero(body)
    queries = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    values = np.zeros((queries.shape[0], 3), dtype=np.float32)
    if xy.shape[0] >= 3:
        try:
            for dim in range(3):
                interp = LinearNDInterpolator(xy, xyz[:, dim], fill_value=np.nan)
                values[:, dim] = np.asarray(interp(queries), dtype=np.float32)
        except (QhullError, ValueError):
            values[:] = np.nan
    missing = ~np.isfinite(values).all(axis=-1)
    if np.any(missing):
        for dim in range(3):
            interp = NearestNDInterpolator(xy, xyz[:, dim])
            values[missing, dim] = np.asarray(interp(queries[missing]), dtype=np.float32)
    out[ys, xs] = values
    valid[ys, xs] = np.isfinite(values).all(axis=-1).astype(np.float32)
    return out, valid


def _triangulate_joint(
    observations: Sequence[tuple[np.ndarray, np.ndarray, float]],
    min_views: int,
    reproj_threshold_px: float,
) -> tuple[Optional[np.ndarray], int, float]:
    obs = list(observations)
    if len(obs) < int(min_views):
        return None, 0, float("inf")
    while len(obs) >= int(min_views):
        a_rows: list[np.ndarray] = []
        for proj, xy, _ in obs:
            u = float(xy[0])
            v = float(xy[1])
            a_rows.append(u * proj[2] - proj[0])
            a_rows.append(v * proj[2] - proj[1])
        a = np.stack(a_rows, axis=0).astype(np.float64)
        _, _, vt = np.linalg.svd(a)
        x_h = vt[-1]
        if abs(float(x_h[3])) <= 1e-12:
            return None, 0, float("inf")
        xyz = (x_h[:3] / x_h[3]).astype(np.float64)
        errs: list[float] = []
        for proj, xy, _ in obs:
            proj_xyz = np.asarray(proj, dtype=np.float64) @ np.concatenate([xyz, np.ones(1, dtype=np.float64)], axis=0)
            if proj_xyz[2] <= 1e-8:
                errs.append(float("inf"))
                continue
            uv = proj_xyz[:2] / proj_xyz[2]
            errs.append(float(np.linalg.norm(uv - np.asarray(xy, dtype=np.float64))))
        max_err = max(errs)
        if max_err <= float(reproj_threshold_px) or len(obs) == int(min_views):
            return xyz.astype(np.float32), len(obs), float(np.mean(errs))
        worst = int(np.argmax(np.asarray(errs, dtype=np.float64)))
        obs.pop(worst)
    return None, 0, float("inf")


def _estimate_sim3(src_xyz: np.ndarray, dst_xyz: np.ndarray, min_align_joints: int) -> Sim3:
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    if src.shape[0] < int(min_align_joints):
        raise RuntimeError(f"need at least {min_align_joints} aligned joints, got {src.shape[0]}")
    scale, rotation, translation = _umeyama_similarity(src, dst)
    pred = scale * (src @ rotation.T) + translation[None, :]
    resid = np.linalg.norm(pred - dst, axis=-1)
    if resid.shape[0] >= max(int(min_align_joints), 6):
        keep_thr = float(np.percentile(resid, 75.0))
        inlier = resid <= max(keep_thr, 1e-6)
        if int(inlier.sum()) >= int(min_align_joints):
            scale, rotation, translation = _umeyama_similarity(src[inlier], dst[inlier])
            pred = scale * (src @ rotation.T) + translation[None, :]
            resid = np.linalg.norm(pred - dst, axis=-1)
            return Sim3(
                scale=float(scale),
                rotation=np.asarray(rotation, dtype=np.float32),
                translation=np.asarray(translation, dtype=np.float32),
                rmse=float(np.sqrt(np.mean(resid * resid))),
                pairs=int(src.shape[0]),
                inliers=int(inlier.sum()),
                source="current_frame_refit",
            )
    return Sim3(
        scale=float(scale),
        rotation=np.asarray(rotation, dtype=np.float32),
        translation=np.asarray(translation, dtype=np.float32),
        rmse=float(np.sqrt(np.mean(resid * resid))),
        pairs=int(src.shape[0]),
        inliers=int(src.shape[0]),
        source="current_frame",
    )


def _load_h5_array(node: Any, preferred_keys: Sequence[str]) -> Optional[np.ndarray]:
    if isinstance(node, h5py.Dataset):
        return np.asarray(node)
    if not isinstance(node, h5py.Group):
        return None
    for key in preferred_keys:
        if key in node and isinstance(node[key], h5py.Dataset):
            return np.asarray(node[key])
    for key in sorted(node.keys()):
        if isinstance(node[key], h5py.Dataset):
            return np.asarray(node[key])
    return None


def _init_smplx_runtime(model_dir: Optional[Path], gender: str, ext: str) -> Optional[SmplxRuntime]:
    if model_dir is None:
        return None
    try:
        import smplx
    except Exception as exc:
        raise RuntimeError(f"failed to import smplx runtime: {exc}") from exc
    model_root = Path(model_dir)
    leaf_file = model_root / f"SMPLX_{str(gender).upper()}.{str(ext)}"
    if leaf_file.is_file():
        model_root = model_root.parent
    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=str(gender),
        ext=str(ext),
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    )
    model.eval()
    return SmplxRuntime(
        model=model,
        model_dir=str(model_root),
        gender=str(gender),
        ext=str(ext),
    )


def _run_smplx_frame(runtime: SmplxRuntime, smplx_params: Dict[str, np.ndarray], frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    fullpose = np.asarray(smplx_params["fullpose"][frame_id], dtype=np.float32)
    if fullpose.shape != (55, 3):
        raise RuntimeError(f"unexpected SMPL-X fullpose shape at frame {frame_id}: {fullpose.shape}")
    betas = np.asarray(smplx_params["betas"][frame_id], dtype=np.float32).reshape(1, -1)
    expression = np.asarray(smplx_params["expression"][frame_id], dtype=np.float32).reshape(1, -1)
    transl = np.asarray(smplx_params["transl"][frame_id], dtype=np.float32).reshape(1, 3)
    scale = float(np.asarray(smplx_params.get("scale", np.asarray([1.0], dtype=np.float32))).reshape(-1)[0])
    with torch.no_grad():
        out = runtime.model(
            betas=torch.from_numpy(betas),
            expression=torch.from_numpy(expression),
            transl=torch.from_numpy(transl),
            global_orient=torch.from_numpy(fullpose[0:1].reshape(1, 3)),
            body_pose=torch.from_numpy(fullpose[1:22].reshape(1, 21, 3)),
            jaw_pose=torch.from_numpy(fullpose[22:23].reshape(1, 3)),
            leye_pose=torch.from_numpy(fullpose[23:24].reshape(1, 3)),
            reye_pose=torch.from_numpy(fullpose[24:25].reshape(1, 3)),
            left_hand_pose=torch.from_numpy(fullpose[25:40].reshape(1, 15, 3)),
            right_hand_pose=torch.from_numpy(fullpose[40:55].reshape(1, 15, 3)),
            return_verts=True,
        )
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    if math.isfinite(scale) and abs(scale - 1.0) > 1e-6:
        verts *= float(scale)
        joints *= float(scale)
    return verts, joints


def _project_world_points(points_world: np.ndarray, extrinsic_w2c: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float32)
    cam = pts @ np.asarray(extrinsic_w2c[:3, :3], dtype=np.float32).T + np.asarray(extrinsic_w2c[:3, 3], dtype=np.float32)[None, :]
    z = cam[:, 2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    valid = np.isfinite(cam).all(axis=-1) & np.isfinite(z) & (z > 1e-6)
    xy = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        xy_valid = np.stack(
            [
                fx * (cam[valid, 0] / z[valid]) + cx,
                fy * (cam[valid, 1] / z[valid]) + cy,
            ],
            axis=-1,
        ).astype(np.float32)
        xy[valid] = xy_valid
    return xy, z.astype(np.float32), valid


def _rasterize_vertex_splat(
    points_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    body_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = body_mask.shape
    xy, depth, valid = _project_world_points(points_world=points_world, extrinsic_w2c=extrinsic_w2c, intrinsic=intrinsic)
    xi = np.rint(xy[:, 0]).astype(np.int32)
    yi = np.rint(xy[:, 1]).astype(np.int32)
    inside = (
        valid
        & np.isfinite(xy).all(axis=-1)
        & (xi >= 0)
        & (xi < w)
        & (yi >= 0)
        & (yi < h)
    )
    if not np.any(inside):
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32), 0
    keep = inside & (body_mask[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)] > 0.5)
    if not np.any(keep):
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32), 0
    seed_map = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    order = np.argsort(depth[keep])
    keep_idx = np.where(keep)[0][order]
    for idx in keep_idx:
        x = int(xi[idx])
        y = int(yi[idx])
        if float(depth[idx]) < float(zbuf[y, x]):
            zbuf[y, x] = float(depth[idx])
            seed_map[y, x] = np.asarray(points_world[idx], dtype=np.float32)
    seed_valid = np.isfinite(zbuf).astype(np.float32)
    return seed_map, seed_valid, int(seed_valid.sum())


def _nearest_fill_from_seed_map(seed_map: np.ndarray, seed_valid: np.ndarray, body_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body = np.asarray(body_mask, dtype=np.float32) > 0.5
    out = np.zeros_like(seed_map, dtype=np.float32)
    valid = np.zeros(body.shape, dtype=np.float32)
    seed = np.asarray(seed_valid, dtype=np.float32) > 0.5
    if int(body.sum()) <= 0 or int(seed.sum()) <= 0:
        return out, valid
    seed_y, seed_x = np.nonzero(seed)
    seed_xy = np.stack([seed_x.astype(np.float32), seed_y.astype(np.float32)], axis=-1)
    seed_xyz = np.asarray(seed_map[seed_y, seed_x], dtype=np.float32)
    query_y, query_x = np.nonzero(body)
    query_xy = np.stack([query_x.astype(np.float32), query_y.astype(np.float32)], axis=-1)
    tree = cKDTree(np.asarray(seed_xy, dtype=np.float64))
    _, nn_idx = tree.query(np.asarray(query_xy, dtype=np.float64), k=1)
    out[query_y, query_x] = seed_xyz[np.asarray(nn_idx, dtype=np.int64)]
    valid[query_y, query_x] = 1.0
    return out, valid


def _load_annots_cache(
    annots_path: Path,
) -> tuple[
    dict[str, Dict[str, np.ndarray]],
    dict[str, np.ndarray],
    Optional[np.ndarray],
    Optional[dict[str, np.ndarray]],
]:
    cameras: dict[str, Dict[str, np.ndarray]] = {}
    keypoints_2d: dict[str, np.ndarray] = {}
    keypoints_3d: Optional[np.ndarray] = None
    smplx_params: Optional[dict[str, np.ndarray]] = None
    with h5py.File(annots_path, "r") as handle:
        cam_group = handle["Camera_Parameter"]
        kpt_group = handle["Keypoints_2D"]
        for cam_id in sorted(cam_group.keys(), key=lambda x: int(x)):
            node = cam_group[cam_id]
            cameras[cam_id] = {
                "K": np.asarray(node["K"], dtype=np.float64),
                "D": np.asarray(node["D"], dtype=np.float64).reshape(-1),
                "RT": np.asarray(node["RT"], dtype=np.float64)[:3, :4],
            }
            keypoints_2d[cam_id] = np.asarray(kpt_group[cam_id], dtype=np.float32)
        if "Keypoints_3D" in handle:
            keypoints_3d_arr = _load_h5_array(handle["Keypoints_3D"], preferred_keys=("keypoints3d", "keypoints_3d"))
            if keypoints_3d_arr is not None:
                keypoints_3d = np.asarray(keypoints_3d_arr, dtype=np.float32)
        if "SMPLx" in handle:
            smpl_group = handle["SMPLx"]
            required = ("betas", "expression", "fullpose", "transl")
            if all(key in smpl_group for key in required):
                smplx_params = {
                    "betas": np.asarray(smpl_group["betas"], dtype=np.float32),
                    "expression": np.asarray(smpl_group["expression"], dtype=np.float32),
                    "fullpose": np.asarray(smpl_group["fullpose"], dtype=np.float32),
                    "transl": np.asarray(smpl_group["transl"], dtype=np.float32),
                    "scale": np.asarray(smpl_group["scale"], dtype=np.float32) if "scale" in smpl_group else np.asarray([1.0], dtype=np.float32),
                }
    return cameras, keypoints_2d, keypoints_3d, smplx_params


def main() -> int:
    args = _parse_args()
    seq_root = Path(args.seq_root).expanduser().resolve()
    geom_root = _resolve_geom_root(seq_root=seq_root, geom_root=args.geom_root)
    annots_path = _resolve_annots_smc(seq_root=seq_root, annots_smc=args.annots_smc, manifest_path=args.manifest_path)
    output_root = (seq_root / str(args.output_subdir)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    geom_paths = _select_geom_npz_paths(
        geom_root=geom_root,
        frames=args.frames,
        max_frames=int(args.max_frames),
    )
    smplx_model_dir = _resolve_smplx_model_dir(raw=args.smplx_model_dir, disable_smplx=bool(args.disable_smplx))
    cameras, keypoints_2d, keypoints_3d, smplx_params = _load_annots_cache(annots_path=annots_path)
    smplx_runtime = _init_smplx_runtime(
        model_dir=smplx_model_dir,
        gender=str(args.smplx_gender),
        ext=str(args.smplx_ext),
    ) if smplx_model_dir is not None else None
    if smplx_model_dir is not None and smplx_params is None:
        raise RuntimeError(f"SMPL-X model dir provided, but SMPLx params are missing in {annots_path}")

    summary_rows: list[dict[str, Any]] = []
    last_good_sim3: Optional[Sim3] = None
    pointmap_frame_override: Optional[str] = None

    for geom_path in geom_paths:
        frame_id = _frame_id_from_stem(geom_path.stem)
        frame_name = f"{frame_id:06d}.png"
        out_path = output_root / geom_path.name
        if out_path.is_file() and not args.overwrite:
            summary_rows.append({
                "frame_id": int(frame_id),
                "status": "skipped_existing",
                "output_path": str(out_path),
            })
            continue

        with np.load(geom_path, allow_pickle=True) as data:
            cam_names = [str(x) for x in data["cam_names"].tolist()]
            pointmap = np.asarray(data["pointmap"], dtype=np.float32)
            extrinsic = np.asarray(data["extrinsic"], dtype=np.float32)
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
            pointmap_frame_raw = ""
            if "pointmap_frame" in data:
                raw = data["pointmap_frame"]
                if isinstance(raw, np.ndarray) and raw.size > 0:
                    pointmap_frame_raw = str(raw.reshape(-1)[0])
                else:
                    pointmap_frame_raw = str(raw)

        if pointmap_frame_override is None:
            pointmap_world, resolved_pointmap_frame, frame_info = _resolve_pointmap_world(
                pointmap=pointmap,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                declared_frame=pointmap_frame_raw,
            )
            pointmap_frame_override = str(resolved_pointmap_frame)
        else:
            if pointmap_frame_override == "camera":
                pointmap_world = np.stack(
                    [_cam_to_world_points(pointmap[v].reshape(-1, 3), extrinsic[v]).reshape(pointmap[v].shape) for v in range(pointmap.shape[0])],
                    axis=0,
                ).astype(np.float32)
            else:
                pointmap_world = pointmap.astype(np.float32, copy=False)
            frame_info = {}

        image_sizes: dict[str, tuple[int, int]] = {}
        body_masks: list[np.ndarray] = []
        for cam_name in cam_names:
            image_path = seq_root / cam_name / frame_name
            if not image_path.is_file():
                raise RuntimeError(f"missing bridged RGB image: {image_path}")
            with Image.open(image_path) as img:
                image_sizes[cam_name] = img.size
            body_masks.append(
                _load_processed_mask(
                    mask_path=seq_root / "mask" / cam_name / frame_name,
                    threshold=int(args.mask_threshold),
                )
            )

        body_masks_np = np.stack(body_masks, axis=0).astype(np.float32)
        selected_cam_ids = [_camera_name_to_id(name) for name in cam_names]
        joint_count = int(keypoints_2d[selected_cam_ids[0]].shape[1])

        source_keypoints_world = np.full((joint_count, 3), np.nan, dtype=np.float32)
        source_keypoints_valid = np.zeros((joint_count,), dtype=bool)
        if keypoints_3d is not None and frame_id < int(keypoints_3d.shape[0]):
            copy_count = min(int(keypoints_3d.shape[1]), joint_count)
            source_keypoints_world[:copy_count] = np.asarray(keypoints_3d[frame_id, :copy_count, :3], dtype=np.float32)
            source_keypoints_valid[:copy_count] = np.isfinite(source_keypoints_world[:copy_count]).all(axis=-1)
            if int(keypoints_3d.shape[-1]) > 3:
                source_keypoints_valid[:copy_count] &= (
                    np.isfinite(np.asarray(keypoints_3d[frame_id, :copy_count, 3], dtype=np.float32))
                    & (np.asarray(keypoints_3d[frame_id, :copy_count, 3], dtype=np.float32) > 0.0)
                )

        triangulated = np.full((joint_count, 3), np.nan, dtype=np.float32)
        triangulated_support = np.zeros((joint_count,), dtype=np.int32)
        triangulated_reproj = np.full((joint_count,), np.nan, dtype=np.float32)

        for joint_idx in range(joint_count):
            obs: list[tuple[np.ndarray, np.ndarray, float]] = []
            for cam_id in sorted(cameras.keys(), key=int):
                kp = np.asarray(keypoints_2d[cam_id][frame_id, joint_idx], dtype=np.float32)
                conf = float(kp[2])
                if not math.isfinite(conf) or conf < float(args.conf_threshold):
                    continue
                und = cv2.undistortPoints(
                    np.asarray(kp[:2], dtype=np.float64).reshape(1, 1, 2),
                    cameras[cam_id]["K"],
                    cameras[cam_id]["D"],
                    P=cameras[cam_id]["K"],
                ).reshape(2)
                proj = cameras[cam_id]["K"] @ cameras[cam_id]["RT"]
                obs.append((proj.astype(np.float64), und.astype(np.float64), conf))
            obs.sort(key=lambda item: float(item[2]), reverse=True)
            xyz, support_views, reproj_err = _triangulate_joint(
                observations=obs[: int(args.max_triangulation_views)],
                min_views=int(args.min_triangulated_views),
                reproj_threshold_px=float(args.reproj_threshold_px),
            )
            if xyz is None:
                continue
            triangulated[joint_idx] = np.asarray(xyz, dtype=np.float32)
            triangulated_support[joint_idx] = int(support_views)
            triangulated_reproj[joint_idx] = float(reproj_err)

        processed_keypoints_per_view: list[np.ndarray] = []
        for view_idx, cam_name in enumerate(cam_names):
            cam_id = selected_cam_ids[view_idx]
            width, height = image_sizes[cam_name]
            kps = np.asarray(keypoints_2d[cam_id][frame_id], dtype=np.float32)
            xy_proc = _preprocess_xy_to_518(kps[:, :2], width=width, height=height)
            processed = np.concatenate([xy_proc, kps[:, 2:3]], axis=-1).astype(np.float32)
            processed_keypoints_per_view.append(processed)

        pseudo_samples: list[list[np.ndarray]] = [[] for _ in range(joint_count)]
        for view_idx, processed in enumerate(processed_keypoints_per_view):
            body_mask = body_masks_np[view_idx]
            for joint_idx in range(joint_count):
                if not np.isfinite(triangulated[joint_idx]).all():
                    continue
                if float(processed[joint_idx, 2]) < float(args.conf_threshold):
                    continue
                xy = processed[joint_idx, :2]
                x_int = int(round(float(xy[0])))
                y_int = int(round(float(xy[1])))
                if x_int < 0 or y_int < 0 or x_int >= body_mask.shape[1] or y_int >= body_mask.shape[0]:
                    continue
                if body_mask[y_int, x_int] <= 0.5:
                    continue
                sample = _bilinear_sample_pointmap(pointmap_world[view_idx], xy)
                if sample is None:
                    continue
                pseudo_samples[joint_idx].append(sample)

        sim3: Optional[Sim3] = None
        align_source = ""
        aligned_keypoints_world = np.full((joint_count, 3), np.nan, dtype=np.float32)
        alignment_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        if int(source_keypoints_valid.sum()) > 0:
            alignment_candidates.append(("keypoints3d", source_keypoints_world, source_keypoints_valid))
        alignment_candidates.append(("triangulated", triangulated, np.isfinite(triangulated).all(axis=-1)))

        for candidate_name, candidate_xyz, candidate_valid in alignment_candidates:
            align_src: list[np.ndarray] = []
            align_dst: list[np.ndarray] = []
            for joint_idx in range(joint_count):
                if not bool(candidate_valid[joint_idx]):
                    continue
                if len(pseudo_samples[joint_idx]) <= 0:
                    continue
                align_src.append(np.asarray(candidate_xyz[joint_idx], dtype=np.float32))
                align_dst.append(np.median(np.stack(pseudo_samples[joint_idx], axis=0), axis=0))
            if len(align_src) < int(args.min_align_joints):
                continue
            try:
                sim3 = _estimate_sim3(
                    src_xyz=np.stack(align_src, axis=0),
                    dst_xyz=np.stack(align_dst, axis=0),
                    min_align_joints=int(args.min_align_joints),
                )
                last_good_sim3 = sim3
                align_source = candidate_name
                aligned_keypoints_world = _apply_sim3(candidate_xyz, sim3=sim3)
                break
            except Exception:
                continue

        if sim3 is None:
            if last_good_sim3 is None:
                raise RuntimeError(f"failed to estimate sim3 for {geom_path.name}")
            sim3 = Sim3(
                scale=float(last_good_sim3.scale),
                rotation=np.asarray(last_good_sim3.rotation, dtype=np.float32),
                translation=np.asarray(last_good_sim3.translation, dtype=np.float32),
                rmse=float(last_good_sim3.rmse),
                pairs=int(last_good_sim3.pairs),
                inliers=int(last_good_sim3.inliers),
                source="fallback_previous_frame",
            )
            align_source = "fallback_previous_frame"
            fallback_anchor = source_keypoints_world if int(source_keypoints_valid.sum()) > 0 else triangulated
            aligned_keypoints_world = _apply_sim3(fallback_anchor, sim3=sim3)

        smplx_vertices_world: Optional[np.ndarray] = None
        smplx_vertex_count = 0
        smplx_error = ""
        if smplx_runtime is not None and smplx_params is not None and frame_id < int(smplx_params["fullpose"].shape[0]):
            try:
                smplx_vertices_source, _ = _run_smplx_frame(runtime=smplx_runtime, smplx_params=smplx_params, frame_id=frame_id)
                smplx_vertices_world = _apply_sim3(smplx_vertices_source, sim3=sim3)
                smplx_vertex_count = int(smplx_vertices_world.shape[0])
            except Exception as exc:
                smplx_error = str(exc)

        prior_pointmap = np.zeros_like(pointmap_world, dtype=np.float32)
        prior_valid_mask = np.zeros(pointmap_world.shape[:3], dtype=np.float32)
        prior_head_mask = np.zeros(pointmap_world.shape[:3], dtype=np.float32)
        prior_face_mask = np.zeros(pointmap_world.shape[:3], dtype=np.float32)
        smplx_seed_pixels_per_view: list[int] = []
        mesh_views_used = 0
        sparse_views_used = 0

        for view_idx, processed in enumerate(processed_keypoints_per_view):
            used_mesh = False
            seed_pixels = 0
            if smplx_vertices_world is not None:
                seed_map, seed_valid, seed_pixels = _rasterize_vertex_splat(
                    points_world=smplx_vertices_world,
                    extrinsic_w2c=extrinsic[view_idx],
                    intrinsic=intrinsic[view_idx],
                    body_mask=body_masks_np[view_idx],
                )
                if seed_pixels > 0:
                    interp_map, interp_valid = _nearest_fill_from_seed_map(
                        seed_map=seed_map,
                        seed_valid=seed_valid,
                        body_mask=body_masks_np[view_idx],
                    )
                    prior_pointmap[view_idx] = interp_map
                    prior_valid_mask[view_idx] = interp_valid
                    mesh_views_used += 1
                    used_mesh = True
            if not used_mesh:
                finite = np.isfinite(aligned_keypoints_world).all(axis=-1)
                conf_ok = processed[:, 2] >= float(args.conf_threshold)
                inside = (
                    (processed[:, 0] >= 0.0)
                    & (processed[:, 0] <= float(body_masks_np.shape[2] - 1))
                    & (processed[:, 1] >= 0.0)
                    & (processed[:, 1] <= float(body_masks_np.shape[1] - 1))
                )
                keep = finite & conf_ok & inside
                seed_xy = processed[keep, :2]
                seed_xyz = aligned_keypoints_world[keep]
                interp_map, interp_valid = _interpolate_pointmap(
                    seed_xy=seed_xy,
                    seed_xyz=seed_xyz,
                    body_mask=body_masks_np[view_idx],
                )
                prior_pointmap[view_idx] = interp_map
                prior_valid_mask[view_idx] = interp_valid
                sparse_views_used += 1
            smplx_seed_pixels_per_view.append(int(seed_pixels))
            head_mask = _build_top_band_mask(body_masks_np[view_idx], top_ratio=float(args.head_top_ratio), min_height_px=8)
            face_mask = _build_top_band_mask(head_mask if head_mask.sum() > 0 else body_masks_np[view_idx], top_ratio=float(args.face_top_ratio), min_height_px=6)
            prior_head_mask[view_idx] = head_mask.astype(np.float32)
            prior_face_mask[view_idx] = face_mask.astype(np.float32)

        prior_source_value = f"{align_source}_sim3_interp"
        if mesh_views_used > 0:
            prior_source_value = "smplx_mesh_sim3_nearestfill"
            if sparse_views_used > 0:
                prior_source_value += "+keypoint_fallback"

        np.savez_compressed(
            out_path,
            cam_names=np.asarray(cam_names),
            prior_pointmap=prior_pointmap.astype(np.float32),
            prior_valid_mask=prior_valid_mask.astype(np.float32),
            body_mask=body_masks_np.astype(np.float32),
            head_mask=prior_head_mask.astype(np.float32),
            face_mask=prior_face_mask.astype(np.float32),
            pointmap_frame=np.asarray(["world"]),
            prior_source=np.asarray([prior_source_value]),
            align_source=np.asarray([align_source]),
            aligned_keypoints_world=aligned_keypoints_world.astype(np.float32),
            source_keypoints_world=source_keypoints_world.astype(np.float32),
            source_keypoints_valid=source_keypoints_valid.astype(np.float32),
            triangulated_keypoints_world=triangulated.astype(np.float32),
            triangulated_support=triangulated_support.astype(np.int32),
            triangulated_reproj_px=triangulated_reproj.astype(np.float32),
            processed_keypoints_xy=np.stack([x[:, :2] for x in processed_keypoints_per_view], axis=0).astype(np.float32),
            processed_keypoints_conf=np.stack([x[:, 2] for x in processed_keypoints_per_view], axis=0).astype(np.float32),
            sim3_scale=np.asarray([sim3.scale], dtype=np.float32),
            sim3_rotation=np.asarray(sim3.rotation, dtype=np.float32),
            sim3_translation=np.asarray(sim3.translation, dtype=np.float32),
            sim3_rmse=np.asarray([sim3.rmse], dtype=np.float32),
            sim3_pairs=np.asarray([sim3.pairs], dtype=np.int32),
            sim3_inliers=np.asarray([sim3.inliers], dtype=np.int32),
            sim3_source=np.asarray([sim3.source]),
            smplx_vertex_count=np.asarray([smplx_vertex_count], dtype=np.int32),
            smplx_seed_pixels=np.asarray(smplx_seed_pixels_per_view, dtype=np.int32),
            smplx_model_dir=np.asarray([str(smplx_runtime.model_dir if smplx_runtime is not None else "")]),
            smplx_gender=np.asarray([str(smplx_runtime.gender if smplx_runtime is not None else "")]),
            smplx_error=np.asarray([str(smplx_error)]),
        )

        summary_rows.append({
            "frame_id": int(frame_id),
            "status": "ok",
            "output_path": str(out_path),
            "pointmap_frame_resolved": str(pointmap_frame_override),
            "point_frame_err_world": float(frame_info.get("point_frame_err_world", np.nan)),
            "point_frame_err_camera": float(frame_info.get("point_frame_err_camera", np.nan)),
            "align_source": str(align_source),
            "triangulated_joints": int(np.isfinite(triangulated).all(axis=-1).sum()),
            "align_pairs": int(sim3.pairs),
            "align_inliers": int(sim3.inliers),
            "sim3_rmse": float(sim3.rmse),
            "sim3_source": str(sim3.source),
            "smplx_vertex_count": int(smplx_vertex_count),
            "smplx_mesh_used": int(mesh_views_used),
            "sparse_views_used": int(sparse_views_used),
            "smplx_seed_pixels_mean": float(np.mean(np.asarray(smplx_seed_pixels_per_view, dtype=np.float32))) if smplx_seed_pixels_per_view else 0.0,
            "prior_source": str(prior_source_value),
            "smplx_error": str(smplx_error),
            "prior_valid_cover": float(prior_valid_mask.mean()),
            "body_cover": float(body_masks_np.mean()),
            "head_cover": float(prior_head_mask.mean()),
            "face_cover": float(prior_face_mask.mean()),
        })
        print(
            f"[human-prior] frame={frame_id:06d} align={summary_rows[-1]['align_source']} "
            f"tri_joints={summary_rows[-1]['triangulated_joints']} "
            f"mesh_views={summary_rows[-1]['smplx_mesh_used']} sim3_rmse={summary_rows[-1]['sim3_rmse']:.5f} "
            f"valid_cover={summary_rows[-1]['prior_valid_cover']:.4f}"
        )

    summary_json = output_root / "human_prior_export_summary.json"
    summary_csv = output_root / "human_prior_export_summary.csv"
    summary_blob = {
        "seq_root": str(seq_root),
        "geom_root": str(geom_root),
        "annots_smc": str(annots_path),
        "output_root": str(output_root),
        "smplx_model_dir": str(smplx_model_dir) if smplx_model_dir is not None else "",
        "smplx_runtime_enabled": bool(smplx_runtime is not None),
        "selected_frames": [int(_frame_id_from_stem(p.stem)) for p in geom_paths],
        "pointmap_frame_resolved": str(pointmap_frame_override or ""),
        "rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary_blob, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_fields = sorted({key for row in summary_rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[human-prior] summary_json={summary_json}")
    print(f"[human-prior] summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
