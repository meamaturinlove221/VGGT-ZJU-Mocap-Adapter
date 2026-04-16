import argparse
import os
import os.path as osp
import re
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from select_views_uniform_yaw import (
    camera_centers_from_extrinsic,
    estimate_subject_center,
    pick_uniform_yaw_indices,
    yaw_degrees_from_centers,
)


def _to_3x4_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic)
    if e.ndim == 2:
        e = e[None, ...]
    if e.shape[-2:] == (4, 4):
        e = e[..., :3, :4]
    if e.shape[-2:] != (3, 4):
        raise ValueError(f"invalid extrinsic shape: {e.shape}")
    return e.astype(np.float64, copy=False)


def _resolve_img_path(path_str: str, zju_root: str, seq_names: list[str]) -> str:
    if isinstance(path_str, bytes):
        path_str = path_str.decode("utf-8")
    s = str(path_str).strip().replace("\\", "/")
    if osp.exists(s):
        return s
    if osp.isabs(s) and osp.exists(s):
        return s
    if re.match(r"^[A-Za-z]:/", s):
        key = "/zju_mocap/"
        if key in s:
            s = s.split(key, 1)[1]
        else:
            parts = s.split("/")
            cut = None
            for i, p in enumerate(parts):
                if p.startswith("CoreView_"):
                    cut = i
                    break
            if cut is not None:
                s = "/".join(parts[cut:])
            else:
                for seq in seq_names:
                    if seq in s:
                        s = seq + s.split(seq, 1)[1]
                        break
    return osp.join(zju_root, s.lstrip("/"))


def _infer_mask_path(img_path: str) -> Optional[str]:
    if not img_path:
        return None
    s = str(img_path).replace("\\", "/")
    parts = s.split("/")
    idx_images = None
    idx_cam = None
    if "images" in parts:
        idx_images = parts.index("images")
    else:
        for i, p in enumerate(parts):
            if p.startswith("Camera_"):
                idx_cam = i
                break
    if idx_images is None and idx_cam is None:
        return None

    def _build(mask_dir: str):
        if idx_images is not None:
            p2 = list(parts)
            p2[idx_images] = mask_dir
        else:
            p2 = list(parts[:idx_cam]) + [mask_dir] + list(parts[idx_cam:])
        out = "/".join(p2)
        base, _ = osp.splitext(out)
        return base + ".png"

    p_mask = _build("mask")
    if osp.isfile(p_mask):
        return p_mask
    p_cihp = _build("mask_cihp")
    if osp.isfile(p_cihp):
        return p_cihp
    return None


def _load_rgb_u8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask01(path: Optional[str], size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if (path is None) or (not osp.isfile(path)):
        return np.zeros((h, w), dtype=np.float32)
    m = Image.open(path).convert("L")
    if m.size != (w, h):
        m = m.resize((w, h), Image.NEAREST)
    arr = np.array(m, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    if float(arr.max()) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _overlay_mask(rgb_u8: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    m = (mask01 > 0.5)
    out = rgb_u8.astype(np.float32).copy()
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    out[m] = out[m] * (1.0 - float(alpha)) + green * float(alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _jet_colormap(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _heatmap_from_scalar(x: np.ndarray, qlo: float = 0.02, qhi: float = 0.98) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    lo = float(np.quantile(arr[valid], qlo))
    hi = float(np.quantile(arr[valid], qhi))
    if hi <= lo:
        hi = lo + 1e-6
    x01 = (arr - lo) / (hi - lo)
    return _jet_colormap(x01)


def _to_pointmap_hw3(pm: np.ndarray) -> np.ndarray:
    p = np.asarray(pm)
    if p.ndim == 3 and p.shape[-1] == 3:
        return p
    if p.ndim == 3 and p.shape[0] == 3:
        return np.transpose(p, (1, 2, 0))
    raise ValueError(f"invalid pointmap shape: {p.shape}")


def _to_3x3_intrinsic(k: np.ndarray) -> np.ndarray:
    kk = np.asarray(k)
    if kk.shape[-2:] == (4, 4):
        kk = kk[:3, :3]
    if kk.shape[-2:] != (3, 3):
        raise ValueError(f"invalid intrinsic shape: {kk.shape}")
    return kk.astype(np.float64, copy=False)


def _project_world_to_pixel(pointmap_hw3: np.ndarray, k_3x3: np.ndarray, t_3x4: np.ndarray):
    pts = pointmap_hw3.reshape(-1, 3).T
    r = t_3x4[:3, :3]
    t = t_3x4[:3, 3:4]
    cam = (r @ pts) + t
    z = cam[2, :]
    x = cam[0, :] / (z + 1e-8)
    y = cam[1, :] / (z + 1e-8)
    u = k_3x3[0, 0] * x + k_3x3[0, 2]
    v = k_3x3[1, 1] * y + k_3x3[1, 2]
    return u, v, z


def _overlay_reprojection(
    rgb_u8: np.ndarray,
    tgt_pointmap_hw3: np.ndarray,
    src_k_3x3: np.ndarray,
    src_t_3x4: np.ndarray,
    stride: int = 24,
    select_mask01: Optional[np.ndarray] = None,
    select_thr: float = 0.5,
) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    out = Image.fromarray(rgb_u8.copy())
    draw = ImageDraw.Draw(out)
    u, v, z = _project_world_to_pixel(tgt_pointmap_hw3, src_k_3x3, src_t_3x4)
    s = max(int(stride), 1)
    h0, w0, _ = tgt_pointmap_hw3.shape
    grid_v, grid_u = np.meshgrid(np.arange(h0), np.arange(w0), indexing="ij")
    take = ((grid_u % s) == 0) & ((grid_v % s) == 0)
    if select_mask01 is not None:
        sm = np.asarray(select_mask01, dtype=np.float32)
        if sm.ndim == 3:
            sm = sm[..., 0]
        if sm.shape != (h0, w0):
            sm_img = Image.fromarray((np.clip(sm, 0.0, 1.0) * 255.0).astype(np.uint8))
            sm = np.array(sm_img.resize((w0, h0), Image.NEAREST), dtype=np.float32) / 255.0
        take = take & (sm > float(select_thr))
    uu = u.reshape(h0, w0)[take]
    vv = v.reshape(h0, w0)[take]
    zz = z.reshape(h0, w0)[take]
    valid = np.isfinite(uu) & np.isfinite(vv) & np.isfinite(zz) & (zz > 1e-6)
    uu = uu[valid]
    vv = vv[valid]
    inside = (uu >= 0) & (uu <= (w - 1)) & (vv >= 0) & (vv <= (h - 1))
    uu = uu[inside]
    vv = vv[inside]
    for x, y in zip(uu.tolist(), vv.tolist()):
        draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=(255, 64, 64))
    return np.array(out, dtype=np.uint8)


def _fit_tile_u8(img_u8: np.ndarray, tile_hw: tuple[int, int]) -> np.ndarray:
    th, tw = tile_hw
    im = Image.fromarray(img_u8)
    im.thumbnail((tw, th), Image.BICUBIC)
    canvas = Image.new("RGB", (tw, th), color=(20, 20, 20))
    x = (tw - im.size[0]) // 2
    y = (th - im.size[1]) // 2
    canvas.paste(im, (x, y))
    return np.array(canvas, dtype=np.uint8)


def _label_tile(img_u8: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img_u8.copy())
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.size[0], 22), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.array(im, dtype=np.uint8)


def _concat_h(tiles: list[np.ndarray], pad: int = 6) -> np.ndarray:
    if not tiles:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h = max(t.shape[0] for t in tiles)
    w = sum(t.shape[1] for t in tiles) + pad * (len(tiles) - 1)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x = 0
    for i, t in enumerate(tiles):
        th, tw = t.shape[:2]
        y = (h - th) // 2
        out[y:y + th, x:x + tw] = t
        x += tw
        if i + 1 < len(tiles):
            x += pad
    return out


def _concat_v(rows: list[np.ndarray], pad: int = 8) -> np.ndarray:
    if not rows:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h = sum(r.shape[0] for r in rows) + pad * (len(rows) - 1)
    w = max(r.shape[1] for r in rows)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y = 0
    for i, r in enumerate(rows):
        rh, rw = r.shape[:2]
        x = (w - rw) // 2
        out[y:y + rh, x:x + rw] = r
        y += rh
        if i + 1 < len(rows):
            y += pad
    return out


def _make_target_block(
    data: dict,
    tgt_idx: int,
    src_idxs: list[int],
    zju_root: str,
    seq_names: list[str],
    tile_hw: tuple[int, int],
    point_stride: int,
    yaw_deg: np.ndarray,
) -> np.ndarray:
    img_paths = data["img_paths"]
    depth = data["depth"]
    depth_conf = data["depth_conf"]
    pointmap = data["pointmap"]
    extrinsic = _to_3x4_extrinsic(data["extrinsic"])
    intrinsic = np.asarray(data["intrinsic"])
    cam_names = data.get("cam_names", None)
    if cam_names is not None:
        cam_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in cam_names]

    tgt_path = _resolve_img_path(img_paths[int(tgt_idx)], zju_root, seq_names)
    tgt_rgb = _load_rgb_u8(tgt_path)
    tgt_mask = _load_mask01(_infer_mask_path(tgt_path), size_hw=tgt_rgb.shape[:2])
    tgt_overlay = _overlay_mask(tgt_rgb, tgt_mask)
    tgt_reproj_mask = tgt_mask if float(tgt_mask.mean()) > 1e-4 else None
    tgt_pm = _to_pointmap_hw3(pointmap[int(tgt_idx)])
    tgt_k = _to_3x3_intrinsic(intrinsic[int(tgt_idx)])
    tgt_t = _to_3x4_extrinsic(extrinsic[int(tgt_idx)])[0]

    src_row_overlay = []
    src_row_reproj = []
    for sidx in src_idxs:
        sidx = int(sidx)
        src_path = _resolve_img_path(img_paths[sidx], zju_root, seq_names)
        src_rgb = _load_rgb_u8(src_path)
        src_mask = _load_mask01(_infer_mask_path(src_path), size_hw=src_rgb.shape[:2])
        src_overlay = _overlay_mask(src_rgb, src_mask)
        src_overlay = _fit_tile_u8(src_overlay, tile_hw)
        src_overlay = _label_tile(
            src_overlay,
            f"src[{sidx}] yaw={yaw_deg[sidx]:.1f}" + (f" {cam_names[sidx]}" if cam_names is not None else ""),
        )
        src_row_overlay.append(src_overlay)

        src_k = _to_3x3_intrinsic(intrinsic[sidx])
        src_t = _to_3x4_extrinsic(extrinsic[sidx])[0]
        src_reproj = _overlay_reprojection(
            rgb_u8=src_rgb,
            tgt_pointmap_hw3=tgt_pm,
            src_k_3x3=src_k,
            src_t_3x4=src_t,
            stride=point_stride,
            select_mask01=tgt_reproj_mask,
        )
        src_reproj = _fit_tile_u8(src_reproj, tile_hw)
        src_reproj = _label_tile(src_reproj, "tgt pointmap reproj")
        src_row_reproj.append(src_reproj)

    tgt_tile = _label_tile(_fit_tile_u8(tgt_overlay, tile_hw), f"tgt[{tgt_idx}] RGB+mask")
    depth_tile = _label_tile(_fit_tile_u8(_heatmap_from_scalar(depth[int(tgt_idx)]), tile_hw), "tgt depth")
    conf_tile = _label_tile(_fit_tile_u8(_heatmap_from_scalar(depth_conf[int(tgt_idx)]), tile_hw), "tgt depth_conf")
    z_t = _project_world_to_pixel(tgt_pm, tgt_k, tgt_t)[2].reshape(tgt_pm.shape[:2])
    z_tile = _label_tile(_fit_tile_u8(_heatmap_from_scalar(z_t), tile_hw), "tgt pointmap z_cam")

    row0 = _concat_h(src_row_overlay, pad=6)
    row1 = _concat_h(src_row_reproj, pad=6)
    row2 = _concat_h([tgt_tile, depth_tile, conf_tile, z_tile], pad=6)

    header = np.zeros((30, max(row0.shape[1], row1.shape[1], row2.shape[1]), 3), dtype=np.uint8)
    header_im = Image.fromarray(header)
    draw = ImageDraw.Draw(header_im)
    draw.text((8, 7), f"target={tgt_idx} yaw={yaw_deg[tgt_idx]:.1f}", fill=(255, 255, 255))
    header = np.array(header_im, dtype=np.uint8)
    return _concat_v([header, row0, row1, row2], pad=6)


def _collect_npz_paths(zju_root: str, seq_names: list[str], geom_subdir: str) -> list[str]:
    paths = []
    for seq in seq_names:
        d = osp.join(zju_root, seq, str(geom_subdir))
        if not osp.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npz"):
                paths.append(osp.join(d, fn))
    return paths


def render_mosaic_from_npz(
    npz_path: str,
    zju_root: str,
    seq_names: list[str],
    out_path: Optional[str] = None,
    num_targets: int = 3,
    num_src_views: int = 6,
    yaw_jitter_deg: float = 20.0,
    yaw_phase_jitter_deg: float = 20.0,
    yaw_axis_x: int = 0,
    yaw_axis_z: int = 2,
    tile_size: int = 300,
    point_stride: int = 24,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    v = int(data["img_paths"].shape[0])
    if v < 2:
        raise RuntimeError(f"need at least 2 views in npz: {npz_path}")

    if isinstance(seq_names, str):
        seq_names = [s for s in re.split(r"[,\s]+", str(seq_names)) if s]
    if not seq_names:
        seq_names = []

    if rng is None:
        rng = np.random.RandomState(2026)

    extrinsic = _to_3x4_extrinsic(data["extrinsic"])
    centers = camera_centers_from_extrinsic(extrinsic)
    subject_center = estimate_subject_center(data.get("pointmap", None), camera_centers=centers)
    yaw_deg = yaw_degrees_from_centers(
        camera_centers=centers,
        subject_center=subject_center,
        axis_x=int(yaw_axis_x),
        axis_z=int(yaw_axis_z),
    )

    tgt_count = min(int(num_targets), v)
    tgt_idxs = list(np.asarray(np.arange(v))[rng.permutation(v)[:tgt_count]].tolist())
    blocks = []
    for tgt_idx in tgt_idxs:
        src_idxs = pick_uniform_yaw_indices(
            yaw_deg=yaw_deg,
            num_select=min(int(num_src_views), v - 1),
            rng=rng,
            exclude_indices=[int(tgt_idx)],
            jitter_deg=float(yaw_jitter_deg),
            phase_jitter_deg=float(yaw_phase_jitter_deg),
        )
        blocks.append(
            _make_target_block(
                data=data,
                tgt_idx=int(tgt_idx),
                src_idxs=src_idxs,
                zju_root=str(zju_root),
                seq_names=list(seq_names),
                tile_hw=(int(tile_size), int(tile_size)),
                point_stride=int(point_stride),
                yaw_deg=yaw_deg,
            )
        )

    if not blocks:
        raise RuntimeError(f"no target blocks built for npz: {npz_path}")
    mosaic = _concat_v(blocks, pad=14)

    if out_path:
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        Image.fromarray(mosaic).save(out_path)
    return mosaic


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zju_root", type=str, required=True)
    ap.add_argument("--seq_names", type=str, default="CoreView_390")
    ap.add_argument("--geom_subdir", type=str, default="vggt_geom")
    ap.add_argument("--out", type=str, default="inspect_batch_mosaic_out")
    ap.add_argument("--num_samples", type=int, default=1, help="How many random frame npz to inspect.")
    ap.add_argument("--num_targets", type=int, default=3, help="How many target views per frame.")
    ap.add_argument("--num_src_views", type=int, default=6)
    ap.add_argument("--yaw_jitter_deg", type=float, default=20.0)
    ap.add_argument("--yaw_phase_jitter_deg", type=float, default=20.0)
    ap.add_argument("--yaw_axis_x", type=int, default=0)
    ap.add_argument("--yaw_axis_z", type=int, default=2)
    ap.add_argument("--tile_size", type=int, default=300)
    ap.add_argument("--point_stride", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    seq_names = [s for s in re.split(r"[,\s]+", str(args.seq_names)) if s]
    if not seq_names:
        raise RuntimeError("seq_names is empty")
    zju_root = str(args.zju_root)
    out_dir = str(args.out)
    os.makedirs(out_dir, exist_ok=True)

    paths = _collect_npz_paths(zju_root, seq_names, geom_subdir=str(args.geom_subdir))
    if not paths:
        raise RuntimeError(
            f"no npz found under {zju_root} for seq_names={seq_names} geom_subdir={args.geom_subdir}"
        )

    rng = np.random.RandomState(int(args.seed))
    order = rng.permutation(len(paths))
    take_n = min(int(args.num_samples), len(paths))

    for rank in range(take_n):
        npz_path = paths[int(order[rank])]
        bn = osp.splitext(osp.basename(npz_path))[0]
        out_path = osp.join(out_dir, f"{bn}_mosaic_rank{rank:02d}.png")
        render_mosaic_from_npz(
            npz_path=npz_path,
            zju_root=zju_root,
            seq_names=seq_names,
            out_path=out_path,
            num_targets=int(args.num_targets),
            num_src_views=int(args.num_src_views),
            yaw_jitter_deg=float(args.yaw_jitter_deg),
            yaw_phase_jitter_deg=float(args.yaw_phase_jitter_deg),
            yaw_axis_x=int(args.yaw_axis_x),
            yaw_axis_z=int(args.yaw_axis_z),
            tile_size=int(args.tile_size),
            point_stride=int(args.point_stride),
            rng=rng,
        )
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    _main()
