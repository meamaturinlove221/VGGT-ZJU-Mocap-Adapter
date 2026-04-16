import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from precompute_zju_vggt_geom import _apply_support_to_depth_conf, _preserve_fg_depth_conf


def _map_remote_to_local(path: str, dataset_root: Path) -> Path:
    p = str(path).replace('\\', '/')
    prefix = '/mnt/data/zju_mocap/'
    if p.startswith(prefix):
        rel = p[len(prefix):]
        return dataset_root / rel.replace('/', os.sep)
    return Path(path)


def _infer_mask_path_local(img_path: Path, preferred: str = 'auto') -> tuple[Path | None, str]:
    if len(img_path.parts) < 3:
        return None, ''
    cam = img_path.parent.name
    seq_root = img_path.parent.parent
    stem = img_path.stem + '.png'
    pref = (preferred or 'auto').strip().lower()
    if pref not in {'auto', 'mask', 'mask_cihp'}:
        pref = 'auto'
    order = ['mask', 'mask_cihp'] if pref == 'auto' else [pref]
    for tok in order:
        cand = seq_root / tok / cam / stem
        if cand.is_file():
            return cand, tok
    return None, ''


def _normalize_mask_binary_np(mask_like: np.ndarray) -> np.ndarray:
    x = np.asarray(mask_like).astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    if x.ndim != 2:
        raise RuntimeError(f'unexpected mask ndim: {x.ndim}')
    if x.size <= 0:
        return np.zeros_like(x, dtype=np.float32)
    mx = float(np.nanmax(x))
    if mx <= 1.5:
        fg = x >= 0.5
    else:
        fg = x > 0.0
    return fg.astype(np.float32)


def _load_fg_masks(img_paths: list[str], dataset_root: Path, preferred: str = 'auto') -> tuple[np.ndarray, str]:
    masks = []
    src_tok = ''
    for ip in img_paths:
        local_img = _map_remote_to_local(ip, dataset_root)
        mp, tok = _infer_mask_path_local(local_img, preferred=preferred)
        if mp is None:
            raise FileNotFoundError(f'foreground mask not found for image: {local_img}')
        arr = np.array(Image.open(mp))
        masks.append(_normalize_mask_binary_np(arr))
        if not src_tok:
            src_tok = tok
    fg = np.stack(masks, axis=0)
    return fg, src_tok


def _resize_fg_masks(fg_np: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    out = []
    for m in fg_np:
        img = Image.fromarray((np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8), mode='L')
        img = img.resize((tw, th), Image.NEAREST)
        out.append((np.array(img).astype(np.float32) > 127).astype(np.float32))
    return np.stack(out, axis=0)


def _erode_mask(mask01: torch.Tensor, erode_px: int) -> torch.Tensor:
    k = int(max(0, erode_px))
    if k <= 0:
        return mask01
    if mask01.ndim != 3:
        raise RuntimeError(f'mask must be (V,H,W), got {tuple(mask01.shape)}')
    v, h, w = mask01.shape
    x = mask01.clamp(0.0, 1.0).reshape(v, 1, h, w)
    kk = 2 * k + 1
    x_inv = 1.0 - x
    dil_inv = F.max_pool2d(x_inv, kernel_size=kk, stride=1, padding=k)
    out = (1.0 - dil_inv).reshape(v, h, w)
    return out.clamp(0.0, 1.0)


def _to_vis_gray(x: np.ndarray) -> Image.Image:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr, dtype=np.uint8)
        return Image.fromarray(arr, mode='L').convert('RGB')
    vals = arr[finite]
    lo = np.percentile(vals, 2.0)
    hi = np.percentile(vals, 98.0)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return Image.fromarray(u8, mode='L').convert('RGB')


def _to_vis_unit(x: np.ndarray) -> Image.Image:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(u8, mode='L').convert('RGB')


def _mean_over_views(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().mean(dim=0).numpy()


def _build_multiview_support_diag(
    *,
    point_world: torch.Tensor,
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    tol_abs: float,
    tol_rel: float,
    stride: int,
    conf_valid_floor: float,
    generation_gate: torch.Tensor | None = None,
    min_depth: float = 1e-6,
) -> dict:
    point_world = torch.as_tensor(point_world, dtype=torch.float32)
    depth = torch.as_tensor(depth, dtype=torch.float32)
    depth_conf = torch.as_tensor(depth_conf, dtype=torch.float32)
    extrinsic_w2c = torch.as_tensor(extrinsic_w2c, dtype=torch.float32)
    intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth3 = depth[..., 0]
    elif depth.ndim == 3:
        depth3 = depth
    else:
        raise RuntimeError(f'unexpected depth shape {tuple(depth.shape)}')
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        conf3 = depth_conf[..., 0]
    elif depth_conf.ndim == 3:
        conf3 = depth_conf
    else:
        raise RuntimeError(f'unexpected depth_conf shape {tuple(depth_conf.shape)}')

    v, h, w, _ = point_world.shape
    s = max(1, int(stride))
    if s > 1:
        hh = max(1, h // s)
        ww = max(1, w // s)
        pw = F.interpolate(point_world.permute(0, 3, 1, 2), size=(hh, ww), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        dz = F.interpolate(depth3.unsqueeze(1), size=(hh, ww), mode='bilinear', align_corners=False)[:, 0]
        cf = F.interpolate(conf3.unsqueeze(1), size=(hh, ww), mode='bilinear', align_corners=False)[:, 0]
        gate = None
        if generation_gate is not None:
            gate = F.interpolate(generation_gate.unsqueeze(1), size=(hh, ww), mode='nearest')[:, 0]
    else:
        hh, ww = h, w
        pw = point_world
        dz = depth3
        cf = conf3
        gate = generation_gate

    if w > 1:
        sx = float(max(ww - 1, 0)) / float(w - 1)
    else:
        sx = 1.0
    if h > 1:
        sy = float(max(hh - 1, 0)) / float(h - 1)
    else:
        sy = 1.0

    valid = torch.isfinite(dz) & (dz > float(min_depth)) & torch.isfinite(cf) & (cf >= float(conf_valid_floor))
    if gate is not None:
        valid = valid & (gate > 0.5)

    out = torch.zeros((v, hh, ww), dtype=torch.float32)
    cover = torch.zeros((v, hh, ww), dtype=torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    ta = float(max(0.0, tol_abs))
    tr = float(max(0.0, tol_rel))

    for vi in range(v):
        xw = pw[vi].reshape(-1, 3)
        src_valid = valid[vi].reshape(-1) & torch.isfinite(xw).all(dim=-1)
        if int(src_valid.sum().item()) <= 0:
            out[vi].fill_(1.0)
            continue
        support = torch.zeros((xw.shape[0],), dtype=torch.float32)
        cover_flat = torch.zeros((xw.shape[0],), dtype=torch.float32)
        for vj in range(v):
            if vj == vi:
                continue
            e = extrinsic_w2c[vj]
            k = intrinsic[vj]
            r = e[:3, :3]
            t = e[:3, 3]
            cam = xw @ r.transpose(0, 1) + t.unsqueeze(0)
            z = cam[:, 2]
            proj_ok = src_valid & torch.isfinite(z) & (z > float(min_depth))
            if int(proj_ok.sum().item()) <= 0:
                continue
            fx = k[0, 0]
            fy = k[1, 1]
            cx = k[0, 2]
            cy = k[1, 2]
            u = fx * (cam[:, 0] / (z + 1e-8)) + cx
            vv = fy * (cam[:, 1] / (z + 1e-8)) + cy
            ui = torch.round(u * sx).long()
            vvi = torch.round(vv * sy).long()
            inside = proj_ok & (ui >= 0) & (ui < ww) & (vvi >= 0) & (vvi < hh)
            if int(inside.sum().item()) <= 0:
                continue
            idx = torch.where(inside)[0]
            dt = dz[vj][vvi[idx], ui[idx]]
            cft = cf[vj][vvi[idx], ui[idx]]
            valid_tgt = torch.isfinite(dt) & (dt > float(min_depth)) & torch.isfinite(cft) & (cft >= float(conf_valid_floor))
            if gate is not None:
                valid_tgt = valid_tgt & (gate[vj][vvi[idx], ui[idx]] > 0.5)
            if int(valid_tgt.sum().item()) <= 0:
                continue
            idx2 = idx[valid_tgt]
            dt2 = dt[valid_tgt]
            zz2 = z[idx2]
            tol = ta + tr * dt2.abs()
            agree = (zz2 - dt2).abs() <= tol
            cover_flat.index_add_(0, idx2, one.expand(idx2.shape[0]))
            if int(agree.sum().item()) > 0:
                idx3 = idx2[agree]
                support.index_add_(0, idx3, one.expand(idx3.shape[0]))
        ratio = support / cover_flat.clamp_min(1.0)
        ratio = ratio.reshape(hh, ww)
        cov = cover_flat.reshape(hh, ww)
        ratio = torch.where(cov > 0.0, ratio, torch.ones_like(ratio))
        out[vi] = ratio
        cover[vi] = cov

    if s > 1:
        out = F.interpolate(out.unsqueeze(1), size=(h, w), mode='nearest')[:, 0]
        cover = F.interpolate(cover.unsqueeze(1), size=(h, w), mode='nearest')[:, 0]
        valid = F.interpolate(valid.float().unsqueeze(1), size=(h, w), mode='nearest')[:, 0] > 0.5
        if gate is not None:
            gate = F.interpolate(gate.unsqueeze(1), size=(h, w), mode='nearest')[:, 0]
    return {
        'support': out.clamp(0.0, 1.0),
        'cover': cover,
        'valid': valid,
        'gate': gate,
    }


def _compute_stats(support: torch.Tensor, valid: torch.Tensor, fg: torch.Tensor, depth_conf_raw: torch.Tensor, depth_conf_after: torch.Tensor, cover: torch.Tensor) -> dict:
    support_np = support.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(np.float32)
    fg_np = fg.detach().cpu().numpy().astype(np.float32)
    bg_np = 1.0 - fg_np
    conf_raw_np = depth_conf_raw.detach().cpu().numpy().astype(np.float32)
    conf_after_np = depth_conf_after.detach().cpu().numpy().astype(np.float32)
    cover_np = cover.detach().cpu().numpy().astype(np.float32)

    def masked_mean(x, m):
        m = m > 0.5
        if not np.any(m):
            return float('nan')
        return float(np.mean(x[m]))

    stats = {
        'support_raw_mean': float(np.mean(support_np)),
        'support_fg_mean': masked_mean(support_np, fg_np),
        'support_bg_mean': masked_mean(support_np, bg_np),
        'support_valid_ratio': float(np.mean(valid_np)),
        'support_fg_valid_ratio': masked_mean(valid_np, fg_np),
        'support_bg_valid_ratio': masked_mean(valid_np, bg_np),
        'support_pair_count_eff': masked_mean(cover_np, valid_np),
        'support_conf_mean': masked_mean(conf_raw_np, valid_np),
        'support_nan_ratio': float(np.mean(~np.isfinite(support_np))),
        'depth_conf_raw_mean': float(np.mean(conf_raw_np)),
        'depth_conf_after_support_mean': float(np.mean(conf_after_np)),
        'depth_conf_delta_mean': float(np.mean(conf_after_np - conf_raw_np)),
        'depth_conf_delta_fg_mean': masked_mean(conf_after_np - conf_raw_np, fg_np),
        'depth_conf_delta_bg_mean': masked_mean(conf_after_np - conf_raw_np, bg_np),
    }
    return stats


def _make_grid(mode_name: str, support: torch.Tensor, fg: torch.Tensor, depth_conf_raw: torch.Tensor, depth_conf_after: torch.Tensor, out_path: Path):
    support_mean = _mean_over_views(support)
    fg_mean = _mean_over_views(fg)
    support_fg = support_mean * fg_mean
    support_bg = support_mean * (1.0 - fg_mean)
    conf_raw = _mean_over_views(depth_conf_raw)
    conf_after = _mean_over_views(depth_conf_after)
    delta = conf_after - conf_raw

    tiles = [
        ('support_map', _to_vis_unit(support_mean)),
        ('fg_mask', _to_vis_unit(fg_mean)),
        ('support_x_fg', _to_vis_unit(support_fg)),
        ('support_x_bg', _to_vis_unit(support_bg)),
        ('depth_conf_raw', _to_vis_gray(conf_raw)),
        ('depth_conf_after', _to_vis_gray(conf_after)),
        ('depth_conf_delta', _to_vis_gray(delta)),
    ]

    pad = 16
    header_h = 28
    tile_w, tile_h = tiles[0][1].size
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    canvas_w = pad + cols * (tile_w + pad)
    canvas_h = pad + rows * (header_h + tile_h + pad) + 36
    canvas = Image.new('RGB', (canvas_w, canvas_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), f'mode={mode_name}', fill=(20,20,20), font=font)
    for idx, (label, img) in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_w + pad)
        y = pad + 24 + r * (header_h + tile_h + pad)
        draw.rectangle([x, y, x + tile_w, y + header_h], fill=(230,230,230))
        draw.text((x + 8, y + 8), label, fill=(20,20,20), font=font)
        canvas.paste(img, (x, y + header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--dataset-root', default=r'F:\datasets\ZJU_MoCap\data\zju_mocap')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mask-source', default='auto', choices=['auto', 'mask', 'mask_cihp'])
    ap.add_argument('--fg-erode-px', type=int, default=5)
    ap.add_argument('--fg-preserve-px', type=int, default=3)
    ap.add_argument('--tol-abs', type=float, default=0.06)
    ap.add_argument('--tol-rel', type=float, default=0.10)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--conf-valid-floor', type=float, default=0.05)
    ap.add_argument('--mv-support-mode', default='linear')
    ap.add_argument('--mv-support-floor', type=float, default=0.05)
    ap.add_argument('--mv-support-gamma', type=float, default=1.0)
    ap.add_argument('--mv-support-clip-thr', type=float, default=0.20)
    ap.add_argument('--mv-support-clip-floor', type=float, default=0.30)
    ap.add_argument('--mv-support-hard-thr', type=float, default=-1.0)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz_path, allow_pickle=True)
    img_paths = [str(x) for x in z['img_paths'].tolist()]
    fg_np, fg_src = _load_fg_masks(img_paths, Path(args.dataset_root), preferred=args.mask_source)

    pointmap = torch.from_numpy(z['pointmap']).float()
    depth = torch.from_numpy(z['depth']).float()
    depth_conf = torch.from_numpy(z['depth_conf']).float()
    extrinsic = torch.from_numpy(z['extrinsic']).float()
    intrinsic = torch.from_numpy(z['intrinsic']).float()
    target_hw = (int(depth.shape[1]), int(depth.shape[2]))
    if fg_np.shape[1:] != target_hw:
        fg_np = _resize_fg_masks(fg_np, target_hw)
    fg = torch.from_numpy(fg_np).float()
    fg_eroded = _erode_mask(fg, args.fg_erode_px)

    mode_gates = {
        'all': None,
        'bg_only': (1.0 - fg).clamp(0.0, 1.0),
        'fg_eroded_off': (1.0 - fg_eroded).clamp(0.0, 1.0),
    }

    all_stats = {
        'npz': str(npz_path),
        'dataset_root': str(args.dataset_root),
        'mask_source_resolved': fg_src,
        'views': int(pointmap.shape[0]),
        'frame_shape': [int(depth.shape[1]), int(depth.shape[2])],
        'modes': {},
    }

    for mode_name, gate in mode_gates.items():
        t0 = time.perf_counter()
        diag = _build_multiview_support_diag(
            point_world=pointmap,
            depth=depth,
            depth_conf=depth_conf,
            extrinsic_w2c=extrinsic,
            intrinsic=intrinsic,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
            stride=args.stride,
            conf_valid_floor=args.conf_valid_floor,
            generation_gate=gate,
        )
        depth_conf_after, mv_weight = _apply_support_to_depth_conf(
            depth_conf=depth_conf,
            support01=diag['support'],
            mode=args.mv_support_mode,
            floor=args.mv_support_floor,
            gamma=args.mv_support_gamma,
            clip_thr=args.mv_support_clip_thr,
            clip_floor=args.mv_support_clip_floor,
            hard_thr=args.mv_support_hard_thr,
        )
        depth_conf_after, fg_preserve_stats = _preserve_fg_depth_conf(
            depth_conf_raw=depth_conf,
            depth_conf_after_support=depth_conf_after,
            fg_mask=fg,
            region_mode_resolved=mode_name,
            preserve_px=args.fg_preserve_px,
        )
        stats = _compute_stats(
            support=diag['support'],
            valid=diag['valid'],
            fg=fg,
            depth_conf_raw=depth_conf,
            depth_conf_after=depth_conf_after if depth_conf_after.ndim == 3 else depth_conf_after[..., 0],
            cover=diag['cover'],
        )
        stats.update(fg_preserve_stats)
        stats['elapsed_sec'] = round(float(time.perf_counter() - t0), 4)
        all_stats['modes'][mode_name] = stats
        _make_grid(
            mode_name,
            diag['support'],
            fg,
            depth_conf,
            depth_conf_after if depth_conf_after.ndim == 3 else depth_conf_after[..., 0],
            out_dir / f'support_diag_{mode_name}.png',
        )

    summary_lines = ['# Support Generation Diagnostic', '', f'- npz: `{npz_path}`', f'- mask_source: `{fg_src}`', '']
    summary_lines.append('| mode | support_fg_mean | support_bg_mean | fg_valid_ratio | bg_valid_ratio | pair_count_eff | conf_delta_fg | conf_delta_bg | fg_preserve_ratio | fg_final_conf |')
    summary_lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for mode_name, stats in all_stats['modes'].items():
        summary_lines.append(
            f"| {mode_name} | {stats['support_fg_mean']:.4f} | {stats['support_bg_mean']:.4f} | {stats['support_fg_valid_ratio']:.4f} | {stats['support_bg_valid_ratio']:.4f} | {stats['support_pair_count_eff']:.4f} | {stats['depth_conf_delta_fg_mean']:.4f} | {stats['depth_conf_delta_bg_mean']:.4f} | {stats.get('depth_conf_fg_preserve_ratio', float('nan')):.4f} | {stats.get('depth_conf_fg_final_mean', float('nan')):.4f} |"
        )
    (out_dir / 'support_diag_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    (out_dir / 'support_diag_summary.json').write_text(json.dumps(all_stats, indent=2), encoding='utf-8')
    print(out_dir)


if __name__ == '__main__':
    main()
