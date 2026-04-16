import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
import sys
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from diagnose_support_generation import (
    _build_multiview_support_diag,
    _compute_stats,
    _erode_mask,
    _load_fg_masks,
    _resize_fg_masks,
)
from precompute_zju_vggt_geom import _apply_support_to_depth_conf, _preserve_fg_depth_conf


def _infer_group_name(npz_path: Path) -> str:
    stem = npz_path.stem
    if '_frame_' in stem:
        return stem.split('_frame_', 1)[0]
    return stem


def _to_float(x):
    try:
        v = float(x)
    except Exception:
        return float('nan')
    return v


def _safe_mean(vals):
    good = [float(v) for v in vals if math.isfinite(float(v))]
    if not good:
        return float('nan')
    return float(mean(good))


def _safe_std(vals):
    good = [float(v) for v in vals if math.isfinite(float(v))]
    if len(good) <= 1:
        return 0.0 if good else float('nan')
    return float(pstdev(good))


def _load_npz_stats(npz_path: Path, dataset_root: Path, mask_source: str, fg_erode_px: int, fg_preserve_px: int, tol_abs: float, tol_rel: float, stride: int, conf_valid_floor: float, mv_support_mode: str, mv_support_floor: float, mv_support_gamma: float, mv_support_clip_thr: float, mv_support_clip_floor: float, mv_support_hard_thr: float):
    z = np.load(npz_path, allow_pickle=True)
    img_paths = [str(x) for x in z['img_paths'].tolist()]
    fg_np, fg_src = _load_fg_masks(img_paths, dataset_root, preferred=mask_source)

    pointmap = torch.from_numpy(z['pointmap']).float()
    depth = torch.from_numpy(z['depth']).float()
    depth_conf = torch.from_numpy(z['depth_conf']).float()
    extrinsic = torch.from_numpy(z['extrinsic']).float()
    intrinsic = torch.from_numpy(z['intrinsic']).float()
    target_hw = (int(depth.shape[1]), int(depth.shape[2]))
    if fg_np.shape[1:] != target_hw:
        fg_np = _resize_fg_masks(fg_np, target_hw)
    fg = torch.from_numpy(fg_np).float()
    fg_eroded = _erode_mask(fg, fg_erode_px)

    mode_gates = {
        'all': None,
        'bg_only': (1.0 - fg).clamp(0.0, 1.0),
        'fg_eroded_off': (1.0 - fg_eroded).clamp(0.0, 1.0),
    }
    result = {
        'npz': str(npz_path),
        'group': _infer_group_name(npz_path),
        'mask_source_resolved': fg_src,
        'views': int(pointmap.shape[0]),
        'frame_shape': [int(depth.shape[1]), int(depth.shape[2])],
        'modes': {},
    }
    for mode_name, gate in mode_gates.items():
        diag = _build_multiview_support_diag(
            point_world=pointmap,
            depth=depth,
            depth_conf=depth_conf,
            extrinsic_w2c=extrinsic,
            intrinsic=intrinsic,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            stride=stride,
            conf_valid_floor=conf_valid_floor,
            generation_gate=gate,
        )
        depth_conf_after, _mv_weight = _apply_support_to_depth_conf(
            depth_conf=depth_conf,
            support01=diag['support'],
            mode=mv_support_mode,
            floor=mv_support_floor,
            gamma=mv_support_gamma,
            clip_thr=mv_support_clip_thr,
            clip_floor=mv_support_clip_floor,
            hard_thr=mv_support_hard_thr,
        )
        depth_conf_after, fg_preserve_stats = _preserve_fg_depth_conf(
            depth_conf_raw=depth_conf,
            depth_conf_after_support=depth_conf_after,
            fg_mask=fg,
            region_mode_resolved=mode_name,
            preserve_px=fg_preserve_px,
        )
        result['modes'][mode_name] = _compute_stats(
            support=diag['support'],
            valid=diag['valid'],
            fg=fg,
            depth_conf_raw=depth_conf,
            depth_conf_after=depth_conf_after if depth_conf_after.ndim == 3 else depth_conf_after[..., 0],
            cover=diag['cover'],
        )
        result['modes'][mode_name].update(fg_preserve_stats)
    return result


def _aggregate(entries):
    metric_names = [
        'support_raw_mean',
        'support_fg_mean',
        'support_bg_mean',
        'support_valid_ratio',
        'support_fg_valid_ratio',
        'support_bg_valid_ratio',
        'support_pair_count_eff',
        'support_conf_mean',
        'support_nan_ratio',
        'depth_conf_delta_mean',
        'depth_conf_delta_fg_mean',
        'depth_conf_delta_bg_mean',
        'depth_conf_fg_preserved_active',
        'depth_conf_fg_preserve_ratio',
        'depth_conf_fg_raw_mean',
        'depth_conf_fg_after_support_mean',
        'depth_conf_fg_final_mean',
    ]
    out = {}
    for mode in ('all', 'bg_only', 'fg_eroded_off'):
        out[mode] = {}
        for metric in metric_names:
            vals = [_to_float(e['modes'][mode].get(metric, float('nan'))) for e in entries]
            out[mode][metric + '_mean'] = _safe_mean(vals)
            out[mode][metric + '_std'] = _safe_std(vals)
    return out


def _render_summary_png(lines, out_path: Path):
    font = ImageFont.load_default()
    pad = 16
    line_h = 18
    width = 1600
    height = pad * 2 + line_h * (len(lines) + 1)
    canvas = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(20, 20, 20), font=font)
        y += line_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-glob', action='append', required=True)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    npz_paths = []
    for pat in args.npz_glob:
        npz_paths.extend(sorted(Path().glob(pat)))
    npz_paths = sorted({p.resolve() for p in npz_paths if p.is_file()})
    if not npz_paths:
        raise SystemExit('no npz files matched')

    entries = []
    for npz_path in npz_paths:
        entries.append(_load_npz_stats(
            npz_path=npz_path,
            dataset_root=dataset_root,
            mask_source=args.mask_source,
            fg_erode_px=args.fg_erode_px,
            fg_preserve_px=args.fg_preserve_px,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
            stride=args.stride,
            conf_valid_floor=args.conf_valid_floor,
            mv_support_mode=args.mv_support_mode,
            mv_support_floor=args.mv_support_floor,
            mv_support_gamma=args.mv_support_gamma,
            mv_support_clip_thr=args.mv_support_clip_thr,
            mv_support_clip_floor=args.mv_support_clip_floor,
            mv_support_hard_thr=args.mv_support_hard_thr,
        ))

    groups = {}
    for e in entries:
        groups.setdefault(e['group'], []).append(e)

    payload = {
        'generated_at': __import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds'),
        'dataset_root': str(dataset_root),
        'frame_count': len(entries),
        'groups': {},
    }
    lines = ['# Support Generation Multi-frame Diagnosis', '']
    lines.append(f'- frames: {len(entries)}')
    lines.append('')

    for group_name, group_entries in sorted(groups.items()):
        agg = _aggregate(group_entries)
        payload['groups'][group_name] = {
            'frame_count': len(group_entries),
            'frames': [e['npz'] for e in group_entries],
            'aggregate': agg,
        }
        lines.append(f'## {group_name} ({len(group_entries)} frames)')
        lines.append('')
        lines.append('| mode | support_fg_mean | support_bg_mean | fg_valid_ratio | bg_valid_ratio | pair_eff | conf_delta_fg | conf_delta_bg | fg_preserve_ratio | fg_final_conf |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for mode in ('all', 'bg_only', 'fg_eroded_off'):
            s = agg[mode]
            lines.append(
                f"| {mode} | {s['support_fg_mean_mean']:.4f} | {s['support_bg_mean_mean']:.4f} | {s['support_fg_valid_ratio_mean']:.4f} | {s['support_bg_valid_ratio_mean']:.4f} | {s['support_pair_count_eff_mean']:.4f} | {s['depth_conf_delta_fg_mean_mean']:.4f} | {s['depth_conf_delta_bg_mean_mean']:.4f} | {s['depth_conf_fg_preserve_ratio_mean']:.4f} | {s['depth_conf_fg_final_mean_mean']:.4f} |"
            )
        all_fg = agg['all']['depth_conf_delta_fg_mean_mean']
        bg_fg = agg['bg_only']['depth_conf_delta_fg_mean_mean']
        fg_gap = agg['all']['support_bg_mean_mean'] - agg['all']['support_fg_mean_mean']
        lines.append('')
        lines.append(f'- all fg/bg support gap: {fg_gap:.4f}')
        lines.append(f'- bg_only vs all fg depth_conf delta improvement: {(bg_fg - all_fg):.4f}')
        lines.append('')

    md_path = out_dir / 'support_generation_multiframe_summary.md'
    json_path = out_dir / 'support_generation_multiframe_summary.json'
    png_path = out_dir / 'support_generation_multiframe_summary.png'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    _render_summary_png(lines, png_path)
    print(out_dir)


if __name__ == '__main__':
    main()
