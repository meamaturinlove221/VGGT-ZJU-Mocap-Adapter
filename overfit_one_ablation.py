import os
import time
import argparse
import numpy as np
from PIL import Image
import torch

from zju_dataset_view import ZJUViewSynthDataset
from view_decoder_ablation import GeomViewDecoderAblation
from train_view_decoder_ablation import (
    autocast_ctx, build_masks_from_batch, masked_l1, save_debug_pack
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zju_root', type=str, required=True)
    ap.add_argument('--seq_names', type=str, nargs='+', required=True)
    ap.add_argument('--index', type=int, default=0,
                    help='sample index inside the split subset')
    ap.add_argument('--split', type=str, default='train',
                    choices=['train', 'val', 'test', 'None'])
    ap.add_argument('--num_src_views', type=int, default=3)
    ap.add_argument('--frame_subsample', type=int, default=1)
    ap.add_argument('--train_ratio', type=float, default=0.9)
    ap.add_argument('--split_seed', type=int, default=0)
    ap.add_argument('--view_seed', type=int, default=2025)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_dir', type=str, default='overfit_debug')
    ap.add_argument('--run_name', type=str, default='')
    ap.add_argument('--stat_every', type=int, default=20)
    ap.add_argument('--diff_against', type=str, default='')

    # mask/weight knobs (keep same spirit as training defaults)
    ap.add_argument('--conf_thr', type=float, default=0.2)
    ap.add_argument('--conf_temp', type=float, default=0.06)
    ap.add_argument('--use_conf_loss_gate', type=int, default=1, choices=[0, 1])
    ap.add_argument('--train_min_cover', type=float, default=0.10)
    ap.add_argument('--fg_thr', type=float, default=0.5)
    ap.add_argument('--fg_min_cover', type=float, default=0.05)
    ap.add_argument('--fg_dilate_k', type=int, default=7)
    ap.add_argument('--fg_keep_largest_cc', type=int, default=1, choices=[0, 1])
    ap.add_argument('--fg_lcc_min_pixels', type=int, default=32)
    ap.add_argument('--valid_min_cover', type=float, default=0.10)
    ap.add_argument('--valid_dilate_k', type=int, default=7)
    ap.add_argument('--valid_k_max', type=int, default=31)
    ap.add_argument('--bg_weight', type=float, default=0.05)
    ap.add_argument('--conf_raw_min', type=float, default=0.0)
    ap.add_argument('--conf_raw_max', type=float, default=1.0)
    ap.add_argument('--conf_auto_norm', type=int, default=0, choices=[0, 1])
    ap.add_argument('--conf_use_quantile', type=int, default=0, choices=[0, 1])
    ap.add_argument('--conf_qlo', type=float, default=0.05)
    ap.add_argument('--conf_qhi', type=float, default=0.95)

    # model knobs (match ablation)
    ap.add_argument('--ref_mode', type=str, default='first',
                    choices=['first', 'mean'])
    ap.add_argument('--use_conf_gate', type=int, default=1, choices=[0, 1])
    ap.add_argument('--conf_gate_detach', type=int, default=0, choices=[0, 1])
    ap.add_argument('--conf_gate_floor', type=float, default=0.0)
    ap.add_argument('--conf_bias_init', type=float, default=-1.0)
    ap.add_argument('--use_tone', type=int, default=0, choices=[0, 1])
    ap.add_argument('--init_alpha', type=float, default=1.0)
    ap.add_argument('--rgb_sigmoid_temp', type=float, default=1.0)
    ap.add_argument('--conf_sigmoid_temp', type=float, default=1.0)
    ap.add_argument('--print_logits', type=int, default=0, choices=[0, 1])

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir
    if str(args.run_name).strip():
        out_dir = os.path.join(out_dir, str(args.run_name).strip())
    os.makedirs(out_dir, exist_ok=True)

    split = None if args.split == 'None' else args.split
    deterministic_views = True  # overfit 必须固定 src/tgt，不然你在追移动靶
    ds = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=args.seq_names,
        split=split,
        num_src_views=int(args.num_src_views),
        frame_subsample=int(args.frame_subsample),
        train_ratio=float(args.train_ratio),
        split_seed=int(args.split_seed),
        deterministic_views=deterministic_views,
        view_seed=int(args.view_seed),
    )
    if len(ds) == 0:
        raise SystemExit('dataset empty')

    idx = int(args.index) % len(ds)
    batch = ds[idx]
    # --- ENSURE_BATCH_DIMS: overfit uses a single sample; add batch dim to match train/eval ---
    def _ensure_dim(x, want_dim):
        if not torch.is_tensor(x):
            return x
        if x.dim() == 0 and want_dim == 1:
            return x.view(1)
        if x.dim() == want_dim - 1:
            return x.unsqueeze(0)
        return x

    batch['src_imgs'] = _ensure_dim(batch.get('src_imgs'), 5)
    batch['src_depth'] = _ensure_dim(batch.get('src_depth'), 5)
    batch['src_depth_conf'] = _ensure_dim(batch.get('src_depth_conf'), 5)
    batch['src_pointmap'] = _ensure_dim(batch.get('src_pointmap'), 5)
    batch['tgt_img'] = _ensure_dim(batch.get('tgt_img'), 4)
    batch['tgt_depth'] = _ensure_dim(batch.get('tgt_depth'), 4)
    batch['tgt_depth_conf'] = _ensure_dim(batch.get('tgt_depth_conf'), 4)
    batch['tgt_conf'] = _ensure_dim(batch.get('tgt_conf'), 4)
    batch['tgt_pointmap'] = _ensure_dim(batch.get('tgt_pointmap'), 4)
    batch['src_vids'] = _ensure_dim(batch.get('src_vids'), 2)
    batch['tgt_vid'] = _ensure_dim(batch.get('tgt_vid'), 1)

    # move device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}

    model = GeomViewDecoderAblation(
        ref_mode=str(args.ref_mode),
        use_conf_gate=bool(args.use_conf_gate),
        use_tone=bool(args.use_tone),
        init_alpha=float(args.init_alpha),
        use_view_cond=False,
        num_views=0,
        view_dim=16,
        view_affine_strength=1.0,
        view_cond_mode='tgt',
        rgb_sigmoid_temp=float(args.rgb_sigmoid_temp),
        conf_sigmoid_temp=float(args.conf_sigmoid_temp),
        conf_gate_detach=bool(args.conf_gate_detach),
        conf_gate_floor=float(args.conf_gate_floor),
        conf_bias_init=(None if float(args.conf_bias_init) < 0 else float(args.conf_bias_init)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    # fixed target
    src_imgs = batch['src_imgs']
    src_depth = batch['src_depth']
    src_depth_conf = batch['src_depth_conf']
    src_pointmap = batch['src_pointmap']
    tgt_img = batch['tgt_img']
    tgt_vid = batch.get('tgt_vid', None)
    src_vids = batch.get('src_vids', None)

    print(
        f'[overfit] split={split} len={len(ds)} pick index={idx} device={device} '
        f'use_conf_gate={int(bool(args.use_conf_gate))} '
        f'use_conf_loss_gate={int(bool(args.use_conf_loss_gate))} '
        f'conf_gate_detach={int(bool(args.conf_gate_detach))} '
        f'conf_gate_floor={float(args.conf_gate_floor):.3f} '
        f'conf_bias_init={float(args.conf_bias_init):.3f} '
        f'out_dir={out_dir}'
    )
    t0 = time.time()

    def _stats(x: torch.Tensor, mask: torch.Tensor = None):
        if x is None or (not torch.is_tensor(x)):
            return None
        x = x.detach()
        if mask is not None:
            m = mask.detach()
            if m.ndim != x.ndim:
                m = m.expand_as(x)
            m = (m > 0.5)
            if m.sum().item() == 0:
                return None
            x = x[m]
        else:
            x = x.reshape(-1)
        if x.numel() == 0:
            return None
        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item())
        mn = float(x.min().item())
        mx = float(x.max().item())
        return mean, mn, mx, std

    def _fmt_stats(name, s_full, s_mask):
        if s_full is None:
            return f'{name}: NA'
        fmean, fmin, fmax, fstd = s_full
        if s_mask is None:
            return (f'{name} full(mean/min/max/std)='
                    f'{fmean:.4f}/{fmin:.4f}/{fmax:.4f}/{fstd:.4f} '
                    f'train=NA')
        mmean, mmin, mmax, mstd = s_mask
        return (f'{name} full(mean/min/max/std)='
                f'{fmean:.4f}/{fmin:.4f}/{fmax:.4f}/{fstd:.4f} '
                f'train(mean/min/max/std)='
                f'{mmean:.4f}/{mmin:.4f}/{mmax:.4f}/{mstd:.4f}')

    def _pct(x: torch.Tensor, thr: float, mode: str):
        if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
            return float('nan')
        if mode == "lt":
            return float((x < thr).float().mean().item() * 100.0)
        return float((x > thr).float().mean().item() * 100.0)

    def _diff_png(path_a: str, path_b: str, out_path: str):
        if (not os.path.isfile(path_a)) or (not os.path.isfile(path_b)):
            return False
        a = np.array(Image.open(path_a))
        b = np.array(Image.open(path_b))
        if a.shape != b.shape:
            return False
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
        Image.fromarray(diff).save(out_path)
        return True
    for step in range(1, int(args.steps)+1):
        model.train()
        opt.zero_grad(set_to_none=True)

        with autocast_ctx(device=str(device), enabled=(str(device).startswith('cuda'))):
            pred_rgb, pred_conf, aux = model(
                src_imgs, src_depth, src_depth_conf, src_pointmap,
                tgt_vid=tgt_vid, src_vids=src_vids, return_aux=True
            )
            H, W = pred_rgb.shape[-2:]
            train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf_t, aux_masks = build_masks_from_batch(
                batch={'tgt_depth_conf': batch.get('tgt_depth_conf', None),
                       'tgt_conf': batch.get('tgt_conf', None),
                       'tgt_depth': batch.get('tgt_depth', None),
                       'tgt_pointmap': batch.get('tgt_pointmap', None),
                       **batch},
                pred_hw=(H, W),
                device=str(device),
                conf_thr=float(args.conf_thr),
                conf_temp=float(args.conf_temp),
                train_min_cover=float(args.train_min_cover),
                fg_thr=float(args.fg_thr),
                fg_min_cover=float(args.fg_min_cover),
                fg_dilate_k=int(args.fg_dilate_k),
                fg_keep_largest_cc=bool(args.fg_keep_largest_cc),
                fg_lcc_min_pixels=int(args.fg_lcc_min_pixels),
                valid_min_cover=float(args.valid_min_cover),
                valid_dilate_k=int(args.valid_dilate_k),
                valid_k_max=int(args.valid_k_max),
                bg_weight=float(args.bg_weight),
                conf_raw_min=float(args.conf_raw_min),
                conf_raw_max=float(args.conf_raw_max),
                conf_auto_norm=bool(args.conf_auto_norm),
                conf_use_quantile=bool(args.conf_use_quantile),
                conf_qlo=float(args.conf_qlo),
                conf_qhi=float(args.conf_qhi),
                use_conf_in_train_mask=bool(args.use_conf_loss_gate),
            )
            loss = masked_l1(pred_rgb, tgt_img, recon_weight)

        loss.backward()
        opt.step()

        if step % int(args.stat_every) == 0 or step == 1:
            dt = time.time() - t0
            with torch.no_grad():
                l1_full = (pred_rgb - tgt_img).abs().mean()
                train_mean = train_mask.float().mean()
                weight_mean = recon_weight.float().mean()
                weight_ratio = weight_mean / (train_mean + 1e-8)
                gate = aux.get('gate', None) if isinstance(aux, dict) else None
                gate_active = bool(aux.get('use_conf_gate', False)) if isinstance(aux, dict) else False
                gate_mode = "conf" if gate_active else "disabled->ones"
                if (gate is None) and pred_conf is not None:
                    gate = torch.ones_like(pred_conf)
                if int(args.print_logits):
                    rgb_logits = aux.get('rgb_logits', None) if isinstance(aux, dict) else None
                    conf_logits = aux.get('conf_logits', None) if isinstance(aux, dict) else None
                    if rgb_logits is not None and conf_logits is not None:
                        rgb_lmin = rgb_logits.min().item()
                        rgb_lmax = rgb_logits.max().item()
                        rgb_lmean = rgb_logits.mean().item()
                        conf_lmin = conf_logits.min().item()
                        conf_lmax = conf_logits.max().item()
                        conf_lmean = conf_logits.mean().item()
                    else:
                        rgb_lmin = rgb_lmax = rgb_lmean = float('nan')
                        conf_lmin = conf_lmax = conf_lmean = float('nan')
            print(
                f'step {step:04d}  loss={float(loss.item()):.6f}  '
                f'l1_full={float(l1_full.item()):.6f}  '
                f'w_mean={float(weight_mean.item()):.6f}  '
                f'train_mean={float(train_mean.item()):.6f}  '
                f'w_over_train={float(weight_ratio.item()):.6f}  '
                + (f'  rgb_logit=[{rgb_lmin:.2f},{rgb_lmax:.2f}] mean={rgb_lmean:.2f}'
                   f'  conf_logit=[{conf_lmin:.2f},{conf_lmax:.2f}] mean={conf_lmean:.2f}'
                   if int(args.print_logits) else '')
                + f'  dt={dt:.1f}s'
            )
            pc_full = _stats(pred_conf, None)
            pc_mask = _stats(pred_conf, train_mask)
            gate_full = _stats(gate, None)
            gate_mask = _stats(gate, train_mask)
            rw_full = _stats(recon_weight, None)
            rw_mask = _stats(recon_weight, train_mask)
            pct_lo = _pct(gate, float(args.conf_gate_floor), "lt")
            pct_hi = _pct(gate, 0.9, "gt")
            print(f'[stats] {_fmt_stats("pred_conf", pc_full, pc_mask)}')
            print(f'[stats] {_fmt_stats(f"gate({gate_mode})", gate_full, gate_mask)} '
                  f'pct<floor={pct_lo:.2f}% pct>0.9={pct_hi:.2f}%')
            print(f'[stats] {_fmt_stats("recon_weight", rw_full, rw_mask)} '
                  f'conf_gate_mode={aux_masks.get("conf_gate_mode", "conf_soft")}')

        if step % 100 == 0 or step in (1,):
            model.eval()
            aux_dbg = {
                'pred_conf': pred_conf.detach(),
                'gate': aux.get('gate', None) if isinstance(aux, dict) else None,
                'tgt_depth_conf': aux_masks.get('tgt_depth_conf', None),
                'tgt_depth_conf_raw': aux_masks.get('tgt_depth_conf_raw', None),
                'fg_mask': fg_mask.detach(),
                'train_mask': train_mask.detach(),
                'recon_weight': recon_weight.detach(),
                'valid_mask': valid_mask.detach(),
            }
            save_debug_pack(pred_rgb.detach(), tgt_img.detach(), aux_dbg, step,
                            out_dir=out_dir, prefix=f'overfit_i{idx}', split_cat_panels=True)

            diff_against = str(args.diff_against).strip()
            if diff_against and os.path.abspath(diff_against) != os.path.abspath(out_dir):
                diff_dir = os.path.join(
                    out_dir, f'diff_vs_{os.path.basename(diff_against)}')
                os.makedirs(diff_dir, exist_ok=True)
                for key in ("pred_conf", "recon_weight", "gate"):
                    fname = f'overfit_i{idx}_{key}_step{step:06d}.png'
                    path_a = os.path.join(out_dir, fname)
                    path_b = os.path.join(diff_against, fname)
                    out_path = os.path.join(diff_dir, f'diff_{key}_step{step:06d}.png')
                    _diff_png(path_a, path_b, out_path)

    print('[done] check images in', out_dir)


if __name__ == '__main__':
    main()
