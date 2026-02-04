import os
import math
import time
import json
import argparse
import numpy as np
from PIL import Image
import torch

from zju_dataset_view import ZJUViewSynthDataset
from view_decoder_ablation import GeomViewDecoderAblation
from train_view_decoder_ablation import (
    autocast_ctx, build_masks_from_batch, masked_l1, save_debug_pack, build_vis_ranges
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zju_root', type=str, required=True)
    ap.add_argument('--seq_names', type=str, nargs='+', required=True)
    ap.add_argument('--index', type=int, default=0,
                    help='sample index inside the split subset')
    ap.add_argument('--seed', type=int, default=0)
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
    ap.add_argument('--train_mask_mode', type=str, default='fg_conf',
                    choices=['fg_conf', 'valid_conf', 'valid_only'])
    ap.add_argument('--recon_mask_mode', type=str, default='valid',
                    choices=['fg', 'train', 'valid'])
    ap.add_argument('--recon_weight_renorm', type=int,
                    default=0, choices=[0, 1])
    ap.add_argument('--recon_weight_clip_max',
                    type=float, default=1.0)

    # model knobs (match ablation)
    ap.add_argument('--ref_mode', type=str, default='first',
                    choices=['first', 'mean'])
    ap.add_argument('--use_conf_gate', type=int, default=1, choices=[0, 1])
    ap.add_argument('--conf_gate_detach', type=int, default=1, choices=[0, 1])
    ap.add_argument('--conf_weight_detach', type=int, default=1, choices=[0, 1])
    ap.add_argument('--conf_gate_floor', type=float, default=0.0)
    ap.add_argument('--conf_gate_gamma', type=float, default=2.0)
    ap.add_argument('--recon_gate_floor', type=float, default=0.1)
    ap.add_argument('--conf_gate_warmup', type=int, default=1000)
    ap.add_argument('--conf_gate_ramp', type=int, default=0)
    ap.add_argument('--conf_gate_ramp_mode', type=str, default='smoothstep',
                    choices=['linear', 'cosine', 'exp', 'smoothstep'])
    ap.add_argument('--conf_gate_ramp_k', type=float, default=5.0)
    ap.add_argument('--conf_bias_init', type=float, default=-1.0,
                    help='Init conf bias; -1 disables, (0,1) treated as prob, other values as logits (temp-aware)')
    ap.add_argument('--use_tone', type=int, default=0, choices=[0, 1])
    ap.add_argument('--init_alpha', type=float, default=1.0)
    ap.add_argument('--rgb_sigmoid_temp', type=float, default=1.0)
    ap.add_argument('--conf_sigmoid_temp', type=float, default=1.0)
    ap.add_argument('--split_conf_head', type=int, default=0, choices=[0, 1])
    ap.add_argument('--logit_clip', type=float, default=10.0)
    ap.add_argument('--print_logits', type=int, default=0, choices=[0, 1])
    ap.add_argument('--conf_head_lr_mult', type=float, default=2.0)
    ap.add_argument('--use_depth_head', type=int, default=0, choices=[0, 1])
    ap.add_argument('--vis_conf_min', type=float, default=0.0)
    ap.add_argument('--vis_conf_max', type=float, default=1.0)
    ap.add_argument('--vis_depth_min', type=float, default=0.0)
    ap.add_argument('--vis_depth_max', type=float, default=5.0)
    ap.add_argument('--vis_mask_min', type=float, default=0.0)
    ap.add_argument('--vis_mask_max', type=float, default=1.0)
    ap.add_argument('--vis_weight_min', type=float, default=0.0)
    ap.add_argument('--vis_weight_max', type=float, default=1.0)

    args = ap.parse_args()
    vis_ranges = build_vis_ranges(args)

    def _seed_all(seed: int):
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_all(args.seed)

    def _resolve_conf_bias_init(raw: float):
        try:
            val = float(raw)
        except Exception:
            return None
        if math.isnan(val) or val == -1.0:
            return None
        return val

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir
    if str(args.run_name).strip():
        out_dir = os.path.join(out_dir, str(args.run_name).strip())
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    config_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

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

    conf_bias_used = _resolve_conf_bias_init(args.conf_bias_init)
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
        split_conf_head=bool(args.split_conf_head),
        conf_gate_detach=bool(args.conf_gate_detach),
        conf_gate_floor=float(args.conf_gate_floor),
        conf_bias_init=conf_bias_used,
        logit_clip=float(args.logit_clip),
        use_depth_head=bool(args.use_depth_head),
    ).to(device)

    conf_bias_param = None
    try:
        if bool(getattr(model.core, "split_conf_head", False)) and hasattr(model.core, "out_conf"):
            bias = model.core.out_conf.bias
            if bias is not None and bias.numel() >= 1:
                conf_bias_param = float(bias[0].detach().cpu().item())
        else:
            bias = model.core.out_conv.bias
            if bias is not None and bias.numel() >= 4:
                conf_bias_param = float(bias[3].detach().cpu().item())
    except Exception:
        pass
    conf_sigmoid_temp = float(getattr(model.core, "conf_sigmoid_temp", 1.0))
    conf_bias_prob = None
    if conf_bias_param is not None:
        t_conf = conf_sigmoid_temp if conf_sigmoid_temp > 0 else 1.0
        conf_bias_prob = float(torch.sigmoid(
            torch.tensor(conf_bias_param / t_conf)).item())
    conf_bias_used_str = "NA" if conf_bias_used is None else f"{conf_bias_used:.3f}"
    conf_bias_param_str = "NA" if conf_bias_param is None else f"{conf_bias_param:.3f}"
    conf_bias_prob_str = "NA" if conf_bias_prob is None else f"{conf_bias_prob:.4f}"
    print(
        f"[conf_head] conf_bias_arg={float(args.conf_bias_init):.3f} "
        f"conf_bias_used={conf_bias_used_str} "
        f"conf_bias_param={conf_bias_param_str} "
        f"conf_sigmoid_temp={conf_sigmoid_temp:.3f} "
        f"bias_as_conf={conf_bias_prob_str}"
    )

    conf_head_lr_mult = float(getattr(args, "conf_head_lr_mult", 1.0))
    if conf_head_lr_mult != 1.0:
        head_params = []
        base_params = []
        head_key = "core.out_conf" if bool(
            getattr(model.core, "split_conf_head", False)) else "core.out_conv"
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if head_key in name:
                head_params.append(p)
            else:
                base_params.append(p)
        if head_params and base_params:
            opt = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": float(args.lr)},
                    {"params": head_params, "lr": float(args.lr) *
                        conf_head_lr_mult},
                ],
                weight_decay=1e-4,
            )
            print(
                f"[conf_head] lr_mult={conf_head_lr_mult:.3f} "
                f"head_key={head_key} base={len(base_params)} head={len(head_params)}"
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    else:
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
        f'train_mask_mode={str(args.train_mask_mode)} '
        f'recon_mask_mode={str(args.recon_mask_mode)} '
        f'conf_gate_detach={int(bool(args.conf_gate_detach))} '
        f'conf_weight_detach={int(bool(args.conf_weight_detach))} '
        f'conf_gate_floor={float(args.conf_gate_floor):.3f} '
        f'conf_gate_gamma={float(args.conf_gate_gamma):.3f} '
        f'recon_gate_floor={float(args.recon_gate_floor):.3f} '
        f'conf_gate_warmup={int(args.conf_gate_warmup)} '
        f'conf_gate_ramp={int(args.conf_gate_ramp)} '
        f'conf_gate_ramp_mode={str(args.conf_gate_ramp_mode)} '
        f'recon_weight_renorm={int(bool(args.recon_weight_renorm))} '
        f'recon_weight_clip_max={float(args.recon_weight_clip_max):.3f} '
        f'conf_bias_init={float(args.conf_bias_init):.3f} '
        f'split_conf_head={int(bool(args.split_conf_head))} '
        f'logit_clip={float(args.logit_clip):.3f} '
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

    def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
        mse = torch.mean((x - y) ** 2).clamp_min(eps)
        return float((10.0 * torch.log10(1.0 / mse)).item())

    def _ssim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> float:
        # simple SSIM on luminance
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)
        xg = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        yg = 0.2989 * y[:, 0:1] + 0.5870 * y[:, 1:2] + 0.1140 * y[:, 2:3]
        # gaussian blur (11x11)
        def _gauss_kernel(device):
            ksize = 11
            sigma = 1.5
            coords = torch.arange(ksize, device=device) - ksize // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
            g = g / g.sum()
            k = (g[:, None] * g[None, :]).view(1, 1, ksize, ksize)
            return k
        k = _gauss_kernel(x.device)
        mu_x = torch.nn.functional.conv2d(xg, k, padding=5)
        mu_y = torch.nn.functional.conv2d(yg, k, padding=5)
        sigma_x = torch.nn.functional.conv2d(xg * xg, k, padding=5) - mu_x * mu_x
        sigma_y = torch.nn.functional.conv2d(yg * yg, k, padding=5) - mu_y * mu_y
        sigma_xy = torch.nn.functional.conv2d(xg * yg, k, padding=5) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / (
            (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + eps
        )
        return float(ssim_map.mean().item())

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

    def _conf_gate_strength(step: int) -> float:
        if not bool(args.use_conf_gate):
            return 0.0
        warm = int(getattr(args, "conf_gate_warmup", 0))
        ramp = int(getattr(args, "conf_gate_ramp", 0))
        if warm > 0 and int(step) < warm:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = max(0, int(step) - warm)
        progress = min(1.0, float(t) / float(ramp))
        mode = str(getattr(args, "conf_gate_ramp_mode", "linear")).lower()
        if mode == "smoothstep":
            return progress * progress * (3.0 - 2.0 * progress)
        if mode == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        if mode == "exp":
            k = float(getattr(args, "conf_gate_ramp_k", 5.0))
            k = max(1e-6, k)
            return (1.0 - math.exp(-k * progress)) / (1.0 - math.exp(-k))
        return progress
    for step in range(1, int(args.steps)+1):
        model.train()
        opt.zero_grad(set_to_none=True)

        with autocast_ctx(device=str(device), enabled=(str(device).startswith('cuda'))):
            conf_gate_strength = _conf_gate_strength(step)
            pred_rgb, pred_conf, aux = model(
                src_imgs, src_depth, src_depth_conf, src_pointmap,
                tgt_vid=tgt_vid, src_vids=src_vids, return_aux=True,
                use_conf_gate_override=bool(args.use_conf_gate),
                conf_gate_strength=conf_gate_strength
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
                train_mask_mode=str(args.train_mask_mode),
                pred_conf_gate=pred_conf,
                use_conf_gate=bool(args.use_conf_gate),
                conf_gate_detach=bool(args.conf_gate_detach),
                conf_weight_detach=bool(args.conf_weight_detach),
                conf_gate_floor=float(args.conf_gate_floor),
                conf_gate_gamma=float(args.conf_gate_gamma),
                conf_gate_strength=float(conf_gate_strength),
                recon_gate_floor=float(args.recon_gate_floor),
                recon_mask_mode=str(args.recon_mask_mode),
                recon_weight_renorm=bool(args.recon_weight_renorm),
                recon_weight_clip_max=float(args.recon_weight_clip_max),
            )
            recon_weight_loss = recon_weight.detach()
            loss = masked_l1(pred_rgb, tgt_img, recon_weight_loss)

        loss.backward()
        opt.step()

        if step % int(args.stat_every) == 0 or step == 1:
            dt = time.time() - t0
            with torch.no_grad():
                l1_full = (pred_rgb - tgt_img).abs().mean()
                psnr_full = _psnr(pred_rgb, tgt_img)
                ssim_full = _ssim(pred_rgb, tgt_img)
                train_mean = train_mask.float().mean()
                weight_mean = recon_weight.float().mean()
                weight_ratio = weight_mean / (train_mean + 1e-8)
                gate = aux.get('gate', None) if isinstance(aux, dict) else None
                gate_active = bool(aux.get('use_conf_gate', False)) if isinstance(aux, dict) else False
                gate_mode = "conf" if gate_active else "disabled->ones"
                gate_loss = aux_masks.get(
                    'gate_loss', None) if isinstance(aux_masks, dict) else None
                gate_loss_active = bool(aux_masks.get(
                    'use_conf_gate_loss', False)) if isinstance(aux_masks, dict) else False
                gate_loss_mode = "conf" if gate_loss_active else "disabled->ones"
                if (gate is None) and pred_conf is not None:
                    gate = torch.ones_like(pred_conf)
                if (gate_loss is None) and pred_conf is not None:
                    gate_loss = torch.ones_like(pred_conf)
                conf_re_diff = float('nan')
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
                        if pred_conf is not None:
                            t_conf = float(getattr(model.core, "conf_sigmoid_temp", 1.0))
                            if t_conf <= 0:
                                t_conf = 1.0
                            conf_re = torch.sigmoid(conf_logits / t_conf)
                            conf_re_diff = float(
                                (conf_re - pred_conf).abs().max().item())
                    else:
                        rgb_lmin = rgb_lmax = rgb_lmean = float('nan')
                        conf_lmin = conf_lmax = conf_lmean = float('nan')
            print(
                f'step {step:04d}  loss={float(loss.item()):.6f}  '
                f'l1_full={float(l1_full.item()):.6f}  '
                f'psnr={psnr_full:.2f}  '
                f'ssim={ssim_full:.4f}  '
                f'w_mean={float(weight_mean.item()):.6f}  '
                f'train_mean={float(train_mean.item()):.6f}  '
                f'w_over_train={float(weight_ratio.item()):.6f}  '
                f'gate_strength={float(conf_gate_strength):.3f}  '
                + (f'  rgb_logit=[{rgb_lmin:.2f},{rgb_lmax:.2f}] mean={rgb_lmean:.2f}'
                   f'  conf_logit=[{conf_lmin:.2f},{conf_lmax:.2f}] mean={conf_lmean:.2f}'
                   f'  conf_re_diff={conf_re_diff:.3e}'
                   if int(args.print_logits) else '')
                + f'  dt={dt:.1f}s'
            )
            pc_full = _stats(pred_conf, None)
            pc_mask = _stats(pred_conf, train_mask)
            gate_full = _stats(gate, None)
            gate_mask = _stats(gate, train_mask)
            gate_loss_full = _stats(gate_loss, None)
            gate_loss_mask = _stats(gate_loss, train_mask)
            rw_raw = aux_masks.get(
                "recon_weight_raw", None) if isinstance(aux_masks, dict) else None
            rw_raw_full = _stats(rw_raw, None)
            rw_raw_mask = _stats(rw_raw, train_mask)
            rw_full = _stats(recon_weight, None)
            rw_mask = _stats(recon_weight, train_mask)
            pct_lo = _pct(gate_loss, float(args.conf_gate_floor), "lt")
            pct_hi = _pct(gate_loss, 0.9, "gt")
            print(f'[stats] {_fmt_stats("pred_conf", pc_full, pc_mask)}')
            print(f'[stats] {_fmt_stats(f"gate_skip({gate_mode})", gate_full, gate_mask)}')
            print(f'[stats] {_fmt_stats(f"gate_loss({gate_loss_mode})", gate_loss_full, gate_loss_mask)} '
                  f'pct<floor={pct_lo:.2f}% pct>0.9={pct_hi:.2f}%')
            print(f'[stats] {_fmt_stats("recon_weight_raw", rw_raw_full, rw_raw_mask)}')
            print(f'[stats] {_fmt_stats("recon_weight", rw_full, rw_mask)} '
                  f'conf_gate_mode={aux_masks.get("conf_gate_mode", "conf_soft")} '
                  f'train_mask_mode={aux_masks.get("train_mask_mode", "fg_conf")} '
                  f'recon_mask_mode={aux_masks.get("recon_mask_mode", "fg")} '
                  f'renorm={int(aux_masks.get("recon_weight_renorm", 0))} '
                  f'clip_max={float(aux_masks.get("recon_weight_clip_max", 1.0)):.3f} '
                  f'gate_gamma={float(aux_masks.get("conf_gate_gamma", 1.0)):.3f}')
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": int(step),
                    "loss": float(loss.item()),
                    "l1_full": float(l1_full.item()),
                    "psnr": float(psnr_full),
                    "ssim": float(ssim_full),
                    "train_mean": float(train_mean.item()),
                    "weight_mean": float(weight_mean.item()),
                    "gate_strength": float(conf_gate_strength),
                    "time_sec": float(dt),
                }) + "\n")

        if step % 100 == 0 or step in (1,):
            model.eval()
            aux_dbg = {
                'pred_conf': pred_conf.detach(),
                'gate': aux.get('gate', None) if isinstance(aux, dict) else None,
                'gate_loss': aux_masks.get('gate_loss', None),
                'tgt_depth_conf': aux_masks.get('tgt_depth_conf', None),
                'tgt_depth_conf_raw': aux_masks.get('tgt_depth_conf_raw', None),
                'tgt_depth': batch.get('tgt_depth', None),
                'fg_mask': fg_mask.detach(),
                'train_mask': train_mask.detach(),
                'recon_weight_raw': aux_masks.get('recon_weight_raw', None),
                'recon_weight': recon_weight.detach(),
                'valid_mask': valid_mask.detach(),
            }
            if isinstance(aux, dict) and aux.get("pred_depth", None) is not None:
                aux_dbg["pred_depth"] = aux.get("pred_depth")
            save_debug_pack(pred_rgb.detach(), tgt_img.detach(), aux_dbg, step,
                            out_dir=out_dir, prefix=f'overfit_i{idx}', split_cat_panels=True,
                            fixed_ranges=vis_ranges)

            diff_against = str(args.diff_against).strip()
            if diff_against and os.path.abspath(diff_against) != os.path.abspath(out_dir):
                diff_dir = os.path.join(
                    out_dir, f'diff_vs_{os.path.basename(diff_against)}')
                os.makedirs(diff_dir, exist_ok=True)
                for key in ("pred_conf", "recon_weight_raw", "recon_weight",
                            "gate", "gate_loss"):
                    fname = f'overfit_i{idx}_{key}_step{step:06d}.png'
                    path_a = os.path.join(out_dir, fname)
                    path_b = os.path.join(diff_against, fname)
                    out_path = os.path.join(diff_dir, f'diff_{key}_step{step:06d}.png')
                    _diff_png(path_a, path_b, out_path)

    print('[done] check images in', out_dir)


if __name__ == '__main__':
    main()
