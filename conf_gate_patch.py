# conf_gate_patch.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------
# 1) Robust helpers
# -------------------------
def to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert tensor to float32 in [0, 1].
    - If values look like 0~255, divide by 255.
    - Clamp to [0,1].
    """
    if x is None:
        raise ValueError("to_01 got None")

    x = x.float()
    # Heuristic: if max is > 1.5, it's probably 0~255 (or 0~65535, but your masks/conf are typically 8-bit)
    maxv = float(x.detach().amax().cpu())
    if maxv > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask
    # maxpool dilation
    return F.max_pool2d(mask, kernel_size=2 * k + 1, stride=1, padding=k)


def _erode(mask: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask
    # erosion via dilation on inverted mask
    return 1.0 - _dilate(1.0 - mask, k)


def morph(mask: torch.Tensor, erode_k: int = 0, dilate_k: int = 0) -> torch.Tensor:
    mask = _erode(mask, erode_k)
    mask = _dilate(mask, dilate_k)
    return mask


def quantile_norm_01(conf: torch.Tensor, q_min: float = 0.05, q_max: float = 0.95, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-image quantile normalization to [0,1].
    conf: [B,1,H,W] or [B,H,W]
    """
    conf = to_01(conf)
    if conf.dim() == 3:
        conf = conf.unsqueeze(1)

    B = conf.shape[0]
    flat = conf.view(B, -1)

    lo = torch.quantile(flat, q_min, dim=1, keepdim=True)
    hi = torch.quantile(flat, q_max, dim=1, keepdim=True)

    flat_n = (flat - lo) / (hi - lo + eps)
    conf_n = flat_n.view_as(conf).clamp(0.0, 1.0)
    return conf_n


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss for smoothing (expects [B,1,H,W] or [B,C,H,W]).
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


# -------------------------
# 2) Config
# -------------------------
@dataclass
class ConfGateCfg:
    enable: bool = True

    # gate modes: "soft" | "hard" | "sigmoid" | "pow"
    mode: str = "sigmoid"

    # threshold in [0,1] (only meaningful after to_01 / quantile_norm_01)
    thr: float = 0.2

    # for sigmoid gate
    temp: float = 0.05

    # quantile norm
    q_min: float = 0.05
    q_max: float = 0.95

    # avoid zero-weight gradient death
    min_w: float = 0.05

    # for pow gate
    gamma: float = 1.0

    # whether to multiply fg/valid masks into weight
    use_fg: bool = True
    use_valid: bool = True

    # morphology
    fg_erode: int = 0
    fg_dilate: int = 0
    valid_erode: int = 0
    valid_dilate: int = 0


# -------------------------
# 3) Build masks + recon weight
# -------------------------
@torch.no_grad()
def build_fg_mask(train_mask: torch.Tensor, thr: float = 0.5, erode_k: int = 0, dilate_k: int = 0) -> torch.Tensor:
    """
    train_mask: [B,1,H,W] or [B,H,W] possibly uint8 0~255
    returns fg_mask float {0,1} same shape as [B,1,H,W]
    """
    m = to_01(train_mask)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    fg = (m > thr).float()
    fg = morph(fg, erode_k=erode_k, dilate_k=dilate_k)
    return fg


@torch.no_grad()
def build_valid_mask(depth_conf_raw: torch.Tensor, thr: float, erode_k: int = 0, dilate_k: int = 0) -> torch.Tensor:
    """
    depth_conf_raw: [B,1,H,W] or [B,H,W] possibly uint8 0~255
    returns valid_mask float {0,1}
    """
    c = to_01(depth_conf_raw)
    if c.dim() == 3:
        c = c.unsqueeze(1)
    vm = (c > thr).float()
    vm = morph(vm, erode_k=erode_k, dilate_k=dilate_k)
    return vm


def compute_recon_weight(
    pred_conf: torch.Tensor,
    cfg: ConfGateCfg,
    fg_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    pred_conf: [B,1,H,W] (can be logits or prob-like; we normalize robustly anyway)
    returns:
      recon_weight: [B,1,H,W] in [min_w, 1]
      debug dict
    """
    # Normalize to [0,1] robustly + quantile stretch
    conf_n = quantile_norm_01(
        pred_conf, q_min=cfg.q_min, q_max=cfg.q_max)  # [B,1,H,W]

    if not cfg.enable:
        w = torch.ones_like(conf_n)
    else:
        mode = cfg.mode.lower()
        if mode == "hard":
            w = (conf_n > cfg.thr).float()
        elif mode == "sigmoid":
            w = torch.sigmoid((conf_n - cfg.thr) / max(cfg.temp, 1e-6))
        elif mode == "pow":
            w = conf_n.clamp(0.0, 1.0).pow(cfg.gamma)
        elif mode == "soft":
            w = conf_n
        else:
            raise ValueError(f"Unknown conf gate mode: {cfg.mode}")

    if cfg.use_fg and fg_mask is not None:
        w = w * fg_mask
    if cfg.use_valid and valid_mask is not None:
        w = w * valid_mask

    # prevent full zero weights
    w = cfg.min_w + (1.0 - cfg.min_w) * w
    w = w.clamp(cfg.min_w, 1.0)

    dbg = {
        "pred_conf_qnorm": conf_n.detach(),
        "recon_weight": w.detach(),
        "mean_conf": conf_n.mean().detach(),
        "mean_w": w.mean().detach(),
    }
    if fg_mask is not None:
        dbg["mean_fg"] = fg_mask.mean().detach()
    if valid_mask is not None:
        dbg["mean_valid"] = valid_mask.mean().detach()

    return w, dbg


# -------------------------
# 4) Weighted losses (stable denominators)
# -------------------------
def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/tgt: [B,C,H,W]
    w: [B,1,H,W] or [B,C,H,W]
    """
    if w.dim() == 4 and w.shape[1] == 1 and pred.shape[1] != 1:
        w = w.expand(-1, pred.shape[1], -1, -1)
    num = (w * (pred - tgt).abs()).sum()
    den = w.sum().clamp_min(eps)
    return num / den


def compute_total_loss(
    pred_rgb: torch.Tensor,
    tgt_rgb: torch.Tensor,
    pred_depth: Optional[torch.Tensor],
    tgt_depth: Optional[torch.Tensor],
    pred_conf: torch.Tensor,
    train_mask: torch.Tensor,
    tgt_depth_conf_raw: Optional[torch.Tensor],
    cfg: ConfGateCfg,
    lambda_rgb: float = 1.0,
    lambda_depth: float = 1.0,
    lambda_conf: float = 1.0,
    lambda_conf_tv: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns total loss and a log dict.
    - train_mask / tgt_depth_conf_raw can be uint8 0~255; handled safely.
    """
    fg_mask = build_fg_mask(train_mask, thr=0.5,
                            erode_k=cfg.fg_erode, dilate_k=cfg.fg_dilate)

    valid_mask = None
    if tgt_depth_conf_raw is not None:
        valid_mask = build_valid_mask(
            tgt_depth_conf_raw, thr=cfg.thr, erode_k=cfg.valid_erode, dilate_k=cfg.valid_dilate)

    recon_w, dbg = compute_recon_weight(
        pred_conf, cfg, fg_mask=fg_mask, valid_mask=valid_mask)

    # weighted reconstruction losses
    loss_rgb = masked_l1(pred_rgb, tgt_rgb, recon_w)

    loss_depth = torch.tensor(0.0, device=pred_rgb.device)
    if pred_depth is not None and tgt_depth is not None:
        # ensure shapes [B,1,H,W]
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if tgt_depth.dim() == 3:
            tgt_depth = tgt_depth.unsqueeze(1)
        loss_depth = masked_l1(pred_depth, tgt_depth, recon_w)

    # confidence regularization: discourage trivial low-confidence everywhere
    # (simple & effective) — push conf upward slightly inside fg/valid region
    conf01 = to_01(pred_conf)
    if conf01.dim() == 3:
        conf01 = conf01.unsqueeze(1)
    reg_region = fg_mask if valid_mask is None else (fg_mask * valid_mask)
    loss_conf_reg = ((1.0 - conf01) * reg_region).mean()

    loss_conf_tv = tv_loss(conf01)

    total = (
        lambda_rgb * loss_rgb
        + lambda_depth * loss_depth
        + lambda_conf * loss_conf_reg
        + lambda_conf_tv * loss_conf_tv
    )

    logs = {
        "loss_total": total.detach(),
        "loss_rgb": loss_rgb.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_conf_reg": loss_conf_reg.detach(),
        "loss_conf_tv": loss_conf_tv.detach(),
        **dbg,
    }

    # also log RAW (unweighted) metrics for sanity
    with torch.no_grad():
        raw_rgb_l1 = (pred_rgb - tgt_rgb).abs().mean()
        logs["raw_rgb_l1"] = raw_rgb_l1.detach()
        if pred_depth is not None and tgt_depth is not None:
            logs["raw_depth_l1"] = (
                pred_depth - tgt_depth).abs().mean().detach()

    return total, logs
