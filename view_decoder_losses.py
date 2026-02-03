# view_decoder_losses.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn.functional as F


def _as_1chw(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Convert mask to (B,1,H,W) aligned with ref (B,C,H,W).
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask[None, None, ...]
    elif mask.dim() == 3:
        mask = mask[:, None, ...] if mask.shape[0] == ref.shape[0] else mask[None, ...]
        if mask.dim() == 3:
            mask = mask[:, None, ...]
    elif mask.dim() == 4:
        if mask.shape[1] != 1:
            mask = mask[:, :1, ...]
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return mask.float()


def masked_l1(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask01: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    pred/tgt: (B,C,H,W)
    mask01: (B,1,H,W) or broadcastable
    """
    m = mask01
    # broadcast to channels
    if m.shape[1] == 1 and pred.shape[1] != 1:
        m = m.expand(-1, pred.shape[1], -1, -1)
    diff = (pred - tgt).abs() * m
    denom = m.sum().clamp_min(eps)
    return diff.sum() / denom


def masked_charbonnier(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask01: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Charbonnier loss: (diff^2 + eps^2)^{alpha}
    """
    diff = pred - tgt
    loss = (diff * diff + (eps * eps)).pow(alpha)
    if mask01 is None:
        return loss.mean()
    m = _as_1chw(mask01, pred)
    if m.shape[1] == 1 and pred.shape[1] != 1:
        m = m.expand(-1, pred.shape[1], -1, -1)
    loss = loss * m
    denom = m.sum().clamp_min(1e-8)
    return loss.sum() / denom


def masked_huber(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask01: Optional[torch.Tensor] = None,
    delta: float = 0.01,
) -> torch.Tensor:
    """
    Huber loss (smooth L1) with delta threshold.
    """
    diff = (pred - tgt).abs()
    delta = float(delta)
    loss = torch.where(diff <= delta, 0.5 * diff * diff / (delta + 1e-8), diff - 0.5 * delta)
    if mask01 is None:
        return loss.mean()
    m = _as_1chw(mask01, pred)
    if m.shape[1] == 1 and pred.shape[1] != 1:
        m = m.expand(-1, pred.shape[1], -1, -1)
    loss = loss * m
    denom = m.sum().clamp_min(1e-8)
    return loss.sum() / denom


def edge_aware_depth_smoothness(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    mask01: Optional[torch.Tensor] = None,
    alpha: float = 10.0,
) -> torch.Tensor:
    """
    Edge-aware smoothness: encourage small depth gradients where RGB is smooth.
    depth: (B,1,H,W), rgb: (B,3,H,W)
    """
    if depth.dim() == 3:
        depth = depth[:, None, ...]
    if rgb.dim() == 3:
        rgb = rgb[:, None, ...]
    depth_dx = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs()
    depth_dy = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()
    img_dx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    img_dy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    wx = torch.exp(-alpha * img_dx)
    wy = torch.exp(-alpha * img_dy)
    loss = (depth_dx * wx).mean() + (depth_dy * wy).mean()
    if mask01 is None:
        return loss
    m = _as_1chw(mask01, depth)
    if m.shape[-2:] != depth.shape[-2:]:
        m = F.interpolate(m, size=depth.shape[-2:], mode="nearest")
    mx = m[:, :, :, 1:]
    my = m[:, :, 1:, :]
    num = (depth_dx * wx * mx).sum() + (depth_dy * wy * my).sum()
    den = mx.sum() + my.sum()
    return num / den.clamp_min(1e-8)


@torch.no_grad()
def split_l1_fg_bg(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    fg_mask01: torch.Tensor,
    valid_mask01: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Return l1_fg, l1_bg, cover_fg, cover_bg, cover_valid, cover_mask
    cover_mask = cover_valid if valid provided else 1.0
    """
    B, C, H, W = pred.shape
    fg = _as_1chw(fg_mask01, pred)
    fg = (fg > 0.5).float()

    if valid_mask01 is None:
        valid = torch.ones((B, 1, H, W), device=pred.device, dtype=pred.dtype)
    else:
        valid = _as_1chw(valid_mask01, pred)
        valid = (valid > 0.5).float()

    # effective fg/bg under valid
    eff_fg = fg * valid
    eff_bg = (1.0 - fg) * valid

    l1_fg = masked_l1(pred, tgt, eff_fg, eps=eps).item(
    ) if eff_fg.sum() > 0 else float("nan")
    l1_bg = masked_l1(pred, tgt, eff_bg, eps=eps).item(
    ) if eff_bg.sum() > 0 else float("nan")

    cover_valid = float(valid.mean().item())
    cover_fg = float((eff_fg.mean()).item())
    cover_bg = float((eff_bg.mean()).item())
    cover_mask = cover_valid  # naming consistent with your logs

    return dict(
        l1_fg=l1_fg,
        l1_bg=l1_bg,
        cover_valid=cover_valid,
        cover_fg=cover_fg,
        cover_bg=cover_bg,
        cover_mask=cover_mask,
    )


def view_decoder_total_loss(
    pred_rgb: torch.Tensor,
    tgt_rgb: torch.Tensor,
    valid_mask01: torch.Tensor,
    fg_mask01: torch.Tensor,
    fg_weight: float = 5.0,
    global_weight: float = 0.05,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    你的三件事都在这里：
    1) masked photometric (valid 内)
    2) fg upweight (fg 小时强迫模型学前景)
    3) global weak loss (防止“除 mask 外随便输出”)

    Returns dict:
      total, loss_valid, loss_fg, loss_global, stats(可在外部用 split_l1_fg_bg 打印)
    """
    B, C, H, W = pred_rgb.shape
    valid = valid_mask01
    fg = fg_mask01
    if valid is None:
        valid = torch.ones(
            (B, 1, H, W), device=pred_rgb.device, dtype=pred_rgb.dtype)
    else:
        if valid.dim() == 3:
            valid = valid[:, None, ...]
        valid = (valid > 0.5).float()

    if fg is None:
        fg = torch.zeros((B, 1, H, W), device=pred_rgb.device,
                         dtype=pred_rgb.dtype)
    else:
        if fg.dim() == 3:
            fg = fg[:, None, ...]
        fg = (fg > 0.5).float()

    eff_fg = fg * valid
    eff_valid = valid

    # base valid loss
    loss_valid = masked_l1(pred_rgb, tgt_rgb, eff_valid, eps=eps)

    # foreground loss (weighted)
    if eff_fg.sum() > 0:
        loss_fg = masked_l1(pred_rgb, tgt_rgb, eff_fg, eps=eps)
    else:
        loss_fg = torch.zeros([], device=pred_rgb.device, dtype=pred_rgb.dtype)

    # weak global loss to stabilize output everywhere
    loss_global = F.l1_loss(pred_rgb, tgt_rgb)

    total = loss_valid + fg_weight * loss_fg + global_weight * loss_global

    return dict(
        total=total,
        loss_valid=loss_valid,
        loss_fg=loss_fg,
        loss_global=loss_global,
    )
