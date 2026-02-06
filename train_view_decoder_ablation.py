from __future__ import annotations
import os
import sys
import time
import math
import copy
import json
import argparse
import configparser
import re
import subprocess
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from pathlib import Path

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    from scipy import ndimage as _scipy_ndimage  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _scipy_ndimage = None
    _HAS_SCIPY = False

from view_decoder_ablation import GeomViewDecoderAblation
from zju_dataset_view import ZJUViewSynthDataset
from mask_ops import mask_stats as _mask_stats
from view_decoder_losses import (
    masked_charbonnier,
    masked_huber,
    edge_aware_depth_smoothness,
)

# Global args holder for optional access in helpers.
args = None

# train_view_decoder_ablation_v3.py
# -*- coding: utf-8 -*-
"""
ZJU ViewSynth - View Decoder Ablation Trainer v3

在 v2 的基础上新增/修复：
- ? 自动把三合一 cat 图切成 3 张：*_p0/_p1/_p2（方便你逐栏看）
- ? debug/val 时对 conf/weight/mask 做分位数统计（写 ini），定位“conf 塌缩/权重发黑”等问题
- ? 修复 pred_conf clamp 截断梯度：自动判断 logits -> sigmoid，否则再 clamp
- ? conf supervision 的归一化可以和 mask 归一化解耦（防止 quantile 每 batch 抖导致 conf 学歪）

v3.1 (本次修复)：
- ? FIX: fg/valid/train 等“二值掩码”在阈值比较前自动识别 0~255 并归一到 0~1
      避免 uint8 掩码直接 >0.5 导致 mask 全白（你现在遇到的核心问题）

运行：
  python train_view_decoder_ablation_v3.py

常用：
  python train_view_decoder_ablation_v3.py --seq_names CoreView_390 --zju_root F:\datasets\ZJU_MoCap\data\zju_mocap
  python train_view_decoder_ablation_v3.py --resume ckpt\viewdec_ablation_last.pth
"""

# --- make local imports stable (Windows + run-from-anywhere) ---
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# ---------------------------
# INI logger
# ---------------------------
class IniLogger:
    def __init__(self, path: str):
        self.path = path
        self.cfg = configparser.ConfigParser()
        if os.path.isfile(path):
            try:
                self.cfg.read(path, encoding="utf-8")
            except Exception:
                self.cfg.read(path)

    def log(self, section: str, kv: dict):
        if section not in self.cfg:
            self.cfg[section] = {}
        for k, v in kv.items():
            self.cfg[section][str(k)] = str(v)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            self.cfg.write(f)


# ---------------------------
# Torch / cuDNN stability
# ---------------------------
def setup_torch_stability(tf32: bool = True, cudnn_benchmark: bool = True):
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32 else "ieee"
            torch.backends.cudnn.conv.fp32_precision = "tf32" if tf32 else "ieee"
        except Exception:
            try:
                torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
                torch.backends.cudnn.allow_tf32 = bool(tf32)
            except Exception:
                pass

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# AMP helpers (兼容新旧 API)
# ---------------------------
def make_grad_scaler(device: str, enabled: bool):
    if (not enabled) or device != "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def autocast_ctx(device: str, enabled: bool):
    if device != "cuda" or (not enabled):
        return _NullCtx()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)

def resolve_conf_bias_init(raw: float) -> Optional[float]:
    try:
        val = float(raw)
    except Exception:
        return None
    if math.isnan(val) or val == -1.0:
        return None
    return val


# ---------------------------
# Utils
# ---------------------------
def save_png(arr_u8_hwc: np.ndarray, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    Image.fromarray(arr_u8_hwc).save(path)


def _parse_only_steps_env(env_key: str = "ONLY_STEPS") -> set:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return set()
    parts = re.split(r"[,\s;/]+", raw)
    out = set()
    for p in parts:
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _parse_int_list(raw: str) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = []
        for v in raw:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\s;/]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def _parse_str_list(raw: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw if str(v).strip()]
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\s;/]+", s)
    return [p for p in parts if p]


def _as_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _ensure_4d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def apply_valid_to_depth_conf(
    src_depth_conf: torch.Tensor,
    src_depth: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Keep source depth confidence only on valid source depth pixels.
    valid := finite(depth) and depth > 0
    """
    if not torch.is_tensor(src_depth_conf):
        return src_depth_conf

    conf_in = src_depth_conf
    conf = src_depth_conf.float()
    valid = torch.isfinite(conf)

    if src_depth is not None and torch.is_tensor(src_depth):
        depth = src_depth.to(device=conf.device).float()
        depth_valid = torch.isfinite(depth) & (depth > 0)

        # Align rank for common layouts: (B,V,H,W) / (B,V,1,H,W).
        while depth_valid.dim() < conf.dim():
            depth_valid = depth_valid.unsqueeze(-3)

        # Channel fallback: reduce depth channel if conf has singleton channel.
        if depth_valid.dim() == conf.dim() and depth_valid.shape[-3] != conf.shape[-3]:
            if conf.shape[-3] == 1 and depth_valid.shape[-3] > 1:
                depth_valid = depth_valid.any(dim=-3, keepdim=True)

        try:
            depth_valid = depth_valid.expand_as(conf)
            valid = valid & depth_valid
        except Exception:
            # Keep finite-conf behavior when shapes are not broadcastable.
            pass

    out = torch.where(valid, conf, torch.zeros_like(conf))
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(dtype=conf_in.dtype)


# ---------------------------
# FIX: robust [0,1] for binary-like masks
# ---------------------------
def to_01_mask(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    仅用于“掩码类”张量（valid/fg/train/alpha/seg等）：
    - 如果看起来是 uint8 0~255，则自动 /255
    - clamp 到 [0,1]
    注意：不要用它处理深度conf那种可能在 1~8 的连续量。
    """
    m = _ensure_4d(mask)
    if m is None:
        return None
    m = m.float()
    # 关键：mask 若来自 PNG/uint8，通常 max=255。
    # 用 >8.0 作为阈值，避免把 1~8 的 conf 误判为 255 范围。
    mx = float(m.detach().amax().cpu())
    mn = float(m.detach().amin().cpu())
    if mx > 8.0 and mn >= -eps:
        m = m / 255.0
    return m.clamp(0.0, 1.0)


def resize_mask_nearest(mask: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    mask = _ensure_4d(mask)
    if mask is None:
        return None
    if mask.shape[-2:] == tuple(out_hw):
        return mask
    return F.interpolate(mask.float(), size=out_hw, mode="nearest")


def binarize(mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    mask = _ensure_4d(mask)
    if mask is None:
        return None
    # FIX: 0~255 -> 0~1，再阈值
    mask01 = to_01_mask(mask)
    return (mask01 > float(thr)).float()


def dilate_mask(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    mask = _ensure_4d(mask)
    if mask is None:
        return None
    # FIX: 0~255 -> 0~1
    mask01 = to_01_mask(mask).float()
    if k <= 1:
        return (mask01 > 0.5).float()
    pad = k // 2
    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)
    return (dil > 0.5).float()


def _largest_cc_numpy(mask_u8: np.ndarray, min_pixels: int = 16) -> np.ndarray:
    if mask_u8.sum() < min_pixels:
        return mask_u8
    if _HAS_CV2 and cv2 is not None:
        try:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
            if num <= 1:
                return mask_u8
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = int(np.argmax(areas)) + 1
            if areas[max_idx - 1] < min_pixels:
                return mask_u8
            return (labels == max_idx).astype(np.uint8)
        except Exception:
            pass
    if _HAS_SCIPY and _scipy_ndimage is not None:
        try:
            labels, num = _scipy_ndimage.label(mask_u8)
            if num <= 1:
                return mask_u8
            counts = np.bincount(labels.reshape(-1))
            if counts.size <= 1:
                return mask_u8
            counts[0] = 0
            max_label = int(np.argmax(counts))
            if counts[max_label] < min_pixels:
                return mask_u8
            return (labels == max_label).astype(np.uint8)
        except Exception:
            pass

    H, W = mask_u8.shape
    labels = np.zeros((H, W), dtype=np.int32)
    max_area = 0
    max_label = 0
    label = 0
    for y in range(H):
        for x in range(W):
            if mask_u8[y, x] == 0 or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            area = 1
            while stack:
                cy, cx = stack.pop()
                ny = cy - 1
                if ny >= 0 and mask_u8[ny, cx] and labels[ny, cx] == 0:
                    labels[ny, cx] = label
                    stack.append((ny, cx))
                    area += 1
                ny = cy + 1
                if ny < H and mask_u8[ny, cx] and labels[ny, cx] == 0:
                    labels[ny, cx] = label
                    stack.append((ny, cx))
                    area += 1
                nx = cx - 1
                if nx >= 0 and mask_u8[cy, nx] and labels[cy, nx] == 0:
                    labels[cy, nx] = label
                    stack.append((cy, nx))
                    area += 1
                nx = cx + 1
                if nx < W and mask_u8[cy, nx] and labels[cy, nx] == 0:
                    labels[cy, nx] = label
                    stack.append((cy, nx))
                    area += 1
            if area > max_area:
                max_area = area
                max_label = label
    if max_area < min_pixels or max_label == 0:
        return mask_u8
    return (labels == max_label).astype(np.uint8)


def keep_largest_connected_component(mask: torch.Tensor, min_pixels: int = 16) -> torch.Tensor:
    m = _ensure_4d(mask)
    if m is None:
        return None
    m = (m > 0.5).float()
    B, _, H, W = m.shape
    out = torch.zeros_like(m)
    for b in range(B):
        mb = m[b, 0].detach().cpu().numpy().astype(np.uint8)
        out_np = _largest_cc_numpy(mb, min_pixels=int(min_pixels))
        out[b, 0] = torch.from_numpy(out_np).to(device=m.device, dtype=m.dtype)
    return out


def ensure_min_cover_by_dilation(mask_bin: torch.Tensor, min_cover: float, k0: int = 7, k_max: int = 31) -> torch.Tensor:
    mask_bin = _ensure_4d(mask_bin)
    if mask_bin is None:
        return None
    # FIX: 保证输入是 {0,1} 掩码
    mask_bin = binarize(mask_bin, 0.5).float()
    if min_cover is None or float(min_cover) <= 0:
        return mask_bin
    cover = mask_bin.mean(dim=(1, 2, 3))
    if torch.all(cover >= float(min_cover)):
        return mask_bin
    k = int(k0)
    while True:
        out = dilate_mask(mask_bin, k=k)
        cover2 = out.mean(dim=(1, 2, 3))
        if torch.all(cover2 >= float(min_cover)) or k >= int(k_max):
            return out
        k = min(int(k_max), k * 2 + 1)


def _safe_quantile(x: torch.Tensor, q: float) -> Optional[torch.Tensor]:
    if x is None or x.numel() == 0:
        return None
    qv = float(q)
    if qv <= 0.0:
        return x.min()
    if qv >= 1.0:
        return x.max()
    try:
        return torch.quantile(x, qv)
    except Exception:
        flat = x.reshape(-1)
        k = int(round(qv * (flat.numel() - 1)))
        k = max(0, min(flat.numel() - 1, k))
        return flat.kthvalue(k + 1).values


def drop_ground_from_fg(
    fg_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    pointmap: Optional[torch.Tensor],
    out_hw: Tuple[int, int],
    axis: int = 1,
    q: float = 0.05,
    margin: float = 0.02,
    min_points: int = 64,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Remove ground-like pixels from fg_mask using pointmap height quantile.
    Assumes pointmap is (B,3,H,W) or (3,H,W) with a stable vertical axis.
    """
    info = {"applied": False, "floor_vals": []}
    if fg_mask is None or pointmap is None:
        return fg_mask, info

    pm = pointmap
    if not torch.is_tensor(pm):
        return fg_mask, info
    if pm.dim() == 3:
        if pm.shape[0] == 3:
            pm = pm.unsqueeze(0)
        elif pm.shape[-1] == 3:
            pm = pm.permute(2, 0, 1).unsqueeze(0)
        else:
            return fg_mask, info
    elif pm.dim() == 4:
        if pm.shape[1] != 3 and pm.shape[-1] == 3:
            pm = pm.permute(0, 3, 1, 2)
        elif pm.shape[1] != 3:
            return fg_mask, info
    else:
        return fg_mask, info

    ax = int(axis)
    if ax < 0:
        ax = 3 + ax
    if ax < 0 or ax > 2:
        return fg_mask, info

    if pm.shape[-2:] != tuple(out_hw):
        pm = F.interpolate(pm.float(), size=out_hw, mode="bilinear", align_corners=False)
    y = pm[:, ax:ax + 1, :, :]  # (B,1,H,W)

    fg_bin = (fg_mask > 0.5)
    if valid_mask is not None:
        fg_bin = fg_bin & (valid_mask > 0.5)

    out = fg_mask.clone()
    B = int(out.shape[0])
    for b in range(B):
        mb = fg_bin[b, 0]
        if int(mb.sum().item()) < int(min_points):
            info["floor_vals"].append(float("nan"))
            continue
        yb = y[b, 0][mb]
        floor = _safe_quantile(yb, q)
        if floor is None:
            info["floor_vals"].append(float("nan"))
            continue
        floor_v = float(floor.item())
        info["floor_vals"].append(floor_v)
        keep = (y[b, 0] > (floor + float(margin))).float()
        out[b, 0] = out[b, 0] * keep

    info["applied"] = True
    return out, info


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], eps=1e-6) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, tgt, reduction="mean")
    diff = (pred - tgt).abs() * mask
    C = float(pred.shape[1])
    denom = (mask.sum(dim=(2, 3)) * C).clamp_min(eps)  # (B,1)
    num = diff.sum(dim=(1, 2, 3), keepdim=False)       # (B,)
    return (num / denom.squeeze(1)).mean()


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], eps=1e-6) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, tgt, reduction="mean")
    diff2 = (pred - tgt).pow(2) * mask
    C = float(pred.shape[1])
    denom = (mask.sum(dim=(2, 3)) * C).clamp_min(eps)  # (B,1)
    num = diff2.sum(dim=(1, 2, 3), keepdim=False)      # (B,)
    return (num / denom.squeeze(1)).mean()


def masked_bce(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], eps=1e-6) -> torch.Tensor:
    pred = pred.clamp(min=eps, max=1.0 - eps)
    loss_map = F.binary_cross_entropy(pred, tgt, reduction="none")
    if mask is None:
        return loss_map.mean()
    denom = mask.sum().clamp_min(eps)
    return (loss_map * mask).sum() / denom


def compute_photo_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss_type: str = "charbonnier",
    huber_delta: float = 0.01,
    charb_eps: float = 1e-3,
    charb_alpha: float = 0.5,
) -> torch.Tensor:
    lt = str(loss_type).lower()
    if lt == "huber":
        return masked_huber(pred, tgt, mask, delta=float(huber_delta))
    if lt == "charbonnier":
        return masked_charbonnier(
            pred, tgt, mask, eps=float(charb_eps), alpha=float(charb_alpha)
        )
    return masked_l1(pred, tgt, mask)


def compute_depth_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss_type: str = "charbonnier",
    huber_delta: float = 0.01,
    charb_eps: float = 1e-3,
    charb_alpha: float = 0.5,
) -> torch.Tensor:
    return compute_photo_loss(
        pred, tgt, mask,
        loss_type=loss_type,
        huber_delta=huber_delta,
        charb_eps=charb_eps,
        charb_alpha=charb_alpha,
    )


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor, eps=1e-6):
    B = x.shape[0]
    denom = (mask.sum(dim=(2, 3)) * 3.0).clamp_min(eps)
    mean = (x * mask).sum(dim=(1, 2, 3)) / denom.squeeze(1)
    mean_b = mean.view(B, 1, 1, 1)
    var = ((x - mean_b) ** 2 * mask).sum(dim=(1, 2, 3)) / denom.squeeze(1)
    std = torch.sqrt(var.clamp_min(eps))
    return mean, std


def _to_4d_mask(m: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    if m is None:
        return None
    if m.dim() == 0:
        m = m.view(1, 1, 1, 1)
    elif m.dim() == 1:
        m = m.view(-1, 1, 1, 1)
    elif m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    if m.dim() == 3:
        m = m[:, None, :, :]
    if m.dim() == 4 and m.size(1) == 1 and like.dim() == 4 and like.size(1) > 1:
        m = m.expand(-1, like.size(1), -1, -1)
    return m


@torch.no_grad()
def compute_l1_splits(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    fg_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> dict:
    assert pred.shape == tgt.shape
    abs_err = (pred - tgt).abs()
    out = {"l1_full": abs_err.mean().item()}
    if valid_mask is not None:
        vm = _to_4d_mask(valid_mask, abs_err).float()
        denom = vm.mean().clamp_min(eps)
        out.update({"l1_valid": ((abs_err * vm).mean() / denom).item(),
                    "cover_valid": denom.item()})
    if fg_mask is not None:
        fm = _to_4d_mask(fg_mask, abs_err).float().clamp(0, 1)
        if valid_mask is not None:
            vm = _to_4d_mask(valid_mask, abs_err).float()
            fm_eff = fm * vm
            bg_eff = (1.0 - fm) * vm
        else:
            fm_eff = fm
            bg_eff = (1.0 - fm)
        fg_denom = fm_eff.mean().clamp_min(eps)
        bg_denom = bg_eff.mean().clamp_min(eps)
        out.update(
            {
                "l1_fg": ((abs_err * fm_eff).mean() / fg_denom).item(),
                "l1_bg": ((abs_err * bg_eff).mean() / bg_denom).item(),
                "cover_fg": fg_denom.item(),
                "cover_bg": bg_denom.item(),
            }
        )
    return out


# ---------------------------
# robust visualization: for 1ch maps auto normalize
# ---------------------------
def _robust_norm01(x: torch.Tensor, qlo=0.01, qhi=0.99, eps=1e-6):
    x = x.float()
    flat = x.flatten()
    if flat.numel() < 16:
        mn, mx = flat.min(), flat.max()
        return (x - mn) / (mx - mn + eps)
    try:
        lo = torch.quantile(flat, torch.tensor(qlo, device=x.device))
        hi = torch.quantile(flat, torch.tensor(qhi, device=x.device))
        return (x - lo) / (hi - lo + eps)
    except Exception:
        mn, mx = flat.min(), flat.max()
        return (x - mn) / (mx - mn + eps)


def to_u8_img(x, *, name=""):
    x = _as_torch(x).detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            f"{name} unexpected ndim={x.ndim}, shape={tuple(x.shape)}")

    # CHW
    if x.shape[0] in (1, 3):
        chw = x
        mn, mx = chw.min().item(), chw.max().item()
        if chw.shape[0] == 3:
            if mn < -0.05 and mx <= 1.05:
                chw = (chw + 1.0) * 0.5
            chw = chw.clamp(0, 1)
        else:
            if mn < -0.05 or mx > 1.05:
                chw = _robust_norm01(chw, 0.01, 0.99).clamp(0, 1)
            else:
                chw = chw.clamp(0, 1)
        hwc = chw.permute(1, 2, 0).contiguous()

    # HWC
    elif x.shape[2] in (1, 3):
        hwc = x
        mn, mx = hwc.min().item(), hwc.max().item()
        if hwc.shape[2] == 3:
            if mn < -0.05 and mx <= 1.05:
                hwc = (hwc + 1.0) * 0.5
            hwc = hwc.clamp(0, 1)
        else:
            if mn < -0.05 or mx > 1.05:
                hwc = _robust_norm01(hwc, 0.01, 0.99).clamp(0, 1)
            else:
                hwc = hwc.clamp(0, 1)
    else:
        raise ValueError(
            f"{name} cannot infer CHW/HWC from shape={tuple(x.shape)}")

    arr = (hwc * 255.0).round().to(torch.uint8).numpy()
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def to_u8_img_fixed(x, vmin: float, vmax: float, *, name: str = ""):
    """
    Fixed-range visualization for 1ch (or 3ch) maps.
    vmin/vmax define linear mapping to [0,1].
    """
    x = _as_torch(x).detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            f"{name} unexpected ndim={x.ndim}, shape={tuple(x.shape)}")

    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        return to_u8_img(x, name=name)

    # CHW
    if x.shape[0] in (1, 3):
        chw = x
        chw = (chw - vmin) / (vmax - vmin + 1e-8)
        chw = chw.clamp(0, 1)
        hwc = chw.permute(1, 2, 0).contiguous()
    # HWC
    elif x.shape[2] in (1, 3):
        hwc = x
        hwc = (hwc - vmin) / (vmax - vmin + 1e-8)
        hwc = hwc.clamp(0, 1)
    else:
        raise ValueError(
            f"{name} cannot infer CHW/HWC from shape={tuple(x.shape)}")

    arr = (hwc * 255.0).round().to(torch.uint8).numpy()
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _jet_colormap(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    r = torch.clamp(1.5 - (4.0 * x - 3.0).abs(), 0.0, 1.0)
    g = torch.clamp(1.5 - (4.0 * x - 2.0).abs(), 0.0, 1.0)
    b = torch.clamp(1.5 - (4.0 * x - 1.0).abs(), 0.0, 1.0)
    return torch.stack([r, g, b], dim=-1)


def to_u8_heatmap(x, *, name: str = ""):
    x = _as_torch(x).detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[-1] == 1:
            x = x[:, :, 0]
        elif x.shape[0] == 3:
            x = x.mean(dim=0)
        elif x.shape[-1] == 3:
            x = x.mean(dim=2)
        else:
            raise ValueError(
                f"{name} cannot infer 2D map from shape={tuple(x.shape)}")
    if x.ndim != 2:
        raise ValueError(
            f"{name} unexpected ndim={x.ndim}, shape={tuple(x.shape)}")
    x01 = _robust_norm01(x, 0.01, 0.99).clamp(0, 1)
    rgb = _jet_colormap(x01)
    arr = (rgb * 255.0).round().to(torch.uint8).numpy()
    return arr


# ---------------------------
# cat panels splitter (new)
# ---------------------------
def split_three_panels_u8(cat_u8_hwc: np.ndarray):
    """把横向拼接的三合一图切成 3 张（尽量均匀切）"""
    H, W, C = cat_u8_hwc.shape
    b1 = W // 3
    b2 = (2 * W) // 3
    p0 = cat_u8_hwc[:, :b1, :]
    p1 = cat_u8_hwc[:, b1:b2, :]
    p2 = cat_u8_hwc[:, b2:, :]
    return p0, p1, p2


def save_cat_panels(cat_u8_hwc: np.ndarray, cat_path: str):
    base, ext = os.path.splitext(cat_path)
    p0, p1, p2 = split_three_panels_u8(cat_u8_hwc)
    save_png(p0, f"{base}_p0{ext}")
    save_png(p1, f"{base}_p1{ext}")
    save_png(p2, f"{base}_p2{ext}")


# ---------------------------
# tensor stats for ini (new)
# ---------------------------
@torch.no_grad()
def tensor_stats_1d(vals: torch.Tensor, prefix: str = "") -> dict:
    vals = vals.float().flatten()
    if vals.numel() == 0:
        return {f"{prefix}n": 0}
    out = {
        f"{prefix}n": int(vals.numel()),
        f"{prefix}min": float(vals.min().item()),
        f"{prefix}mean": float(vals.mean().item()),
        f"{prefix}max": float(vals.max().item()),
    }
    try:
        qs = torch.quantile(vals, torch.tensor(
            [0.01, 0.05, 0.50, 0.95, 0.99], device=vals.device))
        out.update({
            f"{prefix}q01": float(qs[0].item()),
            f"{prefix}q05": float(qs[1].item()),
            f"{prefix}q50": float(qs[2].item()),
            f"{prefix}q95": float(qs[3].item()),
            f"{prefix}q99": float(qs[4].item()),
        })
    except Exception:
        pass
    out.update({
        f"{prefix}gt005": float((vals > 0.05).float().mean().item()),
        f"{prefix}gt01":  float((vals > 0.10).float().mean().item()),
        f"{prefix}gt02":  float((vals > 0.20).float().mean().item()),
        f"{prefix}gt05":  float((vals > 0.50).float().mean().item()),
    })
    return out


@torch.no_grad()
def tensor_stats_map(x: torch.Tensor, mask: Optional[torch.Tensor] = None, prefix: str = "") -> dict:
    """x: (B,1,H,W) or (B,H,W) or any -> 取全部像素/或 mask>0.5 的像素统计"""
    if x is None:
        return {f"{prefix}n": 0}
    t = x.detach()
    if t.dim() == 4 and t.size(1) > 1:
        t = t.mean(dim=1, keepdim=True)
    if t.dim() == 3:
        t = t[:, None, :, :]
    if t.dim() != 4:
        t = t.view(-1)

    if mask is not None:
        m = mask.detach().float()
        if m.dim() == 3:
            m = m[:, None, :, :]
        if m.shape[-2:] != t.shape[-2:]:
            m = F.interpolate(m, size=t.shape[-2:], mode="nearest")
        vals = t[m > 0.5]
    else:
        vals = t.flatten()
    return tensor_stats_1d(vals, prefix=prefix)


# ---------------------------
# conf normalization (supports quantile)
# ---------------------------
@torch.no_grad()
def normalize_conf_to_01(
    conf: torch.Tensor,
    raw_min: float = 1.0,
    raw_max: float = 8.0,
    auto: bool = True,
    use_quantile: bool = True,
    qlo: float = 0.05,
    qhi: float = 0.95,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_dim = conf.dim()
    c = _ensure_4d(conf).float()
    mn = float(c.min().item())
    mx = float(c.max().item())

    # FIX: 如果 conf 来自 uint8 0~255（或类似），先缩放到 0~1 再走后续逻辑
    # 用 >16 判定，避免把 1~8 的深度置信误判
    if auto and mx > 16.0 and mn >= -0.2:
        c = (c / 255.0).clamp(0.0, 1.0)
        mn = float(c.min().item())
        mx = float(c.max().item())

    vm = None
    if valid_mask is not None:
        vm = _ensure_4d(valid_mask).float()
        if vm.shape[-2:] != c.shape[-2:]:
            vm = F.interpolate(vm, size=c.shape[-2:], mode="nearest")

    def _restore(out_t: torch.Tensor) -> torch.Tensor:
        if orig_dim == 3:
            return out_t[:, 0, ...]
        if orig_dim == 2:
            return out_t[0, 0, ...]
        return out_t

    already01 = (auto and mx <= 1.2 and mn >= -0.2)
    if already01 and (not use_quantile):
        out = c.clamp(0, 1)
        if vm is not None:
            out = out * vm
        return _restore(out), {
            "mode": "already01",
            "conf_raw_min": 0.0,
            "conf_raw_max": 1.0,
            "mn": mn,
            "mx": mx,
        }

    if use_quantile:
        try:
            out = torch.zeros_like(c)
            lo_vals = []
            hi_vals = []
            for b in range(c.shape[0]):
                cb = c[b].view(-1)
                if vm is not None:
                    vb = vm[b].view(-1) > 0.5
                    cb = cb[vb]
                if cb.numel() < 32:
                    lo_vals.append(float("nan"))
                    hi_vals.append(float("nan"))
                    continue
                lo = torch.quantile(cb, float(qlo))
                hi = torch.quantile(cb, float(qhi))
                denom = (hi - lo).clamp_min(eps)
                out_b = (c[b] - lo) / denom
                out_b = out_b.clamp(0, 1)
                if vm is not None:
                    out_b = out_b * (vm[b] > 0.5).float()
                out[b] = out_b
                lo_vals.append(float(lo.item()))
                hi_vals.append(float(hi.item()))

            lo_ok = [v for v in lo_vals if math.isfinite(v)]
            hi_ok = [v for v in hi_vals if math.isfinite(v)]
            lo_f = float(sum(lo_ok) / len(lo_ok)) if lo_ok else None
            hi_f = float(sum(hi_ok) / len(hi_ok)) if hi_ok else None
            return _restore(out), {
                "mode": "quantile_valid",
                "qlo": qlo,
                "qhi": qhi,
                "lo": lo_f,
                "hi": hi_f,
                "mn": mn,
                "mx": mx,
            }
        except Exception:
            pass

    denom = float(raw_max - raw_min)
    denom = denom if abs(denom) > eps else 1.0
    out = ((c - float(raw_min)) / denom).clamp(0, 1)
    if vm is not None:
        out = out * vm
    return _restore(out), {"mode": "fixed", "raw_min": raw_min, "raw_max": raw_max, "mn": mn, "mx": mx}


def make_soft_mask_from_conf(
    conf: torch.Tensor,
    out_hw: Tuple[int, int],
    thr=0.2,
    temp=0.06,
    conf_raw_min: float = 1.0,
    conf_raw_max: float = 8.0,
    conf_auto_norm: bool = True,
    conf_use_quantile: bool = True,
    conf_qlo: float = 0.05,
    conf_qhi: float = 0.95,
    valid_mask: Optional[torch.Tensor] = None,
):
    conf_up = F.interpolate(conf.float(), size=out_hw,
                            mode="bilinear", align_corners=False)
    conf01, info = normalize_conf_to_01(
        conf_up,
        raw_min=conf_raw_min,
        raw_max=conf_raw_max,
        auto=conf_auto_norm,
        use_quantile=conf_use_quantile,
        qlo=conf_qlo,
        qhi=conf_qhi,
        valid_mask=valid_mask,
    )
    soft = torch.sigmoid((conf01 - float(thr)) / float(temp))
    return soft, conf01, info


# ---------------------------
# pred_conf normalization (new, avoids clamp killing gradients)
# ---------------------------
def normalize_pred_conf(pred_conf: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    如果 pred_conf 看起来像 logits（范围明显超出[0,1]），就用 sigmoid。
    否则按概率图处理，轻微 clamp。
    """
    pc = pred_conf.float()
    mn = float(pc.min().item())
    mx = float(pc.max().item())
    if mn < -0.05 or mx > 1.05:
        return torch.sigmoid(pc), "sigmoid_logits"
    eps = 1e-4
    return pc.clamp(eps, 1.0 - eps), "clamp_eps"


# ---------------------------
# Batch key picking helpers
# ---------------------------
def pick_first_tensor(batch: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in batch and batch[k] is not None:
            return k, batch[k]
    return None, None


def _assert_tensor_ok(t: Any, *, name: str, where: str, ndims: Tuple[int, ...]):
    if not torch.is_tensor(t):
        raise TypeError(f"[batch-assert] {where}: {name} is not a tensor")
    if t.ndim not in ndims:
        raise ValueError(
            f"[batch-assert] {where}: {name} ndim={t.ndim} not in {ndims}")
    if t.numel() == 0 or any(int(d) <= 0 for d in t.shape):
        raise ValueError(
            f"[batch-assert] {where}: {name} has empty shape {tuple(t.shape)}")
    if t.shape[-1] <= 0 or t.shape[-2] <= 0:
        raise ValueError(
            f"[batch-assert] {where}: {name} has invalid H/W {tuple(t.shape[-2:])}")


def _require_any_tensor(batch: Dict[str, Any], keys: List[str], where: str):
    key, t = pick_first_tensor(batch, keys)
    if t is None:
        raise KeyError(
            f"[batch-assert] {where}: missing required key (any of {keys})")
    return key, t


def assert_batch_shapes(batch: Dict[str, Any], where: str):
    if not isinstance(batch, dict):
        raise TypeError(f"[batch-assert] {where}: batch is not a dict")

    # src
    _assert_tensor_ok(batch.get("src_imgs", None),
                      name="src_imgs", where=where, ndims=(5,))
    _assert_tensor_ok(batch.get("src_depth", None),
                      name="src_depth", where=where, ndims=(5,))
    _assert_tensor_ok(batch.get("src_depth_conf", None),
                      name="src_depth_conf", where=where, ndims=(5,))
    _assert_tensor_ok(batch.get("src_pointmap", None),
                      name="src_pointmap", where=where, ndims=(5,))

    # tgt core
    _assert_tensor_ok(batch.get("tgt_img", None),
                      name="tgt_img", where=where, ndims=(4,))

    _, tgt_depth = _require_any_tensor(
        batch, ["tgt_depth", "depth_tgt", "tgt_depth_map", "depth"], where)
    _assert_tensor_ok(tgt_depth, name="tgt_depth", where=where, ndims=(3, 4))

    _, tgt_depth_conf = _require_any_tensor(
        batch, ["tgt_depth_conf", "depth_conf_tgt",
                "tgt_conf_depth", "tgt_depthconf", "tgt_conf", "tgt_mask_conf", "conf_tgt"],
        where
    )
    _assert_tensor_ok(
        tgt_depth_conf, name="tgt_depth_conf", where=where, ndims=(3, 4))

    _, tgt_fg = _require_any_tensor(
        batch, ["tgt_fg", "fg_mask", "tgt_fg_mask", "tgt_mask", "tgt_silhouette", "silhouette",
                "human_mask", "person_mask", "tgt_alpha", "alpha", "seg", "tgt_seg"],
        where
    )
    _assert_tensor_ok(tgt_fg, name="tgt_fg", where=where, ndims=(3, 4))

    if args is not None and bool(getattr(args, "fg_drop_ground", False)):
        _, tgt_pointmap = _require_any_tensor(
            batch, ["tgt_pointmap", "tgt_point_map", "pointmap_tgt", "tgt_points"], where)
        _assert_tensor_ok(
            tgt_pointmap, name="tgt_pointmap", where=where, ndims=(4,))


def _ensure_batch_dim(x: Any, want_dim: int):
    if not torch.is_tensor(x):
        return x
    if x.dim() >= want_dim:
        return x
    while x.dim() < want_dim:
        x = x.unsqueeze(0)
    return x


def ensure_batch_dims(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a single-sample dict has a batch dimension for debug/infer reuse.
    """
    if not isinstance(sample, dict):
        return sample
    out = dict(sample)
    dim_map = {
        "src_imgs": 5,
        "src_depth": 5,
        "src_depth_conf": 5,
        "src_pointmap": 5,
        "tgt_img": 4,
        "tgt_depth": 4,
        "tgt_depth_conf": 4,
        "tgt_conf": 4,
        "tgt_pointmap": 4,
        "tgt_fg": 4,
        "valid_mask": 4,
        "tgt_vid": 1,
        "src_vids": 2,
        "bad_tgt_masked": 1,
    }
    for k, d in dim_map.items():
        if k in out:
            out[k] = _ensure_batch_dim(out[k], d)
    return out


def _get_dataset_item(ds, index: int):
    try:
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            base_idx = ds.indices[int(index) % len(ds.indices)]
            return ds.dataset[base_idx]
    except Exception:
        pass
    return ds[int(index) % len(ds)]


# ---------------------------
# Masks builder (key-robust)
# ---------------------------
def build_masks_from_batch(
    batch: Dict[str, Any],
    pred_hw: Tuple[int, int],
    device: str,
    conf_thr: float = 0.2,
    conf_temp: float = 0.06,
    train_min_cover: float = 0.10,
    fg_thr: float = 0.5,
    fg_min_cover: float = 0.05,
    fg_dilate_k: int = 7,
    fg_keep_largest_cc: bool = True,
    fg_lcc_min_pixels: int = 32,
    fg_drop_ground: Optional[bool] = None,
    fg_ground_axis: Optional[int] = None,
    fg_ground_q: Optional[float] = None,
    fg_ground_margin: Optional[float] = None,
    fg_ground_min_points: Optional[int] = None,
    valid_min_cover: float = 0.10,
    valid_dilate_k: int = 7,
    valid_k_max: int = 31,
    bg_weight: float = 0.05,
    conf_raw_min: float = 1.0,
    conf_raw_max: float = 8.0,
    conf_auto_norm: bool = True,
    conf_use_quantile: bool = True,
    conf_qlo: float = 0.05,
    conf_qhi: float = 0.95,
    use_conf_in_train_mask: bool = True,
    train_mask_mode: Optional[str] = None,
    pred_conf_gate: Optional[torch.Tensor] = None,
    use_conf_gate: Optional[bool] = None,
    conf_gate_detach: Optional[bool] = None,
    conf_weight_detach: Optional[bool] = None,
    conf_gate_floor: Optional[float] = None,
    conf_gate_gamma: Optional[float] = None,
    conf_gate_strength: Optional[float] = None,
    conf_weight_strength: Optional[float] = None,
    conf_weight_min: Optional[float] = None,
    recon_gate_floor: Optional[float] = None,
    recon_mask_mode: Optional[str] = None,
    recon_weight_renorm: Optional[bool] = None,
    recon_weight_clip_max: Optional[float] = None,
):
    H, W = pred_hw

    def _resolve_opt(value, name: str, default):
        if value is not None:
            return value
        if args is not None and hasattr(args, name):
            return getattr(args, name)
        return default

    def _to_batch_bool(mask_like, B: int, device):
        if mask_like is None:
            return None
        if not torch.is_tensor(mask_like):
            m = torch.as_tensor(mask_like, device=device)
        else:
            m = mask_like.to(device)
        if m.dim() == 0:
            m = m.view(1)
        if m.dim() > 1:
            m = m.view(m.shape[0], -1)[:, 0]
        if m.numel() == 1 and B > 1:
            m = m.expand(B)
        if m.numel() < B:
            m = torch.cat([m, m.new_zeros(B - m.numel())], dim=0)
        elif m.numel() > B:
            m = m[:B]
        return m > 0.5

    # ---- depth conf ----
    depth_conf_key, tgt_depth_conf = pick_first_tensor(
        batch, ["tgt_depth_conf", "depth_conf_tgt",
                "tgt_conf_depth", "tgt_depthconf"]
    )
    if tgt_depth_conf is None:
        raise KeyError("batch missing tgt_depth_conf-like key.")
    tgt_depth_conf = tgt_depth_conf.to(device)
    tgt_depth_conf_raw = _ensure_4d(tgt_depth_conf).float()

    # ---- valid mask ----
    valid_key, valid_raw = pick_first_tensor(
        batch, ["valid_mask", "tgt_valid_mask", "tgt_mask_valid", "mask_valid",
                "tgt_depth_valid_mask", "tgt_depth_mask", "depth_valid_mask"]
    )

    # fallback: tgt_depth > 0
    tgt_depth_key, tgt_depth = pick_first_tensor(
        batch, ["tgt_depth", "depth_tgt", "tgt_depth_map", "depth"])
    if valid_raw is not None:
        # FIX: binarize 内部会自动把 0~255 归一到 0~1
        valid_mask = binarize(resize_mask_nearest(
            valid_raw.to(device), (H, W)), 0.5)
        source_valid_key = valid_key
    elif tgt_depth is not None:
        vm0 = (tgt_depth.to(device) > 0).float()
        valid_mask = binarize(resize_mask_nearest(vm0, (H, W)), 0.5)
        source_valid_key = f"__from_{tgt_depth_key}>0__"
    else:
        valid_mask = torch.ones(
            (tgt_depth_conf.shape[0], 1, H, W), device=device, dtype=torch.float32)
        source_valid_key = "__all_ones__"

    cover_valid0 = float(valid_mask.mean().item())
    if cover_valid0 < float(valid_min_cover):
        valid_mask = ensure_min_cover_by_dilation(valid_mask, float(
            valid_min_cover), int(valid_dilate_k), int(valid_k_max))

    bad_tgt_masked = None
    if batch is not None:
        bad_tgt_masked = batch.get("bad_tgt_masked", None)
    bad_tgt_masked = _to_batch_bool(
        bad_tgt_masked, valid_mask.shape[0], valid_mask.device)
    if bad_tgt_masked is not None and torch.any(bad_tgt_masked):
        valid_mask = valid_mask.clone()
        valid_mask[bad_tgt_masked] = 0

    # ---- apply valid to depth conf (only supervise on valid) ----
    tgt_depth_conf = tgt_depth_conf_raw
    if valid_mask is not None:
        vm_conf = resize_mask_nearest(valid_mask, tgt_depth_conf.shape[-2:])
        tgt_depth_conf = tgt_depth_conf * vm_conf

    # ---- fg mask ----
    fg_key, fg_raw = pick_first_tensor(
        batch, ["tgt_fg", "fg_mask", "tgt_fg_mask", "tgt_mask", "tgt_silhouette", "silhouette",
                "human_mask", "person_mask", "tgt_alpha", "alpha", "seg", "tgt_seg"]
    )

    tgt_conf_key, tgt_conf_raw = pick_first_tensor(
        batch, ["tgt_conf", "tgt_mask_conf", "conf_tgt"])
    if fg_raw is not None:
        fg_up = resize_mask_nearest(fg_raw.to(device), (H, W))
        fg_mask0 = binarize(fg_up, thr=float(fg_thr))  # FIX: 内部 to_01_mask
        source_fg_key = fg_key
    elif tgt_conf_raw is not None:
        conf_up = F.interpolate(
            tgt_conf_raw.to(device).float(), size=(H, W), mode="nearest"
        )
        conf_up = to_01_mask(conf_up)  # FIX: 0~255 -> 0~1，避免 clamp(0,1) 直接全 1
        fg_mask0 = (conf_up > float(fg_thr)).float()
        source_fg_key = tgt_conf_key
    else:
        fg_mask0 = valid_mask.clone()
        source_fg_key = "__fallback_valid__"

    # optional: remove ground using pointmap height quantile
    source_pointmap_key = None
    drop_ground = bool(_resolve_opt(fg_drop_ground, "fg_drop_ground", False))
    fg_ground_axis_v = int(_resolve_opt(fg_ground_axis, "fg_ground_axis", 1))
    fg_ground_q_v = float(_resolve_opt(fg_ground_q, "fg_ground_q", 0.05))
    fg_ground_margin_v = float(_resolve_opt(fg_ground_margin, "fg_ground_margin", 0.02))
    fg_ground_min_points_v = int(_resolve_opt(fg_ground_min_points, "fg_ground_min_points", 64))
    ground_info = None
    if drop_ground:
        pm_key, tgt_pointmap = pick_first_tensor(
            batch, ["tgt_pointmap", "tgt_point_map", "pointmap_tgt", "tgt_points"]
        )
        source_pointmap_key = pm_key
        if tgt_pointmap is not None:
            fg_mask0, ground_info = drop_ground_from_fg(
                fg_mask0,
                valid_mask,
                tgt_pointmap.to(device),
                out_hw=(H, W),
                axis=fg_ground_axis_v,
                q=fg_ground_q_v,
                margin=fg_ground_margin_v,
                min_points=fg_ground_min_points_v,
            )
        else:
            ground_info = {"applied": False, "floor_vals": []}

    if fg_keep_largest_cc:
        fg_mask0 = keep_largest_connected_component(
            fg_mask0, min_pixels=int(fg_lcc_min_pixels))

    fg_mask = ensure_min_cover_by_dilation(
        fg_mask0, float(fg_min_cover), int(fg_dilate_k), 31)
    if bad_tgt_masked is not None and torch.any(bad_tgt_masked):
        fg_mask = fg_mask.clone()
        fg_mask[bad_tgt_masked] = 0

    # --- conf gate (optional) ---

    use_conf_gate = bool(_resolve_opt(use_conf_gate, "use_conf_gate", False))
    conf_gate_detach = bool(_resolve_opt(
        conf_gate_detach, "conf_gate_detach", False))
    conf_weight_detach = bool(_resolve_opt(
        conf_weight_detach, "conf_weight_detach", conf_gate_detach))
    conf_gate_floor = float(_resolve_opt(
        conf_gate_floor, "conf_gate_floor", 0.0))
    conf_gate_gamma = float(_resolve_opt(
        conf_gate_gamma, "conf_gate_gamma", 1.0))
    conf_gate_strength = _resolve_opt(
        conf_gate_strength, "conf_gate_strength", None)
    if conf_gate_strength is None:
        conf_gate_strength = 1.0
    conf_gate_strength = float(conf_gate_strength)
    if conf_gate_strength < 0.0:
        conf_gate_strength = 0.0
    if conf_gate_strength > 1.0:
        conf_gate_strength = 1.0
    conf_weight_strength = _resolve_opt(
        conf_weight_strength, "conf_weight_strength", conf_gate_strength)
    conf_weight_strength = float(conf_weight_strength)
    if conf_weight_strength < 0.0:
        conf_weight_strength = 0.0
    if conf_weight_strength > 1.0:
        conf_weight_strength = 1.0
    conf_weight_min = _resolve_opt(conf_weight_min, "conf_weight_min", None)
    if conf_weight_min is not None:
        conf_weight_min = float(conf_weight_min)
        if conf_weight_min < 0.0:
            conf_weight_min = 0.0
        if conf_weight_min > 1.0:
            conf_weight_min = 1.0
    recon_gate_floor = _resolve_opt(
        recon_gate_floor, "recon_gate_floor", None)
    if recon_gate_floor is None:
        recon_gate_floor = conf_gate_floor
    recon_gate_floor = float(recon_gate_floor)
    if recon_gate_floor < 0.0:
        recon_gate_floor = 0.0
    if recon_gate_floor > 1.0:
        recon_gate_floor = 1.0
    recon_weight_renorm = bool(_resolve_opt(
        recon_weight_renorm, "recon_weight_renorm", False))
    recon_weight_clip_max = float(_resolve_opt(
        recon_weight_clip_max, "recon_weight_clip_max", 1.0))
    recon_mask_mode = str(_resolve_opt(
        recon_mask_mode, "recon_mask_mode", "fg")).lower()
    if recon_mask_mode not in ("fg", "train", "valid"):
        recon_mask_mode = "fg"

    # ---- conf masks ----
    conf_geom_soft, conf01_full, conf_info = make_soft_mask_from_conf(
        tgt_depth_conf.float(),
        out_hw=(H, W),
        thr=conf_thr,
        temp=conf_temp,
        conf_raw_min=conf_raw_min,
        conf_raw_max=conf_raw_max,
        conf_auto_norm=conf_auto_norm,
        conf_use_quantile=conf_use_quantile,
        conf_qlo=conf_qlo,
        conf_qhi=conf_qhi,
        valid_mask=valid_mask,
    )
    conf_geom_soft = conf_geom_soft.clamp(0, 1) * valid_mask

    conf_mask_mode = "depth_conf"
    conf_mask = conf_geom_soft
    if pred_conf_gate is not None:
        conf_mask = pred_conf_gate
        conf_mask_mode = "pred_conf"

    if conf_mask is None:
        conf_mask = torch.ones_like(valid_mask)
        conf_mask_mode = "all_ones"
    else:
        conf_mask = conf_mask.to(device).float()
        if conf_mask.dim() == 2:
            conf_mask = conf_mask[None, None, ...]
        elif conf_mask.dim() == 3:
            conf_mask = conf_mask.unsqueeze(1)
        if conf_mask.dim() == 4 and conf_mask.shape[1] != 1:
            conf_mask = conf_mask[:, :1, ...]
        if conf_mask.shape[-2:] != (H, W):
            conf_mask = F.interpolate(conf_mask, size=(H, W), mode="bilinear", align_corners=False)
        conf_mask = conf_mask.clamp(0.0, 1.0)
        if conf_weight_detach:
            conf_mask = conf_mask.detach()
        if float(conf_gate_floor) > 0:
            conf_mask = float(conf_gate_floor) + (1.0 - float(conf_gate_floor)) * conf_mask
        if float(conf_weight_strength) < 1.0:
            conf_mask = (1.0 - float(conf_weight_strength)) + float(conf_weight_strength) * conf_mask

    if not use_conf_in_train_mask:
        conf_mask = torch.ones_like(valid_mask)
        conf_mask_mode = "all_ones"

    # ---- train mask ----
    train_mask_mode = str(_resolve_opt(
        train_mask_mode, "train_mask_mode", "fg_conf")).lower()
    if train_mask_mode not in ("fg_conf", "valid_conf", "valid_only"):
        train_mask_mode = "fg_conf"

    if train_mask_mode == "valid_only":
        train_mask = valid_mask.clone()
        cover_train = float(train_mask.mean().item())
    else:
        base_mask = valid_mask if train_mask_mode == "valid_conf" else fg_mask
        train_mask = (base_mask * conf_mask).clamp(0, 1)
        cover_train = float(train_mask.mean().item())
        if cover_train < float(train_min_cover):
            if train_mask_mode == "fg_conf":
                train_mask = (fg_mask * valid_mask).clamp(0, 1)
                cover_train = float(train_mask.mean().item())
            if cover_train < float(train_min_cover):
                train_mask = valid_mask.clone()
                cover_train = float(train_mask.mean().item())

    gate_conf = pred_conf_gate
    if gate_conf is None and batch is not None:
        # prefer model predicted confidence if present
        gate_conf = batch.get('pred_conf', None)
    if gate_conf is None:
        gate_conf = train_mask.new_ones(train_mask.shape)
    else:
        gate_conf = gate_conf.to(device).float()
        if gate_conf.dim() == 2:
            gate_conf = gate_conf[None, None, ...]
        elif gate_conf.dim() == 3:
            gate_conf = gate_conf.unsqueeze(1)
        if gate_conf.dim() == 4 and gate_conf.shape[1] != 1:
            gate_conf = gate_conf[:, :1, ...]
        if gate_conf.shape[-2:] != (H, W):
            gate_conf = F.interpolate(
                gate_conf, size=(H, W), mode="bilinear", align_corners=False)

    # gate_conf expected in [0,1]; clamp/gamma/soft-floor + strength if configured
    if use_conf_gate:
        gate_conf = gate_conf.clamp(0.0, 1.0)
        if abs(conf_gate_gamma - 1.0) > 1e-6:
            gate_conf = gate_conf.pow(conf_gate_gamma)
        if recon_gate_floor > 0.0:
            gate_conf = recon_gate_floor + \
                (1.0 - recon_gate_floor) * gate_conf
        if conf_weight_detach:
            gate_conf = gate_conf.detach()
        if conf_weight_strength < 1.0:
            gate_conf = (1.0 - conf_weight_strength) + \
                conf_weight_strength * gate_conf
    else:
        gate_conf = train_mask.new_ones(train_mask.shape)

    if recon_mask_mode == "train":
        recon_base = train_mask
    elif recon_mask_mode == "valid":
        recon_base = valid_mask
    else:
        recon_base = fg_mask
    recon_base = recon_base.clamp(0.0, 1.0)

    conf_weight_map = None
    if conf_weight_min is not None:
        conf_weight_map = conf01_full
        if conf_weight_map.shape[-2:] != recon_base.shape[-2:]:
            conf_weight_map = F.interpolate(
                conf_weight_map, size=recon_base.shape[-2:], mode="nearest")
        if conf_weight_min > 0.0:
            conf_weight_map = conf_weight_min + \
                (1.0 - conf_weight_min) * conf_weight_map
        conf_weight_map = conf_weight_map.clamp(0.0, 1.0)

    recon_fg = recon_base * gate_conf
    if conf_weight_map is not None:
        recon_fg = recon_fg * conf_weight_map
    if float(bg_weight) > 0.0:
        recon_weight_raw = recon_fg + (1.0 - recon_base) * float(bg_weight)
    else:
        recon_weight_raw = recon_fg
    recon_weight = recon_weight_raw
    if recon_weight_renorm:
        renorm_mask = recon_base
        denom = (recon_weight * renorm_mask).sum()
        target = renorm_mask.sum()
        if float(target.item()) > 0.0:
            scale = (target / (denom + 1e-8)).detach()
            recon_weight = recon_weight * scale
    if recon_weight_clip_max > 0:
        recon_weight = recon_weight.clamp(
            min=0.0, max=float(recon_weight_clip_max))
    recon_weight = recon_weight * valid_mask

    aux = {
        "tgt_depth_conf_raw": tgt_depth_conf_raw.detach(),
        "tgt_depth_conf": conf01_full.detach(),
        "conf_info": conf_info,
        "fg_mask": fg_mask.detach(),
        "train_mask": train_mask.detach(),
        "gate_loss": gate_conf.detach(),
        "use_conf_gate_loss": bool(use_conf_gate),
        "conf_gate_gamma": float(conf_gate_gamma),
        "conf_gate_strength": float(conf_gate_strength),
        "conf_weight_strength": float(conf_weight_strength),
        "conf_weight_min": float(conf_weight_min)
        if conf_weight_min is not None else None,
        "recon_gate_floor": float(recon_gate_floor),
        "recon_weight_renorm": bool(recon_weight_renorm),
        "recon_weight_clip_max": float(recon_weight_clip_max),
        "recon_weight_raw": recon_weight_raw.detach(),
        "recon_weight": recon_weight.detach(),
        "valid_mask": valid_mask.detach(),
        "cover_train": cover_train,
        "cover_fg": float(fg_mask.mean().item()),
        "cover_conf": float(conf_mask.mean().item()),
        "cover_conf_geom": float(conf_geom_soft.mean().item()),
        "cover_valid": float(valid_mask.mean().item()),
        "conf_mask_mode": conf_mask_mode,
        "train_mask_mode": train_mask_mode,
        "recon_mask_mode": recon_mask_mode,
        "source_fg_key": source_fg_key,
        "source_valid_key": source_valid_key,
        "source_depth_conf_key": depth_conf_key,
        "source_pointmap_key": source_pointmap_key,
        "fg_drop_ground": bool(drop_ground),
        "fg_ground_axis": int(fg_ground_axis_v),
        "fg_ground_q": float(fg_ground_q_v),
        "fg_ground_margin": float(fg_ground_margin_v),
        "fg_ground_min_points": int(fg_ground_min_points_v),
        "fg_ground_applied": bool(
            ground_info.get("applied", False) if isinstance(ground_info, dict) else False
        ),
        "fg_ground_floor": (ground_info.get("floor_vals") if isinstance(ground_info, dict) else None),
    }
    log_mask_stats = bool(_resolve_opt(None, "log_mask_stats", False))
    if log_mask_stats:
        try:
            mstats = {}
            mstats.update(_mask_stats(valid_mask, "valid_"))
            mstats.update(_mask_stats(fg_mask, "fg_"))
            mstats.update(_mask_stats(conf_mask, "conf_"))
            mstats.update(_mask_stats(train_mask, "train_"))
            aux["mask_stats"] = mstats
        except Exception:
            pass
    return train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf, aux


# ---------------------------
# Metrics
# ---------------------------
def psnr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    tgt = tgt.clamp(0, 1)
    mse = F.mse_loss(pred, tgt, reduction="mean")
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_kernel(window_size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device,
                          dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).contiguous()


def ssim(pred: torch.Tensor, tgt: torch.Tensor, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03, eps=1e-8) -> torch.Tensor:
    pred = pred.clamp(0, 1).float()
    tgt = tgt.clamp(0, 1).float()
    B, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype
    kernel = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    kernel = kernel.view(1, 1, window_size, window_size).repeat(C, 1, 1, 1)

    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(tgt, kernel, padding=window_size // 2, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel,
                        padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(tgt * tgt, kernel,
                        padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * tgt, kernel,
                        padding=window_size // 2, groups=C) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / (den + eps)).mean()


# ---------------------------
# VGG perceptual (robust weights)
# ---------------------------
def _make_vgg16_features():
    from torchvision.models import vgg16
    try:
        from torchvision.models import VGG16_Weights
        try:
            return vgg16(weights=VGG16_Weights.DEFAULT).features
        except Exception:
            return vgg16(weights=None).features
    except Exception:
        try:
            return vgg16(pretrained=True).features
        except Exception:
            return vgg16(pretrained=False).features


class VGGPerceptualLoss(nn.Module):
    def __init__(self, slice_to=16):
        super().__init__()
        vgg = _make_vgg16_features()
        self.slice = nn.Sequential(*list(vgg.children())[:slice_to]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target, mask=None):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        feat_p = self.slice(self._norm(pred))
        feat_t = self.slice(self._norm(target))
        if mask is None:
            return F.l1_loss(feat_p, feat_t, reduction="mean")
        mf = F.interpolate(
            mask, size=feat_p.shape[-2:], mode="bilinear", align_corners=False)
        return masked_l1(feat_p, feat_t, mf)


# ---------------------------
# Edge / TV losses (lightweight)
# ---------------------------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0,  0,  0],
                       [1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    kx, ky = _sobel_kernels(x.device, x.dtype)
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def tv_l1(x: torch.Tensor) -> torch.Tensor:
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


# ---------------------------
# Debug pack saver (updated: split cat panels)
# ---------------------------
def _write_debug_error(out_dir: str, prefix: str, step: int, msg: str):
    os.makedirs(out_dir, exist_ok=True)
    text = str(msg).strip()
    if not text:
        text = "unknown error"
    # create a simple white panel with error text
    w, h = 960, 540
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        while len(line) > 80:
            lines.append(line[:80])
            line = line[80:]
        lines.append(line)
    if not lines:
        lines = [text]
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 18
    path = os.path.join(out_dir, f"{prefix}_ERROR_step{step:06d}.png")
    img.save(path)
    try:
        with open(path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def save_debug_pack(
    pred,
    tgt,
    aux,
    step,
    out_dir="debug_viewdec_ablation",
    prefix="train",
    split_cat_panels: bool = True,
    fixed_ranges: Optional[dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    def _fail(msg: str):
        _write_debug_error(out_dir, prefix, int(step), msg)
        raise RuntimeError(msg)

    # hard checks to avoid silent "whiteboard" debug panels
    if not torch.is_tensor(pred) or pred.ndim != 4:
        _fail(f"[debug] pred invalid: type={type(pred)} ndim={getattr(pred, 'ndim', None)}")
    if not torch.is_tensor(tgt) or tgt.ndim != 4:
        _fail(f"[debug] tgt invalid: type={type(tgt)} ndim={getattr(tgt, 'ndim', None)}")
    if pred.numel() == 0 or tgt.numel() == 0:
        _fail(f"[debug] pred/tgt empty: pred_shape={tuple(pred.shape)} tgt_shape={tuple(tgt.shape)}")
    if pred.shape[0] != tgt.shape[0]:
        _fail(f"[debug] batch mismatch: pred_shape={tuple(pred.shape)} tgt_shape={tuple(tgt.shape)}")
    if pred.shape[-2:] != tgt.shape[-2:]:
        _fail(f"[debug] HW mismatch: pred_shape={tuple(pred.shape)} tgt_shape={tuple(tgt.shape)}")

    pred_raw = pred
    pred_vis = pred
    vis_weight = None
    if isinstance(aux, dict):
        gate = aux.get("gate", None)
        recon_weight = aux.get("recon_weight", None)
        if torch.is_tensor(gate):
            vis_weight = gate
        if torch.is_tensor(recon_weight):
            vis_weight = recon_weight if vis_weight is None else (vis_weight * recon_weight)
    if vis_weight is not None:
        pred_vis = (pred * vis_weight).clamp(0, 1)

    pred_raw_img = to_u8_img(pred_raw[0], name="pred_raw")
    pred_vis_img = to_u8_img(pred_vis[0], name="pred_vis")
    pred_img = pred_raw_img
    tgt_img = to_u8_img(tgt[0], name="tgt")

    path_pred = os.path.join(out_dir, f"{prefix}_pred_step{step:06d}.png")
    path_tgt = os.path.join(out_dir, f"{prefix}_tgt_step{step:06d}.png")
    save_png(pred_img, path_pred)
    save_png(tgt_img, path_tgt)

    path_pred_raw = os.path.join(out_dir, f"{prefix}_pred_raw_step{step:06d}.png")
    path_pred_vis = os.path.join(out_dir, f"{prefix}_pred_vis_step{step:06d}.png")
    save_png(pred_raw_img, path_pred_raw)
    save_png(pred_vis_img, path_pred_vis)

    cat_pt = np.concatenate([pred_img, tgt_img], axis=1)
    path_cat_pt = os.path.join(
        out_dir, f"{prefix}_cat_pred_tgt_step{step:06d}.png")
    save_png(cat_pt, path_cat_pt)
    if split_cat_panels:
        pass

    # diff heatmap (pred vs tgt)
    diff = (pred_raw - tgt).abs().mean(dim=1, keepdim=True)
    diff_img = to_u8_heatmap(diff[0], name="diff")
    path_diff = os.path.join(out_dir, f"{prefix}_diff_step{step:06d}.png")
    save_png(diff_img, path_diff)
    cat_ptd = np.concatenate([pred_img, tgt_img, diff_img], axis=1)
    path_cat_ptd = os.path.join(
        out_dir, f"{prefix}_cat_pred_tgt_diff_step{step:06d}.png")
    save_png(cat_ptd, path_cat_ptd)

    if not aux:
        return

    H, W = int(pred.shape[-2]), int(pred.shape[-1])

    def _resize_to_hw(x: torch.Tensor, is_mask: bool):
        if x is None or (not isinstance(x, torch.Tensor)) or x.ndim != 4:
            return None
        if x.shape[-2:] == (H, W):
            return x
        if is_mask or x.shape[1] == 1:
            return F.interpolate(x, size=(H, W), mode="nearest")
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def _pick_range(key: str):
        if not isinstance(fixed_ranges, dict):
            return None
        if key in fixed_ranges:
            return fixed_ranges[key]
        k = key.lower()
        if "conf" in k and "conf" in fixed_ranges:
            return fixed_ranges["conf"]
        if "depth" in k and "depth" in fixed_ranges:
            return fixed_ranges["depth"]
        if ("mask" in k) and ("mask" in fixed_ranges):
            return fixed_ranges["mask"]
        if ("weight" in k) and ("weight" in fixed_ranges):
            return fixed_ranges["weight"]
        return None

    def _save1(name, x1):
        if not torch.is_tensor(x1) or x1.ndim != 4 or x1.numel() == 0:
            _fail(f"[debug] aux[{name}] invalid: type={type(x1)} shape={getattr(x1, 'shape', None)}")
        x1 = _resize_to_hw(x1, is_mask=True)
        if x1 is None:
            _fail(f"[debug] aux[{name}] resize failed: shape={getattr(x1, 'shape', None)}")
        r = _pick_range(name)
        if r is not None and isinstance(r, (tuple, list)) and len(r) == 2:
            img = to_u8_img_fixed(x1[0], float(r[0]), float(r[1]), name=name)
        else:
            img = to_u8_img(x1[0], name=name)
        path_single = os.path.join(
            out_dir, f"{prefix}_{name}_step{step:06d}.png")
        save_png(img, path_single)

        cat = np.concatenate([img, pred_img, tgt_img], axis=1)
        path_cat = os.path.join(
            out_dir, f"{prefix}_cat_{name}_pred_tgt_step{step:06d}.png")
        save_png(cat, path_cat)

        if split_cat_panels:
            save_cat_panels(cat, path_cat)

    if isinstance(aux, dict):
        tgt_fg = aux.get("tgt_fg", None)
        if torch.is_tensor(tgt_fg):
            try:
                if tgt_fg.dim() == 2:
                    tgt_fg = tgt_fg.unsqueeze(0).unsqueeze(0)
                elif tgt_fg.dim() == 3:
                    tgt_fg = tgt_fg.unsqueeze(1)
                if tgt_fg.ndim == 4:
                    tgt_fg = _resize_to_hw(tgt_fg, is_mask=True)
                    fg = (tgt_fg[0, 0] > 0.5).float()
                    mask = fg.detach().cpu().numpy().astype(bool)
                    if mask.any():
                        overlay = tgt_img.astype(np.float32)
                        alpha = 0.35
                        green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
                        overlay[mask] = overlay[mask] * (1.0 - alpha) + green * alpha
                        overlay = np.clip(overlay, 0, 255).round().astype(np.uint8)
                    else:
                        overlay = tgt_img.copy()
                    path_overlay = os.path.join(
                        out_dir, f"{prefix}_gt_with_fg_overlay_step{step:06d}.png")
                    save_png(overlay, path_overlay)
            except Exception:
                pass

    for key in ["valid_mask", "fg_mask", "train_mask",
                "recon_weight_raw", "recon_weight",
                "tgt_depth_conf", "tgt_depth_conf_raw",
                "tgt_depth", "pred_depth",
                "pred_conf", "gate", "gate_loss"]:
        if key in aux and isinstance(aux[key], torch.Tensor):
            _save1(key, aux[key])


def build_vis_ranges(args) -> Optional[dict]:
    ranges = {}
    try:
        if float(args.vis_conf_max) > float(args.vis_conf_min):
            ranges["conf"] = (float(args.vis_conf_min),
                              float(args.vis_conf_max))
    except Exception:
        pass
    try:
        if float(args.vis_depth_max) > float(args.vis_depth_min):
            ranges["depth"] = (float(args.vis_depth_min),
                               float(args.vis_depth_max))
    except Exception:
        pass
    try:
        if float(args.vis_mask_max) > float(args.vis_mask_min):
            ranges["mask"] = (float(args.vis_mask_min),
                              float(args.vis_mask_max))
    except Exception:
        pass
    try:
        if float(args.vis_weight_max) > float(args.vis_weight_min):
            ranges["weight"] = (float(args.vis_weight_min),
                                float(args.vis_weight_max))
    except Exception:
        pass
    return ranges if ranges else None


def _tensor_to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def dump_batch_npz(batch: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keep_keys = [
        "src_imgs", "src_depth", "src_depth_conf", "src_pointmap",
        "tgt_img", "tgt_depth", "tgt_depth_conf", "tgt_pointmap",
        "src_K", "src_T", "tgt_K", "tgt_T",
        "src_vids", "tgt_vid",
        "geom_path", "tgt_img_path", "src_img_paths", "cam_names",
    ]
    out = {}
    for k in keep_keys:
        if k in batch:
            out[k] = _tensor_to_np(batch[k])
    if not out:
        # fallback: dump all tensor keys
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = _tensor_to_np(v)
    np.savez_compressed(path, **out)


def save_repro_pack(out_dir: str, args: Any, batch: Optional[Dict[str, Any]] = None, note: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _has_nonfinite(x: Any) -> bool:
    if not torch.is_tensor(x):
        return False
    return not torch.isfinite(x).all().item()


def dump_nan_guard(
    out_dir: str,
    args: Any,
    batch: Dict[str, Any],
    pred_rgb: Optional[torch.Tensor],
    tgt_img: Optional[torch.Tensor],
    aux_dbg: Optional[Dict[str, Any]],
    model: Optional[nn.Module] = None,
    note: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    try:
        save_repro_pack(out_dir, args, batch=batch, note=note)
    except Exception:
        pass


def log_param_groups(optimizer, model=None, batch=None, logger=print):
    """Log optimizer param-groups safely.

    Args:
        optimizer: torch optimizer
        model: optional nn.Module; if given, we map params -> names for samples
        batch: optional dict-like; if given, we log keys and a few shapes
        logger: callable
    """
    try:
        id2name = {}
        if model is not None:
            try:
                for n, p in model.named_parameters():
                    id2name[id(p)] = n
            except Exception:
                id2name = {}

        for gi, g in enumerate(getattr(optimizer, "param_groups", [])):
            params = g.get("params", [])
            lr = g.get("lr", None)
            wd = g.get("weight_decay", None)
            n_params = len(params)

            sample = []
            for p in params:
                nm = id2name.get(id(p))
                if nm is not None:
                    sample.append(nm)
                if len(sample) >= 3:
                    break

            logger(
                f"[lr_group {gi}] n_params={n_params} lr={lr} wd={wd} "
                f"sample={', '.join(sample) if sample else 'n/a'}"
            )

        if batch is not None and hasattr(batch, "keys"):
            keys = list(batch.keys())
            logger(f"[batch] keys={keys}")
            for k in keys[:8]:
                v = batch[k]
                shp = getattr(v, "shape", None)
                if shp is not None:
                    logger(f"  - {k}: shape={tuple(shp)}")

    except Exception as e:
        logger(f"[log_param_groups] warn: {type(e).__name__}: {e}")


# ---------------------------
# Checkpoint helpers
# ---------------------------
def _try_load_torch(path: str, device: str):
    return torch.load(path, map_location=device)


def load_checkpoint(path: str, device: str):
    obj = _try_load_torch(path, device)
    if isinstance(obj, dict) and ("model" in obj or "state_dict" in obj or "model_state_dict" in obj):
        return obj
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return {"model": obj}
    return {"model": obj}


def save_checkpoint(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


# ---------------------------
# LR schedule: Warmup + Cosine (step-based)
# ---------------------------
def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int):
        s = step + 1
        if warmup_steps > 0 and s <= warmup_steps:
            return float(s) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        t = (s - warmup_steps) / float(total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------
# Ramp / Schedule helpers (supports smoothstep)
# ---------------------------
def _ramp_progress(step: int, warmup: int, ramp: int) -> float:
    s = int(step)
    warmup = int(max(0, warmup))
    ramp = int(max(0, ramp))
    if warmup > 0 and s < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    t = max(0, s - warmup)
    return max(0.0, min(1.0, float(t) / float(ramp)))


def _apply_ramp_mode(p: float, mode: str, k: float = 5.0) -> float:
    m = str(mode).lower()
    if m == "smoothstep":
        return p * p * (3.0 - 2.0 * p)
    if m == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * p)
    if m == "exp":
        k = max(1e-6, float(k))
        return (1.0 - math.exp(-k * p)) / (1.0 - math.exp(-k))
    return p


def schedule_value(step: int, warmup: int, ramp: int, mode: str, k: float,
                   vmin: float = 0.0, vmax: float = 1.0) -> float:
    p = _ramp_progress(step, warmup, ramp)
    p = _apply_ramp_mode(p, mode, k)
    vmin = float(vmin)
    vmax = float(vmax)
    return vmin + (vmax - vmin) * p


# ---------------------------
# EMA
# ---------------------------
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    decay = float(decay)
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        src = msd[k]
        if not torch.is_tensor(v) or not torch.is_tensor(src):
            esd[k] = src
            continue
        if v.dtype.is_floating_point:
            esd[k].mul_(decay).add_(src, alpha=(1.0 - decay))
        else:
            esd[k].copy_(src)
    ema_model.load_state_dict(esd, strict=False)


@torch.no_grad()
def ema_sanity_check(model: nn.Module, ema_model: nn.Module):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in msd.items():
        if k in esd and torch.is_tensor(v) and torch.is_tensor(esd[k]):
            diff = (v.detach() - esd[k].detach()).abs().max().item()
            print(f"[info] ema sanity: key={k} max_abs_diff={diff:.6e}")
            return diff
    print("[warn] ema sanity: no common tensor keys found.")
    return None


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--zju_root", type=str,
                   default=r"F:\datasets\ZJU_MoCap\data\zju_mocap")
    p.add_argument("--seq_names", type=str, default="CoreView_390",
                   help="comma separated, e.g. CoreView_390,CoreView_392")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1,
                   help="gradient accumulation steps")

    p.add_argument("--num_workers_train", type=int, default=8)
    p.add_argument("--num_workers_val", type=int, default=4)

    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--conf_head_lr_mult", type=float, default=2.0,
                   help="LR multiplier for conf head (core.out_conf if split_conf_head else core.out_conv). Set 1.0 to disable.")
    p.add_argument("--print_param_groups", action="store_true", default=False,
                   help="Print optimizer param group names for debugging.")

    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="random",
                   choices=["random", "contiguous"],
                   help="Split frames by random shuffle or contiguous tail for val.")
    p.add_argument("--frame_subsample", type=int, default=1)
    p.add_argument("--num_src_views", type=int, default=3)
    p.add_argument("--holdout_view_ids", type=str, default="",
                   help="Comma-separated camera IDs to hold out (val-only targets).")
    p.add_argument("--holdout_view_names", type=str, default="",
                   help="Comma-separated camera names to hold out (overrides ids if provided).")
    p.add_argument("--bad_sample_policy", type=str, default="warn",
                   choices=["warn", "skip", "raise", "mask", "drop_src", "drop-src"],
                   help="How to handle detected bad/white images in dataset.")
    p.add_argument("--white_mean_thr", type=float, default=0.98,
                   help="Mean threshold for white-like image detection.")
    p.add_argument("--white_std_thr", type=float, default=1e-3,
                   help="Std threshold for near-constant image detection.")
    p.add_argument("--bad_sample_max_retry", type=int, default=3,
                   help="Max retries when bad_sample_policy=skip.")
    p.add_argument("--report_bad_samples", dest="report_bad_samples",
                   action="store_true", default=True)
    p.add_argument("--no_report_bad_samples", dest="report_bad_samples",
                   action="store_false")

    p.add_argument("--resume", type=str, default="", help="ckpt path")
    p.add_argument("--log_dir", type=str, default="debug_viewdec_ablation")
    p.add_argument("--ckpt_dir", type=str, default="ckpt")

    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    p.add_argument("--tf32", dest="tf32", action="store_true")
    p.add_argument("--no_tf32", dest="tf32", action="store_false")
    p.set_defaults(tf32=True)

    p.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true")
    p.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    p.set_defaults(cudnn_benchmark=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--compile", action="store_true", default=False)

    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", dest="use_ema",
                   action="store_false", help="Disable EMA")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--ema_start_step", type=int, default=0,
                   help="Start EMA updates after this optimizer step (0 = start immediately).")
    p.add_argument("--ema_start_after_warmup", action="store_true", default=False,
                   help="Start EMA after warmup_steps (overrides ema_start_step if larger).")
    p.add_argument("--ema_start_hardcopy", dest="ema_start_hardcopy",
                   action="store_true", help="Hard-copy model -> EMA when EMA starts.")
    p.add_argument("--no_ema_start_hardcopy", dest="ema_start_hardcopy",
                   action="store_false", help="Disable hard-copy at EMA start.")
    p.set_defaults(ema_start_hardcopy=True)
    p.add_argument("--best_by", type=str, default="raw_psnr",
                   choices=["raw_psnr", "raw_ssim", "raw", "raw_l1", "ema", "ema_psnr", "ema_ssim", "psnr_fg"])

    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["cosine", "plateau"])
    p.add_argument("--warmup_steps", type=int, default=400)
    p.add_argument("--min_lr_ratio", type=float,
                   default=0.1)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_patience", type=int, default=1)

    p.add_argument("--ref_mode", type=str, default="first",
                   choices=["first", "mean", "random"])
    p.add_argument("--use_conf_gate", action="store_true", default=True)
    p.add_argument("--no_use_conf_gate", dest="use_conf_gate",
                   action="store_false", help="Disable confidence gate")
    p.add_argument("--conf_gate_detach", action="store_true", default=True,
                   help="Detach pred_conf before gating RGB skip/loss weights (default on)")
    p.add_argument("--no_conf_gate_detach", dest="conf_gate_detach",
                   action="store_false", help="Allow RGB loss to backprop into pred_conf gate")
    p.add_argument("--conf_weight_detach", action="store_true", default=True,
                   help="Detach pred_conf when used as loss weights/masks")
    p.add_argument("--no_conf_weight_detach", dest="conf_weight_detach",
                   action="store_false", help="Allow loss weights to backprop into pred_conf")
    p.add_argument("--conf_gate_floor", type=float, default=0.05,
                   help="Skip gate floor (model ref skip), 0~1")
    p.add_argument("--conf_gate_gamma", type=float, default=2.0,
                   help="Loss gate gamma (>1 emphasizes high-conf, <1 flattens)")
    p.add_argument("--recon_gate_floor", type=float, default=0.1,
                   help="Soft floor for recon weight gate: w = fg * (floor + (1-floor)*conf)")
    p.add_argument("--conf_gate_warmup", type=int, default=1000,
                   help="Disable pred_conf gate for first N optimizer steps (loss weighting only)")
    p.add_argument("--conf_gate_ramp", type=int, default=0,
                   help="Ramp steps after warmup to smoothly enable conf gate (0 = hard switch)")
    p.add_argument("--conf_gate_ramp_mode", type=str, default="smoothstep",
                   choices=["linear", "cosine", "exp", "smoothstep"],
                   help="Ramp mode for conf gate strength")
    p.add_argument("--conf_gate_ramp_k", type=float, default=5.0,
                   help="Exp ramp sharpness (only for exp mode)")
    p.add_argument("--conf_gate_min", type=float, default=0.05,
                   help="Global gate strength min (avoid fully off)")
    p.add_argument("--conf_weight_warmup", type=int, default=None,
                   help="Warmup steps for loss-weight conf mask (default=conf_gate_warmup)")
    p.add_argument("--conf_weight_ramp", type=int, default=None,
                   help="Ramp steps for loss-weight conf mask (default=conf_gate_ramp)")
    p.add_argument("--conf_weight_ramp_mode", type=str, default=None,
                   choices=["linear", "cosine", "exp", "smoothstep"],
                   help="Ramp mode for loss-weight conf mask (default=conf_gate_ramp_mode)")
    p.add_argument("--conf_weight_ramp_k", type=float, default=None,
                   help="Exp ramp sharpness for loss-weight conf mask (default=conf_gate_ramp_k)")
    p.add_argument("--conf_weight_min", type=float, default=None,
                   help="Min strength for loss-weight conf mask (default=0.0)")
    p.add_argument("--conf_bias_init", type=float, default=-1.0,
                   help="Init conf bias; -1 disables, (0,1) treated as prob, other values as logits (temp-aware)")
    p.add_argument("--use_tone", action="store_true", default=True)
    p.add_argument("--no_use_tone", dest="use_tone",
                   action="store_false", help="Disable tone head")
    p.add_argument("--init_alpha", type=float, default=0.12)
    p.add_argument("--use_view_cond", nargs="?", const=1, default=0,
                   type=int, choices=[0, 1])
    p.add_argument("--num_views", type=int, default=0,
                   help="0 means infer from dataset")
    p.add_argument("--view_dim", type=int, default=16)
    p.add_argument("--view_affine_strength", type=float, default=1.0)
    p.add_argument("--view_cond_mode", type=str, default="tgt",
                   choices=["tgt", "tgt_src_mean"])
    p.add_argument("--finetune_view_only", nargs="?", const=1, default=0,
                   type=int, choices=[0, 1])
    p.add_argument("--reset_optim", action="store_true", default=False)
    p.add_argument("--rgb_sigmoid_temp", type=float, default=1.0,
                   help="RGB sigmoid temperature (larger => softer)")
    p.add_argument("--conf_sigmoid_temp", type=float, default=1.0,
                   help="Conf sigmoid temperature (larger => softer)")
    p.add_argument("--split_conf_head", action="store_true", default=False,
                   help="Use separate RGB/Conf heads (conf head can use lower LR)")
    p.add_argument("--logit_clip", type=float, default=10.0,
                   help="Clamp RGB/Conf logits to [-clip, clip] (<=0 to disable)")

    p.add_argument("--lambda_photo", type=float, default=1.0,
                   help="Photometric base weight")
    p.add_argument("--photo_loss", type=str, default="charbonnier",
                   choices=["l1", "huber", "charbonnier"],
                   help="Photometric loss type")
    p.add_argument("--photo_huber_delta", type=float, default=0.01)
    p.add_argument("--photo_charb_eps", type=float, default=1e-3)
    p.add_argument("--photo_charb_alpha", type=float, default=0.5)
    p.add_argument("--lambda_percep", type=float, default=0.05)
    p.add_argument("--lambda_conf", type=float, default=1e-3)
    p.add_argument("--lambda_depth", type=float, default=0.05,
                   help="Depth loss weight (if use_depth_head)")
    p.add_argument("--depth_loss", type=str, default="charbonnier",
                   choices=["l1", "huber", "charbonnier"],
                   help="Depth loss type")
    p.add_argument("--depth_huber_delta", type=float, default=0.01)
    p.add_argument("--depth_charb_eps", type=float, default=1e-3)
    p.add_argument("--depth_charb_alpha", type=float, default=0.5)
    p.add_argument("--lambda_depth_edge", type=float, default=0.0,
                   help="Edge-aware depth smoothness weight")
    p.add_argument("--lambda_bright", type=float, default=0.5)
    p.add_argument("--lambda_contrast", type=float, default=0.5)
    p.add_argument("--lambda_alpha_reg", type=float, default=1e-4)

    p.add_argument("--lambda_edge", type=float, default=0.02,
                   help="edge/gradient loss weight (small but useful)")
    p.add_argument("--lambda_tv_conf", type=float, default=1e-4,
                   help="TV loss on pred_conf to suppress salt-pepper")
    p.add_argument("--lambda_ssim", type=float, default=0.0,
                   help="(optional) add 1-SSIM loss, usually keep 0 or tiny")
    p.add_argument("--lambda_conf_mean", type=float, default=0.0,
                   help="Mean-conf regularizer weight (anti-gaming)")
    p.add_argument("--conf_mean_target", type=float, default=0.5,
                   help="Target mean for conf regularizer")
    p.add_argument("--conf_err_weight", type=float, default=1.0,
                   help="Weight for error-based conf supervision (inside conf_reg); 0 disables")
    p.add_argument("--conf_err_target", type=str, default="exp",
                   choices=["exp", "linear"],
                   help="Target type from reconstruction error")
    p.add_argument("--conf_err_k", type=float, default=1.0,
                   help="Scale for error->target mapping (exp or linear)")
    p.add_argument("--conf_err_loss", type=str, default="l1",
                   choices=["l1", "bce"],
                   help="Loss type for error-based conf supervision")
    p.add_argument("--lambda_mv", type=float, default=0.0,
                   help="Lightweight multi-view consistency weight")
    p.add_argument("--mv_mode", type=str, default="mean",
                   choices=["mean", "first"], help="Source view aggregation for mv consistency")
    p.add_argument("--use_depth_head", action="store_true", default=False,
                   help="Enable extra depth head and depth loss")
    p.add_argument("--photo_warmup", type=int, default=0)
    p.add_argument("--photo_ramp", type=int, default=0)
    p.add_argument("--photo_ramp_mode", type=str, default="smoothstep",
                   choices=["linear", "cosine", "exp", "smoothstep"])
    p.add_argument("--photo_ramp_k", type=float, default=5.0)
    p.add_argument("--photo_min_ratio", type=float, default=0.05)
    p.add_argument("--conf_loss_warmup", type=int, default=0)
    p.add_argument("--conf_loss_ramp", type=int, default=0)
    p.add_argument("--conf_loss_ramp_mode", type=str, default="smoothstep",
                   choices=["linear", "cosine", "exp", "smoothstep"])
    p.add_argument("--conf_loss_ramp_k", type=float, default=5.0)
    p.add_argument("--conf_loss_min_ratio", type=float, default=0.05)
    p.add_argument("--depth_loss_warmup", type=int, default=0)
    p.add_argument("--depth_loss_ramp", type=int, default=0)
    p.add_argument("--depth_loss_ramp_mode", type=str, default="smoothstep",
                   choices=["linear", "cosine", "exp", "smoothstep"])
    p.add_argument("--depth_loss_ramp_k", type=float, default=5.0)
    p.add_argument("--depth_loss_min_ratio", type=float, default=0.05)

    p.add_argument("--conf_thr", type=float, default=0.2)
    p.add_argument("--conf_temp", type=float, default=0.06)
    p.add_argument("--use_conf_loss_gate", action="store_true", default=True,
                   help="Use depth_conf soft gate in train/recon weights")
    p.add_argument("--no_use_conf_loss_gate", dest="use_conf_loss_gate",
                   action="store_false", help="Disable depth_conf gating in loss weights")
    p.add_argument("--train_mask_mode", type=str, default="fg_conf",
                   choices=["fg_conf", "valid_conf", "valid_only"],
                   help="Train mask base: fg*conf, valid*conf, or valid only")
    p.add_argument("--recon_mask_mode", type=str, default="valid",
                   choices=["fg", "train", "valid"],
                   help="Recon-weight base mask: fg, train, or valid")
    p.add_argument("--log_mask_stats", dest="log_mask_stats", action="store_true", default=True,
                   help="Log mask coverage/CC/boundary stats")
    p.add_argument("--no_log_mask_stats", dest="log_mask_stats", action="store_false",
                   help="Disable mask stats logging")
    p.add_argument("--train_min_cover", type=float, default=0.10)
    p.add_argument("--fg_thr", type=float, default=0.5)
    p.add_argument("--fg_min_cover", type=float, default=0.05)
    p.add_argument("--fg_dilate_k", type=int, default=7)
    p.add_argument("--fg_keep_largest_cc", type=int, default=1, choices=[0, 1])
    p.add_argument("--fg_lcc_min_pixels", type=int, default=32)
    p.add_argument("--fg_drop_ground", dest="fg_drop_ground",
                   action="store_true", help="Remove ground from fg mask using pointmap height quantile")
    p.add_argument("--no_fg_drop_ground", dest="fg_drop_ground",
                   action="store_false", help="Disable pointmap ground removal for fg mask")
    p.set_defaults(fg_drop_ground=True)
    p.add_argument("--fg_ground_axis", type=int, default=1,
                   help="Vertical axis index in pointmap (0=x,1=y,2=z)")
    p.add_argument("--fg_ground_q", type=float, default=0.05,
                   help="Ground height quantile within fg&valid region")
    p.add_argument("--fg_ground_margin", type=float, default=0.02,
                   help="Margin above ground height to keep (same units as pointmap)")
    p.add_argument("--fg_ground_min_points", type=int, default=64,
                   help="Min fg points to estimate ground height per sample")
    p.add_argument("--valid_min_cover", type=float, default=0.10)
    p.add_argument("--valid_dilate_k", type=int, default=7)
    p.add_argument("--valid_k_max", type=int, default=31)
    p.add_argument("--bg_weight", type=float, default=0.05)
    p.add_argument("--recon_weight_renorm", action="store_true", default=False,
                   help="Renormalize recon_weight so train-mask mean ~1")
    p.add_argument("--no_recon_weight_renorm", dest="recon_weight_renorm",
                   action="store_false", help="Disable recon_weight renorm")
    p.add_argument("--recon_weight_clip_max", type=float, default=1.0,
                   help="Clamp recon_weight max (<=0 to disable)")

    p.add_argument("--conf_raw_min", type=float, default=1.0)
    p.add_argument("--conf_raw_max", type=float, default=8.0)
    p.add_argument("--conf_auto_norm", action="store_true", default=True)
    p.add_argument("--no_conf_auto_norm", dest="conf_auto_norm",
                   action="store_false", help="Disable confidence auto normalization")
    p.add_argument("--conf_use_quantile", action="store_true", default=True)
    p.add_argument("--no_conf_use_quantile", dest="conf_use_quantile",
                   action="store_false", help="Disable quantile scaling for confidence")
    p.add_argument("--conf_qlo", type=float, default=0.05)
    p.add_argument("--conf_qhi", type=float, default=0.95)

    p.add_argument("--conf_sup_use_quantile", action="store_true", default=False,
                   help="conf supervision 用 quantile 归一化（默认关，避免每 batch 抖动）")
    p.add_argument("--conf_sup_gamma", type=float, default=1.0,
                   help="对 conf target 做幂次（>1 更强调高置信；<1 更平滑）")
    p.add_argument("--conf_floor_thr", type=float,
                   default=0.05, help="pred_conf 的最低值软约束阈值")
    p.add_argument("--conf_floor_w", type=float,
                   default=0.1, help="pred_conf floor 约束权重")

    p.add_argument("--debug_train_every", type=int, default=200)
    p.add_argument("--nan_check_every", type=int, default=200,
                   help="Check NaN/Inf on key tensors every N steps")
    p.add_argument("--debug_val_every_epoch",
                   action="store_true", default=True)
    p.add_argument("--debug_fixed_batch", action="store_true", default=True,
                   help="Use a fixed cached batch for train debug snapshots")
    p.add_argument("--no_debug_fixed_batch", dest="debug_fixed_batch",
                   action="store_false", help="Disable fixed debug batch")
    p.add_argument("--debug_fixed_index", type=int, default=0,
                   help="Index within train_dataset for fixed debug snapshot")

    p.add_argument("--vis_conf_min", type=float, default=0.0)
    p.add_argument("--vis_conf_max", type=float, default=1.0)
    p.add_argument("--vis_depth_min", type=float, default=0.0)
    p.add_argument("--vis_depth_max", type=float, default=5.0)
    p.add_argument("--vis_mask_min", type=float, default=0.0)
    p.add_argument("--vis_mask_max", type=float, default=1.0)
    p.add_argument("--vis_weight_min", type=float, default=0.0)
    p.add_argument("--vis_weight_max", type=float, default=1.0)

    # Debug viz: 是否把 *_cat_*.png 进一步切成三栏（p0/p1/p2）方便对比
    p.add_argument("--split_cat_panels", action="store_true", default=True,
                   help="保存 *_cat_*.png 时额外输出 *_p0/_p1/_p2 三栏切片")
    p.add_argument("--no_split_cat_panels", dest="split_cat_panels", action="store_false",
                   help="Disable split cat panels in debug viz")

    p.add_argument("--min_improve", type=float, default=1e-4)
    p.add_argument("--early_stop", type=int, default=0,
                   help="0=disable; otherwise patience epochs")

    return p.parse_args()


def _normalize_seq_names(seq_names):
    if seq_names is None:
        return []
    if isinstance(seq_names, str):
        s = seq_names.strip()
        if not s:
            return []
        return [p for p in re.split(r"[,\s]+", s) if p]
    if isinstance(seq_names, (list, tuple)):
        out = []
        for item in seq_names:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                out.extend([p for p in re.split(r"[,\s]+", s) if p])
            else:
                out.append(str(item))
        return out
    return [str(seq_names)]


def _best_is_higher(best_by: str) -> bool:
    key = str(best_by).lower()
    return key in ("raw_psnr", "ema_psnr", "raw_ssim", "ema_ssim", "psnr_fg")


def _best_init(best_by: str) -> float:
    return -float("inf") if _best_is_higher(best_by) else float("inf")


def _select_best_metric(
    best_by: str,
    raw_val: float,
    raw_psnr: float,
    raw_ssim: float,
    ema_val: Optional[float],
    ema_psnr: Optional[float],
    ema_ssim: Optional[float],
    raw_psnr_fg: Optional[float] = None,
) -> Tuple[float, bool, str]:
    key = str(best_by).lower()
    if key in ("ema", "ema_l1") and ema_val is not None:
        return float(ema_val), False, "ema_l1"
    if key in ("raw", "raw_l1"):
        return float(raw_val), False, "raw_l1"
    if key == "ema_psnr":
        if ema_psnr is not None:
            return float(ema_psnr), True, "ema_psnr"
        return float(raw_psnr), True, "raw_psnr"
    if key == "raw_psnr":
        return float(raw_psnr), True, "raw_psnr"
    if key == "ema_ssim":
        if ema_ssim is not None:
            return float(ema_ssim), True, "ema_ssim"
        return float(raw_ssim), True, "raw_ssim"
    if key == "raw_ssim":
        return float(raw_ssim), True, "raw_ssim"
    if key == "psnr_fg":
        if raw_psnr_fg is not None:
            return float(raw_psnr_fg), True, "psnr_fg"
        return float(raw_psnr), True, "raw_psnr"
    if ema_val is not None:
        return float(ema_val), False, "ema_l1"
    return float(raw_val), False, "raw_l1"


def main():
    global args
    args = parse_args()
    args.use_view_cond = bool(args.use_view_cond)
    args.finetune_view_only = bool(args.finetune_view_only)
    if args.ema_start_after_warmup:
        args.ema_start_step = max(int(args.ema_start_step), int(args.warmup_steps))
    args.ema_start_step = max(0, int(args.ema_start_step))
    setup_torch_stability(tf32=args.tf32, cudnn_benchmark=args.cudnn_benchmark)
    seed_everything(args.seed)

    seq_names = _normalize_seq_names(args.seq_names)
    if not seq_names:
        raise ValueError("seq_names 为空。示例：--seq_names CoreView_390")
    args.seq_names = ",".join(seq_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device == "cuda")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    only_steps = _parse_only_steps_env()
    if only_steps:
        print(f"[info] ONLY_STEPS (train debug pack) = {sorted(only_steps)}")

    vis_ranges = build_vis_ranges(args)

    holdout_view_ids = _parse_int_list(args.holdout_view_ids)
    holdout_view_names = _parse_str_list(args.holdout_view_names)
    if holdout_view_names:
        holdout_view_ids = []

    ini_logger = IniLogger(os.path.join(args.log_dir, "run_log.ini"))
    ini_logger.log("meta", {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seq_names": ",".join(seq_names),
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "bg_weight": args.bg_weight,
        "conf_use_quantile(mask)": args.conf_use_quantile,
        "conf_qlo": args.conf_qlo,
        "conf_qhi": args.conf_qhi,
        "conf_sup_use_quantile": args.conf_sup_use_quantile,
        "conf_sup_gamma": args.conf_sup_gamma,
        "train_mask_mode": args.train_mask_mode,
        "recon_mask_mode": args.recon_mask_mode,
        "split_mode": args.split_mode,
        "holdout_view_ids": ",".join(str(v) for v in holdout_view_ids),
        "holdout_view_names": ",".join(holdout_view_names),
        "fg_drop_ground": args.fg_drop_ground,
        "fg_ground_axis": args.fg_ground_axis,
        "fg_ground_q": args.fg_ground_q,
        "fg_ground_margin": args.fg_ground_margin,
        "device": device,
        "amp": args.amp,
        "tf32": args.tf32,
        "cudnn_benchmark": args.cudnn_benchmark,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "ema_start_step": args.ema_start_step,
        "ema_start_after_warmup": args.ema_start_after_warmup,
        "ema_start_hardcopy": args.ema_start_hardcopy,
        "best_by": args.best_by,
        "use_view_cond": args.use_view_cond,
        "view_cond_mode": args.view_cond_mode,
        "split_cat_panels": args.split_cat_panels,
    })
    try:
        with open(os.path.join(args.log_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ---------------------------
    # Dataset
    # ---------------------------
    base_ds = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=args.num_src_views,
        frame_subsample=args.frame_subsample,
        split=None,
        bad_sample_policy=args.bad_sample_policy,
        white_mean_thr=args.white_mean_thr,
        white_std_thr=args.white_std_thr,
        report_bad_samples=args.report_bad_samples,
        bad_sample_max_retry=args.bad_sample_max_retry,
    )
    train_dataset = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=args.num_src_views,
        frame_subsample=args.frame_subsample,
        split="train",
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        tgt_view_ids_exclude=(holdout_view_ids if holdout_view_ids else None),
        tgt_view_names_exclude=(holdout_view_names if holdout_view_names else None),
        deterministic_views=False,
        bad_sample_policy=args.bad_sample_policy,
        white_mean_thr=args.white_mean_thr,
        white_std_thr=args.white_std_thr,
        report_bad_samples=args.report_bad_samples,
        bad_sample_max_retry=args.bad_sample_max_retry,
    )
    val_dataset = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=args.num_src_views,
        frame_subsample=args.frame_subsample,
        split="val",
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        tgt_view_ids=(holdout_view_ids if holdout_view_ids else None),
        tgt_view_names=(holdout_view_names if holdout_view_names else None),
        deterministic_views=True,
        bad_sample_policy=args.bad_sample_policy,
        white_mean_thr=args.white_mean_thr,
        white_std_thr=args.white_std_thr,
        report_bad_samples=args.report_bad_samples,
        bad_sample_max_retry=args.bad_sample_max_retry,
    )
    if args.use_view_cond and args.num_views <= 0:
        args.num_views = int(getattr(base_ds, "num_views", 0))
        if args.num_views <= 0:
            raise ValueError(
                "Cannot auto-infer num_views from dataset. Please set --num_views explicitly."
            )
    if args.use_view_cond:
        print(f"[info] num_views = {args.num_views}")

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_train,
        pin_memory=pin,
        drop_last=True,
    )
    if args.num_workers_train > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers_val,
        pin_memory=pin,
    )

    print(f"[info] total frames(all) = {len(base_ds)}")
    print(
        f"[info] total samples = {len(train_dataset) + len(val_dataset)} (train = {len(train_dataset)}, val = {len(val_dataset)})")
    print(f"[info] num_train_batches per epoch = {len(train_loader)}")

    try:
        b0 = next(iter(train_loader))
        if isinstance(b0, dict):
            print(f"[debug] batch keys = {list(b0.keys())}")
            ini_logger.log(
                "batch_keys", {"keys": ",".join([str(k) for k in b0.keys()])})
            assert_batch_shapes(b0, where="train/boot")
    except Exception as e:
        print(f"[fatal] batch sanity check failed: {e}")
        raise

    try:
        save_repro_pack(os.path.join(args.log_dir, "repro"), args, batch=b0)
    except Exception:
        pass

    fixed_debug_batch = None
    if bool(getattr(args, "debug_fixed_batch", False)):
        try:
            sample = _get_dataset_item(
                train_dataset, int(args.debug_fixed_index))
            try:
                batch1 = default_collate([sample])
            except Exception:
                batch1 = ensure_batch_dims(sample)
            fixed_debug_batch = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in batch1.items()
            }
            print(
                f"[info] fixed_debug_batch ready (index={int(args.debug_fixed_index)})")
        except Exception as e:
            print(f"[warn] failed to prepare fixed_debug_batch: {e}")
            fixed_debug_batch = None

    # ---------------------------
    # Model
    # ---------------------------
    conf_bias_init = resolve_conf_bias_init(args.conf_bias_init)
    model = GeomViewDecoderAblation(
        ref_mode=args.ref_mode,
        use_conf_gate=args.use_conf_gate,
        use_tone=args.use_tone,
        init_alpha=args.init_alpha,
        use_view_cond=args.use_view_cond,
        num_views=args.num_views,
        view_dim=args.view_dim,
        view_affine_strength=args.view_affine_strength,
        view_cond_mode=args.view_cond_mode,
        rgb_sigmoid_temp=float(args.rgb_sigmoid_temp),
        conf_sigmoid_temp=float(args.conf_sigmoid_temp),
        split_conf_head=bool(args.split_conf_head),
        conf_gate_detach=bool(args.conf_gate_detach),
        conf_gate_floor=float(args.conf_gate_floor),
        conf_bias_init=conf_bias_init,
        logit_clip=float(args.logit_clip),
        use_depth_head=bool(args.use_depth_head),
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[info] torch.compile enabled.")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        try:
            ema_model.load_state_dict(model.state_dict(), strict=True)
            print("[info] ema initialized from model (hard copy, strict=True).")
        except Exception as e:
            print(f"[warn] ema strict load failed: {e}")
            ema_model.load_state_dict(model.state_dict(), strict=False)
            print("[info] ema initialized from model (hard copy, strict=False).")
        ema_sanity_check(model, ema_model)

    percep_loss_fn = VGGPerceptualLoss(slice_to=16).to(device)

    if args.finetune_view_only:
        for p in model.parameters():
            p.requires_grad = False
        view_keys = ("view_affine", "view_emb", "view_embed", "view_to_gb", "view_to_rgb", "rgb_affine", "view_delta")
        for name, p in model.named_parameters():
            if any(k in name for k in view_keys):
                p.requires_grad = True
        train_params = [p for p in model.parameters() if p.requires_grad]
        if len(train_params) == 0:
            print("[warn] finetune_view_only set but no view params found.")
            for p in model.parameters():
                p.requires_grad = True
            train_params = [p for p in model.parameters() if p.requires_grad]
        else:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            print("[info] finetune_view_only trainables:")
            for n in trainable:
                print(f"  {n}")
            if len(trainable) == 0:
                raise RuntimeError("No trainable params in finetune_view_only.")
    else:
        train_params = [p for p in model.parameters() if p.requires_grad]

    conf_head_lr_mult = float(getattr(args, "conf_head_lr_mult", 1.0))
    if conf_head_lr_mult != 1.0:
        head_params = []
        head_names = []
        base_params = []
        base_names = []
        head_key = "core.out_conf" if bool(
            getattr(model.core, "split_conf_head", False)) else "core.out_conv"
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if head_key in name:
                head_params.append(p)
                head_names.append(name)
            else:
                base_params.append(p)
                base_names.append(name)
        if head_params and base_params:
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": args.lr},
                    {"params": head_params, "lr": args.lr *
                        conf_head_lr_mult},
                ],
                weight_decay=args.weight_decay,
            )
            print(
                f"[info] conf_head_lr_mult={conf_head_lr_mult:.3f} "
                f"head_key={head_key} base={len(base_params)} head={len(head_params)}"
            )
            if (not bool(getattr(model.core, "split_conf_head", False))):
                print(
                    "[warn] split_conf_head=0: conf_head_lr_mult applies to core.out_conv (RGB+conf). "
                    "Use --split_conf_head to target conf-only head."
                )
            if bool(getattr(model.core, "split_conf_head", False)) and not head_params:
                print(
                    "[warn] split_conf_head=1 but no params matched core.out_conf; "
                    "check module naming or split_conf_head wiring."
                )
            if args.print_param_groups:
                def _preview(names, limit=30):
                    if len(names) <= limit:
                        return names
                    return names[:limit] + [f"... (+{len(names)-limit} more)"]
                overlap = set(head_names) & set(base_names)
                if overlap:
                    print(f"[warn] param group overlap detected: {sorted(list(overlap))[:5]}")
                print("[debug] head params:")
                for n in _preview(head_names):
                    print(f"  {n}")
                print("[debug] base params:")
                for n in _preview(base_names):
                    print(f"  {n}")
        else:
            optimizer = torch.optim.AdamW(
                train_params, lr=args.lr, weight_decay=args.weight_decay)
            if bool(getattr(model.core, "split_conf_head", False)) and not head_params:
                print(
                    "[warn] split_conf_head=1 but no params matched core.out_conf; "
                    "falling back to single param group."
                )
    else:
        optimizer = torch.optim.AdamW(
            train_params, lr=args.lr, weight_decay=args.weight_decay)

    log_param_groups(optimizer, model)

    scheduler = None
    total_update_steps = max(
        1, (args.epochs * len(train_loader)) // max(1, args.accum_steps))
    if args.lr_schedule == "cosine":
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_update_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        plateau_mode = "max" if _best_is_higher(args.best_by) else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, factor=args.plateau_factor, patience=args.plateau_patience
        )

    scaler = make_grad_scaler(
        device=device, enabled=(args.amp and device == "cuda"))

    best_val = _best_init(args.best_by)
    global_step = 0
    start_epoch = 0
    epochs_no_improve = 0

    # ---------------------------
    # Resume
    # ---------------------------
    if args.resume and os.path.isfile(args.resume):
        ck = load_checkpoint(args.resume, device=device)
        if "model" in ck and isinstance(ck["model"], dict):
            missing, unexpected = model.load_state_dict(
                ck["model"], strict=False)
            print(
                f"[info] resumed model from {args.resume} (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
        if args.use_ema and ema_model is not None and "ema" in ck and isinstance(ck["ema"], dict):
            ema_model.load_state_dict(ck["ema"], strict=False)
            print("[info] resumed ema.")
        skip_optim_resume = bool(args.finetune_view_only) or bool(args.reset_optim)
        if skip_optim_resume:
            reason = "finetune_view_only=1" if args.finetune_view_only else "reset_optim=1"
            print(
                f"[info] {reason} => skip optimizer/scheduler/scaler resume.")
        if not skip_optim_resume:
            if "optimizer" in ck:
                try:
                    optimizer.load_state_dict(ck["optimizer"])
                    print("[info] resumed optimizer.")
                except Exception as e:
                    print(f"[warn] resume optimizer failed: {e}")
            if "scaler" in ck:
                try:
                    scaler.load_state_dict(ck["scaler"])
                    print("[info] resumed scaler.")
                except Exception as e:
                    print(f"[warn] resume scaler failed: {e}")
            if "scheduler" in ck and args.lr_schedule == "cosine":
                try:
                    scheduler.load_state_dict(ck["scheduler"])
                    print("[info] resumed scheduler.")
                except Exception as e:
                    print(f"[warn] resume scheduler failed: {e}")
        if args.finetune_view_only:
            print(
                "[info] finetune_view_only=1 => reset epoch/global_step/best_val/no_improve.")
            start_epoch = 0
            global_step = 0
            best_val = _best_init(args.best_by)
            epochs_no_improve = 0
        else:
            start_epoch = int(ck.get("epoch", 0))
            global_step = int(ck.get("global_step", 0))
            ck_best_by = ck.get("best_by", None)
            if ck_best_by is not None and str(ck_best_by) != str(args.best_by):
                print(
                    f"[warn] best_by changed ({ck_best_by} -> {args.best_by}); reset best_val.")
                best_val = _best_init(args.best_by)
            else:
                best_val = float(ck.get("best_val", best_val))
            epochs_no_improve = int(ck.get("epochs_no_improve", 0))
            print(
                f"[info] resume meta: epoch={start_epoch} global_step={global_step} best_val={best_val:.6f} no_improve={epochs_no_improve}")

    ema_start_step = int(args.ema_start_step)
    if args.use_ema and ema_model is not None and ema_start_step > 0:
        ema_started = (global_step >= ema_start_step)
        print(
            f"[info] EMA start_step={ema_start_step} hardcopy={bool(args.ema_start_hardcopy)} "
            f"(started={ema_started})"
        )
    else:
        ema_started = True

    def run_val(
        eval_model: nn.Module,
        *,
        save_debug: bool,
        tag: str,
        collect_by_view: bool = False,
        conf_gate_strength_override: Optional[float] = None,
    ):
        eval_model.eval()
        val_sum = 0.0
        val_cnt = 0
        m_psnr = 0.0
        m_ssim = 0.0
        m_mse_full = 0.0
        m_mse_masked = 0.0
        m_psnr_masked = 0.0
        fg_num = 0.0
        fg_den = 0.0
        bg_num = 0.0
        bg_den = 0.0
        by_view = {}

        def _acc_view(vid, l1v, ps):
            v = int(vid)
            entry = by_view.get(v)
            if entry is None:
                entry = {"sum_l1": 0.0, "sum_psnr": 0.0, "n": 0}
                by_view[v] = entry
            entry["sum_l1"] += float(l1v)
            entry["sum_psnr"] += float(ps)
            entry["n"] += 1

        with torch.no_grad():
            for it, batch in enumerate(val_loader):
                assert_batch_shapes(batch, where=f"val/e{epoch:03d}/it{it:04d}")
                src_imgs = batch["src_imgs"].to(device, non_blocking=True)
                src_depth = batch["src_depth"].to(device, non_blocking=True)
                src_depth_conf = batch["src_depth_conf"].to(
                    device, non_blocking=True)
                src_depth_conf = apply_valid_to_depth_conf(
                    src_depth_conf, src_depth)
                src_pointmap = batch["src_pointmap"].to(
                    device, non_blocking=True)
                tgt_img = batch["tgt_img"].to(device, non_blocking=True)
                tgt_vid = batch.get("tgt_vid", None)
                if tgt_vid is not None:
                    tgt_vid = tgt_vid.to(device, non_blocking=True)
                src_vids = batch.get("src_vids", None)
                if src_vids is not None:
                    src_vids = src_vids.to(device, non_blocking=True)

                with autocast_ctx(device=device, enabled=(args.amp and device == "cuda")):
                    conf_gate_strength = conf_gate_strength_override
                    if conf_gate_strength is None:
                        conf_gate_strength = _conf_gate_strength(global_step)
                    conf_weight_strength = float(_schedule_step(global_step)["conf_weight_strength"])
                    pred_rgb, pred_conf, aux_pred = eval_model(
                        src_imgs,
                        src_depth,
                        src_depth_conf,
                        src_pointmap,
                        tgt_vid=tgt_vid,
                        src_vids=src_vids,
                        return_aux=True,
                        use_conf_gate_override=bool(args.use_conf_gate),
                        conf_gate_strength=conf_gate_strength,
                    )
                    pred_conf_safe, pred_conf_mode = normalize_pred_conf(
                        pred_conf)

                    H, W = pred_rgb.shape[-2:]
                    train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf_t, aux_masks = build_masks_from_batch(
                        batch=batch,
                        pred_hw=(H, W),
                        device=device,
                        conf_thr=args.conf_thr,
                        conf_temp=args.conf_temp,
                        train_min_cover=args.train_min_cover,
                        fg_thr=args.fg_thr,
                        fg_min_cover=args.fg_min_cover,
                        fg_dilate_k=args.fg_dilate_k,
                        fg_keep_largest_cc=args.fg_keep_largest_cc,
                        fg_lcc_min_pixels=args.fg_lcc_min_pixels,
                        fg_drop_ground=args.fg_drop_ground,
                        fg_ground_axis=args.fg_ground_axis,
                        fg_ground_q=args.fg_ground_q,
                        fg_ground_margin=args.fg_ground_margin,
                        fg_ground_min_points=args.fg_ground_min_points,
                        valid_min_cover=args.valid_min_cover,
                        valid_dilate_k=args.valid_dilate_k,
                        valid_k_max=args.valid_k_max,
                        bg_weight=args.bg_weight,
                        conf_raw_min=args.conf_raw_min,
                        conf_raw_max=args.conf_raw_max,
                        conf_auto_norm=args.conf_auto_norm,
                        conf_use_quantile=args.conf_use_quantile,
                        conf_qlo=args.conf_qlo,
                        conf_qhi=args.conf_qhi,
                        use_conf_in_train_mask=args.use_conf_loss_gate,
                        train_mask_mode=args.train_mask_mode,
                        pred_conf_gate=pred_conf,
                        use_conf_gate=bool(args.use_conf_gate),
                        conf_gate_detach=args.conf_gate_detach,
                        conf_weight_detach=args.conf_weight_detach,
                        conf_gate_floor=args.conf_gate_floor,
                        conf_gate_gamma=args.conf_gate_gamma,
                        conf_gate_strength=conf_gate_strength,
                        conf_weight_strength=conf_weight_strength,
                        conf_weight_min=args.conf_weight_min,
                        recon_gate_floor=args.recon_gate_floor,
                        recon_mask_mode=args.recon_mask_mode,
                        recon_weight_renorm=args.recon_weight_renorm,
                        recon_weight_clip_max=args.recon_weight_clip_max,
                    )
                    l1w = masked_l1(pred_rgb, tgt_img, recon_weight)

                bs = tgt_img.size(0)
                val_sum += float(l1w.item()) * bs
                val_cnt += bs

                pred_f = pred_rgb.float().clamp(0, 1)
                tgt_f = tgt_img.float().clamp(0, 1)
                mse_full = F.mse_loss(pred_f, tgt_f, reduction="mean")
                mse_masked = masked_mse(pred_f, tgt_f, recon_weight)
                psnr_full = psnr(pred_f, tgt_f)
                psnr_masked = 10.0 * torch.log10(1.0 / (mse_masked.clamp_min(1e-8)))
                m_psnr += float(psnr_full.item()) * bs
                m_ssim += float(ssim(pred_f, tgt_f).item()) * bs
                m_mse_full += float(mse_full.item()) * bs
                m_mse_masked += float(mse_masked.item()) * bs
                m_psnr_masked += float(psnr_masked.item()) * bs

                tgt_fg = batch.get("tgt_fg", None)
                if torch.is_tensor(tgt_fg):
                    tgt_fg = tgt_fg.to(device, non_blocking=True)
                    if tgt_fg.dim() == 3:
                        tgt_fg = tgt_fg.unsqueeze(1)
                    fg = (tgt_fg > 0.5).float()
                    bg = 1.0 - fg

                    m = fg.expand_as(pred_f)
                    fg_num_b = float((m * (pred_f - tgt_f) ** 2).sum().item())
                    fg_den_b = float(m.sum().item())
                    if fg_den_b >= 1.0:
                        fg_num += fg_num_b
                        fg_den += fg_den_b

                    m = bg.expand_as(pred_f)
                    bg_num_b = float((m * (pred_f - tgt_f) ** 2).sum().item())
                    bg_den_b = float(m.sum().item())
                    if bg_den_b >= 1.0:
                        bg_num += bg_num_b
                        bg_den += bg_den_b

                if collect_by_view and tgt_vid is not None:
                    tv = tgt_vid.view(-1)
                    for b in range(int(tv.numel())):
                        l1_b = masked_l1(
                            pred_rgb[b:b + 1], tgt_img[b:b + 1], recon_weight[b:b + 1]
                        ).item()
                        psnr_b = psnr(
                            pred_f[b:b + 1], tgt_f[b:b + 1]
                        ).item()
                        _acc_view(tv[b].item(), l1_b, psnr_b)

                if save_debug and args.debug_val_every_epoch and it == 0:
                    stat_key = f"val_epoch0_stats_{epoch:03d}" if tag == "raw" else f"val_epoch0_stats_{tag}_{epoch:03d}"
                    st_conf = tensor_stats_map(
                        pred_conf_safe, mask=valid_mask, prefix="val_pred_conf_")
                    st_w = tensor_stats_map(
                        recon_weight,  mask=valid_mask, prefix="val_recon_w_")
                    st_tm = tensor_stats_map(
                        train_mask,    mask=None,      prefix="val_train_mask_")
                    st_vm = tensor_stats_map(
                        valid_mask,    mask=None,      prefix="val_valid_mask_")
                    st_fg = tensor_stats_map(
                        fg_mask,       mask=None,      prefix="val_fg_mask_")

                    ini_logger.log(stat_key, {
                        "pred_conf_mode": pred_conf_mode,
                        **st_conf, **st_w, **st_tm, **st_vm, **st_fg
                    })

                    aux_dbg = {
                        "pred_conf": pred_conf_safe.detach(),
                        "tgt_depth_conf": aux_masks.get("tgt_depth_conf", None),
                        "tgt_depth_conf_raw": aux_masks.get("tgt_depth_conf_raw", None),
                        "tgt_depth": batch.get("tgt_depth", None),
                        "tgt_fg": batch.get("tgt_fg", None),
                        "fg_mask": fg_mask.detach(),
                        "train_mask": train_mask.detach(),
                        "recon_weight": recon_weight.detach(),
                        "valid_mask": valid_mask.detach(),
                    }
                    if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
                        aux_dbg["gate"] = aux_pred.get("gate")
                    if isinstance(aux_pred, dict) and aux_pred.get("pred_depth", None) is not None:
                        aux_dbg["pred_depth"] = aux_pred.get("pred_depth")
                    save_debug_pack(
                        pred_rgb, tgt_img, aux_dbg, global_step,
                        out_dir=os.path.join(args.log_dir, "val"),
                        prefix=(f"val_e{epoch:03d}" if tag == "raw" else f"val_{tag}_e{epoch:03d}"),
                        split_cat_panels=args.split_cat_panels,
                        fixed_ranges=vis_ranges
                    )

        mean_val = val_sum / max(1, val_cnt)
        mean_psnr = m_psnr / max(1, val_cnt)
        mean_ssim = m_ssim / max(1, val_cnt)
        mean_mse_full = m_mse_full / max(1, val_cnt)
        mean_mse_masked = m_mse_masked / max(1, val_cnt)
        mean_psnr_masked = m_psnr_masked / max(1, val_cnt)
        mean_mse_fg = None
        mean_psnr_fg = None
        mean_mse_bg = None
        mean_psnr_bg = None
        if fg_den >= 1.0:
            mean_mse_fg = fg_num / fg_den
            mean_psnr_fg = -10.0 * math.log10(max(mean_mse_fg, 1e-12))
        if bg_den >= 1.0:
            mean_mse_bg = bg_num / bg_den
            mean_psnr_bg = -10.0 * math.log10(max(mean_mse_bg, 1e-12))
        by_view_rows = None
        if collect_by_view and len(by_view) > 0:
            rows = []
            for vid, s in by_view.items():
                n = max(1, s["n"])
                rows.append((vid, s["sum_l1"] / n, s["sum_psnr"] / n, n))
            rows.sort(key=lambda x: x[1], reverse=True)
            by_view_rows = rows
        return (
            mean_val,
            mean_psnr,
            mean_ssim,
            mean_mse_full,
            mean_mse_masked,
            mean_psnr_masked,
            mean_mse_fg,
            mean_psnr_fg,
            mean_mse_bg,
            mean_psnr_bg,
            by_view_rows,
        )

    # ---------------------------
    # Train
    # ---------------------------
    accum_steps = max(1, int(args.accum_steps))

    def _schedule_step(step: int) -> Dict[str, float]:
        skip_gate = 0.0
        if bool(args.use_conf_gate):
            skip_gate = schedule_value(
                step,
                warmup=int(getattr(args, "conf_gate_warmup", 0)),
                ramp=int(getattr(args, "conf_gate_ramp", 0)),
                mode=str(getattr(args, "conf_gate_ramp_mode", "smoothstep")),
                k=float(getattr(args, "conf_gate_ramp_k", 5.0)),
                vmin=float(getattr(args, "conf_gate_min", 0.0)),
                vmax=1.0,
            )
        def _pick_opt(value, fallback):
            return fallback if value is None else value

        conf_weight_strength = 0.0
        if bool(getattr(args, "use_conf_loss_gate", True)):
            conf_weight_strength = schedule_value(
                step,
                warmup=int(_pick_opt(getattr(args, "conf_weight_warmup", None),
                                    getattr(args, "conf_gate_warmup", 0))),
                ramp=int(_pick_opt(getattr(args, "conf_weight_ramp", None),
                                  getattr(args, "conf_gate_ramp", 0))),
                mode=str(_pick_opt(getattr(args, "conf_weight_ramp_mode", None),
                                   getattr(args, "conf_gate_ramp_mode", "smoothstep"))),
                k=float(_pick_opt(getattr(args, "conf_weight_ramp_k", None),
                                  getattr(args, "conf_gate_ramp_k", 5.0))),
                vmin=float(_pick_opt(getattr(args, "conf_weight_min", None), 0.0)),
                vmax=1.0,
            )
        photo_ratio = schedule_value(
            step,
            warmup=int(getattr(args, "photo_warmup", 0)),
            ramp=int(getattr(args, "photo_ramp", 0)),
            mode=str(getattr(args, "photo_ramp_mode", "smoothstep")),
            k=float(getattr(args, "photo_ramp_k", 5.0)),
            vmin=float(getattr(args, "photo_min_ratio", 0.05)),
            vmax=1.0,
        )
        conf_ratio = schedule_value(
            step,
            warmup=int(getattr(args, "conf_loss_warmup", 0)),
            ramp=int(getattr(args, "conf_loss_ramp", 0)),
            mode=str(getattr(args, "conf_loss_ramp_mode", "smoothstep")),
            k=float(getattr(args, "conf_loss_ramp_k", 5.0)),
            vmin=float(getattr(args, "conf_loss_min_ratio", 0.05)),
            vmax=1.0,
        )
        depth_ratio = schedule_value(
            step,
            warmup=int(getattr(args, "depth_loss_warmup", 0)),
            ramp=int(getattr(args, "depth_loss_ramp", 0)),
            mode=str(getattr(args, "depth_loss_ramp_mode", "smoothstep")),
            k=float(getattr(args, "depth_loss_ramp_k", 5.0)),
            vmin=float(getattr(args, "depth_loss_min_ratio", 0.05)),
            vmax=1.0,
        )
        return {
            "skip_gate_strength": float(skip_gate),
            "conf_gate_strength": float(skip_gate),
            "conf_weight_strength": float(conf_weight_strength),
            "photo_weight": float(args.lambda_photo) * float(photo_ratio),
            "conf_weight": float(args.lambda_conf) * float(conf_ratio),
            "depth_weight": float(args.lambda_depth) * float(depth_ratio),
        }

    def _conf_gate_strength(step: int) -> float:
        return float(_schedule_step(step)["skip_gate_strength"])

    def _debug_forward_on_batch(batch_cpu: Optional[Dict[str, Any]],
                                conf_gate_strength: float):
        if batch_cpu is None:
            return None
        conf_weight_strength = float(_schedule_step(global_step)["conf_weight_strength"])
        b = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch_cpu.items()
        }
        src_imgs = b["src_imgs"]
        src_depth = b["src_depth"]
        src_depth_conf = b["src_depth_conf"]
        src_depth_conf = apply_valid_to_depth_conf(
            src_depth_conf, src_depth)
        src_pointmap = b["src_pointmap"]
        tgt_img_dbg = b["tgt_img"]
        tgt_vid_dbg = b.get("tgt_vid", None)
        if torch.is_tensor(tgt_vid_dbg):
            tgt_vid_dbg = tgt_vid_dbg.to(device, non_blocking=True)
        src_vids_dbg = b.get("src_vids", None)
        if torch.is_tensor(src_vids_dbg):
            src_vids_dbg = src_vids_dbg.to(device, non_blocking=True)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred_rgb_dbg, pred_conf_dbg, aux_pred_dbg = model(
                src_imgs,
                src_depth,
                src_depth_conf,
                src_pointmap,
                tgt_vid=tgt_vid_dbg,
                src_vids=src_vids_dbg,
                return_aux=True,
                use_conf_gate_override=bool(args.use_conf_gate),
                conf_gate_strength=float(conf_gate_strength),
            )
            H, W = pred_rgb_dbg.shape[-2:]
            train_mask_dbg, valid_mask_dbg, fg_mask_dbg, recon_weight_dbg, tgt_depth_conf_dbg, aux_masks_dbg = build_masks_from_batch(
                batch=b,
                pred_hw=(H, W),
                device=device,
                conf_thr=args.conf_thr,
                conf_temp=args.conf_temp,
                train_min_cover=args.train_min_cover,
                fg_thr=args.fg_thr,
                fg_min_cover=args.fg_min_cover,
                fg_dilate_k=args.fg_dilate_k,
                fg_keep_largest_cc=args.fg_keep_largest_cc,
                fg_lcc_min_pixels=args.fg_lcc_min_pixels,
                fg_drop_ground=args.fg_drop_ground,
                fg_ground_axis=args.fg_ground_axis,
                fg_ground_q=args.fg_ground_q,
                fg_ground_margin=args.fg_ground_margin,
                fg_ground_min_points=args.fg_ground_min_points,
                valid_min_cover=args.valid_min_cover,
                valid_dilate_k=args.valid_dilate_k,
                valid_k_max=args.valid_k_max,
                bg_weight=args.bg_weight,
                conf_raw_min=args.conf_raw_min,
                conf_raw_max=args.conf_raw_max,
                conf_auto_norm=args.conf_auto_norm,
                conf_use_quantile=args.conf_use_quantile,
                conf_qlo=args.conf_qlo,
                conf_qhi=args.conf_qhi,
                use_conf_in_train_mask=args.use_conf_loss_gate,
                train_mask_mode=args.train_mask_mode,
                pred_conf_gate=pred_conf_dbg,
                use_conf_gate=bool(args.use_conf_gate),
                conf_gate_detach=args.conf_gate_detach,
                conf_weight_detach=args.conf_weight_detach,
                conf_gate_floor=args.conf_gate_floor,
                conf_gate_gamma=args.conf_gate_gamma,
                conf_gate_strength=float(conf_gate_strength),
                conf_weight_strength=float(conf_weight_strength),
                conf_weight_min=args.conf_weight_min,
                recon_gate_floor=args.recon_gate_floor,
                recon_mask_mode=args.recon_mask_mode,
                recon_weight_renorm=args.recon_weight_renorm,
                recon_weight_clip_max=args.recon_weight_clip_max,
            )
            pred_conf_safe_dbg, _ = normalize_pred_conf(pred_conf_dbg)
            aux_dbg = {
                "pred_conf": pred_conf_safe_dbg.detach(),
                "tgt_depth_conf": aux_masks_dbg.get("tgt_depth_conf", None),
                "tgt_depth_conf_raw": aux_masks_dbg.get("tgt_depth_conf_raw", None),
                "tgt_depth": b.get("tgt_depth", None),
                "tgt_fg": b.get("tgt_fg", None),
                "fg_mask": fg_mask_dbg.detach(),
                "train_mask": train_mask_dbg.detach(),
                "recon_weight": recon_weight_dbg.detach(),
                "valid_mask": valid_mask_dbg.detach(),
            }
            if isinstance(aux_pred_dbg, dict) and aux_pred_dbg.get("gate", None) is not None:
                aux_dbg["gate"] = aux_pred_dbg.get("gate")
            if isinstance(aux_pred_dbg, dict) and aux_pred_dbg.get("pred_depth", None) is not None:
                aux_dbg["pred_depth"] = aux_pred_dbg.get("pred_depth")
        model.train(was_training)
        return pred_rgb_dbg, tgt_img_dbg, aux_dbg

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for inner, batch in enumerate(train_loader):
            assert_batch_shapes(batch, where=f"train/e{epoch:03d}/it{inner:04d}")
            src_imgs = batch["src_imgs"].to(device, non_blocking=True)
            src_depth = batch["src_depth"].to(device, non_blocking=True)
            src_depth_conf = batch["src_depth_conf"].to(
                device, non_blocking=True)
            src_depth_conf = apply_valid_to_depth_conf(
                src_depth_conf, src_depth)
            src_pointmap = batch["src_pointmap"].to(device, non_blocking=True)
            tgt_img = batch["tgt_img"].to(device, non_blocking=True)
            tgt_vid = batch.get("tgt_vid", None)
            if tgt_vid is not None:
                tgt_vid = tgt_vid.to(device, non_blocking=True)
            src_vids = batch.get("src_vids", None)
            if src_vids is not None:
                src_vids = src_vids.to(device, non_blocking=True)

            with autocast_ctx(device=device, enabled=(args.amp and device == "cuda")):
                sched = _schedule_step(global_step)
                conf_gate_strength = float(sched["skip_gate_strength"])
                conf_weight_strength = float(sched["conf_weight_strength"])
                pred_rgb, pred_conf, aux_pred = model(
                    src_imgs,
                    src_depth,
                    src_depth_conf,
                    src_pointmap,
                    tgt_vid=tgt_vid,
                    src_vids=src_vids,
                    return_aux=True,
                    use_conf_gate_override=bool(args.use_conf_gate),
                    conf_gate_strength=conf_gate_strength,
                )

                H, W = pred_rgb.shape[-2:]
                train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf_t, aux_masks = build_masks_from_batch(
                    batch=batch,
                    pred_hw=(H, W),
                    device=device,
                    conf_thr=args.conf_thr,
                    conf_temp=args.conf_temp,
                    train_min_cover=args.train_min_cover,
                    fg_thr=args.fg_thr,
                    fg_min_cover=args.fg_min_cover,
                    fg_dilate_k=args.fg_dilate_k,
                    fg_keep_largest_cc=args.fg_keep_largest_cc,
                    fg_lcc_min_pixels=args.fg_lcc_min_pixels,
                    fg_drop_ground=args.fg_drop_ground,
                    fg_ground_axis=args.fg_ground_axis,
                    fg_ground_q=args.fg_ground_q,
                    fg_ground_margin=args.fg_ground_margin,
                    fg_ground_min_points=args.fg_ground_min_points,
                    valid_min_cover=args.valid_min_cover,
                    valid_dilate_k=args.valid_dilate_k,
                    valid_k_max=args.valid_k_max,
                    bg_weight=args.bg_weight,
                    conf_raw_min=args.conf_raw_min,
                    conf_raw_max=args.conf_raw_max,
                    conf_auto_norm=args.conf_auto_norm,
                    conf_use_quantile=args.conf_use_quantile,
                    conf_qlo=args.conf_qlo,
                    conf_qhi=args.conf_qhi,
                    use_conf_in_train_mask=args.use_conf_loss_gate,
                    train_mask_mode=args.train_mask_mode,
                    pred_conf_gate=pred_conf,
                    use_conf_gate=bool(args.use_conf_gate),
                    conf_gate_detach=args.conf_gate_detach,
                    conf_weight_detach=args.conf_weight_detach,
                    conf_gate_floor=args.conf_gate_floor,
                    conf_gate_gamma=args.conf_gate_gamma,
                    conf_gate_strength=conf_gate_strength,
                    conf_weight_strength=conf_weight_strength,
                    conf_weight_min=args.conf_weight_min,
                    recon_gate_floor=args.recon_gate_floor,
                    recon_mask_mode=args.recon_mask_mode,
                    recon_weight_renorm=args.recon_weight_renorm,
                    recon_weight_clip_max=args.recon_weight_clip_max,
                )

                recon_weight_loss = recon_weight.detach() if torch.is_tensor(recon_weight) else recon_weight

                recon_loss = compute_photo_loss(
                    pred_rgb, tgt_img, recon_weight_loss,
                    loss_type=args.photo_loss,
                    huber_delta=args.photo_huber_delta,
                    charb_eps=args.photo_charb_eps,
                    charb_alpha=args.photo_charb_alpha,
                )

                vgg_h, vgg_w = 256, 256
                pred_small = F.interpolate(pred_rgb, size=(
                    vgg_h, vgg_w), mode="bilinear", align_corners=False)
                tgt_small = F.interpolate(tgt_img,  size=(
                    vgg_h, vgg_w), mode="bilinear", align_corners=False)
                w_small = F.interpolate(recon_weight_loss, size=(
                    vgg_h, vgg_w), mode="bilinear", align_corners=False)
                percep_loss = percep_loss_fn(pred_small, tgt_small, w_small)

                pred_conf_safe, pred_conf_mode = normalize_pred_conf(pred_conf)

                tgt_conf_up_raw = F.interpolate(
                    tgt_depth_conf_t.float(),
                    size=pred_conf_safe.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                tgt_conf_up, tgt_conf_info = normalize_conf_to_01(
                    tgt_conf_up_raw,
                    raw_min=args.conf_raw_min,
                    raw_max=args.conf_raw_max,
                    auto=args.conf_auto_norm,
                    use_quantile=args.conf_sup_use_quantile,
                    qlo=args.conf_qlo,
                    qhi=args.conf_qhi,
                    valid_mask=F.interpolate(
                        valid_mask, size=pred_conf_safe.shape[-2:], mode="nearest"),
                )
                tgt_conf_up = tgt_conf_up.clamp(0, 1)
                if abs(float(args.conf_sup_gamma) - 1.0) > 1e-6:
                    tgt_conf_up = tgt_conf_up.pow(float(args.conf_sup_gamma))

                mask_for_conf = F.interpolate(
                    train_mask, size=pred_conf_safe.shape[-2:], mode="nearest")
                conf_match = masked_l1(
                    pred_conf_safe, tgt_conf_up, mask_for_conf)
                conf_floor = F.relu(
                    float(args.conf_floor_thr) - pred_conf_safe).mean()
                conf_err_loss = torch.tensor(0.0, device=device)
                if args.conf_err_weight and args.conf_err_weight > 0:
                    err = (pred_rgb - tgt_img).abs().mean(
                        dim=1, keepdim=True).detach()
                    vm = None
                    if valid_mask is not None:
                        vm = F.interpolate(
                            valid_mask, size=err.shape[-2:], mode="nearest")
                    if fg_mask is not None:
                        fm = F.interpolate(
                            fg_mask, size=err.shape[-2:], mode="nearest")
                        vm = fm if vm is None else (vm * fm)
                    if vm is not None:
                        denom = vm.mean().clamp_min(1e-6)
                        err_mean = (err * vm).mean() / denom
                    else:
                        err_mean = err.mean()
                    err_n = err / (err_mean + 1e-6)
                    if str(args.conf_err_target).lower() == "exp":
                        t = torch.exp(-float(args.conf_err_k) * err_n)
                    else:
                        t = (1.0 - float(args.conf_err_k) * err_n).clamp(0, 1)
                    if str(args.conf_err_loss).lower() == "bce":
                        conf_err_loss = masked_bce(pred_conf_safe, t, vm)
                    else:
                        conf_err_loss = masked_l1(pred_conf_safe, t, vm)
                conf_reg = (
                    conf_match
                    + float(args.conf_floor_w) * conf_floor
                    + float(args.conf_err_weight) * conf_err_loss
                )

                mean_p, std_p = masked_mean_std(pred_rgb, recon_weight_loss)
                mean_t, std_t = masked_mean_std(tgt_img, recon_weight_loss)
                bright_loss = (mean_p - mean_t).abs().mean()
                contrast_loss = (std_p - std_t).abs().mean()

                try:
                    alpha = model.get_alpha()
                    if not torch.is_tensor(alpha):
                        alpha = torch.as_tensor(
                            alpha, device=device, dtype=torch.float32)
                except Exception:
                    alpha = torch.tensor(0.0, device=device)
                alpha_reg = alpha * alpha

                edge_loss = torch.tensor(0.0, device=device)
                if args.lambda_edge and args.lambda_edge > 0:
                    g_pred = sobel_grad_mag(pred_rgb.clamp(0, 1))
                    g_tgt = sobel_grad_mag(tgt_img.clamp(0, 1))
                    edge_loss = masked_l1(g_pred, g_tgt, recon_weight_loss)

                tv_conf = torch.tensor(0.0, device=device)
                if args.lambda_tv_conf and args.lambda_tv_conf > 0:
                    tv_conf = tv_l1(pred_conf_safe)

                ssim_loss = torch.tensor(0.0, device=device)
                if args.lambda_ssim and args.lambda_ssim > 0:
                    p2 = F.interpolate(pred_rgb, size=(
                        256, 256), mode="bilinear", align_corners=False)
                    t2 = F.interpolate(tgt_img,  size=(
                        256, 256), mode="bilinear", align_corners=False)
                    ssim_loss = 1.0 - ssim(p2, t2)

                # conf mean regularizer (anti-gaming)
                conf_mean_loss = torch.tensor(0.0, device=device)
                if args.lambda_conf_mean and args.lambda_conf_mean > 0:
                    if valid_mask is not None:
                        vm = F.interpolate(
                            valid_mask, size=pred_conf_safe.shape[-2:], mode="nearest")
                        if vm.sum().item() > 0:
                            conf_mean = pred_conf_safe[vm > 0.5].mean()
                        else:
                            conf_mean = pred_conf_safe.mean()
                    else:
                        conf_mean = pred_conf_safe.mean()
                    conf_mean_loss = (conf_mean - float(args.conf_mean_target)) ** 2

                # optional depth loss
                depth_loss = torch.tensor(0.0, device=device)
                depth_edge = torch.tensor(0.0, device=device)
                pred_depth = aux_pred.get("pred_depth", None) if isinstance(aux_pred, dict) else None
                if args.use_depth_head and torch.is_tensor(pred_depth):
                    tgt_depth = batch.get("tgt_depth", None)
                    if torch.is_tensor(tgt_depth):
                        if tgt_depth.device != pred_depth.device:
                            tgt_depth = tgt_depth.to(pred_depth.device, non_blocking=True)
                        if tgt_depth.dim() == 3:
                            tgt_depth = tgt_depth[:, None, :, :]
                        if tgt_depth.shape[-2:] != pred_depth.shape[-2:]:
                            tgt_depth = F.interpolate(
                                tgt_depth.float(), size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
                        vm = valid_mask
                        if vm is not None and vm.shape[-2:] != pred_depth.shape[-2:]:
                            vm = F.interpolate(vm, size=pred_depth.shape[-2:], mode="nearest")
                        depth_loss = compute_depth_loss(
                            pred_depth, tgt_depth.float(), vm,
                            loss_type=args.depth_loss,
                            huber_delta=args.depth_huber_delta,
                            charb_eps=args.depth_charb_eps,
                            charb_alpha=args.depth_charb_alpha,
                        )
                        if args.lambda_depth_edge and args.lambda_depth_edge > 0:
                            depth_edge = edge_aware_depth_smoothness(
                                pred_depth, tgt_img, vm, alpha=10.0)

                # lightweight multi-view consistency (optional)
                mv_loss = torch.tensor(0.0, device=device)
                if args.lambda_mv and args.lambda_mv > 0:
                    if args.mv_mode == "first":
                        ref_rgb = src_imgs[:, 0]
                    else:
                        ref_rgb = src_imgs.mean(dim=1)
                    mv_loss = compute_photo_loss(
                        pred_rgb, ref_rgb, recon_weight_loss,
                        loss_type=args.photo_loss,
                        huber_delta=args.photo_huber_delta,
                        charb_eps=args.photo_charb_eps,
                        charb_alpha=args.photo_charb_alpha,
                    )

                photo_w = float(sched["photo_weight"])
                conf_w = float(sched["conf_weight"])
                depth_w = float(sched["depth_weight"])

                loss = (
                    photo_w * recon_loss
                    + args.lambda_percep * percep_loss
                    + conf_w * conf_reg
                    + args.lambda_conf_mean * conf_mean_loss
                    + depth_w * depth_loss
                    + depth_w * args.lambda_depth_edge * depth_edge
                    + args.lambda_bright * bright_loss
                    + args.lambda_contrast * contrast_loss
                    + args.lambda_alpha_reg * alpha_reg
                    + args.lambda_edge * edge_loss
                    + args.lambda_tv_conf * tv_conf
                    + args.lambda_ssim * ssim_loss
                    + args.lambda_mv * mv_loss
                )

                loss = loss / float(accum_steps)

            if args.nan_check_every and args.nan_check_every > 0 and (global_step % int(args.nan_check_every) == 0) and ((inner + 1) % accum_steps == 0):
                bad = []
                if _has_nonfinite(loss):
                    bad.append("loss")
                if _has_nonfinite(pred_rgb):
                    bad.append("pred_rgb")
                if _has_nonfinite(pred_conf):
                    bad.append("pred_conf")
                if _has_nonfinite(recon_weight):
                    bad.append("recon_weight")
                if _has_nonfinite(conf_reg):
                    bad.append("conf_reg")
                if _has_nonfinite(depth_loss):
                    bad.append("depth_loss")
                if bad:
                    note = f"nonfinite tensors: {','.join(bad)}"
                    aux_dbg = {
                        "pred_conf": pred_conf.detach() if torch.is_tensor(pred_conf) else None,
                        "fg_mask": fg_mask.detach() if torch.is_tensor(fg_mask) else None,
                        "train_mask": train_mask.detach() if torch.is_tensor(train_mask) else None,
                        "recon_weight": recon_weight.detach() if torch.is_tensor(recon_weight) else None,
                        "valid_mask": valid_mask.detach() if torch.is_tensor(valid_mask) else None,
                    }
                    dump_nan_guard(
                        os.path.join(args.log_dir, "nan_guard", f"step{global_step:06d}"),
                        args,
                        batch,
                        pred_rgb,
                        tgt_img,
                        aux_dbg,
                        model=model,
                        note=note,
                    )
                    print(f"[warn] {note} -> dumped nan_guard and skip batch")
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        scaler.update()
                    except Exception:
                        pass
                    continue
            if not torch.isfinite(loss):
                print(
                    f"[warn] non-finite loss at epoch={epoch} inner={inner}, skip batch.")
                optimizer.zero_grad(set_to_none=True)
                try:
                    scaler.update()
                except Exception:
                    pass
                continue

            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f"[warn] backward runtime error (skip batch): {e}")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    scaler.update()
                except Exception:
                    pass
                continue

            do_step = ((inner + 1) % accum_steps == 0)

            if do_step:
                try:
                    scaler.unscale_(optimizer)
                except RuntimeError as e:
                    print(f"[warn] unscale error (skip step): {e}")
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scaler.update()
                    continue

                grad_bad = False
                total_norm = 0.0
                if args.nan_check_every and args.nan_check_every > 0 and (global_step % int(args.nan_check_every) == 0):
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            grad_bad = True
                            break
                        gn = float(p.grad.data.norm(2).item())
                        total_norm += gn * gn
                    if grad_bad:
                        note = "nonfinite gradients"
                        dump_nan_guard(
                            os.path.join(args.log_dir, "nan_guard", f"step{global_step:06d}_grad"),
                            args,
                            batch,
                            pred_rgb,
                            tgt_img,
                            aux_dbg={
                                "pred_conf": pred_conf.detach() if torch.is_tensor(pred_conf) else None,
                                "recon_weight": recon_weight.detach() if torch.is_tensor(recon_weight) else None,
                                "valid_mask": valid_mask.detach() if torch.is_tensor(valid_mask) else None,
                            },
                            model=model,
                            note=note,
                        )
                        print(f"[warn] {note} -> dumped nan_guard and skip step")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    total_norm = math.sqrt(total_norm)

                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)

                try:
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as e:
                    print(f"[warn] optimizer step error (skip): {e}")
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scaler.update()
                    continue

                optimizer.zero_grad(set_to_none=True)

                if args.lr_schedule == "cosine":
                    scheduler.step()

                global_step += 1

                if args.use_ema and ema_model is not None:
                    if (not ema_started) and (global_step >= ema_start_step):
                        ema_started = True
                        if args.ema_start_hardcopy:
                            ema_model.load_state_dict(
                                model.state_dict(), strict=False)
                            print("[info] EMA start: hard-copied model weights.")
                    if ema_started:
                        ema_update(ema_model, model, decay=args.ema_decay)

            epoch_loss += float(loss.item()) * float(accum_steps)

            if do_step and (global_step % 10 == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch:02d} step {global_step:06d}] "
                      f"loss={loss.item()*accum_steps:.4f} lr={lr_now:.6g}")

            should_debug = False
            if only_steps:
                should_debug = int(global_step) in only_steps
            else:
                should_debug = (global_step % args.debug_train_every == 0)

            if do_step and should_debug:
                with torch.no_grad():
                    pred_f = pred_rgb.float().clamp(0, 1)
                    tgt_f = tgt_img.float().clamp(0, 1)
                    splits = compute_l1_splits(
                        pred_f, tgt_f, valid_mask=valid_mask, fg_mask=fg_mask)

                    pc_min = float(pred_conf_safe.min().item())
                    pc_mean = float(pred_conf_safe.mean().item())
                    pc_max = float(pred_conf_safe.max().item())

                    st_conf = tensor_stats_map(
                        pred_conf_safe, mask=valid_mask, prefix="pred_conf_")
                    st_w = tensor_stats_map(
                        recon_weight,  mask=valid_mask, prefix="recon_w_")
                    st_tm = tensor_stats_map(
                        train_mask,    mask=None,      prefix="train_mask_")
                    st_vm = tensor_stats_map(
                        valid_mask,    mask=None,      prefix="valid_mask_")
                    st_fg = tensor_stats_map(
                        fg_mask,       mask=None,      prefix="fg_mask_")
                    st_tgtc = tensor_stats_map(
                        tgt_conf_up,   mask=valid_mask, prefix="tgt_conf01_")
                    eps = 1e-6
                    fg_cover_per = fg_mask.mean(
                        dim=(1, 2, 3)).detach().cpu()
                    valid_cover_per = valid_mask.mean(
                        dim=(1, 2, 3)).detach().cpu()
                    fg_cover_mean = float(fg_cover_per.mean().item())
                    fg_cover_std = float(fg_cover_per.std(
                        unbiased=False).item())
                    valid_cover_mean = float(valid_cover_per.mean().item())
                    valid_cover_std = float(valid_cover_per.std(
                        unbiased=False).item())
                    fg_valid_cover_per = (
                        fg_mask * valid_mask).mean(dim=(1, 2, 3)).detach().cpu()
                    fg_over_valid = fg_valid_cover_per / \
                        (valid_cover_per + eps)
                    fg_over_valid_mean = float(
                        fg_over_valid.mean().item())
                    fg_over_valid_std = float(
                        fg_over_valid.std(unbiased=False).item())

                ci = aux_masks.get("conf_info", {})
                print(
                    f"[debug step {global_step:06d}] "
                    f"recon={recon_loss.item():.4f} percep={percep_loss.item():.4f} conf_reg={conf_reg.item():.4f} conf_err={conf_err_loss.item():.4f} "
                    f"conf_mean={conf_mean_loss.item():.4f} depth={depth_loss.item():.4f} d_edge={depth_edge.item():.4f} "
                    f"mv={mv_loss.item():.4f} "
                    f"edge={edge_loss.item():.4f} tvc={tv_conf.item():.4f} ssimL={ssim_loss.item():.4f} "
                    f"bright={bright_loss.item():.4f} contrast={contrast_loss.item():.4f} "
                    f"cover_train={aux_masks['cover_train']:.4f} cover_fg={aux_masks['cover_fg']:.4f} "
                    f"cover_conf={aux_masks['cover_conf']:.4f} cover_conf_geom={aux_masks.get('cover_conf_geom', 0.0):.4f} "
                    f"cover_valid={aux_masks['cover_valid']:.4f} "
                    f"pred_conf(min/mean/max)=({pc_min:.3f}/{pc_mean:.3f}/{pc_max:.3f}) "
                    f"pred_conf_mode={pred_conf_mode} "
                    f"conf_mask_mode={aux_masks.get('conf_mask_mode', '')} "
                    f"conf_mode(depth_conf)={ci.get('mode', '')} conf_mode(sup)={tgt_conf_info.get('mode', '')}"
                )

                kv = {
                    "loss": loss.item() * accum_steps,
                    "recon": recon_loss.item(),
                    "percep": percep_loss.item(),
                    "conf_reg": conf_reg.item(),
                    "conf_err": conf_err_loss.item(),
                    "conf_mean": conf_mean_loss.item(),
                    "depth": depth_loss.item(),
                    "depth_edge": depth_edge.item(),
                    "mv": mv_loss.item(),
                    "edge": edge_loss.item(),
                    "tv_conf": tv_conf.item(),
                    "ssim_loss": ssim_loss.item(),
                    "bright": bright_loss.item(),
                    "contrast": contrast_loss.item(),
                    "photo_w": float(photo_w),
                    "conf_w": float(conf_w),
                    "depth_w": float(depth_w),
                    "cover_train": aux_masks["cover_train"],
                    "cover_fg": aux_masks["cover_fg"],
                    "cover_conf": aux_masks["cover_conf"],
                    "cover_conf_geom": aux_masks.get("cover_conf_geom", 0.0),
                    "cover_valid": aux_masks["cover_valid"],
                    "l1_full": splits["l1_full"],
                    "l1_fg": splits.get("l1_fg", splits["l1_full"]),
                    "l1_bg": splits.get("l1_bg", splits["l1_full"]),
                    "pred_conf_min": pc_min,
                    "pred_conf_mean": pc_mean,
                    "pred_conf_max": pc_max,
                    "pred_conf_mode": pred_conf_mode,
                    "source_fg_key": aux_masks["source_fg_key"],
                    "source_valid_key": aux_masks["source_valid_key"],
                    "source_depth_conf_key": aux_masks["source_depth_conf_key"],
                    "conf_mask_mode": aux_masks.get("conf_mask_mode", ""),
                    "conf_mode_mask": ci.get("mode", ""),
                    "conf_mode_sup": tgt_conf_info.get("mode", ""),
                    "lr": optimizer.param_groups[0]["lr"],
                    "conf_gate_strength": float(conf_gate_strength),
                    "conf_gate_warmup": int(args.conf_gate_warmup),
                    "conf_gate_ramp": int(args.conf_gate_ramp),
                    "conf_weight_strength": float(conf_weight_strength),
                    "conf_weight_warmup": int(args.conf_weight_warmup) if args.conf_weight_warmup is not None else -1,
                    "conf_weight_ramp": int(args.conf_weight_ramp) if args.conf_weight_ramp is not None else -1,
                    "recon_gate_floor": float(args.recon_gate_floor),
                    "fg_cover_mean": fg_cover_mean,
                    "fg_cover_std": fg_cover_std,
                    "valid_cover_mean": valid_cover_mean,
                    "valid_cover_std": valid_cover_std,
                    "fg_over_valid_mean": fg_over_valid_mean,
                    "fg_over_valid_std": fg_over_valid_std,
                }
                gate_mean = None
                if isinstance(aux_pred, dict):
                    gate_t = aux_pred.get("gate", None)
                    if torch.is_tensor(gate_t):
                        gate_mean = float(gate_t.mean().item())
                    gate_loss_t = aux_pred.get("gate_loss", None)
                    if torch.is_tensor(gate_loss_t):
                        kv["gate_loss_mean"] = float(gate_loss_t.mean().item())
                if gate_mean is not None:
                    kv["gate_mean"] = gate_mean
                if torch.is_tensor(recon_weight):
                    kv["recon_weight_mean"] = float(
                        recon_weight.mean().item())
                kv.update(st_conf)
                kv.update(st_w)
                kv.update(st_tm)
                kv.update(st_vm)
                kv.update(st_fg)
                kv.update(st_tgtc)
                if isinstance(aux_masks, dict) and "mask_stats" in aux_masks:
                    try:
                        kv.update(aux_masks["mask_stats"])
                    except Exception:
                        pass

                ini_logger.log(f"train_step_{global_step:06d}", kv)

                aux_dbg = {
                    "pred_conf": pred_conf_safe.detach(),
                    "tgt_depth_conf": aux_masks.get("tgt_depth_conf", None),
                    "tgt_depth_conf_raw": aux_masks.get("tgt_depth_conf_raw", None),
                    "tgt_depth": batch.get("tgt_depth", None),
                    "tgt_fg": batch.get("tgt_fg", None),
                    "fg_mask": fg_mask.detach(),
                    "train_mask": train_mask.detach(),
                    "recon_weight": recon_weight.detach(),
                    "valid_mask": valid_mask.detach(),
                }
                if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
                    aux_dbg["gate"] = aux_pred.get("gate")
                if isinstance(aux_pred, dict) and aux_pred.get("pred_depth", None) is not None:
                    aux_dbg["pred_depth"] = aux_pred.get("pred_depth")

                dbg_pack = None
                if fixed_debug_batch is not None:
                    try:
                        dbg_pack = _debug_forward_on_batch(
                            fixed_debug_batch, conf_gate_strength)
                    except Exception as e:
                        print(f"[warn] fixed_debug_batch failed: {e}")
                        dbg_pack = None
                if dbg_pack is not None:
                    dbg_pred, dbg_tgt, dbg_aux = dbg_pack
                else:
                    dbg_pred, dbg_tgt, dbg_aux = pred_rgb, tgt_img, aux_dbg

                save_debug_pack(
                    dbg_pred, dbg_tgt, dbg_aux, global_step,
                    out_dir=os.path.join(args.log_dir, "train"),
                    prefix="train",
                    split_cat_panels=args.split_cat_panels,
                    fixed_ranges=vis_ranges
                )

        mean_train = epoch_loss / max(1, len(train_loader))
        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] mean train loss = {mean_train:.6f} time={dt:.1f}s")

        # ---------------------------
        # Val
        # ---------------------------
        model.eval()
        val_gate_strength = _conf_gate_strength(global_step)
        raw_val, raw_psnr, raw_ssim, raw_mse_full, raw_mse_masked, raw_psnr_masked, raw_mse_fg, raw_psnr_fg, raw_mse_bg, raw_psnr_bg, raw_by_view = run_val(
            model, save_debug=True, tag="raw", collect_by_view=True,
            conf_gate_strength_override=val_gate_strength)
        ema_val, ema_psnr, ema_ssim = None, None, None
        ema_mse_full, ema_mse_masked, ema_psnr_masked = None, None, None
        ema_mse_fg, ema_psnr_fg, ema_mse_bg, ema_psnr_bg = None, None, None, None
        if args.use_ema and ema_model is not None and ema_started:
            ema_val, ema_psnr, ema_ssim, ema_mse_full, ema_mse_masked, ema_psnr_masked, ema_mse_fg, ema_psnr_fg, ema_mse_bg, ema_psnr_bg, _ = run_val(
                ema_model, save_debug=False, tag="ema", collect_by_view=False,
                conf_gate_strength_override=val_gate_strength)

        psnr_fg_s = f"{raw_psnr_fg:.2f}" if raw_psnr_fg is not None else "n/a"
        psnr_bg_s = f"{raw_psnr_bg:.2f}" if raw_psnr_bg is not None else "n/a"

        if ema_val is not None:
            print(
                f"[Epoch {epoch:02d}] val raw={raw_val:.6f} PSNR={raw_psnr:.2f} SSIM={raw_ssim:.4f} | "
                f"MSE_full={raw_mse_full:.6e} PSNR_masked={raw_psnr_masked:.2f} MSE_masked={raw_mse_masked:.6e} | "
                f"PSNR_fg={psnr_fg_s} PSNR_bg={psnr_bg_s} | "
                f"ema={ema_val:.6f} PSNR={ema_psnr:.2f} SSIM={ema_ssim:.4f}"
            )
        else:
            print(
                f"[Epoch {epoch:02d}] val raw={raw_val:.6f} PSNR={raw_psnr:.2f} SSIM={raw_ssim:.4f} | "
                f"MSE_full={raw_mse_full:.6e} PSNR_masked={raw_psnr_masked:.2f} MSE_masked={raw_mse_masked:.6e} | "
                f"PSNR_fg={psnr_fg_s} PSNR_bg={psnr_bg_s}"
            )

        ini_logger.log(f"val_epoch_{epoch:03d}", {
            "wL1_raw": raw_val,
            "PSNR_raw": raw_psnr,
            "SSIM_raw": raw_ssim,
            "MSE_full_raw": raw_mse_full,
            "MSE_masked_raw": raw_mse_masked,
            "PSNR_masked_raw": raw_psnr_masked,
            "MSE_fg_raw": (raw_mse_fg if raw_mse_fg is not None else -1.0),
            "PSNR_fg_raw": (raw_psnr_fg if raw_psnr_fg is not None else -1.0),
            "MSE_bg_raw": (raw_mse_bg if raw_mse_bg is not None else -1.0),
            "PSNR_bg_raw": (raw_psnr_bg if raw_psnr_bg is not None else -1.0),
            "wL1_ema": (ema_val if ema_val is not None else -1.0),
            "PSNR_ema": (ema_psnr if ema_psnr is not None else -1.0),
            "SSIM_ema": (ema_ssim if ema_ssim is not None else -1.0),
            "MSE_full_ema": (ema_mse_full if ema_mse_full is not None else -1.0),
            "MSE_masked_ema": (ema_mse_masked if ema_mse_masked is not None else -1.0),
            "PSNR_masked_ema": (ema_psnr_masked if ema_psnr_masked is not None else -1.0),
            "MSE_fg_ema": (ema_mse_fg if ema_mse_fg is not None else -1.0),
            "PSNR_fg_ema": (ema_psnr_fg if ema_psnr_fg is not None else -1.0),
            "MSE_bg_ema": (ema_mse_bg if ema_mse_bg is not None else -1.0),
            "PSNR_bg_ema": (ema_psnr_bg if ema_psnr_bg is not None else -1.0),
            "best_by": args.best_by,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if raw_by_view:
            print("[by_view] worst->best (vid, meanL1, meanPSNR, N):")
            for row in raw_by_view:
                print(f"  {row}")

        metric_val, higher_is_better, metric_name = _select_best_metric(
            args.best_by,
            raw_val,
            raw_psnr,
            raw_ssim,
            ema_val,
            ema_psnr,
            ema_ssim,
            raw_psnr_fg,
        )

        if args.lr_schedule == "plateau":
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(metric_val)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"[scheduler] lr reduced: {prev_lr:.6g} -> {new_lr:.6g}")

        payload = {
            "model": model.state_dict(),
            "ema": (ema_model.state_dict() if (args.use_ema and ema_model is not None) else None),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": (scheduler.state_dict() if args.lr_schedule == "cosine" else None),
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_val": best_val,
            "val_raw": raw_val,
            "val_ema": ema_val,
            "best_by": args.best_by,
            "best_metric": metric_name,
            "epochs_no_improve": epochs_no_improve,
            "args": vars(args),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        ckpt_last = os.path.join(args.ckpt_dir, "viewdec_ablation_last.pth")
        save_checkpoint(ckpt_last, payload)

        if higher_is_better:
            improved = (metric_val - best_val) > float(args.min_improve)
        else:
            improved = (best_val - metric_val) > float(args.min_improve)
        if improved:
            best_val = metric_val
            epochs_no_improve = 0
            ckpt_path = os.path.join(
                args.ckpt_dir, f"viewdec_ablation_best_epoch{epoch:02d}.pth")
            payload["best_val"] = best_val
            payload["epochs_no_improve"] = epochs_no_improve
            save_checkpoint(ckpt_path, payload)
            save_checkpoint(os.path.join(
                args.ckpt_dir, "viewdec_ablation_best.pth"), payload)
            print(
                f" -> val improved ({metric_name}={metric_val:.6f}), saved new best: {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f" -> val not improved (no_improve = {epochs_no_improve})")

        if args.early_stop and epochs_no_improve >= int(args.early_stop):
            print(
                f"[early-stop] stop: no_improve={epochs_no_improve} >= patience={args.early_stop}")
            break


if __name__ == "__main__":
    main()
