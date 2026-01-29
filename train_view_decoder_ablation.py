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
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
def setup_torch_stability(tf32: bool = True):
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

    torch.backends.cudnn.benchmark = False
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


def _as_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _ensure_4d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


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


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], eps=1e-6) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, tgt, reduction="mean")
    diff = (pred - tgt).abs() * mask
    C = float(pred.shape[1])
    denom = (mask.sum(dim=(2, 3)) * C).clamp_min(eps)  # (B,1)
    num = diff.sum(dim=(1, 2, 3), keepdim=False)       # (B,)
    return (num / denom.squeeze(1)).mean()


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
    c = conf.float()
    mn = float(c.min().item())
    mx = float(c.max().item())

    # FIX: 如果 conf 来自 uint8 0~255（或类似），先缩放到 0~1 再走后续逻辑
    # 用 >16 判定，避免把 1~8 的深度置信误判
    if auto and mx > 16.0 and mn >= -0.2:
        c = (c / 255.0).clamp(0.0, 1.0)
        mn = float(c.min().item())
        mx = float(c.max().item())

    already01 = (auto and mx <= 1.2 and mn >= -0.2)
    if already01 and (not use_quantile):
        return c.clamp(0, 1), {
            "mode": "already01",
            "conf_raw_min": 0.0,
            "conf_raw_max": 1.0,
            "mn": mn,
            "mx": mx,
        }

    if use_quantile:
        try:
            x = c
            if valid_mask is not None:
                vm = valid_mask.float()
                if vm.shape[-2:] != x.shape[-2:]:
                    vm = F.interpolate(vm, size=x.shape[-2:], mode="nearest")
                vals = x[vm > 0.5]
            else:
                vals = x.flatten()

            if vals.numel() >= 32:
                lo = torch.quantile(
                    vals, torch.tensor(qlo, device=vals.device))
                hi = torch.quantile(
                    vals, torch.tensor(qhi, device=vals.device))
                lo_f = float(lo.item())
                hi_f = float(hi.item())
                denom = max(hi_f - lo_f, eps)
                out = ((c - lo_f) / denom).clamp(0, 1)
                return out, {"mode": "quantile", "qlo": qlo, "qhi": qhi,
                             "lo": lo_f, "hi": hi_f, "mn": mn, "mx": mx}
        except Exception:
            pass

    denom = float(raw_max - raw_min)
    denom = denom if abs(denom) > eps else 1.0
    out = ((c - float(raw_min)) / denom).clamp(0, 1)
    return out, {"mode": "fixed", "raw_min": raw_min, "raw_max": raw_max, "mn": mn, "mx": mx}


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
    return pc.clamp(0.0, 1.0), "clamp01"


# ---------------------------
# Batch key picking helpers
# ---------------------------
def pick_first_tensor(batch: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in batch and batch[k] is not None:
            return k, batch[k]
    return None, None


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
    pred_conf_gate: Optional[torch.Tensor] = None,
    use_conf_gate: Optional[bool] = None,
    conf_gate_detach: Optional[bool] = None,
    conf_gate_floor: Optional[float] = None,
    conf_gate_gamma: Optional[float] = None,
    conf_gate_strength: Optional[float] = None,
    recon_gate_floor: Optional[float] = None,
    recon_weight_renorm: Optional[bool] = None,
    recon_weight_clip_max: Optional[float] = None,
):
    H, W = pred_hw

    # ---- depth conf ----
    depth_conf_key, tgt_depth_conf = pick_first_tensor(
        batch, ["tgt_depth_conf", "depth_conf_tgt",
                "tgt_conf_depth", "tgt_depthconf"]
    )
    if tgt_depth_conf is None:
        raise KeyError("batch missing tgt_depth_conf-like key.")
    tgt_depth_conf = tgt_depth_conf.to(device)

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

    # ---- fg mask ----
    fg_key, fg_raw = pick_first_tensor(
        batch, ["fg_mask", "tgt_fg_mask", "tgt_mask", "tgt_silhouette", "silhouette",
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

    if fg_keep_largest_cc:
        fg_mask0 = keep_largest_connected_component(
            fg_mask0, min_pixels=int(fg_lcc_min_pixels))

    fg_mask = ensure_min_cover_by_dilation(
        fg_mask0, float(fg_min_cover), int(fg_dilate_k), 31)

    # ---- conf soft (gate) ----
    conf_soft, conf01_full, conf_info = make_soft_mask_from_conf(
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
    conf_soft = conf_soft.clamp(0, 1) * valid_mask
    conf_gate_mode = "conf_soft"
    if not use_conf_in_train_mask:
        conf_soft = torch.ones_like(valid_mask)
        conf_gate_mode = "all_ones"

    # ---- train mask ----
    train_mask = (fg_mask * conf_soft).clamp(0, 1)
    cover_train = float(train_mask.mean().item())

    if cover_train < float(train_min_cover):
        train_mask = (fg_mask * valid_mask).clamp(0, 1)
        cover_train = float(train_mask.mean().item())
        if cover_train < float(train_min_cover):
            train_mask = valid_mask.clone()
            cover_train = float(train_mask.mean().item())

    # --- conf gate (optional) ---
    def _resolve_opt(value, name: str, default):
        if value is not None:
            return value
        if args is not None and hasattr(args, name):
            return getattr(args, name)
        return default

    use_conf_gate = bool(_resolve_opt(use_conf_gate, "use_conf_gate", False))
    conf_gate_detach = bool(_resolve_opt(
        conf_gate_detach, "conf_gate_detach", False))
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
        if conf_gate_detach:
            gate_conf = gate_conf.detach()
        if conf_gate_strength < 1.0:
            gate_conf = (1.0 - conf_gate_strength) + \
                conf_gate_strength * gate_conf
    else:
        gate_conf = train_mask.new_ones(train_mask.shape)

    # NOTE: recon weight anchors on fg_mask; bg weight is explicit & separate
    fg_mask_safe = fg_mask.clamp(0.0, 1.0)
    recon_fg = fg_mask_safe * gate_conf
    if float(bg_weight) > 0.0:
        recon_weight_raw = recon_fg + (1.0 - fg_mask_safe) * float(bg_weight)
    else:
        recon_weight_raw = recon_fg
    recon_weight = recon_weight_raw
    if recon_weight_renorm:
        renorm_mask = fg_mask_safe
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
        "tgt_depth_conf_raw": tgt_depth_conf.detach(),
        "tgt_depth_conf": conf01_full.detach(),
        "conf_info": conf_info,
        "fg_mask": fg_mask.detach(),
        "train_mask": train_mask.detach(),
        "gate_loss": gate_conf.detach(),
        "use_conf_gate_loss": bool(use_conf_gate),
        "conf_gate_gamma": float(conf_gate_gamma),
        "conf_gate_strength": float(conf_gate_strength),
        "recon_gate_floor": float(recon_gate_floor),
        "recon_weight_renorm": bool(recon_weight_renorm),
        "recon_weight_clip_max": float(recon_weight_clip_max),
        "recon_weight_raw": recon_weight_raw.detach(),
        "recon_weight": recon_weight.detach(),
        "valid_mask": valid_mask.detach(),
        "cover_train": cover_train,
        "cover_fg": float(fg_mask.mean().item()),
        "cover_conf": float(conf_soft.mean().item()),
        "cover_valid": float(valid_mask.mean().item()),
        "conf_gate_mode": conf_gate_mode,
        "source_fg_key": source_fg_key,
        "source_valid_key": source_valid_key,
        "source_depth_conf_key": depth_conf_key,
    }
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
def save_debug_pack(pred, tgt, aux, step, out_dir="debug_viewdec_ablation", prefix="train", split_cat_panels: bool = True):
    os.makedirs(out_dir, exist_ok=True)
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

    def _save1(name, x1):
        x1 = _resize_to_hw(x1, is_mask=True)
        if x1 is None:
            return
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

    for key in ["valid_mask", "fg_mask", "train_mask",
                "recon_weight_raw", "recon_weight",
                "tgt_depth_conf", "tgt_depth_conf_raw",
                "pred_conf", "gate", "gate_loss"]:
        if key in aux and isinstance(aux[key], torch.Tensor):
            _save1(key, aux[key])


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


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--zju_root", type=str,
                   default=r"F:\datasets\ZJU_MoCap\data\zju_mocap")
    p.add_argument("--seq_names", type=str, default="CoreView_390",
                   help="comma separated, e.g. CoreView_390,CoreView_392")

    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--accum_steps", type=int, default=1,
                   help="gradient accumulation steps")

    p.add_argument("--num_workers_train", type=int, default=4)
    p.add_argument("--num_workers_val", type=int, default=0)

    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--conf_head_lr_mult", type=float, default=0.1,
                   help="LR multiplier for core.out_conv (RGB+conf head). Set 1.0 to disable.")

    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--frame_subsample", type=int, default=1)
    p.add_argument("--num_src_views", type=int, default=3)

    p.add_argument("--resume", type=str, default="", help="ckpt path")
    p.add_argument("--log_dir", type=str, default="debug_viewdec_ablation")
    p.add_argument("--ckpt_dir", type=str, default="ckpt")

    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    p.add_argument("--tf32", dest="tf32", action="store_true")
    p.add_argument("--no_tf32", dest="tf32", action="store_false")
    p.set_defaults(tf32=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--compile", action="store_true", default=False)

    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", dest="use_ema",
                   action="store_false", help="Disable EMA")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--best_by", type=str, default="ema",
                   choices=["ema", "raw"])

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
    p.add_argument("--conf_gate_floor", type=float, default=0.0,
                   help="Skip gate floor (model ref skip), 0~1")
    p.add_argument("--conf_gate_gamma", type=float, default=0.5,
                   help="Loss gate gamma (>1 emphasizes high-conf, <1 flattens)")
    p.add_argument("--recon_gate_floor", type=float, default=0.2,
                   help="Soft floor for recon weight gate: w = fg * (floor + (1-floor)*conf)")
    p.add_argument("--conf_gate_warmup", type=int, default=200,
                   help="Disable pred_conf gate for first N optimizer steps (loss weighting only)")
    p.add_argument("--conf_gate_ramp", type=int, default=0,
                   help="Ramp steps after warmup to smoothly enable conf gate (0 = hard switch)")
    p.add_argument("--conf_gate_ramp_mode", type=str, default="linear",
                   choices=["linear", "cosine", "exp"],
                   help="Ramp mode for conf gate strength")
    p.add_argument("--conf_gate_ramp_k", type=float, default=5.0,
                   help="Exp ramp sharpness (only for exp mode)")
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
    p.add_argument("--conf_sigmoid_temp", type=float, default=2.0,
                   help="Conf sigmoid temperature (larger => softer)")
    p.add_argument("--split_conf_head", action="store_true", default=False,
                   help="Use separate RGB/Conf heads (conf head can use lower LR)")
    p.add_argument("--logit_clip", type=float, default=10.0,
                   help="Clamp RGB/Conf logits to [-clip, clip] (<=0 to disable)")

    p.add_argument("--lambda_percep", type=float, default=0.05)
    p.add_argument("--lambda_conf", type=float, default=1e-3)
    p.add_argument("--lambda_bright", type=float, default=0.5)
    p.add_argument("--lambda_contrast", type=float, default=0.5)
    p.add_argument("--lambda_alpha_reg", type=float, default=1e-4)

    p.add_argument("--lambda_edge", type=float, default=0.02,
                   help="edge/gradient loss weight (small but useful)")
    p.add_argument("--lambda_tv_conf", type=float, default=1e-4,
                   help="TV loss on pred_conf to suppress salt-pepper")
    p.add_argument("--lambda_ssim", type=float, default=0.0,
                   help="(optional) add 1-SSIM loss, usually keep 0 or tiny")

    p.add_argument("--conf_thr", type=float, default=0.2)
    p.add_argument("--conf_temp", type=float, default=0.06)
    p.add_argument("--use_conf_loss_gate", action="store_true", default=True,
                   help="Use depth_conf soft gate in train/recon weights")
    p.add_argument("--no_use_conf_loss_gate", dest="use_conf_loss_gate",
                   action="store_false", help="Disable depth_conf gating in loss weights")
    p.add_argument("--train_min_cover", type=float, default=0.10)
    p.add_argument("--fg_thr", type=float, default=0.5)
    p.add_argument("--fg_min_cover", type=float, default=0.05)
    p.add_argument("--fg_dilate_k", type=int, default=7)
    p.add_argument("--fg_keep_largest_cc", type=int, default=1, choices=[0, 1])
    p.add_argument("--fg_lcc_min_pixels", type=int, default=32)
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
    p.add_argument("--debug_val_every_epoch",
                   action="store_true", default=True)

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


def main():
    global args
    args = parse_args()
    args.use_view_cond = bool(args.use_view_cond)
    args.finetune_view_only = bool(args.finetune_view_only)
    setup_torch_stability(tf32=args.tf32)
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
        "device": device,
        "amp": args.amp,
        "tf32": args.tf32,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
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
    )
    train_dataset = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=args.num_src_views,
        frame_subsample=args.frame_subsample,
        split="train",
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        deterministic_views=False,
    )
    val_dataset = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=args.num_src_views,
        frame_subsample=args.frame_subsample,
        split="val",
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        deterministic_views=True,
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
    except Exception:
        pass

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
        ema_model.load_state_dict(model.state_dict(), strict=False)
        print("[info] ema initialized from model (hard copy).")

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
        else:
            optimizer = torch.optim.AdamW(
                train_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            train_params, lr=args.lr, weight_decay=args.weight_decay)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience
        )

    scaler = make_grad_scaler(
        device=device, enabled=(args.amp and device == "cuda"))

    best_val = float("inf")
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
            best_val = float("inf")
            epochs_no_improve = 0
        else:
            start_epoch = int(ck.get("epoch", 0))
            global_step = int(ck.get("global_step", 0))
            best_val = float(ck.get("best_val", best_val))
            epochs_no_improve = int(ck.get("epochs_no_improve", 0))
            print(
                f"[info] resume meta: epoch={start_epoch} global_step={global_step} best_val={best_val:.6f} no_improve={epochs_no_improve}")

    def run_val(eval_model: nn.Module, *, save_debug: bool, tag: str, collect_by_view: bool = False):
        eval_model.eval()
        val_sum = 0.0
        val_cnt = 0
        m_psnr = 0.0
        m_ssim = 0.0
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
                src_imgs = batch["src_imgs"].to(device, non_blocking=True)
                src_depth = batch["src_depth"].to(device, non_blocking=True)
                src_depth_conf = batch["src_depth_conf"].to(
                    device, non_blocking=True)
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
                    conf_gate_strength = _conf_gate_strength(global_step)
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
                        pred_conf_gate=pred_conf,
                        use_conf_gate=bool(args.use_conf_gate),
                        conf_gate_detach=args.conf_gate_detach,
                        conf_gate_floor=args.conf_gate_floor,
                        conf_gate_gamma=args.conf_gate_gamma,
                        conf_gate_strength=conf_gate_strength,
                        recon_gate_floor=args.recon_gate_floor,
                        recon_weight_renorm=args.recon_weight_renorm,
                        recon_weight_clip_max=args.recon_weight_clip_max,
                    )
                    l1w = masked_l1(pred_rgb, tgt_img, recon_weight)

                bs = tgt_img.size(0)
                val_sum += float(l1w.item()) * bs
                val_cnt += bs

                pred_f = pred_rgb.float().clamp(0, 1)
                tgt_f = tgt_img.float().clamp(0, 1)
                m_psnr += float(psnr(pred_f, tgt_f).item()) * bs
                m_ssim += float(ssim(pred_f, tgt_f).item()) * bs

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
                        "fg_mask": fg_mask.detach(),
                        "train_mask": train_mask.detach(),
                        "recon_weight": recon_weight.detach(),
                        "valid_mask": valid_mask.detach(),
                    }
                    if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
                        aux_dbg["gate"] = aux_pred.get("gate")
                    save_debug_pack(
                        pred_rgb, tgt_img, aux_dbg, global_step,
                        out_dir=os.path.join(args.log_dir, "val"),
                        prefix=(f"val_e{epoch:03d}" if tag == "raw" else f"val_{tag}_e{epoch:03d}"),
                        split_cat_panels=args.split_cat_panels
                    )

        mean_val = val_sum / max(1, val_cnt)
        mean_psnr = m_psnr / max(1, val_cnt)
        mean_ssim = m_ssim / max(1, val_cnt)
        by_view_rows = None
        if collect_by_view and len(by_view) > 0:
            rows = []
            for vid, s in by_view.items():
                n = max(1, s["n"])
                rows.append((vid, s["sum_l1"] / n, s["sum_psnr"] / n, n))
            rows.sort(key=lambda x: x[1], reverse=True)
            by_view_rows = rows
        return mean_val, mean_psnr, mean_ssim, by_view_rows

    # ---------------------------
    # Train
    # ---------------------------
    accum_steps = max(1, int(args.accum_steps))

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
        if mode == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        if mode == "exp":
            k = float(getattr(args, "conf_gate_ramp_k", 5.0))
            k = max(1e-6, k)
            return (1.0 - math.exp(-k * progress)) / (1.0 - math.exp(-k))
        return progress

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for inner, batch in enumerate(train_loader):
            src_imgs = batch["src_imgs"].to(device, non_blocking=True)
            src_depth = batch["src_depth"].to(device, non_blocking=True)
            src_depth_conf = batch["src_depth_conf"].to(
                device, non_blocking=True)
            src_pointmap = batch["src_pointmap"].to(device, non_blocking=True)
            tgt_img = batch["tgt_img"].to(device, non_blocking=True)
            tgt_vid = batch.get("tgt_vid", None)
            if tgt_vid is not None:
                tgt_vid = tgt_vid.to(device, non_blocking=True)
            src_vids = batch.get("src_vids", None)
            if src_vids is not None:
                src_vids = src_vids.to(device, non_blocking=True)

            with autocast_ctx(device=device, enabled=(args.amp and device == "cuda")):
                conf_gate_strength = _conf_gate_strength(global_step)
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
                    pred_conf_gate=pred_conf,
                    use_conf_gate=bool(args.use_conf_gate),
                    conf_gate_detach=args.conf_gate_detach,
                    conf_gate_floor=args.conf_gate_floor,
                    conf_gate_gamma=args.conf_gate_gamma,
                    conf_gate_strength=conf_gate_strength,
                    recon_gate_floor=args.recon_gate_floor,
                    recon_weight_renorm=args.recon_weight_renorm,
                    recon_weight_clip_max=args.recon_weight_clip_max,
                )

                recon_loss = masked_l1(pred_rgb, tgt_img, recon_weight)

                vgg_h, vgg_w = 256, 256
                pred_small = F.interpolate(pred_rgb, size=(
                    vgg_h, vgg_w), mode="bilinear", align_corners=False)
                tgt_small = F.interpolate(tgt_img,  size=(
                    vgg_h, vgg_w), mode="bilinear", align_corners=False)
                w_small = F.interpolate(recon_weight, size=(
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
                conf_reg = conf_match + float(args.conf_floor_w) * conf_floor

                mean_p, std_p = masked_mean_std(pred_rgb, recon_weight)
                mean_t, std_t = masked_mean_std(tgt_img, recon_weight)
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
                    edge_loss = masked_l1(g_pred, g_tgt, recon_weight)

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

                loss = (
                    recon_loss
                    + args.lambda_percep * percep_loss
                    + args.lambda_conf * conf_reg
                    + args.lambda_bright * bright_loss
                    + args.lambda_contrast * contrast_loss
                    + args.lambda_alpha_reg * alpha_reg
                    + args.lambda_edge * edge_loss
                    + args.lambda_tv_conf * tv_conf
                    + args.lambda_ssim * ssim_loss
                )

                loss = loss / float(accum_steps)

            if not torch.isfinite(loss):
                print(
                    f"[warn] non-finite loss at epoch={epoch} inner={inner}, skip batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f"[warn] backward runtime error (skip batch): {e}")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            do_step = ((inner + 1) % accum_steps == 0)

            if do_step:
                try:
                    scaler.unscale_(optimizer)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if args.lr_schedule == "cosine":
                        scheduler.step()

                    global_step += 1

                    if args.use_ema and ema_model is not None:
                        ema_update(ema_model, model, decay=args.ema_decay)

                except RuntimeError as e:
                    print(f"[warn] optimizer step error (skip): {e}")
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

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
                    f"recon={recon_loss.item():.4f} percep={percep_loss.item():.4f} conf_reg={conf_reg.item():.4f} "
                    f"edge={edge_loss.item():.4f} tvc={tv_conf.item():.4f} ssimL={ssim_loss.item():.4f} "
                    f"bright={bright_loss.item():.4f} contrast={contrast_loss.item():.4f} "
                    f"cover_train={aux_masks['cover_train']:.4f} cover_fg={aux_masks['cover_fg']:.4f} "
                    f"cover_conf={aux_masks['cover_conf']:.4f} cover_valid={aux_masks['cover_valid']:.4f} "
                    f"pred_conf(min/mean/max)=({pc_min:.3f}/{pc_mean:.3f}/{pc_max:.3f}) "
                    f"pred_conf_mode={pred_conf_mode} "
                    f"conf_mode(mask)={ci.get('mode', '')} conf_mode(sup)={tgt_conf_info.get('mode', '')}"
                )

                kv = {
                    "loss": loss.item() * accum_steps,
                    "recon": recon_loss.item(),
                    "percep": percep_loss.item(),
                    "conf_reg": conf_reg.item(),
                    "edge": edge_loss.item(),
                    "tv_conf": tv_conf.item(),
                    "ssim_loss": ssim_loss.item(),
                    "bright": bright_loss.item(),
                    "contrast": contrast_loss.item(),
                    "cover_train": aux_masks["cover_train"],
                    "cover_fg": aux_masks["cover_fg"],
                    "cover_conf": aux_masks["cover_conf"],
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
                    "conf_mode_mask": ci.get("mode", ""),
                    "conf_mode_sup": tgt_conf_info.get("mode", ""),
                    "lr": optimizer.param_groups[0]["lr"],
                    "conf_gate_strength": float(conf_gate_strength),
                    "conf_gate_warmup": int(args.conf_gate_warmup),
                    "conf_gate_ramp": int(args.conf_gate_ramp),
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

                ini_logger.log(f"train_step_{global_step:06d}", kv)

                aux_dbg = {
                    "pred_conf": pred_conf_safe.detach(),
                    "tgt_depth_conf": aux_masks.get("tgt_depth_conf", None),
                    "tgt_depth_conf_raw": aux_masks.get("tgt_depth_conf_raw", None),
                    "fg_mask": fg_mask.detach(),
                    "train_mask": train_mask.detach(),
                    "recon_weight": recon_weight.detach(),
                    "valid_mask": valid_mask.detach(),
                }
                if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
                    aux_dbg["gate"] = aux_pred.get("gate")
                save_debug_pack(
                    pred_rgb, tgt_img, aux_dbg, global_step,
                    out_dir=os.path.join(args.log_dir, "train"),
                    prefix="train",
                    split_cat_panels=args.split_cat_panels
                )

        mean_train = epoch_loss / max(1, len(train_loader))
        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] mean train loss = {mean_train:.6f} time={dt:.1f}s")

        # ---------------------------
        # Val
        # ---------------------------
        model.eval()
        raw_val, raw_psnr, raw_ssim, raw_by_view = run_val(
            model, save_debug=True, tag="raw", collect_by_view=True)
        ema_val, ema_psnr, ema_ssim = None, None, None
        if args.use_ema and ema_model is not None:
            ema_val, ema_psnr, ema_ssim, _ = run_val(
                ema_model, save_debug=False, tag="ema", collect_by_view=False)

        if ema_val is not None:
            print(
                f"[Epoch {epoch:02d}] val raw={raw_val:.6f} PSNR={raw_psnr:.2f} SSIM={raw_ssim:.4f} | "
                f"ema={ema_val:.6f} PSNR={ema_psnr:.2f} SSIM={ema_ssim:.4f}"
            )
        else:
            print(
                f"[Epoch {epoch:02d}] val raw={raw_val:.6f} PSNR={raw_psnr:.2f} SSIM={raw_ssim:.4f}"
            )

        ini_logger.log(f"val_epoch_{epoch:03d}", {
            "wL1_raw": raw_val,
            "PSNR_raw": raw_psnr,
            "SSIM_raw": raw_ssim,
            "wL1_ema": (ema_val if ema_val is not None else -1.0),
            "PSNR_ema": (ema_psnr if ema_psnr is not None else -1.0),
            "SSIM_ema": (ema_ssim if ema_ssim is not None else -1.0),
            "best_by": args.best_by,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if raw_by_view:
            print("[by_view] worst->best (vid, meanL1, meanPSNR, N):")
            for row in raw_by_view:
                print(f"  {row}")

        metric_val = raw_val
        if args.best_by == "ema" and ema_val is not None:
            metric_val = ema_val

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
            "epochs_no_improve": epochs_no_improve,
            "args": vars(args),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        ckpt_last = os.path.join(args.ckpt_dir, "viewdec_ablation_last.pth")
        save_checkpoint(ckpt_last, payload)

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
            print(f" -> val 明显下降，保存新 best: {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f" -> val 没明显下降 (no_improve = {epochs_no_improve})")

        if args.early_stop and epochs_no_improve >= int(args.early_stop):
            print(
                f"[early-stop] stop: no_improve={epochs_no_improve} >= patience={args.early_stop}")
            break


if __name__ == "__main__":
    main()
