import argparse
import collections
import json
import os
import os.path as osp
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None:
        return str(default)
    s = str(raw).strip()
    return s if s else str(default)


def _env_int(key: str, default: int) -> int:
    s = _env_str(key, str(default))
    try:
        return int(s)
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    s = _env_str(key, str(default))
    try:
        return float(s)
    except Exception:
        return float(default)


def _env_bool(key: str, default: bool) -> bool:
    raw = str(os.environ.get(key, "") or "").strip().lower()
    if raw == "":
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if raw in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _split_tokens(raw: str) -> List[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [x for x in re.split(r"[,\s;/|]+", s) if x]


def _normalize_seq_names(seq_names) -> List[str]:
    if seq_names is None:
        return []
    if isinstance(seq_names, str):
        return _split_tokens(seq_names)
    if isinstance(seq_names, (list, tuple)):
        out = []
        for x in seq_names:
            out.extend(_split_tokens(str(x)))
        return out
    return _split_tokens(str(seq_names))


def _resolve_img_path(path_str: str, zju_root: str, seq_names: Sequence[str]) -> str:
    s = str(path_str).strip().replace("\\", "/")
    if osp.exists(s):
        return s
    if osp.isabs(s):
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
    s = s.lstrip("/")
    return osp.join(zju_root, s)


def _to_depth01(conf_like: torch.Tensor) -> torch.Tensor:
    x = conf_like.float()
    mx = float(x.max().item()) if x.numel() > 0 else 0.0
    if mx <= 1.5:
        return x.clamp(0.0, 1.0)
    if mx <= 32.0:
        return (x / (mx + 1e-8)).clamp(0.0, 1.0)
    if mx <= 255.0 + 1e-3:
        return (x / 255.0).clamp(0.0, 1.0)
    return (x / (mx + 1e-8)).clamp(0.0, 1.0)


def _build_conf_weight(
    conf01: torch.Tensor,
    valid01: torch.Tensor,
    thr: float,
    gamma: float,
) -> torch.Tensor:
    x = conf01.clamp(0.0, 1.0)
    t = float(min(0.95, max(0.0, thr)))
    g = float(max(0.1, gamma))
    if t > 0.0:
        x = ((x - t) / max(1e-6, 1.0 - t)).clamp(0.0, 1.0)
    if abs(g - 1.0) > 1e-6:
        x = torch.pow(x, g)
    return (x * valid01).clamp(0.0, 1.0)


def _build_per_view_conf_quantile_mask(
    conf01: torch.Tensor,
    valid01: torch.Tensor,
    quantile: float,
    min_valid: int = 16,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Build per-view confidence keep mask by quantile on each (B,V) slice.
    Keep pixels with confidence >= q-th quantile within valid pixels.
    """
    q = float(np.clip(float(quantile), 0.0, 1.0))
    mv = int(max(1, int(min_valid)))
    keep = valid01.float().clone()
    info: Dict[str, float] = {
        "conf_pvq_enabled": 0.0,
        "conf_pvq_quantile": q,
        "conf_pvq_min_valid": float(mv),
        "conf_pvq_keep_ratio": 1.0,
        "conf_pvq_skipped_ratio": 1.0,
    }
    if q <= 0.0:
        return keep, info
    info["conf_pvq_enabled"] = 1.0

    c = conf01.float().clamp(0.0, 1.0)
    if c.ndim == 5 and c.shape[-1] == 1:
        c = c[..., 0]
    v = (valid01 > 0.5)
    if v.ndim == 5 and v.shape[-1] == 1:
        v = v[..., 0]
    if c.ndim != 4 or v.ndim != 4:
        return keep, info

    keep2 = torch.zeros_like(c, dtype=torch.float32)
    total = int(v.numel())
    skipped = 0
    for bi in range(c.shape[0]):
        for vi in range(c.shape[1]):
            m = v[bi, vi]
            n = int(m.sum().item())
            if n < mv:
                keep2[bi, vi] = m.float()
                skipped += int(m.numel())
                continue
            qv = torch.quantile(c[bi, vi][m], q)
            keep2[bi, vi] = ((c[bi, vi] >= qv).float() * m.float())

    keep = keep * keep2
    info["conf_pvq_keep_ratio"] = float((keep > 0.0).float().mean().item())
    info["conf_pvq_skipped_ratio"] = float(skipped / max(1, total))
    return keep, info


def _build_gram_dynamic_weight_from_tokens(
    aggregated_tokens_list: Sequence[torch.Tensor],
    patch_start_idx: int,
    target_hw: Sequence[int],
    layer_idx: int = -1,
    quantile: float = 0.30,
    weight_floor: float = 0.25,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """
    Build soft dynamic/static weight map from cross-view token Gram similarity.
    Output weight is in [weight_floor, 1], larger means more cross-view static-consistent.
    """
    info: Dict[str, float] = {
        "gram_dyn_enabled": 0.0,
        "gram_dyn_layer_idx": float(layer_idx),
        "gram_dyn_quantile": float(quantile),
        "gram_dyn_weight_floor": float(weight_floor),
        "gram_dyn_keep_ratio": 1.0,
        "gram_dyn_mean": 1.0,
        "gram_dyn_p10": 1.0,
        "gram_dyn_p90": 1.0,
    }
    if aggregated_tokens_list is None or len(aggregated_tokens_list) <= 0:
        return None, info
    try:
        q = float(np.clip(float(quantile), 0.0, 0.95))
        wf = float(np.clip(float(weight_floor), 0.0, 1.0))
        h = int(target_hw[0])
        w = int(target_hw[1])
        if h <= 0 or w <= 0:
            return None, info

        n_layers = int(len(aggregated_tokens_list))
        li = int(layer_idx)
        if li < 0:
            li = n_layers + li
        li = int(max(0, min(n_layers - 1, li)))

        tok = aggregated_tokens_list[li]
        if tok is None or tok.ndim != 4:
            return None, info
        if int(tok.shape[1]) <= 1:
            # Single-view cannot estimate cross-view dynamics.
            return None, info

        patch = tok[:, :, int(patch_start_idx):, :].float()  # (B,V,P,C)
        if patch.ndim != 4:
            return None, info
        b, v, p, c = patch.shape
        if p <= 0 or c <= 0:
            return None, info

        hp = int(round(float(np.sqrt(float(p)))))
        hp = max(1, hp)
        wp = int(max(1, p // hp))
        if hp * wp != p:
            # Fallback to 1xP strip if token count is not a perfect grid.
            hp = 1
            wp = int(p)

        x = F.normalize(patch, dim=-1, eps=1e-6)
        x_mean = x.mean(dim=1, keepdim=True)  # (B,1,P,C)
        if int(v) > 1:
            x_others = (x_mean * float(v) - x) / float(max(1, int(v) - 1))
            x_others = F.normalize(x_others, dim=-1, eps=1e-6)
        else:
            x_others = x_mean.expand_as(x)
        sim = torch.sum(x * x_others, dim=-1).clamp(-1.0, 1.0)  # (B,V,P)
        sim01 = (sim + 1.0) * 0.5

        flat = sim01.reshape(b, v, -1)
        if q > 0.0:
            thr = torch.quantile(flat, q, dim=-1, keepdim=True)
            den = (1.0 - thr).clamp_min(1e-6)
            w_patch = ((flat - thr) / den).clamp(0.0, 1.0)
        else:
            w_patch = flat.clamp(0.0, 1.0)
        w_patch = wf + (1.0 - wf) * w_patch

        w_patch = w_patch.reshape(b * v, 1, hp, wp)
        w_full = F.interpolate(
            w_patch, size=(h, w), mode="bilinear", align_corners=False
        ).reshape(b, v, h, w).clamp(0.0, 1.0)
        arr = w_full.detach()
        info["gram_dyn_enabled"] = 1.0
        info["gram_dyn_keep_ratio"] = float((arr > (wf + 1e-6)).float().mean().item())
        info["gram_dyn_mean"] = float(arr.mean().item())
        info["gram_dyn_p10"] = float(torch.quantile(arr.reshape(-1), 0.10).item())
        info["gram_dyn_p90"] = float(torch.quantile(arr.reshape(-1), 0.90).item())
        return w_full, info
    except Exception:
        return None, info


def _build_dyn_proxy_weight(
    fg_mask01: Optional[torch.Tensor],
    gram_static_weight: Optional[torch.Tensor],
    support01: Optional[torch.Tensor],
    use_gram: bool = True,
    use_support: bool = True,
    floor: float = 0.35,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """
    Build a foreground-only static soft weight proxy from Gram/static consistency
    and multi-view point support. Output is 1.0 outside foreground and
    in [floor, 1.0] inside foreground.
    """
    info: Dict[str, float] = {
        "dyn_proxy_enabled": 0.0,
        "dyn_proxy_keep_ratio": 1.0,
        "dyn_proxy_mean": 1.0,
        "dyn_proxy_p10": 1.0,
        "dyn_proxy_p90": 1.0,
        "dyn_proxy_fg_mean": 1.0,
        "dyn_proxy_bg_mean": 1.0,
        "dyn_proxy_use_gram_effective": 0.0,
        "dyn_proxy_use_support_effective": 0.0,
    }
    if fg_mask01 is None:
        return None, info

    fg = fg_mask01
    if fg.ndim == 5 and fg.shape[-1] == 1:
        fg = fg[..., 0]
    if fg.ndim != 4:
        return None, info
    fg = fg.float().clamp(0.0, 1.0)

    comps = []
    if bool(use_gram) and gram_static_weight is not None:
        g = gram_static_weight
        if g.ndim == 5 and g.shape[-1] == 1:
            g = g[..., 0]
        if g.ndim == 4 and g.shape == fg.shape:
            comps.append(g.float().clamp(0.0, 1.0))
            info["dyn_proxy_use_gram_effective"] = 1.0
    if bool(use_support) and support01 is not None:
        s = support01
        if s.ndim == 5 and s.shape[-1] == 1:
            s = s[..., 0]
        if s.ndim == 4 and s.shape == fg.shape:
            comps.append(s.float().clamp(0.0, 1.0))
            info["dyn_proxy_use_support_effective"] = 1.0
    if len(comps) <= 0:
        return None, info

    floor_v = float(np.clip(float(floor), 0.0, 1.0))
    if len(comps) == 1:
        static_soft = comps[0]
    else:
        stack = torch.stack(comps, dim=0).clamp_min(1e-6)
        static_soft = torch.exp(torch.log(stack).mean(dim=0))
    static_soft = (floor_v + (1.0 - floor_v) * static_soft.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    out = ((1.0 - fg) + fg * static_soft).clamp(0.0, 1.0)

    arr = out.detach()
    fg_active = (fg > 0.5)
    bg_active = ~fg_active
    info["dyn_proxy_enabled"] = 1.0
    if int(arr.numel()) > 0:
        flat = arr.reshape(-1)
        info["dyn_proxy_mean"] = float(flat.mean().item())
        info["dyn_proxy_p10"] = float(torch.quantile(flat, 0.10).item())
        info["dyn_proxy_p90"] = float(torch.quantile(flat, 0.90).item())
    if int(fg_active.sum().item()) > 0:
        fg_vals = arr[fg_active]
        info["dyn_proxy_keep_ratio"] = float((fg_vals > (floor_v + 1e-6)).float().mean().item())
        info["dyn_proxy_fg_mean"] = float(fg_vals.mean().item())
    if int(bg_active.sum().item()) > 0:
        info["dyn_proxy_bg_mean"] = float(arr[bg_active].mean().item())
    return out, info


def _append_region_distribution_stats(
    info: Dict[str, float],
    prefix: str,
    value01: Optional[torch.Tensor],
    fg_mask01: Optional[torch.Tensor],
    active_mask01: Optional[torch.Tensor] = None,
) -> None:
    """
    Record global + foreground/background distribution stats for a scalar map.
    The tensor is expected to be (B,V,H,W) or (B,V,H,W,1). Stats are detached
    and intended purely for diagnostics.
    """
    if value01 is None:
        return
    arr = value01
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 4:
        return
    arr = arr.detach().float()
    finite = torch.isfinite(arr)
    if active_mask01 is not None:
        active = active_mask01
        if active.ndim == 5 and active.shape[-1] == 1:
            active = active[..., 0]
        if active.ndim == 4 and active.shape == arr.shape:
            finite = finite & (active > 0.5)
    flat = arr[finite]
    if int(flat.numel()) > 0:
        info[f"{prefix}_mean"] = float(flat.mean().item())
        info[f"{prefix}_p10"] = float(torch.quantile(flat, 0.10).item())
        info[f"{prefix}_p90"] = float(torch.quantile(flat, 0.90).item())
    if fg_mask01 is None:
        return
    fg = fg_mask01
    if fg.ndim == 5 and fg.shape[-1] == 1:
        fg = fg[..., 0]
    if fg.ndim != 4 or fg.shape != arr.shape:
        return
    fg = fg.detach().float().clamp(0.0, 1.0)
    fg_active = finite & (fg > 0.5)
    bg_active = finite & (fg <= 0.5)
    if int(fg_active.sum().item()) > 0:
        fg_vals = arr[fg_active]
        info[f"{prefix}_fg_mean"] = float(fg_vals.mean().item())
        info[f"{prefix}_fg_p10"] = float(torch.quantile(fg_vals, 0.10).item())
        info[f"{prefix}_fg_p90"] = float(torch.quantile(fg_vals, 0.90).item())
    if int(bg_active.sum().item()) > 0:
        bg_vals = arr[bg_active]
        info[f"{prefix}_bg_mean"] = float(bg_vals.mean().item())
        info[f"{prefix}_bg_p10"] = float(torch.quantile(bg_vals, 0.10).item())
        info[f"{prefix}_bg_p90"] = float(torch.quantile(bg_vals, 0.90).item())


def _masked_scalar_mean(
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    x = value.float()
    if mask is None:
        return x.mean()
    m = mask.float().clamp(0.0, 1.0)
    return (x * m).sum() / m.sum().clamp_min(eps)


def _apply_fg_supervision_boost(
    base_weight01: torch.Tensor,
    fg_mask01: Optional[torch.Tensor],
    fg_boost: float = 1.0,
    fg_stats_mask01: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    info: Dict[str, float] = {
        "fg_supervision_boost": float(fg_boost),
        "fg_supervision_boost_applied": 0.0,
        "fg_supervision_profile_mean": 1.0,
        "fg_supervision_profile_fg_mean": 1.0,
        "fg_supervision_profile_bg_mean": 1.0,
        "fg_supervision_weight_mean": float(base_weight01.mean().item()),
    }
    if fg_mask01 is None:
        return base_weight01, info

    boost = float(max(0.0, fg_boost))
    fg = fg_mask01.float().clamp(0.0, 1.0)
    if fg.shape != base_weight01.shape:
        raise RuntimeError(
            f"fg supervision boost shape mismatch: weight={tuple(base_weight01.shape)} fg={tuple(fg.shape)}"
        )
    fg_stats = fg
    if fg_stats_mask01 is not None:
        fg_stats = fg_stats_mask01.float().clamp(0.0, 1.0)
        if fg_stats.shape != base_weight01.shape:
            raise RuntimeError(
                f"fg supervision stats shape mismatch: weight={tuple(base_weight01.shape)} fg_stats={tuple(fg_stats.shape)}"
            )
    if abs(boost - 1.0) <= 1e-6:
        _append_region_distribution_stats(
            info=info,
            prefix="fg_supervision_profile",
            value01=torch.ones_like(base_weight01),
            fg_mask01=fg_stats,
        )
        return base_weight01, info

    profile = (1.0 + (boost - 1.0) * fg).clamp_min(0.0)
    out = base_weight01 * profile
    info["fg_supervision_boost_applied"] = 1.0
    info["fg_supervision_weight_mean"] = float(out.mean().item())
    _append_region_distribution_stats(
        info=info,
        prefix="fg_supervision_profile",
        value01=profile,
        fg_mask01=fg_stats,
    )
    return out, info


def _apply_region_weight_boost(
    base_weight01: torch.Tensor,
    region_mask01: Optional[torch.Tensor],
    boost: float,
    info_prefix: str,
    fg_stats_mask01: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    prefix = str(info_prefix or "region")
    info: Dict[str, float] = {
        f"{prefix}_boost": float(boost),
        f"{prefix}_boost_applied": 0.0,
        f"{prefix}_weight_mean": float(base_weight01.mean().item()),
    }
    if region_mask01 is None:
        return base_weight01, info

    gain = float(max(0.0, boost))
    region = region_mask01.float().clamp(0.0, 1.0)
    if region.shape != base_weight01.shape:
        raise RuntimeError(
            f"{prefix} boost shape mismatch: weight={tuple(base_weight01.shape)} region={tuple(region.shape)}"
        )
    if abs(gain - 1.0) <= 1e-6:
        return base_weight01, info

    profile = (1.0 + (gain - 1.0) * region).clamp_min(0.0)
    out = base_weight01 * profile
    info[f"{prefix}_boost_applied"] = 1.0
    info[f"{prefix}_weight_mean"] = float(out.mean().item())
    _append_region_distribution_stats(
        info=info,
        prefix=f"{prefix}_profile",
        value01=profile,
        fg_mask01=fg_stats_mask01,
        active_mask01=region,
    )
    return out, info


def _build_fg_supervision_boost_mask(
    fg_mask01: Optional[torch.Tensor],
    region_mode: str = "all",
    region_erode_px: int = 0,
) -> tuple[Optional[torch.Tensor], Dict[str, float | str]]:
    mode = str(region_mode or "all").strip().lower()
    if mode not in {"all", "interior_only"}:
        mode = "all"
    erode_px_i = int(max(0, int(region_erode_px)))
    info: Dict[str, float | str] = {
        "fg_supervision_region_mode": mode,
        "fg_supervision_region_erode_px": float(erode_px_i),
        "fg_supervision_boost_cover": -1.0,
        "fg_supervision_boost_cover_ratio_in_fg": 0.0,
        "fg_supervision_boundary_ring_cover": 0.0,
        "fg_supervision_boundary_ring_ratio_in_fg": 0.0,
    }
    if fg_mask01 is None:
        return None, info

    fg = fg_mask01.float().clamp(0.0, 1.0)
    if fg.ndim != 4:
        raise RuntimeError(f"fg supervision mask must be (B,V,H,W), got {tuple(fg.shape)}")

    if mode == "interior_only" and erode_px_i > 0:
        boost_mask = _erode_mask_tensor(fg, erode_px_i)
    else:
        boost_mask = fg
    ring = (fg - boost_mask).clamp(0.0, 1.0)
    fg_sum = float(fg.sum().item())
    boost_sum = float(boost_mask.sum().item())
    ring_sum = float(ring.sum().item())
    info["fg_supervision_boost_cover"] = float(boost_mask.mean().item())
    if fg_sum > 1e-6:
        info["fg_supervision_boost_cover_ratio_in_fg"] = float(boost_sum / fg_sum)
        info["fg_supervision_boundary_ring_ratio_in_fg"] = float(ring_sum / fg_sum)
    info["fg_supervision_boundary_ring_cover"] = float(ring.mean().item())
    _append_region_distribution_stats(
        info=info,
        prefix="fg_supervision_boost_mask",
        value01=boost_mask,
        fg_mask01=fg,
    )
    _append_region_distribution_stats(
        info=info,
        prefix="fg_supervision_boundary_ring",
        value01=ring,
        fg_mask01=fg,
    )
    return boost_mask, info


def _fg_conf_presence_floor_loss(
    pred_conf01: torch.Tensor,
    tgt_conf01: torch.Tensor,
    fg_mask01: Optional[torch.Tensor],
    valid01: Optional[torch.Tensor] = None,
    target_ratio: float = 0.9,
) -> tuple[torch.Tensor, Dict[str, float]]:
    z = torch.zeros([], device=pred_conf01.device, dtype=torch.float32)
    info: Dict[str, float] = {
        "fg_conf_presence_enabled": 0.0,
        "fg_conf_presence_target_ratio": float(target_ratio),
        "fg_conf_presence_pred_mean": 0.0,
        "fg_conf_presence_tgt_mean": 0.0,
        "fg_conf_presence_target_floor": 0.0,
        "fg_conf_presence_active_ratio": 0.0,
        "fg_conf_presence_loss": 0.0,
    }
    if fg_mask01 is None:
        return z, info

    mask = fg_mask01.float().clamp(0.0, 1.0)
    if valid01 is not None:
        mask = (mask * valid01.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    active = mask.sum()
    if float(active.item()) <= 1e-6:
        return z, info

    ratio = float(max(0.0, target_ratio))
    pred_mean = _masked_scalar_mean(pred_conf01, mask)
    tgt_mean = _masked_scalar_mean(tgt_conf01, mask)
    target_floor = ratio * tgt_mean.detach()
    loss = F.relu(target_floor - pred_mean)
    info["fg_conf_presence_enabled"] = 1.0
    info["fg_conf_presence_pred_mean"] = float(pred_mean.item())
    info["fg_conf_presence_tgt_mean"] = float(tgt_mean.item())
    info["fg_conf_presence_target_floor"] = float(target_floor.item())
    info["fg_conf_presence_active_ratio"] = float((mask > 0.0).float().mean().item())
    info["fg_conf_presence_loss"] = float(loss.item())
    return loss, info


def _normal_from_world_point_map(
    point_world: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Estimate normals from world-space point map using central differences.
    Input: point_world (B,V,H,W,3), valid_mask (B,V,H,W)
    Returns: normal (B,V,H-2,W-2,3), valid_center (B,V,H-2,W-2)
    """
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        return None, None
    b, v, h, w, _ = point_world.shape
    if h < 3 or w < 3:
        return None, None

    px = point_world[:, :, 1:-1, 2:, :] - point_world[:, :, 1:-1, :-2, :]
    py = point_world[:, :, 2:, 1:-1, :] - point_world[:, :, :-2, 1:-1, :]
    n = torch.cross(px, py, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-6)

    vm = (valid_mask > 0.5)
    if vm.ndim == 5 and vm.shape[-1] == 1:
        vm = vm[..., 0]
    v_center = (
        vm[:, :, 1:-1, 1:-1]
        & vm[:, :, 1:-1, 2:]
        & vm[:, :, 1:-1, :-2]
        & vm[:, :, 2:, 1:-1]
        & vm[:, :, :-2, 1:-1]
    ).float()
    return n, v_center


def _point_normal_consistency_loss(
    point_world_pred: torch.Tensor,
    point_world_tgt: torch.Tensor,
    valid_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    n_pred, v_pred = _normal_from_world_point_map(point_world_pred, valid_mask)
    n_tgt, v_tgt = _normal_from_world_point_map(point_world_tgt, valid_mask)
    if n_pred is None or n_tgt is None or v_pred is None or v_tgt is None:
        z = torch.zeros([], device=point_world_pred.device, dtype=torch.float32)
        return z, {
            "point_normal_valid_ratio": 0.0,
            "point_normal_cos_mean": 1.0,
            "point_normal_loss_raw": 0.0,
        }

    valid = (v_pred * v_tgt).clamp(0.0, 1.0)
    w = valid
    if support_weight is not None:
        sw = support_weight.float()
        if sw.ndim == 5 and sw.shape[-1] == 1:
            sw = sw[..., 0]
        if sw.ndim == 4:
            w = w * sw[:, :, 1:-1, 1:-1].clamp(0.0, 1.0)

    cos = torch.sum(n_pred * n_tgt, dim=-1).clamp(-1.0, 1.0)
    loss_map = 1.0 - cos
    den = w.sum() + 1e-6
    loss = (loss_map * w).sum() / den
    info = {
        "point_normal_valid_ratio": float((valid > 0.0).float().mean().item()),
        "point_normal_cos_mean": float((cos * valid).sum().item() / (valid.sum().item() + 1e-6)),
        "point_normal_loss_raw": float(loss.item()),
    }
    return loss, info


def _map_support_weight(
    support01: Optional[torch.Tensor],
    mode: str,
    floor: float,
) -> Optional[torch.Tensor]:
    """
    Map multiview support weights for different supervision strategies.
    - direct: high support -> high weight
    - inverse: low support -> high weight (focus on ghost-prone pixels)
    - off: disable support weighting
    """
    if support01 is None:
        return None
    m = str(mode or "direct").strip().lower()
    if m in {"off", "none"}:
        return None
    x = support01.clamp(0.0, 1.0)
    f = float(np.clip(float(floor), 0.0, 1.0))
    if m == "inverse":
        return (f + (1.0 - f) * (1.0 - x)).clamp(0.0, 1.0)
    return (f + (1.0 - f) * x).clamp(0.0, 1.0)


def _safe_resize_like(x: torch.Tensor, ref_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref_hw:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref_hw, mode="nearest")
    return F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)


def _augment_images(imgs: torch.Tensor, jitter: float, noise_std: float) -> torch.Tensor:
    # imgs: (B, V, 3, H, W) in [0,1]
    if jitter <= 0 and noise_std <= 0:
        return imgs
    b, v = imgs.shape[:2]
    out = imgs.clone()
    for bi in range(b):
        for vi in range(v):
            x = out[bi, vi]
            if jitter > 0:
                alpha = 1.0 + random.uniform(-jitter, jitter)  # contrast
                beta = random.uniform(-jitter, jitter)  # brightness
                x = x * alpha + beta
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
            out[bi, vi] = x.clamp(0.0, 1.0)
    return out


def _align_depth_median_scale(
    depth_pred: torch.Tensor,
    depth_tgt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pred = depth_pred[..., 0]
    tgt = depth_tgt[..., 0]
    out = pred.clone()
    scales: list[float] = []

    B, V = pred.shape[:2]
    for bi in range(B):
        for vi in range(V):
            m = (
                (valid_mask[bi, vi] > 0.5)
                & torch.isfinite(pred[bi, vi])
                & torch.isfinite(tgt[bi, vi])
                & (pred[bi, vi].abs() > 1e-6)
                & (tgt[bi, vi].abs() > 1e-6)
            )
            if int(m.sum().item()) < 16:
                scale = 1.0
            else:
                p_med = pred[bi, vi][m].detach().median()
                t_med = tgt[bi, vi][m].detach().median()
                scale = float((t_med / (p_med + 1e-6)).clamp(0.1, 10.0).item())
            out[bi, vi] = pred[bi, vi] * scale
            scales.append(scale)

    info: Dict[str, float] = {}
    if scales:
        arr = np.asarray(scales, dtype=np.float64)
        info["depth_scale_mean"] = float(arr.mean())
        info["depth_scale_std"] = float(arr.std())
        info["depth_scale_min"] = float(arr.min())
        info["depth_scale_max"] = float(arr.max())
    return out.unsqueeze(-1), info


def _apply_freeze_mode(model: torch.nn.Module, freeze_mode: str) -> Dict[str, bool]:
    mode = str(freeze_mode).strip().lower()
    if mode not in {"depth_point", "depth_only", "point_only", "all_trainable"}:
        raise ValueError(f"unsupported freeze_mode: {freeze_mode}")

    # all_trainable: keep full model trainable (mentor-aligned, not head-only tuning).
    if mode == "all_trainable":
        for p in model.parameters():
            p.requires_grad = True
        if model.aggregator is not None:
            model.aggregator.train()
        if model.camera_head is not None:
            model.camera_head.train()
        return {
            "depth_trainable": bool(model.depth_head is not None),
            "point_trainable": bool(model.point_head is not None),
        }

    for p in model.parameters():
        p.requires_grad = False

    depth_trainable = False
    point_trainable = False

    if mode in {"depth_point", "depth_only"} and model.depth_head is not None:
        for p in model.depth_head.parameters():
            p.requires_grad = True
        depth_trainable = True

    if mode in {"depth_point", "point_only"} and model.point_head is not None:
        for p in model.point_head.parameters():
            p.requires_grad = True
        point_trainable = True

    if model.aggregator is not None:
        model.aggregator.eval()
    if model.camera_head is not None:
        model.camera_head.eval()

    return {
        "depth_trainable": depth_trainable,
        "point_trainable": point_trainable,
    }


@dataclass
class HumanPriorSample:
    pointmap: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    body_mask: Optional[np.ndarray] = None
    head_mask: Optional[np.ndarray] = None
    face_mask: Optional[np.ndarray] = None
    pointmap_frame: str = ""
    source: str = ""
    path: str = ""


@dataclass
class Sample:
    cam_names: List[str]
    img_paths: List[str]
    depth: np.ndarray
    depth_conf: np.ndarray
    pointmap: np.ndarray
    extrinsic: Optional[np.ndarray]
    intrinsic: Optional[np.ndarray]
    pointmap_source: str = ""
    pointmap_frame: str = ""
    human_prior: Optional[HumanPriorSample] = None


def _optional_npz_array(data: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _optional_npz_string(data: Any, keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key not in data:
            continue
        raw = data[key]
        if isinstance(raw, np.ndarray):
            if raw.size <= 0:
                continue
            return str(raw.reshape(-1)[0])
        return str(raw)
    return str(default)


def _normalize_mask_stack_np(mask_like: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask_like is None:
        return None
    x = np.asarray(mask_like)
    if x.ndim == 4 and x.shape[-1] in {1, 3, 4}:
        x = x[..., 0]
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        return None
    out = [_normalize_mask_binary_np(xi) for xi in x]
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def _slice_optional_view_array(
    arr: Optional[np.ndarray],
    indices: Sequence[int],
    src_views: int,
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    x = np.asarray(arr)
    if x.ndim >= 1 and int(x.shape[0]) == int(src_views):
        return x[list(indices)]
    return x


def _resolve_sidecar_npz_path(
    *,
    zju_root: str,
    seq: str,
    sidecar_subdir: str,
    geom_npz_path: str,
) -> Optional[Path]:
    raw = str(sidecar_subdir or "").strip()
    if not raw:
        return None
    try:
        raw = raw.format(seq=str(seq))
    except Exception:
        pass
    basename = Path(str(geom_npz_path)).name
    p = Path(raw)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p / str(seq) / basename)
        candidates.append(p / basename)
    else:
        candidates.append(Path(str(zju_root)) / str(seq) / raw / basename)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_human_prior_sidecar(
    *,
    zju_root: str,
    seq: str,
    geom_npz_path: str,
    sidecar_subdir: str,
    target_cam_names: Sequence[str],
    source_cam_names_before_filter: Sequence[str],
    source_keep_indices: Sequence[int],
    strict: bool,
) -> Optional[HumanPriorSample]:
    prior_path = _resolve_sidecar_npz_path(
        zju_root=zju_root,
        seq=seq,
        sidecar_subdir=sidecar_subdir,
        geom_npz_path=geom_npz_path,
    )
    if prior_path is None:
        if strict:
            raise RuntimeError(
                f"human prior sidecar missing for seq={seq} geom={geom_npz_path} subdir={sidecar_subdir}"
            )
        return None

    with np.load(prior_path, allow_pickle=True) as data:
        prior_cam_names = [str(x) for x in data["cam_names"].tolist()] if "cam_names" in data else []
        pointmap = _optional_npz_array(data, ["prior_pointmap", "smpl_pointmap", "smplx_pointmap", "pointmap"])
        valid_mask = _optional_npz_array(data, ["prior_valid_mask", "valid_mask", "smpl_valid_mask"])
        body_mask = _optional_npz_array(data, ["body_mask", "prior_body_mask", "smpl_body_mask", "mask"])
        head_mask = _optional_npz_array(data, ["head_mask", "prior_head_mask", "smpl_head_mask"])
        face_mask = _optional_npz_array(data, ["face_mask", "prior_face_mask", "smpl_face_mask"])
        pointmap_frame = _optional_npz_string(data, ["pointmap_frame", "prior_pointmap_frame"], default="")
        source = _optional_npz_string(data, ["prior_source", "source", "prior_type"], default=prior_path.parent.name)

    source_views = int(len(source_cam_names_before_filter))
    keep = list(source_keep_indices)
    if prior_cam_names:
        index_map = {str(name): i for i, name in enumerate(prior_cam_names)}
        aligned = [index_map[c] for c in target_cam_names if c in index_map]
        if len(aligned) != len(target_cam_names):
            if strict:
                missing = [c for c in target_cam_names if c not in index_map]
                raise RuntimeError(
                    f"human prior cam_names mismatch for {prior_path}: missing={missing}"
                )
            return None
        pointmap = _slice_optional_view_array(pointmap, aligned, len(prior_cam_names))
        valid_mask = _slice_optional_view_array(valid_mask, aligned, len(prior_cam_names))
        body_mask = _slice_optional_view_array(body_mask, aligned, len(prior_cam_names))
        head_mask = _slice_optional_view_array(head_mask, aligned, len(prior_cam_names))
        face_mask = _slice_optional_view_array(face_mask, aligned, len(prior_cam_names))
    elif keep and len(keep) != source_views:
        pointmap = _slice_optional_view_array(pointmap, keep, source_views)
        valid_mask = _slice_optional_view_array(valid_mask, keep, source_views)
        body_mask = _slice_optional_view_array(body_mask, keep, source_views)
        head_mask = _slice_optional_view_array(head_mask, keep, source_views)
        face_mask = _slice_optional_view_array(face_mask, keep, source_views)

    if pointmap is not None:
        pointmap = np.asarray(pointmap)
        if pointmap.ndim != 4 or int(pointmap.shape[-1]) != 3:
            raise RuntimeError(f"unexpected human prior pointmap shape {tuple(pointmap.shape)} from {prior_path}")
        if int(pointmap.shape[0]) != len(target_cam_names):
            if strict:
                raise RuntimeError(
                    f"human prior view count mismatch {tuple(pointmap.shape)} vs target cams={len(target_cam_names)} from {prior_path}"
                )
            return None

    valid_mask = _normalize_mask_stack_np(valid_mask)
    body_mask = _normalize_mask_stack_np(body_mask)
    head_mask = _normalize_mask_stack_np(head_mask)
    face_mask = _normalize_mask_stack_np(face_mask)

    return HumanPriorSample(
        pointmap=pointmap.astype(np.float32, copy=False) if pointmap is not None else None,
        valid_mask=valid_mask,
        body_mask=body_mask,
        head_mask=head_mask,
        face_mask=face_mask,
        pointmap_frame=str(pointmap_frame or ""),
        source=str(source or ""),
        path=str(prior_path),
    )


class PseudoGeomDataset(Dataset):
    def __init__(
        self,
        zju_root: str,
        seq_names: Sequence[str],
        cam_names: Optional[Sequence[str]] = None,
        max_frames: int = 0,
        geom_subdir: str = "vggt_geom",
        human_prior_enable: bool = False,
        human_prior_subdir: str = "",
        human_prior_strict: bool = False,
    ):
        self.zju_root = str(zju_root)
        self.seq_names = [str(s) for s in seq_names]
        self.cam_names = set([str(c) for c in (cam_names or [])])
        self.human_prior_enable = bool(human_prior_enable)
        self.human_prior_subdir = str(human_prior_subdir or "")
        self.human_prior_strict = bool(human_prior_strict)
        self.items: List[tuple[str, str]] = []
        for seq in self.seq_names:
            gdir = Path(self.zju_root) / seq / geom_subdir
            if not gdir.is_dir():
                continue
            files = sorted([p for p in gdir.glob("*.npz") if p.is_file()])
            if max_frames > 0:
                files = files[: max_frames]
            for p in files:
                self.items.append((seq, str(p)))
        if not self.items:
            raise RuntimeError(
                f"no pseudo geometry found under {self.zju_root} for seq_names={self.seq_names}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        seq, path = self.items[idx]
        with np.load(path, allow_pickle=True) as data:
            img_paths = [str(x) for x in data["img_paths"].tolist()]
            cam_names = [str(x) for x in data["cam_names"].tolist()] if "cam_names" in data else []
            depth = np.asarray(data["depth"])
            depth_conf = np.asarray(data["depth_conf"])
            pointmap = np.asarray(data["pointmap"])
            extrinsic = np.asarray(data["extrinsic"]) if "extrinsic" in data else None
            intrinsic = np.asarray(data["intrinsic"]) if "intrinsic" in data else None
            if "pointmap_source" in data:
                src_raw = data["pointmap_source"]
                if isinstance(src_raw, np.ndarray):
                    if src_raw.size > 0:
                        pointmap_source = str(src_raw.reshape(-1)[0])
                    else:
                        pointmap_source = ""
                else:
                    pointmap_source = str(src_raw)
            else:
                pointmap_source = ""
            if "pointmap_frame" in data:
                frame_raw = data["pointmap_frame"]
                if isinstance(frame_raw, np.ndarray):
                    if frame_raw.size > 0:
                        pointmap_frame = str(frame_raw.reshape(-1)[0])
                    else:
                        pointmap_frame = ""
                else:
                    pointmap_frame = str(frame_raw)
            else:
                pointmap_frame = ""

        source_cam_names_before_filter = list(cam_names)
        src_view_count = int(len(source_cam_names_before_filter))
        keep: list[int] = list(range(len(cam_names)))
        if self.cam_names and cam_names:
            keep = [i for i, c in enumerate(cam_names) if c in self.cam_names]
            if len(keep) >= 2:
                cam_names = [cam_names[i] for i in keep]
                img_paths = [img_paths[i] for i in keep]
                depth = depth[keep]
                depth_conf = depth_conf[keep]
                pointmap = pointmap[keep]
                if extrinsic is not None and extrinsic.shape[0] == src_view_count:
                    extrinsic = extrinsic[keep]
                if intrinsic is not None and intrinsic.shape[0] == src_view_count:
                    intrinsic = intrinsic[keep]
            else:
                keep = list(range(len(cam_names)))

        if not cam_names:
            cam_names = [Path(p).parent.name for p in img_paths]

        human_prior = None
        if self.human_prior_enable:
            human_prior = _load_human_prior_sidecar(
                zju_root=self.zju_root,
                seq=seq,
                geom_npz_path=path,
                sidecar_subdir=self.human_prior_subdir,
                target_cam_names=cam_names,
                source_cam_names_before_filter=source_cam_names_before_filter,
                source_keep_indices=keep,
                strict=self.human_prior_strict,
            )

        img_paths = [
            _resolve_img_path(p, self.zju_root, self.seq_names) for p in img_paths
        ]
        return Sample(
            cam_names=cam_names,
            img_paths=img_paths,
            depth=depth,
            depth_conf=depth_conf,
            pointmap=pointmap,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            pointmap_source=pointmap_source,
            pointmap_frame=pointmap_frame,
            human_prior=human_prior,
        )


def _sample_to_tensors(
    sample: Sample,
    device: str,
    use_fg_mask: bool = False,
    fg_mask_source: str = "auto",
    use_human_prior: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    str,
    str,
    Optional[torch.Tensor],
    str,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    str,
    str,
]:
    imgs = load_and_preprocess_images(sample.img_paths).unsqueeze(0).to(device)  # (1,V,3,H,W)

    depth = torch.as_tensor(sample.depth, device=device).float()
    if depth.ndim == 3:
        depth = depth[..., None]
    depth = depth.unsqueeze(0)  # (1,V,H,W,1)

    conf = torch.as_tensor(sample.depth_conf, device=device).float()
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    conf = conf.unsqueeze(0)  # (1,V,H,W)

    point = torch.as_tensor(sample.pointmap, device=device).float()
    if point.ndim != 4 or point.shape[-1] != 3:
        raise RuntimeError(f"unexpected pointmap shape: {tuple(point.shape)}")
    point = point.unsqueeze(0)  # (1,V,H,W,3)

    extrinsic = None
    intrinsic = None
    if sample.extrinsic is not None:
        extrinsic = torch.as_tensor(sample.extrinsic, device=device).float()
        if extrinsic.ndim == 2:
            extrinsic = extrinsic[None, ...]
        if tuple(extrinsic.shape[-2:]) == (4, 4):
            extrinsic = extrinsic[..., :3, :4]
        if tuple(extrinsic.shape[-2:]) != (3, 4):
            raise RuntimeError(f"unexpected extrinsic shape: {tuple(extrinsic.shape)}")
        extrinsic = extrinsic.unsqueeze(0)  # (1,V,3,4)
    if sample.intrinsic is not None:
        intrinsic = torch.as_tensor(sample.intrinsic, device=device).float()
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None, ...]
        if tuple(intrinsic.shape[-2:]) == (4, 4):
            intrinsic = intrinsic[..., :3, :3]
        if tuple(intrinsic.shape[-2:]) != (3, 3):
            raise RuntimeError(f"unexpected intrinsic shape: {tuple(intrinsic.shape)}")
        intrinsic = intrinsic.unsqueeze(0)  # (1,V,3,3)

    fg_mask = None
    fg_mask_source_resolved = ""
    if bool(use_fg_mask):
        fg_arrs: list[np.ndarray] = []
        fg_src: Optional[str] = None
        for ip in sample.img_paths:
            mp, src = _infer_mask_path_from_image(ip, preferred=str(fg_mask_source))
            if (mp is None) or (not osp.isfile(mp)):
                fg_arrs = []
                fg_src = None
                break
            try:
                arr = np.asarray(Image.open(mp))
                fg_arrs.append(_normalize_mask_binary_np(arr))
                if fg_src is None:
                    fg_src = str(src)
            except Exception:
                fg_arrs = []
                fg_src = None
                break
        if len(fg_arrs) == len(sample.img_paths) and len(fg_arrs) > 0:
            fg_mask = torch.as_tensor(np.stack(fg_arrs, axis=0), device=device).float().unsqueeze(0)
            fg_mask_source_resolved = str(fg_src or "")

    prior_point = None
    prior_valid_mask = None
    prior_body_mask = None
    prior_head_mask = None
    prior_face_mask = None
    prior_source = ""
    prior_pointmap_frame = ""
    if bool(use_human_prior) and sample.human_prior is not None:
        hp = sample.human_prior
        prior_source = str(hp.source or "")
        prior_pointmap_frame = str(hp.pointmap_frame or "")
        if hp.pointmap is not None:
            prior_point = torch.as_tensor(hp.pointmap, device=device).float()
            if prior_point.ndim != 4 or prior_point.shape[-1] != 3:
                raise RuntimeError(f"unexpected human prior pointmap shape: {tuple(prior_point.shape)}")
            prior_point = prior_point.unsqueeze(0)
        if hp.valid_mask is not None:
            prior_valid_mask = torch.as_tensor(hp.valid_mask, device=device).float()
            if prior_valid_mask.ndim != 3:
                raise RuntimeError(f"unexpected human prior valid_mask shape: {tuple(prior_valid_mask.shape)}")
            prior_valid_mask = prior_valid_mask.unsqueeze(0)
        if hp.body_mask is not None:
            prior_body_mask = torch.as_tensor(hp.body_mask, device=device).float()
            if prior_body_mask.ndim != 3:
                raise RuntimeError(f"unexpected human prior body_mask shape: {tuple(prior_body_mask.shape)}")
            prior_body_mask = prior_body_mask.unsqueeze(0)
        if hp.head_mask is not None:
            prior_head_mask = torch.as_tensor(hp.head_mask, device=device).float()
            if prior_head_mask.ndim != 3:
                raise RuntimeError(f"unexpected human prior head_mask shape: {tuple(prior_head_mask.shape)}")
            prior_head_mask = prior_head_mask.unsqueeze(0)
        if hp.face_mask is not None:
            prior_face_mask = torch.as_tensor(hp.face_mask, device=device).float()
            if prior_face_mask.ndim != 3:
                raise RuntimeError(f"unexpected human prior face_mask shape: {tuple(prior_face_mask.shape)}")
            prior_face_mask = prior_face_mask.unsqueeze(0)

    return (
        imgs,
        depth,
        conf,
        point,
        extrinsic,
        intrinsic,
        str(sample.pointmap_source or ""),
        str(sample.pointmap_frame or ""),
        fg_mask,
        fg_mask_source_resolved,
        prior_point,
        prior_valid_mask,
        prior_body_mask,
        prior_head_mask,
        prior_face_mask,
        prior_source,
        prior_pointmap_frame,
    )


def _infer_mask_path_from_image(img_path: str, preferred: str = "auto") -> tuple[Optional[str], str]:
    p = Path(str(img_path))
    if len(p.parts) < 3:
        return None, ""
    cam = p.parent.name
    seq_root = p.parent.parent
    stem = p.stem + ".png"
    pref = str(preferred or "auto").strip().lower()
    if pref not in {"auto", "mask", "mask_cihp"}:
        pref = "auto"
    if pref == "auto":
        order = ["mask", "mask_cihp"]
    else:
        order = [pref]
    for tok in order:
        cand = seq_root / tok / cam / stem
        if cand.is_file():
            return str(cand), tok
    return None, ""


def _normalize_mask_binary_np(mask_like: np.ndarray) -> np.ndarray:
    x = np.asarray(mask_like).astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    if x.ndim != 2:
        raise RuntimeError(f"unexpected mask ndim: {x.ndim}")
    if x.size <= 0:
        return np.zeros_like(x, dtype=np.float32)
    mx = float(np.nanmax(x))
    if mx <= 1.5:
        fg = (x >= 0.5)
    else:
        fg = (x > 0.0)
    return fg.astype(np.float32)


def _erode_mask_tensor(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    """
    Binary erosion on mask tensor (B,V,H,W) using max-pool on inverted mask.
    """
    k = int(max(0, erode_px))
    if k <= 0:
        return mask
    if mask.ndim != 4:
        raise RuntimeError(f"mask must be (B,V,H,W), got {tuple(mask.shape)}")
    b, v, h, w = mask.shape
    kk = int(2 * k + 1)
    x = mask.clamp(0.0, 1.0).reshape(b * v, 1, h, w)
    # erosion(x) = 1 - dilation(1 - x)
    x_inv = 1.0 - x
    dil_inv = F.max_pool2d(x_inv, kernel_size=kk, stride=1, padding=k)
    out = (1.0 - dil_inv).reshape(b, v, h, w)
    return out.clamp(0.0, 1.0)


def _dilate_mask_tensor(mask: torch.Tensor, dilate_px: int) -> torch.Tensor:
    """
    Binary dilation on mask tensor (B,V,H,W) using max-pool.
    """
    k = int(max(0, dilate_px))
    if k <= 0:
        return mask.clamp(0.0, 1.0)
    if mask.ndim != 4:
        raise RuntimeError(f"mask must be (B,V,H,W), got {tuple(mask.shape)}")
    b, v, h, w = mask.shape
    kk = int(2 * k + 1)
    x = mask.clamp(0.0, 1.0).reshape(b * v, 1, h, w)
    out = F.max_pool2d(x, kernel_size=kk, stride=1, padding=k).reshape(b, v, h, w)
    return out.clamp(0.0, 1.0)


def _build_fg_bbox_mask(
    fg_mask01: Optional[torch.Tensor],
    margin_px: int = 12,
    min_side_px: int = 24,
) -> Optional[torch.Tensor]:
    if fg_mask01 is None:
        return None
    fg = fg_mask01.float().clamp(0.0, 1.0)
    if fg.ndim != 4:
        raise RuntimeError(f"fg bbox mask expects (B,V,H,W), got {tuple(fg.shape)}")
    b, v, h, w = fg.shape
    margin = int(max(0, margin_px))
    min_side = int(max(1, min_side_px))
    out = torch.zeros_like(fg)
    for bi in range(b):
        for vi in range(v):
            active = fg[bi, vi] > 0.5
            if int(active.sum().item()) <= 0:
                continue
            ys, xs = torch.where(active)
            y0 = int(ys.min().item())
            y1 = int(ys.max().item())
            x0 = int(xs.min().item())
            x1 = int(xs.max().item())
            y0 = max(0, y0 - margin)
            y1 = min(h - 1, y1 + margin)
            x0 = max(0, x0 - margin)
            x1 = min(w - 1, x1 + margin)
            box_h = y1 - y0 + 1
            box_w = x1 - x0 + 1
            if box_h < min_side:
                extra = min_side - box_h
                y0 = max(0, y0 - extra // 2)
                y1 = min(h - 1, y1 + (extra - extra // 2))
                box_h = y1 - y0 + 1
                if box_h < min_side:
                    if y0 <= 0:
                        y1 = min(h - 1, y0 + min_side - 1)
                    else:
                        y0 = max(0, y1 - min_side + 1)
            if box_w < min_side:
                extra = min_side - box_w
                x0 = max(0, x0 - extra // 2)
                x1 = min(w - 1, x1 + (extra - extra // 2))
                box_w = x1 - x0 + 1
                if box_w < min_side:
                    if x0 <= 0:
                        x1 = min(w - 1, x0 + min_side - 1)
                    else:
                        x0 = max(0, x1 - min_side + 1)
            out[bi, vi, y0:y1 + 1, x0:x1 + 1] = 1.0
    return out


def _build_top_band_mask(
    mask01: Optional[torch.Tensor],
    top_ratio: float,
    min_height_px: int = 8,
) -> Optional[torch.Tensor]:
    if mask01 is None:
        return None
    fg = mask01.float().clamp(0.0, 1.0)
    if fg.ndim != 4:
        raise RuntimeError(f"top band mask expects (B,V,H,W), got {tuple(fg.shape)}")
    ratio = float(np.clip(float(top_ratio), 0.01, 1.0))
    min_h = int(max(1, min_height_px))
    out = torch.zeros_like(fg)
    b, v, h, w = fg.shape
    for bi in range(b):
        for vi in range(v):
            active = fg[bi, vi] > 0.5
            if int(active.sum().item()) <= 0:
                continue
            ys, xs = torch.where(active)
            y0 = int(ys.min().item())
            y1 = int(ys.max().item())
            x0 = int(xs.min().item())
            x1 = int(xs.max().item())
            band_h = max(min_h, int(round((y1 - y0 + 1) * ratio)))
            band_h = min(band_h, y1 - y0 + 1)
            yb1 = min(h - 1, y0 + band_h - 1)
            out[bi, vi, y0:yb1 + 1, x0:x1 + 1] = 1.0
    return out


def _resolve_human_prior_region_mask(
    *,
    region_mode: str,
    valid_mask01: Optional[torch.Tensor],
    body_mask01: Optional[torch.Tensor],
    head_mask01: Optional[torch.Tensor],
    face_mask01: Optional[torch.Tensor],
    head_fallback_top_ratio: float,
    face_fallback_top_ratio: float,
) -> Optional[torch.Tensor]:
    mode = str(region_mode or "off").strip().lower()
    if mode in {"", "off", "none"}:
        return None

    body = body_mask01.float().clamp(0.0, 1.0) if body_mask01 is not None else None
    valid = valid_mask01.float().clamp(0.0, 1.0) if valid_mask01 is not None else None
    head = head_mask01.float().clamp(0.0, 1.0) if head_mask01 is not None else None
    face = face_mask01.float().clamp(0.0, 1.0) if face_mask01 is not None else None
    base = body if body is not None else valid
    if head is None and base is not None:
        head = _build_top_band_mask(base, top_ratio=float(head_fallback_top_ratio), min_height_px=8)
    if face is None:
        face_src = head if head is not None else base
        if face_src is not None:
            face = _build_top_band_mask(face_src, top_ratio=float(face_fallback_top_ratio), min_height_px=6)

    if mode == "all":
        out = valid if valid is not None else base
    elif mode == "body":
        out = body if body is not None else valid
    elif mode == "head":
        out = head if head is not None else body
    elif mode == "face":
        out = face if face is not None else head if head is not None else body
    elif mode == "head_face":
        out = None
        if head is not None and face is not None:
            out = torch.maximum(head, face)
        elif head is not None:
            out = head
        elif face is not None:
            out = face
        else:
            out = body
    else:
        raise ValueError(f"unsupported human prior region_mode={region_mode}")

    if out is None:
        return None
    return out.clamp(0.0, 1.0)


def _build_fg_boundary_band_mask(
    fg_mask01: Optional[torch.Tensor],
    erode_px: int = 3,
) -> Optional[torch.Tensor]:
    if fg_mask01 is None:
        return None
    fg = (fg_mask01.float().clamp(0.0, 1.0) > 0.5).float()
    erode_px_i = int(max(0, erode_px))
    if erode_px_i <= 0:
        interior = fg
    else:
        interior = (_erode_mask_tensor(fg, erode_px_i) > 0.5).float()
    return (fg - interior).clamp(0.0, 1.0)


def _build_fg_structure_region_mask(
    fg_mask01: Optional[torch.Tensor],
    fg_bbox_mask01: Optional[torch.Tensor],
    region_mode: str = "bbox",
    region_erode_px: int = 0,
) -> Optional[torch.Tensor]:
    if fg_bbox_mask01 is None:
        return None
    bbox = (fg_bbox_mask01.float().clamp(0.0, 1.0) > 0.5).float()
    mode = str(region_mode or "bbox").strip().lower()
    if mode == "bbox":
        return bbox
    if fg_mask01 is None:
        return bbox
    fg = (fg_mask01.float().clamp(0.0, 1.0) > 0.5).float()
    erode_px_i = int(max(0, region_erode_px))
    if mode == "bbox_fg_interior":
        if erode_px_i > 0:
            fg = (_erode_mask_tensor(fg, erode_px_i) > 0.5).float()
        return (bbox * fg).clamp(0.0, 1.0)
    raise ValueError(f"unsupported fg_structure_region_mode={region_mode}")


def _build_fg_outside_ring_mask(
    fg_mask01: Optional[torch.Tensor],
    ring_px: int = 3,
) -> Optional[torch.Tensor]:
    if fg_mask01 is None:
        return None
    fg = (fg_mask01.float().clamp(0.0, 1.0) > 0.5).float()
    dil = (_dilate_mask_tensor(fg, int(max(0, ring_px))) > 0.5).float()
    return (dil - fg).clamp(0.0, 1.0)


def _build_fg_structure_target_edge_support_mask(
    target_edge01: torch.Tensor,
    valid01: torch.Tensor,
    fg_structure_region_mask01: Optional[torch.Tensor],
    view_active01: Optional[torch.Tensor] = None,
    mode: str = "off",
    quantile: float = 0.0,
    min_support_px: int = 32,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    info: Dict[str, float] = {
        "fg_structure_target_edge_support_active": 0.0,
        "fg_structure_target_edge_support_views": 0.0,
        "fg_structure_target_edge_support_cover": 0.0,
        "fg_structure_target_edge_support_region_share": 0.0,
        "fg_structure_target_edge_support_threshold_mean": 0.0,
    }
    mode_norm = str(mode or "off").strip().lower()
    if mode_norm == "off" or fg_structure_region_mask01 is None:
        return None, info
    if target_edge01.ndim != 4:
        raise RuntimeError(f"target_edge01 must be (B,V,H,W), got {tuple(target_edge01.shape)}")
    if valid01.ndim != 4:
        raise RuntimeError(f"valid01 must be (B,V,H,W), got {tuple(valid01.shape)}")
    if fg_structure_region_mask01.ndim != 4:
        raise RuntimeError(
        )
    if view_active01 is not None and view_active01.ndim != 2:
        raise RuntimeError(f"view_active01 must be (B,V), got {tuple(view_active01.shape)}")

    edge = target_edge01.float()
    region_active = (
        (valid01.float() > 0.5)
        & (fg_structure_region_mask01.float() > 0.5)
        & torch.isfinite(edge)
    )
    if view_active01 is not None:
        region_active = region_active & (view_active01[:, :, None, None] > 0.5)

    support = torch.zeros_like(edge, dtype=torch.float32)
    thresholds: list[float] = []
    active_views = 0
    region_total = 0
    support_total = 0
    min_support = int(max(1, min_support_px))
    q = float(min(0.999, max(0.0, quantile)))
    b, v, _, _ = edge.shape
    with torch.no_grad():
        for bi in range(b):
            for vi in range(v):
                mask = region_active[bi, vi]
                region_count = int(mask.sum().item())
                if region_count <= 0:
                    continue
                vals = edge[bi, vi][mask].detach()
                if region_count <= min_support:
                    support_mask = mask
                    threshold = float(vals.min().item())
                else:
                    threshold_t = torch.quantile(vals, q)
                    support_mask = mask & (edge[bi, vi] >= threshold_t)
                    support_count = int(support_mask.sum().item())
                    if support_count < min_support:
                        k = min(region_count, min_support)
                        coords = mask.nonzero(as_tuple=False)
                        topk_idx = torch.topk(vals, k=k, largest=True).indices
                        support_mask = torch.zeros_like(mask, dtype=torch.bool)
                        support_mask[coords[topk_idx, 0], coords[topk_idx, 1]] = True
                    threshold = float(threshold_t.item())
                support_count = int(support_mask.sum().item())
                if support_count <= 0:
                    continue
                support[bi, vi][support_mask] = 1.0
                thresholds.append(threshold)
                active_views += 1
                region_total += region_count
                support_total += support_count

    if active_views > 0:
        info["fg_structure_target_edge_support_active"] = 1.0
        info["fg_structure_target_edge_support_views"] = float(active_views)
        info["fg_structure_target_edge_support_cover"] = float(support.mean().item())
        if region_total > 0:
            info["fg_structure_target_edge_support_region_share"] = float(support_total / float(region_total))
        if thresholds:
            info["fg_structure_target_edge_support_threshold_mean"] = float(np.mean(thresholds))
    return support, info


def _build_inside_distance_weight(
    mask01: Optional[torch.Tensor],
    falloff_px: int = 0,
) -> Optional[torch.Tensor]:
    if mask01 is None:
        return None
    mask = (mask01.float().clamp(0.0, 1.0) > 0.5).float()
    falloff = int(max(0, falloff_px))
    if falloff <= 0:
        return mask
    prev = mask
    weight = torch.zeros_like(mask)
    denom = float(falloff + 1)
    for d in range(1, falloff + 1):
        eroded = (_erode_mask_tensor(mask, d) > 0.5).float()
        shell = (prev - eroded).clamp(0.0, 1.0)
        weight = torch.maximum(weight, shell * (float(d) / denom))
        prev = eroded
    weight = torch.maximum(weight, prev)
    return (weight * mask).clamp(0.0, 1.0)


def _binary_component_stats_np(mask01: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask01, dtype=np.float32) > 0.5
    if mask.ndim != 2:
        raise RuntimeError(f"component mask must be 2D, got {tuple(mask.shape)}")
    h, w = mask.shape
    if h <= 0 or w <= 0 or not bool(mask.any()):
        return {
            "component_count": 0,
            "largest_component_share": 0.0,
            "top2_component_share": 0.0,
            "centroid_distance_mean": 0.0,
            "largest_mask": np.zeros_like(mask, dtype=np.float32),
        }

    visited = np.zeros_like(mask, dtype=np.uint8)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    components: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        queue = collections.deque([(int(y0), int(x0))])
        visited[y0, x0] = 1
        coords: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.pop()
            coords.append((cy, cx))
            for dy, dx in neighbors:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or (not mask[ny, nx]):
                    continue
                visited[ny, nx] = 1
                queue.append((ny, nx))
        coord_arr = np.asarray(coords, dtype=np.int32)
        size = int(coord_arr.shape[0])
        centroid = coord_arr.astype(np.float32).mean(axis=0) if size > 0 else np.zeros(2, dtype=np.float32)
        components.append({
            "size": size,
            "coords": coord_arr,
            "centroid": centroid,
        })

    components.sort(key=lambda item: item["size"], reverse=True)
    total = int(sum(int(comp["size"]) for comp in components))
    largest_mask = np.zeros_like(mask, dtype=np.float32)
    if components:
        coords0 = components[0]["coords"]
        largest_mask[coords0[:, 0], coords0[:, 1]] = 1.0
    largest_share = 0.0 if total <= 0 else float(components[0]["size"] / float(total))
    top2 = int(sum(int(comp["size"]) for comp in components[:2]))
    top2_share = 0.0 if total <= 0 else float(top2 / float(total))
    centroid_distance_mean = 0.0
    if len(components) > 1:
        diag = float(max(np.sqrt(float(h * h + w * w)), 1.0))
        ref = np.asarray(components[0]["centroid"], dtype=np.float32)
        dist_num = 0.0
        dist_den = 0.0
        for comp in components[1:]:
            ctr = np.asarray(comp["centroid"], dtype=np.float32)
            dist = float(np.linalg.norm(ctr - ref) / diag)
            wt = float(comp["size"])
            dist_num += dist * wt
            dist_den += wt
        if dist_den > 1e-8:
            centroid_distance_mean = float(dist_num / dist_den)
    return {
        "component_count": int(len(components)),
        "largest_component_share": largest_share,
        "top2_component_share": top2_share,
        "centroid_distance_mean": centroid_distance_mean,
        "largest_mask": largest_mask,
    }


def _summarize_main_support_components(
    weight_map01: Optional[torch.Tensor],
    threshold_ratio: float = 0.25,
    active_eps: float = 1e-6,
) -> tuple[Dict[str, float], Optional[torch.Tensor]]:
    info: Dict[str, float] = {
        "main_support_component_count": 0.0,
        "main_support_largest_component_share": 0.0,
        "main_support_top2_component_share": 0.0,
        "main_support_centroid_distance_mean": 0.0,
        "main_support_component_active_views": 0.0,
    }
    if weight_map01 is None:
        return info, None
    if weight_map01.ndim != 4:
        raise RuntimeError(f"weight_map01 must be (B,V,H,W), got {tuple(weight_map01.shape)}")

    weight = weight_map01.detach().float()
    largest_mask = torch.zeros_like(weight)
    counts: list[float] = []
    largest_shares: list[float] = []
    top2_shares: list[float] = []
    centroid_dists: list[float] = []
    active_views = 0
    b, v = weight.shape[:2]
    thr_ratio = float(max(0.0, threshold_ratio))
    eps = float(max(0.0, active_eps))

    with torch.no_grad():
        for bi in range(b):
            for vi in range(v):
                view = weight[bi, vi]
                max_val = float(view.max().item()) if view.numel() > 0 else 0.0
                if max_val <= eps:
                    continue
                thr = max(eps, thr_ratio * max_val)
                comp_mask = (view >= thr)
                if int(comp_mask.sum().item()) <= 0:
                    comp_mask = view > eps
                stats = _binary_component_stats_np(comp_mask.detach().cpu().numpy())
                if int(stats["component_count"]) <= 0:
                    continue
                active_views += 1
                counts.append(float(stats["component_count"]))
                largest_shares.append(float(stats["largest_component_share"]))
                top2_shares.append(float(stats["top2_component_share"]))
                centroid_dists.append(float(stats["centroid_distance_mean"]))
                largest_mask[bi, vi] = torch.from_numpy(stats["largest_mask"]).to(weight.device, dtype=weight.dtype)

    if active_views > 0:
        info["main_support_component_count"] = float(np.mean(counts))
        info["main_support_largest_component_share"] = float(np.mean(largest_shares))
        info["main_support_top2_component_share"] = float(np.mean(top2_shares))
        info["main_support_centroid_distance_mean"] = float(np.mean(centroid_dists))
        info["main_support_component_active_views"] = float(active_views)
    return info, largest_mask


def _build_largest_component_soft_bias(
    weight_map01: Optional[torch.Tensor],
    threshold_ratio: float = 0.25,
    other_scale: float = 0.35,
    active_eps: float = 1e-6,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    info = {
        "main_support_component_bias_weight_share": 1.0,
    }
    if weight_map01 is None:
        return None, info
    if weight_map01.ndim != 4:
        raise RuntimeError(f"weight_map01 must be (B,V,H,W), got {tuple(weight_map01.shape)}")

    weight = weight_map01.detach().float()
    _, largest_mask = _summarize_main_support_components(
        weight_map01=weight,
        threshold_ratio=threshold_ratio,
        active_eps=active_eps,
    )
    if largest_mask is None:
        return None, info
    other = float(min(1.0, max(0.0, other_scale)))
    bias = torch.zeros_like(weight)
    positive = weight > float(max(0.0, active_eps))
    if other > 0.0:
        bias = torch.where(positive, torch.full_like(weight, other), bias)
    bias = torch.where(largest_mask > 0.5, torch.ones_like(weight), bias)
    pre = float(torch.clamp(weight, min=0.0).sum().item())
    post = float(torch.clamp(weight * bias, min=0.0).sum().item())
    if pre > 1e-8:
        info["main_support_component_bias_weight_share"] = float(post / pre)
    elif post > 1e-8:
        info["main_support_component_bias_weight_share"] = 1.0
    return bias, info


def _summarize_main_support_depth_modes(
    depth_tgt01: Optional[torch.Tensor],
    weight_map01: Optional[torch.Tensor],
    bbox_active01: Optional[torch.Tensor],
    center_quantile: float = 0.5,
    hist_bins: int = 24,
    peak_min_ratio: float = 0.15,
    min_active_px: int = 64,
    active_eps: float = 1e-6,
) -> Dict[str, float]:
    info: Dict[str, float] = {
        "main_support_depth_mode_count": 0.0,
        "main_support_back_mode_share": 0.0,
        "main_support_front_back_gap": 0.0,
        "main_support_depth_hist_peak_ratio": 0.0,
        "main_support_secondary_risk": 0.0,
        "main_support_depth_mode_active_views": 0.0,
    }
    if depth_tgt01 is None or weight_map01 is None or bbox_active01 is None:
        return info
    if depth_tgt01.ndim != 4:
        raise RuntimeError(f"depth_tgt01 must be (B,V,H,W), got {tuple(depth_tgt01.shape)}")
    if weight_map01.ndim != 4:
        raise RuntimeError(f"weight_map01 must be (B,V,H,W), got {tuple(weight_map01.shape)}")
    if bbox_active01.ndim != 4:
        raise RuntimeError(f"bbox_active01 must be (B,V,H,W), got {tuple(bbox_active01.shape)}")

    depth = depth_tgt01.detach().float()
    weight = weight_map01.detach().float().clamp_min(0.0)
    bbox_active = bbox_active01.detach().bool()
    q_center = float(np.clip(float(center_quantile), 0.0, 1.0))
    bins = int(max(8, hist_bins))
    peak_floor = float(np.clip(float(peak_min_ratio), 0.0, 1.0))
    min_px = int(max(1, min_active_px))
    eps = float(max(0.0, active_eps))

    mode_counts: list[float] = []
    back_shares: list[float] = []
    front_back_gaps: list[float] = []
    peak_ratios: list[float] = []
    secondary_risks: list[float] = []
    active_views = 0

    with torch.no_grad():
        hist_edges = torch.linspace(-3.0, 3.0, bins + 1, device=depth.device, dtype=depth.dtype)
        hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        smooth_kernel = torch.tensor([0.25, 0.5, 0.25], device=depth.device, dtype=depth.dtype)
        b, v = depth.shape[:2]
        for bi in range(b):
            for vi in range(v):
                support_mask = (
                    (weight[bi, vi] > eps)
                    & torch.isfinite(weight[bi, vi])
                    & torch.isfinite(depth[bi, vi])
                )
                if int(support_mask.sum().item()) < min_px:
                    continue
                stats_mask = bbox_active[bi, vi] & torch.isfinite(depth[bi, vi])
                if int(stats_mask.sum().item()) < min_px:
                    continue
                vals = depth[bi, vi][stats_mask]
                center = torch.quantile(vals, q_center)
                q25 = torch.quantile(vals, 0.25)
                q75 = torch.quantile(vals, 0.75)
                scale = torch.clamp(q75 - q25, min=1e-3)
                z = ((depth[bi, vi] - center) / scale).clamp(-3.0, 3.0)
                z_support = z[support_mask]
                w_support = weight[bi, vi][support_mask]
                total_w = float(w_support.sum().item())
                if total_w <= 1e-8:
                    continue
                hist = torch.zeros(bins, device=depth.device, dtype=depth.dtype)
                hist_idx = torch.bucketize(z_support, hist_edges[1:-1], right=False)
                hist.scatter_add_(0, hist_idx, w_support)
                if bins >= 3:
                    hist = F.conv1d(
                        hist.view(1, 1, bins),
                        smooth_kernel.view(1, 1, -1),
                        padding=1,
                    ).view(-1)
                max_hist = float(hist.max().item())
                if max_hist <= 1e-8:
                    continue
                peaks: list[int] = []
                thr = peak_floor * max_hist
                for idx in range(bins):
                    cur = float(hist[idx].item())
                    if cur < thr:
                        continue
                    prev = float(hist[idx - 1].item()) if idx > 0 else cur
                    nxt = float(hist[idx + 1].item()) if idx < (bins - 1) else cur
                    if cur >= prev and cur >= nxt:
                        peaks.append(idx)
                if not peaks:
                    peaks = [int(torch.argmax(hist).item())]
                peaks_sorted_by_height = sorted(peaks, key=lambda idx: float(hist[idx].item()), reverse=True)
                mode_count = float(len(peaks))
                peak_ratio = 0.0
                if len(peaks_sorted_by_height) >= 2:
                    denom = float(hist[peaks_sorted_by_height[0]].item())
                    if denom > 1e-8:
                        peak_ratio = float(hist[peaks_sorted_by_height[1]].item() / denom)
                front_idx = min(peaks)
                back_idx = max(peaks)
                gap = 0.0
                back_share = 0.0
                if back_idx > front_idx:
                    front_z = float(hist_centers[front_idx].item())
                    back_z = float(hist_centers[back_idx].item())
                    gap = float(max(0.0, back_z - front_z))
                    split_z = 0.5 * (front_z + back_z)
                    back_share = float(w_support[z_support >= split_z].sum().item() / total_w)
                secondary_risk = float(back_share * peak_ratio)

                active_views += 1
                mode_counts.append(mode_count)
                back_shares.append(back_share)
                front_back_gaps.append(gap)
                peak_ratios.append(peak_ratio)
                secondary_risks.append(secondary_risk)

    if active_views > 0:
        info["main_support_depth_mode_count"] = float(np.mean(mode_counts))
        info["main_support_back_mode_share"] = float(np.mean(back_shares))
        info["main_support_front_back_gap"] = float(np.mean(front_back_gaps))
        info["main_support_depth_hist_peak_ratio"] = float(np.mean(peak_ratios))
        info["main_support_secondary_risk"] = float(np.mean(secondary_risks))
        info["main_support_depth_mode_active_views"] = float(active_views)
    return info


def _build_front_depth_soft_bias(
    depth_tgt01: Optional[torch.Tensor],
    weight_map01: Optional[torch.Tensor],
    bbox_active01: Optional[torch.Tensor],
    mode: str = "off",
    tau: float = 0.75,
    center_quantile: float = 0.55,
    min_active_px: int = 64,
    active_eps: float = 1e-6,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    info: Dict[str, float] = {
        "fg_structure_front_depth_bias_weight_share": 1.0,
        "fg_structure_front_depth_bias_active_views": 0.0,
    }
    mode_norm = str(mode or "off").strip().lower()
    if mode_norm == "off" or depth_tgt01 is None or weight_map01 is None or bbox_active01 is None:
        return None, info
    if depth_tgt01.ndim != 4:
        raise RuntimeError(f"depth_tgt01 must be (B,V,H,W), got {tuple(depth_tgt01.shape)}")
    if weight_map01.ndim != 4:
        raise RuntimeError(f"weight_map01 must be (B,V,H,W), got {tuple(weight_map01.shape)}")
    if bbox_active01.ndim != 4:
        raise RuntimeError(f"bbox_active01 must be (B,V,H,W), got {tuple(bbox_active01.shape)}")

    depth = depth_tgt01.detach().float()
    weight = weight_map01.detach().float()
    bbox_active = bbox_active01.detach().bool()
    bias = torch.zeros_like(weight)
    tau_v = float(max(1e-3, tau))
    q_center = float(np.clip(float(center_quantile), 0.0, 1.0))
    min_px = int(max(1, min_active_px))
    eps = float(max(0.0, active_eps))
    active_views = 0

    with torch.no_grad():
        b, v = depth.shape[:2]
        for bi in range(b):
            for vi in range(v):
                support_mask = (
                    (weight[bi, vi] > eps)
                    & torch.isfinite(weight[bi, vi])
                    & torch.isfinite(depth[bi, vi])
                )
                if int(support_mask.sum().item()) < min_px:
                    continue
                stats_mask = bbox_active[bi, vi] & torch.isfinite(depth[bi, vi])
                if int(stats_mask.sum().item()) < min_px:
                    continue
                vals = depth[bi, vi][stats_mask]
                center = torch.quantile(vals, q_center)
                q25 = torch.quantile(vals, 0.25)
                q75 = torch.quantile(vals, 0.75)
                scale = torch.clamp(q75 - q25, min=1e-3)
                rel = ((depth[bi, vi] - center) / scale).clamp(-3.0, 3.0)
                view_bias = torch.exp(-torch.relu(rel) / tau_v)
                bias[bi, vi] = torch.where(support_mask, view_bias, torch.zeros_like(view_bias))
                active_views += 1

    pre = float(torch.clamp(weight, min=0.0).sum().item())
    post = float(torch.clamp(weight * bias, min=0.0).sum().item())
    if pre > 1e-8:
        info["fg_structure_front_depth_bias_weight_share"] = float(post / pre)
    elif post > 1e-8:
        info["fg_structure_front_depth_bias_weight_share"] = 1.0
    info["fg_structure_front_depth_bias_active_views"] = float(active_views)
    return bias, info


def _robust_abs(x: torch.Tensor, eps: float) -> torch.Tensor:
    e = float(max(0.0, eps))
    if e <= 0.0:
        return x.abs()
    return torch.sqrt(x * x + (e * e)) - e


def _masked_l1_map(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, tgt, reduction="mean")
    diff = (pred - tgt).abs() * mask
    c = float(pred.shape[1]) if pred.ndim >= 4 else 1.0
    denom = (mask.sum(dim=(2, 3)) * c).clamp_min(eps)
    num = diff.sum(dim=(1, 2, 3), keepdim=False)
    return (num / denom.squeeze(1)).mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def _sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(f"sobel input must be (N,C,H,W), got {tuple(x.shape)}")
    _, c, _, _ = x.shape
    kx, ky = _sobel_kernels(x.device, x.dtype)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _fg_structure_depth_edge_loss(
    depth_pred: torch.Tensor,
    depth_tgt: torch.Tensor,
    valid01: torch.Tensor,
    fg_bbox_mask01: Optional[torch.Tensor],
    fg_structure_region_mask01: Optional[torch.Tensor],
    boundary_probe_mask01: Optional[torch.Tensor] = None,
    edge_support_mode: str = "off",
    edge_support_quantile: float = 0.0,
    min_edge_support_px: int = 32,
    edge_weight_mode: str = "uniform",
    boundary_falloff_px: int = 0,
    component_bias_mode: str = "off",
    component_bias_threshold_ratio: float = 0.25,
    component_bias_other_scale: float = 1.0,
    front_depth_bias_mode: str = "off",
    front_depth_bias_tau: float = 0.75,
    front_depth_bias_center_quantile: float = 0.55,
    min_active_px: int = 64,
    min_boundary_px: int = 32,
) -> tuple[torch.Tensor, Dict[str, float]]:
    z = torch.zeros([], device=depth_pred.device, dtype=torch.float32)
    info: Dict[str, float] = {
        "fg_structure_depth_edge_active": 0.0,
        "fg_structure_bbox_cover": 0.0,
        "fg_structure_region_cover": 0.0,
        "fg_structure_effective_cover": 0.0,
        "fg_structure_boundary_probe_cover": 0.0,
        "fg_structure_bbox_active_ratio": 0.0,
        "fg_structure_region_active_ratio": 0.0,
        "fg_structure_boundary_band_active_ratio": 0.0,
        "fg_structure_depth_edge_active_views": 0.0,
        "fg_structure_depth_edge_boundary_active_views": 0.0,
        "fg_structure_depth_edge_loss": 0.0,
        "fg_structure_depth_edge_loss_main": 0.0,
        "fg_structure_depth_edge_loss_boundary_probe": 0.0,
        "fg_structure_depth_edge_loss_interior": 0.0,
        "fg_structure_depth_edge_loss_boundary_band": 0.0,
        "fg_structure_depth_edge_pred_mean": 0.0,
        "fg_structure_depth_edge_tgt_mean": 0.0,
        "fg_structure_depth_edge_boundary_probe_pred_mean": 0.0,
        "fg_structure_depth_edge_boundary_probe_tgt_mean": 0.0,
        "fg_structure_depth_edge_boundary_pred_mean": 0.0,
        "fg_structure_depth_edge_boundary_tgt_mean": 0.0,
        "fg_structure_target_edge_support_active": 0.0,
        "fg_structure_target_edge_support_views": 0.0,
        "fg_structure_target_edge_support_cover": 0.0,
        "fg_structure_target_edge_support_region_share": 0.0,
        "fg_structure_target_edge_support_threshold_mean": 0.0,
        "fg_structure_main_weight_mean": 0.0,
        "fg_structure_boundary_distance_weight_share": 1.0,
        "main_support_component_count": 0.0,
        "main_support_largest_component_share": 0.0,
        "main_support_top2_component_share": 0.0,
        "main_support_centroid_distance_mean": 0.0,
        "main_support_component_active_views": 0.0,
        "main_support_component_bias_weight_share": 1.0,
        "fg_structure_front_depth_bias_weight_share": 1.0,
        "fg_structure_front_depth_bias_active_views": 0.0,
        "main_support_depth_mode_count": 0.0,
        "main_support_back_mode_share": 0.0,
        "main_support_front_back_gap": 0.0,
        "main_support_depth_hist_peak_ratio": 0.0,
        "main_support_secondary_risk": 0.0,
        "main_support_depth_mode_active_views": 0.0,
    }
    if fg_bbox_mask01 is None or fg_structure_region_mask01 is None:
        return z, info
    if depth_pred.ndim != 5 or depth_pred.shape[-1] != 1:
        raise RuntimeError(f"depth_pred must be (B,V,H,W,1), got {tuple(depth_pred.shape)}")
    if depth_tgt.ndim != 5 or depth_tgt.shape[-1] != 1:
        raise RuntimeError(f"depth_tgt must be (B,V,H,W,1), got {tuple(depth_tgt.shape)}")
    if valid01.ndim != 4:
        raise RuntimeError(f"valid01 must be (B,V,H,W), got {tuple(valid01.shape)}")
    if fg_bbox_mask01.ndim != 4:
        raise RuntimeError(f"fg_bbox_mask01 must be (B,V,H,W), got {tuple(fg_bbox_mask01.shape)}")
    if fg_structure_region_mask01.ndim != 4:
        raise RuntimeError(f"fg_structure_region_mask01 must be (B,V,H,W), got {tuple(fg_structure_region_mask01.shape)}")
    if boundary_probe_mask01 is not None and boundary_probe_mask01.ndim != 4:
        raise RuntimeError(f"boundary_probe_mask01 must be (B,V,H,W), got {tuple(boundary_probe_mask01.shape)}")

    pred = depth_pred[..., 0].float()
    tgt = depth_tgt[..., 0].float()
    bbox_active = (
        (valid01.float() > 0.5)
        & (fg_bbox_mask01.float() > 0.5)
        & torch.isfinite(pred)
        & torch.isfinite(tgt)
    )
    b, v, h, w = pred.shape
    pred_norm = torch.zeros_like(pred)
    tgt_norm = torch.zeros_like(tgt)
    view_active = torch.zeros((b, v), device=pred.device, dtype=torch.float32)
    min_active = int(max(1, min_active_px))
    with torch.no_grad():
        for bi in range(b):
            for vi in range(v):
                mask = bbox_active[bi, vi]
                if int(mask.sum().item()) < min_active:
                    continue
                vals = tgt[bi, vi][mask].detach()
                center = vals.median()
                mad = (vals - center).abs().median()
                scale = torch.clamp(mad * 1.4826, min=1e-3)
                pred_norm[bi, vi] = ((pred[bi, vi] - center) / scale).clamp(-3.0, 3.0)
                tgt_norm[bi, vi] = ((tgt[bi, vi] - center) / scale).clamp(-3.0, 3.0)
                view_active[bi, vi] = 1.0

    if float(view_active.sum().item()) <= 0.0:
        return z, info

    pred_edge = _sobel_grad_mag(pred_norm.reshape(b * v, 1, h, w)).reshape(b, v, h, w)
    tgt_edge = _sobel_grad_mag(tgt_norm.reshape(b * v, 1, h, w)).reshape(b, v, h, w)
    region_active = (
        (valid01.float() > 0.5)
        & (fg_structure_region_mask01.float() > 0.5)
        & torch.isfinite(pred)
        & torch.isfinite(tgt)
    )
    edge_mask = region_active.float() * view_active[:, :, None, None]
    if str(edge_support_mode or "off").strip().lower() != "off":
        edge_support_mask, edge_support_info = _build_fg_structure_target_edge_support_mask(
            target_edge01=tgt_edge.detach(),
            valid01=valid01.detach(),
            fg_structure_region_mask01=fg_structure_region_mask01.detach(),
            view_active01=view_active.detach(),
            mode=edge_support_mode,
            quantile=float(edge_support_quantile),
            min_support_px=int(max(1, min_edge_support_px)),
        )
        info.update(edge_support_info)
        if edge_support_mask is not None:
            edge_mask = edge_mask * (edge_support_mask.float() > 0.5).float()
    weight_map = edge_mask
    edge_weight_mode_norm = str(edge_weight_mode or "uniform").strip().lower()
    if edge_weight_mode_norm == "target_edge_sqrt":
        edge_weight = torch.zeros_like(weight_map)
        with torch.no_grad():
            for bi in range(b):
                for vi in range(v):
                    mask = weight_map[bi, vi] > 0.5
                    if int(mask.sum().item()) <= 0:
                        continue
                    vals = tgt_edge[bi, vi][mask]
                    scale = torch.clamp(torch.quantile(vals, 0.95), min=1e-6)
                    edge_weight[bi, vi] = torch.sqrt((tgt_edge[bi, vi] / scale).clamp(0.0, 1.0))
        weight_map = weight_map * edge_weight
    pre_boundary_weight_sum = float(weight_map.sum().item())
    if int(max(0, boundary_falloff_px)) > 0:
        boundary_weight = _build_inside_distance_weight(
            mask01=fg_structure_region_mask01.detach(),
            falloff_px=int(boundary_falloff_px),
        )
        if boundary_weight is not None:
            weight_map = weight_map * boundary_weight.float()
    post_boundary_weight_sum = float(weight_map.sum().item())
    if pre_boundary_weight_sum > 1e-8:
        info["fg_structure_boundary_distance_weight_share"] = float(post_boundary_weight_sum / pre_boundary_weight_sum)
    elif post_boundary_weight_sum > 1e-8:
        info["fg_structure_boundary_distance_weight_share"] = 1.0
    component_bias_mode_norm = str(component_bias_mode or "off").strip().lower()
    if component_bias_mode_norm == "largest_soft":
        component_bias, component_bias_info = _build_largest_component_soft_bias(
            weight_map01=weight_map.detach(),
            threshold_ratio=float(component_bias_threshold_ratio),
            other_scale=float(component_bias_other_scale),
        )
        info.update(component_bias_info)
        if component_bias is not None:
            weight_map = weight_map * component_bias.to(weight_map.device, dtype=weight_map.dtype)
    front_bias_mode_norm = str(front_depth_bias_mode or "off").strip().lower()
    if front_bias_mode_norm == "front_soft":
        front_bias, front_bias_info = _build_front_depth_soft_bias(
            depth_tgt01=tgt.detach(),
            weight_map01=weight_map.detach(),
            bbox_active01=bbox_active.detach(),
            mode=front_bias_mode_norm,
            tau=float(front_depth_bias_tau),
            center_quantile=float(front_depth_bias_center_quantile),
            min_active_px=int(max(1, min_active_px)),
        )
        info.update(front_bias_info)
        if front_bias is not None:
            weight_map = weight_map * front_bias.to(weight_map.device, dtype=weight_map.dtype)
    component_info, _ = _summarize_main_support_components(
        weight_map01=weight_map.detach(),
        threshold_ratio=float(component_bias_threshold_ratio),
    )
    info.update(component_info)
    depth_mode_info = _summarize_main_support_depth_modes(
        depth_tgt01=tgt.detach(),
        weight_map01=weight_map.detach(),
        bbox_active01=bbox_active.detach(),
        center_quantile=float(front_depth_bias_center_quantile),
        min_active_px=int(max(1, min_active_px)),
    )
    info.update(depth_mode_info)
    final_weight_sum = float(weight_map.sum().item())
    if final_weight_sum <= 1e-8:
        return z, info
    loss = _masked_l1_map(
        pred_edge.reshape(b * v, 1, h, w),
        tgt_edge.reshape(b * v, 1, h, w),
        weight_map.reshape(b * v, 1, h, w),
    )
    info["fg_structure_depth_edge_active"] = 1.0
    info["fg_structure_bbox_cover"] = float(fg_bbox_mask01.float().mean().item())
    info["fg_structure_region_cover"] = float(fg_structure_region_mask01.float().mean().item())
    info["fg_structure_effective_cover"] = float((weight_map > 1e-6).float().mean().item())
    info["fg_structure_bbox_active_ratio"] = info["fg_structure_bbox_cover"]
    info["fg_structure_region_active_ratio"] = info["fg_structure_region_cover"]
    info["fg_structure_depth_edge_active_views"] = float(view_active.sum().item())
    info["fg_structure_depth_edge_loss"] = float(loss.item())
    info["fg_structure_depth_edge_loss_main"] = float(loss.item())
    info["fg_structure_depth_edge_loss_interior"] = float(loss.item())
    structure_weight_mask = fg_structure_region_mask01.float() > 0.5
    if int(structure_weight_mask.sum().item()) > 0:
        info["fg_structure_main_weight_mean"] = float(weight_map[structure_weight_mask].mean().item())
    active_edge = weight_map > 1e-6
    if int(active_edge.sum().item()) > 0:
        info["fg_structure_depth_edge_pred_mean"] = float(pred_edge[active_edge].mean().item())
        info["fg_structure_depth_edge_tgt_mean"] = float(tgt_edge[active_edge].mean().item())
    if boundary_probe_mask01 is not None:
        boundary_active = (
            (valid01.float() > 0.5)
            & (boundary_probe_mask01.float() > 0.5)
            & torch.isfinite(pred)
            & torch.isfinite(tgt)
        )
        info["fg_structure_boundary_probe_cover"] = float(boundary_probe_mask01.float().mean().item())
        info["fg_structure_boundary_band_active_ratio"] = info["fg_structure_boundary_probe_cover"]
        boundary_view_active = (
            (boundary_active.reshape(b, v, -1).sum(dim=-1) >= int(max(1, min_boundary_px))).float() * view_active
        )
        info["fg_structure_depth_edge_boundary_active_views"] = float(boundary_view_active.sum().item())
        boundary_mask = boundary_active.float() * boundary_view_active[:, :, None, None]
        if float(boundary_view_active.sum().item()) > 0.0:
            boundary_loss = _masked_l1_map(
                pred_edge.reshape(b * v, 1, h, w),
                tgt_edge.reshape(b * v, 1, h, w),
                boundary_mask.reshape(b * v, 1, h, w),
            )
            info["fg_structure_depth_edge_loss_boundary_probe"] = float(boundary_loss.item())
            info["fg_structure_depth_edge_loss_boundary_band"] = float(boundary_loss.item())
            active_boundary = boundary_mask > 0.5
            if int(active_boundary.sum().item()) > 0:
                info["fg_structure_depth_edge_boundary_probe_pred_mean"] = float(pred_edge[active_boundary].mean().item())
                info["fg_structure_depth_edge_boundary_probe_tgt_mean"] = float(tgt_edge[active_boundary].mean().item())
                info["fg_structure_depth_edge_boundary_pred_mean"] = float(pred_edge[active_boundary].mean().item())
                info["fg_structure_depth_edge_boundary_tgt_mean"] = float(tgt_edge[active_boundary].mean().item())
    return loss, info


def _cam_to_world_point_map_torch(
    point_cam: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
) -> torch.Tensor:
    """
    Convert point map from camera coordinates to world coordinates.
    point_cam: (B,V,H,W,3)
    extrinsic_w2c: (B,V,3,4)
    """
    if point_cam.ndim != 5 or point_cam.shape[-1] != 3:
        raise RuntimeError(f"point_cam must be (B,V,H,W,3), got {tuple(point_cam.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    r = extrinsic_w2c[..., :3, :3]
    t = extrinsic_w2c[..., :3, 3]
    centered = point_cam - t.unsqueeze(-2).unsqueeze(-2)
    return torch.einsum("bvij,bvhwj->bvhwi", r.transpose(-1, -2), centered)


def _self_reproj_err_px(
    point_world: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> float:
    """
    Mean self-reprojection pixel error under world-coordinate assumption.
    """
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        return float("inf")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        return float("inf")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        return float("inf")

    b, v, h, w = point_world.shape[:4]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=point_world.device, dtype=point_world.dtype),
        torch.arange(w, device=point_world.device, dtype=point_world.dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)

    r = extrinsic_w2c[..., :3, :3]
    t = extrinsic_w2c[..., :3, 3]
    cam = torch.einsum("bvij,bvhwj->bvhwi", r, point_world) + t.unsqueeze(-2).unsqueeze(-2)
    z = cam[..., 2]

    fx = intrinsic[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsic[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsic[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsic[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    u = fx * (cam[..., 0] / (z + 1e-8)) + cx
    vv = fy * (cam[..., 1] / (z + 1e-8)) + cy

    err = torch.sqrt((u - xs) * (u - xs) + (vv - ys) * (vv - ys))
    valid = torch.isfinite(err) & torch.isfinite(z) & (z > 1e-6)
    if int(valid.sum().item()) <= 0:
        return float("inf")
    return float(err[valid].mean().item())


def _resolve_point_frame_auto(
    point_map: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[str, Dict[str, float]]:
    """
    Resolve whether a point map is in world or camera coordinates by comparing
    self-reprojection errors.
    """
    cand_world = point_map
    cand_cam2world = _cam_to_world_point_map_torch(point_map, extrinsic_w2c)
    err_world = _self_reproj_err_px(cand_world, extrinsic_w2c, intrinsic)
    err_cam = _self_reproj_err_px(cand_cam2world, extrinsic_w2c, intrinsic)
    frame = "camera" if (err_cam < err_world) else "world"
    info = {
        "point_frame_err_world": float(err_world),
        "point_frame_err_camera": float(err_cam),
    }
    return frame, info


def _depth_to_world_point_map_torch(
    depth_map: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable depth->world unprojection.
    depth_map: (B,V,H,W,1) or (V,H,W,1)
    extrinsic_w2c: (B,V,3,4) or (V,3,4), camera-from-world (OpenCV)
    intrinsic: (B,V,3,3) or (V,3,3)
    returns: (B,V,H,W,3)
    """
    d = depth_map
    e = extrinsic_w2c
    k = intrinsic
    if d.ndim == 4:
        d = d.unsqueeze(0)
    if e.ndim == 3:
        e = e.unsqueeze(0)
    if k.ndim == 3:
        k = k.unsqueeze(0)
    if d.ndim != 5 or d.shape[-1] != 1:
        raise RuntimeError(f"depth_map must be (B,V,H,W,1), got {tuple(d.shape)}")
    if e.ndim != 4 or tuple(e.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic must be (B,V,3,4), got {tuple(e.shape)}")
    if k.ndim != 4 or tuple(k.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(k.shape)}")

    b, v, h, w = d.shape[:4]
    if not (e.shape[0] == b and e.shape[1] == v and k.shape[0] == b and k.shape[1] == v):
        raise RuntimeError(f"batch/view mismatch depth={tuple(d.shape)} extrinsic={tuple(e.shape)} intrinsic={tuple(k.shape)}")

    z = d[..., 0]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=d.device, dtype=d.dtype),
        torch.arange(w, device=d.device, dtype=d.dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)

    fx = k[..., 0, 0].unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
    fy = k[..., 1, 1].unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
    cx = k[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = k[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (B,V,H,W,3)

    r = e[..., :3, :3]
    t = e[..., :3, 3]
    cam_centered = cam - t.unsqueeze(-2).unsqueeze(-2)
    world = torch.einsum("bvij,bvhwj->bvhwi", r.transpose(-1, -2), cam_centered)
    return world


def _point_multiview_support_weight(
    point_world: torch.Tensor,
    depth_tgt: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    valid_mask: torch.Tensor,
    tol_abs: float,
    tol_rel: float,
    weight_floor: float,
    stride: int,
) -> torch.Tensor:
    """
    Build per-pixel support weights for pseudo pointmap using cross-view depth agreement.
    A world point from source view is projected to other views, and gets higher weight if
    projected depth agrees with pseudo depth in those target views.
    """
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        raise RuntimeError(f"point_world must be (B,V,H,W,3), got {tuple(point_world.shape)}")
    if depth_tgt.ndim != 5 or depth_tgt.shape[-1] != 1:
        raise RuntimeError(f"depth_tgt must be (B,V,H,W,1), got {tuple(depth_tgt.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(intrinsic.shape)}")
    if valid_mask.ndim != 4:
        raise RuntimeError(f"valid_mask must be (B,V,H,W), got {tuple(valid_mask.shape)}")

    b, v, h, w = point_world.shape[:4]
    if depth_tgt.shape[:4] != (b, v, h, w):
        raise RuntimeError("shape mismatch between point_world and depth_tgt")
    if valid_mask.shape != (b, v, h, w):
        raise RuntimeError("shape mismatch for valid_mask")
    if extrinsic_w2c.shape[:2] != (b, v) or intrinsic.shape[:2] != (b, v):
        raise RuntimeError("shape mismatch for camera tensors")

    s = max(1, int(stride))
    if s > 1:
        point_lr = point_world[:, :, ::s, ::s, :]
        depth_lr = depth_tgt[:, :, ::s, ::s, 0]
        valid_lr = valid_mask[:, :, ::s, ::s]
    else:
        point_lr = point_world
        depth_lr = depth_tgt[..., 0]
        valid_lr = valid_mask

    _, _, hh, ww = point_lr.shape[:4]
    if w > 1:
        sx = float(max(ww - 1, 0)) / float(w - 1)
    else:
        sx = 1.0
    if h > 1:
        sy = float(max(hh - 1, 0)) / float(h - 1)
    else:
        sy = 1.0
    floor = float(min(1.0, max(0.0, weight_floor)))
    ta = float(max(0.0, tol_abs))
    tr = float(max(0.0, tol_rel))

    out = torch.zeros((b, v, hh, ww), device=point_world.device, dtype=point_world.dtype)
    one = torch.tensor(1.0, device=point_world.device, dtype=point_world.dtype)

    for bi in range(b):
        for vi in range(v):
            xw = point_lr[bi, vi].reshape(-1, 3).to(dtype=torch.float32)
            src_valid = (valid_lr[bi, vi] > 0.5).reshape(-1)
            src_valid = src_valid & torch.isfinite(xw).all(dim=-1)
            if int(src_valid.sum().item()) <= 0:
                continue

            support = torch.zeros((xw.shape[0],), device=point_world.device, dtype=torch.float32)
            total_views = 0

            for vj in range(v):
                if vj == vi:
                    continue
                total_views += 1
                e = extrinsic_w2c[bi, vj]
                k = intrinsic[bi, vj]
                r = e[:3, :3]
                t = e[:3, 3]

                cam = xw @ r.transpose(0, 1) + t.unsqueeze(0)
                z = cam[:, 2]
                proj_ok = src_valid & torch.isfinite(z) & (z > 1e-6)
                if int(proj_ok.sum().item()) <= 0:
                    continue

                u = k[0, 0] * (cam[:, 0] / (z + 1e-8)) + k[0, 2]
                vpx = k[1, 1] * (cam[:, 1] / (z + 1e-8)) + k[1, 2]
                u_lr = u * sx
                v_lr = vpx * sy
                ui = torch.round(u_lr).long()
                vii = torch.round(v_lr).long()

                inside = (
                    proj_ok
                    & (ui >= 0)
                    & (ui < ww)
                    & (vii >= 0)
                    & (vii < hh)
                )
                if int(inside.sum().item()) <= 0:
                    continue

                idx = torch.where(inside)[0]
                dt = depth_lr[bi, vj][vii[idx], ui[idx]]
                zz = z[idx]
                valid_depth = torch.isfinite(dt) & (dt > 1e-6) & torch.isfinite(zz)
                if int(valid_depth.sum().item()) <= 0:
                    continue

                idx2 = idx[valid_depth]
                dt2 = dt[valid_depth]
                zz2 = zz[valid_depth]
                tol = ta + tr * dt2.abs()
                agree = (zz2 - dt2).abs() <= tol
                if int(agree.sum().item()) <= 0:
                    continue

                idx3 = idx2[agree]
                support.index_add_(
                    0,
                    idx3,
                    one.expand(idx3.shape[0]).to(dtype=support.dtype),
                )

            if total_views > 0:
                ratio = support / float(total_views)
            else:
                ratio = torch.zeros_like(support)
            ratio = ratio.reshape(hh, ww)
            weight = floor + (1.0 - floor) * ratio
            weight = weight * (valid_lr[bi, vi] > 0.5).to(weight.dtype)
            out[bi, vi] = weight

    out = out.clamp(0.0, 1.0)
    if s > 1:
        up = F.interpolate(
            out.reshape(b * v, 1, hh, ww),
            size=(h, w),
            mode="nearest",
        ).reshape(b, v, h, w)
        return up.clamp(0.0, 1.0)
    return out


def _point_multiview_depth_reproj_loss(
    point_world: torch.Tensor,
    depth_tgt: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    valid_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor],
    robust_eps: float,
    tol_abs: float,
    tol_rel: float,
    weight_floor: float,
    stride: int,
    max_pairs: int,
    pair_mode: str,
    inlier_only: bool,
    err_quantile: float,
    outlier_boost: float,
    outlier_cap: float,
    tgt_valid_mode: str,
    tgt_valid_floor: float,
    tgt_valid_min_ratio: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Reproject predicted world points to other views and align projected depth with
    pseudo depth. This directly constrains point geometry with cross-view cameras.
    """
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        raise RuntimeError(f"point_world must be (B,V,H,W,3), got {tuple(point_world.shape)}")
    if depth_tgt.ndim != 5 or depth_tgt.shape[-1] != 1:
        raise RuntimeError(f"depth_tgt must be (B,V,H,W,1), got {tuple(depth_tgt.shape)}")
    if valid_mask.ndim != 4:
        raise RuntimeError(f"valid_mask must be (B,V,H,W), got {tuple(valid_mask.shape)}")
    if support_weight is not None and support_weight.ndim != 4:
        raise RuntimeError(f"support_weight must be (B,V,H,W), got {tuple(support_weight.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(intrinsic.shape)}")

    b, v, h, w = point_world.shape[:4]
    s = max(1, int(stride))
    hh = max(1, h // s)
    ww = max(1, w // s)
    if s > 1:
        pw = F.interpolate(
            point_world.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, 3, hh, ww).permute(0, 1, 3, 4, 2)
        dt = F.interpolate(
            depth_tgt.permute(0, 1, 4, 2, 3).reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, hh, ww, 1)
        vm = F.interpolate(
            valid_mask.reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="nearest",
        ).reshape(b, v, hh, ww)
        sw = None
        if support_weight is not None:
            sw = F.interpolate(
                support_weight.reshape(b * v, 1, h, w),
                size=(hh, ww),
                mode="nearest",
            ).reshape(b, v, hh, ww)
    else:
        pw = point_world
        dt = depth_tgt
        vm = valid_mask
        sw = support_weight
    if w > 1:
        sx = float(max(ww - 1, 0)) / float(w - 1)
    else:
        sx = 1.0
    if h > 1:
        sy = float(max(hh - 1, 0)) / float(h - 1)
    else:
        sy = 1.0

    ta = float(max(0.0, tol_abs))
    tr = float(max(0.0, tol_rel))
    wf = float(min(1.0, max(0.0, weight_floor)))
    max_pairs = max(0, int(max_pairs))
    pmode = str(pair_mode or "sequential").strip().lower()
    if pmode not in {"sequential", "adjacent", "farthest", "random"}:
        pmode = "sequential"
    q = float(min(1.0, max(0.0, err_quantile)))
    inlier_only = bool(inlier_only)
    outlier_boost = float(max(0.0, outlier_boost))
    outlier_cap = float(max(1.0, outlier_cap))
    tgt_mode = str(tgt_valid_mode).lower()
    if tgt_mode not in {"hard", "soft", "soft_zero", "off"}:
        tgt_mode = "hard"
    tgt_floor = float(np.clip(float(tgt_valid_floor), 0.0, 1.0))
    tgt_min_ratio = float(np.clip(float(tgt_valid_min_ratio), 0.0, 1.0))

    losses: list[torch.Tensor] = []
    valid_ratios: list[float] = []
    base_proj_ratios: list[float] = []
    inlier_ratios: list[float] = []
    outlier_ratios: list[float] = []
    tgt_valid_ratios: list[float] = []
    skip_low_tgt_valid_pairs = 0

    for bi in range(b):
        for vi in range(v):
            xw = pw[bi, vi]  # (H,W,3)
            pair_candidates = [vj for vj in range(v) if vj != vi]
            if pmode == "adjacent":
                pair_candidates = sorted(
                    pair_candidates,
                    key=lambda j: (min(abs(j - vi), v - abs(j - vi)), j),
                )
            elif pmode == "farthest":
                pair_candidates = sorted(
                    pair_candidates,
                    key=lambda j: (-min(abs(j - vi), v - abs(j - vi)), j),
                )
            elif pmode == "random":
                pair_candidates = random.sample(pair_candidates, k=len(pair_candidates))

            pair_cnt = 0
            for vj in pair_candidates:
                if max_pairs > 0 and pair_cnt >= max_pairs:
                    break
                pair_cnt += 1

                e = extrinsic_w2c[bi, vj]
                k = intrinsic[bi, vj]
                r = e[:3, :3]
                t = e[:3, 3]

                cam = torch.einsum("ij,hwj->hwi", r, xw) + t.view(1, 1, 3)
                z = cam[..., 2]
                u = k[0, 0] * (cam[..., 0] / (z + 1e-8)) + k[0, 2]
                vv = k[1, 1] * (cam[..., 1] / (z + 1e-8)) + k[1, 2]
                u_lr = u * sx
                vv_lr = vv * sy

                gx = (u_lr / max(1.0, float(ww - 1))) * 2.0 - 1.0
                gy = (vv_lr / max(1.0, float(hh - 1))) * 2.0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(dtype=torch.float32)

                depth_map = dt[bi, vj, ..., 0].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
                valid_map = vm[bi, vj].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
                depth_s = F.grid_sample(depth_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[
                    0, 0
                ]
                valid_s = F.grid_sample(valid_map, grid, mode="nearest", padding_mode="zeros", align_corners=True)[
                    0, 0
                ]

                inside = (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)
                src_proj_valid = (
                    (vm[bi, vi] > 0.5)
                    & inside
                    & torch.isfinite(z)
                    & (z > 1e-6)
                )
                src_proj_ratio = float(src_proj_valid.float().mean().item())
                valid = (
                    src_proj_valid
                    & torch.isfinite(depth_s)
                    & (depth_s > 1e-6)
                )
                if tgt_mode == "hard":
                    valid = valid & (valid_s > 0.5)
                valid_ratio = float(valid.float().mean().item())
                if valid_ratio <= 0.0:
                    continue
                tgt_valid_ratio_cur = (
                    float(valid_s[src_proj_valid].mean().item())
                    if int(src_proj_valid.sum().item()) > 0
                    else 0.0
                )
                if tgt_valid_ratio_cur < tgt_min_ratio:
                    skip_low_tgt_valid_pairs += 1
                    continue

                err = _robust_abs(z - depth_s, robust_eps)
                tol = ta + tr * depth_s.abs()
                inlier = (err <= tol).to(dtype=err.dtype)

                wt = valid.to(dtype=err.dtype)
                if tgt_mode == "soft":
                    tgt_w = torch.clamp(valid_s.to(dtype=err.dtype), min=tgt_floor, max=1.0)
                    wt = wt * tgt_w
                elif tgt_mode == "soft_zero":
                    valid_s_f = valid_s.to(dtype=err.dtype)
                    tgt_w = torch.clamp(valid_s_f, min=tgt_floor, max=1.0)
                    # Do not apply floor on truly invalid target samples.
                    tgt_w = torch.where(valid_s_f > 1e-6, tgt_w, torch.zeros_like(tgt_w))
                    wt = wt * tgt_w
                if sw is not None:
                    wt = wt * sw[bi, vi].to(dtype=err.dtype).clamp(0.0, 1.0)

                if inlier_only:
                    wt = wt * inlier
                else:
                    inlier_w = torch.clamp(wf + (1.0 - wf) * inlier, 0.0, 1.0)
                    if outlier_boost > 0.0:
                        # Emphasize cross-view outliers to actively suppress ghost geometry.
                        boost = 1.0 + outlier_boost * (1.0 - inlier)
                        boost = torch.clamp(boost, 1.0, outlier_cap)
                        inlier_w = inlier_w * boost
                    wt = wt * inlier_w

                if q < 1.0:
                    valid_w = wt > 0.0
                    if int(valid_w.sum().item()) >= 16:
                        err_det = err.detach()
                        qv = torch.quantile(err_det[valid_w], q)
                        keep = (err_det <= qv).to(dtype=wt.dtype)
                        wt = wt * keep

                denom = wt.sum() + 1e-6
                if float(denom.detach().item()) <= 1e-8:
                    continue
                pair_loss = (err * wt).sum() / denom
                losses.append(pair_loss)
                valid_ratios.append(valid_ratio)
                base_proj_ratios.append(src_proj_ratio)
                tgt_valid_ratios.append(tgt_valid_ratio_cur)
                valid_w_for_ratio = wt > 0.0
                if int(valid_w_for_ratio.sum().item()) > 0:
                    inlier_ratio = float(inlier[valid_w_for_ratio].mean().item())
                    inlier_ratios.append(inlier_ratio)
                    outlier_ratios.append(1.0 - inlier_ratio)
                else:
                    inlier_ratios.append(0.0)
                    outlier_ratios.append(1.0)

    if losses:
        loss = torch.stack(losses).mean()
        info = {
            "point_mv_depth_pairs": float(len(losses)),
            "point_mv_depth_base_proj_ratio": float(np.mean(base_proj_ratios)) if base_proj_ratios else 0.0,
            "point_mv_depth_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "point_mv_depth_tgt_valid_ratio": float(np.mean(tgt_valid_ratios)) if tgt_valid_ratios else 0.0,
            "point_mv_depth_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
            "point_mv_depth_outlier_ratio": float(np.mean(outlier_ratios)) if outlier_ratios else 1.0,
            "point_mv_depth_skip_low_tgt_valid_pairs": float(skip_low_tgt_valid_pairs),
        }
        return loss, info

    zero = torch.zeros([], device=point_world.device, dtype=point_world.dtype)
    return zero, {
        "point_mv_depth_pairs": 0.0,
        "point_mv_depth_base_proj_ratio": 0.0,
        "point_mv_depth_valid_ratio": 0.0,
        "point_mv_depth_tgt_valid_ratio": 0.0,
        "point_mv_depth_inlier_ratio": 0.0,
        "point_mv_depth_outlier_ratio": 0.0,
        "point_mv_depth_skip_low_tgt_valid_pairs": float(skip_low_tgt_valid_pairs),
    }


def _point_multiview_projected_mask_pairs(
    point_world: torch.Tensor,
    target_mask_tgt: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    src_valid_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor],
    stride: int,
    max_pairs: int,
    pair_mode: str,
    soft_blur_px: int,
    soft_blur_iters: int,
    soft_mix: float,
):
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        raise RuntimeError(f"point_world must be (B,V,H,W,3), got {tuple(point_world.shape)}")
    if target_mask_tgt.ndim != 4:
        raise RuntimeError(f"target_mask_tgt must be (B,V,H,W), got {tuple(target_mask_tgt.shape)}")
    if src_valid_mask.ndim != 4:
        raise RuntimeError(f"src_valid_mask must be (B,V,H,W), got {tuple(src_valid_mask.shape)}")
    if support_weight is not None and support_weight.ndim != 4:
        raise RuntimeError(f"support_weight must be (B,V,H,W), got {tuple(support_weight.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(intrinsic.shape)}")

    b, v, h, w = point_world.shape[:4]
    if target_mask_tgt.shape != (b, v, h, w):
        raise RuntimeError("target_mask_tgt shape mismatch")
    if src_valid_mask.shape != (b, v, h, w):
        raise RuntimeError("src_valid_mask shape mismatch")
    if support_weight is not None and support_weight.shape != (b, v, h, w):
        raise RuntimeError("support_weight shape mismatch")

    s = max(1, int(stride))
    hh = max(1, h // s)
    ww = max(1, w // s)
    if s > 1:
        pw = F.interpolate(
            point_world.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, 3, hh, ww).permute(0, 1, 3, 4, 2)
        tgt_mask = F.interpolate(
            target_mask_tgt.reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="nearest",
        ).reshape(b, v, hh, ww)
        vm = F.interpolate(
            src_valid_mask.reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="nearest",
        ).reshape(b, v, hh, ww)
        sw = None
        if support_weight is not None:
            sw = F.interpolate(
                support_weight.reshape(b * v, 1, h, w),
                size=(hh, ww),
                mode="nearest",
            ).reshape(b, v, hh, ww)
    else:
        pw = point_world
        tgt_mask = target_mask_tgt
        vm = src_valid_mask
        sw = support_weight
    if w > 1:
        sx = float(max(ww - 1, 0)) / float(w - 1)
    else:
        sx = 1.0
    if h > 1:
        sy = float(max(hh - 1, 0)) / float(h - 1)
    else:
        sy = 1.0

    max_pairs = max(0, int(max_pairs))
    pmode = str(pair_mode or "sequential").strip().lower()
    if pmode not in {"sequential", "adjacent", "farthest", "random"}:
        pmode = "sequential"
    soft_blur_px = max(0, int(soft_blur_px))
    soft_blur_iters = max(1, int(soft_blur_iters))
    soft_mix = float(np.clip(float(soft_mix), 0.0, 1.0))

    for bi in range(b):
        target_soft_cache: dict[int, torch.Tensor] = {}
        for vi in range(v):
            xw = pw[bi, vi]
            pair_candidates = [vj for vj in range(v) if vj != vi]
            if pmode == "adjacent":
                pair_candidates = sorted(
                    pair_candidates,
                    key=lambda j: (min(abs(j - vi), v - abs(j - vi)), j),
                )
            elif pmode == "farthest":
                pair_candidates = sorted(
                    pair_candidates,
                    key=lambda j: (-min(abs(j - vi), v - abs(j - vi)), j),
                )
            elif pmode == "random":
                pair_candidates = random.sample(pair_candidates, k=len(pair_candidates))

            pair_cnt = 0
            for vj in pair_candidates:
                if max_pairs > 0 and pair_cnt >= max_pairs:
                    break
                pair_cnt += 1

                e = extrinsic_w2c[bi, vj]
                k = intrinsic[bi, vj]
                r = e[:3, :3]
                t = e[:3, 3]

                cam = torch.einsum("ij,hwj->hwi", r, xw) + t.view(1, 1, 3)
                z = cam[..., 2]
                u = k[0, 0] * (cam[..., 0] / (z + 1e-8)) + k[0, 2]
                vv = k[1, 1] * (cam[..., 1] / (z + 1e-8)) + k[1, 2]
                u_lr = u * sx
                vv_lr = vv * sy

                gx = (u_lr / max(1.0, float(ww - 1))) * 2.0 - 1.0
                gy = (vv_lr / max(1.0, float(hh - 1))) * 2.0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(dtype=torch.float32)

                target_map = tgt_mask[bi, vj].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
                sampled = F.grid_sample(target_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
                sampled_soft = None
                if (soft_mix > 0.0) and (soft_blur_px > 0):
                    target_soft_map = target_soft_cache.get(vj)
                    if target_soft_map is None:
                        target_soft_map = target_map
                        kernel = int(2 * soft_blur_px + 1)
                        for _ in range(soft_blur_iters):
                            target_soft_map = F.avg_pool2d(
                                target_soft_map,
                                kernel_size=kernel,
                                stride=1,
                                padding=soft_blur_px,
                            )
                        target_soft_map = target_soft_map.clamp(0.0, 1.0)
                        target_soft_cache[vj] = target_soft_map
                    sampled_soft = F.grid_sample(
                        target_soft_map,
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )[0, 0]

                inside = (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)
                src_proj_valid = (
                    (vm[bi, vi] > 0.5)
                    & inside
                    & torch.isfinite(z)
                    & (z > 1e-6)
                )
                src_proj_ratio = float(src_proj_valid.float().mean().item())
                valid = src_proj_valid & torch.isfinite(sampled)
                valid_ratio = float(valid.float().mean().item())
                if valid_ratio <= 0.0:
                    continue

                wt = valid.to(dtype=torch.float32)
                if sw is not None:
                    wt = wt * sw[bi, vi].to(dtype=torch.float32).clamp(0.0, 1.0)

                yield {
                    "bi": int(bi),
                    "vi": int(vi),
                    "vj": int(vj),
                    "sampled": sampled,
                    "sampled_soft": sampled_soft,
                    "src_proj_valid": src_proj_valid,
                    "valid": valid,
                    "valid_ratio": valid_ratio,
                    "src_proj_ratio": src_proj_ratio,
                    "wt": wt,
                    "target_active_px": int((target_mask_tgt[bi, vj] > 0.5).sum().item()),
                }


def _point_multiview_fg_reproj_loss(
    point_world: torch.Tensor,
    fg_mask_tgt: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    src_valid_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor],
    robust_eps: float,
    stride: int,
    max_pairs: int,
    pair_mode: str,
    min_tgt_fg_ratio: float,
    hit_thr: float,
    soft_blur_px: int,
    soft_blur_iters: int,
    soft_mix: float,
    soft_hit_thr: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    losses: list[torch.Tensor] = []
    valid_ratios: list[float] = []
    base_proj_ratios: list[float] = []
    tgt_fg_ratios: list[float] = []
    miss_ratios: list[float] = []
    soft_fg_ratios: list[float] = []
    soft_miss_ratios: list[float] = []
    skip_low_tgt_fg_pairs = 0
    min_tgt_fg_ratio = float(np.clip(float(min_tgt_fg_ratio), 0.0, 1.0))
    hit_thr = float(np.clip(float(hit_thr), 1e-4, 1.0))
    soft_hit_thr = float(np.clip(float(soft_hit_thr), 1e-4, 1.0))

    for pair in _point_multiview_projected_mask_pairs(
        point_world=point_world,
        target_mask_tgt=fg_mask_tgt,
        extrinsic_w2c=extrinsic_w2c,
        intrinsic=intrinsic,
        src_valid_mask=src_valid_mask,
        support_weight=support_weight,
        stride=stride,
        max_pairs=max_pairs,
        pair_mode=pair_mode,
        soft_blur_px=soft_blur_px,
        soft_blur_iters=soft_blur_iters,
        soft_mix=soft_mix,
    ):
        fg_s = pair["sampled"]
        fg_soft_s = pair["sampled_soft"]
        src_proj_valid = pair["src_proj_valid"]
        wt = pair["wt"]

        tgt_fg_ratio_cur = (
            float(fg_s[src_proj_valid].mean().item())
            if int(src_proj_valid.sum().item()) > 0
            else 0.0
        )
        if tgt_fg_ratio_cur < min_tgt_fg_ratio:
            skip_low_tgt_fg_pairs += 1
            continue

        miss_hard = torch.relu(hit_thr - fg_s) / hit_thr
        miss = miss_hard
        if fg_soft_s is not None:
            miss_soft = torch.relu(soft_hit_thr - fg_soft_s) / soft_hit_thr
            miss = (1.0 - soft_mix) * miss_hard + soft_mix * miss_soft
        err = _robust_abs(miss, robust_eps)
        denom = wt.sum() + 1e-6
        if float(denom.detach().item()) <= 1e-8:
            continue
        pair_loss = (err * wt).sum() / denom
        losses.append(pair_loss)
        valid_ratios.append(pair["valid_ratio"])
        base_proj_ratios.append(pair["src_proj_ratio"])
        tgt_fg_ratios.append(tgt_fg_ratio_cur)
        valid_w = wt > 0.0
        if int(valid_w.sum().item()) > 0:
            miss_ratios.append(float((fg_s[valid_w] < hit_thr).float().mean().item()))
            if fg_soft_s is not None:
                soft_fg_ratios.append(float(fg_soft_s[valid_w].mean().item()))
                soft_miss_ratios.append(float((fg_soft_s[valid_w] < soft_hit_thr).float().mean().item()))
        else:
            miss_ratios.append(1.0)
            if fg_soft_s is not None:
                soft_fg_ratios.append(0.0)
                soft_miss_ratios.append(1.0)

    if losses:
        loss = torch.stack(losses).mean()
        info = {
            "point_mv_mask_pairs": float(len(losses)),
            "point_mv_mask_base_proj_ratio": float(np.mean(base_proj_ratios)) if base_proj_ratios else 0.0,
            "point_mv_mask_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "point_mv_mask_tgt_fg_ratio": float(np.mean(tgt_fg_ratios)) if tgt_fg_ratios else 0.0,
            "point_mv_mask_miss_ratio": float(np.mean(miss_ratios)) if miss_ratios else 1.0,
            "point_mv_mask_skip_low_tgt_fg_pairs": float(skip_low_tgt_fg_pairs),
            "point_mv_mask_soft_mix": float(soft_mix),
            "point_mv_mask_soft_fg_ratio": float(np.mean(soft_fg_ratios)) if soft_fg_ratios else 0.0,
            "point_mv_mask_soft_miss_ratio": float(np.mean(soft_miss_ratios)) if soft_miss_ratios else 0.0,
        }
        return loss, info

    zero = torch.zeros([], device=point_world.device, dtype=point_world.dtype)
    return zero, {
        "point_mv_mask_pairs": 0.0,
        "point_mv_mask_base_proj_ratio": 0.0,
        "point_mv_mask_valid_ratio": 0.0,
        "point_mv_mask_tgt_fg_ratio": 0.0,
        "point_mv_mask_miss_ratio": 0.0,
        "point_mv_mask_skip_low_tgt_fg_pairs": float(skip_low_tgt_fg_pairs),
        "point_mv_mask_soft_mix": float(soft_mix),
        "point_mv_mask_soft_fg_ratio": 0.0,
        "point_mv_mask_soft_miss_ratio": 0.0,
    }


def _point_mv_outside_ring_loss(
    point_world: torch.Tensor,
    outside_ring_mask_tgt: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    src_valid_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor],
    robust_eps: float,
    stride: int,
    min_active_ring_px: int = 32,
) -> tuple[torch.Tensor, Dict[str, float]]:
    losses: list[torch.Tensor] = []
    hit_ratios: list[float] = []
    valid_ratios: list[float] = []
    base_proj_ratios: list[float] = []
    active_targets: set[tuple[int, int]] = set()
    min_active_ring = int(max(1, min_active_ring_px))

    for pair in _point_multiview_projected_mask_pairs(
        point_world=point_world,
        target_mask_tgt=outside_ring_mask_tgt,
        extrinsic_w2c=extrinsic_w2c,
        intrinsic=intrinsic,
        src_valid_mask=src_valid_mask,
        support_weight=support_weight,
        stride=stride,
        max_pairs=1,
        pair_mode="adjacent",
        soft_blur_px=0,
        soft_blur_iters=1,
        soft_mix=0.0,
    ):
        if int(pair["target_active_px"]) < min_active_ring:
            continue
        wt = pair["wt"]
        denom = wt.sum() + 1e-6
        if float(denom.detach().item()) <= 1e-8:
            continue
        sampled = pair["sampled"]
        pair_loss = (_robust_abs(sampled, robust_eps) * wt).sum() / denom
        losses.append(pair_loss)
        base_proj_ratios.append(pair["src_proj_ratio"])
        valid_ratios.append(pair["valid_ratio"])
        valid_w = wt > 0.0
        if int(valid_w.sum().item()) > 0:
            hit_ratios.append(float((sampled[valid_w] > 0.5).float().mean().item()))
            active_targets.add((int(pair["bi"]), int(pair["vj"])))

    if losses:
        loss = torch.stack(losses).mean()
        return loss, {
            "point_mv_outside_ring_active": 1.0,
            "point_mv_outside_ring_active_views": float(len(active_targets)),
            "point_mv_outside_ring_hit_ratio": float(np.mean(hit_ratios)) if hit_ratios else 0.0,
            "point_mv_outside_ring_loss": float(loss.item()),
            "point_mv_outside_ring_base_proj_ratio": float(np.mean(base_proj_ratios)) if base_proj_ratios else 0.0,
            "point_mv_outside_ring_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        }

    zero = torch.zeros([], device=point_world.device, dtype=point_world.dtype)
    return zero, {
        "point_mv_outside_ring_active": 0.0,
        "point_mv_outside_ring_active_views": 0.0,
        "point_mv_outside_ring_hit_ratio": 0.0,
        "point_mv_outside_ring_loss": 0.0,
        "point_mv_outside_ring_base_proj_ratio": 0.0,
        "point_mv_outside_ring_valid_ratio": 0.0,
    }


def _point_self_reprojection_loss(
    point_world: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    valid_mask: torch.Tensor,
    robust_eps: float,
    clamp_px: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Encourage predicted world point at each pixel to lie on the same camera ray
    by reprojection to its own view.
    """
    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        raise RuntimeError(f"point_world must be (B,V,H,W,3), got {tuple(point_world.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(intrinsic.shape)}")
    if valid_mask.ndim != 4:
        raise RuntimeError(f"valid_mask must be (B,V,H,W), got {tuple(valid_mask.shape)}")

    b, v, h, w = point_world.shape[:4]
    if extrinsic_w2c.shape[:2] != (b, v) or intrinsic.shape[:2] != (b, v):
        raise RuntimeError("camera shape mismatch")
    if valid_mask.shape != (b, v, h, w):
        raise RuntimeError("valid_mask shape mismatch")

    ys, xs = torch.meshgrid(
        torch.arange(h, device=point_world.device, dtype=point_world.dtype),
        torch.arange(w, device=point_world.device, dtype=point_world.dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)

    losses: list[torch.Tensor] = []
    mean_px: list[float] = []
    p95_px: list[float] = []
    clamp_v = float(max(0.0, clamp_px))
    scale = float(max(1.0, max(h, w)))

    for bi in range(b):
        for vi in range(v):
            e = extrinsic_w2c[bi, vi]
            k = intrinsic[bi, vi]
            r = e[:3, :3]
            t = e[:3, 3]

            xw = point_world[bi, vi]
            cam = torch.einsum("ij,hwj->hwi", r, xw) + t.view(1, 1, 3)
            z = cam[..., 2]
            u = k[0, 0] * (cam[..., 0] / (z + 1e-8)) + k[0, 2]
            vv = k[1, 1] * (cam[..., 1] / (z + 1e-8)) + k[1, 2]

            du = u - xs[0, 0]
            dv = vv - ys[0, 0]
            if clamp_v > 0.0:
                du = du.clamp(-clamp_v, clamp_v)
                dv = dv.clamp(-clamp_v, clamp_v)

            valid = (
                (valid_mask[bi, vi] > 0.5)
                & torch.isfinite(du)
                & torch.isfinite(dv)
                & torch.isfinite(z)
                & (z > 1e-6)
            )
            if int(valid.sum().item()) <= 0:
                continue

            err_px = torch.sqrt(du * du + dv * dv)
            err = (_robust_abs(du, robust_eps) + _robust_abs(dv, robust_eps)) / scale
            wt = valid.to(dtype=err.dtype)
            denom = wt.sum() + 1e-6
            losses.append((err * wt).sum() / denom)

            ep = err_px[valid].detach().float()
            mean_px.append(float(ep.mean().item()))
            p95_px.append(float(torch.quantile(ep, 0.95).item()))

    if losses:
        out = torch.stack(losses).mean()
        return out, {
            "point_reproj_views": float(len(losses)),
            "point_reproj_px_mean": float(np.mean(mean_px)) if mean_px else 0.0,
            "point_reproj_px_p95": float(np.mean(p95_px)) if p95_px else 0.0,
        }
    zero = torch.zeros([], device=point_world.device, dtype=point_world.dtype)
    return zero, {
        "point_reproj_views": 0.0,
        "point_reproj_px_mean": 0.0,
        "point_reproj_px_p95": 0.0,
    }


def _camera_pose_losses(
    pred_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    robust_eps: float,
    rot_weight: float,
    fov_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Camera supervision on pose encoding:
    - translation: robust L1
    - rotation: quaternion cosine distance with sign ambiguity handled by abs(dot)
    - fov: robust L1
    """
    trans_pred = pred_pose[..., :3]
    quat_pred = pred_pose[..., 3:7]
    fov_pred = pred_pose[..., 7:9]

    trans_tgt = tgt_pose[..., :3]
    quat_tgt = tgt_pose[..., 3:7]
    fov_tgt = tgt_pose[..., 7:9]

    loss_trans = _robust_abs(trans_pred - trans_tgt, robust_eps).mean()

    quat_pred_n = F.normalize(quat_pred, dim=-1, eps=1e-6)
    quat_tgt_n = F.normalize(quat_tgt, dim=-1, eps=1e-6)
    quat_dot = (quat_pred_n * quat_tgt_n).sum(dim=-1).abs().clamp(0.0, 1.0)
    loss_rot = (1.0 - quat_dot).mean()

    loss_fov = _robust_abs(fov_pred - fov_tgt, robust_eps).mean()
    total = loss_trans + float(rot_weight) * loss_rot + float(fov_weight) * loss_fov
    return total, {
        "loss_cam_trans": float(loss_trans.detach().item()),
        "loss_cam_rot": float(loss_rot.detach().item()),
        "loss_cam_fov": float(loss_fov.detach().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser("finetune_vggt_pseudo")
    parser.add_argument("--zju_root", type=str, default=_env_str("VGGT_ZJU_ROOT", "/mnt/data/zju_mocap"))
    parser.add_argument("--seq_names", type=str, default=_env_str("VGGT_SEQ_NAMES", "CoreView_390"))
    parser.add_argument("--cam_names", type=str, default=_env_str("VGGT_CAM_NAMES", ""))
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=_env_str("VGGT_CKPT", _env_str("VGGT_PRECOMPUTE_CKPT", "model.pt")),
    )
    parser.add_argument("--resume_ckpt", type=str, default=_env_str("VGGT_FT_RESUME_CKPT", ""))
    parser.add_argument("--epochs", type=int, default=_env_int("VGGT_FT_EPOCHS", _env_int("VGGT_EPOCHS", 2)))
    parser.add_argument("--max_frames", type=int, default=_env_int("VGGT_FT_MAX_FRAMES", _env_int("VGGT_MAX_FRAMES", 0)))
    parser.add_argument("--lr", type=float, default=_env_float("VGGT_FT_LR", 2e-5))
    parser.add_argument("--weight_decay", type=float, default=_env_float("VGGT_FT_WEIGHT_DECAY", 1e-4))
    parser.add_argument("--lambda_depth", type=float, default=_env_float("VGGT_FT_LAMBDA_DEPTH", 1.0))
    parser.add_argument("--lambda_point", type=float, default=_env_float("VGGT_FT_LAMBDA_POINT", 0.5))
    parser.add_argument("--lambda_point_prior", type=float, default=_env_float("VGGT_FT_LAMBDA_POINT_PRIOR", 0.0))
    parser.add_argument("--lambda_point_reproj", type=float, default=_env_float("VGGT_FT_LAMBDA_POINT_REPROJ", 0.0))
    parser.add_argument(
        "--lambda_point_normal_consis",
        type=float,
        default=_env_float("VGGT_FT_LAMBDA_POINT_NORMAL_CONSIS", 0.0),
    )
    parser.add_argument("--lambda_conf", type=float, default=_env_float("VGGT_FT_LAMBDA_CONF", 0.05))
    parser.add_argument("--lambda_geom_cons", type=float, default=_env_float("VGGT_FT_LAMBDA_GEOM_CONS", 0.0))
    parser.add_argument("--lambda_cam", type=float, default=_env_float("VGGT_FT_LAMBDA_CAM", 0.0))
    parser.add_argument(
        "--lambda_conf_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_LAMBDA_CONF_WARMUP_STEPS", 0)),
        help="Linear warmup steps for effective lambda_conf.",
    )
    parser.add_argument(
        "--lambda_cam_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_LAMBDA_CAM_WARMUP_STEPS", 0)),
        help="Linear warmup steps for effective lambda_cam.",
    )
    parser.add_argument("--cam_rot_weight", type=float, default=_env_float("VGGT_FT_CAM_ROT_WEIGHT", 1.0))
    parser.add_argument("--cam_fov_weight", type=float, default=_env_float("VGGT_FT_CAM_FOV_WEIGHT", 0.2))
    parser.add_argument("--cam_warmup_steps", type=int, default=max(1, _env_int("VGGT_FT_CAM_WARMUP_STEPS", 40)))
    parser.add_argument("--jitter", type=float, default=_env_float("VGGT_FT_JITTER", 0.12))
    parser.add_argument("--noise_std", type=float, default=_env_float("VGGT_FT_NOISE_STD", 0.01))
    parser.add_argument("--robust_l1_eps", type=float, default=_env_float("VGGT_FT_ROBUST_L1_EPS", 0.0))
    parser.add_argument(
        "--conf_weight_thr",
        type=float,
        default=_env_float("VGGT_FT_CONF_WEIGHT_THR", 0.0),
        help="Confidence gate threshold in [0,1]. Values below threshold are down-weighted to 0.",
    )
    parser.add_argument(
        "--conf_weight_gamma",
        type=float,
        default=_env_float("VGGT_FT_CONF_WEIGHT_GAMMA", 1.0),
        help="Confidence gate exponent (>1 sharpens high-confidence region).",
    )
    parser.add_argument(
        "--conf_weight_per_view_quantile",
        type=float,
        default=_env_float("VGGT_FT_CONF_WEIGHT_PER_VIEW_QUANTILE", 0.0),
        help="Per-view confidence quantile keep ratio in [0,1]; 0 disables per-view quantile gating.",
    )
    parser.add_argument(
        "--conf_weight_per_view_min_valid",
        type=int,
        default=max(1, _env_int("VGGT_FT_CONF_WEIGHT_PER_VIEW_MIN_VALID", 16)),
        help="Minimum valid pixels required to apply per-view quantile confidence gating.",
    )
    parser.add_argument(
        "--gram_dyn_enable",
        type=str,
        default=_env_str("VGGT_FT_GRAM_DYN_ENABLE", "off"),
        choices=["on", "off"],
        help="Enable token-Gram dynamic soft weighting.",
    )
    parser.add_argument(
        "--gram_dyn_layer_idx",
        type=int,
        default=_env_int("VGGT_FT_GRAM_DYN_LAYER_IDX", -1),
        help="Aggregator layer index for Gram dynamic weighting. Negative uses python-style indexing.",
    )
    parser.add_argument(
        "--gram_dyn_quantile",
        type=float,
        default=_env_float("VGGT_FT_GRAM_DYN_QUANTILE", 0.30),
        help="Per-view Gram similarity quantile to define low-consistency region.",
    )
    parser.add_argument(
        "--gram_dyn_weight_floor",
        type=float,
        default=_env_float("VGGT_FT_GRAM_DYN_WEIGHT_FLOOR", 0.25),
        help="Lower bound of dynamic soft weight map.",
    )
    parser.add_argument(
        "--gram_dyn_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_GRAM_DYN_WARMUP_STEPS", 40)),
        help="Linear warmup steps for applying Gram dynamic soft weights.",
    )
    parser.add_argument(
        "--dyn_proxy_enable",
        type=str,
        default=_env_str("VGGT_FT_DYN_PROXY_ENABLE", "off"),
        choices=["on", "off"],
        help="Enable foreground-only static soft proxy for geometry supervision.",
    )
    parser.add_argument(
        "--dyn_proxy_mode",
        type=str,
        default=_env_str("VGGT_FT_DYN_PROXY_MODE", "fg_static_soft"),
        choices=["fg_static_soft"],
        help="Dynamic proxy mode.",
    )
    parser.add_argument(
        "--dyn_proxy_use_gram",
        type=str,
        default=_env_str("VGGT_FT_DYN_PROXY_USE_GRAM", "on"),
        choices=["on", "off"],
        help="Use Gram static consistency inside dyn_proxy.",
    )
    parser.add_argument(
        "--dyn_proxy_use_support",
        type=str,
        default=_env_str("VGGT_FT_DYN_PROXY_USE_SUPPORT", "on"),
        choices=["on", "off"],
        help="Use point multiview support inside dyn_proxy.",
    )
    parser.add_argument(
        "--dyn_proxy_floor",
        type=float,
        default=_env_float("VGGT_FT_DYN_PROXY_FLOOR", 0.35),
        help="Lower bound of dyn_proxy weight inside foreground.",
    )
    parser.add_argument(
        "--dyn_proxy_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_DYN_PROXY_WARMUP_STEPS", 40)),
        help="Linear warmup steps for applying dyn_proxy to geometry supervision.",
    )
    parser.add_argument(
        "--point_cons_tau",
        type=float,
        default=_env_float("VGGT_FT_POINT_CONS_TAU", 0.03),
        help="Scale for pseudo point consistency reliability: exp(-err/tau).",
    )
    parser.add_argument(
        "--point_cons_weight_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_CONS_WEIGHT_FLOOR", 0.2),
        help="Lower bound of reliability weight from pseudo point consistency.",
    )
    parser.add_argument(
        "--point_cons_clip_min_qv",
        type=float,
        default=_env_float("VGGT_FT_POINT_CONS_CLIP_MIN_QV", 1e-6),
        help="Skip point consistency quantile clipping when q-th error is below this value.",
    )
    parser.add_argument(
        "--point_cons_quantile",
        type=float,
        default=_env_float("VGGT_FT_POINT_CONS_QUANTILE", 1.0),
        help="Keep only low consistency-error pixels for point loss (<=quantile). 1.0 disables clipping.",
    )
    parser.add_argument(
        "--point_cons_focus",
        type=str,
        default=_env_str("VGGT_FT_POINT_CONS_FOCUS", "inlier"),
        choices=["inlier", "outlier", "all"],
        help="Quantile focus mode for consistency error: low-error, high-error, or disable clipping.",
    )
    parser.add_argument(
        "--point_residual_quantile",
        type=float,
        default=_env_float("VGGT_FT_POINT_RESIDUAL_QUANTILE", 1.0),
        help="Keep only low point-residual pixels for point loss (<=quantile). 1.0 disables clipping.",
    )
    parser.add_argument(
        "--point_residual_focus",
        type=str,
        default=_env_str("VGGT_FT_POINT_RESIDUAL_FOCUS", "inlier"),
        choices=["inlier", "outlier", "all"],
        help="Quantile focus mode for point residual: low-error, high-error, or disable clipping.",
    )
    parser.add_argument(
        "--point_residual_boost",
        type=float,
        default=_env_float("VGGT_FT_POINT_RESIDUAL_BOOST", 0.0),
        help="Extra weight on large point residuals (0 disables).",
    )
    parser.add_argument(
        "--point_residual_boost_cap",
        type=float,
        default=_env_float("VGGT_FT_POINT_RESIDUAL_BOOST_CAP", 4.0),
        help="Upper cap for residual boost multiplier.",
    )
    parser.add_argument(
        "--point_target_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_TARGET_MODE", "pointmap"),
        choices=["pointmap", "depth_unproject", "blend", "depth_consensus_unproject"],
        help="Point supervision target: pseudo pointmap, depth unprojection, or blend of both.",
    )
    parser.add_argument(
        "--point_target_blend_alpha",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_BLEND_ALPHA", 0.7),
        help="For point_target_mode=blend: base weight of depth-unproject target.",
    )
    parser.add_argument(
        "--point_target_blend_alpha_min",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_BLEND_ALPHA_MIN", 0.0),
        help="For blend mode, lower clamp of depth-unproject ratio.",
    )
    parser.add_argument(
        "--point_target_blend_alpha_max",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_BLEND_ALPHA_MAX", 1.0),
        help="For blend mode, upper clamp of depth-unproject ratio.",
    )
    parser.add_argument(
        "--point_target_blend_rel_gain",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_BLEND_REL_GAIN", 1.0),
        help="For blend mode, gain of reliability-driven alpha increase (0 disables).",
    )
    parser.add_argument(
        "--point_target_blend_mv_gain",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_BLEND_MV_GAIN", 1.0),
        help="For blend mode, gain of multiview-support-driven alpha increase (0 disables).",
    )
    parser.add_argument(
        "--point_target_blend_by_reliability",
        type=str,
        default=_env_str("VGGT_FT_POINT_TARGET_BLEND_BY_RELIABILITY", "on"),
        choices=["off", "on"],
        help="For blend mode, increase depth target ratio where pseudo pointmap reliability is low.",
    )
    parser.add_argument(
        "--point_target_blend_by_mv_support",
        type=str,
        default=_env_str("VGGT_FT_POINT_TARGET_BLEND_BY_MV_SUPPORT", "off"),
        choices=["off", "on"],
        help="For blend mode, increase depth target ratio where pseudo point has weak multiview support.",
    )
    parser.add_argument(
        "--point_target_blend_mv_region_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_TARGET_BLEND_MV_REGION_MODE", "all"),
        choices=["all", "bg_only"],
        help="Restrict the mv-support-driven part of point-target blending to all pixels or background only.",
    )
    parser.add_argument(
        "--point_target_blend_mv_policy",
        type=str,
        default=_env_str("VGGT_FT_POINT_TARGET_BLEND_MV_POLICY", "weak_to_depth"),
        choices=["weak_to_depth", "strong_to_depth"],
        help="How multiview support modulates depth target ratio in blend mode.",
    )
    parser.add_argument(
        "--point_target_consensus_alpha_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_TARGET_CONSENSUS_ALPHA_FLOOR", 0.0),
        help="For depth_consensus_unproject mode: minimum depth-unproject ratio.",
    )
    parser.add_argument(
        "--target_point_frame",
        type=str,
        default=_env_str("VGGT_FT_TARGET_POINT_FRAME", "auto"),
        choices=["auto", "world", "camera"],
        help="Point frame of pseudo pointmap target. auto uses NPZ metadata then falls back to reprojection check.",
    )
    parser.add_argument(
        "--pred_point_frame",
        type=str,
        default=_env_str("VGGT_FT_PRED_POINT_FRAME", "auto"),
        choices=["auto", "world", "camera"],
        help="Point frame of point_head output used for loss. auto resolves with reprojection error.",
    )
    parser.add_argument(
        "--point_loss_scale_depth_unproject",
        type=float,
        default=_env_float("VGGT_FT_POINT_LOSS_SCALE_DEPTH_UNPROJECT", 0.5),
        help="Scale point loss when pseudo pointmap source is depth_unproject.",
    )
    parser.add_argument(
        "--point_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_WARMUP_STEPS", 0)),
        help="Linear warmup steps for point loss scale.",
    )
    parser.add_argument(
        "--point_normal_consis_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_NORMAL_CONSIS_WARMUP_STEPS", 40)),
        help="Linear warmup steps for point normal consistency loss.",
    )
    parser.add_argument(
        "--point_reproj_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_REPROJ_WARMUP_STEPS", 40)),
        help="Linear warmup steps for point reprojection consistency loss.",
    )
    parser.add_argument(
        "--point_reproj_clamp_px",
        type=float,
        default=_env_float("VGGT_FT_POINT_REPROJ_CLAMP_PX", 64.0),
        help="Clamp reprojection residual (pixels) for stability.",
    )
    parser.add_argument(
        "--human_prior_enable",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_ENABLE", "off"),
        choices=["off", "on"],
        help="Enable optional human-prior sidecar loading and supervision.",
    )
    parser.add_argument(
        "--human_prior_subdir",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_SUBDIR", "human_prior"),
        help="Per-sequence sidecar subdir that stores human prior npz files aligned to geometry cache names.",
    )
    parser.add_argument(
        "--human_prior_strict",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_STRICT", "off"),
        choices=["off", "on"],
        help="When on, fail if any requested human-prior sidecar is missing or misaligned.",
    )
    parser.add_argument(
        "--human_prior_point_blend_alpha",
        type=float,
        default=_env_float("VGGT_FT_HUMAN_PRIOR_POINT_BLEND_ALPHA", 0.0),
        help="Blend ratio of human prior point target inside the selected prior region.",
    )
    parser.add_argument(
        "--human_prior_point_blend_region",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_POINT_BLEND_REGION", "body"),
        choices=["off", "all", "body", "head", "face", "head_face"],
        help="Region where human prior point target participates in point target blending.",
    )
    parser.add_argument(
        "--human_prior_weight_boost",
        type=float,
        default=_env_float("VGGT_FT_HUMAN_PRIOR_WEIGHT_BOOST", 1.0),
        help="Extra supervision boost multiplier inside selected human prior region.",
    )
    parser.add_argument(
        "--human_prior_weight_region",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_WEIGHT_REGION", "head_face"),
        choices=["off", "all", "body", "head", "face", "head_face"],
        help="Region where human prior multiplies supervision weights.",
    )
    parser.add_argument(
        "--human_prior_loss_region",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_LOSS_REGION", "body"),
        choices=["off", "all", "body", "head", "face", "head_face"],
        help="Region used by lambda_point_prior when comparing point_head to human prior pointmap.",
    )
    parser.add_argument(
        "--human_prior_complete_weight",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_HUMAN_PRIOR_COMPLETE_WEIGHT", 0.0)),
        help="Completion supervision floor injected into prior-valid holes where pseudo depth/point supervision is missing.",
    )
    parser.add_argument(
        "--human_prior_complete_region",
        type=str,
        default=_env_str("VGGT_FT_HUMAN_PRIOR_COMPLETE_REGION", "body"),
        choices=["off", "all", "body", "head", "face", "head_face"],
        help="Region where human prior can fill missing supervision coverage to improve point-cloud completeness.",
    )
    parser.add_argument(
        "--human_prior_region_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_HUMAN_PRIOR_REGION_ERODE_PX", 0)),
        help="Optional erosion applied to human prior masks before blending or weight boosting.",
    )
    parser.add_argument(
        "--human_prior_head_fallback_top_ratio",
        type=float,
        default=_env_float("VGGT_FT_HUMAN_PRIOR_HEAD_FALLBACK_TOP_RATIO", 0.32),
        help="If head mask is absent, use this top-of-body ratio to synthesize a coarse head region.",
    )
    parser.add_argument(
        "--human_prior_face_fallback_top_ratio",
        type=float,
        default=_env_float("VGGT_FT_HUMAN_PRIOR_FACE_FALLBACK_TOP_RATIO", 0.18),
        help="If face mask is absent, use this top-of-head/body ratio to synthesize a coarse face region.",
    )
    parser.add_argument(
        "--use_fg_mask",
        type=str,
        default=_env_str("VGGT_FT_USE_FG_MASK", "off"),
        choices=["off", "on"],
        help="Use foreground mask to gate valid supervision area.",
    )
    parser.add_argument(
        "--fg_mask_source",
        type=str,
        default=_env_str("VGGT_FT_FG_MASK_SOURCE", "auto"),
        choices=["auto", "mask", "mask_cihp"],
        help="Preferred source for foreground mask.",
    )
    parser.add_argument(
        "--fg_mask_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_MASK_ERODE_PX", 0)),
        help="Foreground mask erosion pixels to suppress boundary noise.",
    )
    parser.add_argument(
        "--point_loss_fg_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_LOSS_FG_ERODE_PX", 0)),
        help="Extra FG erosion for point losses only.",
    )
    parser.add_argument(
        "--fg_supervision_boost",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_FG_SUPERVISION_BOOST", 1.0)),
        help="Relative supervision boost inside foreground compared with background/other valid pixels.",
    )
    parser.add_argument(
        "--fg_supervision_bg_floor",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_FG_SUPERVISION_BG_FLOOR", 0.0)),
        help="Background validity floor when FG-only supervision is enabled; keeps a weak BG lane so fg boost is meaningful.",
    )
    parser.add_argument(
        "--fg_supervision_region_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_SUPERVISION_REGION_MODE", "all"),
        choices=["all", "interior_only"],
        help="Region mode for foreground supervision boost only; does not change valid-mask semantics.",
    )
    parser.add_argument(
        "--fg_supervision_region_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_SUPERVISION_REGION_ERODE_PX", 0)),
        help="Extra erosion applied only to the foreground boost mask when fg_supervision_region_mode=interior_only.",
    )
    parser.add_argument(
        "--lambda_fg_conf_presence",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_LAMBDA_FG_CONF_PRESENCE", 0.0)),
        help="Weak lower-bound regularizer that prevents foreground confidence from collapsing too low.",
    )
    parser.add_argument(
        "--fg_conf_presence_target_ratio",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_FG_CONF_PRESENCE_TARGET_RATIO", 0.9)),
        help="Foreground confidence floor target as a ratio of GT foreground confidence mean.",
    )
    parser.add_argument(
        "--lambda_fg_structure_depth_edge",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_LAMBDA_FG_STRUCTURE_DEPTH_EDGE", 0.0)),
        help="Geometry structure loss on target depth-edge gradients inside a tight foreground bbox.",
    )
    parser.add_argument(
        "--fg_structure_bbox_margin_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_STRUCTURE_BBOX_MARGIN_PX", 12)),
        help="Margin around foreground bbox for H-family structure loss.",
    )
    parser.add_argument(
        "--fg_structure_bbox_min_side_px",
        type=int,
        default=max(1, _env_int("VGGT_FT_FG_STRUCTURE_BBOX_MIN_SIDE_PX", 24)),
        help="Minimum bbox side for H-family structure loss.",
    )
    parser.add_argument(
        "--fg_structure_region_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_STRUCTURE_REGION_MODE", "bbox"),
        choices=["bbox", "bbox_fg_interior"],
        help="Region mode for H-family structure loss.",
    )
    parser.add_argument(
        "--fg_structure_region_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_STRUCTURE_REGION_ERODE_PX", 0)),
        help="Extra erosion applied to foreground before intersecting the H-family structure region.",
    )
    parser.add_argument(
        "--fg_structure_depth_edge_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_STRUCTURE_DEPTH_EDGE_WARMUP_STEPS", 0)),
        help="Linear warmup steps for the H-family depth-edge structure loss scale.",
    )
    parser.add_argument(
        "--fg_structure_boundary_probe_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_STRUCTURE_BOUNDARY_PROBE_PX", 2)),
        help="Fixed diagnostic boundary probe width for H-family structure reporting.",
    )
    parser.add_argument(
        "--fg_structure_edge_support_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_STRUCTURE_EDGE_SUPPORT_MODE", "off"),
        choices=["off", "target_edge_quantile"],
        help="Optional target-edge support mask for H-family depth-edge loss.",
    )
    parser.add_argument(
        "--fg_structure_edge_support_quantile",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_FG_STRUCTURE_EDGE_SUPPORT_QUANTILE", 0.0)),
        help="Target-edge quantile for H-family support mask when fg_structure_edge_support_mode=target_edge_quantile.",
    )
    parser.add_argument(
        "--fg_structure_edge_support_min_px",
        type=int,
        default=max(1, _env_int("VGGT_FT_FG_STRUCTURE_EDGE_SUPPORT_MIN_PX", 32)),
        help="Minimum support pixels per view for H-family target-edge support mask.",
    )
    parser.add_argument(
        "--fg_structure_edge_weight_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_STRUCTURE_EDGE_WEIGHT_MODE", "uniform"),
        choices=["uniform", "target_edge_sqrt"],
        help="Optional continuous weighting mode for H-family structure supervision.",
    )
    parser.add_argument(
        "--fg_structure_boundary_falloff_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_FG_STRUCTURE_BOUNDARY_FALLOFF_PX", 0)),
        help="Inside-structure boundary falloff width for H-family soft weighting.",
    )
    parser.add_argument(
        "--fg_structure_component_bias_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_STRUCTURE_COMPONENT_BIAS_MODE", "off"),
        choices=["off", "largest_soft"],
        help="Optional soft single-component bias for H-family main support.",
    )
    parser.add_argument(
        "--fg_structure_component_bias_threshold_ratio",
        type=float,
        default=_env_float("VGGT_FT_FG_STRUCTURE_COMPONENT_BIAS_THRESHOLD_RATIO", 0.25),
        help="Relative threshold used to detect fragmented support components.",
    )
    parser.add_argument(
        "--fg_structure_component_bias_other_scale",
        type=float,
        default=_env_float("VGGT_FT_FG_STRUCTURE_COMPONENT_BIAS_OTHER_SCALE", 1.0),
        help="Soft downweight for non-largest support components when component bias is enabled.",
    )
    parser.add_argument(
        "--fg_structure_front_depth_bias_mode",
        type=str,
        default=_env_str("VGGT_FT_FG_STRUCTURE_FRONT_DEPTH_BIAS_MODE", "off"),
        choices=["off", "front_soft"],
        help="Optional front-surface soft bias inside H-family structure supervision.",
    )
    parser.add_argument(
        "--fg_structure_front_depth_bias_tau",
        type=float,
        default=max(1e-3, _env_float("VGGT_FT_FG_STRUCTURE_FRONT_DEPTH_BIAS_TAU", 0.75)),
        help="Softness for front-depth bias decay on deeper-than-center structure pixels.",
    )
    parser.add_argument(
        "--fg_structure_front_depth_bias_center_quantile",
        type=float,
        default=_env_float("VGGT_FT_FG_STRUCTURE_FRONT_DEPTH_BIAS_CENTER_QUANTILE", 0.55),
        help="Target-depth quantile used as the front-surface center for H-family depth bias.",
    )
    parser.add_argument(
        "--lambda_point_mv_outside_ring",
        type=float,
        default=max(0.0, _env_float("VGGT_FT_LAMBDA_POINT_MV_OUTSIDE_RING", 0.0)),
        help="Weak outside-ring halo penalty using the existing multiview reprojection path.",
    )
    parser.add_argument(
        "--point_mv_outside_ring_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_OUTSIDE_RING_PX", 3)),
        help="Outside-ring width in pixels for the H-family halo penalty.",
    )
    parser.add_argument(
        "--supervision_weight_mode",
        type=str,
        default=_env_str("VGGT_FT_SUPERVISION_WEIGHT_MODE", "conf"),
        choices=["conf", "uniform", "mix"],
        help="Weight mode for depth/point supervision.",
    )
    parser.add_argument(
        "--supervision_weight_mix_alpha",
        type=float,
        default=_env_float("VGGT_FT_SUPERVISION_WEIGHT_MIX_ALPHA", 0.5),
        help="For supervision_weight_mode=mix: alpha*conf + (1-alpha)*uniform.",
    )
    parser.add_argument(
        "--point_mv_consistency",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_CONSISTENCY", "off"),
        choices=["off", "on"],
        help="Cross-view depth agreement weighting for pseudo point supervision.",
    )
    parser.add_argument(
        "--point_mv_tol_abs",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_TOL_ABS", 0.03),
        help="Absolute depth tolerance for point multiview agreement.",
    )
    parser.add_argument(
        "--point_mv_tol_rel",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_TOL_REL", 0.05),
        help="Relative depth tolerance for point multiview agreement.",
    )
    parser.add_argument(
        "--point_mv_weight_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_WEIGHT_FLOOR", 0.2),
        help="Minimum weight after multiview support weighting.",
    )
    parser.add_argument(
        "--point_mv_stride",
        type=int,
        default=max(1, _env_int("VGGT_FT_POINT_MV_STRIDE", 2)),
        help="Downsample stride for multiview support computation.",
    )
    parser.add_argument(
        "--lambda_point_mv_depth",
        type=float,
        default=_env_float("VGGT_FT_LAMBDA_POINT_MV_DEPTH", 0.0),
        help="Cross-view depth reprojection loss on predicted pointmap.",
    )
    parser.add_argument(
        "--lambda_point_mv_mask",
        type=float,
        default=_env_float("VGGT_FT_LAMBDA_POINT_MV_MASK", 0.0),
        help="Cross-view foreground-mask reprojection loss on predicted pointmap.",
    )
    parser.add_argument(
        "--point_mv_depth_max_pairs",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_DEPTH_MAX_PAIRS", 3)),
        help="Max target views per source view for mv-depth loss. 0 means all.",
    )
    parser.add_argument(
        "--point_mv_depth_pair_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_PAIR_MODE", "adjacent"),
        choices=["sequential", "adjacent", "farthest", "random"],
        help="Pair selection strategy for mv-depth supervision.",
    )
    parser.add_argument(
        "--point_mv_depth_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_DEPTH_WARMUP_STEPS", 40)),
        help="Linear warmup steps for point mv-depth loss scale.",
    )
    parser.add_argument(
        "--point_mv_depth_region_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_REGION_MODE", "all"),
        choices=["all", "bg_only"],
        help="Apply mv-depth reprojection on all valid pixels or background-only valid pixels.",
    )
    parser.add_argument(
        "--point_mv_mask_warmup_steps",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_MASK_WARMUP_STEPS", 40)),
        help="Linear warmup steps for point mv-mask loss scale.",
    )
    parser.add_argument(
        "--point_mv_depth_inlier_only",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_INLIER_ONLY", "off"),
        choices=["off", "on"],
        help="If on, mv-depth loss uses inlier pixels only.",
    )
    parser.add_argument(
        "--point_mv_depth_err_quantile",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_ERR_QUANTILE", 1.0),
        help="Keep only low-residual mv-depth pixels (<=quantile). 1.0 disables clipping.",
    )
    parser.add_argument(
        "--point_mv_depth_outlier_boost",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_OUTLIER_BOOST", 0.0),
        help="Additional weight boost for mv-depth outliers (0 disables).",
    )
    parser.add_argument(
        "--point_mv_depth_outlier_cap",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_OUTLIER_CAP", 3.0),
        help="Upper cap for boosted outlier weights in mv-depth loss.",
    )
    parser.add_argument(
        "--point_mv_depth_tgt_valid_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_TGT_VALID_MODE", "hard"),
        choices=["hard", "soft", "soft_zero", "off"],
        help="How target-view validity is used in mv-depth loss: hard gate, soft weight, soft weight with strict zero on invalid target samples, or off.",
    )
    parser.add_argument(
        "--point_mv_depth_tgt_valid_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_TGT_VALID_FLOOR", 0.2),
        help="When tgt_valid_mode=soft, minimum sampled target-valid weight.",
    )
    parser.add_argument(
        "--point_mv_depth_min_tgt_valid_ratio",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_MIN_TGT_VALID_RATIO", 0.0),
        help="Skip mv-depth pair when sampled target-valid ratio is below this threshold.",
    )
    parser.add_argument(
        "--point_mv_mask_min_tgt_fg_ratio",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_MASK_MIN_TGT_FG_RATIO", 0.0),
        help="Skip mv-mask pair when sampled target-foreground ratio is below this threshold.",
    )
    parser.add_argument(
        "--point_mv_mask_hit_thr",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_MASK_HIT_THR", 0.5),
        help="Foreground hit threshold for mv-mask miss penalty.",
    )
    parser.add_argument(
        "--point_mv_mask_soft_blur_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_MASK_SOFT_BLUR_PX", 0)),
        help="Blur radius (px) for soft foreground field used by mv-mask supervision.",
    )
    parser.add_argument(
        "--point_mv_mask_soft_blur_iters",
        type=int,
        default=max(1, _env_int("VGGT_FT_POINT_MV_MASK_SOFT_BLUR_ITERS", 1)),
        help="Number of blur iterations for soft foreground field.",
    )
    parser.add_argument(
        "--point_mv_mask_soft_mix",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_MASK_SOFT_MIX", 0.0),
        help="Mix ratio between hard-miss and soft-miss terms in mv-mask loss.",
    )
    parser.add_argument(
        "--point_mv_mask_soft_hit_thr",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_MASK_SOFT_HIT_THR", 0.35),
        help="Foreground hit threshold for the soft-miss branch in mv-mask loss.",
    )
    parser.add_argument(
        "--point_mv_depth_tgt_valid_scale_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_TGT_VALID_SCALE_MODE", "off"),
        choices=["off", "linear"],
        help="Extra scale for mv-depth loss from sampled target-valid ratio.",
    )
    parser.add_argument(
        "--point_mv_depth_tgt_valid_scale_thr",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_TGT_VALID_SCALE_THR", 0.01),
        help="When scale_mode=linear, full mv-depth weight is reached at this target-valid ratio.",
    )
    parser.add_argument(
        "--point_mv_depth_adapt_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_ADAPT_MODE", "off"),
        choices=["off", "valid_ratio", "on"],
        help="Adaptive scale mode for mv-depth loss. valid_ratio boosts when valid ratio is too low.",
    )
    parser.add_argument(
        "--point_mv_depth_adapt_target_valid",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_ADAPT_TARGET_VALID", 0.01),
        help="Target valid ratio for mv-depth adaptive scaling when adapt_mode=valid_ratio.",
    )
    parser.add_argument(
        "--point_mv_depth_adapt_min_scale",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_ADAPT_MIN_SCALE", 1.0),
        help="Lower bound of adaptive scaling for mv-depth loss.",
    )
    parser.add_argument(
        "--point_mv_depth_adapt_max_scale",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_ADAPT_MAX_SCALE", 32.0),
        help="Upper bound of adaptive scaling for mv-depth loss.",
    )
    parser.add_argument(
        "--point_support_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_SUPPORT_MODE", "direct"),
        choices=["direct", "inverse", "off"],
        help="How multiview support weights are applied to point loss.",
    )
    parser.add_argument(
        "--point_support_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_SUPPORT_FLOOR", 0.0),
        help="Floor after support mapping for point loss weight.",
    )
    parser.add_argument(
        "--point_mv_depth_support_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_DEPTH_SUPPORT_MODE", "direct"),
        choices=["direct", "inverse", "off"],
        help="How multiview support weights are applied to mv-depth loss.",
    )
    parser.add_argument(
        "--point_mv_depth_support_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_DEPTH_SUPPORT_FLOOR", 0.0),
        help="Floor after support mapping for mv-depth support weight.",
    )
    parser.add_argument(
        "--point_mv_mask_support_mode",
        type=str,
        default=_env_str("VGGT_FT_POINT_MV_MASK_SUPPORT_MODE", "inverse"),
        choices=["direct", "inverse", "off"],
        help="How multiview support weights are applied to mv-mask loss.",
    )
    parser.add_argument(
        "--point_mv_mask_support_floor",
        type=float,
        default=_env_float("VGGT_FT_POINT_MV_MASK_SUPPORT_FLOOR", 0.0),
        help="Floor after support mapping for mv-mask support weight.",
    )
    parser.add_argument(
        "--point_mv_depth_fg_erode_px",
        type=int,
        default=max(0, _env_int("VGGT_FT_POINT_MV_DEPTH_FG_ERODE_PX", 0)),
        help="Extra erosion on valid/fg mask before mv-depth loss.",
    )
    parser.add_argument("--lr_backbone_scale", type=float, default=_env_float("VGGT_FT_LR_BACKBONE_SCALE", 0.2))
    parser.add_argument("--lr_head_scale", type=float, default=_env_float("VGGT_FT_LR_HEAD_SCALE", 1.0))
    parser.add_argument("--lr_camera_scale", type=float, default=_env_float("VGGT_FT_LR_CAMERA_SCALE", 0.1))
    parser.add_argument("--grad_clip", type=float, default=_env_float("VGGT_FT_GRAD_CLIP", 1.0))
    parser.add_argument("--geom_subdir", type=str, default=_env_str("VGGT_FT_GEOM_SUBDIR", _env_str("VGGT_GEOM_SUBDIR", "vggt_geom")))
    parser.add_argument("--log_dir", type=str, default=_env_str("VGGT_FT_LOG_DIR", "logs"))
    parser.add_argument("--ckpt_dir", type=str, default=_env_str("VGGT_FT_CKPT_DIR", "ckpt"))
    parser.add_argument("--seed", type=int, default=_env_int("VGGT_FT_SEED", 0))
    parser.add_argument("--device", type=str, default=_env_str("VGGT_DEVICE", "auto"))
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("VGGT_TF32", True),
        help="Enable TF32 matmul/cudnn on CUDA.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("VGGT_AMP", True),
        help="Reserved precision flag for parity with the runner; training path remains fp32 today.",
    )
    parser.add_argument(
        "--freeze_mode",
        type=str,
        default=_env_str("VGGT_FT_FREEZE_MODE", "all_trainable"),
        choices=["depth_point", "depth_only", "point_only", "all_trainable"],
    )
    parser.add_argument(
        "--depth_scale_align",
        type=str,
        default=_env_str("VGGT_FT_DEPTH_SCALE_ALIGN", "off"),
        choices=["off", "median"],
    )
    parser.add_argument("--eval_every_steps", type=int, default=max(1, _env_int("VGGT_FT_EVAL_EVERY_STEPS", 20)))
    parser.add_argument("--debug_metrics_every_steps", type=int, default=max(0, _env_int("VGGT_FT_DEBUG_METRICS_EVERY_STEPS", 0)))
    parser.add_argument("--log_heartbeat_sec", type=float, default=max(0.0, _env_float("VGGT_FT_LOG_HEARTBEAT_SEC", 30.0)))
    parser.add_argument("--debug_vis_every_steps", type=int, default=max(0, _env_int("VGGT_FT_DEBUG_VIS_EVERY_STEPS", 0)))
    parser.add_argument("--debug_vis_max_steps", type=int, default=max(0, _env_int("VGGT_FT_DEBUG_VIS_MAX_STEPS", 0)))
    parser.add_argument("--debug_vis_views", type=int, default=max(1, _env_int("VGGT_FT_DEBUG_VIS_VIEWS", 1)))
    parser.add_argument("--debug_vis_dir", type=str, default=_env_str("VGGT_FT_DEBUG_VIS_DIR", ""))
    parser.add_argument("--early_stop_patience", type=int, default=max(0, _env_int("VGGT_FT_EARLY_STOP_PATIENCE", 0)))
    parser.add_argument("--min_improve", type=float, default=max(0.0, _env_float("VGGT_FT_MIN_IMPROVE", 0.0)))
    parser.add_argument("--max_steps_per_epoch", type=int, default=max(0, _env_int("VGGT_FT_MAX_STEPS_PER_EPOCH", 0)))
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[info] ignored unknown args: {unknown}")
    return args


def _resolve_device(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"", "auto", "none"}:
        return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    if s.startswith("cuda") and torch.cuda.device_count() <= 0:
        return "cpu"
    return s


def _load_state_dict(ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


def _atomic_torch_save(state_dict: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    torch.save(state_dict, tmp_path)
    # Validate readability before replacing target to avoid broken checkpoints.
    _ = torch.load(tmp_path, map_location="cpu")
    os.replace(tmp_path, out_path)


def _masked_percentile_range(
    x: np.ndarray,
    mask01: Optional[np.ndarray] = None,
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> tuple[float, float]:
    xx = np.asarray(x, dtype=np.float32)
    if mask01 is not None:
        mm = np.asarray(mask01, dtype=np.float32) > 0.5
        vals = xx[mm]
    else:
        vals = xx.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        vals = xx[np.isfinite(xx)]
    if vals.size <= 0:
        return 0.0, 1.0
    ql = float(np.percentile(vals, q_low))
    qh = float(np.percentile(vals, q_high))
    if not np.isfinite(ql):
        ql = float(np.min(vals))
    if not np.isfinite(qh):
        qh = float(np.max(vals))
    if qh <= ql:
        qh = ql + 1e-6
    return ql, qh


def _vis_gray_u8(x: np.ndarray, mask01: Optional[np.ndarray] = None) -> np.ndarray:
    lo, hi = _masked_percentile_range(x, mask01=mask01, q_low=2.0, q_high=98.0)
    y = (np.asarray(x, dtype=np.float32) - lo) / max(1e-6, (hi - lo))
    y = np.clip(y, 0.0, 1.0)
    y_u8 = (y * 255.0).astype(np.uint8)
    return np.repeat(y_u8[..., None], 3, axis=2)


def _vis_unit_u8(x01: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x01, dtype=np.float32), 0.0, 1.0)
    y_u8 = (y * 255.0).astype(np.uint8)
    return np.repeat(y_u8[..., None], 3, axis=2)


def _vis_vec3_u8(x: np.ndarray, mask01: Optional[np.ndarray] = None) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float32)
    if xx.ndim != 3 or xx.shape[-1] != 3:
        raise RuntimeError(f"expected vec3 map (H,W,3), got {tuple(xx.shape)}")
    out = np.zeros_like(xx, dtype=np.float32)
    for ci in range(3):
        lo, hi = _masked_percentile_range(xx[..., ci], mask01=mask01, q_low=2.0, q_high=98.0)
        out[..., ci] = np.clip((xx[..., ci] - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _save_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8)).save(path)


def _hstack_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or b.shape != c.shape:
        raise RuntimeError(f"triplet shape mismatch: {a.shape}, {b.shape}, {c.shape}")
    return np.concatenate([a, b, c], axis=1)


def _write_step_debug_vis(
    out_dir: Path,
    epoch: int,
    step: int,
    view_idx: int,
    point_target_mode: str,
    rgb: torch.Tensor,              # (3,H,W) [0,1]
    depth_tgt: torch.Tensor,        # (H,W)
    depth_pred: torch.Tensor,       # (H,W)
    point_tgt_raw: torch.Tensor,    # (H,W,3)
    point_tgt_loss: torch.Tensor,   # (H,W,3)
    point_pred_raw: torch.Tensor,   # (H,W,3) raw point_head output
    point_pred_loss: torch.Tensor,  # (H,W,3) point used in losses (frame-aligned)
    conf_tgt: torch.Tensor,         # (H,W)
    conf_pred: torch.Tensor,        # (H,W)
    valid01: torch.Tensor,          # (H,W)
    weight01: torch.Tensor,         # (H,W)
    fg01: Optional[torch.Tensor],
    extra: Dict[str, float],
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    stem = f"e{int(epoch):03d}_step{int(step):06d}_v{int(view_idx):02d}"

    rgb_np = (
        rgb.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    )
    rgb_u8 = (rgb_np * 255.0).astype(np.uint8)

    depth_t = depth_tgt.detach().float().cpu().numpy()
    depth_p = depth_pred.detach().float().cpu().numpy()
    depth_err = np.abs(depth_p - depth_t)

    point_t_raw = point_tgt_raw.detach().float().cpu().numpy()
    point_t_loss = point_tgt_loss.detach().float().cpu().numpy()
    point_p_raw = point_pred_raw.detach().float().cpu().numpy()
    point_p = point_pred_loss.detach().float().cpu().numpy()
    point_err = np.linalg.norm(point_p - point_t_loss, axis=-1)
    point_pred_shift = np.linalg.norm(point_p - point_p_raw, axis=-1)
    point_t_shift = np.linalg.norm(point_t_loss - point_t_raw, axis=-1)

    conf_t = conf_tgt.detach().float().cpu().numpy()
    conf_p = conf_pred.detach().float().cpu().numpy()
    valid = valid01.detach().float().cpu().numpy()
    weight = weight01.detach().float().cpu().numpy()
    fg = fg01.detach().float().cpu().numpy() if fg01 is not None else None

    depth_t_u8 = _vis_gray_u8(depth_t, mask01=valid)
    depth_p_u8 = _vis_gray_u8(depth_p, mask01=valid)
    depth_e_u8 = _vis_gray_u8(depth_err, mask01=valid)

    point_t_raw_u8 = _vis_vec3_u8(point_t_raw, mask01=valid)
    point_t_loss_u8 = _vis_vec3_u8(point_t_loss, mask01=valid)
    point_p_raw_u8 = _vis_vec3_u8(point_p_raw, mask01=valid)
    point_p_u8 = _vis_vec3_u8(point_p, mask01=valid)
    point_e_u8 = _vis_gray_u8(point_err, mask01=valid)
    point_shift_u8 = _vis_gray_u8(point_t_shift, mask01=valid)
    point_pred_shift_u8 = _vis_gray_u8(point_pred_shift, mask01=valid)

    conf_t_u8 = _vis_unit_u8(conf_t)
    conf_p_u8 = _vis_unit_u8(conf_p)
    valid_u8 = _vis_unit_u8(valid)
    weight_u8 = _vis_unit_u8(weight)
    fg_u8 = _vis_unit_u8(fg) if fg is not None else _vis_unit_u8(np.zeros_like(valid))

    depth_triplet = _hstack_triplet(depth_t_u8, depth_p_u8, depth_e_u8)
    point_triplet = _hstack_triplet(point_t_loss_u8, point_p_u8, point_e_u8)
    point_target_triplet = _hstack_triplet(point_t_raw_u8, point_t_loss_u8, point_shift_u8)
    point_pred_triplet = _hstack_triplet(point_p_raw_u8, point_p_u8, point_pred_shift_u8)
    conf_triplet = _hstack_triplet(conf_t_u8, conf_p_u8, weight_u8)
    mask_triplet = _hstack_triplet(valid_u8, fg_u8, _vis_unit_u8((weight > 0.0).astype(np.float32)))

    p_rgb = out_dir / f"{stem}_rgb.png"
    p_depth = out_dir / f"{stem}_triplet_depth_tgt_pred_err.png"
    p_point = out_dir / f"{stem}_triplet_point_tgt_pred_err.png"
    p_point_tgt = out_dir / f"{stem}_triplet_point_raw_vs_loss_shift.png"
    p_point_pred = out_dir / f"{stem}_triplet_point_pred_raw_vs_loss_shift.png"
    p_conf = out_dir / f"{stem}_triplet_conf_tgt_pred_weight.png"
    p_mask = out_dir / f"{stem}_triplet_valid_fg_weightnz.png"
    p_sidecar = out_dir / f"{stem}_meta.json"

    _save_rgb(p_rgb, rgb_u8)
    _save_rgb(p_depth, depth_triplet)
    _save_rgb(p_point, point_triplet)
    _save_rgb(p_point_tgt, point_target_triplet)
    _save_rgb(p_point_pred, point_pred_triplet)
    _save_rgb(p_conf, conf_triplet)
    _save_rgb(p_mask, mask_triplet)

    meta: Dict[str, Any] = {
        "event": "ft_debug_vis",
        "epoch": int(epoch),
        "step": int(step),
        "view_idx": int(view_idx),
        "point_target_mode": str(point_target_mode),
        "semantics": {
            "triplet_depth_tgt_pred_err": ["left=depth_tgt", "mid=depth_pred_for_loss", "right=abs_err"],
            "triplet_point_tgt_pred_err": ["left=point_tgt_for_loss", "mid=point_pred", "right=l2_err"],
            "triplet_point_raw_vs_loss_shift": ["left=point_tgt_raw_from_npz", "mid=point_tgt_for_loss_world", "right=l2_shift"],
            "triplet_point_pred_raw_vs_loss_shift": ["left=point_pred_raw", "mid=point_pred_for_loss", "right=l2_shift"],
            "triplet_conf_tgt_pred_weight": ["left=conf_tgt01", "mid=conf_pred01", "right=supervision_weight"],
            "triplet_valid_fg_weightnz": ["left=valid", "mid=fg_mask_or_zero", "right=weight>0"],
        },
        "paths": {
            "rgb": str(p_rgb),
            "depth_triplet": str(p_depth),
            "point_triplet": str(p_point),
            "point_target_triplet": str(p_point_tgt),
            "point_pred_triplet": str(p_point_pred),
            "conf_triplet": str(p_conf),
            "mask_triplet": str(p_mask),
        },
        "stats": {
            "depth_abs_mean": float(np.mean(depth_err[np.isfinite(depth_err)])) if np.isfinite(depth_err).any() else 0.0,
            "point_abs_mean": float(np.mean(point_err[np.isfinite(point_err)])) if np.isfinite(point_err).any() else 0.0,
            "point_target_shift_mean": float(np.mean(point_t_shift[np.isfinite(point_t_shift)])) if np.isfinite(point_t_shift).any() else 0.0,
            "point_pred_shift_mean": float(np.mean(point_pred_shift[np.isfinite(point_pred_shift)])) if np.isfinite(point_pred_shift).any() else 0.0,
            "valid_cover": float(np.mean(valid)),
            "fg_cover": float(np.mean(fg)) if fg is not None else -1.0,
            "weight_mean": float(np.mean(weight)),
        },
    }
    for k, v in (extra or {}).items():
        if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
            meta["stats"][str(k)] = float(v)

    with p_sidecar.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _strip_prefix_if_present(sd: dict, prefix: str) -> dict:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(str(k).startswith(prefix) for k in keys):
        return {str(k)[len(prefix):]: v for k, v in sd.items()}
    return sd


def _load_model_compat(model: torch.nn.Module, ckpt_path: str) -> None:
    sd = _load_state_dict(ckpt_path)
    if not isinstance(sd, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(sd)}")

    sd = _strip_prefix_if_present(sd, "module.")
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(str(k) for k in sd.keys())
    matched = len(model_keys & ckpt_keys)
    if matched <= 0:
        raise RuntimeError(
            f"no matching keys between checkpoint and model, ckpt={ckpt_path}"
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        "[finetune] load_state_dict strict=False",
        f"matched={matched}",
        f"missing={len(missing)}",
        f"unexpected={len(unexpected)}",
    )
    if unexpected:
        print("[finetune] unexpected(sample)=", unexpected[:8])
    if missing:
        print("[finetune] missing(sample)=", missing[:8])


def main():
    args = parse_args()
    if str(args.point_mv_depth_adapt_mode).lower() == "on":
        args.point_mv_depth_adapt_mode = "valid_ratio"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    print(f"[finetune] device={device}")
    strict_deterministic = _env_bool("VGGT_STRICT_DETERMINISTIC", False)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if strict_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    print(
        "[finetune] precision"
        f" tf32={bool(args.tf32)}"
        f" amp_flag={bool(args.amp)}"
        f" strict_deterministic={strict_deterministic}"
    )

    seq_names = _normalize_seq_names(args.seq_names)
    if not seq_names:
        raise RuntimeError("seq_names is empty")
    cam_names = _split_tokens(args.cam_names)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_path = Path(args.log_dir) / "finetune_vggt_metrics.jsonl"
    summary_path = Path(args.log_dir) / "finetune_vggt_summary.json"

    ds = PseudoGeomDataset(
        zju_root=args.zju_root,
        seq_names=seq_names,
        cam_names=cam_names,
        max_frames=int(args.max_frames),
        geom_subdir=str(args.geom_subdir),
        human_prior_enable=(str(args.human_prior_enable).lower() == "on"),
        human_prior_subdir=str(args.human_prior_subdir),
        human_prior_strict=(str(args.human_prior_strict).lower() == "on"),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda b: b[0])
    print(f"[finetune] samples={len(ds)} seq={seq_names}")

    model = VGGT(enable_track=False).to(device)
    init_ckpt = (str(args.resume_ckpt).strip() or str(args.pretrained_ckpt).strip())
    _load_model_compat(model, init_ckpt)
    print(f"[finetune] init_ckpt={init_ckpt}")

    freeze_info = _apply_freeze_mode(model, freeze_mode=args.freeze_mode)
    depth_trainable = bool(freeze_info["depth_trainable"])
    point_trainable = bool(freeze_info["point_trainable"])
    print(
        "[finetune] freeze_mode="
        f"{args.freeze_mode} depth_trainable={depth_trainable} point_trainable={point_trainable}"
    )

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_trainable:
        raise RuntimeError("no trainable params found")
    groups: Dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "depth": [],
        "point": [],
        "camera": [],
    }
    for n, p in named_trainable:
        if n.startswith("depth_head."):
            groups["depth"].append(p)
        elif n.startswith("point_head."):
            groups["point"].append(p)
        elif n.startswith("camera_head."):
            groups["camera"].append(p)
        else:
            groups["backbone"].append(p)

    base_lr = float(args.lr)
    lr_backbone = base_lr * float(args.lr_backbone_scale)
    lr_head = base_lr * float(args.lr_head_scale)
    lr_camera = base_lr * float(args.lr_camera_scale)
    param_groups: list[Dict[str, Any]] = []
    if groups["backbone"]:
        param_groups.append({"params": groups["backbone"], "lr": lr_backbone})
    if groups["depth"]:
        param_groups.append({"params": groups["depth"], "lr": lr_head})
    if groups["point"]:
        param_groups.append({"params": groups["point"], "lr": lr_head})
    if groups["camera"]:
        param_groups.append({"params": groups["camera"], "lr": lr_camera})
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=float(args.weight_decay))
    print(
        "[finetune] lr_groups="
        f" backbone={lr_backbone:.3e}({len(groups['backbone'])})"
        f" depth={lr_head:.3e}({len(groups['depth'])})"
        f" point={lr_head:.3e}({len(groups['point'])})"
        f" camera={lr_camera:.3e}({len(groups['camera'])})"
    )

    lambda_depth = float(args.lambda_depth) if depth_trainable else 0.0
    lambda_point = float(args.lambda_point) if point_trainable else 0.0
    lambda_point_prior = float(args.lambda_point_prior) if point_trainable else 0.0
    lambda_point_reproj = float(args.lambda_point_reproj) if point_trainable else 0.0
    lambda_point_normal_consis = float(args.lambda_point_normal_consis) if point_trainable else 0.0
    lambda_point_mv_depth = float(args.lambda_point_mv_depth) if point_trainable else 0.0
    lambda_point_mv_mask = float(args.lambda_point_mv_mask) if point_trainable else 0.0
    lambda_fg_structure_depth_edge = float(args.lambda_fg_structure_depth_edge) if depth_trainable else 0.0
    lambda_point_mv_outside_ring = float(args.lambda_point_mv_outside_ring) if point_trainable else 0.0
    h_family_enabled = (lambda_fg_structure_depth_edge > 0.0) or (lambda_point_mv_outside_ring > 0.0)
    lambda_conf = float(args.lambda_conf) if depth_trainable else 0.0
    lambda_geom_cons = float(args.lambda_geom_cons) if (depth_trainable and point_trainable) else 0.0
    camera_trainable = bool(
        model.camera_head is not None
        and any(p.requires_grad for p in model.camera_head.parameters())
    )
    lambda_cam = float(args.lambda_cam) if camera_trainable else 0.0
    print(
        "[finetune] effective_lambdas="
        f" depth={lambda_depth} point={lambda_point} point_prior={lambda_point_prior} point_reproj={lambda_point_reproj}"
        f" point_normal_consis={lambda_point_normal_consis}"
        f" point_mv_depth={lambda_point_mv_depth} point_mv_mask={lambda_point_mv_mask}"
        f" fg_structure_edge={lambda_fg_structure_depth_edge} point_mv_outside_ring={lambda_point_mv_outside_ring}"
        f" conf={lambda_conf}"
        f" geom_cons={lambda_geom_cons} cam={lambda_cam}"
    )

    run_meta: Dict[str, Any] = {
        "event": "run_meta",
        "seq_names": seq_names,
        "cam_names": cam_names,
        "geom_subdir": str(args.geom_subdir),
        "epochs": int(args.epochs),
        "max_frames": int(args.max_frames),
        "max_steps_per_epoch": int(args.max_steps_per_epoch),
        "tf32": bool(args.tf32),
        "amp": bool(args.amp),
        "strict_deterministic": bool(strict_deterministic),
        "lr": float(args.lr),
        "lr_backbone_scale": float(args.lr_backbone_scale),
        "lr_head_scale": float(args.lr_head_scale),
        "lr_camera_scale": float(args.lr_camera_scale),
        "weight_decay": float(args.weight_decay),
        "freeze_mode": str(args.freeze_mode),
        "depth_scale_align": str(args.depth_scale_align),
        "lambda_depth": float(lambda_depth),
        "lambda_point": float(lambda_point),
        "lambda_point_prior": float(lambda_point_prior),
        "lambda_point_reproj": float(lambda_point_reproj),
        "lambda_point_normal_consis": float(lambda_point_normal_consis),
        "lambda_point_mv_depth": float(lambda_point_mv_depth),
        "lambda_point_mv_mask": float(lambda_point_mv_mask),
        "lambda_conf": float(lambda_conf),
        "lambda_conf_warmup_steps": int(args.lambda_conf_warmup_steps),
        "lambda_geom_cons": float(lambda_geom_cons),
        "lambda_cam": float(lambda_cam),
        "lambda_cam_warmup_steps": int(args.lambda_cam_warmup_steps),
        "cam_rot_weight": float(args.cam_rot_weight),
        "cam_fov_weight": float(args.cam_fov_weight),
        "cam_warmup_steps": int(args.cam_warmup_steps),
        "point_loss_scale_depth_unproject": float(args.point_loss_scale_depth_unproject),
        "point_warmup_steps": int(args.point_warmup_steps),
        "point_normal_consis_warmup_steps": int(args.point_normal_consis_warmup_steps),
        "point_reproj_warmup_steps": int(args.point_reproj_warmup_steps),
        "point_reproj_clamp_px": float(args.point_reproj_clamp_px),
        "human_prior_enable": str(args.human_prior_enable),
        "human_prior_subdir": str(args.human_prior_subdir),
        "human_prior_strict": str(args.human_prior_strict),
        "human_prior_point_blend_alpha": float(args.human_prior_point_blend_alpha),
        "human_prior_point_blend_region": str(args.human_prior_point_blend_region),
        "human_prior_weight_boost": float(args.human_prior_weight_boost),
        "human_prior_weight_region": str(args.human_prior_weight_region),
        "human_prior_loss_region": str(args.human_prior_loss_region),
        "human_prior_complete_weight": float(args.human_prior_complete_weight),
        "human_prior_complete_region": str(args.human_prior_complete_region),
        "human_prior_region_erode_px": int(args.human_prior_region_erode_px),
        "human_prior_head_fallback_top_ratio": float(args.human_prior_head_fallback_top_ratio),
        "human_prior_face_fallback_top_ratio": float(args.human_prior_face_fallback_top_ratio),
        "use_fg_mask": str(args.use_fg_mask),
        "fg_mask_source": str(args.fg_mask_source),
        "fg_mask_erode_px": int(args.fg_mask_erode_px),
        "point_loss_fg_erode_px": int(args.point_loss_fg_erode_px),
        "fg_supervision_boost": float(args.fg_supervision_boost),
        "fg_supervision_bg_floor": float(args.fg_supervision_bg_floor),
        "fg_supervision_region_mode": str(args.fg_supervision_region_mode),
        "fg_supervision_region_erode_px": int(args.fg_supervision_region_erode_px),
        "lambda_fg_conf_presence": float(args.lambda_fg_conf_presence),
        "fg_conf_presence_target_ratio": float(args.fg_conf_presence_target_ratio),
        "lambda_fg_structure_depth_edge": float(args.lambda_fg_structure_depth_edge),
        "fg_structure_bbox_margin_px": int(args.fg_structure_bbox_margin_px),
        "fg_structure_bbox_min_side_px": int(args.fg_structure_bbox_min_side_px),
        "fg_structure_region_mode": str(args.fg_structure_region_mode),
        "fg_structure_region_erode_px": int(args.fg_structure_region_erode_px),
        "fg_structure_depth_edge_warmup_steps": int(args.fg_structure_depth_edge_warmup_steps),
        "fg_structure_boundary_probe_px": int(args.fg_structure_boundary_probe_px),
        "fg_structure_edge_support_mode": str(args.fg_structure_edge_support_mode),
        "fg_structure_edge_support_quantile": float(args.fg_structure_edge_support_quantile),
        "fg_structure_edge_support_min_px": int(args.fg_structure_edge_support_min_px),
        "fg_structure_edge_weight_mode": str(args.fg_structure_edge_weight_mode),
        "fg_structure_boundary_falloff_px": int(args.fg_structure_boundary_falloff_px),
        "fg_structure_component_bias_mode": str(args.fg_structure_component_bias_mode),
        "fg_structure_component_bias_threshold_ratio": float(args.fg_structure_component_bias_threshold_ratio),
        "fg_structure_component_bias_other_scale": float(args.fg_structure_component_bias_other_scale),
        "fg_structure_front_depth_bias_mode": str(args.fg_structure_front_depth_bias_mode),
        "fg_structure_front_depth_bias_tau": float(args.fg_structure_front_depth_bias_tau),
        "fg_structure_front_depth_bias_center_quantile": float(args.fg_structure_front_depth_bias_center_quantile),
        "lambda_point_mv_outside_ring": float(args.lambda_point_mv_outside_ring),
        "point_mv_outside_ring_px": int(args.point_mv_outside_ring_px),
        "supervision_weight_mode": str(args.supervision_weight_mode),
        "supervision_weight_mix_alpha": float(args.supervision_weight_mix_alpha),
        "point_mv_consistency": str(args.point_mv_consistency),
        "point_mv_tol_abs": float(args.point_mv_tol_abs),
        "point_mv_tol_rel": float(args.point_mv_tol_rel),
        "point_mv_weight_floor": float(args.point_mv_weight_floor),
        "point_mv_stride": int(args.point_mv_stride),
        "point_mv_depth_max_pairs": int(args.point_mv_depth_max_pairs),
        "point_mv_depth_pair_mode": str(args.point_mv_depth_pair_mode),
        "point_mv_depth_warmup_steps": int(args.point_mv_depth_warmup_steps),
        "point_mv_depth_region_mode": str(args.point_mv_depth_region_mode),
        "point_mv_mask_warmup_steps": int(args.point_mv_mask_warmup_steps),
        "point_mv_depth_inlier_only": str(args.point_mv_depth_inlier_only),
        "point_mv_depth_err_quantile": float(args.point_mv_depth_err_quantile),
        "point_mv_depth_outlier_boost": float(args.point_mv_depth_outlier_boost),
        "point_mv_depth_outlier_cap": float(args.point_mv_depth_outlier_cap),
        "point_mv_depth_tgt_valid_mode": str(args.point_mv_depth_tgt_valid_mode),
        "point_mv_depth_tgt_valid_floor": float(args.point_mv_depth_tgt_valid_floor),
        "point_mv_depth_min_tgt_valid_ratio": float(args.point_mv_depth_min_tgt_valid_ratio),
        "point_mv_mask_min_tgt_fg_ratio": float(args.point_mv_mask_min_tgt_fg_ratio),
        "point_mv_mask_hit_thr": float(args.point_mv_mask_hit_thr),
        "point_mv_mask_soft_blur_px": int(args.point_mv_mask_soft_blur_px),
        "point_mv_mask_soft_blur_iters": int(args.point_mv_mask_soft_blur_iters),
        "point_mv_mask_soft_mix": float(args.point_mv_mask_soft_mix),
        "point_mv_mask_soft_hit_thr": float(args.point_mv_mask_soft_hit_thr),
        "point_mv_depth_tgt_valid_scale_mode": str(args.point_mv_depth_tgt_valid_scale_mode),
        "point_mv_depth_tgt_valid_scale_thr": float(args.point_mv_depth_tgt_valid_scale_thr),
        "point_mv_depth_adapt_mode": str(args.point_mv_depth_adapt_mode),
        "point_mv_depth_adapt_target_valid": float(args.point_mv_depth_adapt_target_valid),
        "point_mv_depth_adapt_min_scale": float(args.point_mv_depth_adapt_min_scale),
        "point_mv_depth_adapt_max_scale": float(args.point_mv_depth_adapt_max_scale),
        "point_support_mode": str(args.point_support_mode),
        "point_support_floor": float(args.point_support_floor),
        "point_mv_depth_support_mode": str(args.point_mv_depth_support_mode),
        "point_mv_depth_support_floor": float(args.point_mv_depth_support_floor),
        "point_mv_mask_support_mode": str(args.point_mv_mask_support_mode),
        "point_mv_mask_support_floor": float(args.point_mv_mask_support_floor),
        "point_mv_depth_fg_erode_px": int(args.point_mv_depth_fg_erode_px),
        "point_cons_quantile": float(args.point_cons_quantile),
        "point_cons_focus": str(args.point_cons_focus),
        "point_residual_quantile": float(args.point_residual_quantile),
        "point_residual_focus": str(args.point_residual_focus),
        "point_residual_boost": float(args.point_residual_boost),
        "point_residual_boost_cap": float(args.point_residual_boost_cap),
        "point_target_mode": str(args.point_target_mode),
        "point_target_blend_alpha": float(args.point_target_blend_alpha),
        "point_target_blend_alpha_min": float(args.point_target_blend_alpha_min),
        "point_target_blend_alpha_max": float(args.point_target_blend_alpha_max),
        "point_target_blend_rel_gain": float(args.point_target_blend_rel_gain),
        "point_target_blend_mv_gain": float(args.point_target_blend_mv_gain),
        "point_target_blend_by_reliability": str(args.point_target_blend_by_reliability),
        "point_target_blend_by_mv_support": str(args.point_target_blend_by_mv_support),
        "point_target_blend_mv_region_mode": str(args.point_target_blend_mv_region_mode),
        "point_target_blend_mv_policy": str(args.point_target_blend_mv_policy),
        "point_target_consensus_alpha_floor": float(args.point_target_consensus_alpha_floor),
        "target_point_frame": str(args.target_point_frame),
        "pred_point_frame": str(args.pred_point_frame),
        "conf_weight_thr": float(args.conf_weight_thr),
        "conf_weight_gamma": float(args.conf_weight_gamma),
        "conf_weight_per_view_quantile": float(args.conf_weight_per_view_quantile),
        "conf_weight_per_view_min_valid": int(args.conf_weight_per_view_min_valid),
        "gram_dyn_enable": str(args.gram_dyn_enable),
        "gram_dyn_layer_idx": int(args.gram_dyn_layer_idx),
        "gram_dyn_quantile": float(args.gram_dyn_quantile),
        "gram_dyn_weight_floor": float(args.gram_dyn_weight_floor),
        "gram_dyn_warmup_steps": int(args.gram_dyn_warmup_steps),
        "dyn_proxy_enable": str(args.dyn_proxy_enable),
        "dyn_proxy_mode": str(args.dyn_proxy_mode),
        "dyn_proxy_use_gram": str(args.dyn_proxy_use_gram),
        "dyn_proxy_use_support": str(args.dyn_proxy_use_support),
        "dyn_proxy_floor": float(args.dyn_proxy_floor),
        "dyn_proxy_warmup_steps": int(args.dyn_proxy_warmup_steps),
        "robust_l1_eps": float(args.robust_l1_eps),
        "point_cons_tau": float(args.point_cons_tau),
        "point_cons_weight_floor": float(args.point_cons_weight_floor),
        "point_cons_clip_min_qv": float(args.point_cons_clip_min_qv),
        "grad_clip": float(args.grad_clip),
        "eval_every_steps": int(args.eval_every_steps),
        "debug_metrics_every_steps": int(args.debug_metrics_every_steps),
        "log_heartbeat_sec": float(args.log_heartbeat_sec),
        "debug_vis_every_steps": int(args.debug_vis_every_steps),
        "debug_vis_max_steps": int(args.debug_vis_max_steps),
        "debug_vis_views": int(args.debug_vis_views),
        "debug_vis_dir": str(args.debug_vis_dir),
        "early_stop_patience": int(args.early_stop_patience),
        "min_improve": float(args.min_improve),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "resume_ckpt": str(args.resume_ckpt),
        "device": device,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")

    eval_every_steps = max(1, int(args.eval_every_steps))
    debug_metrics_every_steps = max(0, int(args.debug_metrics_every_steps))
    debug_vis_every_steps = max(0, int(args.debug_vis_every_steps))
    debug_vis_max_steps = max(0, int(args.debug_vis_max_steps))
    debug_vis_views = max(1, int(args.debug_vis_views))
    debug_vis_saved = 0
    debug_vis_dir = Path(args.debug_vis_dir) if str(args.debug_vis_dir).strip() else (Path(args.log_dir) / "ft_debug_vis")
    if debug_vis_every_steps > 0:
        debug_vis_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[finetune] debug_vis enabled "
            f"every={debug_vis_every_steps} views={debug_vis_views} "
            f"max_steps={debug_vis_max_steps} out={debug_vis_dir}"
        )

    heartbeat_every_sec = max(0.0, float(args.log_heartbeat_sec))
    last_heartbeat_wall = time.monotonic()

    step = 0
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    out_last = Path(args.ckpt_dir) / "model_ft_zju_last.pt"
    out_best = Path(args.ckpt_dir) / "model_ft_zju.pt"
    last_epoch = -1
    pred_point_frame_resolved: Optional[str] = None

    for epoch in range(int(args.epochs)):
        last_epoch = int(epoch)
        model.train()
        if model.aggregator is not None:
            if str(args.freeze_mode).lower() == "all_trainable":
                model.aggregator.train()
            else:
                model.aggregator.eval()
        if model.camera_head is not None:
            if str(args.freeze_mode).lower() == "all_trainable":
                model.camera_head.train()
            else:
                model.camera_head.eval()

        loss_sum = 0.0
        loss_depth_sum = 0.0
        loss_point_sum = 0.0
        loss_point_prior_sum = 0.0
        loss_point_reproj_sum = 0.0
        loss_point_normal_consis_sum = 0.0
        loss_point_mv_depth_sum = 0.0
        loss_point_mv_mask_sum = 0.0
        loss_fg_structure_depth_edge_sum = 0.0
        loss_point_mv_outside_ring_sum = 0.0
        loss_conf_sum = 0.0
        loss_fg_conf_presence_sum = 0.0
        loss_geom_cons_sum = 0.0
        loss_cam_sum = 0.0
        grad_norm_sum = 0.0
        step_update_norm_sum = 0.0
        step_update_ratio_sum = 0.0
        count = 0
        steps_this_epoch = 0
        last_scale_info: Dict[str, float] = {}
        print(f"[finetune] epoch_start epoch={epoch} step={step}", flush=True)
        for sample in dl:
            need_fg_mask = (
                (str(args.use_fg_mask).lower() == "on")
                or (lambda_point_mv_mask > 0.0)
                or (str(args.point_mv_depth_region_mode).strip().lower() == "bg_only")
                or (str(args.point_target_blend_mv_region_mode).strip().lower() == "bg_only")
            )
            need_human_prior = (
                (str(args.human_prior_enable).lower() == "on")
                and (
                    float(args.human_prior_point_blend_alpha) > 0.0
                    or abs(float(args.human_prior_weight_boost) - 1.0) > 1e-6
                    or float(args.human_prior_complete_weight) > 0.0
                    or lambda_point_prior > 0.0
                )
            )
            (
                imgs,
                depth_tgt,
                conf_tgt_raw,
                point_tgt,
                extrinsic_tgt,
                intrinsic_tgt,
                pointmap_source,
                pointmap_frame,
                fg_mask_tgt,
                fg_mask_source,
                human_prior_point_tgt,
                human_prior_valid_mask,
                human_prior_body_mask,
                human_prior_head_mask,
                human_prior_face_mask,
                human_prior_source,
                human_prior_pointmap_frame,
            ) = _sample_to_tensors(
                sample,
                device=device,
                use_fg_mask=need_fg_mask,
                fg_mask_source=str(args.fg_mask_source),
                use_human_prior=need_human_prior,
            )
            last_scale_info.update({
                "point_support_path_active": 0.0,
                "point_mv_depth_support_path_active": 0.0,
                "point_mv_mask_support_path_active": 0.0,
                "point_target_blend_mv_support_active": 0.0,
                "human_prior_requested": 1.0 if need_human_prior else 0.0,
                "human_prior_present": 1.0 if (human_prior_point_tgt is not None or human_prior_body_mask is not None or human_prior_head_mask is not None or human_prior_face_mask is not None) else 0.0,
            })

            imgs_aug = _augment_images(imgs, jitter=float(args.jitter), noise_std=float(args.noise_std))

            agg_tokens_list, ps_idx = model.aggregator(imgs_aug)
            pose_pred = None
            if model.camera_head is not None:
                pose_enc_list = model.camera_head(agg_tokens_list)
                if len(pose_enc_list) > 0:
                    pose_pred = pose_enc_list[-1]
            depth_pred, conf_pred_raw = model.depth_head(agg_tokens_list, imgs_aug, ps_idx)
            point_pred, _ = model.point_head(agg_tokens_list, imgs_aug, ps_idx)

            # Align pseudo target resolution if needed.
            if depth_pred.shape[-3:-1] != depth_tgt.shape[-3:-1]:
                dt = depth_tgt.permute(0, 1, 4, 2, 3).reshape(-1, 1, depth_tgt.shape[2], depth_tgt.shape[3])
                dt = _safe_resize_like(dt, depth_pred.shape[-3:-1], mode="bilinear")
                depth_tgt = dt.reshape(depth_tgt.shape[0], depth_tgt.shape[1], 1, dt.shape[-2], dt.shape[-1]).permute(0, 1, 3, 4, 2)

            if conf_tgt_raw.shape[-2:] != conf_pred_raw.shape[-2:]:
                ct = conf_tgt_raw.reshape(-1, 1, conf_tgt_raw.shape[-2], conf_tgt_raw.shape[-1])
                ct = _safe_resize_like(ct, conf_pred_raw.shape[-2:], mode="nearest")
                conf_tgt_raw = ct.reshape(conf_tgt_raw.shape[0], conf_tgt_raw.shape[1], ct.shape[-2], ct.shape[-1])

            if fg_mask_tgt is not None and fg_mask_tgt.shape[-2:] != conf_pred_raw.shape[-2:]:
                mt = fg_mask_tgt.reshape(-1, 1, fg_mask_tgt.shape[-2], fg_mask_tgt.shape[-1])
                mt = _safe_resize_like(mt, conf_pred_raw.shape[-2:], mode="nearest")
                fg_mask_tgt = mt.reshape(fg_mask_tgt.shape[0], fg_mask_tgt.shape[1], mt.shape[-2], mt.shape[-1])

            if point_pred.shape[-3:-1] != point_tgt.shape[-3:-1]:
                pt = point_tgt.permute(0, 1, 4, 2, 3).reshape(-1, 3, point_tgt.shape[2], point_tgt.shape[3])
                pt = _safe_resize_like(pt, point_pred.shape[-3:-1], mode="bilinear")
                point_tgt = pt.reshape(point_tgt.shape[0], point_tgt.shape[1], 3, pt.shape[-2], pt.shape[-1]).permute(0, 1, 3, 4, 2)

            if human_prior_point_tgt is not None and point_pred.shape[-3:-1] != human_prior_point_tgt.shape[-3:-1]:
                hp = human_prior_point_tgt.permute(0, 1, 4, 2, 3).reshape(-1, 3, human_prior_point_tgt.shape[2], human_prior_point_tgt.shape[3])
                hp = _safe_resize_like(hp, point_pred.shape[-3:-1], mode="bilinear")
                human_prior_point_tgt = hp.reshape(
                    human_prior_point_tgt.shape[0],
                    human_prior_point_tgt.shape[1],
                    3,
                    hp.shape[-2],
                    hp.shape[-1],
                ).permute(0, 1, 3, 4, 2)
            if human_prior_valid_mask is not None and human_prior_valid_mask.shape[-2:] != conf_pred_raw.shape[-2:]:
                hm = human_prior_valid_mask.reshape(-1, 1, human_prior_valid_mask.shape[-2], human_prior_valid_mask.shape[-1])
                hm = _safe_resize_like(hm, conf_pred_raw.shape[-2:], mode="nearest")
                human_prior_valid_mask = hm.reshape(human_prior_valid_mask.shape[0], human_prior_valid_mask.shape[1], hm.shape[-2], hm.shape[-1])
            if human_prior_body_mask is not None and human_prior_body_mask.shape[-2:] != conf_pred_raw.shape[-2:]:
                hm = human_prior_body_mask.reshape(-1, 1, human_prior_body_mask.shape[-2], human_prior_body_mask.shape[-1])
                hm = _safe_resize_like(hm, conf_pred_raw.shape[-2:], mode="nearest")
                human_prior_body_mask = hm.reshape(human_prior_body_mask.shape[0], human_prior_body_mask.shape[1], hm.shape[-2], hm.shape[-1])
            if human_prior_head_mask is not None and human_prior_head_mask.shape[-2:] != conf_pred_raw.shape[-2:]:
                hm = human_prior_head_mask.reshape(-1, 1, human_prior_head_mask.shape[-2], human_prior_head_mask.shape[-1])
                hm = _safe_resize_like(hm, conf_pred_raw.shape[-2:], mode="nearest")
                human_prior_head_mask = hm.reshape(human_prior_head_mask.shape[0], human_prior_head_mask.shape[1], hm.shape[-2], hm.shape[-1])
            if human_prior_face_mask is not None and human_prior_face_mask.shape[-2:] != conf_pred_raw.shape[-2:]:
                hm = human_prior_face_mask.reshape(-1, 1, human_prior_face_mask.shape[-2], human_prior_face_mask.shape[-1])
                hm = _safe_resize_like(hm, conf_pred_raw.shape[-2:], mode="nearest")
                human_prior_face_mask = hm.reshape(human_prior_face_mask.shape[0], human_prior_face_mask.shape[1], hm.shape[-2], hm.shape[-1])

            point_tgt_input_raw = point_tgt
            point_pred_raw = point_pred

            point_tgt_frame = str(pointmap_frame or "").strip().lower()
            tgt_point_frame_cfg = str(args.target_point_frame).strip().lower()
            if tgt_point_frame_cfg in {"world", "camera"}:
                point_tgt_frame = tgt_point_frame_cfg
            if point_tgt_frame not in {"world", "camera"}:
                if extrinsic_tgt is not None and intrinsic_tgt is not None:
                    with torch.no_grad():
                        point_tgt_frame, tgt_frame_info = _resolve_point_frame_auto(
                            point_tgt.detach(), extrinsic_tgt.detach(), intrinsic_tgt.detach()
                        )
                    last_scale_info["point_tgt_frame_auto"] = 1.0
                    last_scale_info["point_tgt_frame_err_world"] = float(tgt_frame_info["point_frame_err_world"])
                    last_scale_info["point_tgt_frame_err_camera"] = float(tgt_frame_info["point_frame_err_camera"])
                else:
                    point_tgt_frame = "world"
            if point_tgt_frame == "camera" and extrinsic_tgt is not None:
                point_tgt = _cam_to_world_point_map_torch(point_tgt, extrinsic_tgt)
                last_scale_info["point_tgt_cam2world"] = 1.0
            last_scale_info["point_tgt_frame"] = str(point_tgt_frame)

            pred_point_frame_cfg = str(args.pred_point_frame).strip().lower()
            pred_point_frame = pred_point_frame_cfg
            if pred_point_frame_cfg == "auto":
                if pred_point_frame_resolved is None:
                    if extrinsic_tgt is not None and intrinsic_tgt is not None:
                        with torch.no_grad():
                            pred_point_frame_resolved, pred_frame_info = _resolve_point_frame_auto(
                                point_pred.detach(), extrinsic_tgt.detach(), intrinsic_tgt.detach()
                            )
                        print(
                            "[finetune] resolved pred_point_frame:",
                            pred_point_frame_resolved,
                            f"(err_world={pred_frame_info['point_frame_err_world']:.3f}, "
                            f"err_camera={pred_frame_info['point_frame_err_camera']:.3f})",
                        )
                    else:
                        pred_point_frame_resolved = "world"
                pred_point_frame = str(pred_point_frame_resolved or "world")
            if pred_point_frame == "camera" and extrinsic_tgt is not None:
                point_pred = _cam_to_world_point_map_torch(point_pred, extrinsic_tgt)
                last_scale_info["point_pred_cam2world"] = 1.0
            last_scale_info["point_pred_frame"] = str(pred_point_frame)

            human_prior_point_world = human_prior_point_tgt
            human_prior_point_frame = str(human_prior_pointmap_frame or "").strip().lower()
            if human_prior_point_world is not None:
                if human_prior_point_frame not in {"world", "camera"}:
                    if extrinsic_tgt is not None and intrinsic_tgt is not None:
                        with torch.no_grad():
                            human_prior_point_frame, human_prior_frame_info = _resolve_point_frame_auto(
                                human_prior_point_world.detach(),
                                extrinsic_tgt.detach(),
                                intrinsic_tgt.detach(),
                            )
                        last_scale_info["human_prior_point_frame_auto"] = 1.0
                        last_scale_info["human_prior_point_frame_err_world"] = float(human_prior_frame_info["point_frame_err_world"])
                        last_scale_info["human_prior_point_frame_err_camera"] = float(human_prior_frame_info["point_frame_err_camera"])
                    else:
                        human_prior_point_frame = "world"
                if human_prior_point_frame == "camera" and extrinsic_tgt is not None:
                    human_prior_point_world = _cam_to_world_point_map_torch(human_prior_point_world, extrinsic_tgt)
                    last_scale_info["human_prior_point_cam2world"] = 1.0
                last_scale_info["human_prior_point_frame"] = str(human_prior_point_frame)
                last_scale_info["human_prior_source"] = str(human_prior_source or "")
            else:
                last_scale_info["human_prior_point_frame"] = ""

            if human_prior_valid_mask is None and human_prior_point_world is not None:
                human_prior_valid_mask = torch.isfinite(human_prior_point_world).all(dim=-1).float()
            if human_prior_body_mask is None and human_prior_valid_mask is not None:
                human_prior_body_mask = human_prior_valid_mask.detach().clone()

            human_prior_erode_px = int(max(0, int(args.human_prior_region_erode_px)))
            if human_prior_body_mask is not None and human_prior_erode_px > 0:
                human_prior_body_mask = _erode_mask_tensor(human_prior_body_mask, human_prior_erode_px)
            if human_prior_head_mask is not None and human_prior_erode_px > 0:
                human_prior_head_mask = _erode_mask_tensor(human_prior_head_mask, human_prior_erode_px)
            if human_prior_face_mask is not None and human_prior_erode_px > 0:
                human_prior_face_mask = _erode_mask_tensor(human_prior_face_mask, human_prior_erode_px)

            conf_tgt = _to_depth01(conf_tgt_raw)
            conf_pred = _to_depth01(conf_pred_raw)
            valid_all = (depth_tgt[..., 0] > 1e-6).float()
            valid = valid_all
            fg_mask_used = None
            if fg_mask_tgt is not None:
                fg_mask_used = fg_mask_tgt.clamp(0.0, 1.0)
                erode_px = int(max(0, int(args.fg_mask_erode_px)))
                if erode_px > 0:
                    fg_mask_used = _erode_mask_tensor(fg_mask_used, erode_px)
                    last_scale_info["fg_mask_erode_px"] = float(erode_px)
                    last_scale_info["fg_cover_eroded"] = float(fg_mask_used.mean().item())
                last_scale_info["fg_cover"] = float(fg_mask_used.mean().item())
                last_scale_info["fg_mask_source"] = str(fg_mask_source or "")
                if str(args.use_fg_mask).lower() == "on":
                    valid = valid_all * fg_mask_used
                    last_scale_info["fg_mask_applied_to_valid"] = 1.0
                else:
                    last_scale_info["fg_mask_applied_to_valid"] = 0.0
            else:
                last_scale_info["fg_cover"] = -1.0
            supervision_valid = valid
            fg_supervision_bg_floor = float(np.clip(float(args.fg_supervision_bg_floor), 0.0, 1.0))
            last_scale_info["fg_supervision_bg_floor"] = float(fg_supervision_bg_floor)
            if (
                fg_mask_used is not None
                and str(args.use_fg_mask).lower() == "on"
                and fg_supervision_bg_floor > 0.0
            ):
                supervision_valid = (
                    valid_all * (
                        fg_mask_used
                        + fg_supervision_bg_floor * (1.0 - fg_mask_used)
                    )
                ).clamp(0.0, 1.0)
            last_scale_info["supervision_valid_cover"] = float(supervision_valid.mean().item())
            if fg_mask_used is not None:
                fg_region = (fg_mask_used > 0.5)
                bg_region = ~fg_region
                if int(fg_region.sum().item()) > 0:
                    last_scale_info["supervision_valid_fg_mean"] = float(supervision_valid[fg_region].mean().item())
                if int(bg_region.sum().item()) > 0:
                    last_scale_info["supervision_valid_bg_mean"] = float(supervision_valid[bg_region].mean().item())

            human_prior_blend_mask = _resolve_human_prior_region_mask(
                region_mode=str(args.human_prior_point_blend_region),
                valid_mask01=human_prior_valid_mask,
                body_mask01=human_prior_body_mask,
                head_mask01=human_prior_head_mask,
                face_mask01=human_prior_face_mask,
                head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
                face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
            )
            human_prior_weight_mask = _resolve_human_prior_region_mask(
                region_mode=str(args.human_prior_weight_region),
                valid_mask01=human_prior_valid_mask,
                body_mask01=human_prior_body_mask,
                head_mask01=human_prior_head_mask,
                face_mask01=human_prior_face_mask,
                head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
                face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
            )
            human_prior_loss_mask = _resolve_human_prior_region_mask(
                region_mode=str(args.human_prior_loss_region),
                valid_mask01=human_prior_valid_mask,
                body_mask01=human_prior_body_mask,
                head_mask01=human_prior_head_mask,
                face_mask01=human_prior_face_mask,
                head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
                face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
            )
            human_prior_complete_mask = _resolve_human_prior_region_mask(
                region_mode=str(args.human_prior_complete_region),
                valid_mask01=human_prior_valid_mask,
                body_mask01=human_prior_body_mask,
                head_mask01=human_prior_head_mask,
                face_mask01=human_prior_face_mask,
                head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
                face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
            )
            if human_prior_blend_mask is not None and human_prior_valid_mask is not None:
                human_prior_blend_mask = (human_prior_blend_mask * human_prior_valid_mask).clamp(0.0, 1.0)
            if human_prior_weight_mask is not None and human_prior_valid_mask is not None:
                human_prior_weight_mask = (human_prior_weight_mask * human_prior_valid_mask).clamp(0.0, 1.0)
            if human_prior_loss_mask is not None and human_prior_valid_mask is not None:
                human_prior_loss_mask = (human_prior_loss_mask * human_prior_valid_mask).clamp(0.0, 1.0)
            if human_prior_complete_mask is not None and human_prior_valid_mask is not None:
                human_prior_complete_mask = (human_prior_complete_mask * human_prior_valid_mask).clamp(0.0, 1.0)
            if human_prior_complete_mask is not None:
                human_prior_complete_mask = (human_prior_complete_mask * (1.0 - valid_all)).clamp(0.0, 1.0)
            if human_prior_valid_mask is not None:
                last_scale_info["human_prior_valid_cover"] = float(human_prior_valid_mask.mean().item())
            if human_prior_body_mask is not None:
                last_scale_info["human_prior_body_cover"] = float(human_prior_body_mask.mean().item())
            if human_prior_head_mask is not None:
                last_scale_info["human_prior_head_cover"] = float(human_prior_head_mask.mean().item())
            if human_prior_face_mask is not None:
                last_scale_info["human_prior_face_cover"] = float(human_prior_face_mask.mean().item())
            if human_prior_blend_mask is not None:
                last_scale_info["human_prior_blend_cover"] = float(human_prior_blend_mask.mean().item())
            if human_prior_weight_mask is not None:
                last_scale_info["human_prior_weight_cover"] = float(human_prior_weight_mask.mean().item())
            if human_prior_loss_mask is not None:
                last_scale_info["human_prior_loss_cover"] = float(human_prior_loss_mask.mean().item())
            if human_prior_complete_mask is not None:
                last_scale_info["human_prior_complete_cover"] = float(human_prior_complete_mask.mean().item())

            conf_weight_base = _build_conf_weight(
                conf01=conf_tgt,
                valid01=supervision_valid,
                thr=float(args.conf_weight_thr),
                gamma=float(args.conf_weight_gamma),
            )
            conf_pvq_mask, conf_pvq_info = _build_per_view_conf_quantile_mask(
                conf01=conf_tgt,
                valid01=supervision_valid,
                quantile=float(args.conf_weight_per_view_quantile),
                min_valid=int(args.conf_weight_per_view_min_valid),
            )
            conf_weight_base = conf_weight_base * conf_pvq_mask
            gram_dyn_weight = None
            gram_dyn_weight_raw = None
            gram_dyn_info = {
                "gram_dyn_enabled": 0.0,
                "gram_dyn_keep_ratio": 1.0,
                "gram_dyn_mean": 1.0,
                "gram_dyn_p10": 1.0,
                "gram_dyn_p90": 1.0,
            }
            dyn_proxy_enabled = str(args.dyn_proxy_enable).strip().lower() == "on"
            if str(args.gram_dyn_enable).strip().lower() == "on":
                gram_dyn_weight_raw, gram_dyn_info_raw = _build_gram_dynamic_weight_from_tokens(
                    aggregated_tokens_list=agg_tokens_list,
                    patch_start_idx=int(ps_idx),
                    target_hw=(int(conf_pred_raw.shape[-2]), int(conf_pred_raw.shape[-1])),
                    layer_idx=int(args.gram_dyn_layer_idx),
                    quantile=float(args.gram_dyn_quantile),
                    weight_floor=float(args.gram_dyn_weight_floor),
                )
                gram_dyn_info.update(gram_dyn_info_raw)
                if gram_dyn_weight_raw is not None:
                    gram_warm = max(0, int(args.gram_dyn_warmup_steps))
                    gram_scale = 1.0 if gram_warm <= 0 else min(1.0, float(step + 1) / float(max(1, gram_warm)))
                    gram_dyn_weight = (
                        1.0 - float(gram_scale) * (1.0 - gram_dyn_weight_raw.clamp(0.0, 1.0))
                    ).clamp(0.0, 1.0)
                    gram_dyn_info["gram_dyn_scale"] = float(gram_scale)
                    if not dyn_proxy_enabled:
                        conf_weight_base = conf_weight_base * gram_dyn_weight
                        gram_dyn_info["gram_dyn_applied"] = 1.0
                    else:
                        gram_dyn_info["gram_dyn_applied"] = 0.0
                else:
                    gram_dyn_info["gram_dyn_scale"] = 0.0
                    gram_dyn_info["gram_dyn_applied"] = 0.0
            else:
                gram_dyn_info["gram_dyn_scale"] = 0.0
                gram_dyn_info["gram_dyn_applied"] = 0.0
            weight_mode = str(args.supervision_weight_mode).strip().lower()
            if weight_mode == "uniform":
                conf_weight = supervision_valid
            elif weight_mode == "mix":
                alpha = float(min(1.0, max(0.0, args.supervision_weight_mix_alpha)))
                conf_weight = supervision_valid * (alpha * conf_weight_base + (1.0 - alpha))
                last_scale_info["sup_w_mix_alpha"] = float(alpha)
            else:
                conf_weight = conf_weight_base
                weight_mode = "conf"
            w_base = conf_weight.detach()
            fg_boost_mask_used, fg_boost_mask_info = _build_fg_supervision_boost_mask(
                fg_mask01=(fg_mask_used.detach() if fg_mask_used is not None else None),
                region_mode=str(args.fg_supervision_region_mode),
                region_erode_px=int(args.fg_supervision_region_erode_px),
            )
            w_base, fg_boost_info = _apply_fg_supervision_boost(
                base_weight01=w_base,
                fg_mask01=fg_boost_mask_used,
                fg_boost=float(args.fg_supervision_boost),
                fg_stats_mask01=(fg_mask_used.detach() if fg_mask_used is not None else None),
            )
            w_base, human_prior_weight_info = _apply_region_weight_boost(
                base_weight01=w_base,
                region_mask01=(human_prior_weight_mask.detach() if human_prior_weight_mask is not None else None),
                boost=float(args.human_prior_weight_boost),
                info_prefix="human_prior_weight",
                fg_stats_mask01=(fg_mask_used.detach() if fg_mask_used is not None else None),
            )
            w = w_base
            denom = w.sum() + 1e-6
            last_scale_info["sup_w_mode"] = weight_mode
            last_scale_info["sup_w_nonzero_ratio"] = float((conf_weight > 0.0).float().mean().item())
            last_scale_info["conf_w_mean"] = float(conf_weight.mean().item())
            last_scale_info["conf_w_thr"] = float(args.conf_weight_thr)
            last_scale_info["conf_w_gamma"] = float(args.conf_weight_gamma)
            last_scale_info["human_prior_blend_region"] = str(args.human_prior_point_blend_region)
            last_scale_info["human_prior_weight_region"] = str(args.human_prior_weight_region)
            last_scale_info["human_prior_loss_region"] = str(args.human_prior_loss_region)
            last_scale_info["human_prior_complete_region"] = str(args.human_prior_complete_region)
            last_scale_info["human_prior_complete_weight"] = float(args.human_prior_complete_weight)
            last_scale_info.update(conf_pvq_info)
            last_scale_info.update(gram_dyn_info)
            last_scale_info.update(fg_boost_mask_info)
            last_scale_info.update(fg_boost_info)
            last_scale_info.update(human_prior_weight_info)
            if str(args.depth_scale_align).lower() == "median":
                depth_pred_for_loss, scale_info = _align_depth_median_scale(depth_pred, depth_tgt, valid)
                last_scale_info.update(scale_info)
            else:
                depth_pred_for_loss = depth_pred

            point_loss_scale = 1.0
            src_l = str(pointmap_source).strip().lower()
            if src_l == "depth_unproject":
                point_loss_scale = float(args.point_loss_scale_depth_unproject)

            point_from_depth_tgt: Optional[torch.Tensor] = None
            if (
                lambda_point > 0.0
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
                and depth_tgt.shape[:2] == point_tgt.shape[:2]
            ):
                with torch.no_grad():
                    point_from_depth_tgt = _depth_to_world_point_map_torch(
                        depth_tgt, extrinsic_tgt, intrinsic_tgt
                    )

            point_reliability = None
            if (
                lambda_point > 0.0
                and point_from_depth_tgt is not None
            ):
                with torch.no_grad():
                    cons_err_tgt = torch.linalg.norm((point_from_depth_tgt - point_tgt), dim=-1)
                    tau = float(max(1e-6, args.point_cons_tau))
                    floor = float(min(1.0, max(0.0, args.point_cons_weight_floor)))
                    point_reliability = torch.exp(-cons_err_tgt / tau).clamp(min=floor, max=1.0)
                    q = float(min(1.0, max(0.0, args.point_cons_quantile)))
                    cons_focus = str(args.point_cons_focus).strip().lower()
                    if cons_focus not in {"inlier", "outlier", "all"}:
                        cons_focus = "inlier"
                    last_scale_info["point_cons_focus"] = cons_focus
                    valid_rel = (valid > 0.5)
                    if q < 1.0 and cons_focus != "all" and int(valid_rel.sum().item()) >= 16:
                        q_sel = max(0.0, 1.0 - q) if cons_focus == "outlier" else q
                        qv = torch.quantile(cons_err_tgt[valid_rel], q_sel)
                        qv_f = float(qv.item())
                        clip_min_qv = float(max(0.0, float(args.point_cons_clip_min_qv)))
                        last_scale_info["point_cons_clip_min_qv"] = clip_min_qv
                        last_scale_info["point_cons_q"] = float(q)
                        last_scale_info["point_cons_err_qv"] = qv_f
                        if qv_f > clip_min_qv:
                            if cons_focus == "outlier":
                                keep = (cons_err_tgt >= qv).float()
                            else:
                                keep = (cons_err_tgt <= qv).float()
                            point_reliability = point_reliability * keep
                            last_scale_info["point_rel_keep_ratio"] = float(keep[valid_rel].mean().item())
                            last_scale_info["point_cons_clip_applied"] = 1.0
                        else:
                            last_scale_info["point_cons_clip_applied"] = 0.0
                            last_scale_info["point_cons_clip_skipped_low_qv"] = 1.0
                    last_scale_info["point_rel_mean"] = float(point_reliability.mean().item())
                    last_scale_info["point_rel_min"] = float(point_reliability.min().item())
                    last_scale_info["point_rel_max"] = float(point_reliability.max().item())

            point_tgt_for_loss = point_tgt
            point_target_mode = str(args.point_target_mode).strip().lower()
            last_scale_info["point_target_mode"] = point_target_mode
            point_mv_support_pseudo = None
            blend_by_mv_support = (
                lambda_point > 0.0
                and point_from_depth_tgt is not None
                and (point_target_mode in {"blend", "depth_consensus_unproject"})
                and str(args.point_target_blend_by_mv_support).strip().lower() == "on"
                and str(args.point_mv_consistency).strip().lower() == "on"
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            )
            if blend_by_mv_support:
                with torch.no_grad():
                    point_mv_support_pseudo = _point_multiview_support_weight(
                        point_world=point_tgt.detach(),
                        depth_tgt=depth_tgt.detach(),
                        extrinsic_w2c=extrinsic_tgt.detach(),
                        intrinsic=intrinsic_tgt.detach(),
                        valid_mask=valid.detach(),
                        tol_abs=float(args.point_mv_tol_abs),
                        tol_rel=float(args.point_mv_tol_rel),
                        weight_floor=float(args.point_mv_weight_floor),
                        stride=int(args.point_mv_stride),
                    )
                    last_scale_info["point_mv_pseudo_mean"] = float(point_mv_support_pseudo.mean().item())
                    last_scale_info["point_mv_pseudo_min"] = float(point_mv_support_pseudo.min().item())
                    last_scale_info["point_mv_pseudo_max"] = float(point_mv_support_pseudo.max().item())
                    _append_region_distribution_stats(
                        info=last_scale_info,
                        prefix="point_mv_pseudo_support",
                        value01=point_mv_support_pseudo,
                        fg_mask01=fg_mask_used,
                        active_mask01=valid,
                    )
            if lambda_point > 0.0 and point_from_depth_tgt is not None:
                if point_target_mode == "depth_unproject":
                    point_tgt_for_loss = point_from_depth_tgt
                elif point_target_mode == "depth_consensus_unproject":
                    if point_mv_support_pseudo is not None:
                        alpha_map = point_mv_support_pseudo.clamp(0.0, 1.0)
                        mv_policy = str(args.point_target_blend_mv_policy).strip().lower()
                        if mv_policy == "weak_to_depth":
                            alpha_map = 1.0 - alpha_map
                        else:
                            mv_policy = "strong_to_depth"
                        last_scale_info["point_target_consensus_mv_policy"] = str(mv_policy)
                        alpha_floor = float(min(1.0, max(0.0, args.point_target_consensus_alpha_floor)))
                        if alpha_floor > 0.0:
                            alpha_map = torch.maximum(alpha_map, torch.full_like(alpha_map, alpha_floor))
                            last_scale_info["point_target_consensus_alpha_floor"] = float(alpha_floor)
                        blend_mv_region_mode = str(args.point_target_blend_mv_region_mode).strip().lower()
                        if blend_mv_region_mode not in {"all", "bg_only"}:
                            blend_mv_region_mode = "all"
                        last_scale_info["point_target_blend_mv_region_mode"] = str(blend_mv_region_mode)
                        if blend_mv_region_mode == "bg_only" and fg_mask_used is not None:
                            bg_gate = (1.0 - fg_mask_used.detach()).clamp(0.0, 1.0)
                            alpha_map = alpha_map * bg_gate + (1.0 - bg_gate)
                            last_scale_info["point_target_consensus_bg_gate_mean"] = float(bg_gate.mean().item())
                        point_tgt_for_loss = (
                            alpha_map[..., None] * point_from_depth_tgt
                            + (1.0 - alpha_map[..., None]) * point_tgt
                        )
                        last_scale_info["point_target_consensus_alpha_mean"] = float(alpha_map.mean().item())
                        last_scale_info["point_target_consensus_alpha_min"] = float(alpha_map.min().item())
                        last_scale_info["point_target_consensus_alpha_max"] = float(alpha_map.max().item())
                    else:
                        point_tgt_for_loss = point_from_depth_tgt
                        last_scale_info["point_target_consensus_fallback_depth"] = 1.0
                elif point_target_mode == "blend":
                    alpha_base = float(min(1.0, max(0.0, args.point_target_blend_alpha)))
                    alpha_lo = float(min(1.0, max(0.0, args.point_target_blend_alpha_min)))
                    alpha_hi = float(min(1.0, max(alpha_lo, args.point_target_blend_alpha_max)))
                    rel_gain = float(min(1.0, max(0.0, args.point_target_blend_rel_gain)))
                    mv_gain = float(min(1.0, max(0.0, args.point_target_blend_mv_gain)))
                    if str(args.point_target_blend_by_reliability).strip().lower() == "on" and point_reliability is not None:
                        alpha_map = alpha_base + (1.0 - alpha_base) * rel_gain * (1.0 - point_reliability)
                        alpha_map = alpha_map.clamp(alpha_lo, alpha_hi)
                    else:
                        alpha_map = torch.full_like(valid, alpha_base)
                    blend_mv_region_mode = str(args.point_target_blend_mv_region_mode).strip().lower()
                    if blend_mv_region_mode not in {"all", "bg_only"}:
                        blend_mv_region_mode = "all"
                    last_scale_info["point_target_blend_mv_region_mode"] = str(blend_mv_region_mode)
                    if point_mv_support_pseudo is not None:
                        alpha_before_mv = alpha_map
                        mv_policy = str(args.point_target_blend_mv_policy).strip().lower()
                        if mv_policy == "strong_to_depth":
                            # Increase depth-unproject ratio where pseudo pointmap has strong cross-view support.
                            alpha_map = alpha_map + (1.0 - alpha_map) * mv_gain * point_mv_support_pseudo
                        else:
                            # Increase depth-unproject ratio where pseudo pointmap has weak cross-view support.
                            alpha_map = alpha_map + (1.0 - alpha_map) * mv_gain * (1.0 - point_mv_support_pseudo)
                            mv_policy = "weak_to_depth"
                        if blend_mv_region_mode == "bg_only" and fg_mask_used is not None:
                            bg_gate = (1.0 - fg_mask_used.detach()).clamp(0.0, 1.0)
                            alpha_map = alpha_before_mv + (alpha_map - alpha_before_mv) * bg_gate
                            last_scale_info["point_target_blend_mv_bg_gate_mean"] = float(bg_gate.mean().item())
                        alpha_map = alpha_map.clamp(alpha_lo, alpha_hi)
                        last_scale_info["point_target_blend_by_mv_support"] = 1.0
                        last_scale_info["point_target_blend_mv_support_active"] = 1.0
                        last_scale_info["point_target_blend_mv_policy"] = str(mv_policy)
                    point_tgt_for_loss = alpha_map[..., None] * point_from_depth_tgt + (1.0 - alpha_map[..., None]) * point_tgt
                    last_scale_info["point_target_blend_alpha_base"] = float(alpha_base)
                    last_scale_info["point_target_blend_alpha_min_cfg"] = float(alpha_lo)
                    last_scale_info["point_target_blend_alpha_max_cfg"] = float(alpha_hi)
                    last_scale_info["point_target_blend_rel_gain"] = float(rel_gain)
                    last_scale_info["point_target_blend_mv_gain"] = float(mv_gain)
                    last_scale_info["point_target_blend_alpha_mean"] = float(alpha_map.mean().item())
                    last_scale_info["point_target_blend_alpha_min"] = float(alpha_map.min().item())
                    last_scale_info["point_target_blend_alpha_max"] = float(alpha_map.max().item())
            elif point_target_mode != "pointmap":
                last_scale_info["point_target_fallback"] = 1.0

            human_prior_blend_alpha = float(min(1.0, max(0.0, args.human_prior_point_blend_alpha)))
            last_scale_info["human_prior_blend_applied"] = 0.0
            last_scale_info["human_prior_blend_alpha_cfg"] = float(human_prior_blend_alpha)
            if (
                lambda_point > 0.0
                and human_prior_point_world is not None
                and human_prior_blend_mask is not None
                and human_prior_blend_alpha > 0.0
            ):
                human_prior_finite_mask = torch.isfinite(human_prior_point_world).all(dim=-1).float()
                alpha_map = (human_prior_blend_mask * human_prior_finite_mask * human_prior_blend_alpha).clamp(0.0, 1.0)
                if float(alpha_map.max().item()) > 0.0:
                    point_tgt_for_loss_before_human_prior = point_tgt_for_loss
                    point_tgt_for_loss = (
                        alpha_map[..., None] * human_prior_point_world
                        + (1.0 - alpha_map[..., None]) * point_tgt_for_loss
                    )
                    last_scale_info["human_prior_blend_applied"] = 1.0
                    last_scale_info["human_prior_blend_alpha_mean"] = float(alpha_map.mean().item())
                    last_scale_info["human_prior_blend_alpha_min"] = float(alpha_map.min().item())
                    last_scale_info["human_prior_blend_alpha_max"] = float(alpha_map.max().item())
                    blend_active = (alpha_map > 0.0)
                    if int(blend_active.sum().item()) > 0:
                        last_scale_info["human_prior_blend_active_ratio"] = float(blend_active.float().mean().item())
                        blend_shift = torch.linalg.norm(
                            (point_tgt_for_loss.detach() - point_tgt_for_loss_before_human_prior.detach()),
                            dim=-1,
                        )
                        last_scale_info["human_prior_blend_shift_l2"] = float(blend_shift[blend_active].mean().item())

            human_prior_complete_weight = float(max(0.0, args.human_prior_complete_weight))
            last_scale_info["human_prior_complete_applied"] = 0.0
            if (
                lambda_point > 0.0
                and human_prior_point_world is not None
                and human_prior_complete_mask is not None
                and human_prior_complete_weight > 0.0
            ):
                human_prior_complete_finite = torch.isfinite(human_prior_point_world).all(dim=-1).float()
                completion_mask = (human_prior_complete_mask * human_prior_complete_finite).clamp(0.0, 1.0)
                if float(completion_mask.max().item()) > 0.0:
                    point_tgt_for_loss = (
                        completion_mask[..., None] * human_prior_point_world
                        + (1.0 - completion_mask[..., None]) * point_tgt_for_loss
                    )
                    last_scale_info["human_prior_complete_applied"] = 1.0
                    last_scale_info["human_prior_complete_active_ratio"] = float((completion_mask > 0.0).float().mean().item())

            point_mv_support = None
            if (
                (lambda_point > 0.0 or lambda_point_mv_depth > 0.0)
                and str(args.point_mv_consistency).lower() == "on"
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                point_tgt_for_support = point_tgt_for_loss if lambda_point > 0.0 else point_tgt
                with torch.no_grad():
                    point_mv_support = _point_multiview_support_weight(
                        point_world=point_tgt_for_support.detach(),
                        depth_tgt=depth_tgt.detach(),
                        extrinsic_w2c=extrinsic_tgt.detach(),
                        intrinsic=intrinsic_tgt.detach(),
                        valid_mask=valid.detach(),
                        tol_abs=float(args.point_mv_tol_abs),
                        tol_rel=float(args.point_mv_tol_rel),
                        weight_floor=float(args.point_mv_weight_floor),
                        stride=int(args.point_mv_stride),
                    )
                    last_scale_info["point_mv_mean"] = float(point_mv_support.mean().item())
                    last_scale_info["point_mv_min"] = float(point_mv_support.min().item())
                    last_scale_info["point_mv_max"] = float(point_mv_support.max().item())
                    mv_active = (valid > 0.5)
                    if int(mv_active.sum().item()) > 0:
                        last_scale_info["point_mv_active_ratio"] = float(point_mv_support[mv_active].mean().item())
                    _append_region_distribution_stats(
                        info=last_scale_info,
                        prefix="point_mv_support",
                        value01=point_mv_support,
                        fg_mask01=fg_mask_used,
                        active_mask01=valid,
                    )

            if point_mv_support_pseudo is not None:
                if point_mv_support is None:
                    point_mv_support = point_mv_support_pseudo
                    last_scale_info["point_mv_mode"] = "pseudo_only"
                else:
                    point_mv_support = torch.minimum(point_mv_support, point_mv_support_pseudo)
                    last_scale_info["point_mv_mode"] = "min_target_pseudo"
                last_scale_info["point_mv_mean"] = float(point_mv_support.mean().item())
                last_scale_info["point_mv_min"] = float(point_mv_support.min().item())
                last_scale_info["point_mv_max"] = float(point_mv_support.max().item())
                mv_active = (valid > 0.5)
                if int(mv_active.sum().item()) > 0:
                    last_scale_info["point_mv_active_ratio"] = float(point_mv_support[mv_active].mean().item())
                _append_region_distribution_stats(
                    info=last_scale_info,
                    prefix="point_mv_support",
                    value01=point_mv_support,
                    fg_mask01=fg_mask_used,
                    active_mask01=valid,
                )

            dyn_proxy_info = {
                "dyn_proxy_enabled": 1.0 if dyn_proxy_enabled else 0.0,
                "dyn_proxy_keep_ratio": 1.0,
                "dyn_proxy_mean": 1.0,
                "dyn_proxy_p10": 1.0,
                "dyn_proxy_p90": 1.0,
                "dyn_proxy_fg_mean": 1.0,
                "dyn_proxy_bg_mean": 1.0,
                "dyn_proxy_scale": 0.0,
                "dyn_proxy_applied": 0.0,
                "dyn_proxy_floor": float(args.dyn_proxy_floor),
                "dyn_proxy_use_gram": 1.0 if str(args.dyn_proxy_use_gram).strip().lower() == "on" else 0.0,
                "dyn_proxy_use_support": 1.0 if str(args.dyn_proxy_use_support).strip().lower() == "on" else 0.0,
            }
            dyn_proxy_weight = None
            if dyn_proxy_enabled:
                dyn_proxy_weight_raw, dyn_proxy_info_raw = _build_dyn_proxy_weight(
                    fg_mask01=fg_mask_used.detach() if fg_mask_used is not None else None,
                    gram_static_weight=(
                        gram_dyn_weight_raw.detach() if gram_dyn_weight_raw is not None else None
                    ),
                    support01=point_mv_support.detach() if point_mv_support is not None else None,
                    use_gram=(str(args.dyn_proxy_use_gram).strip().lower() == "on"),
                    use_support=(str(args.dyn_proxy_use_support).strip().lower() == "on"),
                    floor=float(args.dyn_proxy_floor),
                )
                dyn_proxy_info.update(dyn_proxy_info_raw)
                if dyn_proxy_weight_raw is not None:
                    dyn_warm = max(0, int(args.dyn_proxy_warmup_steps))
                    dyn_scale = 1.0 if dyn_warm <= 0 else min(1.0, float(step + 1) / float(max(1, dyn_warm)))
                    dyn_proxy_weight = (
                        1.0 - float(dyn_scale) * (1.0 - dyn_proxy_weight_raw.clamp(0.0, 1.0))
                    ).clamp(0.0, 1.0).detach()
                    dyn_proxy_info["dyn_proxy_scale"] = float(dyn_scale)
                    dyn_proxy_info["dyn_proxy_applied"] = 1.0
                    w = (w_base * dyn_proxy_weight).detach()
                    denom = w.sum() + 1e-6
            last_scale_info.update(dyn_proxy_info)

            if lambda_depth > 0.0:
                depth_abs = _robust_abs(depth_pred_for_loss - depth_tgt, float(args.robust_l1_eps))[..., 0]
                loss_depth = (depth_abs * w).sum() / denom
            else:
                loss_depth = torch.zeros([], device=device, dtype=torch.float32)

            point_scale = 1.0
            if lambda_point > 0.0 or lambda_point_prior > 0.0:
                point_warm = max(0, int(args.point_warmup_steps))
                point_scale = 1.0 if point_warm <= 0 else min(1.0, float(step + 1) / float(max(1, point_warm)))
                last_scale_info["point_scale"] = float(point_scale)

            if lambda_point > 0.0:
                point_abs = _robust_abs(point_pred - point_tgt_for_loss, float(args.robust_l1_eps)).mean(dim=-1)
                if point_reliability is not None:
                    w_point = (w * point_reliability).detach()
                else:
                    w_point = w
                point_support_for_loss = _map_support_weight(
                    support01=point_mv_support.detach() if point_mv_support is not None else None,
                    mode=str(args.point_support_mode),
                    floor=float(args.point_support_floor),
                )
                if point_support_for_loss is not None:
                    last_scale_info["point_support_path_active"] = 1.0
                    w_point = (w_point * point_support_for_loss).detach()
                    last_scale_info["point_support_mode"] = str(args.point_support_mode)
                    last_scale_info["point_support_floor"] = float(args.point_support_floor)
                    last_scale_info["point_support_eff_mean"] = float(point_support_for_loss.mean().item())
                    mv_active = (valid > 0.5)
                    if int(mv_active.sum().item()) > 0:
                        last_scale_info["point_support_eff_active_ratio"] = float(point_support_for_loss[mv_active].mean().item())
                    _append_region_distribution_stats(
                        info=last_scale_info,
                        prefix="point_support_eff",
                        value01=point_support_for_loss,
                        fg_mask01=fg_mask_used,
                        active_mask01=valid,
                    )
                point_fg_erode = int(max(0, int(args.point_loss_fg_erode_px)))
                if fg_mask_used is not None and point_fg_erode > 0:
                    fg_point = _erode_mask_tensor(fg_mask_used.detach(), point_fg_erode)
                    w_point = (w_point * fg_point).detach()
                    last_scale_info["point_loss_fg_erode_px"] = float(point_fg_erode)
                    last_scale_info["point_loss_fg_cover_eroded"] = float(fg_point.mean().item())
                q_res = float(min(1.0, max(0.0, args.point_residual_quantile)))
                res_focus = str(args.point_residual_focus).strip().lower()
                if res_focus not in {"inlier", "outlier", "all"}:
                    res_focus = "inlier"
                last_scale_info["point_res_focus"] = res_focus
                valid_point = (w_point > 0.0)
                if q_res < 1.0 and res_focus != "all" and int(valid_point.sum().item()) >= 16:
                    point_abs_det = point_abs.detach()
                    if res_focus == "outlier":
                        qv_res = torch.quantile(point_abs_det[valid_point], max(0.0, 1.0 - q_res))
                        keep_res = (point_abs_det >= qv_res).float()
                    else:
                        qv_res = torch.quantile(point_abs_det[valid_point], q_res)
                        keep_res = (point_abs_det <= qv_res).float()
                    w_point = w_point * keep_res
                    last_scale_info["point_res_q"] = float(q_res)
                    last_scale_info["point_res_qv"] = float(qv_res.item())
                    last_scale_info["point_res_keep_ratio"] = float(keep_res[valid_point].mean().item())
                boost_gain = float(max(0.0, float(args.point_residual_boost)))
                if boost_gain > 0.0:
                    valid_point = (w_point > 0.0)
                    if int(valid_point.sum().item()) >= 16:
                        point_abs_det = point_abs.detach()
                        qv_boost = torch.quantile(point_abs_det[valid_point], 0.75)
                        qv_boost_f = float(qv_boost.item())
                        if qv_boost_f > 1e-8:
                            boost_cap = float(max(1.0, float(args.point_residual_boost_cap)))
                            rel = (point_abs_det / qv_boost).clamp(min=0.0)
                            boost_map = 1.0 + boost_gain * (rel - 1.0).clamp(min=0.0, max=(boost_cap - 1.0))
                            w_point = w_point * boost_map
                            last_scale_info["point_residual_boost"] = float(boost_gain)
                            last_scale_info["point_residual_boost_cap"] = float(boost_cap)
                            last_scale_info["point_residual_boost_q75"] = float(qv_boost_f)
                            last_scale_info["point_residual_boost_mean"] = float(boost_map[valid_point].mean().item())
                if human_prior_complete_mask is not None and human_prior_complete_weight > 0.0:
                    completion_weight_map = (human_prior_complete_mask * human_prior_complete_weight).detach()
                    w_point = torch.maximum(w_point, completion_weight_map)
                    last_scale_info["human_prior_complete_weight_mean"] = float(completion_weight_map.mean().item())
                    last_scale_info["human_prior_complete_weight_nonzero"] = float((completion_weight_map > 0.0).float().mean().item())
                denom_point = w_point.sum() + 1e-6
                loss_point = (point_abs * w_point).sum() / denom_point
                loss_point = loss_point * float(max(0.0, point_loss_scale)) * float(point_scale)
            else:
                loss_point = torch.zeros([], device=device, dtype=torch.float32)

            last_scale_info["human_prior_loss_active"] = 0.0
            if (
                lambda_point_prior > 0.0
                and human_prior_point_world is not None
                and human_prior_loss_mask is not None
            ):
                human_prior_finite_mask = torch.isfinite(human_prior_point_world).all(dim=-1).float()
                w_point_prior = (w * human_prior_loss_mask * human_prior_finite_mask).detach()
                if human_prior_complete_mask is not None and human_prior_complete_weight > 0.0:
                    completion_weight_map = (human_prior_complete_mask * human_prior_complete_weight).detach()
                    w_point_prior = torch.maximum(w_point_prior, completion_weight_map)
                denom_point_prior = w_point_prior.sum() + 1e-6
                point_prior_abs = _robust_abs(
                    point_pred - human_prior_point_world,
                    float(args.robust_l1_eps),
                ).mean(dim=-1)
                loss_point_prior = (point_prior_abs * w_point_prior).sum() / denom_point_prior
                loss_point_prior = loss_point_prior * float(point_scale)
                last_scale_info["human_prior_loss_active"] = 1.0
                last_scale_info["human_prior_loss_weight_mean"] = float(w_point_prior.mean().item())
                last_scale_info["human_prior_loss_weight_nonzero"] = float((w_point_prior > 0.0).float().mean().item())
            else:
                loss_point_prior = torch.zeros([], device=device, dtype=torch.float32)

            if (
                lambda_point_reproj > 0.0
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                loss_point_reproj_raw, reproj_info = _point_self_reprojection_loss(
                    point_world=point_pred,
                    extrinsic_w2c=extrinsic_tgt.detach(),
                    intrinsic=intrinsic_tgt.detach(),
                    valid_mask=valid.detach(),
                    robust_eps=float(args.robust_l1_eps),
                    clamp_px=float(args.point_reproj_clamp_px),
                )
                warm_pr = max(0, int(args.point_reproj_warmup_steps))
                pr_scale = 1.0 if warm_pr <= 0 else min(1.0, float(step + 1) / float(max(1, warm_pr)))
                loss_point_reproj = loss_point_reproj_raw * float(pr_scale)
                last_scale_info["point_reproj_scale"] = float(pr_scale)
                last_scale_info.update(reproj_info)
            else:
                loss_point_reproj = torch.zeros([], device=device, dtype=torch.float32)

            if (
                lambda_point_normal_consis > 0.0
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                point_from_depth_pred_norm = _depth_to_world_point_map_torch(
                    depth_pred_for_loss, extrinsic_tgt, intrinsic_tgt
                )
                point_from_depth_tgt_norm = _depth_to_world_point_map_torch(
                    depth_tgt.detach(), extrinsic_tgt.detach(), intrinsic_tgt.detach()
                )
                loss_point_normal_consis_raw, normal_info = _point_normal_consistency_loss(
                    point_world_pred=point_from_depth_pred_norm,
                    point_world_tgt=point_from_depth_tgt_norm,
                    valid_mask=valid.detach(),
                    support_weight=w_base.detach(),
                )
                warm_pn = max(0, int(args.point_normal_consis_warmup_steps))
                pn_scale = 1.0 if warm_pn <= 0 else min(1.0, float(step + 1) / float(max(1, warm_pn)))
                loss_point_normal_consis = loss_point_normal_consis_raw * float(pn_scale)
                last_scale_info["point_normal_consis_scale"] = float(pn_scale)
                last_scale_info.update(normal_info)
            else:
                loss_point_normal_consis = torch.zeros([], device=device, dtype=torch.float32)

            if (
                lambda_point_mv_depth > 0.0
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                valid_for_mv = valid.detach()
                mv_fg_erode = int(max(0, int(args.point_mv_depth_fg_erode_px)))
                if mv_fg_erode > 0:
                    valid_for_mv = _erode_mask_tensor(valid_for_mv, mv_fg_erode)
                    last_scale_info["point_mv_depth_fg_erode_px"] = float(mv_fg_erode)
                    last_scale_info["point_mv_depth_valid_cover_eroded"] = float(valid_for_mv.mean().item())
                mv_depth_region_mode = str(args.point_mv_depth_region_mode).strip().lower()
                if mv_depth_region_mode not in {"all", "bg_only"}:
                    mv_depth_region_mode = "all"
                last_scale_info["point_mv_depth_region_mode"] = str(mv_depth_region_mode)
                if mv_depth_region_mode == "bg_only" and fg_mask_used is not None:
                    bg_gate = (1.0 - fg_mask_used.detach()).clamp(0.0, 1.0)
                    valid_for_mv = valid_for_mv * bg_gate
                    last_scale_info["point_mv_depth_bg_gate_mean"] = float(bg_gate.mean().item())
                    last_scale_info["point_mv_depth_valid_cover_bg_only"] = float(valid_for_mv.mean().item())

                point_mv_support_for_mv = _map_support_weight(
                    support01=point_mv_support.detach() if point_mv_support is not None else None,
                    mode=str(args.point_mv_depth_support_mode),
                    floor=float(args.point_mv_depth_support_floor),
                )
                if dyn_proxy_weight is not None:
                    if point_mv_support_for_mv is not None:
                        point_mv_support_for_mv = (point_mv_support_for_mv * dyn_proxy_weight).clamp(0.0, 1.0)
                    else:
                        point_mv_support_for_mv = dyn_proxy_weight
                elif gram_dyn_weight is not None:
                    if point_mv_support_for_mv is not None:
                        point_mv_support_for_mv = (point_mv_support_for_mv * gram_dyn_weight.detach()).clamp(0.0, 1.0)
                    else:
                        point_mv_support_for_mv = gram_dyn_weight.detach().clamp(0.0, 1.0)
                if point_mv_support_for_mv is not None:
                    if str(args.point_mv_depth_support_mode).strip().lower() not in {"", "off", "none"}:
                        last_scale_info["point_mv_depth_support_path_active"] = 1.0
                    last_scale_info["point_mv_depth_support_mode"] = str(args.point_mv_depth_support_mode)
                    last_scale_info["point_mv_depth_support_floor"] = float(args.point_mv_depth_support_floor)
                    last_scale_info["point_mv_depth_support_eff_mean"] = float(point_mv_support_for_mv.mean().item())
                    mv_active = (valid_for_mv > 0.5)
                    if int(mv_active.sum().item()) > 0:
                        last_scale_info["point_mv_depth_support_eff_active_ratio"] = float(point_mv_support_for_mv[mv_active].mean().item())
                    _append_region_distribution_stats(
                        info=last_scale_info,
                        prefix="point_mv_depth_support_eff",
                        value01=point_mv_support_for_mv,
                        fg_mask01=fg_mask_used,
                        active_mask01=valid_for_mv,
                    )

                loss_point_mv_depth_raw, mv_depth_info = _point_multiview_depth_reproj_loss(
                    point_world=point_pred,
                    depth_tgt=depth_tgt.detach(),
                    extrinsic_w2c=extrinsic_tgt.detach(),
                    intrinsic=intrinsic_tgt.detach(),
                    valid_mask=valid_for_mv,
                    support_weight=point_mv_support_for_mv.detach() if point_mv_support_for_mv is not None else None,
                    robust_eps=float(args.robust_l1_eps),
                    tol_abs=float(args.point_mv_tol_abs),
                    tol_rel=float(args.point_mv_tol_rel),
                    weight_floor=float(args.point_mv_weight_floor),
                    stride=int(args.point_mv_stride),
                    max_pairs=int(args.point_mv_depth_max_pairs),
                    pair_mode=str(args.point_mv_depth_pair_mode),
                    inlier_only=(str(args.point_mv_depth_inlier_only).lower() == "on"),
                    err_quantile=float(args.point_mv_depth_err_quantile),
                    outlier_boost=float(args.point_mv_depth_outlier_boost),
                    outlier_cap=float(args.point_mv_depth_outlier_cap),
                    tgt_valid_mode=str(args.point_mv_depth_tgt_valid_mode),
                    tgt_valid_floor=float(args.point_mv_depth_tgt_valid_floor),
                    tgt_valid_min_ratio=float(args.point_mv_depth_min_tgt_valid_ratio),
                )
                warm_mv = max(0, int(args.point_mv_depth_warmup_steps))
                mv_scale_warm = 1.0 if warm_mv <= 0 else min(1.0, float(step + 1) / float(max(1, warm_mv)))
                mv_scale_adapt = 1.0
                adapt_mode = str(args.point_mv_depth_adapt_mode).lower()
                if adapt_mode == "on":
                    adapt_mode = "valid_ratio"
                if adapt_mode == "valid_ratio":
                    valid_ratio_mv = float(mv_depth_info.get("point_mv_depth_valid_ratio", 0.0))
                    target_valid_mv = max(1e-8, float(args.point_mv_depth_adapt_target_valid))
                    min_scale_mv = max(0.0, float(args.point_mv_depth_adapt_min_scale))
                    max_scale_mv = max(min_scale_mv, float(args.point_mv_depth_adapt_max_scale))
                    raw_scale_mv = target_valid_mv / max(valid_ratio_mv, 1e-8)
                    mv_scale_adapt = float(np.clip(raw_scale_mv, min_scale_mv, max_scale_mv))
                    last_scale_info["point_mv_depth_adapt_target_valid"] = float(target_valid_mv)
                    last_scale_info["point_mv_depth_adapt_min_scale"] = float(min_scale_mv)
                    last_scale_info["point_mv_depth_adapt_max_scale"] = float(max_scale_mv)
                    last_scale_info["point_mv_depth_adapt_raw_scale"] = float(raw_scale_mv)
                last_scale_info["point_mv_depth_adapt_on"] = 1.0 if adapt_mode == "valid_ratio" else 0.0
                tgt_mode_flag = str(args.point_mv_depth_tgt_valid_mode).lower()
                last_scale_info["point_mv_depth_tgt_valid_hard"] = 1.0 if tgt_mode_flag == "hard" else 0.0
                last_scale_info["point_mv_depth_tgt_valid_soft"] = 1.0 if tgt_mode_flag == "soft" else 0.0
                last_scale_info["point_mv_depth_tgt_valid_soft_zero"] = 1.0 if tgt_mode_flag == "soft_zero" else 0.0
                last_scale_info["point_mv_depth_tgt_valid_floor"] = float(args.point_mv_depth_tgt_valid_floor)
                mv_scale = float(mv_scale_warm) * float(mv_scale_adapt)
                tgt_scale_mode = str(args.point_mv_depth_tgt_valid_scale_mode).strip().lower()
                tgt_scale_thr = float(max(1e-8, float(args.point_mv_depth_tgt_valid_scale_thr)))
                tgt_valid_ratio_mv = float(mv_depth_info.get("point_mv_depth_tgt_valid_ratio", 0.0))
                mv_scale_tgt_valid = 1.0
                if tgt_scale_mode == "linear":
                    mv_scale_tgt_valid = float(np.clip(tgt_valid_ratio_mv / tgt_scale_thr, 0.0, 1.0))
                else:
                    tgt_scale_mode = "off"
                loss_point_mv_depth = (
                    loss_point_mv_depth_raw
                    * float(mv_scale)
                    * float(mv_scale_tgt_valid)
                )
                last_scale_info["point_mv_depth_scale_warm"] = float(mv_scale_warm)
                last_scale_info["point_mv_depth_scale_adapt"] = float(mv_scale_adapt)
                last_scale_info["point_mv_depth_scale"] = float(mv_scale)
                last_scale_info["point_mv_depth_tgt_valid_scale_mode"] = str(tgt_scale_mode)
                last_scale_info["point_mv_depth_tgt_valid_scale_thr"] = float(tgt_scale_thr)
                last_scale_info["point_mv_depth_tgt_valid_scale"] = float(mv_scale_tgt_valid)
                last_scale_info.update(mv_depth_info)
            else:
                loss_point_mv_depth = torch.zeros([], device=device, dtype=torch.float32)

            if (
                lambda_point_mv_mask > 0.0
                and fg_mask_used is not None
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                valid_for_mv_mask = valid.detach()
                fg_for_mv_mask = fg_mask_used.detach()
                mv_fg_erode = int(max(0, int(args.point_mv_depth_fg_erode_px)))
                if mv_fg_erode > 0:
                    valid_for_mv_mask = _erode_mask_tensor(valid_for_mv_mask, mv_fg_erode)
                    fg_for_mv_mask = _erode_mask_tensor(fg_for_mv_mask, mv_fg_erode)
                    last_scale_info["point_mv_mask_fg_erode_px"] = float(mv_fg_erode)
                    last_scale_info["point_mv_mask_valid_cover_eroded"] = float(valid_for_mv_mask.mean().item())
                    last_scale_info["point_mv_mask_fg_cover_eroded"] = float(fg_for_mv_mask.mean().item())

                point_mv_support_for_mask = _map_support_weight(
                    support01=point_mv_support.detach() if point_mv_support is not None else None,
                    mode=str(args.point_mv_mask_support_mode),
                    floor=float(args.point_mv_mask_support_floor),
                )
                if dyn_proxy_weight is not None:
                    if point_mv_support_for_mask is not None:
                        point_mv_support_for_mask = (point_mv_support_for_mask * dyn_proxy_weight).clamp(0.0, 1.0)
                    else:
                        point_mv_support_for_mask = dyn_proxy_weight
                elif gram_dyn_weight is not None:
                    if point_mv_support_for_mask is not None:
                        point_mv_support_for_mask = (point_mv_support_for_mask * gram_dyn_weight.detach()).clamp(0.0, 1.0)
                    else:
                        point_mv_support_for_mask = gram_dyn_weight.detach().clamp(0.0, 1.0)
                if point_mv_support_for_mask is not None:
                    if str(args.point_mv_mask_support_mode).strip().lower() not in {"", "off", "none"}:
                        last_scale_info["point_mv_mask_support_path_active"] = 1.0
                    last_scale_info["point_mv_mask_support_mode"] = str(args.point_mv_mask_support_mode)
                    last_scale_info["point_mv_mask_support_floor"] = float(args.point_mv_mask_support_floor)
                    last_scale_info["point_mv_mask_support_eff_mean"] = float(point_mv_support_for_mask.mean().item())
                    _append_region_distribution_stats(
                        info=last_scale_info,
                        prefix="point_mv_mask_support_eff",
                        value01=point_mv_support_for_mask,
                        fg_mask01=fg_mask_used,
                        active_mask01=valid_for_mv_mask,
                    )

                loss_point_mv_mask_raw, mv_mask_info = _point_multiview_fg_reproj_loss(
                    point_world=point_pred,
                    fg_mask_tgt=fg_for_mv_mask,
                    extrinsic_w2c=extrinsic_tgt.detach(),
                    intrinsic=intrinsic_tgt.detach(),
                    src_valid_mask=valid_for_mv_mask,
                    support_weight=point_mv_support_for_mask.detach() if point_mv_support_for_mask is not None else None,
                    robust_eps=float(args.robust_l1_eps),
                    stride=int(args.point_mv_stride),
                    max_pairs=int(args.point_mv_depth_max_pairs),
                    pair_mode=str(args.point_mv_depth_pair_mode),
                    min_tgt_fg_ratio=float(args.point_mv_mask_min_tgt_fg_ratio),
                    hit_thr=float(args.point_mv_mask_hit_thr),
                    soft_blur_px=int(args.point_mv_mask_soft_blur_px),
                    soft_blur_iters=int(args.point_mv_mask_soft_blur_iters),
                    soft_mix=float(args.point_mv_mask_soft_mix),
                    soft_hit_thr=float(args.point_mv_mask_soft_hit_thr),
                )
                # If configured threshold is too strict and no pair survives, auto-relax once
                # to keep mv-mask supervision active instead of silently becoming no-op.
                if (
                    float(mv_mask_info.get("point_mv_mask_pairs", 0.0)) <= 0.0
                    and float(args.point_mv_mask_min_tgt_fg_ratio) > 0.0
                ):
                    loss_point_mv_mask_raw, mv_mask_info = _point_multiview_fg_reproj_loss(
                        point_world=point_pred,
                        fg_mask_tgt=fg_for_mv_mask,
                        extrinsic_w2c=extrinsic_tgt.detach(),
                        intrinsic=intrinsic_tgt.detach(),
                        src_valid_mask=valid_for_mv_mask,
                        support_weight=point_mv_support_for_mask.detach() if point_mv_support_for_mask is not None else None,
                        robust_eps=float(args.robust_l1_eps),
                        stride=int(args.point_mv_stride),
                        max_pairs=int(args.point_mv_depth_max_pairs),
                        pair_mode=str(args.point_mv_depth_pair_mode),
                        min_tgt_fg_ratio=0.0,
                        hit_thr=float(args.point_mv_mask_hit_thr),
                        soft_blur_px=int(args.point_mv_mask_soft_blur_px),
                        soft_blur_iters=int(args.point_mv_mask_soft_blur_iters),
                        soft_mix=float(args.point_mv_mask_soft_mix),
                        soft_hit_thr=float(args.point_mv_mask_soft_hit_thr),
                    )
                    mv_mask_info["point_mv_mask_min_tgt_fg_ratio_relaxed"] = 1.0
                else:
                    mv_mask_info["point_mv_mask_min_tgt_fg_ratio_relaxed"] = 0.0
                warm_mm = max(0, int(args.point_mv_mask_warmup_steps))
                mm_scale = 1.0 if warm_mm <= 0 else min(1.0, float(step + 1) / float(max(1, warm_mm)))
                loss_point_mv_mask = loss_point_mv_mask_raw * float(mm_scale)
                last_scale_info["point_mv_mask_scale"] = float(mm_scale)
                last_scale_info["point_mv_mask_min_tgt_fg_ratio"] = float(args.point_mv_mask_min_tgt_fg_ratio)
                last_scale_info["point_mv_mask_hit_thr"] = float(args.point_mv_mask_hit_thr)
                last_scale_info["point_mv_mask_soft_blur_px"] = float(args.point_mv_mask_soft_blur_px)
                last_scale_info["point_mv_mask_soft_blur_iters"] = float(args.point_mv_mask_soft_blur_iters)
                last_scale_info["point_mv_mask_soft_mix"] = float(args.point_mv_mask_soft_mix)
                last_scale_info["point_mv_mask_soft_hit_thr"] = float(args.point_mv_mask_soft_hit_thr)
                last_scale_info.update(mv_mask_info)
            else:
                loss_point_mv_mask = torch.zeros([], device=device, dtype=torch.float32)

            loss_fg_structure_depth_edge = torch.zeros([], device=device, dtype=torch.float32)
            loss_point_mv_outside_ring = torch.zeros([], device=device, dtype=torch.float32)
            if h_family_enabled:
                if lambda_fg_structure_depth_edge > 0.0 and fg_mask_used is not None:
                    fg_bbox_mask = _build_fg_bbox_mask(
                        fg_mask01=fg_mask_used.detach(),
                        margin_px=int(args.fg_structure_bbox_margin_px),
                        min_side_px=int(args.fg_structure_bbox_min_side_px),
                    )
                    fg_structure_region_mask = _build_fg_structure_region_mask(
                        fg_mask01=fg_mask_used.detach(),
                        fg_bbox_mask01=fg_bbox_mask,
                        region_mode=str(args.fg_structure_region_mode),
                        region_erode_px=int(args.fg_structure_region_erode_px),
                    )
                    fg_boundary_probe_mask = _build_fg_boundary_band_mask(
                        fg_mask01=fg_mask_used.detach(),
                        erode_px=int(max(1, int(args.fg_structure_boundary_probe_px))),
                    )
                    loss_fg_structure_depth_edge, fg_structure_info = _fg_structure_depth_edge_loss(
                        depth_pred=depth_pred_for_loss,
                        depth_tgt=depth_tgt.detach(),
                        valid01=valid.detach(),
                        fg_bbox_mask01=fg_bbox_mask,
                        fg_structure_region_mask01=fg_structure_region_mask,
                        boundary_probe_mask01=fg_boundary_probe_mask,
                        edge_support_mode=str(args.fg_structure_edge_support_mode),
                        edge_support_quantile=float(args.fg_structure_edge_support_quantile),
                        min_edge_support_px=int(args.fg_structure_edge_support_min_px),
                        edge_weight_mode=str(args.fg_structure_edge_weight_mode),
                        boundary_falloff_px=int(args.fg_structure_boundary_falloff_px),
                        component_bias_mode=str(args.fg_structure_component_bias_mode),
                        component_bias_threshold_ratio=float(args.fg_structure_component_bias_threshold_ratio),
                        component_bias_other_scale=float(args.fg_structure_component_bias_other_scale),
                        front_depth_bias_mode=str(args.fg_structure_front_depth_bias_mode),
                        front_depth_bias_tau=float(args.fg_structure_front_depth_bias_tau),
                        front_depth_bias_center_quantile=float(args.fg_structure_front_depth_bias_center_quantile),
                        min_active_px=64,
                        min_boundary_px=32,
                    )
                    last_scale_info.update(fg_structure_info)

                if (
                    lambda_point_mv_outside_ring > 0.0
                    and fg_mask_used is not None
                    and extrinsic_tgt is not None
                    and intrinsic_tgt is not None
                ):
                    outside_ring_mask = _build_fg_outside_ring_mask(
                        fg_mask01=fg_mask_used.detach(),
                        ring_px=int(args.point_mv_outside_ring_px),
                    )
                    point_mv_support_for_ring = _map_support_weight(
                        support01=point_mv_support.detach() if point_mv_support is not None else None,
                        mode=str(args.point_mv_mask_support_mode),
                        floor=float(args.point_mv_mask_support_floor),
                    )
                    if dyn_proxy_weight is not None:
                        if point_mv_support_for_ring is not None:
                            point_mv_support_for_ring = (point_mv_support_for_ring * dyn_proxy_weight).clamp(0.0, 1.0)
                        else:
                            point_mv_support_for_ring = dyn_proxy_weight
                    elif gram_dyn_weight is not None:
                        if point_mv_support_for_ring is not None:
                            point_mv_support_for_ring = (point_mv_support_for_ring * gram_dyn_weight.detach()).clamp(0.0, 1.0)
                        else:
                            point_mv_support_for_ring = gram_dyn_weight.detach().clamp(0.0, 1.0)
                    loss_point_mv_outside_ring, outside_ring_info = _point_mv_outside_ring_loss(
                        point_world=point_pred,
                        outside_ring_mask_tgt=outside_ring_mask,
                        extrinsic_w2c=extrinsic_tgt.detach(),
                        intrinsic=intrinsic_tgt.detach(),
                        src_valid_mask=valid.detach(),
                        support_weight=point_mv_support_for_ring.detach() if point_mv_support_for_ring is not None else None,
                        robust_eps=float(args.robust_l1_eps),
                        stride=int(args.point_mv_stride),
                        min_active_ring_px=32,
                    )
                    last_scale_info.update(outside_ring_info)

            if lambda_conf > 0.0:
                conf_abs = _robust_abs(conf_pred - conf_tgt, float(args.robust_l1_eps))
                loss_conf = (conf_abs * valid).sum() / (valid.sum() + 1e-6)
            else:
                loss_conf = torch.zeros([], device=device, dtype=torch.float32)

            lambda_fg_conf_presence = float(max(0.0, float(args.lambda_fg_conf_presence)))
            if lambda_fg_conf_presence > 0.0 and fg_mask_used is not None:
                loss_fg_conf_presence, fg_conf_presence_info = _fg_conf_presence_floor_loss(
                    pred_conf01=conf_pred,
                    tgt_conf01=conf_tgt,
                    fg_mask01=fg_mask_used.detach(),
                    valid01=valid_all.detach(),
                    target_ratio=float(args.fg_conf_presence_target_ratio),
                )
                last_scale_info.update(fg_conf_presence_info)
            else:
                loss_fg_conf_presence = torch.zeros([], device=device, dtype=torch.float32)
                last_scale_info["fg_conf_presence_enabled"] = 0.0
                last_scale_info["fg_conf_presence_target_ratio"] = float(args.fg_conf_presence_target_ratio)
                last_scale_info["fg_conf_presence_loss"] = 0.0

            cam_scale = 0.0
            if (
                lambda_cam > 0.0
                and pose_pred is not None
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                # Infer image size from principal point in pseudo intrinsics for FoV encoding.
                h_cam = int(max(2, round(float((intrinsic_tgt[..., 1, 2].detach().mean() * 2.0).item()))))
                w_cam = int(max(2, round(float((intrinsic_tgt[..., 0, 2].detach().mean() * 2.0).item()))))
                pose_tgt = extri_intri_to_pose_encoding(
                    extrinsic_tgt, intrinsic_tgt, image_size_hw=(h_cam, w_cam)
                ).detach()
                loss_cam_raw, cam_info = _camera_pose_losses(
                    pred_pose=pose_pred,
                    tgt_pose=pose_tgt,
                    robust_eps=float(args.robust_l1_eps),
                    rot_weight=float(args.cam_rot_weight),
                    fov_weight=float(args.cam_fov_weight),
                )
                warm = max(1, int(args.cam_warmup_steps))
                cam_scale = min(1.0, float(step + 1) / float(warm))
                loss_cam = loss_cam_raw * cam_scale
                last_scale_info.update(cam_info)
            else:
                loss_cam = torch.zeros([], device=device, dtype=torch.float32)

            if (
                lambda_geom_cons > 0.0
                and extrinsic_tgt is not None
                and intrinsic_tgt is not None
            ):
                point_from_depth_pred = _depth_to_world_point_map_torch(
                    depth_pred_for_loss, extrinsic_tgt, intrinsic_tgt
                )
                geom_abs = _robust_abs(point_from_depth_pred - point_pred, float(args.robust_l1_eps)).mean(dim=-1)
                denom_geom = w_base.sum() + 1e-6
                loss_geom_cons = (geom_abs * w_base).sum() / denom_geom
            else:
                loss_geom_cons = torch.zeros([], device=device, dtype=torch.float32)

            conf_warm = max(0, int(args.lambda_conf_warmup_steps))
            conf_lambda_scale = 1.0 if conf_warm <= 0 else min(1.0, float(step + 1) / float(max(1, conf_warm)))
            lambda_conf_eff = float(lambda_conf) * float(conf_lambda_scale)
            last_scale_info["lambda_conf_scale"] = float(conf_lambda_scale)
            last_scale_info["lambda_conf_eff"] = float(lambda_conf_eff)

            structure_warm = max(0, int(args.fg_structure_depth_edge_warmup_steps))
            structure_lambda_scale = (
                1.0 if structure_warm <= 0 else min(1.0, float(step + 1) / float(max(1, structure_warm)))
            )
            lambda_fg_structure_depth_edge_eff = float(lambda_fg_structure_depth_edge) * float(structure_lambda_scale)
            last_scale_info["lambda_fg_structure_depth_edge_scale"] = float(structure_lambda_scale)
            last_scale_info["lambda_fg_structure_depth_edge_eff"] = float(lambda_fg_structure_depth_edge_eff)

            cam_lambda_warm = max(0, int(args.lambda_cam_warmup_steps))
            cam_lambda_scale = 1.0 if cam_lambda_warm <= 0 else min(1.0, float(step + 1) / float(max(1, cam_lambda_warm)))
            lambda_cam_eff = float(lambda_cam) * float(cam_lambda_scale)
            last_scale_info["lambda_cam_scale"] = float(cam_lambda_scale)
            last_scale_info["lambda_cam_eff"] = float(lambda_cam_eff)

            contrib_depth = float(lambda_depth) * loss_depth
            contrib_point = float(lambda_point) * loss_point
            contrib_point_prior = float(lambda_point_prior) * loss_point_prior
            contrib_point_reproj = float(lambda_point_reproj) * loss_point_reproj
            contrib_point_normal_consis = float(lambda_point_normal_consis) * loss_point_normal_consis
            contrib_point_mv_depth = float(lambda_point_mv_depth) * loss_point_mv_depth
            contrib_point_mv_mask = float(lambda_point_mv_mask) * loss_point_mv_mask
            contrib_fg_structure_depth_edge = float(lambda_fg_structure_depth_edge_eff) * loss_fg_structure_depth_edge
            contrib_point_mv_outside_ring = float(lambda_point_mv_outside_ring) * loss_point_mv_outside_ring
            contrib_conf = float(lambda_conf_eff) * loss_conf
            contrib_fg_conf_presence = float(lambda_fg_conf_presence) * loss_fg_conf_presence
            contrib_geom_cons = float(lambda_geom_cons) * loss_geom_cons
            contrib_cam = float(lambda_cam_eff) * loss_cam

            loss = (
                contrib_depth
                + contrib_point
                + contrib_point_prior
                + contrib_point_reproj
                + contrib_point_normal_consis
                + contrib_point_mv_depth
                + contrib_point_mv_mask
                + contrib_fg_structure_depth_edge
                + contrib_point_mv_outside_ring
                + contrib_conf
                + contrib_fg_conf_presence
                + contrib_geom_cons
                + contrib_cam
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            grad_sq = 0.0
            step_sq = 0.0
            param_sq = 0.0
            for group in optimizer.param_groups:
                lr_group = float(group.get("lr", base_lr))
                lr_sq = lr_group * lr_group
                for p in group.get("params", []):
                    if p is None or p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    g2 = float(torch.sum(g * g).item())
                    grad_sq += g2
                    step_sq += g2 * lr_sq
                    pd = p.detach().float()
                    param_sq += float(torch.sum(pd * pd).item())
            grad_norm_val = float(np.sqrt(max(0.0, grad_sq)))
            step_update_norm_val = float(np.sqrt(max(0.0, step_sq)))
            step_update_ratio_val = float(step_update_norm_val / max(np.sqrt(max(0.0, param_sq)), 1e-12))
            optimizer.step()

            step += 1
            count += 1
            steps_this_epoch += 1
            loss_sum += float(loss.item())
            loss_depth_sum += float(loss_depth.item())
            loss_point_sum += float(loss_point.item())
            loss_point_prior_sum += float(loss_point_prior.item())
            loss_point_reproj_sum += float(loss_point_reproj.item())
            loss_point_normal_consis_sum += float(loss_point_normal_consis.item())
            loss_point_mv_depth_sum += float(loss_point_mv_depth.item())
            loss_point_mv_mask_sum += float(loss_point_mv_mask.item())
            loss_fg_structure_depth_edge_sum += float(loss_fg_structure_depth_edge.item())
            loss_point_mv_outside_ring_sum += float(loss_point_mv_outside_ring.item())
            loss_conf_sum += float(loss_conf.item())
            loss_fg_conf_presence_sum += float(loss_fg_conf_presence.item())
            loss_geom_cons_sum += float(loss_geom_cons.item())
            loss_cam_sum += float(loss_cam.item())
            grad_norm_sum += float(grad_norm_val)
            step_update_norm_sum += float(step_update_norm_val)
            step_update_ratio_sum += float(step_update_ratio_val)
            eval_due = ((step % eval_every_steps) == 0)
            debug_due = (debug_metrics_every_steps > 0) and ((step % debug_metrics_every_steps) == 0)
            vis_due = (
                (debug_vis_every_steps > 0)
                and ((step % debug_vis_every_steps) == 0)
                and ((debug_vis_max_steps <= 0) or (debug_vis_saved < debug_vis_max_steps))
            )

            debug_extra: Dict[str, float] = {}
            if eval_due or debug_due or vis_due:
                with torch.no_grad():
                    denom_valid = valid.sum() + 1e-6
                    if point_from_depth_tgt is not None:
                        pseudo_dp = torch.linalg.norm((point_from_depth_tgt.detach() - point_tgt.detach()), dim=-1)
                        debug_extra["pseudo_point_depth_l2"] = float(((pseudo_dp * valid).sum() / denom_valid).item())
                    if extrinsic_tgt is not None and intrinsic_tgt is not None:
                        point_from_depth_pred_dbg = _depth_to_world_point_map_torch(
                            depth_pred_for_loss.detach(), extrinsic_tgt.detach(), intrinsic_tgt.detach()
                        )
                        pred_dp = torch.linalg.norm((point_from_depth_pred_dbg - point_pred.detach()), dim=-1)
                        debug_extra["pred_point_depth_l2"] = float(((pred_dp * valid).sum() / denom_valid).item())
                    point_shift = torch.linalg.norm((point_tgt_for_loss.detach() - point_tgt.detach()), dim=-1)
                    debug_extra["point_target_shift_l2"] = float(((point_shift * valid).sum() / denom_valid).item())
                    # Keep a frame-conversion diagnostic, but use point_pred_shift_l2 as
                    # supervision residual for easier trend reading during short runs.
                    point_pred_frame_shift = torch.linalg.norm((point_pred.detach() - point_pred_raw.detach()), dim=-1)
                    debug_extra["point_pred_frame_shift_l2"] = float(((point_pred_frame_shift * valid).sum() / denom_valid).item())
                    point_pred_shift = torch.linalg.norm((point_pred.detach() - point_tgt_for_loss.detach()), dim=-1)
                    debug_extra["point_pred_shift_l2"] = float(((point_pred_shift * valid).sum() / denom_valid).item())
                    point_tgt_frame_shift = torch.linalg.norm((point_tgt.detach() - point_tgt_input_raw.detach()), dim=-1)
                    debug_extra["point_tgt_frame_shift_l2"] = float(((point_tgt_frame_shift * valid).sum() / denom_valid).item())
                    # Direct supervision diagnostics. These remain informative even when
                    # frame-alignment shifts are near zero (e.g. world->world no-op).
                    point_pred_to_target = torch.linalg.norm((point_pred.detach() - point_tgt_for_loss.detach()), dim=-1)
                    debug_extra["point_pred_to_target_l2"] = float(((point_pred_to_target * valid).sum() / denom_valid).item())
                    point_pred_to_pointmap = torch.linalg.norm((point_pred.detach() - point_tgt.detach()), dim=-1)
                    debug_extra["point_pred_to_pointmap_l2"] = float(((point_pred_to_pointmap * valid).sum() / denom_valid).item())
                    if point_from_depth_tgt is not None:
                        point_pred_to_depthunproj = torch.linalg.norm((point_pred.detach() - point_from_depth_tgt.detach()), dim=-1)
                        debug_extra["point_pred_to_depthunproj_l2"] = float(((point_pred_to_depthunproj * valid).sum() / denom_valid).item())
                    debug_extra["valid_cover"] = float(valid.mean().item())
                    debug_extra["grad_norm"] = float(grad_norm_val)
                    debug_extra["step_update_norm"] = float(step_update_norm_val)
                    debug_extra["step_update_ratio"] = float(step_update_ratio_val)

                    if fg_mask_used is not None and extrinsic_tgt is not None and intrinsic_tgt is not None:
                        # GT-chain diagnostic: if GT point projection itself misses FG in other views,
                        # ghost mitigation should prioritize data/camera chain instead of LR tuning.
                        _, mv_mask_gt_loss_info = _point_multiview_fg_reproj_loss(
                            point_world=point_tgt_for_loss.detach(),
                            fg_mask_tgt=fg_mask_used.detach(),
                            extrinsic_w2c=extrinsic_tgt.detach(),
                            intrinsic=intrinsic_tgt.detach(),
                            src_valid_mask=valid.detach(),
                            support_weight=None,
                            robust_eps=float(args.robust_l1_eps),
                            stride=int(args.point_mv_stride),
                            max_pairs=1,
                            pair_mode="adjacent",
                            min_tgt_fg_ratio=0.0,
                            hit_thr=float(args.point_mv_mask_hit_thr),
                            soft_blur_px=int(args.point_mv_mask_soft_blur_px),
                            soft_blur_iters=int(args.point_mv_mask_soft_blur_iters),
                            soft_mix=float(args.point_mv_mask_soft_mix),
                            soft_hit_thr=float(args.point_mv_mask_soft_hit_thr),
                        )
                        debug_extra["point_mv_mask_gt_pairs"] = float(mv_mask_gt_loss_info.get("point_mv_mask_pairs", 0.0))
                        debug_extra["point_mv_mask_gt_tgt_fg_ratio"] = float(mv_mask_gt_loss_info.get("point_mv_mask_tgt_fg_ratio", 0.0))
                        debug_extra["point_mv_mask_gt_miss_ratio"] = float(mv_mask_gt_loss_info.get("point_mv_mask_miss_ratio", 0.0))

                        _, mv_mask_pointmap_info = _point_multiview_fg_reproj_loss(
                            point_world=point_tgt.detach(),
                            fg_mask_tgt=fg_mask_used.detach(),
                            extrinsic_w2c=extrinsic_tgt.detach(),
                            intrinsic=intrinsic_tgt.detach(),
                            src_valid_mask=valid.detach(),
                            support_weight=None,
                            robust_eps=float(args.robust_l1_eps),
                            stride=int(args.point_mv_stride),
                            max_pairs=1,
                            pair_mode="adjacent",
                            min_tgt_fg_ratio=0.0,
                            hit_thr=float(args.point_mv_mask_hit_thr),
                            soft_blur_px=int(args.point_mv_mask_soft_blur_px),
                            soft_blur_iters=int(args.point_mv_mask_soft_blur_iters),
                            soft_mix=float(args.point_mv_mask_soft_mix),
                            soft_hit_thr=float(args.point_mv_mask_soft_hit_thr),
                        )
                        debug_extra["point_mv_mask_pointmap_pairs"] = float(mv_mask_pointmap_info.get("point_mv_mask_pairs", 0.0))
                        debug_extra["point_mv_mask_pointmap_tgt_fg_ratio"] = float(mv_mask_pointmap_info.get("point_mv_mask_tgt_fg_ratio", 0.0))
                        debug_extra["point_mv_mask_pointmap_miss_ratio"] = float(mv_mask_pointmap_info.get("point_mv_mask_miss_ratio", 0.0))

            if eval_due or debug_due:
                msg = {
                    "event": "step_eval" if eval_due else "step_debug",
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_depth": float(loss_depth.item()),
                    "loss_point": float(loss_point.item()),
                    "loss_point_prior": float(loss_point_prior.item()),
                    "loss_point_reproj": float(loss_point_reproj.item()),
                    "loss_point_normal_consis": float(loss_point_normal_consis.item()),
                    "loss_point_mv_depth": float(loss_point_mv_depth.item()),
                    "loss_point_mv_mask": float(loss_point_mv_mask.item()),
                    "loss_fg_structure_depth_edge": float(loss_fg_structure_depth_edge.item()),
                    "loss_point_mv_outside_ring": float(loss_point_mv_outside_ring.item()),
                    "loss_conf": float(loss_conf.item()),
                    "loss_fg_conf_presence": float(loss_fg_conf_presence.item()),
                    "loss_geom_cons": float(loss_geom_cons.item()),
                    "loss_cam": float(loss_cam.item()),
                    "cam_scale": float(cam_scale),
                    "loss_contrib_depth": float(contrib_depth.item()),
                    "loss_contrib_point": float(contrib_point.item()),
                    "loss_contrib_point_prior": float(contrib_point_prior.item()),
                    "loss_contrib_point_reproj": float(contrib_point_reproj.item()),
                    "loss_contrib_point_normal_consis": float(contrib_point_normal_consis.item()),
                    "loss_contrib_point_mv_depth": float(contrib_point_mv_depth.item()),
                    "loss_contrib_point_mv_mask": float(contrib_point_mv_mask.item()),
                    "loss_contrib_fg_structure_depth_edge": float(contrib_fg_structure_depth_edge.item()),
                    "loss_contrib_point_mv_outside_ring": float(contrib_point_mv_outside_ring.item()),
                    "loss_contrib_conf": float(contrib_conf.item()),
                    "loss_contrib_fg_conf_presence": float(contrib_fg_conf_presence.item()),
                    "loss_contrib_geom_cons": float(contrib_geom_cons.item()),
                    "loss_contrib_cam": float(contrib_cam.item()),
                    "weight_mean": float(w.mean().item()),
                    "point_loss_scale": float(point_loss_scale),
                }
                msg.update(debug_extra)
                msg.update(last_scale_info)
                print(
                    "[finetune] " + json.dumps(msg, ensure_ascii=True, sort_keys=True),
                    flush=True,
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                last_heartbeat_wall = time.monotonic()
            elif heartbeat_every_sec > 0.0:
                now_wall = time.monotonic()
                if (now_wall - last_heartbeat_wall) >= heartbeat_every_sec:
                    heartbeat_msg = {
                        "event": "step_heartbeat",
                        "epoch": int(epoch),
                        "step": int(step),
                        "loss": float(loss.item()),
                        "loss_depth": float(loss_depth.item()),
                        "loss_point": float(loss_point.item()),
                        "loss_point_prior": float(loss_point_prior.item()),
                        "loss_point_mv_depth": float(loss_point_mv_depth.item()),
                        "loss_point_mv_mask": float(loss_point_mv_mask.item()),
                        "loss_fg_structure_depth_edge": float(loss_fg_structure_depth_edge.item()),
                        "loss_point_mv_outside_ring": float(loss_point_mv_outside_ring.item()),
                        "loss_fg_conf_presence": float(loss_fg_conf_presence.item()),
                        "steps_this_epoch": int(steps_this_epoch),
                    }
                    print(
                        "[finetune] " + json.dumps(heartbeat_msg, ensure_ascii=True, sort_keys=True),
                        flush=True,
                    )
                    last_heartbeat_wall = now_wall

            if vis_due:
                with torch.no_grad():
                    b0 = 0
                    n_views = min(int(imgs.shape[1]), int(debug_vis_views))
                    for vi in range(n_views):
                        _ = _write_step_debug_vis(
                            out_dir=debug_vis_dir,
                            epoch=int(epoch),
                            step=int(step),
                            view_idx=int(vi),
                            point_target_mode=str(point_target_mode),
                            rgb=imgs[b0, vi].detach(),
                            depth_tgt=depth_tgt[b0, vi, ..., 0].detach(),
                            depth_pred=depth_pred_for_loss[b0, vi, ..., 0].detach(),
                            point_tgt_raw=point_tgt_input_raw[b0, vi].detach(),
                            point_tgt_loss=point_tgt_for_loss[b0, vi].detach(),
                            point_pred_raw=point_pred_raw[b0, vi].detach(),
                            point_pred_loss=point_pred[b0, vi].detach(),
                            conf_tgt=conf_tgt[b0, vi].detach(),
                            conf_pred=conf_pred[b0, vi].detach(),
                            valid01=valid[b0, vi].detach(),
                            weight01=w[b0, vi].detach(),
                            fg01=(fg_mask_used[b0, vi].detach() if fg_mask_used is not None else None),
                            extra=debug_extra,
                        )
                    debug_vis_saved += 1

            if int(args.max_steps_per_epoch) > 0 and steps_this_epoch >= int(args.max_steps_per_epoch):
                break

        mean_loss = loss_sum / max(1, count)
        mean_depth = loss_depth_sum / max(1, count)
        mean_point = loss_point_sum / max(1, count)
        mean_point_prior = loss_point_prior_sum / max(1, count)
        mean_point_reproj = loss_point_reproj_sum / max(1, count)
        mean_point_normal_consis = loss_point_normal_consis_sum / max(1, count)
        mean_point_mv_depth = loss_point_mv_depth_sum / max(1, count)
        mean_point_mv_mask = loss_point_mv_mask_sum / max(1, count)
        mean_fg_structure_depth_edge = loss_fg_structure_depth_edge_sum / max(1, count)
        mean_point_mv_outside_ring = loss_point_mv_outside_ring_sum / max(1, count)
        mean_conf = loss_conf_sum / max(1, count)
        mean_fg_conf_presence = loss_fg_conf_presence_sum / max(1, count)
        mean_geom_cons = loss_geom_cons_sum / max(1, count)
        mean_cam = loss_cam_sum / max(1, count)
        improved = mean_loss < (best_loss - float(args.min_improve))
        if improved:
            best_loss = float(mean_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            _atomic_torch_save(model.state_dict(), out_best)
        else:
            bad_epochs += 1

        epoch_msg = {
            "event": "epoch_end",
            "epoch": int(epoch),
            "mean_loss": float(mean_loss),
            "mean_loss_depth": float(mean_depth),
            "mean_loss_point": float(mean_point),
            "mean_loss_point_prior": float(mean_point_prior),
            "mean_loss_point_reproj": float(mean_point_reproj),
            "mean_loss_point_normal_consis": float(mean_point_normal_consis),
            "mean_loss_point_mv_depth": float(mean_point_mv_depth),
            "mean_loss_point_mv_mask": float(mean_point_mv_mask),
            "mean_loss_fg_structure_depth_edge": float(mean_fg_structure_depth_edge),
            "mean_loss_point_mv_outside_ring": float(mean_point_mv_outside_ring),
            "mean_loss_conf": float(mean_conf),
            "mean_loss_fg_conf_presence": float(mean_fg_conf_presence),
            "mean_loss_geom_cons": float(mean_geom_cons),
            "mean_loss_cam": float(mean_cam),
            "mean_grad_norm": float(grad_norm_sum / max(1, count)),
            "mean_step_update_norm": float(step_update_norm_sum / max(1, count)),
            "mean_step_update_ratio": float(step_update_ratio_sum / max(1, count)),
            "steps": int(count),
            "improved": bool(improved),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "bad_epochs": int(bad_epochs),
        }
        epoch_msg.update(last_scale_info)
        print(f"[finetune] epoch={epoch} mean_loss={mean_loss:.6f}")
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_msg, ensure_ascii=False) + "\n")

        if int(args.early_stop_patience) > 0 and bad_epochs >= int(args.early_stop_patience):
            print(
                "[finetune] early_stop "
                f"epoch={epoch} bad_epochs={bad_epochs} patience={int(args.early_stop_patience)}"
            )
            break

    _atomic_torch_save(model.state_dict(), out_last)
    if not out_best.exists():
        _atomic_torch_save(model.state_dict(), out_best)

    summary = {
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_epoch": int(last_epoch),
        "final_step": int(step),
        "out_best": str(out_best),
        "out_last": str(out_last),
        "freeze_mode": str(args.freeze_mode),
        "depth_scale_align": str(args.depth_scale_align),
        "lr": float(args.lr),
        "lr_backbone_scale": float(args.lr_backbone_scale),
        "lr_head_scale": float(args.lr_head_scale),
        "lr_camera_scale": float(args.lr_camera_scale),
        "grad_clip": float(args.grad_clip),
        "lambda_depth": float(lambda_depth),
        "lambda_point": float(lambda_point),
        "lambda_point_prior": float(lambda_point_prior),
        "lambda_point_reproj": float(lambda_point_reproj),
        "lambda_point_normal_consis": float(lambda_point_normal_consis),
        "lambda_point_mv_depth": float(lambda_point_mv_depth),
        "lambda_point_mv_mask": float(lambda_point_mv_mask),
        "lambda_conf": float(lambda_conf),
        "lambda_geom_cons": float(lambda_geom_cons),
        "lambda_cam": float(lambda_cam),
        "cam_rot_weight": float(args.cam_rot_weight),
        "cam_fov_weight": float(args.cam_fov_weight),
        "cam_warmup_steps": int(args.cam_warmup_steps),
        "point_loss_scale_depth_unproject": float(args.point_loss_scale_depth_unproject),
        "point_warmup_steps": int(args.point_warmup_steps),
        "point_normal_consis_warmup_steps": int(args.point_normal_consis_warmup_steps),
        "point_reproj_warmup_steps": int(args.point_reproj_warmup_steps),
        "point_reproj_clamp_px": float(args.point_reproj_clamp_px),
        "human_prior_enable": str(args.human_prior_enable),
        "human_prior_subdir": str(args.human_prior_subdir),
        "human_prior_strict": str(args.human_prior_strict),
        "human_prior_point_blend_alpha": float(args.human_prior_point_blend_alpha),
        "human_prior_point_blend_region": str(args.human_prior_point_blend_region),
        "human_prior_weight_boost": float(args.human_prior_weight_boost),
        "human_prior_weight_region": str(args.human_prior_weight_region),
        "human_prior_loss_region": str(args.human_prior_loss_region),
        "human_prior_complete_weight": float(args.human_prior_complete_weight),
        "human_prior_complete_region": str(args.human_prior_complete_region),
        "human_prior_region_erode_px": int(args.human_prior_region_erode_px),
        "human_prior_head_fallback_top_ratio": float(args.human_prior_head_fallback_top_ratio),
        "human_prior_face_fallback_top_ratio": float(args.human_prior_face_fallback_top_ratio),
        "use_fg_mask": str(args.use_fg_mask),
        "fg_mask_source": str(args.fg_mask_source),
        "fg_mask_erode_px": int(args.fg_mask_erode_px),
        "point_loss_fg_erode_px": int(args.point_loss_fg_erode_px),
        "fg_supervision_boost": float(args.fg_supervision_boost),
        "fg_supervision_bg_floor": float(args.fg_supervision_bg_floor),
        "fg_supervision_region_mode": str(args.fg_supervision_region_mode),
        "fg_supervision_region_erode_px": int(args.fg_supervision_region_erode_px),
        "lambda_fg_conf_presence": float(args.lambda_fg_conf_presence),
        "fg_conf_presence_target_ratio": float(args.fg_conf_presence_target_ratio),
        "lambda_fg_structure_depth_edge": float(args.lambda_fg_structure_depth_edge),
        "fg_structure_bbox_margin_px": int(args.fg_structure_bbox_margin_px),
        "fg_structure_bbox_min_side_px": int(args.fg_structure_bbox_min_side_px),
        "fg_structure_region_mode": str(args.fg_structure_region_mode),
        "fg_structure_region_erode_px": int(args.fg_structure_region_erode_px),
        "fg_structure_depth_edge_warmup_steps": int(args.fg_structure_depth_edge_warmup_steps),
        "fg_structure_boundary_probe_px": int(args.fg_structure_boundary_probe_px),
        "fg_structure_edge_support_mode": str(args.fg_structure_edge_support_mode),
        "fg_structure_edge_support_quantile": float(args.fg_structure_edge_support_quantile),
        "fg_structure_edge_support_min_px": int(args.fg_structure_edge_support_min_px),
        "fg_structure_edge_weight_mode": str(args.fg_structure_edge_weight_mode),
        "fg_structure_boundary_falloff_px": int(args.fg_structure_boundary_falloff_px),
        "fg_structure_component_bias_mode": str(args.fg_structure_component_bias_mode),
        "fg_structure_component_bias_threshold_ratio": float(args.fg_structure_component_bias_threshold_ratio),
        "fg_structure_component_bias_other_scale": float(args.fg_structure_component_bias_other_scale),
        "fg_structure_front_depth_bias_mode": str(args.fg_structure_front_depth_bias_mode),
        "fg_structure_front_depth_bias_tau": float(args.fg_structure_front_depth_bias_tau),
        "fg_structure_front_depth_bias_center_quantile": float(args.fg_structure_front_depth_bias_center_quantile),
        "lambda_point_mv_outside_ring": float(args.lambda_point_mv_outside_ring),
        "point_mv_outside_ring_px": int(args.point_mv_outside_ring_px),
        "supervision_weight_mode": str(args.supervision_weight_mode),
        "supervision_weight_mix_alpha": float(args.supervision_weight_mix_alpha),
        "point_mv_consistency": str(args.point_mv_consistency),
        "point_mv_tol_abs": float(args.point_mv_tol_abs),
        "point_mv_tol_rel": float(args.point_mv_tol_rel),
        "point_mv_weight_floor": float(args.point_mv_weight_floor),
        "point_mv_stride": int(args.point_mv_stride),
        "point_mv_depth_max_pairs": int(args.point_mv_depth_max_pairs),
        "point_mv_depth_pair_mode": str(args.point_mv_depth_pair_mode),
        "point_mv_depth_warmup_steps": int(args.point_mv_depth_warmup_steps),
        "point_mv_depth_region_mode": str(args.point_mv_depth_region_mode),
        "point_mv_mask_warmup_steps": int(args.point_mv_mask_warmup_steps),
        "point_mv_depth_inlier_only": str(args.point_mv_depth_inlier_only),
        "point_mv_depth_err_quantile": float(args.point_mv_depth_err_quantile),
        "point_mv_depth_outlier_boost": float(args.point_mv_depth_outlier_boost),
        "point_mv_depth_outlier_cap": float(args.point_mv_depth_outlier_cap),
        "point_mv_depth_tgt_valid_mode": str(args.point_mv_depth_tgt_valid_mode),
        "point_mv_depth_tgt_valid_floor": float(args.point_mv_depth_tgt_valid_floor),
        "point_mv_depth_min_tgt_valid_ratio": float(args.point_mv_depth_min_tgt_valid_ratio),
        "point_mv_mask_min_tgt_fg_ratio": float(args.point_mv_mask_min_tgt_fg_ratio),
        "point_mv_mask_hit_thr": float(args.point_mv_mask_hit_thr),
        "point_mv_mask_soft_blur_px": int(args.point_mv_mask_soft_blur_px),
        "point_mv_mask_soft_blur_iters": int(args.point_mv_mask_soft_blur_iters),
        "point_mv_mask_soft_mix": float(args.point_mv_mask_soft_mix),
        "point_mv_mask_soft_hit_thr": float(args.point_mv_mask_soft_hit_thr),
        "point_mv_depth_tgt_valid_scale_mode": str(args.point_mv_depth_tgt_valid_scale_mode),
        "point_mv_depth_tgt_valid_scale_thr": float(args.point_mv_depth_tgt_valid_scale_thr),
        "point_mv_depth_adapt_mode": str(args.point_mv_depth_adapt_mode),
        "point_mv_depth_adapt_target_valid": float(args.point_mv_depth_adapt_target_valid),
        "point_mv_depth_adapt_min_scale": float(args.point_mv_depth_adapt_min_scale),
        "point_mv_depth_adapt_max_scale": float(args.point_mv_depth_adapt_max_scale),
        "point_support_mode": str(args.point_support_mode),
        "point_support_floor": float(args.point_support_floor),
        "point_mv_depth_support_mode": str(args.point_mv_depth_support_mode),
        "point_mv_depth_support_floor": float(args.point_mv_depth_support_floor),
        "point_mv_mask_support_mode": str(args.point_mv_mask_support_mode),
        "point_mv_mask_support_floor": float(args.point_mv_mask_support_floor),
        "point_mv_depth_fg_erode_px": int(args.point_mv_depth_fg_erode_px),
        "point_cons_quantile": float(args.point_cons_quantile),
        "point_cons_focus": str(args.point_cons_focus),
        "point_cons_clip_min_qv": float(args.point_cons_clip_min_qv),
        "point_residual_quantile": float(args.point_residual_quantile),
        "point_residual_focus": str(args.point_residual_focus),
        "point_residual_boost": float(args.point_residual_boost),
        "point_residual_boost_cap": float(args.point_residual_boost_cap),
        "point_target_mode": str(args.point_target_mode),
        "point_target_blend_alpha": float(args.point_target_blend_alpha),
        "point_target_blend_alpha_min": float(args.point_target_blend_alpha_min),
        "point_target_blend_alpha_max": float(args.point_target_blend_alpha_max),
        "point_target_blend_rel_gain": float(args.point_target_blend_rel_gain),
        "point_target_blend_mv_gain": float(args.point_target_blend_mv_gain),
        "point_target_blend_by_reliability": str(args.point_target_blend_by_reliability),
        "point_target_blend_by_mv_support": str(args.point_target_blend_by_mv_support),
        "point_target_blend_mv_region_mode": str(args.point_target_blend_mv_region_mode),
        "point_target_blend_mv_policy": str(args.point_target_blend_mv_policy),
        "point_target_consensus_alpha_floor": float(args.point_target_consensus_alpha_floor),
        "target_point_frame": str(args.target_point_frame),
        "pred_point_frame": str(args.pred_point_frame),
        "conf_weight_thr": float(args.conf_weight_thr),
        "conf_weight_gamma": float(args.conf_weight_gamma),
        "conf_weight_per_view_quantile": float(args.conf_weight_per_view_quantile),
        "conf_weight_per_view_min_valid": int(args.conf_weight_per_view_min_valid),
        "gram_dyn_enable": str(args.gram_dyn_enable),
        "gram_dyn_layer_idx": int(args.gram_dyn_layer_idx),
        "gram_dyn_quantile": float(args.gram_dyn_quantile),
        "gram_dyn_weight_floor": float(args.gram_dyn_weight_floor),
        "gram_dyn_warmup_steps": int(args.gram_dyn_warmup_steps),
        "dyn_proxy_enable": str(args.dyn_proxy_enable),
        "dyn_proxy_mode": str(args.dyn_proxy_mode),
        "dyn_proxy_use_gram": str(args.dyn_proxy_use_gram),
        "dyn_proxy_use_support": str(args.dyn_proxy_use_support),
        "dyn_proxy_floor": float(args.dyn_proxy_floor),
        "dyn_proxy_warmup_steps": int(args.dyn_proxy_warmup_steps),
        "robust_l1_eps": float(args.robust_l1_eps),
        "epochs": int(args.epochs),
        "eval_every_steps": int(args.eval_every_steps),
        "debug_metrics_every_steps": int(args.debug_metrics_every_steps),
        "log_heartbeat_sec": float(args.log_heartbeat_sec),
        "debug_vis_every_steps": int(args.debug_vis_every_steps),
        "debug_vis_max_steps": int(args.debug_vis_max_steps),
        "debug_vis_views": int(args.debug_vis_views),
        "debug_vis_dir": str(args.debug_vis_dir),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[finetune] saved: {out_best}")


if __name__ == "__main__":
    main()
