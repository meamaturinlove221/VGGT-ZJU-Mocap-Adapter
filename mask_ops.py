# mask_ops.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

try:
    from scipy import ndimage as _scipy_ndimage  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _scipy_ndimage = None
    _HAVE_SCIPY = False


def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is NCHW.
    Accepts: (H,W), (C,H,W), (N,C,H,W)
    """
    if x.dim() == 2:
        return x[None, None, ...]
    if x.dim() == 3:
        return x[None, ...]
    if x.dim() == 4:
        return x
    raise ValueError(
        f"Unsupported dim={x.dim()} for tensor shape={tuple(x.shape)}")


def resize_mask_nearest(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    mask: (..., H, W) in {0,1} or [0,1]
    size_hw: (H_out, W_out)
    Return: same dtype as input, nearest resized.
    """
    m = ensure_4d(mask.float())
    # nearest is critical for binary masks
    out = F.interpolate(m, size=size_hw, mode="nearest")
    out = out.squeeze(0)
    # keep as float 0/1
    return out


def resize_img_bilinear(img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    img: (..., H, W) float in [0,1] or [0,255]
    bilinear resize for images.
    """
    x = ensure_4d(img.float())
    out = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0)


def binarize(mask: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (mask.float() >= thresh).float()


def smooth_confidence_map(
    conf: Optional[torch.Tensor],
    kernel_size: int = 3,
    passes: int = 1,
) -> Optional[torch.Tensor]:
    """
    Lightly smooth confidence maps to reduce speckle before gating/weighting.
    conf: (H,W)/(1,H,W)/(B,1,H,W)/(B,C,H,W)
    """
    if conf is None:
        return None
    k = int(kernel_size)
    if k <= 1:
        return conf
    if (k % 2) == 0:
        k += 1
    n_pass = max(1, int(passes))

    x = conf.float()
    if x.dim() == 2:
        x = x[None, None, ...]
    elif x.dim() == 3:
        x = x[:, None, ...] if x.shape[0] > 1 else x[None, ...]
        if x.dim() == 3:
            x = x[:, None, ...]
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"smooth_confidence_map expects 2D/3D/4D, got {tuple(x.shape)}")

    for _ in range(n_pass):
        x = F.avg_pool2d(x, kernel_size=k, stride=1,
                         padding=k // 2, count_include_pad=False)
    x = x.clamp(0.0, 1.0)

    if conf.dim() == 2:
        return x[0, 0]
    if conf.dim() == 3:
        if conf.shape[0] > 1:
            return x[:, 0]
        return x[0]
    return x


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


def _as_pointmap_b3hw(pointmap: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if pointmap is None or (not torch.is_tensor(pointmap)):
        return None
    pm = pointmap
    if pm.dim() == 3:
        if pm.shape[0] == 3:
            return pm.unsqueeze(0)
        if pm.shape[-1] == 3:
            return pm.permute(2, 0, 1).unsqueeze(0)
        return None
    if pm.dim() == 4:
        if pm.shape[1] == 3:
            return pm
        if pm.shape[-1] == 3:
            return pm.permute(0, 3, 1, 2)
        return None
    return None


def _fit_plane_ls(u: torch.Tensor, v: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
    if u.numel() < 3 or v.numel() < 3 or y.numel() < 3:
        return None
    A = torch.stack([u, v, torch.ones_like(u)], dim=1)  # [N,3]
    b = y.unsqueeze(1)  # [N,1]
    try:
        sol = torch.linalg.lstsq(A, b).solution  # [3,1]
        if sol.numel() >= 3:
            return sol[:3, 0]
    except Exception:
        pass
    try:
        pinv = torch.linalg.pinv(A)
        sol2 = (pinv @ b).squeeze(1)
        if sol2.numel() >= 3:
            return sol2[:3]
    except Exception:
        pass
    return None


def drop_ground_from_fg_plane(
    fg_mask: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    pointmap: Optional[torch.Tensor],
    out_hw: Tuple[int, int],
    axis: int = 1,
    margin: float = 0.05,
    min_points: int = 256,
    seed_q: float = 0.20,
    inlier_q: float = 0.70,
    refine_iters: int = 2,
    fallback_q: float = 0.05,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Remove likely ground from fg_mask using a robust plane fit on pointmap:
    1) seed by low-height quantile
    2) least-squares fit plane y = a*u + b*v + c
    3) iterative residual inlier refinement
    """
    info: Dict[str, Any] = {
        "applied": False,
        "method": "plane",
        "fallback": [],
        "floor_vals": [],
        "coeffs": [],
        "num_points": [],
        "num_inliers": [],
    }
    if fg_mask is None:
        return fg_mask, info
    pm = _as_pointmap_b3hw(pointmap)
    if pm is None:
        return fg_mask, info

    fg = ensure_4d(fg_mask).float()
    vm = ensure_4d(valid_mask).float() if valid_mask is not None else None
    if fg is None:
        return fg_mask, info
    if pm.shape[0] == 1 and fg.shape[0] > 1:
        pm = pm.expand(fg.shape[0], -1, -1, -1)
    if vm is not None and vm.shape[0] == 1 and fg.shape[0] > 1:
        vm = vm.expand(fg.shape[0], -1, -1, -1)
    if pm.shape[0] != fg.shape[0]:
        return fg_mask, info
    if vm is not None and vm.shape[0] != fg.shape[0]:
        return fg_mask, info

    ax = int(axis)
    if ax < 0:
        ax = 3 + ax
    if ax < 0 or ax > 2:
        return fg_mask, info

    if pm.shape[-2:] != tuple(out_hw):
        pm = F.interpolate(pm.float(), size=out_hw, mode="bilinear", align_corners=False)
    else:
        pm = pm.float()

    keep_axes: List[int] = [0, 1, 2]
    keep_axes.remove(ax)
    u_all = pm[:, keep_axes[0]:keep_axes[0] + 1, :, :]
    v_all = pm[:, keep_axes[1]:keep_axes[1] + 1, :, :]
    y_all = pm[:, ax:ax + 1, :, :]

    out = fg.clone()
    B = int(out.shape[0])
    for b in range(B):
        mb = (fg[b:b + 1] > 0.5)
        if vm is not None:
            mb = mb & (vm[b:b + 1] > 0.5)
        mb2 = mb[0, 0]
        n_pts = int(mb2.sum().item())
        info["num_points"].append(n_pts)
        if n_pts < int(min_points):
            info["fallback"].append(True)
            info["coeffs"].append(None)
            info["num_inliers"].append(0)
            yb = y_all[b, 0][mb2]
            floor = _safe_quantile(yb, float(fallback_q))
            floor_v = float(floor.item()) if floor is not None else float("nan")
            info["floor_vals"].append(floor_v)
            if floor is not None:
                keep = (y_all[b, 0] > (floor + float(margin))).float()
                out[b, 0] = out[b, 0] * keep
            continue

        ub = u_all[b, 0][mb2]
        vb = v_all[b, 0][mb2]
        yb = y_all[b, 0][mb2]

        y_seed_thr = _safe_quantile(yb, float(seed_q))
        if y_seed_thr is not None:
            seed_mask = (yb <= y_seed_thr)
            if int(seed_mask.sum().item()) >= max(32, int(min_points) // 4):
                fit_u = ub[seed_mask]
                fit_v = vb[seed_mask]
                fit_y = yb[seed_mask]
            else:
                fit_u, fit_v, fit_y = ub, vb, yb
        else:
            fit_u, fit_v, fit_y = ub, vb, yb

        coeff = _fit_plane_ls(fit_u, fit_v, fit_y)
        if coeff is None:
            info["fallback"].append(True)
            info["coeffs"].append(None)
            info["num_inliers"].append(0)
            floor = _safe_quantile(yb, float(fallback_q))
            floor_v = float(floor.item()) if floor is not None else float("nan")
            info["floor_vals"].append(floor_v)
            if floor is not None:
                keep = (y_all[b, 0] > (floor + float(margin))).float()
                out[b, 0] = out[b, 0] * keep
            continue

        inliers = torch.ones_like(yb, dtype=torch.bool)
        for _ in range(max(0, int(refine_iters))):
            a, bb, c = coeff
            y_hat = a * ub + bb * vb + c
            resid = (yb - y_hat).abs()
            thr = _safe_quantile(resid, float(inlier_q))
            if thr is None:
                break
            inliers = resid <= thr
            if int(inliers.sum().item()) < max(32, int(min_points) // 4):
                break
            coeff_new = _fit_plane_ls(ub[inliers], vb[inliers], yb[inliers])
            if coeff_new is None:
                break
            coeff = coeff_new

        a, bb, c = coeff
        y_plane = a * u_all[b, 0] + bb * v_all[b, 0] + c
        keep = (y_all[b, 0] > (y_plane + float(margin))).float()
        out[b, 0] = out[b, 0] * keep

        floor_on_fg = y_plane[mb2]
        floor_ref = _safe_quantile(floor_on_fg, 0.5)
        info["fallback"].append(False)
        info["coeffs"].append([float(a.item()), float(bb.item()), float(c.item())])
        info["num_inliers"].append(int(inliers.sum().item()))
        info["floor_vals"].append(
            float(floor_ref.item()) if floor_ref is not None else float("nan"))

    info["applied"] = True
    if fg_mask.dim() == 2:
        return out[0, 0], info
    if fg_mask.dim() == 3:
        return out[0], info
    return out, info


def mask_dilation(mask01: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Binary dilation via max_pool2d (no opencv dependency).
    mask01: (..., H, W) float 0/1
    k: odd kernel size recommended.
    """
    if k <= 1:
        return mask01
    m = ensure_4d(mask01.float())
    pad = k // 2
    # maxpool acts like dilation for binary masks
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
    dil = (dil > 0).float()
    return dil.squeeze(0)


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w)),
            y2=max(0, min(self.y2, h)),
        )

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def bbox_from_mask(mask01: torch.Tensor, min_pixels: int = 10) -> Optional[BBox]:
    """
    Compute bbox from binary mask.
    mask01: (H,W) or (1,H,W) or (C,H,W) but will use first channel.
    """
    m = mask01
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        raise ValueError(f"bbox_from_mask expects 2D/3D mask, got {m.shape}")

    ys, xs = torch.where(m > 0.5)
    if ys.numel() < min_pixels:
        return None
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def expand_bbox(b: BBox, w: int, h: int, scale: float = 1.3) -> BBox:
    """
    Expand bbox around center by scale.
    """
    cx = (b.x1 + b.x2) / 2.0
    cy = (b.y1 + b.y2) / 2.0
    bw = b.width() * scale
    bh = b.height() * scale
    x1 = int(round(cx - bw / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    y2 = int(round(cy + bh / 2.0))
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2).clip(w=w, h=h)


def crop_by_bbox(
    img: torch.Tensor,
    bbox: BBox,
) -> torch.Tensor:
    """
    img: (C,H,W) or (H,W)
    bbox coords in original image.
    """
    if img.dim() == 2:
        return img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    if img.dim() == 3:
        return img[:, bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    raise ValueError(f"crop_by_bbox supports 2D/3D, got {img.shape}")


def roi_crop_and_resize(
    img: torch.Tensor,
    fg_mask01: torch.Tensor,
    valid_mask01: Optional[torch.Tensor],
    out_hw: Tuple[int, int],
    expand_scale: float = 1.3,
    min_pixels: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Crop around fg bbox, resize back to out_hw.
    This boosts fg pixel ratio dramatically if your fg is tiny in full frame.

    Returns dict with cropped-resized tensors:
    - img: (C,H,W)
    - fg_mask: (1,H,W)
    - valid_mask: (1,H,W) if provided
    """
    if img.dim() != 3:
        raise ValueError(f"img should be (C,H,W), got {img.shape}")

    C, H, W = img.shape
    fg = fg_mask01
    if fg.dim() == 2:
        fg = fg[None, ...]
    if fg.dim() != 3:
        raise ValueError(f"fg_mask should be (1,H,W) or (H,W), got {fg.shape}")

    bb = bbox_from_mask(fg[0], min_pixels=min_pixels)
    if bb is None:
        # fallback: no crop
        return {
            "img": resize_img_bilinear(img, out_hw),
            "fg_mask": resize_mask_nearest(fg, out_hw),
            "valid_mask": resize_mask_nearest(valid_mask01, out_hw) if valid_mask01 is not None else None,
        }

    bb = expand_bbox(bb, w=W, h=H, scale=expand_scale)

    img_c = crop_by_bbox(img, bb)
    fg_c = crop_by_bbox(fg, bb)
    val_c = crop_by_bbox(
        valid_mask01, bb) if valid_mask01 is not None else None

    img_r = resize_img_bilinear(img_c, out_hw)
    fg_r = resize_mask_nearest(fg_c, out_hw)
    fg_r = binarize(fg_r)

    out = {"img": img_r, "fg_mask": fg_r}
    if val_c is not None:
        val_r = resize_mask_nearest(val_c, out_hw)
        out["valid_mask"] = binarize(val_r)
    else:
        out["valid_mask"] = None
    return out


def cover_ratio(mask01: torch.Tensor) -> float:
    """
    mask01: (H,W) or (1,H,W)
    """
    m = mask01
    if m.dim() == 3:
        m = m[0]
    return float(m.float().mean().item())


def _erode_mask(mask01: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask01
    m = ensure_4d(mask01.float())
    pad = k // 2
    inv = 1.0 - m
    eroded = 1.0 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=pad)
    eroded = (eroded > 0.5).float()
    return eroded.squeeze(0)


def _count_connected_components(mask01: torch.Tensor) -> int:
    m = mask01
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        raise ValueError(f"count_cc expects 2D/3D mask, got {m.shape}")
    m_np = (m > 0.5).detach().cpu().numpy().astype(np.uint8)
    if m_np.sum() == 0:
        return 0
    if _HAVE_CV2 and cv2 is not None:
        try:
            num, _labels = cv2.connectedComponents(m_np, connectivity=4)
            return max(0, int(num) - 1)
        except Exception:
            pass
    if _HAVE_SCIPY and _scipy_ndimage is not None:
        try:
            _labels, num = _scipy_ndimage.label(m_np)
            return int(num)
        except Exception:
            pass
    # fallback: simple DFS (slow for huge masks, OK for debug)
    H, W = m_np.shape
    visited = np.zeros_like(m_np, dtype=np.uint8)
    cc = 0
    for y in range(H):
        for x in range(W):
            if m_np[y, x] == 0 or visited[y, x]:
                continue
            cc += 1
            stack = [(y, x)]
            visited[y, x] = 1
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and m_np[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    return int(cc)


def mask_stats(mask01: torch.Tensor, name: str = "") -> Dict[str, float]:
    m = mask01
    if m is None:
        return {f"{name}cover": 0.0, f"{name}cc": 0, f"{name}boundary": 0.0}
    if m.dim() == 4:
        m = m[0]
    if m.dim() == 3 and m.shape[0] != 1:
        m = m[:1]
    if m.dim() == 3:
        m2 = m[0]
    else:
        m2 = m
    m2 = (m2 > 0.5).float()
    cover = float(m2.mean().item())
    cc = _count_connected_components(m2)
    er = _erode_mask(m2, k=3)
    boundary = (m2 - er).clamp(0, 1)
    boundary_ratio = float(boundary.sum().item() / (m2.sum().item() + 1e-8))
    return {
        f"{name}cover": cover,
        f"{name}cc": float(cc),
        f"{name}boundary": boundary_ratio,
    }


def compose_masks(
    valid_geom: torch.Tensor,
    fg_mask: Optional[torch.Tensor],
    conf_mask: Optional[torch.Tensor],
    mode: str = "geom_fg_conf",
    min_gate: float = 0.05,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compose masks with explicit physical meaning.
    mode:
      - geom_fg_conf: valid_geom * fg * conf
      - geom_fg: valid_geom * fg
      - geom_conf: valid_geom * conf
      - geom_only: valid_geom
      - fg_conf: fg * conf
      - fg_only: fg
      - conf_only: conf
    """
    v = binarize(valid_geom)
    if fg_mask is None:
        fg = torch.ones_like(v)
    else:
        fg = binarize(fg_mask)
        if fg.shape[-2:] != v.shape[-2:]:
            fg = resize_mask_nearest(fg, v.shape[-2:])
            fg = binarize(fg)

    if conf_mask is None:
        conf = torch.ones_like(v)
    else:
        conf = conf_mask.float()
        if conf.dim() == 2:
            conf = conf[None, None, ...]
        if conf.dim() == 3:
            conf = conf.unsqueeze(1)
        if conf.shape[1] != 1:
            conf = conf[:, :1, ...]
        if conf.shape[-2:] != v.shape[-2:]:
            conf = F.interpolate(conf, size=v.shape[-2:], mode="bilinear", align_corners=False)
        conf = conf.clamp(0.0, 1.0)
        if float(min_gate) > 0:
            conf = float(min_gate) + (1.0 - float(min_gate)) * conf

    m = str(mode).lower()
    if m == "geom_fg_conf":
        out = v * fg * conf
    elif m == "geom_fg":
        out = v * fg
    elif m == "geom_conf":
        out = v * conf
    elif m == "geom_only":
        out = v
    elif m == "fg_conf":
        out = fg * conf
    elif m == "fg_only":
        out = fg
    elif m == "conf_only":
        out = conf
    else:
        out = v * fg * conf

    out = out.clamp(0.0, 1.0)
    stats: Dict[str, float] = {}
    if return_stats:
        stats.update(mask_stats(v, "valid_"))
        stats.update(mask_stats(fg, "fg_"))
        stats.update(mask_stats(conf, "conf_"))
        stats.update(mask_stats(out, "out_"))
    return out, stats

def overlay_mask_pil(
    img_chw: torch.Tensor,
    mask01_hw: torch.Tensor,
    color_rgb=(255, 0, 0),
    alpha: float = 0.45,
) -> Image.Image:
    """
    img_chw: torch (C,H,W) in [0,1] or [0,255]
    mask01_hw: torch (H,W) or (1,H,W) in {0,1}
    Return PIL.Image (RGB) with colored overlay.
    """
    img = img_chw.detach().cpu().float()
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255).byte()
    img_np = img.permute(1, 2, 0).numpy()  # HWC

    m = mask01_hw.detach().cpu().float()
    if m.dim() == 3:
        m = m[0]
    m_np = (m > 0.5).numpy().astype(np.uint8)

    overlay = img_np.copy()
    overlay[m_np == 1] = np.array(color_rgb, dtype=np.uint8)

    out = (img_np.astype(np.float32) * (1 - alpha) +
           overlay.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def save_overlay_triplet(
    out_dir: str,
    prefix: str,
    tgt_img_chw: torch.Tensor,
    fg_mask01: torch.Tensor,
    valid_mask01: Optional[torch.Tensor],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # base
    base = tgt_img_chw.detach().cpu().float()
    if base.max() <= 1.0:
        base = base * 255.0
    base = base.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(base, mode="RGB").save(
        os.path.join(out_dir, f"{prefix}_tgt.png"))

    # fg overlay
    overlay_fg = overlay_mask_pil(
        tgt_img_chw, fg_mask01, color_rgb=(255, 0, 0), alpha=0.45)
    overlay_fg.save(os.path.join(out_dir, f"{prefix}_fg_overlay.png"))

    # valid overlay (if any)
    if valid_mask01 is not None:
        overlay_val = overlay_mask_pil(
            tgt_img_chw, valid_mask01, color_rgb=(0, 255, 0), alpha=0.45)
        overlay_val.save(os.path.join(out_dir, f"{prefix}_valid_overlay.png"))
