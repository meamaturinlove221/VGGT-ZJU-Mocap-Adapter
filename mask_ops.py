# mask_ops.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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
