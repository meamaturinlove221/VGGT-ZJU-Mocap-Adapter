# debug_vis_utils.py
# -*- coding: utf-8 -*-

import os
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torchvision.utils import save_image
except Exception as e:
    raise ImportError("需要 torchvision 才能保存图片：pip install torchvision") from e


# ---------------------------
# 0) 训练稳定性设置（可选）
# ---------------------------
def set_stable_cudnn_tf32(ieee: bool = True) -> None:
    """
    更稳的 cudnn/tf32 设置（Windows 上尤其有用）。
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 新接口：fp32_precision
    if ieee:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.fp32_precision = "high"
        torch.backends.cudnn.conv.fp32_precision = "high"


# ---------------------------
# 1) 张量形状/数值统计打印
# ---------------------------
@torch.no_grad()
def tensor_stats(name: str, x: torch.Tensor) -> None:
    """
    打印 min/max/mean/std/shape/dtype/device，帮你确认到底是哪类张量。
    """
    if x is None:
        print(f"[stats] {name}: None")
        return
    if not torch.is_tensor(x):
        print(f"[stats] {name}: type={type(x)} (not tensor)")
        return

    xx = x.detach()
    # 避免半精度统计溢出
    if xx.dtype in (torch.float16, torch.bfloat16):
        xx = xx.float()

    # 有些张量可能是空的
    if xx.numel() == 0:
        print(
            f"[stats] {name}: EMPTY shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        return

    mn = float(xx.min().cpu())
    mx = float(xx.max().cpu())
    mean = float(xx.mean().cpu())
    std = float(xx.std(unbiased=False).cpu())

    print(
        f"[stats] {name}: "
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}"
    )

    # 额外：RGB 通道均值（判断是不是只有G通道有值）
    # 尽量容错：只要能抽到 [C,H,W] 且 C>=3
    try:
        img = _to_chw(_pick_first(x))
        if img.dim() == 3 and img.size(0) >= 3:
            ch_mean = img[:3].float().mean(dim=(1, 2)).cpu().tolist()
            print(f"[stats] {name}: channel_mean(R,G,B)={ch_mean}")
    except Exception:
        pass


# ---------------------------
# 2) 通用：从任意形状取“第一张”并转 CHW
# ---------------------------
def _pick_first(x: torch.Tensor) -> torch.Tensor:
    """
    尽量兼容这些形状：
    - [B,C,H,W] -> 取 x[0]
    - [B,V,C,H,W] -> 取 x[0,0]
    - [C,H,W] -> 原样
    - [H,W] -> 原样
    - [B,H,W] -> 取 x[0]
    """
    if x.dim() == 5:     # B,V,C,H,W or B,V,H,W,C
        return x[0, 0]
    if x.dim() == 4:     # B,C,H,W or B,H,W,C
        return x[0]
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x
    if x.dim() == 1:
        return x
    return x


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    把张量尽量转成 [C,H,W]：
    - [H,W] -> [1,H,W]
    - [H,W,C] (C=1/3) -> [C,H,W]
    - [C,H,W] -> 原样
    """
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        # 判断是不是 HWC
        if x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            return x.permute(2, 0, 1).contiguous()
        return x
    # 其他形状就直接拍平处理（不常见）
    return x


def _normalize_to_01(img: torch.Tensor, assume: str = "auto") -> torch.Tensor:
    """
    把图像张量转成 [0,1]：
    - assume='auto': 若 max>1 或 min<0，尝试判断 [-1,1] 或任意范围做 min-max
    - assume='minus1_1': 认为是 [-1,1]
    - assume='0_1': 认为已是 [0,1]
    """
    img = img.detach()
    if img.dtype in (torch.float16, torch.bfloat16):
        img = img.float()

    if assume == "0_1":
        return img.clamp(0, 1)
    if assume == "minus1_1":
        return ((img + 1.0) * 0.5).clamp(0, 1)

    # auto
    mn = float(img.min())
    mx = float(img.max())

    # 常见：[-1,1]
    if mn >= -1.1 and mx <= 1.1 and (mn < 0.0):
        return ((img + 1.0) * 0.5).clamp(0, 1)

    # 常见：[0,1]
    if mn >= -1e-6 and mx <= 1.0 + 1e-6:
        return img.clamp(0, 1)

    # 其他：做 min-max（用于 conf/深度/乱七八糟的中间量）
    eps = 1e-8
    return ((img - img.min()) / (img.max() - img.min() + eps)).clamp(0, 1)


@torch.no_grad()
def save_tensor_image(
    path: str,
    x: torch.Tensor,
    assume_range: str = "auto",
    force_3ch: bool = True,
) -> None:
    """
    把任意张量保存为 png：
    - 自动选第一张
    - 自动转 CHW
    - 自动归一化到 [0,1]
    - mask/conf 默认扩成 3 通道便于肉眼看
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = _to_chw(_pick_first(x))

    # 处理 1D 情况（很少）
    if img.dim() == 1:
        img = img.unsqueeze(0).unsqueeze(-1)  # [1, N, 1] 仅为了不报错

    img01 = _normalize_to_01(img, assume=assume_range)

    if force_3ch and img01.dim() == 3 and img01.size(0) == 1:
        img01 = img01.repeat(3, 1, 1)

    save_image(img01.cpu(), path)


# ---------------------------
# 3) valid/mask 构建：阈值 + 平滑 + 膨胀 + 最小覆盖率保护
# ---------------------------
@torch.no_grad()
def dilate_mask(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    """
    用 max_pool2d 做膨胀，k 必须为奇数更直观。
    mask: [B,1,H,W] 或 [1,H,W] 或 [H,W]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1,1,H,W] 或 [1,C,H,W]（我们只取第0通道）
        if mask.size(1) != 1:
            mask = mask[:, :1]
    elif mask.dim() == 4:
        if mask.size(1) != 1:
            mask = mask[:, :1]
    else:
        raise ValueError(f"dilate_mask: unsupported shape {tuple(mask.shape)}")

    k = int(k)
    if k <= 1:
        return (mask > 0.5).float()

    pad = k // 2
    out = F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=pad)
    return (out > 0.5).float()


@torch.no_grad()
def smooth_conf(conf: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    用 avg_pool2d 对 conf 平滑（避免碎点云）。
    conf: [B,1,H,W] / [H,W] / [1,H,W]
    """
    if conf.dim() == 2:
        conf = conf.unsqueeze(0).unsqueeze(0)
    elif conf.dim() == 3:
        conf = conf.unsqueeze(0)
        if conf.size(1) != 1:
            conf = conf[:, :1]
    elif conf.dim() == 4:
        if conf.size(1) != 1:
            conf = conf[:, :1]
    else:
        raise ValueError(f"smooth_conf: unsupported shape {tuple(conf.shape)}")

    if k <= 1:
        return conf.float()

    pad = k // 2
    return F.avg_pool2d(conf.float(), kernel_size=k, stride=1, padding=pad)


@torch.no_grad()
def make_valid_mask(
    conf: torch.Tensor,
    thr: Optional[float] = 0.1,
    quantile: Optional[float] = None,
    smooth_k: int = 0,
    dilate_k: int = 0,
    min_coverage: float = 0.10,
    fallback_quantile: float = 0.20,
    fallback_full: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    从 conf 生成 valid mask，并确保覆盖率不要小得离谱。

    参数：
    - thr：固定阈值（例如 0.1）
    - quantile：用分位数阈值（例如 0.2 表示取 top 80% 当 valid）
    - smooth_k：先平滑 conf 再阈值（建议 3~7）
    - dilate_k：阈值后对 mask 膨胀（建议 5~11）
    - min_coverage：覆盖率下限（比如 0.10 = 至少 10% 像素为 valid）
    - fallback_quantile：如果覆盖率太低，用更宽松的分位数兜底
    - fallback_full：再不行就全图 valid（至少别让 loss 被 1% 像素“骗”）

    返回：
    - mask: [B,1,H,W] float {0,1}
    - info: 覆盖率等信息
    """
    # 统一形状到 [B,1,H,W]
    if conf.dim() == 2:
        conf4 = conf.unsqueeze(0).unsqueeze(0)
    elif conf.dim() == 3:
        conf4 = conf.unsqueeze(0)
        if conf4.size(1) != 1:
            conf4 = conf4[:, :1]
    elif conf.dim() == 4:
        conf4 = conf
        if conf4.size(1) != 1:
            conf4 = conf4[:, :1]
    else:
        raise ValueError(
            f"make_valid_mask: unsupported shape {tuple(conf.shape)}")

    conf4 = conf4.float()
    if smooth_k and smooth_k > 1:
        conf4 = smooth_conf(conf4, k=smooth_k)

    # 选阈值
    if quantile is not None:
        q = float(quantile)
        # per-batch 计算阈值，避免不同样本尺度差异
        flat = conf4.flatten(2)  # [B,1,HW]
        thr_b = torch.quantile(flat, q=q, dim=2, keepdim=True)  # [B,1,1]
        thr_val = thr_b.view(-1, 1, 1, 1)
        mask = (conf4 > thr_val).float()
        used_thr = float(thr_b.mean().item())
        used_mode = f"quantile={q:.3f}"
    else:
        used_thr = float(thr if thr is not None else 0.0)
        mask = (conf4 > used_thr).float()
        used_mode = f"thr={used_thr:.4f}"

    if dilate_k and dilate_k > 1:
        mask = dilate_mask(mask, k=dilate_k)

    # 覆盖率检查
    cov = float(mask.mean().item())

    # 兜底：覆盖率太低就放宽
    if cov < float(min_coverage):
        # 先用 fallback_quantile 再试一次
        flat = conf4.flatten(2)
        thr_b = torch.quantile(flat, q=float(
            fallback_quantile), dim=2, keepdim=True)
        thr_val = thr_b.view(-1, 1, 1, 1)
        mask2 = (conf4 > thr_val).float()
        if dilate_k and dilate_k > 1:
            mask2 = dilate_mask(mask2, k=dilate_k)
        cov2 = float(mask2.mean().item())

        if cov2 >= cov:
            mask, cov = mask2, cov2
            used_thr = float(thr_b.mean().item())
            used_mode = f"fallback_quantile={fallback_quantile:.3f}"

        # 仍然太低：可选全图 valid
        if cov < float(min_coverage) and fallback_full:
            mask = torch.ones_like(mask)
            cov = 1.0
            used_mode = "fallback_full"

    info = {
        "coverage": cov,
        "conf_min": float(conf4.min().item()),
        "conf_max": float(conf4.max().item()),
        "conf_mean": float(conf4.mean().item()),
        "used_thr": float(used_thr),
        "mode": used_mode,
    }
    return mask, info


# ---------------------------
# 4) 指标：full L1 / masked L1 / PSNR
# ---------------------------
@torch.no_grad()
def l1_full(pred_rgb: torch.Tensor, tgt_rgb: torch.Tensor) -> torch.Tensor:
    return (pred_rgb - tgt_rgb).abs().mean()


@torch.no_grad()
def l1_masked(pred_rgb: torch.Tensor, tgt_rgb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/tgt: [B,3,H,W]
    mask: [B,1,H,W] 或 [B,3,H,W]
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1:
        mask3 = mask.repeat(1, 3, 1, 1)
    else:
        mask3 = mask

    diff = (pred_rgb - tgt_rgb).abs() * mask3
    denom = mask3.mean().clamp_min(eps)  # 用均值等价于按像素比例归一
    return diff.mean() / denom


@torch.no_grad()
def psnr_full(pred_rgb: torch.Tensor, tgt_rgb: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/tgt 默认在 [0,1]，data_range=1.0
    """
    mse = (pred_rgb - tgt_rgb).pow(2).mean().clamp_min(eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)


# ---------------------------
# 5) 一键保存 sanity pack（单独图 + 打印）
# ---------------------------
@torch.no_grad()
def save_sanity_pack(
    out_dir: str,
    step: int,
    tensors: Dict[str, torch.Tensor],
    assume_range_rgb: str = "auto",
    assume_range_other: str = "auto",
    print_stats: bool = True,
) -> None:
    """
    tensors 里放你想查的东西，比如：
      {
        "src_rgb": src_imgs,         # 任意形状，内部会取第一张
        "tgt_rgb": tgt_img,
        "pred_rgb": pred_rgb,
        "conf": tgt_conf,
        "mask": valid_mask,
      }

    会输出：
      out_dir/step000123_src_rgb.png
      out_dir/step000123_tgt_rgb.png
      out_dir/step000123_pred_rgb.png
      out_dir/step000123_conf.png
      out_dir/step000123_mask.png
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"step{int(step):06d}"

    for k, v in tensors.items():
        if v is None:
            continue
        if print_stats:
            tensor_stats(k, v)

        # RGB 用 assume_range_rgb，其他用 assume_range_other
        assume = assume_range_rgb if (
            "rgb" in k.lower() or "img" in k.lower()) else assume_range_other
        force_3ch = True
        save_tensor_image(
            path=os.path.join(out_dir, f"{prefix}_{k}.png"),
            x=v,
            assume_range=assume,
            force_3ch=force_3ch,
        )
