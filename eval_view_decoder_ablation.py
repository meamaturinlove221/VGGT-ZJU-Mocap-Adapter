# -*- coding: utf-8 -*-
"""
eval_view_decoder_ablation.py

用法示例（Windows）：
(vit-torch) F:\vggt> python eval_view_decoder_ablation.py ^
  --ckpt ckpt\viewdec_ablation_best_epoch19.pth ^
  --train_script train_view_decoder_ablation.py ^
  --split val ^
  --out_dir runs\eval_epoch19 ^
  --num_vis 24

说明：
- 这个脚本会“动态导入”你训练用的 train_view_decoder_ablation.py，
  复用其中的 Dataset/Model 定义，避免架构不匹配。
- 若你的 train 脚本里类/函数名不同，可在本脚本顶部的“HOOK 名称”里改一下映射。
"""

import os
import sys
import json
import math
import time
import argparse
import importlib.util
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torchvision.utils import save_image, make_grid
except Exception:
    save_image = None
    make_grid = None


# -----------------------------
# HOOK 名称（如你脚本里命名不同，改这里）
# -----------------------------
DATASET_CLS_CANDIDATES = ["ZJUViewSynthDataset"]
MODEL_BUILD_FN_CANDIDATES = ["build_model", "create_model", "make_model"]
MODEL_CLS_CANDIDATES = ["ViewDecoder", "ViewDecoderAblation"]


def import_train_module(train_script_path: str):
    train_script_path = os.path.abspath(train_script_path)
    if not os.path.isfile(train_script_path):
        raise FileNotFoundError(f"train_script not found: {train_script_path}")

    spec = importlib.util.spec_from_file_location("train_mod", train_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import train script: {train_script_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_mod"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def pick_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n), n
    return None, None


def smart_image_range(x: torch.Tensor) -> torch.Tensor:
    """
    自动把图像张量变到 [0,1]：
    - 若发现明显像是 [-1,1]，就做 (x+1)/2
    - 否则直接 clamp
    """
    if x.numel() == 0:
        return x
    xmin = float(x.min().detach().cpu())
    xmax = float(x.max().detach().cpu())
    if xmin < -0.2 and xmax <= 1.2:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    pred/tgt: [B,3,H,W] in [0,1]
    mask: [B,1,H,W] 0/1
    """
    eps = 1e-8
    if mask is None:
        mse = F.mse_loss(pred, tgt, reduction="mean").clamp_min(eps)
        return float((-10.0 * torch.log10(mse)).detach().cpu())
    # masked mse
    w = mask
    diff2 = (pred - tgt) ** 2
    diff2 = diff2.mean(dim=1, keepdim=True)  # [B,1,H,W]
    num = (diff2 * w).sum()
    den = w.sum().clamp_min(1.0)
    mse = (num / den).clamp_min(eps)
    return float((-10.0 * torch.log10(mse)).detach().cpu())


def _gaussian_kernel(ch: int, device, dtype, ksize: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # 1D gaussian
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    # 2D
    g2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
    g2d = g2d.repeat(ch, 1, 1, 1)  # [C,1,K,K]
    return g2d


def ssim(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    简化版 SSIM（全图或 masked），pred/tgt: [B,3,H,W] in [0,1]
    """
    # 常量
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    pred = pred.clamp(0, 1)
    tgt = tgt.clamp(0, 1)

    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype
    k = _gaussian_kernel(C, device, dtype)

    def filt(x):
        return F.conv2d(x, k, padding=5, groups=C)

    mu_x = filt(pred)
    mu_y = filt(tgt)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(tgt * tgt) - mu_y2
    sigma_xy = filt(pred * tgt) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8)
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # [B,1,H,W]

    if mask is None:
        return float(ssim_map.mean().detach().cpu())

    w = mask
    num = (ssim_map * w).sum()
    den = w.sum().clamp_min(1.0)
    return float((num / den).detach().cpu())


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    pred/tgt: [B,3,H,W]
    mask: [B,1,H,W]
    """
    if mask is None:
        return float(F.l1_loss(pred, tgt, reduction="mean").detach().cpu())
    diff = (pred - tgt).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    num = (diff * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return float((num / den).detach().cpu())


def quantile_train_mask(pred_conf: torch.Tensor, cover: float) -> torch.Tensor:
    """
    pred_conf: [B,1,H,W] in [0,1]
    返回 mask: [B,1,H,W] 0/1，选取 top-cover 的像素
    """
    B = pred_conf.shape[0]
    flat = pred_conf.view(B, -1)
    # 阈值 = (1-cover) 分位点
    q = torch.quantile(flat, 1.0 - cover, dim=1, keepdim=True)  # [B,1]
    q = q.view(B, 1, 1, 1)
    return (pred_conf >= q).float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r"ckpt\viewdec_ablation_best_epoch19.pth")
    parser.add_argument("--train_script", type=str, default="train_view_decoder_ablation.py")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default=r"runs\eval_viewdec")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true", help="强制用 ema 权重（若有）")
    parser.add_argument("--no_ema", action="store_true", help="强制不用 ema 权重")
    parser.add_argument("--num_vis", type=int, default=24, help="最多导出多少个样本的可视化")
    parser.add_argument("--cover_train", type=float, default=0.55, help="复刻训练用 quantile mask 的覆盖率")
    parser.add_argument("--valid_mask", type=str, default="tgt_conf",
                        choices=["none", "tgt_conf", "tgt_depth_conf"],
                        help="评测时用哪个 GT mask 作为有效像素")
    parser.add_argument("--save_png", action="store_true", help="强制保存 png（默认保存）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # TF32 新 API（顺手把 warning 压掉）
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 1) 动态导入训练脚本（复用其定义）
    train_mod = import_train_module(args.train_script)

    # 2) 读 ckpt（并尽量把保存的 args/cfg 取出来）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", None)
    saved_cfg = ckpt.get("cfg", None)

    # 3) 构建 Dataset
    DatasetCls, ds_name = pick_attr(train_mod, DATASET_CLS_CANDIDATES)
    if DatasetCls is None:
        raise RuntimeError(
            f"在 {args.train_script} 里找不到 Dataset 类，尝试过: {DATASET_CLS_CANDIDATES}\n"
            "请把你的 Dataset 类名加进 DATASET_CLS_CANDIDATES。"
        )

    # 兼容：Dataset 初始化参数不确定，做“尽力而为”的调用
    # 你训练日志里 split=train/val 都能用 split 字符串。
    ds_kwargs = {}
    if isinstance(saved_args, dict):
        # 常见字段：data_root / root / dataset_root
        for k in ["data_root", "root", "dataset_root"]:
            if k in saved_args:
                ds_kwargs["root"] = saved_args[k] if k != "root" else saved_args[k]
                break
    # 不强依赖 root：若你的 Dataset 内部写死路径，这里也能跑
    ds_kwargs["split"] = args.split

    try:
        dataset = DatasetCls(**ds_kwargs)
    except TypeError:
        # 回退：只传 split
        dataset = DatasetCls(split=args.split)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 4) 构建 Model
    build_fn, build_name = pick_attr(train_mod, MODEL_BUILD_FN_CANDIDATES)
    model = None
    model_name = None

    if build_fn is not None:
        # build_model(args) or build_model() 两种都兼容
        try:
            model = build_fn(args)
        except TypeError:
            model = build_fn()
        model_name = build_name
    else:
        ModelCls, model_name = pick_attr(train_mod, MODEL_CLS_CANDIDATES)
        if ModelCls is None:
            raise RuntimeError(
                f"在 {args.train_script} 里找不到 build_model 或 Model 类。\n"
                f"尝试过 build_fn={MODEL_BUILD_FN_CANDIDATES}, model_cls={MODEL_CLS_CANDIDATES}\n"
                "请把你的模型构建函数/类名加进候选列表。"
            )
        # 这里没法猜 __init__ 参数，所以只支持无参构造（你训练脚本里通常是无参/或默认参数）
        model = ModelCls()

    assert model is not None
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 5) 载入权重（优先 EMA）
    def _find_state(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 常见 key
        for k in ["ema", "model_ema", "ema_state", "ema_sd", "ema_state_dict"]:
            if k in d and isinstance(d[k], dict):
                return d[k]
        for k in ["model", "state_dict", "net", "network"]:
            if k in d and isinstance(d[k], dict):
                return d[k]
        return None

    state_ema = _find_state({k: ckpt[k] for k in ckpt.keys() if k in ["ema", "model_ema", "ema_state", "ema_sd", "ema_state_dict"]})
    state_model = _find_state({k: ckpt[k] for k in ckpt.keys() if k in ["model", "state_dict", "net", "network"]})

    chosen = None
    chosen_tag = None
    if args.no_ema:
        chosen = state_model or state_ema
        chosen_tag = "model(forced)"
    elif args.use_ema:
        chosen = state_ema or state_model
        chosen_tag = "ema(forced)" if state_ema is not None else "model(fallback)"
    else:
        chosen = state_ema or state_model
        chosen_tag = "ema" if state_ema is not None else "model"

    if chosen is None:
        raise RuntimeError(f"ckpt 里没找到可用 state_dict key。ckpt keys = {list(ckpt.keys())}")

    missing, unexpected = model.load_state_dict(chosen, strict=False)
    print(f"[load] using={chosen_tag}  missing={len(missing)} unexpected={len(unexpected)}")

    # 6) 评测循环
    t0 = time.time()
    metrics_sum = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "conf_err_corr": 0.0}
    metrics_n = 0

    vis_dir = os.path.join(args.out_dir, f"vis_{args.split}")
    os.makedirs(vis_dir, exist_ok=True)

    def get_valid_mask(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if args.valid_mask == "none":
            return None
        key = args.valid_mask
        if key not in batch:
            return None
        m = batch[key]
        # 兼容 [B,H,W] / [B,1,H,W]
        if m.dim() == 3:
            m = m.unsqueeze(1)
        m = (m > 0.5).float()
        return m

    # 简单的“置信度是否靠谱”指标：corr(conf, -err)
    def conf_error_corr(pred_conf: torch.Tensor, err_map: torch.Tensor, valid: Optional[torch.Tensor]) -> float:
        # pred_conf/err_map: [B,1,H,W]
        if valid is not None:
            m = valid
            pc = pred_conf[m > 0.5].flatten()
            em = err_map[m > 0.5].flatten()
        else:
            pc = pred_conf.flatten()
            em = err_map.flatten()
        if pc.numel() < 100:
            return 0.0
        # Pearson corr(pc, -em)
        pc = pc.float()
        ne = (-em).float()
        pc = pc - pc.mean()
        ne = ne - ne.mean()
        denom = (pc.std(unbiased=False) * ne.std(unbiased=False)).clamp_min(1e-8)
        return float((pc * ne).mean().detach().cpu() / denom.detach().cpu())

    # forward 兼容：你训练脚本里大概率是 model(batch) 或 model(**batch)
    def run_model(m, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 尽量复用训练脚本的 forward_step（如果存在）
        if hasattr(train_mod, "forward_step"):
            out = train_mod.forward_step(m, batch)
            if isinstance(out, dict):
                return out
        # 直接尝试 model(batch)
        try:
            out = m(batch)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
        # 再尝试 model(**batch)（过滤非 Tensor）
        kw = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        out = m(**kw)
        if isinstance(out, dict):
            return out
        raise RuntimeError("无法从模型 forward 得到 dict 输出。请在训练脚本里提供 forward_step(model,batch)->dict。")

    # 输出里 pred/tgt 的 key 兼容（你训练脚本里很可能叫 pred_tgt / pred_img）
    def pick_pred_keys(out: Dict[str, torch.Tensor]) -> Tuple[str, str]:
        pred_candidates = ["pred_tgt", "pred_img", "pred_rgb", "pred"]
        conf_candidates = ["pred_conf", "conf", "pred_confidence"]
        pk = None
        ck = None
        for k in pred_candidates:
            if k in out:
                pk = k
                break
        for k in conf_candidates:
            if k in out:
                ck = k
                break
        if pk is None:
            raise RuntimeError(f"模型输出里找不到 pred 图像 key，out keys={list(out.keys())}")
        if ck is None:
            raise RuntimeError(f"模型输出里找不到 pred_conf key，out keys={list(out.keys())}")
        return pk, ck

    # 可视化保存（需要 torchvision）
    def save_vis(i_global: int, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]):
        if save_image is None or make_grid is None:
            return
        pred_key, conf_key = pick_pred_keys(out)
        tgt = batch["tgt_img"].to(device)
        pred = out[pred_key].to(device)
        conf = out[conf_key].to(device)

        tgt = smart_image_range(tgt)
        pred = smart_image_range(pred)

        # conf 处理到 [0,1]
        if conf.dim() == 3:
            conf = conf.unsqueeze(1)
        conf = conf.float()
        conf = conf.clamp(0.0, 1.0)

        train_m = quantile_train_mask(conf, cover=args.cover_train)
        err = (pred - tgt).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        err_vis = err / (err.max().clamp_min(1e-6))

        # 拼图（tgt / pred / conf / train_mask / err）
        # 把 1 通道图扩成 3 通道方便看
        conf3 = conf.repeat(1, 3, 1, 1)
        m3 = train_m.repeat(1, 3, 1, 1)
        e3 = err_vis.repeat(1, 3, 1, 1)

        grid = make_grid(torch.cat([tgt, pred, conf3, m3, e3], dim=0), nrow=tgt.shape[0])
        save_image(grid, os.path.join(vis_dir, f"{i_global:06d}_tgt_pred_conf_mask_err.png"))

        # 单独保存
        save_image(tgt, os.path.join(vis_dir, f"{i_global:06d}_tgt.png"))
        save_image(pred, os.path.join(vis_dir, f"{i_global:06d}_pred.png"))
        save_image(conf3, os.path.join(vis_dir, f"{i_global:06d}_pred_conf.png"))
        save_image(m3, os.path.join(vis_dir, f"{i_global:06d}_train_mask.png"))
        save_image(e3, os.path.join(vis_dir, f"{i_global:06d}_err.png"))

    i_global = 0
    for batch in loader:
        # batch to device
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = run_model(model, batch)
        pred_key, conf_key = pick_pred_keys(out)

        tgt = smart_image_range(batch["tgt_img"])
        pred = smart_image_range(out[pred_key])

        valid = get_valid_mask(batch)
        l1v = masked_l1(pred, tgt, valid)
        ps = psnr(pred, tgt, valid)
        ss = ssim(pred, tgt, valid)

        # conf-error corr
        conf = out[conf_key]
        if conf.dim() == 3:
            conf = conf.unsqueeze(1)
        conf = conf.float().clamp(0, 1)
        err_map = (pred - tgt).abs().mean(dim=1, keepdim=True)
        corr = conf_error_corr(conf, err_map, valid)

        metrics_sum["l1"] += l1v
        metrics_sum["psnr"] += ps
        metrics_sum["ssim"] += ss
        metrics_sum["conf_err_corr"] += corr
        metrics_n += 1

        if i_global < args.num_vis:
            save_vis(i_global, batch, out)

        i_global += batch["tgt_img"].shape[0]

        if metrics_n % 50 == 0:
            print(f"[{metrics_n:05d}] L1={metrics_sum['l1']/metrics_n:.6f} "
                  f"PSNR={metrics_sum['psnr']/metrics_n:.3f} "
                  f"SSIM={metrics_sum['ssim']/metrics_n:.4f} "
                  f"corr(conf,-err)={metrics_sum['conf_err_corr']/metrics_n:.4f}")

    elapsed = time.time() - t0
    mean_metrics = {k: (v / max(1, metrics_n)) for k, v in metrics_sum.items()}
    mean_metrics["num_batches"] = metrics_n
    mean_metrics["num_samples"] = i_global
    mean_metrics["elapsed_sec"] = elapsed
    mean_metrics["ckpt"] = os.path.abspath(args.ckpt)
    mean_metrics["split"] = args.split
    mean_metrics["valid_mask"] = args.valid_mask

    print("\n=== FINAL ===")
    print(json.dumps(mean_metrics, indent=2, ensure_ascii=False))

    with open(os.path.join(args.out_dir, f"metrics_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(mean_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[done] metrics saved to: {os.path.join(args.out_dir, f'metrics_{args.split}.json')}")
    print(f"[done] vis saved to: {vis_dir}")


if __name__ == "__main__":
    main()
