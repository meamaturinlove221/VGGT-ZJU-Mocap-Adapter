import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from zju_dataset_view import ZJUViewSynthDataset
from view_decoder import GeomViewDecoder
from train_view_decoder import ZJU_ROOT


# ========= 配置 =========

SEQ_NAMES = [
    "CoreView_390",
]

# 换成你当前想看的 ckpt
CKPT_PATH = r"F:\vggt\ckpts\viewdec_best_epoch02.pth"

BATCH_SIZE = 8
NUM_WORKERS = 8
USE_FP16 = True   # 先关掉半精度排查（True/False 自己切）

# 输出目录
OUT_NAIVE = "view_decoder_vis_naive"   # 只用 pred_rgb 的原始预测
OUT_RAW = "view_decoder_vis_raw"       # pred_rgb * conf^gamma （主要用来观察 conf 有没有用）
OUT_CAT = "view_decoder_vis"           # 可视化：norm 后的 rgb 预测 | GT
OUT_GT = "view_decoder_vis_gt"         # 目标图
OUT_NORM = "view_decoder_vis_norm"     # 对比度放大版（naive/weighted 各一份）
OUT_CONF = "view_decoder_vis_conf"     # 新增：单独保存 conf 的灰度可视化

os.makedirs(OUT_NAIVE, exist_ok=True)
os.makedirs(OUT_RAW,   exist_ok=True)
os.makedirs(OUT_CAT,   exist_ok=True)
os.makedirs(OUT_GT,    exist_ok=True)
os.makedirs(OUT_NORM,  exist_ok=True)
os.makedirs(OUT_CONF,  exist_ok=True)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """默认假设 t 已经在 [0,1]，直接乘 255 存。"""
    t = t.detach().float().cpu().clamp(0.0, 1.0)
    t = (t * 255.0).to(torch.uint8)
    return t.permute(1, 2, 0).numpy()


def tensor_to_img_norm01(t: torch.Tensor) -> np.ndarray:
    """
    对当前这帧做自适应归一化：
      - 把 min 映射到 0
      - 把 max 映射到 1
    用来看“有没有结构”，不在乎绝对亮度。
    """
    t = t.detach().float().cpu()
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max - t_min < 1e-6:
        t = torch.zeros_like(t)
    else:
        t = (t - t_min) / (t_max - t_min)
    t = (t * 255.0).clamp(0, 255).to(torch.uint8)
    return t.permute(1, 2, 0).numpy()


def conf_to_img(t: torch.Tensor) -> np.ndarray:
    """
    把单通道 conf [H,W] 映射到 0~255 灰度，并复制成 3 通道方便保存。
    """
    t = t.detach().float().cpu().clamp(0.0, 1.0)
    t = (t * 255.0).to(torch.uint8)
    t_np = t.numpy()          # [H,W]
    conf_vis = np.stack([t_np] * 3, axis=-1)  # [H,W,3]
    return conf_vis


def main():
    torch.backends.cudnn.benchmark = True

    # ---------- 1) Dataset / Loader ----------
    dataset = ZJUViewSynthDataset(
        root=ZJU_ROOT,
        seq_names=SEQ_NAMES,
        num_src_views=3,
        frame_subsample=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------- 2) 模型 & checkpoint ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = GeomViewDecoder().to(device)

    print(">>> Using checkpoint:", CKPT_PATH)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    gamma = 3.0  # conf 的幂次，只给 debug 用
    frame_counter = 0

    # ---------- 3) 推理 ----------
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            src_imgs = batch["src_imgs"].to(device)
            src_depth = batch["src_depth"].to(device)
            src_depth_conf = batch["src_depth_conf"].to(device)
            src_pointmap = batch["src_pointmap"].to(device)
            tgt_img = batch["tgt_img"].to(device)

            # 可选的 fp16 推理
            if USE_FP16 and device == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    pred_rgb, pred_conf = model(
                        src_imgs, src_depth, src_depth_conf, src_pointmap
                    )
            else:
                pred_rgb, pred_conf = model(
                    src_imgs, src_depth, src_depth_conf, src_pointmap
                )

            pred_rgb = pred_rgb.float()
            pred_conf = pred_conf.float()
            tgt_img = tgt_img.float()

            # 打印前几个 batch 的取值范围
            if batch_idx < 3:
                pr_min = float(pred_rgb.min().item())
                pr_max = float(pred_rgb.max().item())
                pc_min = float(pred_conf.min().item())
                pc_max = float(pred_conf.max().item())
                tg_min = float(tgt_img.min().item())
                tg_max = float(tgt_img.max().item())
                print(
                    f"[debug infer] batch {batch_idx} "
                    f"pred_rgb=({pr_min:.4f},{pr_max:.4f}), "
                    f"pred_conf=({pc_min:.4f},{pc_max:.4f}), "
                    f"tgt_img=({tg_min:.4f},{tg_max:.4f})"
                )

            B = pred_rgb.shape[0]
            for b in range(B):
                frame_id = frame_counter
                frame_counter += 1
                fname = f"{frame_id:05d}.png"

                # ---------- 1) 纯预测（不乘 conf） ----------
                # 原始范围的预测（会偏暗，但保留绝对值）
                pred_naive_orig = tensor_to_img(pred_rgb[b])

                # 自适应归一化后的预测（用来「看清」结构）
                pred_naive_norm = tensor_to_img_norm01(pred_rgb[b])

                # ---------- 2) conf 加权预测（仅做对比 / debug） ----------
                conf_w = pred_conf[b] ** gamma             # (1,H,W)
                pred_rgb_weighted = pred_rgb[b] * conf_w   # (3,H,W)

                pred_weighted = tensor_to_img(pred_rgb_weighted)
                pred_weighted_norm = tensor_to_img_norm01(pred_rgb_weighted)

                # ---------- 3) GT ----------
                tgt_img_np = tensor_to_img(tgt_img[b])

                # === 额外：conf map & error map ===
                # 1) conf 灰度图
                conf_map = pred_conf[b]  # (1,H,W)
                conf_map_3c = conf_map.repeat(3, 1, 1)  # 变成 (3,H,W) 好存
                conf_img = tensor_to_img(conf_map_3c)

                # 2) 误差图 |pred - gt|
                # (1,H,W)，对通道取平均误差
                err = torch.abs(pred_rgb[b] - tgt_img[b]
                                ).mean(dim=0, keepdim=True)
                err_3c = err.repeat(3, 1, 1)
                err_img_norm = tensor_to_img_norm01(err_3c)  # 自适应到 0-255 看结构

                # ---------- 4) 保存各类输出 ----------

                # 4.0 保存 conf 灰度图（单独观察置信度分布）
                # pred_conf[b]: [1,H,W] -> 取第 0 通道
                conf_vis = conf_to_img(pred_conf[b, 0])
                Image.fromarray(conf_vis).save(
                    os.path.join(OUT_CONF, fname)
                )

                # 4.1 只用 pred_rgb 的预测
                Image.fromarray(pred_naive_orig).save(
                    os.path.join(OUT_NAIVE, fname)
                )

                # 4.2 conf 加权后的预测（主要用于观察 conf 是否全 0）
                Image.fromarray(pred_weighted).save(
                    os.path.join(OUT_RAW, fname)
                )

                # 4.3 GT
                Image.fromarray(tgt_img_np).save(
                    os.path.join(OUT_GT, fname)
                )

                # 4.4 对比度放大版（debug）
                Image.fromarray(pred_naive_norm).save(
                    os.path.join(OUT_NORM, f"naive_norm_{fname}")
                )
                Image.fromarray(pred_weighted_norm).save(
                    os.path.join(OUT_NORM, f"weighted_norm_{fname}")
                )

                # 4.5 拼接图：**只用不乘 conf 的 norm 预测 | GT**
                cat = np.concatenate([pred_naive_norm, tgt_img_np], axis=1)
                Image.fromarray(cat).save(
                    os.path.join(OUT_CAT, fname)
                )

                # 存到 OUT_NORM 或新建一个 OUT_ERR 文件夹
                err_dir = "view_decoder_vis_err"
                os.makedirs(err_dir, exist_ok=True)

                Image.fromarray(conf_img).save(
                    os.path.join(OUT_NORM, f"conf_{fname}")
                )
                Image.fromarray(err_img_norm).save(
                    os.path.join(err_dir, f"err_{fname}")
                )

            # 如果只想先看前 N 帧，可以解开这个：
            # if frame_counter >= 200:
            #     break


if __name__ == "__main__":
    main()
