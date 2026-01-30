import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from zju_dataset_view import ZJUViewSynthDataset
from view_decoder import GeomViewDecoder


# ====== 你本地 ZJU 数据路径 ======
ZJU_ROOT = r"F:\datasets\ZJU_MoCap\data\zju_mocap"


class VGGPerceptualLoss(nn.Module):
    """
    一个简单的感知损失：用 VGG16 前几层特征做 L1。
    只用来度量纹理/结构，不参与训练。
    """

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights="DEFAULT").features
        # 取到 relu3_3 左右，足够 low-level 纹理
        self.slice = nn.Sequential(*list(vgg.children())[:16])
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        # pred, target: (B,3,H,W), 范围 [0,1]
        feat_p = self.slice(pred)
        feat_t = self.slice(target)
        return F.l1_loss(feat_p, feat_t)


def save_debug(pred_rgb, tgt, step, out_dir="debug_viewdec"):
    """
    把当前 batch 中第 1 张样本的预测 / GT 存成 png，方便肉眼看效果。

    pred_rgb: (B, 3, H, W)
    tgt:      (B, 3, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)

    pred = pred_rgb.detach().cpu().clamp(0, 1)[0]  # (3,H,W)
    tgt = tgt.detach().cpu().clamp(0, 1)[0]

    pred_img = (pred * 255).permute(1, 2, 0).byte().numpy()  # (H,W,3) uint8
    tgt_img = (tgt * 255).permute(1, 2, 0).byte().numpy()

    Image.fromarray(pred_img).save(
        os.path.join(out_dir, f"pred_step{step:06d}.png"))
    Image.fromarray(tgt_img).save(
        os.path.join(out_dir, f"tgt_step{step:06d}.png"))

    cat = np.concatenate([pred_img, tgt_img], axis=1)
    Image.fromarray(cat).save(os.path.join(out_dir, f"cat_step{step:06d}.png"))


def main():
    # ---------- 0) 一些配置 ----------
    seq_names = [
        "CoreView_390",
        # 想多训几个就在这里加，例如：
        # "CoreView_377",
        # "CoreView_386",
    ]

    batch_size = 8
    num_workers = 8

    # 感知损失 / 亮度 / 对比度 / conf 的权重
    lambda_percep = 0.05       # 比之前 0.1 更弱一些
    lambda_conf = 0.01
    lambda_bright = 0.5        # 强迫整体亮度贴近 GT
    lambda_contrast = 0.5      # 再约束整体对比度（std）

    # 训练/finetune 相关
    num_epochs = 130            # 这一轮主要是 fine-tune
    base_lr = 1e-4
    fine_tune_lr = 1e-5

    # 如果想从 epoch5 的 best ckpt 继续训，就把路径填在这里
    RESUME_CKPT = r"ckpt\viewdec_best_epoch01.pth"   # 不想 resume 就设成 "" 或 None
    # RESUME_CKPT = r""
    # ---------- 1) 构建数据集 & DataLoader ----------
    full_dataset = ZJUViewSynthDataset(
        root=ZJU_ROOT,
        seq_names=seq_names,
        num_src_views=3,
        frame_subsample=1,
    )

    # 做一个稳定的 train/val 划分（例如 90% / 10%）
    val_ratio = 0.1
    num_total = len(full_dataset)
    num_val = max(1, int(num_total * val_ratio))
    num_train = num_total - num_val

    g = torch.Generator().manual_seed(0)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [num_train, num_val],
        generator=g,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[info] total samples = {num_total} "
          f"(train = {len(train_dataset)}, val = {len(val_dataset)})")
    print(f"[info] num_train_batches per epoch = {len(train_loader)}")

    # ---------- 2) 模型 & 优化器 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model = GeomViewDecoder().to(device)

    percep_loss_fn = VGGPerceptualLoss().to(device)

    # 根据是否 resume 来决定学习率
    if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(state)
        lr = fine_tune_lr
        print(f"[info] resume from {RESUME_CKPT}, use lr={lr}")
    else:
        lr = base_lr
        if RESUME_CKPT:
            print(
                f"[warn] RESUME_CKPT = {RESUME_CKPT} not found, train from scratch.")
        else:
            print("[info] train from scratch (no resume ckpt).")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # 学习率调度：当 val loss 不再下降时减半 lr
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        # 你的 torch 没有 verbose 这个参数，所以先去掉
    )

    os.makedirs("ckpt", exist_ok=True)
    best_val_recon = float("inf")
    global_step = 0
    min_improve = 1e-4
    epochs_no_improve = 0
    max_patience = 6   # 如果你想 early stop，可以用这个

    # ---------- 3) 训练循环 ----------
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            src_imgs = batch["src_imgs"].to(device)          # (B,S,3,H,W)
            src_depth = batch["src_depth"].to(device)        # (B,S,1,Hd,Wd)
            src_depth_conf = batch["src_depth_conf"].to(
                device)  # (B,S,1,Hd,Wd)
            src_pointmap = batch["src_pointmap"].to(device)  # (B,S,3,Hd,Wd)
            tgt_img = batch["tgt_img"].to(device)            # (B,3,H,W)

            optimizer.zero_grad()

            pred_rgb, pred_conf = model(
                src_imgs, src_depth, src_depth_conf, src_pointmap
            )  # (B,3,H,W), (B,1,H,W)

            # 1) 重建误差：对 RGB 做 L1
            recon_loss = F.l1_loss(pred_rgb, tgt_img)

            # 2) 感知损失（在下采样后的分辨率上算）
            vgg_h, vgg_w = 256, 256
            pred_small = F.interpolate(
                pred_rgb, size=(vgg_h, vgg_w),
                mode="bilinear", align_corners=False
            )
            tgt_small = F.interpolate(
                tgt_img, size=(vgg_h, vgg_w),
                mode="bilinear", align_corners=False
            )
            loss_percep = percep_loss_fn(pred_small, tgt_small)

            # 3) confidence 正则：惩罚 0.5 一类中间值，鼓励靠近 0 或 1
            # 惩罚过小的 conf（比如小于 0.1）
            conf_reg = F.relu(0.1 - pred_conf).mean()
            lambda_conf = 1e-3

            # 4) 亮度 / 对比度约束：对齐“整张图的平均亮度 + std”
            mean_pred = pred_rgb.mean()
            mean_tgt = tgt_img.mean()
            brightness_loss = torch.abs(mean_pred - mean_tgt)

            std_pred = pred_rgb.std()
            std_tgt = tgt_img.std()
            contrast_loss = torch.abs(std_pred - std_tgt)

            # 5) 汇总总 loss
            loss = (
                recon_loss
                + lambda_conf * conf_reg
                + lambda_percep * loss_percep
                + lambda_bright * brightness_loss
                + lambda_contrast * contrast_loss
            )

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_train_loss += loss.item()

            # 打一点 log
            if global_step % 10 == 0:
                print(
                    f"[epoch {epoch:02d} step {global_step:06d} (inner {step:04d})] "
                    f"loss={loss.item():.4f}"
                )

            if global_step % 200 == 0:
                print(
                    f"[epoch {epoch:02d} step {global_step:06d}] "
                    f"loss = {loss.item():.4f} "
                    f"(recon={recon_loss.item():.4f}, "
                    f"percep={loss_percep.item():.4f}, "
                    f"bright={brightness_loss.item():.4f}, "
                    f"contrast={contrast_loss.item():.4f}, "
                    f"conf_reg={conf_reg.item():.4f})"
                )
                pr_min = float(pred_rgb.min().item())
                pr_max = float(pred_rgb.max().item())
                tg_min = float(tgt_img.min().item())
                tg_max = float(tgt_img.max().item())
                print(
                    f"[debug] step {global_step} "
                    f"pred range=({pr_min:.4f}, {pr_max:.4f}), "
                    f"tgt range=({tg_min:.4f}, {tg_max:.4f})"
                )

                save_debug(pred_rgb, tgt_img, global_step,
                           out_dir="debug_viewdec")

        # ---------- 4) 每个 epoch 结束后：统计 train / val ----------
        mean_train_loss = epoch_train_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch:02d}] mean train loss = {mean_train_loss:.4f}")

        # --- 在 val 集上算 L1 作为指标 ---
        model.eval()
        val_recon_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                src_imgs = batch["src_imgs"].to(device)
                src_depth = batch["src_depth"].to(device)
                src_depth_conf = batch["src_depth_conf"].to(device)
                src_pointmap = batch["src_pointmap"].to(device)
                tgt_img = batch["tgt_img"].to(device)

                pred_rgb, pred_conf = model(
                    src_imgs, src_depth, src_depth_conf, src_pointmap
                )

                recon_val = F.l1_loss(pred_rgb, tgt_img)
                bs = tgt_img.size(0)
                val_recon_sum += recon_val.item() * bs
                val_count += bs

        mean_val_recon = val_recon_sum / max(1, val_count)
        print(f"[Epoch {epoch:02d}] mean val L1 = {mean_val_recon:.4f}")

        # 学习率调度器：看 val L1 是否还在下降
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(mean_val_recon)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"[scheduler] lr reduced: {prev_lr:.6g} -> {new_lr:.6g}")

        # ---------- 5) checkpoint 逻辑：看 val L1 有没有明显下降 ----------
        if best_val_recon - mean_val_recon > min_improve:
            best_val_recon = mean_val_recon
            epochs_no_improve = 0
            ckpt_path = os.path.join(
                "ckpt", f"viewdec_best_epoch{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> val L1 明显下降，保存新 best: {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> val L1 没明显下降 (no_improve = {epochs_no_improve})")

            # 想启用 early stop 的话可以解开：
            # if epochs_no_improve >= max_patience:
            #     print("  -> 连续几轮没变好，认为基本收敛，可以停了")
            #     break


if __name__ == "__main__":
    main()
