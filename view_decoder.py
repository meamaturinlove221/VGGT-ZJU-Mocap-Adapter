# view_decoder.py （只放核心的 SimpleViewDecoder）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleViewDecoder(nn.Module):
    """
    一个非常简单的多视角 → 目标视角 Decoder：

    输入:
        src_imgs:  (B, S, 3, H, W)
        src_depth:(B, S, 1, Hd, Wd)  # Hd, Wd 可以和 H, W 不同

    流程:
        1) depth 双线性插值到和 RGB 一样大小
        2) 拼成 (B*S, 4, H, W)，做一个小 U-Net 编码
        3) 在中间 feature 上对 S 个视角取平均，得到 (B, C, h, w)
        4) 解码回 (B, C, H, W)
        5) out_conv 输出 4 通道：3 个 RGB + 1 个 confidence，分别过 sigmoid

    输出:
        rgb:  (B, 3, H, W),   范围 [0,1]
        conf: (B, 1, H, W),   范围 [0,1]
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        C = base_channels

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)  # H,W -> H/2,W/2

        self.enc2 = nn.Sequential(
            nn.Conv2d(C, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)  # H/2,W/2 -> H/4,W/4

        self.enc3 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder ----
        self.up1 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 最终输出 4 通道：RGB(3) + conf(1)
        self.out_conv = nn.Conv2d(C, 4, kernel_size=3, padding=1)

    def forward(self, src_imgs, src_depth):
        """
        src_imgs:  (B, S, 3, H, W)
        src_depth:(B, S, 1, Hd, Wd)
        """
        B, S, _, H, W = src_imgs.shape

        # 1) depth 插值到 (H,W)
        d = src_depth
        if d.shape[-2:] != (H, W):
            d = d.view(B * S, 1, *d.shape[-2:])  # (B*S,1,Hd,Wd)
            d = F.interpolate(
                d, size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            d = d.view(B, S, 1, H, W)  # (B,S,1,H,W)

        # 2) 拼 RGB+depth，展平视角
        x = torch.cat([src_imgs, d], dim=2)    # (B,S,4,H,W)
        x = x.view(B * S, 4, H, W)            # (B*S,4,H,W)

        # 3) Encoder
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)                      # (B*S, C*4, h, w)

        # 4) 在 feature 上对视角求平均
        Bs, C4, h, w = x.shape
        assert Bs == B * S
        x = x.view(B, S, C4, h, w).mean(dim=1)  # (B, C*4, h, w)

        # 5) Decoder
        x = self.up1(x)          # (B, C*2, H/2, W/2)
        x = self.dec1(x)
        x = self.up2(x)          # (B, C, H, W)
        x = self.dec2(x)

        # 6) 输出 4 通道
        x = self.out_conv(x)     # (B, 4, H, W)

        rgb = torch.sigmoid(x[:, :3, :, :])   # (B,3,H,W)
        conf = torch.sigmoid(x[:, 3:4, :, :])  # (B,1,H,W)

        return rgb, conf

class GeomViewDecoder(nn.Module):
    """
    几何增强版 View Decoder：
      - 输入: src_imgs, src_depth, src_depth_conf, src_pointmap
      - 用 depth_conf 做 per-pixel 视角加权
      - U-Net 结构 + skip connection
      - 输出: rgb(3) + conf(1)
    """

    def __init__(
        self,
        base_channels: int = 64,
        use_view_cond: bool = False,
        num_views: int = 0,
        view_dim: int = 16,
        view_affine_strength: float = 1.0,
        rgb_sigmoid_temp: float = 1.0,
        conf_sigmoid_temp: float = 1.0,
        split_conf_head: bool = False,
        logit_clip: float = 0.0,
        conf_bias_init: float = None,
    ):
        super().__init__()
        C = base_channels
        in_channels = 3 + 1 + 1 + 3   # rgb + depth + depth_conf + pointmap
        self.use_view_cond = bool(use_view_cond)
        self.hidden = int(C)
        self.num_views = int(num_views) if num_views is not None else 0
        self.view_affine_strength = float(view_affine_strength)
        self.rgb_sigmoid_temp = float(rgb_sigmoid_temp)
        self.conf_sigmoid_temp = float(conf_sigmoid_temp)
        self.split_conf_head = bool(split_conf_head)
        self.logit_clip = float(logit_clip)
        self.conf_bias_init = conf_bias_init
        self.view_embed = None
        self.view_to_gb = None
        self.view_to_rgb = None
        if self.use_view_cond and int(num_views) > 0:
            self._init_view_cond(int(num_views), int(view_dim))

        # --------- Encoder with skip --------- #
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)  # H,W -> H/2,W/2

        self.enc2 = nn.Sequential(
            nn.Conv2d(C, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 2, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)  # H/2,W/2 -> H/4,W/4

        self.enc3 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 4, C * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # --------- Decoder with skip --------- #
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C * 4, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 2, C * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C * 2, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 最终输出 RGB(3) + conf(1)
        if self.split_conf_head:
            self.out_rgb = nn.Conv2d(C, 3, kernel_size=3, padding=1)
            self.out_conf = nn.Conv2d(C, 1, kernel_size=3, padding=1)
        else:
            self.out_conv = nn.Conv2d(C, 4, kernel_size=3, padding=1)
        self._init_conf_bias(self.conf_bias_init)

    def _init_view_cond(self, num_views: int, view_dim: int):
        self.view_embed = nn.Embedding(num_views, view_dim)
        self.view_to_gb = nn.Linear(view_dim, 2 * self.hidden)
        self.view_to_rgb = nn.Linear(view_dim * 2, 6)
        nn.init.zeros_(self.view_to_gb.weight)
        nn.init.zeros_(self.view_to_gb.bias)
        nn.init.zeros_(self.view_to_rgb.weight)
        nn.init.zeros_(self.view_to_rgb.bias)

    def _init_conf_bias(self, conf_bias_init: float):
        if conf_bias_init is None:
            return
        val = float(conf_bias_init)
        # Interpret init as target prob or logit before sigmoid; compensate for temp
        t_conf = self.conf_sigmoid_temp if self.conf_sigmoid_temp > 0 else 1.0
        if 0.0 < val < 1.0:
            val = math.log(val / (1.0 - val))
        val = val * float(t_conf)
        with torch.no_grad():
            if self.split_conf_head:
                if hasattr(self, "out_conf") and self.out_conf.bias is not None:
                    if self.out_conf.bias.numel() >= 1:
                        self.out_conf.bias[0] = val
            else:
                if hasattr(self, "out_conv") and self.out_conv.bias is not None:
                    if self.out_conv.bias.numel() >= 4:
                        self.out_conv.bias[3] = val

    def _apply_view_cond(self, feat: torch.Tensor, tgt_vid: torch.Tensor):
        if (not self.use_view_cond) or tgt_vid is None:
            return feat
        if self.view_embed is None or self.view_to_gb is None:
            return feat
        vid = tgt_vid.long().view(-1)
        e = self.view_embed(vid)
        gb = self.view_to_gb(e)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feat * (1.0 + gamma) + beta

    def _apply_view_affine(self, rgb: torch.Tensor, tgt_vid: torch.Tensor, src_vids: torch.Tensor):
        if (not self.use_view_cond) or tgt_vid is None or src_vids is None:
            return rgb
        if self.view_embed is None or self.view_to_rgb is None:
            return rgb
        if self.num_views <= 0:
            return rgb

        B = int(rgb.shape[0])
        tgt = tgt_vid.view(-1)
        if tgt.numel() != B:
            tgt = tgt[:B]
        tgt = tgt.long().clamp(min=0, max=self.num_views - 1)

        src = src_vids
        if src.dim() == 1:
            if src.numel() == B:
                src = src.view(B, 1)
            else:
                src = src.view(1, -1)
        src = src.long().clamp(min=0, max=self.num_views - 1)

        e_t = self.view_embed(tgt)
        e_s = self.view_embed(src).mean(dim=1)
        z = torch.cat([e_t, e_s], dim=1)

        out = self.view_to_rgb(z)
        if abs(self.view_affine_strength - 1.0) > 1e-6:
            out = out * self.view_affine_strength
        dg = out[:, :3].view(B, 3, 1, 1)
        db = out[:, 3:].view(B, 3, 1, 1)
        return rgb * (1.0 + dg) + db

    def forward(
        self,
        src_imgs,
        src_depth,
        src_depth_conf,
        src_pointmap,
        tgt_vid=None,
        src_vids=None,
        return_logits: bool = False,
    ):
        """
        src_imgs:       (B,S,3,H,W)
        src_depth:      (B,S,1,Hd,Wd)
        src_depth_conf: (B,S,1,Hd,Wd)
        src_pointmap:   (B,S,3,Hd,Wd)
        """
        B, S, _, H, W = src_imgs.shape

        # ---- 1) 把 depth / conf / pointmap 插到 RGB 尺寸 ---- #
        d = src_depth
        if d.shape[-2:] != (H, W):
            d = d.view(B * S, 1, *d.shape[-2:])
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            d = d.view(B, S, 1, H, W)

        conf = src_depth_conf
        if conf.shape[-2:] != (H, W):
            conf = conf.view(B * S, 1, *conf.shape[-2:])
            conf = F.interpolate(conf, size=(H, W), mode="bilinear", align_corners=False)
            conf = conf.view(B, S, 1, H, W)

        pm = src_pointmap
        if pm.shape[-2:] != (H, W):
            pm = pm.view(B * S, 3, *pm.shape[-2:])
            pm = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)
            pm = pm.view(B, S, 3, H, W)

        # ---- 2) 拼通道 & 展平视角 ---- #
        x = torch.cat([src_imgs, d, conf, pm], dim=2)   # (B,S,8,H,W)
        x = x.view(B * S, x.shape[2], H, W)             # (B*S,8,H,W)

        # ---- 3) Encoder with skip ---- #
        x1 = self.enc1(x)          # (B*S, C,   H,   W)
        x2 = self.down1(x1)        # (B*S, C,   H/2, W/2)
        x2 = self.enc2(x2)         # (B*S, 2C,  H/2, W/2)
        x3 = self.down2(x2)        # (B*S, 2C,  H/4, W/4)
        x3 = self.enc3(x3)         # (B*S, 4C,  H/4, W/4)

        # ---- 4) 用 depth_conf 做多视角加权聚合 ---- #
        Bs, C4, h, w = x3.shape
        assert Bs == B * S
        x3 = x3.view(B, S, C4, h, w)  # (B,S,4C,h,w)

        # 把 conf 下采样到 h,w 做权重
        conf_b = conf
        if conf_b.shape[-2:] != (h, w):
            conf_b = conf_b.view(B * S, 1, H, W)
            conf_b = F.interpolate(conf_b, size=(h, w), mode="bilinear", align_corners=False)
            conf_b = conf_b.view(B, S, 1, h, w)         # (B,S,1,h,w)

        weights = conf_b + 1e-6
        weights = weights / weights.sum(dim=1, keepdim=True)  # 视角维度归一化

        x_agg = (x3 * weights).sum(dim=1)               # (B,4C,h,w)

        # skip 也做一遍加权聚合
        x1 = x1.view(B, S, -1, H, W)                    # (B,S,C,H,W)
        x2 = x2.view(B, S, -1, H // 2, W // 2)          # (B,S,2C,H/2,W/2)

        # x1 对应的 conf 在 H,W 上已经有了
        w1 = conf + 1e-6                                # (B,S,1,H,W)
        w1 = w1 / w1.sum(dim=1, keepdim=True)
        x1_agg = (x1 * w1).sum(dim=1)                   # (B,C,H,W)
        x1_agg = self._apply_view_cond(x1_agg, tgt_vid)

        # x2 对应再下采样 conf
        conf2 = conf
        if conf2.shape[-2:] != (H // 2, W // 2):
            conf2 = conf2.view(B * S, 1, H, W)
            conf2 = F.interpolate(conf2, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
            conf2 = conf2.view(B, S, 1, H // 2, W // 2)

        w2 = conf2 + 1e-6
        w2 = w2 / w2.sum(dim=1, keepdim=True)
        x2_agg = (x2 * w2).sum(dim=1)                   # (B,2C,H/2,W/2)

        # ---- 5) Decoder with skip ---- #
        u2 = self.up2(x_agg)                            # (B,2C,H/2,W/2)
        u2 = torch.cat([u2, x2_agg], dim=1)             # (B,4C,H/2,W/2)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)                               # (B,C,H,W)
        u1 = torch.cat([u1, x1_agg], dim=1)             # (B,2C,H,W)
        u1 = self.dec1(u1)

        if self.split_conf_head:
            rgb_logits = self.out_rgb(u1)               # (B,3,H,W)
            conf_logits = self.out_conf(u1)             # (B,1,H,W)
            if self.logit_clip and self.logit_clip > 0:
                rgb_logits = rgb_logits.clamp(
                    min=-self.logit_clip, max=self.logit_clip)
                conf_logits = conf_logits.clamp(
                    min=-self.logit_clip, max=self.logit_clip)
        else:
            out = self.out_conv(u1)                     # (B,4,H,W)
            if self.logit_clip and self.logit_clip > 0:
                out = out.clamp(min=-self.logit_clip, max=self.logit_clip)
            rgb_logits = out[:, :3, :, :]
            conf_logits = out[:, 3:4, :, :]
        t_rgb = self.rgb_sigmoid_temp if self.rgb_sigmoid_temp > 0 else 1.0
        t_conf = self.conf_sigmoid_temp if self.conf_sigmoid_temp > 0 else 1.0
        rgb = torch.sigmoid(rgb_logits / t_rgb)         # (B,3,H,W)
        conf_out = torch.sigmoid(conf_logits / t_conf)  # (B,1,H,W)

        rgb = self._apply_view_affine(rgb, tgt_vid, src_vids)

        if return_logits:
            return rgb, conf_out, (rgb_logits, conf_logits)
        return rgb, conf_out
# ============================================================
# Ablation 版：在 GeomViewDecoder 外面包一层
#   - 加一个 global skip：把 src_imgs 拼出一个“参考图像”，做 1x1 conv 后加到输出上
#   - 再加一个 1x1 conv 作为色调微调层（可选）
# 使用方式：
#   from view_decoder import GeomViewDecoderAblation
#   model = GeomViewDecoderAblation(use_global_skip=True, use_tone_conv=True)
# ============================================================

class GeomViewDecoderAblation(nn.Module):
    """
    在原来的 GeomViewDecoder 外面包一层做结构 ablation：

    1) Global skip：
        - 从 src_imgs 得到一个参考 RGB（比如所有视角的平均）
        - 过一个 1x1 Conv 做到同一 feature 空间
        - 和 core 的输出 rgb 做相加

    2) Tone conv：
        - 在 final rgb 上再过一个 1x1 Conv，学习“整体色调/对比度微调”

    注意：
        - 不改 core 的结构，所以你现有的 GeomViewDecoder 代码可以原样保留
        - 这个类是“新模型”，建议重新训练，而不是直接加载老 ckpt
    """

    def __init__(
        self,
        base_channels: int = 64,
        use_global_skip: bool = True,
        use_tone_conv: bool = True,
    ):
        super().__init__()
        # 原来的几何版 decoder 当成一个子模块
        self.core = GeomViewDecoder(base_channels=base_channels)

        self.use_global_skip = use_global_skip
        self.use_tone_conv = use_tone_conv
        # 让网络自己决定“要不要信 skip”，并且一开始默认不信（=0），避免它立刻走歪路
        self.skip_alpha = nn.Parameter(torch.tensor(-4.0))  # sigmoid(-4)≈0.018，初始几乎关掉


        if self.use_global_skip:
            # 把参考 RGB 映射到和输出同一尺度，方便相加
            self.skip_proj = nn.Conv2d(3, 3, kernel_size=1)

        if self.use_tone_conv:
            # 只看 final rgb，再做一次 1x1 卷积当“色调微调层”
            self.tone_conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, src_imgs, src_depth, src_depth_conf, src_pointmap):
        """
        参数和 GeomViewDecoder 一模一样，方便直接替换：

        src_imgs:        (B, S, 3, H, W)
        src_depth:       (B, S, 1, Hd, Wd)
        src_depth_conf:  (B, S, 1, Hd, Wd)
        src_pointmap:    (B, S, 3, Hd, Wd)
        """
        # 先走原来的几何 U-Net
        rgb, conf = self.core(src_imgs, src_depth, src_depth_conf, src_pointmap)
        # rgb: (B,3,H,W), conf: (B,1,H,W)，一般已经是 [0,1] 左右

        # ---------- 1) Global skip from src_imgs ----------
        if self.use_global_skip:
            # 很粗暴地：所有视角取平均，得到一个“参考外观”
            # 你也可以改成取第 0 个视角 / 中间视角，看你数据习惯
            # src_imgs: (B,S,3,H,W)
            ref_rgb = src_imgs[:, 0]   # (B,3,H,W) 只取一个视角
            ref_rgb = self.skip_proj(ref_rgb)  # (B,3,H,W)

            alpha = torch.sigmoid(self.skip_alpha)  # (标量)
            rgb = rgb + alpha * ref_rgb


        # ---------- 2) Tone conv（色调微调） ----------
        if self.use_tone_conv:
            # 在当前 rgb 的基础上再预测一个 Δrgb
            delta = self.tone_conv(rgb)
            rgb = rgb + delta

        # 最后一手 clamp，避免数值飞出 [0,1]
        rgb = rgb.clamp(0.0, 1.0)

        return rgb, conf
