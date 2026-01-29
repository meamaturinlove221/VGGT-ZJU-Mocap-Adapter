# view_decoder_ablation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from view_decoder import GeomViewDecoder


class ViewResidualAffine(nn.Module):
    """
    Residual affine adapter:
      pred' = pred * (1 + ds) + db
    Initialized to identity (ds=0, db=0) while keeping gradients alive.
    """

    def __init__(
        self,
        num_views: int,
        view_dim: int = 32,
        hidden: int = 64,
        mode: str = "tgt",
    ):
        super().__init__()
        self.mode = str(mode)
        self.view_emb = nn.Embedding(int(num_views), int(view_dim))
        in_dim = int(view_dim)
        if self.mode == "tgt_src_mean":
            in_dim = int(view_dim) * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 6),
        )

        nn.init.zeros_(self.view_emb.weight)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tgt_vid: torch.Tensor, src_vids: torch.Tensor = None):
        if tgt_vid.dim() > 1:
            tgt_vid = tgt_vid.view(-1)
        v_t = self.view_emb(tgt_vid.long())
        if self.mode == "tgt_src_mean":
            if src_vids is not None:
                if src_vids.dim() == 1:
                    src_vids = src_vids.view(1, -1)
                v_s = self.view_emb(src_vids.long()).mean(dim=1)
                v = torch.cat([v_t, v_s], dim=-1)
            else:
                v = v_t
        elif self.mode == "tgt":
            v = v_t
        else:
            raise ValueError(f"Unknown view_cond_mode={self.mode}")

        delta = self.mlp(v)
        ds = delta[:, 0:3]
        db = delta[:, 3:6]
        return ds, db


class GeomViewDecoderAblation(nn.Module):
    """
    在不改动原始 view_decoder.py 的前提下，
    给 GeomViewDecoder 包一层可控的 RGB skip（ablation 用）。

    你之前问的：
        ref_rgb = src_imgs.mean(dim=1)
    就在这里（当 ref_mode='mean' 时）。

    参数：
      - ref_mode:
          'mean'  -> ref_rgb = src_imgs.mean(dim=1)   (B,3,H,W)
          'first' -> ref_rgb = src_imgs[:,0]          (B,3,H,W)
          'random'-> 每个 batch 随机选一个视角当 ref
      - use_conf_gate:
          True  -> skip 乘以 pred_conf（通常前景更高，背景更低）
      - use_tone:
          True  -> 额外一个 1x1 conv 做轻微颜色/亮度校正（残差形式）
    """

    def __init__(
        self,
        ref_mode: str = "first",
        use_conf_gate: bool = True,
        use_tone: bool = True,
        init_alpha: float = 0.12,   # 初始 skip 强度（sigmoid 后大概 0.1 左右）
        use_view_cond: bool = False,
        num_views: int = 0,
        view_dim: int = 16,
        view_affine_strength: float = 1.0,
        view_cond_mode: str = "tgt",
        rgb_sigmoid_temp: float = 1.0,
        conf_sigmoid_temp: float = 1.0,
        split_conf_head: bool = False,
        logit_clip: float = 0.0,
        conf_gate_detach: bool = False,
        conf_gate_floor: float = 0.0,
        conf_bias_init: float = None,
    ):
        super().__init__()

        self.core = GeomViewDecoder(
            use_view_cond=False,
            num_views=0,
            view_dim=int(view_dim),
            view_affine_strength=float(view_affine_strength),
            rgb_sigmoid_temp=float(rgb_sigmoid_temp),
            conf_sigmoid_temp=float(conf_sigmoid_temp),
            split_conf_head=bool(split_conf_head),
            conf_bias_init=conf_bias_init,
            logit_clip=float(logit_clip),
        )
        self.ref_mode = ref_mode
        self.use_conf_gate = use_conf_gate
        self.use_tone = use_tone
        self.use_view_cond = bool(use_view_cond)
        self.num_views = int(num_views) if num_views is not None else 0
        self.view_dim = int(view_dim)
        self.view_affine_strength = float(view_affine_strength)
        self.view_cond_mode = str(view_cond_mode)
        self.conf_gate_detach = bool(conf_gate_detach)
        self.conf_gate_floor = float(conf_gate_floor)
        if self.conf_gate_floor < 0.0:
            self.conf_gate_floor = 0.0
        if self.conf_gate_floor > 1.0:
            self.conf_gate_floor = 1.0

        # 把 ref_rgb 先过一个 1x1，给模型一点“学着用 skip”的自由度
        self.skip_proj = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        self._init_1x1_identity(self.skip_proj)

        # 用 logit 参数化 alpha：alpha = sigmoid(alpha_logit)
        # 这样 alpha 永远在 (0,1)，更稳，不会越训越爆
        init_alpha = float(init_alpha)
        init_alpha = max(1e-4, min(0.999, init_alpha))
        init_logit = torch.log(torch.tensor(init_alpha / (1.0 - init_alpha)))
        self.alpha_logit = nn.Parameter(init_logit.clone().detach())

        # tone：残差形式（默认很小/接近 0），避免一上来把颜色洗坏
        if self.use_tone:
            self.tone_conv = nn.Conv2d(3, 3, kernel_size=1, bias=True)
            nn.init.zeros_(self.tone_conv.weight)
            nn.init.zeros_(self.tone_conv.bias)
        else:
            self.tone_conv = None

        if self.use_view_cond:
            if self.num_views <= 0:
                raise ValueError(
                    "use_view_cond=True but num_views<=0. Please pass num_views or auto-set it from dataset."
                )
            self.view_affine = ViewResidualAffine(
                num_views=self.num_views,
                view_dim=self.view_dim,
                hidden=64,
                mode=self.view_cond_mode,
            )
        else:
            self.view_affine = None

    @staticmethod
    def _init_1x1_identity(conv: nn.Conv2d):
        """把 3->3 的 1x1 conv 初始化成近似 identity。"""
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        with torch.no_grad():
            # weight: (out=3, in=3, 1, 1)
            for i in range(min(conv.out_channels, conv.in_channels)):
                conv.weight[i, i, 0, 0] = 1.0

    def get_alpha(self) -> torch.Tensor:
        """返回当前 alpha（标量 Tensor，范围 0~1）。"""
        return torch.sigmoid(self.alpha_logit)

    def pick_ref_rgb(self, src_imgs: torch.Tensor) -> torch.Tensor:
        """
        src_imgs: (B,S,3,H,W) -> ref_rgb: (B,3,H,W)
        """
        if self.ref_mode == "mean":
            # 你问的那句，就在这里 ✅
            ref_rgb = src_imgs.mean(dim=1)
        elif self.ref_mode == "first":
            ref_rgb = src_imgs[:, 0]
        elif self.ref_mode == "random":
            B, S = src_imgs.shape[0], src_imgs.shape[1]
            idx = torch.randint(low=0, high=S, size=(B,),
                                device=src_imgs.device)
            ref_rgb = src_imgs[torch.arange(B, device=src_imgs.device), idx]
        else:
            raise ValueError(f"Unknown ref_mode={self.ref_mode}")
        return ref_rgb

    def forward(
        self,
        src_imgs: torch.Tensor,
        src_depth: torch.Tensor,
        src_depth_conf: torch.Tensor,
        src_pointmap: torch.Tensor,
        tgt_vid: torch.Tensor = None,
        return_aux: bool = False,
        src_vids: torch.Tensor = None,
        use_conf_gate_override: bool = None,
        conf_gate_strength: float = None,
    ):
        """
        返回：
          pred_rgb: (B,3,H,W) in [0,1]
          pred_conf:(B,1,H,W) in [0,1]
          aux: 可选，含 core/ref/after_skip 等中间量
        """
        if return_aux:
            pred_core, pred_conf, logits = self.core(
                src_imgs,
                src_depth,
                src_depth_conf,
                src_pointmap,
                tgt_vid=tgt_vid,
                src_vids=src_vids,
                return_logits=True,
            )
        else:
            pred_core, pred_conf = self.core(
                src_imgs,
                src_depth,
                src_depth_conf,
                src_pointmap,
                tgt_vid=tgt_vid,
                src_vids=src_vids,
            )
        # pred_core 已经是 sigmoid 后的 [0,1]

        ref_rgb = self.pick_ref_rgb(src_imgs)              # (B,3,H,W)
        ref_proj = self.skip_proj(ref_rgb)                 # (B,3,H,W)
        alpha = self.get_alpha()                           # scalar

        use_gate = self.use_conf_gate if use_conf_gate_override is None else bool(
            use_conf_gate_override)
        gate_strength = 1.0 if conf_gate_strength is None else float(
            conf_gate_strength)
        if gate_strength < 0.0:
            gate_strength = 0.0
        if gate_strength > 1.0:
            gate_strength = 1.0
        gate = None
        if use_gate:
            gate = pred_conf                               # (B,1,H,W)
            if self.conf_gate_detach:
                gate = gate.detach()
            if self.conf_gate_floor > 0.0:
                gate = gate.clamp(min=self.conf_gate_floor, max=1.0)
            else:
                gate = gate.clamp(0.0, 1.0)
            if gate_strength < 1.0:
                gate = (1.0 - gate_strength) + gate_strength * gate
            # (B,3,H,W) broadcast
            ref_term = ref_proj * gate
        else:
            ref_term = ref_proj

        pred_after_skip = pred_core + alpha * ref_term

        if self.use_tone:
            delta = self.tone_conv(pred_after_skip)
            pred_final = pred_after_skip + delta
        else:
            delta = None
            pred_final = pred_after_skip

        if self.use_view_cond and tgt_vid is not None and self.view_affine is not None:
            ds, db = self.view_affine(tgt_vid, src_vids=src_vids)
            ds = ds.view(-1, 3, 1, 1)
            db = db.view(-1, 3, 1, 1)
            if abs(self.view_affine_strength - 1.0) > 1e-6:
                ds = ds * self.view_affine_strength
                db = db * self.view_affine_strength
            pred_final = pred_final * (1.0 + ds) + db

        pred_final = pred_final.clamp(0.0, 1.0)

        if not return_aux:
            return pred_final, pred_conf

        aux = {
            "pred_core": pred_core.detach(),
            "ref_rgb": ref_rgb.detach(),
            "ref_proj": ref_proj.detach(),
            "alpha": alpha.detach(),
            "pred_after_skip": pred_after_skip.detach(),
            "use_conf_gate": bool(use_gate),
            "conf_gate_strength": float(gate_strength),
            "conf_gate_detach": bool(self.conf_gate_detach),
            "conf_gate_floor": float(self.conf_gate_floor),
        }
        if gate is not None:
            aux["gate"] = gate.detach()
        if return_aux and logits is not None:
            rgb_logits, conf_logits = logits
            aux["rgb_logits"] = rgb_logits.detach()
            aux["conf_logits"] = conf_logits.detach()
        if delta is not None:
            aux["tone_delta"] = delta.detach()

        return pred_final, pred_conf, aux
