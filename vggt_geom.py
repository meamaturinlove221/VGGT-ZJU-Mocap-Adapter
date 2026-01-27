# F:\vggt\vggt_geom.py

import torch
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class VGGTGeomTeacher(torch.nn.Module):
    """
    封装 VGGT：输入一组图像路径，输出
        - depth:      (V, H, W)
        - depth_conf: (V, H, W)
        - pointmap:   (V, H, W, 3)
        - extrinsic:  (V, 4, 4)
        - intrinsic:  (V, 3, 3)
    其中 V 是视角数。
    """

    def __init__(self, ckpt_path: str, device: str | None = None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 1) 建模型骨架
        model = VGGT().to(device)

        # 2) 加载权重（你从 HF 下的 model.pt）
        print(f"[VGGTGeomTeacher] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        # 有些 ckpt 会包一层 dict，这里简单兼容一下
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, img_paths: list[str]):
        """
        img_paths: List[str]，多视角同一帧图像路径
        返回字典：
            {
              "depth": (V, H, W),
              "depth_conf": (V, H, W),
              "pointmap": (V, H, W, 3),
              "extrinsic": (V, 4, 4),
              "intrinsic": (V, 3, 3),
            }
        """
        device = self.device

        paths = [Path(p) for p in img_paths]
        # (V,3,H,W)
        imgs = load_and_preprocess_images(paths).to(device)
        # VGGT 期望的是 (B,V,3,H,W)，这里只有一帧，B=1
        imgs = imgs.unsqueeze(0)  # (1,V,3,H,W)

        if device == "cuda":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = torch.no_grad()

        with ctx:
            # 1) backbone：aggregator
            agg_tokens_list, ps_idx = self.model.aggregator(imgs)

            # 2) 相机分支
            pose_enc = self.model.camera_head(agg_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, imgs.shape[-2:]
            )  # (1,V,4,4), (1,V,3,3)

            # 3) 深度分支
            depth_map, depth_conf = self.model.depth_head(
                agg_tokens_list, imgs, ps_idx
            )  # (1,V,H,W)

            # 4) 用深度 + 相机反投影到 3D
            #    注意这里 squeeze(0) 去掉 batch 维度
            pointmap = unproject_depth_map_to_point_map(
                depth_map.squeeze(0),       # (V,H,W)
                extrinsic.squeeze(0),       # (V,4,4)
                intrinsic.squeeze(0),       # (V,3,3)
            )  # (V,H,W,3)

        depth = depth_map.squeeze(0)         # (V,H,W)
        depth_conf = depth_conf.squeeze(0)   # (V,H,W)
        extrinsic = extrinsic.squeeze(0)     # (V,4,4)
        intrinsic = intrinsic.squeeze(0)     # (V,3,3)

        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "pointmap": pointmap,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }
