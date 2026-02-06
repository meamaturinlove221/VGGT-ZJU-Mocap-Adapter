import contextlib
import os
import re
from pathlib import Path

import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _has_usable_cuda() -> bool:
    return torch.cuda.is_available() and (torch.cuda.device_count() > 0)


def _resolve_device(device: str | torch.device | None) -> str:
    if device is None:
        raw = os.environ.get("VGGT_DEVICE", "").strip().lower()
        if raw in {"", "auto", "none"}:
            device_str = "cuda" if _has_usable_cuda() else "cpu"
        else:
            device_str = raw
    else:
        device_str = str(device).strip().lower()
        if device_str in {"", "auto", "none"}:
            device_str = "cuda" if _has_usable_cuda() else "cpu"

    if device_str.startswith("cuda") and torch.cuda.device_count() == 0:
        return "cpu"
    return device_str


class VGGTGeomTeacher(torch.nn.Module):
    """Run VGGT on multi-view images and return geometry outputs."""

    def __init__(self, ckpt_path: str, device: str | torch.device | None = None):
        super().__init__()
        self.device = _resolve_device(device)

        model = VGGT().to(self.device)
        print(f"[VGGTGeomTeacher] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, img_paths: list[str]):
        device = self.device

        zju_root = os.environ.get("VGGT_ZJU_ROOT", "")
        paths: list[Path] = []
        for p in img_paths:
            s = str(p)
            if not os.path.exists(s):
                s2 = s.replace("\\", "/")
                if re.match(r"^[A-Za-z]:/", s2):
                    s2 = s2[2:]
                m = re.search(r"(CoreView_\d+/.*)$", s2)
                if zju_root and m:
                    s = os.path.join(zju_root, m.group(1))
                elif zju_root:
                    s = os.path.join(zju_root, s2.lstrip("/"))
            paths.append(Path(s))

        imgs = load_and_preprocess_images(paths).to(device)
        imgs = imgs.unsqueeze(0)  # (1, V, 3, H, W)

        if str(device).startswith("cuda"):
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            agg_tokens_list, ps_idx = self.model.aggregator(imgs)
            pose_enc = self.model.camera_head(agg_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, imgs.shape[-2:]
            )
            depth_map, depth_conf = self.model.depth_head(agg_tokens_list, imgs, ps_idx)
            pointmap = unproject_depth_map_to_point_map(
                depth_map.squeeze(0),
                extrinsic.squeeze(0),
                intrinsic.squeeze(0),
            )

        depth = depth_map.squeeze(0)
        depth_conf = depth_conf.squeeze(0)
        extrinsic = extrinsic.squeeze(0)
        intrinsic = intrinsic.squeeze(0)

        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "pointmap": pointmap,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }
