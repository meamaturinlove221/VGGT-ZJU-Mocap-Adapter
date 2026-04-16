import contextlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw == "":
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if raw in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


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

    def __init__(
        self,
        ckpt_path: str,
        device: str | torch.device | None = None,
        pointmap_source: str = "depth_unproject",
        point_head_frame: str = "auto",
        unproject_impl: str = "legacy",
        amp: bool | None = None,
        tf32: bool | None = None,
        deterministic: bool | None = None,
    ):
        super().__init__()
        self.device = _resolve_device(device)
        src = str(pointmap_source or "depth_unproject").strip().lower()
        if src not in {"depth_unproject", "point_head", "auto"}:
            raise ValueError(
                f"unsupported pointmap_source={pointmap_source}, "
                "expected one of ['depth_unproject','point_head','auto']"
            )
        self.pointmap_source = src
        frame = str(point_head_frame or "auto").strip().lower()
        if frame not in {"auto", "camera", "world"}:
            raise ValueError(
                f"unsupported point_head_frame={point_head_frame}, "
                "expected one of ['auto','camera','world']"
            )
        self.point_head_frame = frame
        self._point_head_frame_resolved: str | None = None
        self.unproject_impl = str(unproject_impl or "legacy").strip().lower()
        if self.unproject_impl not in {"legacy", "upstream433", "pixel_center", "center_0.5"}:
            raise ValueError(
                f"unsupported unproject_impl={unproject_impl}, "
                "expected one of ['legacy','upstream433']"
            )
        self.amp = _env_bool("VGGT_AMP", True) if amp is None else bool(amp)
        self.tf32 = _env_bool("VGGT_TF32", True) if tf32 is None else bool(tf32)
        self.deterministic = (
            _env_bool("VGGT_STRICT_DETERMINISTIC", False)
            if deterministic is None
            else bool(deterministic)
        )
        self.image_load_workers = max(
            1,
            int((os.environ.get("VGGT_PRECOMPUTE_IMAGE_WORKERS", "") or "16").strip() or "16"),
        )
        self._image_executor = (
            ThreadPoolExecutor(max_workers=self.image_load_workers)
            if self.image_load_workers > 1
            else None
        )
        if str(self.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = bool(self.tf32)
            torch.backends.cudnn.allow_tf32 = bool(self.tf32)
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True, warn_only=True)

        print(f"[VGGTGeomTeacher] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError(f"unexpected checkpoint type: {type(state)}")
        keys = [str(k) for k in state.keys()]
        if keys and all(k.startswith("module.") for k in keys):
            state = {k[len("module."):]: v for k, v in state.items()}
            keys = [str(k) for k in state.keys()]

        has_track = any(k.startswith("track_head.") for k in keys)
        model = VGGT(enable_track=has_track).to(self.device)
        model_keys = set(model.state_dict().keys())
        matched = len(model_keys.intersection(set(keys)))
        if matched <= 0:
            raise RuntimeError(
                f"no matching keys between checkpoint and model: {ckpt_path}"
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "[VGGTGeomTeacher] load_state_dict strict=False",
            f"enable_track={has_track}",
            f"matched={matched}",
            f"missing={len(missing)}",
            f"unexpected={len(unexpected)}",
            f"pointmap_source={self.pointmap_source}",
            f"point_head_frame={self.point_head_frame}",
            f"unproject_impl={self.unproject_impl}",
            f"amp={self.amp}",
            f"tf32={self.tf32}",
            f"deterministic={self.deterministic}",
        )
        model.eval()
        self.model = model

    @staticmethod
    def _cam_to_world(point_cam: torch.Tensor, extrinsic_w2c: torch.Tensor) -> torch.Tensor:
        # point_cam: (B,V,H,W,3), extrinsic_w2c: (B,V,3,4)
        r = extrinsic_w2c[..., :3, :3]
        t = extrinsic_w2c[..., :3, 3]
        centered = point_cam - t.unsqueeze(-2).unsqueeze(-2)
        return torch.einsum("bvij,bvhwj->bvhwi", r.transpose(-1, -2), centered)

    @staticmethod
    def _unproject_pixel_center_offset(unproject_impl: str) -> float:
        mode = str(unproject_impl or "legacy").strip().lower()
        if mode in {"legacy", "orig", "default"}:
            return 0.0
        if mode in {"upstream433", "pixel_center", "center_0.5"}:
            return 0.5
        raise ValueError(
            f"unsupported unproject_impl={unproject_impl}, "
            "expected one of ['legacy','upstream433']"
        )

    @classmethod
    def _unproject_depth_to_world_batched(
        cls,
        depth_map: torch.Tensor,
        extrinsic_w2c: torch.Tensor,
        intrinsic: torch.Tensor,
        unproject_impl: str,
    ) -> torch.Tensor:
        if depth_map.ndim == 5 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        if depth_map.ndim != 4:
            raise ValueError(f"expected depth_map shape (B,V,H,W) or (B,V,H,W,1), got {tuple(depth_map.shape)}")

        depth_map_f32 = depth_map.float()
        extrinsic_f32 = extrinsic_w2c.float()
        intrinsic_f32 = intrinsic.float()
        batch, views, height, width = depth_map_f32.shape

        off = cls._unproject_pixel_center_offset(unproject_impl)
        ys, xs = torch.meshgrid(
            torch.arange(height, device=depth_map_f32.device, dtype=depth_map_f32.dtype) + off,
            torch.arange(width, device=depth_map_f32.device, dtype=depth_map_f32.dtype) + off,
            indexing="ij",
        )
        xs = xs.view(1, 1, height, width)
        ys = ys.view(1, 1, height, width)

        fx = intrinsic_f32[..., 0, 0].view(batch, views, 1, 1)
        fy = intrinsic_f32[..., 1, 1].view(batch, views, 1, 1)
        cx = intrinsic_f32[..., 0, 2].view(batch, views, 1, 1)
        cy = intrinsic_f32[..., 1, 2].view(batch, views, 1, 1)

        x_cam = (xs - cx) * depth_map_f32 / fx
        y_cam = (ys - cy) * depth_map_f32 / fy
        cam_coords = torch.stack((x_cam, y_cam, depth_map_f32), dim=-1)
        return cls._cam_to_world(cam_coords, extrinsic_f32)

    @staticmethod
    def _self_reproj_err_px(point_world: torch.Tensor, extrinsic_w2c: torch.Tensor, intrinsic: torch.Tensor) -> float:
        # quick scalar diagnostic for frame disambiguation
        if point_world.ndim != 5:
            return float("inf")
        b, v, h, w = point_world.shape[:4]
        ys, xs = torch.meshgrid(
            torch.arange(h, device=point_world.device, dtype=point_world.dtype),
            torch.arange(w, device=point_world.device, dtype=point_world.dtype),
            indexing="ij",
        )
        xs = xs.view(1, 1, h, w)
        ys = ys.view(1, 1, h, w)

        r = extrinsic_w2c[..., :3, :3]
        t = extrinsic_w2c[..., :3, 3]
        cam = torch.einsum("bvij,bvhwj->bvhwi", r, point_world) + t.unsqueeze(-2).unsqueeze(-2)
        z = cam[..., 2]
        fx = intrinsic[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        fy = intrinsic[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = intrinsic[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = intrinsic[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
        u = fx * (cam[..., 0] / (z + 1e-8)) + cx
        vv = fy * (cam[..., 1] / (z + 1e-8)) + cy
        err = torch.sqrt((u - xs) * (u - xs) + (vv - ys) * (vv - ys))
        valid = torch.isfinite(err) & torch.isfinite(z) & (z > 1e-6)
        if int(valid.sum().item()) <= 0:
            return float("inf")
        return float(err[valid].mean().item())

    def _resolve_paths(self, img_paths: list[str]) -> list[Path]:
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
        return paths

    def _resolve_point_head_frame_batched(
        self,
        pointmap_pred: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> str:
        frame_resolved = self._point_head_frame_resolved
        if frame_resolved is not None:
            return frame_resolved
        frame_resolved = self.point_head_frame
        if frame_resolved == "auto":
            cand_world = pointmap_pred[:1]
            cand_cam2world = self._cam_to_world(pointmap_pred[:1], extrinsic[:1])
            err_world = self._self_reproj_err_px(cand_world, extrinsic[:1], intrinsic[:1])
            err_cam = self._self_reproj_err_px(cand_cam2world, extrinsic[:1], intrinsic[:1])
            frame_resolved = "camera" if (err_cam < err_world) else "world"
            print(
                "[VGGTGeomTeacher] auto point_head_frame resolved:",
                f"{frame_resolved} (err_world={err_world:.3f}, err_cam={err_cam:.3f})"
            )
        self._point_head_frame_resolved = frame_resolved
        return frame_resolved

    def prepare_batch_inputs(self, batched_img_paths: list[list[str]]) -> dict[str, torch.Tensor | int | float]:
        if not batched_img_paths:
            raise ValueError("batched_img_paths must not be empty")
        batch_size = len(batched_img_paths)
        view_count = len(batched_img_paths[0])
        if view_count <= 0:
            raise ValueError("batched_img_paths must contain at least one view per frame")
        for idx, paths in enumerate(batched_img_paths):
            if len(paths) != view_count:
                raise ValueError(
                    f"inconsistent multi-view count in batch: item0={view_count} item{idx}={len(paths)}"
                )

        t0 = time.perf_counter()
        flat_paths: list[Path] = []
        for frame_paths in batched_img_paths:
            flat_paths.extend(self._resolve_paths(frame_paths))
        resolve_dt = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs = load_and_preprocess_images(
            flat_paths,
            num_workers=min(self.image_load_workers, max(1, len(flat_paths))),
            executor=self._image_executor,
        )
        load_dt = time.perf_counter() - t0
        if imgs.ndim != 4 or imgs.shape[0] != batch_size * view_count:
            raise RuntimeError(
                f"unexpected preprocessed image shape={tuple(imgs.shape)} "
                f"for batch={batch_size} views={view_count}"
            )
        imgs = imgs.view(batch_size, view_count, imgs.shape[1], imgs.shape[2], imgs.shape[3])
        return {
            "imgs": imgs,
            "batch_size": batch_size,
            "view_count": view_count,
            "resolve_dt": resolve_dt,
            "load_dt": load_dt,
        }

    def forward_prepared_batch(
        self,
        prepared_batch: dict[str, torch.Tensor | int | float],
    ) -> list[dict[str, torch.Tensor | str]]:
        device = self.device
        t_batch0 = time.perf_counter()
        imgs = prepared_batch["imgs"]
        batch_size = int(prepared_batch["batch_size"])
        view_count = int(prepared_batch["view_count"])
        resolve_dt = float(prepared_batch.get("resolve_dt", 0.0))
        load_dt = float(prepared_batch.get("load_dt", 0.0))
        t0 = time.perf_counter()
        if str(device).startswith("cuda"):
            imgs = imgs.pin_memory().to(device, non_blocking=True)
        else:
            imgs = imgs.to(device)
        transfer_dt = time.perf_counter() - t0

        if str(device).startswith("cuda") and self.amp:
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            t0 = time.perf_counter()
            agg_tokens_list, ps_idx = self.model.aggregator(imgs)
            agg_dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            pose_enc = self.model.camera_head(agg_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, imgs.shape[-2:]
            )
            cam_dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            depth_map, depth_conf = self.model.depth_head(agg_tokens_list, imgs, ps_idx)
            depth_dt = time.perf_counter() - t0
            pointmap_mode = self.pointmap_source
            if pointmap_mode == "auto":
                pointmap_mode = (
                    "point_head" if (self.model.point_head is not None) else "depth_unproject"
                )

            point_dt = 0.0
            if pointmap_mode == "point_head" and (self.model.point_head is not None):
                t0 = time.perf_counter()
                pointmap_pred, _ = self.model.point_head(agg_tokens_list, imgs, ps_idx)
                frame_resolved = self._resolve_point_head_frame_batched(
                    pointmap_pred=pointmap_pred,
                    extrinsic=extrinsic,
                    intrinsic=intrinsic,
                )
                if frame_resolved == "camera":
                    pointmap = self._cam_to_world(pointmap_pred, extrinsic)
                else:
                    pointmap = pointmap_pred
                point_dt = time.perf_counter() - t0
        if not (pointmap_mode == "point_head" and (self.model.point_head is not None)):
            t0 = time.perf_counter()
            pointmap = self._unproject_depth_to_world_batched(
                depth_map=depth_map,
                extrinsic_w2c=extrinsic,
                intrinsic=intrinsic,
                unproject_impl=self.unproject_impl,
            )
            point_dt = time.perf_counter() - t0

        total_dt = time.perf_counter() - t_batch0
        print(
            "[VGGTGeomTeacher] batch="
            f"{batch_size}x{view_count} resolve={resolve_dt:.2f}s load={load_dt:.2f}s "
            f"transfer={transfer_dt:.2f}s agg={agg_dt:.2f}s cam={cam_dt:.2f}s "
            f"depth={depth_dt:.2f}s point={point_dt:.2f}s total={total_dt:.2f}s",
            flush=True,
        )

        pointmap_frame = self._point_head_frame_resolved or self.point_head_frame
        outputs: list[dict[str, torch.Tensor | str]] = []
        for batch_idx in range(batch_size):
            outputs.append({
                "depth": depth_map[batch_idx],
                "depth_conf": depth_conf[batch_idx],
                "pointmap": pointmap[batch_idx],
                "extrinsic": extrinsic[batch_idx],
                "intrinsic": intrinsic[batch_idx],
                "pointmap_frame": pointmap_frame,
                "unproject_impl": self.unproject_impl,
            })
        return outputs

    def forward_batch(self, batched_img_paths: list[list[str]]) -> list[dict[str, torch.Tensor | str]]:
        prepared = self.prepare_batch_inputs(batched_img_paths)
        return self.forward_prepared_batch(prepared)

    @torch.no_grad()
    def forward(self, img_paths: list[str]):
        return self.forward_batch([img_paths])[0]

    def __del__(self):
        try:
            if self._image_executor is not None:
                self._image_executor.shutdown(wait=False, cancel_futures=True)
                self._image_executor = None
        except Exception:
            pass
