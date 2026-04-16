import gc
import json
import math
import os
import os.path as osp
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vggt_geom import VGGTGeomTeacher
from zju_multiview import ZJUMocapSeq


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw == "":
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if raw in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _emit_precompute_heartbeat(
    state: dict | None,
    *,
    seq_name: str,
    batch_idx: int,
    frame_start: int,
    frame_end: int,
    views: int,
    phase: str,
    items_done: int | None = None,
    items_total: int | None = None,
    frame_id: int | None = None,
    phase_started_at: float | None = None,
    total_started_at: float | None = None,
    extra: dict | None = None,
) -> None:
    if state is None:
        return
    state["counter"] = int(state.get("counter", 0)) + 1
    now = time.perf_counter()
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "scene_id": str(seq_name),
        "batch_idx": int(batch_idx),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "views": int(views),
        "phase": str(phase),
        "progress_counter": int(state["counter"]),
        "items_done": None if items_done is None else int(items_done),
        "items_total": None if items_total is None else int(items_total),
        "elapsed_phase_sec": None if phase_started_at is None else float(max(0.0, now - phase_started_at)),
        "elapsed_total_sec": None if total_started_at is None else float(max(0.0, now - total_started_at)),
    }
    if frame_id is not None:
        payload["frame_id"] = int(frame_id)
    if extra:
        payload.update(extra)
    print("[precompute-heartbeat] " + json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)


def _save_npz(out_npz: str, save_dict: dict, compressed: bool) -> float:
    t0 = time.perf_counter()
    if compressed:
        np.savez_compressed(out_npz, **save_dict)
    else:
        np.savez(out_npz, **save_dict)
    return time.perf_counter() - t0


def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def _save_support_stats_json(sidecar_path: str, payload: dict | None) -> None:
    if not sidecar_path or not payload:
        return
    clean = {}
    for key, value in payload.items():
        if value is None:
            continue
        scalar = _json_scalar(value)
        if scalar is None:
            continue
        clean[str(key)] = scalar
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=True, sort_keys=True)


def _save_outputs(
    out_npz: str,
    save_dict: dict,
    compressed: bool,
    support_stats_json_path: str | None = None,
    support_stats_payload: dict | None = None,
) -> float:
    t0 = time.perf_counter()
    if compressed:
        np.savez_compressed(out_npz, **save_dict)
    else:
        np.savez(out_npz, **save_dict)
    _save_support_stats_json(support_stats_json_path, support_stats_payload)
    return time.perf_counter() - t0


class _AsyncSaveQueue:
    def __init__(self, *, max_workers: int, max_pending: int, compressed: bool):
        self.max_workers = max(0, int(max_workers))
        self.max_pending = max(1, int(max_pending))
        self.compressed = bool(compressed)
        self._executor = (
            ThreadPoolExecutor(max_workers=self.max_workers)
            if self.max_workers > 0
            else None
        )
        self._pending: deque[tuple] = deque()

    def _drain_completed(self) -> None:
        while self._pending and self._pending[0][0].done():
            future, meta = self._pending.popleft()
            save_dt = float(future.result())
            meta["on_done"](save_dt)

    def submit(
        self,
        *,
        out_npz: str,
        save_dict: dict,
        on_done,
        support_stats_json_path: str | None = None,
        support_stats_payload: dict | None = None,
    ) -> None:
        if self._executor is None:
            save_dt = _save_outputs(
                out_npz,
                save_dict,
                self.compressed,
                support_stats_json_path=support_stats_json_path,
                support_stats_payload=support_stats_payload,
            )
            on_done(save_dt)
            return

        self._drain_completed()
        while len(self._pending) >= self.max_pending:
            future, meta = self._pending.popleft()
            save_dt = float(future.result())
            meta["on_done"](save_dt)

        future = self._executor.submit(
            _save_outputs,
            out_npz,
            save_dict,
            self.compressed,
            support_stats_json_path,
            support_stats_payload,
        )
        self._pending.append((future, {"on_done": on_done}))

    def flush(self) -> None:
        while self._pending:
            future, meta = self._pending.popleft()
            save_dt = float(future.result())
            meta["on_done"](save_dt)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _depth_like_to_3d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Convert depth/conf tensor to (V,H,W) and return whether original had channel axis."""
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0], True
    if x.ndim == 3:
        return x, False
    raise RuntimeError(f"expected (V,H,W) or (V,H,W,1), got {tuple(x.shape)}")


def _restore_depth_like_3d(x3: torch.Tensor, had_last_ch: bool) -> torch.Tensor:
    if had_last_ch:
        return x3[..., None]
    return x3


def _depth_like_to_4d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Convert depth/conf tensor to (B,V,H,W) and return whether original had channel axis."""
    if x.ndim == 5 and x.shape[-1] == 1:
        return x[..., 0], True
    if x.ndim == 4:
        return x, False
    raise RuntimeError(f"expected (B,V,H,W) or (B,V,H,W,1), got {tuple(x.shape)}")


def _build_multiview_support_batched(
    *,
    point_world: torch.Tensor,     # (B,V,H,W,3), world frame
    depth: torch.Tensor,           # (B,V,H,W) or (B,V,H,W,1)
    depth_conf: torch.Tensor,      # (B,V,H,W) or (B,V,H,W,1)
    extrinsic_w2c: torch.Tensor,   # (B,V,3,4)
    intrinsic: torch.Tensor,       # (B,V,3,3)
    tol_abs: float,
    tol_rel: float,
    stride: int,
    conf_valid_floor: float,
    generation_gate: torch.Tensor | None = None,
    min_depth: float = 1e-6,
    heartbeat_sec: float = 30.0,
    heartbeat_prefix: str = "",
    heartbeat_cb=None,
    return_diag: bool = False,
) -> torch.Tensor | dict:
    """Cross-view depth agreement support in [0,1], shape (B,V,H,W)."""
    point_world = torch.as_tensor(point_world)
    depth = torch.as_tensor(depth)
    depth_conf = torch.as_tensor(depth_conf)
    extrinsic_w2c = torch.as_tensor(extrinsic_w2c)
    intrinsic = torch.as_tensor(intrinsic)
    dev = point_world.device
    point_world = point_world.to(device=dev, dtype=torch.float32)
    depth = depth.to(device=dev, dtype=torch.float32)
    depth_conf = depth_conf.to(device=dev, dtype=torch.float32)
    extrinsic_w2c = extrinsic_w2c.to(device=dev, dtype=torch.float32)
    intrinsic = intrinsic.to(device=dev, dtype=torch.float32)

    if point_world.ndim != 5 or point_world.shape[-1] != 3:
        raise RuntimeError(f"point_world must be (B,V,H,W,3), got {tuple(point_world.shape)}")
    if extrinsic_w2c.ndim != 4 or tuple(extrinsic_w2c.shape[-2:]) != (3, 4):
        raise RuntimeError(f"extrinsic_w2c must be (B,V,3,4), got {tuple(extrinsic_w2c.shape)}")
    if intrinsic.ndim != 4 or tuple(intrinsic.shape[-2:]) != (3, 3):
        raise RuntimeError(f"intrinsic must be (B,V,3,3), got {tuple(intrinsic.shape)}")

    depth4, _ = _depth_like_to_4d(depth)
    conf4, _ = _depth_like_to_4d(depth_conf)

    b, v, h, w, _ = point_world.shape
    if depth4.shape != (b, v, h, w):
        raise RuntimeError(f"depth shape mismatch: expected {(b, v, h, w)} got {tuple(depth4.shape)}")
    if conf4.shape != (b, v, h, w):
        raise RuntimeError(f"depth_conf shape mismatch: expected {(b, v, h, w)} got {tuple(conf4.shape)}")
    if extrinsic_w2c.shape[:2] != (b, v) or intrinsic.shape[:2] != (b, v):
        raise RuntimeError("camera tensor view count mismatch")

    s = max(1, int(stride))
    if s > 1:
        hh = max(1, h // s)
        ww = max(1, w // s)
        pw = F.interpolate(
            point_world.permute(0, 1, 4, 2, 3).reshape(b * v, 3, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, 3, hh, ww).permute(0, 1, 3, 4, 2)
        dz = F.interpolate(
            depth4.reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, hh, ww)
        cf = F.interpolate(
            conf4.reshape(b * v, 1, h, w),
            size=(hh, ww),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, hh, ww)
    else:
        hh, ww = h, w
        pw = point_world
        dz = depth4
        cf = conf4
    if w > 1:
        sx = float(max(ww - 1, 0)) / float(w - 1)
    else:
        sx = 1.0
    if h > 1:
        sy = float(max(hh - 1, 0)) / float(h - 1)
    else:
        sy = 1.0

    valid = (
        torch.isfinite(dz)
        & (dz > float(min_depth))
        & torch.isfinite(cf)
        & (cf >= float(conf_valid_floor))
    )
    gate = None
    if generation_gate is not None:
        gate = torch.as_tensor(generation_gate, device=pw.device, dtype=torch.float32)
        if gate.shape != (b, v, h, w):
            raise RuntimeError(f"generation_gate shape mismatch: expected {(b, v, h, w)} got {tuple(gate.shape)}")
        if s > 1:
            gate = F.interpolate(gate.reshape(b * v, 1, h, w), size=(hh, ww), mode="nearest").reshape(b, v, hh, ww)
        valid = valid & (gate > 0.5)

    out = torch.zeros((b, v, hh, ww), device=pw.device, dtype=torch.float32)
    cover_map = torch.zeros((b, v, hh, ww), device=pw.device, dtype=torch.float32)
    ta = float(max(0.0, tol_abs))
    tr = float(max(0.0, tol_rel))
    hb_every = float(max(0.0, heartbeat_sec))
    view_chunk = max(1, _env_int("VGGT_PRECOMPUTE_MV_SUPPORT_VIEW_CHUNK", 8))
    last_hb = time.perf_counter()
    support_t0 = time.perf_counter()
    view_idx = torch.arange(v, device=pw.device)
    view_idx_grid = view_idx.view(1, 1, v, 1)
    extr_r = extrinsic_w2c[:, :, :3, :3]
    extr_t = extrinsic_w2c[:, :, :3, 3]
    fx_all = intrinsic[:, :, 0, 0].view(b, 1, v, 1)
    fy_all = intrinsic[:, :, 1, 1].view(b, 1, v, 1)
    cx_all = intrinsic[:, :, 0, 2].view(b, 1, v, 1)
    cy_all = intrinsic[:, :, 1, 2].view(b, 1, v, 1)
    pw_flat = pw.reshape(b, v, -1, 3).to(dtype=torch.float32)
    dz_flat = dz.reshape(b, v, -1)
    cf_flat = cf.reshape(b, v, -1)
    gate_flat = gate.reshape(b, v, -1) if gate is not None else None
    flat_cap = max(0, hh * ww - 1)
    src_valid_all = valid.reshape(b, v, -1) & torch.isfinite(pw_flat).all(dim=-1)

    vi = 0
    chunk_limit = min(v, view_chunk)
    while vi < v:
        chunk_n = min(chunk_limit, v - vi)
        source_idx = view_idx[vi:vi + chunk_n]
        try:
            xw = pw_flat[:, source_idx]
            src_valid = src_valid_all[:, source_idx]
            cam = torch.einsum("bsnc,bvdc->bsvnd", xw, extr_r) + extr_t[:, None, :, None, :]
            z = cam[..., 2]
            proj_ok = src_valid.unsqueeze(2) & torch.isfinite(z) & (z > float(min_depth))

            u = fx_all * (cam[..., 0] / (z + 1e-8)) + cx_all
            vv = fy_all * (cam[..., 1] / (z + 1e-8)) + cy_all
            ui = torch.round(u * sx).long()
            vvi = torch.round(vv * sy).long()
            inside = (
                proj_ok
                & (view_idx_grid != source_idx.view(1, chunk_n, 1, 1))
                & (ui >= 0)
                & (ui < ww)
                & (vvi >= 0)
                & (vvi < hh)
            )

            flat_idx = (vvi * ww + ui).clamp_(0, flat_cap)
            dt = torch.gather(dz_flat[:, None, :, :].expand(-1, chunk_n, -1, -1), 3, flat_idx)
            cft = torch.gather(cf_flat[:, None, :, :].expand(-1, chunk_n, -1, -1), 3, flat_idx)
            valid_tgt = (
                inside
                & torch.isfinite(dt)
                & (dt > float(min_depth))
                & torch.isfinite(cft)
                & (cft >= float(conf_valid_floor))
            )
            if gate_flat is not None:
                gate_vals = torch.gather(gate_flat[:, None, :, :].expand(-1, chunk_n, -1, -1), 3, flat_idx)
                valid_tgt = valid_tgt & (gate_vals > 0.5)

            tol = ta + tr * dt.abs()
            agree = valid_tgt & ((z - dt).abs() <= tol)
            cover = valid_tgt.to(dtype=torch.float32).sum(dim=2)
            support = agree.to(dtype=torch.float32).sum(dim=2)

            ratio = support / cover.clamp_min(1.0)
            ratio = torch.where(cover > 0.0, ratio, torch.ones_like(ratio))
            ratio = ratio.reshape(b, chunk_n, hh, ww)
            cover_chunk = cover.reshape(b, chunk_n, hh, ww)

            empty_src = src_valid.sum(dim=2) <= 0
            if bool(empty_src.any().item()):
                ratio = torch.where(empty_src[..., None, None], torch.ones_like(ratio), ratio)
                cover_chunk = torch.where(empty_src[..., None, None], torch.zeros_like(cover_chunk), cover_chunk)

            out[:, source_idx] = ratio
            cover_map[:, source_idx] = cover_chunk
            vi += chunk_n
            chunk_limit = max(chunk_limit, chunk_n)
        except Exception as exc:
            if _is_cuda_oom(exc) and chunk_n > 1:
                next_chunk = max(1, chunk_n // 2)
                prefix = (heartbeat_prefix or "[mv_support]").strip()
                print(f"{prefix} oom view_chunk={chunk_n} -> {next_chunk}", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_limit = next_chunk
                continue
            raise

        now = time.perf_counter()
        if hb_every > 0.0 and ((now - last_hb) >= hb_every):
            prefix = (heartbeat_prefix or "[mv_support]").strip()
            print(
                f"{prefix} progress view={vi}/{v} batch={b} chunk={chunk_n} elapsed={now - support_t0:.1f}s",
                flush=True,
            )
            if heartbeat_cb is not None:
                heartbeat_cb(
                    phase="mv_support_progress",
                    items_done=vi,
                    items_total=v,
                    phase_started_at=support_t0,
                    extra={"view_index": int(min(max(vi - 1, 0), v - 1)), "view_chunk": int(chunk_n)},
                )
            last_hb = now

    out = out.clamp(0.0, 1.0)
    if s > 1:
        out = F.interpolate(
            out.reshape(b * v, 1, hh, ww),
            size=(h, w),
            mode="nearest",
        ).reshape(b, v, h, w)
        cover_map = F.interpolate(
            cover_map.reshape(b * v, 1, hh, ww),
            size=(h, w),
            mode="nearest",
        ).reshape(b, v, h, w)
        valid = F.interpolate(
            valid.float().reshape(b * v, 1, hh, ww),
            size=(h, w),
            mode="nearest",
        ).reshape(b, v, h, w) > 0.5
        if gate is not None:
            gate = F.interpolate(
                gate.reshape(b * v, 1, hh, ww),
                size=(h, w),
                mode="nearest",
            ).reshape(b, v, h, w)

    if not return_diag:
        return out
    return {
        "support": out,
        "cover": cover_map,
        "valid": valid,
        "gate": gate,
    }


def _build_multiview_support(
    *,
    point_world: torch.Tensor,     # (V,H,W,3), world frame
    depth: torch.Tensor,           # (V,H,W) or (V,H,W,1)
    depth_conf: torch.Tensor,      # (V,H,W) or (V,H,W,1)
    extrinsic_w2c: torch.Tensor,   # (V,3,4)
    intrinsic: torch.Tensor,       # (V,3,3)
    tol_abs: float,
    tol_rel: float,
    stride: int,
    conf_valid_floor: float,
    generation_gate: torch.Tensor | None = None,
    min_depth: float = 1e-6,
    heartbeat_sec: float = 30.0,
    heartbeat_prefix: str = "",
    heartbeat_cb=None,
    return_diag: bool = False,
) -> torch.Tensor | dict:
    diag = _build_multiview_support_batched(
        point_world=torch.as_tensor(point_world).unsqueeze(0),
        depth=torch.as_tensor(depth).unsqueeze(0),
        depth_conf=torch.as_tensor(depth_conf).unsqueeze(0),
        extrinsic_w2c=torch.as_tensor(extrinsic_w2c).unsqueeze(0),
        intrinsic=torch.as_tensor(intrinsic).unsqueeze(0),
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        stride=stride,
        conf_valid_floor=conf_valid_floor,
        generation_gate=None if generation_gate is None else torch.as_tensor(generation_gate).unsqueeze(0),
        min_depth=min_depth,
        heartbeat_sec=heartbeat_sec,
        heartbeat_prefix=heartbeat_prefix,
        heartbeat_cb=heartbeat_cb,
        return_diag=True,
    )
    if not return_diag:
        return diag["support"][0]
    return {
        "support": diag["support"][0],
        "cover": diag["cover"][0],
        "valid": diag["valid"][0],
        "gate": None if diag["gate"] is None else diag["gate"][0],
    }


def _apply_support_to_depth_conf(
    depth_conf: torch.Tensor,
    support01: torch.Tensor,
    mode: str,
    floor: float,
    gamma: float,
    clip_thr: float,
    clip_floor: float,
    hard_thr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return filtered depth_conf (same shape as input) and support weight map (V,H,W)."""
    depth_conf = torch.as_tensor(depth_conf, dtype=torch.float32)
    support01 = torch.as_tensor(support01, device=depth_conf.device, dtype=torch.float32)
    conf3, had_last_ch = _depth_like_to_3d(depth_conf)
    if support01.shape != conf3.shape:
        raise RuntimeError(
            f"support/depth_conf shape mismatch: support={tuple(support01.shape)} conf={tuple(conf3.shape)}"
        )

    mode_l = str(mode or "linear").strip().lower()
    if mode_l == "clip":
        thr = float(min(1.0, max(0.0, clip_thr)))
        cfl = float(min(1.0, max(0.0, clip_floor)))
        weight = torch.ones_like(support01)
        weight = torch.where(support01 < thr, torch.full_like(weight, cfl), weight)
    else:
        floor = float(min(1.0, max(0.0, floor)))
        gamma = float(max(1e-6, gamma))
        weight = floor + (1.0 - floor) * torch.pow(support01.clamp(0.0, 1.0), gamma)
    if hard_thr >= 0.0:
        weight = weight * (support01 >= float(hard_thr)).to(weight.dtype)
    filtered = conf3 * weight
    return _restore_depth_like_3d(filtered, had_last_ch), weight


def _preserve_fg_depth_conf(
    *,
    depth_conf_raw: torch.Tensor,
    depth_conf_after_support: torch.Tensor,
    fg_mask: torch.Tensor | None,
    region_mode_resolved: str,
    preserve_px: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Keep raw depth confidence on foreground for safer support-enabled generation."""
    if fg_mask is None:
        return depth_conf_after_support, {"depth_conf_fg_preserved_active": 0.0}
    mode_l = str(region_mode_resolved or "").strip().lower()
    if mode_l not in {"bg_only", "fg_eroded_off"}:
        return depth_conf_after_support, {"depth_conf_fg_preserved_active": 0.0}

    raw3, had_last_ch = _depth_like_to_3d(torch.as_tensor(depth_conf_raw, dtype=torch.float32))
    after3, _ = _depth_like_to_3d(torch.as_tensor(depth_conf_after_support, device=raw3.device, dtype=torch.float32))
    fg3 = torch.as_tensor(fg_mask, device=raw3.device, dtype=torch.float32)
    if fg3.shape != raw3.shape:
        raise RuntimeError(
            f"foreground mask/depth_conf shape mismatch: fg={tuple(fg3.shape)} conf={tuple(raw3.shape)}"
        )

    fg_exact_bool = fg3 > 0.5
    fg_preserve = _dilate_mask_tensor_3d(fg3, preserve_px) if int(max(0, preserve_px)) > 0 else fg3
    fg_bool = fg_preserve > 0.5
    preserved3 = torch.where(fg_bool, raw3, after3)
    fg_count = int(fg_exact_bool.sum().item())
    stats = {
        "depth_conf_fg_preserved_active": 1.0,
        "depth_conf_fg_preserve_px": float(int(max(0, preserve_px))),
        "depth_conf_fg_exact_ratio": round(float(fg_exact_bool.float().mean().item()), 6),
        "depth_conf_fg_preserve_ratio": round(float(fg_bool.float().mean().item()), 6),
    }
    if fg_count > 0:
        stats["depth_conf_fg_raw_mean"] = round(float(raw3[fg_exact_bool].mean().item()), 6)
        stats["depth_conf_fg_after_support_mean"] = round(float(after3[fg_exact_bool].mean().item()), 6)
        stats["depth_conf_fg_final_mean"] = round(float(preserved3[fg_exact_bool].mean().item()), 6)
    else:
        stats["depth_conf_fg_raw_mean"] = float("nan")
        stats["depth_conf_fg_after_support_mean"] = float("nan")
        stats["depth_conf_fg_final_mean"] = float("nan")
    return _restore_depth_like_3d(preserved3, had_last_ch), stats


ZJU_ROOT = r"F:\datasets\ZJU_MoCap\data\zju_mocap"
SEQ_LIST = [
    "CoreView_390",
    # "CoreView_313",
    # "CoreView_377",
]
SELECT_CAMERAS = [
    "Camera_B1",
    "Camera_B5",
    "Camera_B10",
    "Camera_B14",
    "Camera_B19",
    "Camera_B23",
]
CKPT_PATH = "model.pt"


def _split_list(raw: str) -> list[str]:
    raw = (raw or "").replace(",", " ").replace(";", " ")
    return [x for x in raw.split() if x]


def _resolve_ckpt_path() -> str:
    env_ckpt = (os.environ.get("VGGT_CKPT", "") or "").strip()
    if not env_ckpt:
        env_ckpt = (os.environ.get("VGGT_PRECOMPUTE_CKPT", "") or "").strip()
    if env_ckpt:
        return env_ckpt

    local_default = Path(__file__).resolve().with_name("model.pt")
    if local_default.exists():
        return str(local_default)
    return CKPT_PATH


def _resolve_img_path(path_str: str, zju_root: str, seq_root: str) -> str:
    s = str(path_str or "").strip()
    if not s:
        return s
    if osp.exists(s):
        return s
    if osp.isabs(s):
        return s

    s2 = s.replace("\\", "/")
    if len(s2) >= 3 and s2[1] == ":" and s2[2] == "/":
        s2 = s2[3:]
    idx = s2.find("CoreView_")
    if idx >= 0:
        s2 = s2[idx:]
    s2 = s2.lstrip("/")
    if s2.startswith("CoreView_"):
        return osp.join(zju_root, s2)
    return osp.join(seq_root, s2)


def _infer_mask_path_from_image(img_path: str, preferred: str = "auto") -> tuple[str | None, str]:
    p = Path(str(img_path))
    if len(p.parts) < 3:
        return None, ""
    cam = p.parent.name
    seq_root = p.parent.parent
    stem = p.stem + ".png"
    pref = str(preferred or "auto").strip().lower()
    if pref not in {"auto", "mask", "mask_cihp"}:
        pref = "auto"
    order = ["mask", "mask_cihp"] if pref == "auto" else [pref]
    for tok in order:
        cand = seq_root / tok / cam / stem
        if cand.is_file():
            return str(cand), tok
    return None, ""


def _normalize_mask_binary_np(mask_like: np.ndarray) -> np.ndarray:
    x = np.asarray(mask_like).astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    if x.ndim != 2:
        raise RuntimeError(f"unexpected mask ndim: {x.ndim}")
    if x.size <= 0:
        return np.zeros_like(x, dtype=np.float32)
    mx = float(np.nanmax(x))
    if mx <= 1.5:
        fg = x >= 0.5
    else:
        fg = x > 0.0
    return fg.astype(np.float32)


def _erode_mask_tensor_3d(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    k = int(max(0, erode_px))
    if k <= 0:
        return mask
    if mask.ndim != 3:
        raise RuntimeError(f"mask must be (V,H,W), got {tuple(mask.shape)}")
    v, h, w = mask.shape
    x = mask.clamp(0.0, 1.0).reshape(v, 1, h, w)
    kk = 2 * k + 1
    x_inv = 1.0 - x
    dil_inv = F.max_pool2d(x_inv, kernel_size=kk, stride=1, padding=k)
    out = (1.0 - dil_inv).reshape(v, h, w)
    return out.clamp(0.0, 1.0)


def _dilate_mask_tensor_3d(mask: torch.Tensor, dilate_px: int) -> torch.Tensor:
    k = int(max(0, dilate_px))
    if k <= 0:
        return mask
    if mask.ndim != 3:
        raise RuntimeError(f"mask must be (V,H,W), got {tuple(mask.shape)}")
    v, h, w = mask.shape
    x = mask.clamp(0.0, 1.0).reshape(v, 1, h, w)
    kk = 2 * k + 1
    out = F.max_pool2d(x, kernel_size=kk, stride=1, padding=k).reshape(v, h, w)
    return out.clamp(0.0, 1.0)


def _load_fg_mask_tensor_for_img_paths(
    img_paths: list[str],
    *,
    zju_root: str,
    seq_root: str,
    preferred: str,
    target_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    masks = []
    src_tok = ""
    th, tw = int(target_hw[0]), int(target_hw[1])
    for raw_ip in img_paths:
        local_img = _resolve_img_path(raw_ip, zju_root=zju_root, seq_root=seq_root)
        mp, tok = _infer_mask_path_from_image(local_img, preferred=preferred)
        if not mp:
            raise FileNotFoundError(f"foreground mask not found for image: {local_img}")
        arr = np.array(Image.open(mp))
        mask_np = _normalize_mask_binary_np(arr)
        if mask_np.shape != (th, tw):
            mask_img = Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            mask_img = mask_img.resize((tw, th), Image.NEAREST)
            mask_np = (np.array(mask_img).astype(np.float32) > 127).astype(np.float32)
        masks.append(mask_np)
        if not src_tok:
            src_tok = tok
    fg = np.stack(masks, axis=0)
    return torch.from_numpy(fg).to(device=device, dtype=torch.float32), str(src_tok or "")


def _build_support_generation_gate(
    img_paths: list[str],
    *,
    zju_root: str,
    seq_root: str,
    target_hw: tuple[int, int],
    region_mode: str,
    fg_mask_source: str,
    fg_erode_px: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str, str]:
    mode = str(region_mode or "all").strip().lower() or "all"
    if mode == "all":
        return None, None, "", "all"
    if mode == "auto":
        try:
            fg_mask, resolved_src = _load_fg_mask_tensor_for_img_paths(
                img_paths,
                zju_root=zju_root,
                seq_root=seq_root,
                preferred=fg_mask_source,
                target_hw=target_hw,
                device=device,
            )
        except FileNotFoundError:
            return None, None, "", "all"
        return (1.0 - fg_mask).clamp(0.0, 1.0), fg_mask, resolved_src, "bg_only"
    fg_mask, resolved_src = _load_fg_mask_tensor_for_img_paths(
        img_paths,
        zju_root=zju_root,
        seq_root=seq_root,
        preferred=fg_mask_source,
        target_hw=target_hw,
        device=device,
    )
    if mode == "bg_only":
        return (1.0 - fg_mask).clamp(0.0, 1.0), fg_mask, resolved_src, "bg_only"
    if mode == "fg_eroded_off":
        fg_eroded = _erode_mask_tensor_3d(fg_mask, fg_erode_px)
        return (1.0 - fg_eroded).clamp(0.0, 1.0), fg_mask, resolved_src, "fg_eroded_off"
    raise RuntimeError(f"unsupported support generation region mode: {region_mode}")


def _prepare_support_generation_batch(
    batch_entries: list[dict],
    *,
    zju_root: str,
    seq_root: str,
    target_hw: tuple[int, int],
    region_mode: str,
    fg_mask_source: str,
    fg_erode_px: int,
) -> list[dict]:
    if not batch_entries:
        return []
    if str(region_mode or "all").strip().lower() == "all":
        return [{
            "gate": None,
            "fg": None,
            "mask_source": "",
            "region_mode_resolved": "all",
        } for _ in batch_entries]

    jobs = []
    for entry in batch_entries:
        jobs.append({
            "img_paths": list(entry["img_paths"]),
        })

    def _load_one(job: dict) -> dict:
        gate, fg, src, resolved = _build_support_generation_gate(
            job["img_paths"],
            zju_root=zju_root,
            seq_root=seq_root,
            target_hw=target_hw,
            region_mode=region_mode,
            fg_mask_source=fg_mask_source,
            fg_erode_px=fg_erode_px,
            device=torch.device("cpu"),
        )
        return {
            "gate": gate,
            "fg": fg,
            "mask_source": src,
            "region_mode_resolved": resolved,
        }

    worker_count = min(max(1, len(jobs)), 8)
    if worker_count <= 1:
        return [_load_one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        return list(ex.map(_load_one, jobs))


def _compute_mv_support_diag_batch_adaptive(
    geoms: list[dict],
    *,
    generation_gates: list[torch.Tensor] | None,
    tol_abs: float,
    tol_rel: float,
    stride: int,
    conf_valid_floor: float,
    heartbeat_sec: float,
    heartbeat_prefix: str,
) -> list[dict]:
    if not geoms:
        return []
    try:
        point_world_batch = torch.stack([torch.as_tensor(geom["pointmap"]) for geom in geoms], dim=0)
        depth_batch = torch.stack([torch.as_tensor(geom["depth"]) for geom in geoms], dim=0)
        depth_conf_batch = torch.stack([torch.as_tensor(geom["depth_conf"]) for geom in geoms], dim=0)
        extrinsic_batch = torch.stack([torch.as_tensor(geom["extrinsic"]) for geom in geoms], dim=0)
        intrinsic_batch = torch.stack([torch.as_tensor(geom["intrinsic"]) for geom in geoms], dim=0)
        gate_batch = None
        if generation_gates is not None:
            gate_batch = torch.stack([
                torch.as_tensor(gate, device=point_world_batch.device, dtype=torch.float32)
                for gate in generation_gates
            ], dim=0)
        diag_batch = _build_multiview_support_batched(
            point_world=point_world_batch,
            depth=depth_batch,
            depth_conf=depth_conf_batch,
            extrinsic_w2c=extrinsic_batch,
            intrinsic=intrinsic_batch,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            stride=stride,
            conf_valid_floor=conf_valid_floor,
            generation_gate=gate_batch,
            heartbeat_sec=heartbeat_sec,
            heartbeat_prefix=heartbeat_prefix,
            return_diag=True,
        )
        outputs = []
        for idx in range(len(geoms)):
            outputs.append({
                "support": diag_batch["support"][idx],
                "cover": diag_batch["cover"][idx],
                "valid": diag_batch["valid"][idx],
                "gate": None if diag_batch["gate"] is None else diag_batch["gate"][idx],
            })
        return outputs
    except Exception as exc:
        if _is_cuda_oom(exc) and len(geoms) > 1:
            left_n = max(1, len(geoms) // 2)
            right_n = len(geoms) - left_n
            print(
                f"{heartbeat_prefix} oom frame_batch={len(geoms)} -> split {left_n}+{right_n}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            left = _compute_mv_support_diag_batch_adaptive(
                geoms[:left_n],
                generation_gates=None if generation_gates is None else generation_gates[:left_n],
                tol_abs=tol_abs,
                tol_rel=tol_rel,
                stride=stride,
                conf_valid_floor=conf_valid_floor,
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=heartbeat_prefix,
            )
            right = _compute_mv_support_diag_batch_adaptive(
                geoms[left_n:],
                generation_gates=None if generation_gates is None else generation_gates[left_n:],
                tol_abs=tol_abs,
                tol_rel=tol_rel,
                stride=stride,
                conf_valid_floor=conf_valid_floor,
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=heartbeat_prefix,
            )
            return left + right
        raise


def _process_frame_batch(
    *,
    teacher: VGGTGeomTeacher,
    batch_entries: list[dict],
    seq_name: str,
    batch_idx: int,
    seq_root: str,
    zju_root: str,
    pointmap_source: str,
    point_head_frame: str,
    unproject_impl: str,
    mv_support_on: bool,
    mv_support_tol_abs: float,
    mv_support_tol_rel: float,
    mv_support_stride: int,
    mv_support_mode: str,
    mv_support_floor: float,
    mv_support_gamma: float,
    mv_support_clip_thr: float,
    mv_support_clip_floor: float,
    mv_support_hard_thr: float,
    mv_conf_valid_floor: float,
    mv_support_save: bool,
    mv_support_save_raw_conf: bool,
    mv_support_region_mode: str,
    mv_support_fg_mask_source: str,
    mv_support_fg_erode_px: int,
    mv_support_fg_preserve_px: int,
    heartbeat_sec: float,
    save_queue: _AsyncSaveQueue,
    preloaded_batch: dict | None = None,
) -> None:
    if not batch_entries:
        return

    batched_img_paths = [entry["img_paths"] for entry in batch_entries]
    first_fid = int(batch_entries[0]["fid"])
    last_fid = int(batch_entries[-1]["fid"])
    batch_t0 = time.perf_counter()
    batch_views = len(batch_entries[0]["cam_names"]) if batch_entries else 0
    hb_state = {"counter": 0}
    print(
        f"[{seq_name}] batch_begin size={len(batch_entries)} frames={first_fid:06d}-{last_fid:06d}",
        flush=True,
    )
    _emit_precompute_heartbeat(
        hb_state,
        seq_name=seq_name,
        batch_idx=batch_idx,
        frame_start=first_fid,
        frame_end=last_fid,
        views=batch_views,
        phase="load_batch_start",
        items_done=0,
        items_total=len(batch_entries),
        phase_started_at=batch_t0,
        total_started_at=batch_t0,
    )
    save_queue._drain_completed()
    with torch.no_grad():
        teacher_t0 = time.perf_counter()
        _emit_precompute_heartbeat(
            hb_state,
            seq_name=seq_name,
            batch_idx=batch_idx,
            frame_start=first_fid,
            frame_end=last_fid,
            views=batch_views,
            phase="teacher_forward_start",
            items_done=0,
            items_total=len(batch_entries),
            phase_started_at=teacher_t0,
            total_started_at=batch_t0,
        )
        if preloaded_batch is None:
            batched_geom = teacher.forward_batch(batched_img_paths)
        else:
            batched_geom = teacher.forward_prepared_batch(preloaded_batch)
    teacher_dt = time.perf_counter() - teacher_t0
    print(
        f"[{seq_name}] batch_teacher_done frames={first_fid:06d}-{last_fid:06d} "
        f"teacher_forward_sec={teacher_dt:.2f}",
        flush=True,
    )
    _emit_precompute_heartbeat(
        hb_state,
        seq_name=seq_name,
        batch_idx=batch_idx,
        frame_start=first_fid,
        frame_end=last_fid,
        views=batch_views,
        phase="teacher_forward_done",
        items_done=len(batch_entries),
        items_total=len(batch_entries),
        phase_started_at=teacher_t0,
        total_started_at=batch_t0,
        extra={"teacher_forward_sec": round(float(teacher_dt), 4)},
    )
    if len(batched_geom) != len(batch_entries):
        raise RuntimeError(
            f"teacher batch output mismatch: entries={len(batch_entries)} outputs={len(batched_geom)}"
        )

    support_generation_batch = [{
        "gate": None,
        "fg": None,
        "mask_source": "",
        "region_mode_resolved": str(mv_support_region_mode),
    } for _ in batch_entries]
    mv_support_diag_by_idx: dict[int, dict] = {}
    if mv_support_on:
        target_hw = (
            int(batched_geom[0]["depth"].shape[1]),
            int(batched_geom[0]["depth"].shape[2]),
        )
        gate_prep_t0 = time.perf_counter()
        support_generation_batch = _prepare_support_generation_batch(
            batch_entries,
            zju_root=zju_root,
            seq_root=seq_root,
            target_hw=target_hw,
            region_mode=mv_support_region_mode,
            fg_mask_source=mv_support_fg_mask_source,
            fg_erode_px=mv_support_fg_erode_px,
        )
        gate_prep_dt = time.perf_counter() - gate_prep_t0
        print(
            f"[{seq_name}] batch_gate_ready frames={first_fid:06d}-{last_fid:06d} gate_prepare_sec={gate_prep_dt:.2f}",
            flush=True,
        )
        groups: dict[str, list[int]] = {}
        for idx, gate_info in enumerate(support_generation_batch):
            key = "with_gate" if (gate_info.get("gate") is not None) else "no_gate"
            groups.setdefault(key, []).append(idx)
        for idx_list in groups.values():
            geoms_group = [batched_geom[idx] for idx in idx_list]
            gates_group = None
            if support_generation_batch[idx_list[0]].get("gate") is not None:
                gates_group = [support_generation_batch[idx]["gate"] for idx in idx_list]
            diag_group = _compute_mv_support_diag_batch_adaptive(
                geoms_group,
                generation_gates=gates_group,
                tol_abs=mv_support_tol_abs,
                tol_rel=mv_support_tol_rel,
                stride=mv_support_stride,
                conf_valid_floor=mv_conf_valid_floor,
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=f"[{seq_name}] batch={batch_idx} mv_support",
            )
            for local_idx, diag in zip(idx_list, diag_group):
                mv_support_diag_by_idx[local_idx] = diag

    for batch_pos, (entry, geom) in enumerate(zip(batch_entries, batched_geom)):
        frame_t0 = time.perf_counter()
        fid = int(entry["fid"])
        frame_idx = int(entry["frame_idx"])
        out_npz = str(entry["out_npz"])
        cam_names = list(entry["cam_names"])
        img_paths = list(entry["img_paths"])

        depth_conf_out = geom["depth_conf"]
        depth_conf_raw = None
        mv_support = None
        mv_weight = None
        support_generation_gate = None
        support_generation_fg = None
        support_generation_mask_source = ""
        support_generation_region_mode_resolved = str(mv_support_region_mode)
        if mv_support_on:
            try:
                mv_t0 = time.perf_counter()
                gate_info = support_generation_batch[batch_pos]
                support_generation_gate = gate_info.get("gate")
                support_generation_fg = gate_info.get("fg")
                support_generation_mask_source = str(gate_info.get("mask_source") or "")
                support_generation_region_mode_resolved = str(gate_info.get("region_mode_resolved") or mv_support_region_mode)
                if support_generation_gate is not None:
                    support_generation_gate = torch.as_tensor(
                        support_generation_gate,
                        device=geom["pointmap"].device,
                        dtype=torch.float32,
                    )
                if support_generation_fg is not None:
                    support_generation_fg = torch.as_tensor(
                        support_generation_fg,
                        device=geom["pointmap"].device,
                        dtype=torch.float32,
                    )
                _emit_precompute_heartbeat(
                    hb_state,
                    seq_name=seq_name,
                    batch_idx=batch_idx,
                    frame_start=first_fid,
                    frame_end=last_fid,
                    views=len(cam_names),
                    phase="mv_support_start",
                    items_done=0,
                    items_total=len(cam_names),
                    frame_id=fid,
                    phase_started_at=mv_t0,
                    total_started_at=batch_t0,
                )
                depth_conf_raw = depth_conf_out.detach().clone()
                mv_support_diag = mv_support_diag_by_idx.get(batch_pos)
                if mv_support_diag is None:
                    mv_support_diag = _build_multiview_support(
                        point_world=geom["pointmap"],
                        depth=geom["depth"],
                        depth_conf=depth_conf_out,
                        extrinsic_w2c=geom["extrinsic"],
                        intrinsic=geom["intrinsic"],
                        tol_abs=mv_support_tol_abs,
                        tol_rel=mv_support_tol_rel,
                        stride=mv_support_stride,
                        conf_valid_floor=mv_conf_valid_floor,
                        generation_gate=support_generation_gate,
                        heartbeat_sec=heartbeat_sec,
                        heartbeat_prefix=f"[{seq_name}] frame={fid:06d} mv_support",
                        heartbeat_cb=lambda **kwargs: _emit_precompute_heartbeat(
                            hb_state,
                            seq_name=seq_name,
                            batch_idx=batch_idx,
                            frame_start=first_fid,
                            frame_end=last_fid,
                            views=len(cam_names),
                            frame_id=fid,
                            total_started_at=batch_t0,
                            **kwargs,
                        ),
                        return_diag=True,
                    )
                mv_support = mv_support_diag["support"]
                depth_conf_out, mv_weight = _apply_support_to_depth_conf(
                    depth_conf=depth_conf_out,
                    support01=mv_support,
                    mode=mv_support_mode,
                    floor=mv_support_floor,
                    gamma=mv_support_gamma,
                    clip_thr=mv_support_clip_thr,
                    clip_floor=mv_support_clip_floor,
                    hard_thr=mv_support_hard_thr,
                )
                depth_conf_out, fg_preserve_stats = _preserve_fg_depth_conf(
                    depth_conf_raw=depth_conf_raw,
                    depth_conf_after_support=depth_conf_out,
                    fg_mask=support_generation_fg,
                    region_mode_resolved=support_generation_region_mode_resolved,
                    preserve_px=mv_support_fg_preserve_px,
                )
                support_stats_extra = {}
                support_stats_extra.update(fg_preserve_stats)
                conf_delta = (depth_conf_out - depth_conf_raw).detach()
                support_valid = mv_support_diag["valid"].detach()
                support_cover = mv_support_diag["cover"].detach()
                support_valid_count = int(support_valid.sum().item())
                support_stats_extra["mv_support_raw_mean"] = round(float(mv_support.mean().item()), 6)
                support_stats_extra["mv_support_valid_ratio"] = round(float(support_valid.float().mean().item()), 6)
                support_stats_extra["mv_support_pair_count_eff"] = round(float(support_cover[support_valid].mean().item()), 6) if support_valid_count > 0 else float("nan")
                support_stats_extra["mv_support_conf_mean"] = round(float(depth_conf_raw[support_valid].mean().item()), 6) if support_valid_count > 0 else float("nan")
                support_stats_extra["mv_support_nan_ratio"] = round(float((~torch.isfinite(mv_support)).float().mean().item()), 6)
                support_stats_extra["depth_conf_delta_mean"] = round(float(conf_delta.mean().item()), 6)
                support_stats_extra["mv_support_generation_region_mode"] = str(support_generation_region_mode_resolved)
                if support_generation_mask_source:
                    support_stats_extra["mv_support_generation_fg_mask_source"] = str(support_generation_mask_source)
                if support_generation_fg is not None:
                    fg_mask = support_generation_fg > 0.5
                    bg_mask = ~fg_mask
                    if int(fg_mask.sum().item()) > 0:
                        support_stats_extra["mv_support_fg_mean"] = round(float(mv_support[fg_mask].mean().item()), 6)
                        support_stats_extra["mv_support_fg_valid_ratio"] = round(float(support_valid[fg_mask].float().mean().item()), 6)
                        support_stats_extra["depth_conf_delta_fg_mean"] = round(float(conf_delta[fg_mask].mean().item()), 6)
                    if int(bg_mask.sum().item()) > 0:
                        support_stats_extra["mv_support_bg_mean"] = round(float(mv_support[bg_mask].mean().item()), 6)
                        support_stats_extra["mv_support_bg_valid_ratio"] = round(float(support_valid[bg_mask].float().mean().item()), 6)
                        support_stats_extra["depth_conf_delta_bg_mean"] = round(float(conf_delta[bg_mask].mean().item()), 6)
                mv_dt = time.perf_counter() - mv_t0
                stats_suffix = ""
                if "mv_support_fg_mean" in support_stats_extra:
                    stats_suffix += f" fg_mean={support_stats_extra['mv_support_fg_mean']:.4f}"
                if "mv_support_bg_mean" in support_stats_extra:
                    stats_suffix += f" bg_mean={support_stats_extra['mv_support_bg_mean']:.4f}"
                if "depth_conf_delta_fg_mean" in support_stats_extra:
                    stats_suffix += f" conf_delta_fg={support_stats_extra['depth_conf_delta_fg_mean']:.4f}"
                if "depth_conf_delta_bg_mean" in support_stats_extra:
                    stats_suffix += f" conf_delta_bg={support_stats_extra['depth_conf_delta_bg_mean']:.4f}"
                if "depth_conf_fg_final_mean" in support_stats_extra:
                    stats_suffix += f" fg_conf_final={support_stats_extra['depth_conf_fg_final_mean']:.4f}"
                print(
                    f"[{seq_name}] frame={fid:06d} mv_support_done sec={mv_dt:.2f} "
                    f"mean={float(mv_support.mean().item()):.4f} "
                    f"p10={float(torch.quantile(mv_support.reshape(-1), 0.10).item()):.4f} "
                    f"p90={float(torch.quantile(mv_support.reshape(-1), 0.90).item()):.4f}"
                    f"{stats_suffix}",
                    flush=True,
                )
                _emit_precompute_heartbeat(
                    hb_state,
                    seq_name=seq_name,
                    batch_idx=batch_idx,
                    frame_start=first_fid,
                    frame_end=last_fid,
                    views=len(cam_names),
                    phase="mv_support_done",
                    items_done=len(cam_names),
                    items_total=len(cam_names),
                    frame_id=fid,
                    phase_started_at=mv_t0,
                    total_started_at=batch_t0,
                    extra={
                        "mv_support_sec": round(float(mv_dt), 4),
                        "mv_support_mean": round(float(mv_support.mean().item()), 6),
                        **support_stats_extra,
                    },
                )
            except Exception as e:
                print(f"[warn] mv-support failed at {seq_name}/frame_{fid:06d}: {e}", flush=True)
                depth_conf_out = geom["depth_conf"]
                depth_conf_raw = None
                mv_support = None
                mv_weight = None

        save_dict = {
            "cam_names": np.array(cam_names),
            "img_paths": np.array(img_paths),
            "pointmap_source": np.array([pointmap_source]),
            "pointmap_frame": np.array([str(geom.get("pointmap_frame", point_head_frame))]),
            "unproject_impl": np.array([str(geom.get("unproject_impl", unproject_impl))]),
            "mv_support_generation_region_mode": np.array([str(support_generation_region_mode_resolved)]),
            "depth": to_numpy(geom["depth"]),
            "depth_conf": to_numpy(depth_conf_out),
            "pointmap": to_numpy(geom["pointmap"]),
            "extrinsic": to_numpy(geom["extrinsic"]),
            "intrinsic": to_numpy(geom["intrinsic"]),
        }
        if support_generation_mask_source:
            save_dict["mv_support_generation_fg_mask_source"] = np.array([str(support_generation_mask_source)])
        if mv_support is not None:
            for k, v in support_stats_extra.items():
                if isinstance(v, (int, float)):
                    save_dict[k] = np.array([float(v)], dtype=np.float32)
                elif isinstance(v, str):
                    save_dict[k] = np.array([v])
        if mv_support is not None and mv_support_save:
            save_dict["mv_support"] = to_numpy(mv_support)
            if mv_weight is not None:
                save_dict["mv_support_weight"] = to_numpy(mv_weight)
        if depth_conf_raw is not None and mv_support_save_raw_conf:
            save_dict["depth_conf_raw"] = to_numpy(depth_conf_raw)
        save_t0 = time.perf_counter()
        support_stats_json_path = ""
        if mv_support is not None:
            sidecar = Path(out_npz).with_suffix(".support_stats.json")
            support_stats_json_path = str(sidecar)
        _emit_precompute_heartbeat(
            hb_state,
            seq_name=seq_name,
            batch_idx=batch_idx,
            frame_start=first_fid,
            frame_end=last_fid,
            views=len(cam_names),
            phase="save_start",
            items_done=0,
            items_total=1,
            frame_id=fid,
            phase_started_at=save_t0,
            total_started_at=batch_t0,
        )
        def _on_save_done(save_dt: float, *, _fid=fid, _save_t0=save_t0, _frame_t0=frame_t0, _views=len(cam_names)):
            frame_dt = time.perf_counter() - _frame_t0
            print(
                f"[{seq_name}] frame={_fid:06d} save_done sec={save_dt:.2f} total_frame_sec={frame_dt:.2f}",
                flush=True,
            )
            _emit_precompute_heartbeat(
                hb_state,
                seq_name=seq_name,
                batch_idx=batch_idx,
                frame_start=first_fid,
                frame_end=last_fid,
                views=_views,
                phase="save_done",
                items_done=1,
                items_total=1,
                frame_id=_fid,
                phase_started_at=_save_t0,
                total_started_at=batch_t0,
                extra={
                    "save_sec": round(float(save_dt), 4),
                    "frame_total_sec": round(float(frame_dt), 4),
                },
            )

        save_queue.submit(
            out_npz=out_npz,
            save_dict=save_dict,
            on_done=_on_save_done,
            support_stats_json_path=support_stats_json_path,
            support_stats_payload=support_stats_extra if mv_support is not None else None,
        )
    batch_dt = time.perf_counter() - batch_t0
    print(
        f"[{seq_name}] batch_done size={len(batch_entries)} frames={first_fid:06d}-{last_fid:06d} "
        f"total_sec={batch_dt:.2f}",
        flush=True,
    )
    _emit_precompute_heartbeat(
        hb_state,
        seq_name=seq_name,
        batch_idx=batch_idx,
        frame_start=first_fid,
        frame_end=last_fid,
        views=batch_views,
        phase="batch_done",
        items_done=len(batch_entries),
        items_total=len(batch_entries),
        phase_started_at=batch_t0,
        total_started_at=batch_t0,
        extra={"batch_total_sec": round(float(batch_dt), 4)},
    )


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc or "")
    if "out of memory" in msg.lower():
        return True
    return isinstance(exc, torch.OutOfMemoryError)


def main():
    zju_root = (os.environ.get("VGGT_ZJU_ROOT", "") or "").strip() or ZJU_ROOT
    ckpt_path = _resolve_ckpt_path()
    out_dir = (os.environ.get("VGGT_OUT_DIR", "") or "").strip() or "vggt_geom"

    seq_list = _split_list(os.environ.get("VGGT_SEQ_NAMES", "")) or list(SEQ_LIST)
    cam_list = _split_list(os.environ.get("VGGT_CAM_NAMES", "")) or list(SELECT_CAMERAS)

    max_raw = (os.environ.get("VGGT_MAX_FRAMES", "") or "").strip()
    max_frames = int(max_raw) if max_raw.isdigit() else 0
    pointmap_source = (os.environ.get("VGGT_POINTMAP_SOURCE", "") or "").strip() or "depth_unproject"
    point_head_frame = (os.environ.get("VGGT_POINT_HEAD_FRAME", "") or "").strip() or "auto"
    unproject_impl = (os.environ.get("VGGT_UNPROJECT_IMPL", "") or "").strip() or "legacy"
    mv_support_on = _env_bool("VGGT_MV_SUPPORT_ON", False)
    mv_support_tol_abs = _env_float("VGGT_MV_SUPPORT_TOL_ABS", 0.06)
    mv_support_tol_rel = _env_float("VGGT_MV_SUPPORT_TOL_REL", 0.10)
    mv_support_stride = _env_int("VGGT_MV_SUPPORT_STRIDE", 2)
    mv_support_mode = str(os.environ.get("VGGT_MV_SUPPORT_MODE", "") or "").strip().lower() or "linear"
    mv_support_floor = _env_float("VGGT_MV_SUPPORT_FLOOR", 0.05)
    mv_support_gamma = _env_float("VGGT_MV_SUPPORT_GAMMA", 1.0)
    mv_support_clip_thr = _env_float("VGGT_MV_SUPPORT_CLIP_THR", 0.20)
    mv_support_clip_floor = _env_float("VGGT_MV_SUPPORT_CLIP_FLOOR", 0.30)
    mv_support_hard_thr = _env_float("VGGT_MV_SUPPORT_HARD_THR", -1.0)
    mv_conf_valid_floor = _env_float("VGGT_MV_CONF_VALID_FLOOR", 0.02)
    mv_support_save = _env_bool("VGGT_MV_SUPPORT_SAVE", False)
    mv_support_save_raw_conf = _env_bool("VGGT_MV_SUPPORT_SAVE_RAW_CONF", False)
    mv_support_region_mode = str(os.environ.get("VGGT_MV_SUPPORT_REGION_MODE", "") or "").strip().lower() or "auto"
    mv_support_fg_mask_source = str(os.environ.get("VGGT_MV_SUPPORT_FG_MASK_SOURCE", "") or "").strip().lower() or "auto"
    mv_support_fg_erode_px = _env_int("VGGT_MV_SUPPORT_FG_ERODE_PX", 5)
    mv_support_fg_preserve_px = _env_int("VGGT_MV_SUPPORT_FG_PRESERVE_PX", 5)
    precompute_batch_frames = max(1, _env_int("VGGT_PRECOMPUTE_BATCH_FRAMES", 6))
    precompute_heartbeat_sec = max(0.0, _env_float("VGGT_PRECOMPUTE_HEARTBEAT_SEC", 30.0))
    precompute_save_workers = max(0, _env_int("VGGT_PRECOMPUTE_SAVE_WORKERS", 4))
    precompute_max_pending_saves = max(1, _env_int("VGGT_PRECOMPUTE_MAX_PENDING_SAVES", 8))
    precompute_save_compressed = _env_bool("VGGT_PRECOMPUTE_SAVE_COMPRESSED", False)
    precompute_prefetch = _env_bool("VGGT_PRECOMPUTE_PREFETCH", True)
    precompute_mv_support_view_chunk = max(1, _env_int("VGGT_PRECOMPUTE_MV_SUPPORT_VIEW_CHUNK", 8))
    precompute_tf32 = _env_bool("VGGT_TF32", True)
    precompute_amp = _env_bool("VGGT_AMP", True)
    precompute_strict_deterministic = _env_bool("VGGT_STRICT_DETERMINISTIC", False)

    device_env = (os.environ.get("VGGT_DEVICE", "") or "").strip().lower()
    if device_env in {"", "auto", "none"}:
        device_env = ""
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = device_env or ("cuda" if use_cuda else "cpu")
    if str(device).startswith("cuda") and torch.cuda.device_count() <= 0:
        device = "cpu"

    print(
        "[dev]",
        "resolved_device=",
        device,
        "cuda_available=",
        torch.cuda.is_available(),
        "cuda_count=",
        torch.cuda.device_count(),
        "pointmap_source=",
        pointmap_source,
        "point_head_frame=",
        point_head_frame,
        "unproject_impl=",
        unproject_impl,
        "mv_support_on=",
        mv_support_on,
        "mv_support_tol_abs=",
        mv_support_tol_abs,
        "mv_support_tol_rel=",
        mv_support_tol_rel,
        "mv_support_stride=",
        mv_support_stride,
        "mv_support_mode=",
        mv_support_mode,
        "mv_support_floor=",
        mv_support_floor,
        "mv_support_gamma=",
        mv_support_gamma,
        "mv_support_clip_thr=",
        mv_support_clip_thr,
        "mv_support_clip_floor=",
        mv_support_clip_floor,
        "mv_support_hard_thr=",
        mv_support_hard_thr,
        "mv_conf_valid_floor=",
        mv_conf_valid_floor,
        "mv_support_region_mode=",
        mv_support_region_mode,
        "mv_support_fg_mask_source=",
        mv_support_fg_mask_source,
        "mv_support_fg_erode_px=",
        mv_support_fg_erode_px,
        "mv_support_fg_preserve_px=",
        mv_support_fg_preserve_px,
        "precompute_batch_frames=",
        precompute_batch_frames,
        "precompute_save_workers=",
        precompute_save_workers,
        "precompute_max_pending_saves=",
        precompute_max_pending_saves,
        "precompute_save_compressed=",
        precompute_save_compressed,
        "precompute_prefetch=",
        precompute_prefetch,
        "precompute_mv_support_view_chunk=",
        precompute_mv_support_view_chunk,
        "precompute_heartbeat_sec=",
        precompute_heartbeat_sec,
        "precompute_tf32=",
        precompute_tf32,
        "precompute_amp=",
        precompute_amp,
        "precompute_strict_deterministic=",
        precompute_strict_deterministic,
    )
    teacher = VGGTGeomTeacher(
        ckpt_path,
        device=device,
        pointmap_source=pointmap_source,
        point_head_frame=point_head_frame,
        unproject_impl=unproject_impl,
        amp=precompute_amp,
        tf32=precompute_tf32,
        deterministic=precompute_strict_deterministic,
    )
    for seq_name in seq_list:
        seq_root = osp.join(zju_root, seq_name)
        if not osp.isdir(seq_root):
            print("[warn] seq not found:", seq_root)
            continue

        out_root = osp.join(seq_root, out_dir)
        os.makedirs(out_root, exist_ok=True)
        save_queue = _AsyncSaveQueue(
            max_workers=precompute_save_workers,
            max_pending=precompute_max_pending_saves,
            compressed=precompute_save_compressed,
        )

        seq = ZJUMocapSeq(seq_root, cam_names=(cam_list if cam_list else None))
        print(f"[{seq_name}] frames={seq.num_frames()} out={out_root}")

        num_frames = seq.num_frames()
        if max_frames > 0:
            num_frames = min(num_frames, int(max_frames))

        scheduled_batches: list[list[dict]] = []
        batch_entries: list[dict] = []

        def _flush_batch_adaptive(entries: list[dict], batch_idx: int, preloaded_batch: dict | None = None) -> None:
            if not entries:
                return
            first_fid = int(entries[0]["fid"])
            last_fid = int(entries[-1]["fid"])
            try:
                _process_frame_batch(
                    teacher=teacher,
                    batch_entries=entries,
                    seq_name=seq_name,
                    batch_idx=batch_idx,
                    seq_root=seq_root,
                    zju_root=zju_root,
                    pointmap_source=pointmap_source,
                    point_head_frame=point_head_frame,
                    unproject_impl=unproject_impl,
                    mv_support_on=mv_support_on,
                    mv_support_tol_abs=mv_support_tol_abs,
                    mv_support_tol_rel=mv_support_tol_rel,
                    mv_support_stride=mv_support_stride,
                    mv_support_mode=mv_support_mode,
                    mv_support_floor=mv_support_floor,
                    mv_support_gamma=mv_support_gamma,
                    mv_support_clip_thr=mv_support_clip_thr,
                    mv_support_clip_floor=mv_support_clip_floor,
                    mv_support_hard_thr=mv_support_hard_thr,
                    mv_conf_valid_floor=mv_conf_valid_floor,
                    mv_support_save=mv_support_save,
                    mv_support_save_raw_conf=mv_support_save_raw_conf,
                    mv_support_region_mode=mv_support_region_mode,
                    mv_support_fg_mask_source=mv_support_fg_mask_source,
                    mv_support_fg_erode_px=mv_support_fg_erode_px,
                    mv_support_fg_preserve_px=mv_support_fg_preserve_px,
                    heartbeat_sec=precompute_heartbeat_sec,
                    save_queue=save_queue,
                    preloaded_batch=preloaded_batch,
                )
            except Exception as exc:
                if _is_cuda_oom(exc) and len(entries) > 1:
                    left_n = max(1, len(entries) // 2)
                    right_n = len(entries) - left_n
                    print(
                        f"[{seq_name}] precompute_batch oom size={len(entries)} "
                        f"frames={first_fid:06d}-{last_fid:06d} "
                        f"-> split {left_n}+{right_n}"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _flush_batch_adaptive(entries[:left_n], batch_idx)
                    _flush_batch_adaptive(entries[left_n:], batch_idx)
                    return
                raise

        def flush_batch() -> None:
            if not batch_entries:
                return
            scheduled_batches.append(list(batch_entries))
            batch_entries.clear()

        for frame_idx in tqdm(range(num_frames), desc=f"{seq_name}:{out_dir}"):
            fid = seq.get_frame_id(frame_idx)
            out_npz = osp.join(out_root, f"frame_{fid:06d}.npz")
            if osp.exists(out_npz):
                continue

            cam2path = seq.get_frame_paths(frame_idx)
            if not cam2path:
                continue

            cam_names = sorted(cam2path.keys())
            img_paths = [osp.relpath(cam2path[c], seq_root).replace("\\", "/") for c in cam_names]
            img_paths = [_resolve_img_path(p, zju_root, seq_root) for p in img_paths]
            if batch_entries:
                prev_cam_names = batch_entries[0]["cam_names"]
                if cam_names != prev_cam_names:
                    flush_batch()
            batch_entries.append({
                "frame_idx": frame_idx,
                "fid": fid,
                "out_npz": out_npz,
                "cam_names": cam_names,
                "img_paths": img_paths,
            })
            if len(batch_entries) >= precompute_batch_frames:
                flush_batch()

        flush_batch()
        if scheduled_batches:
            prefetch_pool = (
                ThreadPoolExecutor(max_workers=1)
                if (precompute_prefetch and len(scheduled_batches) > 1)
                else None
            )
            prefetch_future = None
            try:
                if scheduled_batches:
                    first_batch_paths = [entry["img_paths"] for entry in scheduled_batches[0]]
                    if prefetch_pool is not None:
                        prefetch_future = prefetch_pool.submit(teacher.prepare_batch_inputs, first_batch_paths)
                    else:
                        prefetch_future = None
                for batch_index, entries in enumerate(scheduled_batches, start=1):
                    first_fid = int(entries[0]["fid"])
                    last_fid = int(entries[-1]["fid"])
                    print(
                        f"[{seq_name}] precompute_batch idx={batch_index} size={len(entries)} "
                        f"frames={first_fid:06d}-{last_fid:06d}",
                        flush=True,
                    )
                    preloaded_batch = None
                    if prefetch_future is not None:
                        preloaded_batch = prefetch_future.result()
                    elif precompute_prefetch:
                        preloaded_batch = teacher.prepare_batch_inputs([entry["img_paths"] for entry in entries])

                    next_entries = scheduled_batches[batch_index] if batch_index < len(scheduled_batches) else None
                    if (prefetch_pool is not None) and (next_entries is not None):
                        next_paths = [entry["img_paths"] for entry in next_entries]
                        prefetch_future = prefetch_pool.submit(teacher.prepare_batch_inputs, next_paths)
                    else:
                        prefetch_future = None

                    _flush_batch_adaptive(entries, batch_index, preloaded_batch=preloaded_batch)
            finally:
                if prefetch_pool is not None:
                    prefetch_pool.shutdown(wait=False, cancel_futures=True)
        save_queue.flush()

        print(f"[{seq_name}] done, saved to {out_root}")


if __name__ == "__main__":
    main()
