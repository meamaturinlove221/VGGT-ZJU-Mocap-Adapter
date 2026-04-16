import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from finetune_vggt_pseudo import (  # noqa: E402
    _build_fg_boundary_band_mask,
    _build_fg_bbox_mask,
    _build_fg_outside_ring_mask,
    _build_largest_component_soft_bias,
    _build_front_depth_soft_bias,
    _build_fg_structure_region_mask,
    _build_fg_structure_target_edge_support_mask,
    _build_inside_distance_weight,
    _summarize_main_support_components,
    _summarize_main_support_depth_modes,
    _infer_mask_path_from_image,
    _normalize_mask_binary_np,
    _resolve_img_path,
    _sobel_grad_mag,
)


PANEL_W = 148
PANEL_H = 110
PANEL_GAP = 8
FRAME_GAP = 16
ROW_GAP = 28
TEXT_H = 28
MARGIN = 18


@dataclass
class StageSpec:
    label: str
    lambda_edge: float
    lambda_ring: float
    bbox_margin_px: int
    bbox_min_side_px: int
    ring_px: int
    region_mode: str = "bbox"
    region_erode_px: int = 0
    warmup_steps: int = 0
    edge_support_mode: str = "off"
    edge_support_quantile: float = 0.0
    edge_support_min_px: int = 32
    edge_weight_mode: str = "uniform"
    boundary_falloff_px: int = 0
    component_bias_mode: str = "off"
    component_bias_threshold_ratio: float = 0.25
    component_bias_other_scale: float = 1.0
    front_depth_bias_mode: str = "off"
    front_depth_bias_tau: float = 0.75
    front_depth_bias_center_quantile: float = 0.55


@dataclass
class FrameDiag:
    step_label: str
    npz_name: str
    rgb_u8: np.ndarray
    fg_mask: np.ndarray
    bbox_mask: np.ndarray
    structure_mask: np.ndarray
    boundary_probe_mask: np.ndarray
    ring_mask: np.ndarray
    edge_heatmap: np.ndarray
    edge_support_mask: np.ndarray
    main_support_mask: np.ndarray
    main_support_active_mask: np.ndarray
    main_support_largest_component_mask: np.ndarray
    bbox_active_ratio: float
    structure_active_ratio: float
    boundary_probe_active_ratio: float
    ring_active_ratio: float
    edge_support_active_ratio: float
    main_support_active_ratio: float
    main_support_density_in_structure: float
    main_support_mean_weight: float
    boundary_distance_weight_share: float
    main_support_component_count: float
    main_support_largest_component_share: float
    main_support_top2_component_share: float
    main_support_centroid_distance_mean: float
    main_support_component_bias_weight_share: float
    main_support_depth_mode_count: float
    main_support_back_mode_share: float
    main_support_front_back_gap: float
    main_support_depth_hist_peak_ratio: float
    main_support_secondary_risk: float
    main_support_front_depth_bias_weight_share: float
    target_edge_structure_share: float
    target_edge_boundary_probe_share: float
    target_edge_outside_ring_share: float
    target_edge_support_share: float
    main_support_boundary_probe_overlap: float
    main_support_outside_ring_overlap: float
    edge_active: bool


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_stage(raw: str) -> StageSpec:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) not in (6, 9, 12, 14, 17, 20):
        raise ValueError(f"invalid --stage spec: {raw}")
    spec = StageSpec(
        label=parts[0],
        lambda_edge=float(parts[1]),
        lambda_ring=float(parts[2]),
        bbox_margin_px=max(0, int(parts[3])),
        bbox_min_side_px=max(1, int(parts[4])),
        ring_px=max(0, int(parts[5])),
    )
    if len(parts) >= 9:
        spec.region_mode = parts[6] or "bbox"
        spec.region_erode_px = max(0, int(parts[7]))
        spec.warmup_steps = max(0, int(parts[8]))
    if len(parts) >= 12:
        spec.edge_support_mode = parts[9] or "off"
        spec.edge_support_quantile = max(0.0, float(parts[10]))
        spec.edge_support_min_px = max(1, int(parts[11]))
    if len(parts) >= 14:
        spec.edge_weight_mode = parts[12] or "uniform"
        spec.boundary_falloff_px = max(0, int(parts[13]))
    if len(parts) >= 17:
        spec.component_bias_mode = parts[14] or "off"
        spec.component_bias_threshold_ratio = max(0.0, float(parts[15]))
        spec.component_bias_other_scale = max(0.0, min(1.0, float(parts[16])))
    if len(parts) >= 20:
        spec.front_depth_bias_mode = parts[17] or "off"
        spec.front_depth_bias_tau = max(1e-3, float(parts[18]))
        spec.front_depth_bias_center_quantile = max(0.0, min(1.0, float(parts[19])))
    return spec


def resolve_seq_name(contract: dict[str, Any], override: str) -> str:
    if override:
        return override
    raw = str(contract.get("seq_names") or "").strip()
    return raw.replace(",", " ").split()[0] if raw else "CoreView_390"


def resolve_geom_subdir(contract: dict[str, Any]) -> str:
    raw = str(contract.get("pseudo_geom_subdir") or contract.get("geom_subdir") or "").strip()
    return raw or "vggt_geom"


def resolve_candidate_geom_subdir(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("geom_subdir", "best_geom_subdir", "pseudo_geom_subdir"):
        raw = str(candidate.get(key) or "").strip()
        if raw:
            return raw
    return ""


def pick_existing_geom_subdir(
    zju_root: str,
    seq_name: str,
    contract: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> str:
    seq_dir = os.path.join(zju_root, seq_name)
    raw_candidates: list[str] = []
    for value in (
        resolve_candidate_geom_subdir(candidate),
        resolve_geom_subdir(contract),
    ):
        value = str(value or "").strip()
        if value and value not in raw_candidates:
            raw_candidates.append(value)
    for geom_subdir in raw_candidates:
        gdir = os.path.join(seq_dir, geom_subdir)
        if os.path.isdir(gdir):
            return geom_subdir
    if os.path.isdir(seq_dir):
        fallbacks = sorted(
            [
                name
                for name in os.listdir(seq_dir)
                if name.startswith("vggt_geom") and os.path.isdir(os.path.join(seq_dir, name))
            ],
            reverse=True,
        )
        if fallbacks:
            return fallbacks[0]
    joined = ", ".join(raw_candidates) if raw_candidates else "<none>"
    raise FileNotFoundError(
        f"failed to resolve geom_subdir under {seq_dir}; tried {joined}"
    )


def _list_npz_files(gdir: str) -> list[str]:
    if not os.path.isdir(gdir):
        return []
    return sorted(
        [
            os.path.join(gdir, name)
            for name in os.listdir(gdir)
            if name.lower().endswith(".npz")
        ]
    )


def resolve_npz_paths(
    zju_root: str,
    seq_name: str,
    geom_subdir: str,
    frame_indices: list[int],
) -> tuple[str, list[str]]:
    seq_dir = os.path.join(zju_root, seq_name)
    need_count = (max(frame_indices) + 1) if frame_indices else 0
    requested_gdir = os.path.join(seq_dir, geom_subdir)
    files = _list_npz_files(requested_gdir)
    resolved_geom_subdir = geom_subdir
    if len(files) < need_count:
        fallback_rows: list[tuple[int, int, float, str, list[str]]] = []
        if os.path.isdir(seq_dir):
            for name in os.listdir(seq_dir):
                gdir = os.path.join(seq_dir, name)
                if not os.path.isdir(gdir) or not name.startswith("vggt_geom"):
                    continue
                cand_files = _list_npz_files(gdir)
                if len(cand_files) < need_count:
                    continue
                score_ft = 1 if "ft_lr" in name else 0
                score_requested = 1 if name == geom_subdir else 0
                fallback_rows.append((score_requested, score_ft, os.path.getmtime(gdir), name, cand_files))
        if not fallback_rows:
            raise FileNotFoundError(
                f"no geom_subdir under {seq_dir} has at least {need_count} npz files; requested={geom_subdir}"
            )
        fallback_rows.sort(key=lambda item: (item[0], item[1], len(item[4]), item[2], item[3]), reverse=True)
        resolved_geom_subdir = fallback_rows[0][3]
        files = fallback_rows[0][4]
    out: list[str] = []
    for idx in frame_indices:
        if idx < 0 or idx >= len(files):
            raise IndexError(f"frame index {idx} out of range for {os.path.join(seq_dir, resolved_geom_subdir)}")
        out.append(files[idx])
    return resolved_geom_subdir, out


def resolve_camera_view(
    npz_path: str,
    camera: str,
    zju_root: str,
    seq_name: str,
) -> tuple[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        img_paths = [str(x) for x in data["img_paths"].tolist()]
        cam_names = [str(x) for x in data["cam_names"].tolist()] if "cam_names" in data else []
        depth = np.asarray(data["depth"])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise RuntimeError(f"unexpected depth shape in {npz_path}: {tuple(depth.shape)}")
    resolved_paths = [_resolve_img_path(path, zju_root, [seq_name]) for path in img_paths]
    if not cam_names:
        cam_names = [os.path.basename(os.path.dirname(path)) for path in resolved_paths]
    try:
        view_idx = cam_names.index(camera)
    except ValueError as exc:
        raise RuntimeError(f"camera {camera} not found in {npz_path}") from exc
    return resolved_paths[view_idx], depth[view_idx]


def build_edge_heatmap(depth_tgt: np.ndarray, bbox_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    if tuple(bbox_mask.shape) != tuple(depth_tgt.shape):
        bbox_mask = np.asarray(
            Image.fromarray((np.asarray(bbox_mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255).resize(
                (int(depth_tgt.shape[1]), int(depth_tgt.shape[0])),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
        bbox_mask = (bbox_mask > 127.5).astype(np.float32)
    depth = torch.from_numpy(np.asarray(depth_tgt, dtype=np.float32))
    bbox = torch.from_numpy(np.asarray(bbox_mask, dtype=np.float32))
    valid = torch.isfinite(depth) & (depth > 1e-6) & (bbox > 0.5)
    if int(valid.sum().item()) < 64:
        return np.zeros_like(depth_tgt, dtype=np.float32), False
    vals = depth[valid]
    center = vals.median()
    mad = (vals - center).abs().median()
    scale = torch.clamp(mad * 1.4826, min=1e-3)
    norm = ((depth - center) / scale).clamp(-3.0, 3.0)
    edge = _sobel_grad_mag(norm[None, None])[0, 0]
    edge = edge * valid.float()
    edge_np = edge.detach().cpu().numpy().astype(np.float32)
    if np.isfinite(edge_np).any():
        mx = float(np.nanmax(edge_np))
        if mx > 1e-6:
            edge_np = edge_np / mx
    return edge_np, True


def align_mask_to_shape(mask01: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask01, dtype=np.float32)
    if tuple(mask.shape) == tuple(shape_hw):
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    resized = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
        (int(shape_hw[1]), int(shape_hw[0])),
        Image.Resampling.NEAREST,
    )
    return (np.asarray(resized, dtype=np.float32) > 127.5).astype(np.float32)


def masked_share(mask01: np.ndarray, weight01: np.ndarray) -> float:
    m = align_mask_to_shape(mask01, np.asarray(weight01).shape[:2])
    w = np.asarray(weight01, dtype=np.float32)
    total = float(np.clip(w, 0.0, None).sum())
    if total <= 1e-8:
        return 0.0
    return float((np.clip(m, 0.0, 1.0) * np.clip(w, 0.0, None)).sum() / total)


def load_frame_diag(
    npz_path: str,
    step_idx: int,
    camera: str,
    zju_root: str,
    seq_name: str,
    stage: StageSpec,
    boundary_probe_px: int,
) -> FrameDiag:
    img_path, depth_tgt = resolve_camera_view(npz_path=npz_path, camera=camera, zju_root=zju_root, seq_name=seq_name)
    mask_path, _ = _infer_mask_path_from_image(img_path, preferred="mask")
    if not mask_path or (not os.path.exists(mask_path)):
        raise FileNotFoundError(f"mask not found for image: {img_path}")
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    fg = _normalize_mask_binary_np(np.asarray(Image.open(mask_path)))
    fg_tensor = torch.from_numpy(fg).unsqueeze(0).unsqueeze(0).float()
    bbox = _build_fg_bbox_mask(
        fg_mask01=fg_tensor,
        margin_px=stage.bbox_margin_px,
        min_side_px=stage.bbox_min_side_px,
    )[0, 0].detach().cpu().numpy().astype(np.float32)
    structure = _build_fg_structure_region_mask(
        fg_mask01=fg_tensor,
        fg_bbox_mask01=torch.from_numpy(bbox).unsqueeze(0).unsqueeze(0),
        region_mode=stage.region_mode,
        region_erode_px=stage.region_erode_px,
    )[0, 0].detach().cpu().numpy().astype(np.float32)
    boundary_probe = _build_fg_boundary_band_mask(
        fg_mask01=fg_tensor,
        erode_px=max(1, int(boundary_probe_px)),
    )[0, 0].detach().cpu().numpy().astype(np.float32)
    ring = _build_fg_outside_ring_mask(
        fg_mask01=fg_tensor,
        ring_px=stage.ring_px,
    )[0, 0].detach().cpu().numpy().astype(np.float32)
    edge_heatmap, edge_active = build_edge_heatmap(depth_tgt=depth_tgt, bbox_mask=bbox)
    bbox_for_support = align_mask_to_shape(bbox, edge_heatmap.shape)
    structure_for_support = align_mask_to_shape(structure, edge_heatmap.shape)
    boundary_probe_for_support = align_mask_to_shape(boundary_probe, edge_heatmap.shape)
    ring_for_support = align_mask_to_shape(ring, edge_heatmap.shape)
    edge_support = np.zeros_like(edge_heatmap, dtype=np.float32)
    if stage.edge_support_mode.strip().lower() != "off":
        edge_tensor = torch.from_numpy(edge_heatmap).unsqueeze(0).unsqueeze(0).float()
        support_mask, _ = _build_fg_structure_target_edge_support_mask(
            target_edge01=edge_tensor,
            valid01=torch.ones_like(edge_tensor),
            fg_structure_region_mask01=torch.from_numpy(structure_for_support).unsqueeze(0).unsqueeze(0).float(),
            view_active01=torch.ones((1, 1), dtype=torch.float32),
            mode=stage.edge_support_mode,
            quantile=float(stage.edge_support_quantile),
            min_support_px=int(stage.edge_support_min_px),
        )
        if support_mask is not None:
            edge_support = support_mask[0, 0].detach().cpu().numpy().astype(np.float32)
    main_support = edge_support.copy() if stage.edge_support_mode.strip().lower() != "off" else structure_for_support.copy()
    if stage.edge_weight_mode.strip().lower() == "target_edge_sqrt":
        active = main_support > 0.5
        if np.any(active):
            scale = float(np.quantile(edge_heatmap[active], 0.95))
            if scale < 1e-6:
                scale = 1.0
            edge_weight = np.sqrt(np.clip(edge_heatmap / scale, 0.0, 1.0)).astype(np.float32)
            main_support = main_support * edge_weight
    pre_boundary_sum = float(np.clip(main_support, 0.0, None).sum())
    if int(stage.boundary_falloff_px) > 0:
        boundary_weight = _build_inside_distance_weight(
            mask01=torch.from_numpy(structure_for_support).unsqueeze(0).unsqueeze(0).float(),
            falloff_px=int(stage.boundary_falloff_px),
        )
        if boundary_weight is not None:
            boundary_weight_np = boundary_weight[0, 0].detach().cpu().numpy().astype(np.float32)
            main_support = main_support * boundary_weight_np
    post_boundary_sum = float(np.clip(main_support, 0.0, None).sum())
    boundary_distance_weight_share = 1.0 if pre_boundary_sum <= 1e-8 else float(post_boundary_sum / pre_boundary_sum)
    component_bias_weight_share = 1.0
    support_tensor = torch.from_numpy(main_support).unsqueeze(0).unsqueeze(0).float()
    if stage.component_bias_mode.strip().lower() == "largest_soft":
        component_bias, component_bias_info = _build_largest_component_soft_bias(
            weight_map01=support_tensor,
            threshold_ratio=float(stage.component_bias_threshold_ratio),
            other_scale=float(stage.component_bias_other_scale),
        )
        component_bias_weight_share = float(component_bias_info.get("main_support_component_bias_weight_share", 1.0))
        if component_bias is not None:
            support_tensor = support_tensor * component_bias.float()
            main_support = support_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    bbox_active = (
        np.isfinite(depth_tgt)
        & (np.asarray(bbox_for_support, dtype=np.float32) > 0.5)
    ).astype(np.float32)
    front_depth_bias_weight_share = 1.0
    if stage.front_depth_bias_mode.strip().lower() == "front_soft":
        front_bias, front_bias_info = _build_front_depth_soft_bias(
            depth_tgt01=torch.from_numpy(np.asarray(depth_tgt, dtype=np.float32)).unsqueeze(0).unsqueeze(0),
            weight_map01=support_tensor,
            bbox_active01=torch.from_numpy(bbox_active).unsqueeze(0).unsqueeze(0).float(),
            mode=stage.front_depth_bias_mode,
            tau=float(stage.front_depth_bias_tau),
            center_quantile=float(stage.front_depth_bias_center_quantile),
            min_active_px=64,
        )
        front_depth_bias_weight_share = float(front_bias_info.get("fg_structure_front_depth_bias_weight_share", 1.0))
        if front_bias is not None:
            support_tensor = support_tensor * front_bias.float()
            main_support = support_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    component_info, largest_mask = _summarize_main_support_components(
        weight_map01=support_tensor,
        threshold_ratio=float(stage.component_bias_threshold_ratio),
    )
    depth_mode_info = _summarize_main_support_depth_modes(
        depth_tgt01=torch.from_numpy(np.asarray(depth_tgt, dtype=np.float32)).unsqueeze(0).unsqueeze(0),
        weight_map01=support_tensor,
        bbox_active01=torch.from_numpy(bbox_active).unsqueeze(0).unsqueeze(0).float(),
        center_quantile=float(stage.front_depth_bias_center_quantile),
        min_active_px=64,
    )
    structure_sum = float(np.clip(structure_for_support, 0.0, 1.0).sum())
    support_sum = float(np.clip(main_support, 0.0, None).sum())
    density_in_structure = 0.0 if structure_sum <= 1e-8 else float(support_sum / structure_sum)
    support_active = (main_support > 1e-6).astype(np.float32)
    structure_active = structure_for_support > 0.5
    mean_weight = 0.0
    if np.any(structure_active):
        mean_weight = float(main_support[structure_active].mean())
    largest_component_mask = np.zeros_like(main_support, dtype=np.float32)
    if largest_mask is not None:
        largest_component_mask = largest_mask[0, 0].detach().cpu().numpy().astype(np.float32)
    return FrameDiag(
        step_label=f"step{step_idx:06d}",
        npz_name=os.path.basename(npz_path),
        rgb_u8=rgb,
        fg_mask=fg.astype(np.float32),
        bbox_mask=bbox,
        structure_mask=structure,
        boundary_probe_mask=boundary_probe,
        ring_mask=ring,
        edge_heatmap=edge_heatmap,
        edge_support_mask=edge_support,
        main_support_mask=main_support,
        main_support_active_mask=support_active,
        main_support_largest_component_mask=largest_component_mask,
        bbox_active_ratio=float(bbox.mean()),
        structure_active_ratio=float(structure.mean()),
        boundary_probe_active_ratio=float(boundary_probe.mean()),
        ring_active_ratio=float(ring.mean()),
        edge_support_active_ratio=float(edge_support.mean()),
        main_support_active_ratio=float(support_active.mean()),
        main_support_density_in_structure=density_in_structure,
        main_support_mean_weight=mean_weight,
        boundary_distance_weight_share=boundary_distance_weight_share,
        main_support_component_count=float(component_info["main_support_component_count"]),
        main_support_largest_component_share=float(component_info["main_support_largest_component_share"]),
        main_support_top2_component_share=float(component_info["main_support_top2_component_share"]),
        main_support_centroid_distance_mean=float(component_info["main_support_centroid_distance_mean"]),
        main_support_component_bias_weight_share=component_bias_weight_share,
        main_support_depth_mode_count=float(depth_mode_info["main_support_depth_mode_count"]),
        main_support_back_mode_share=float(depth_mode_info["main_support_back_mode_share"]),
        main_support_front_back_gap=float(depth_mode_info["main_support_front_back_gap"]),
        main_support_depth_hist_peak_ratio=float(depth_mode_info["main_support_depth_hist_peak_ratio"]),
        main_support_secondary_risk=float(depth_mode_info["main_support_secondary_risk"]),
        main_support_front_depth_bias_weight_share=front_depth_bias_weight_share,
        target_edge_structure_share=masked_share(structure_for_support, edge_heatmap),
        target_edge_boundary_probe_share=masked_share(boundary_probe_for_support, edge_heatmap),
        target_edge_outside_ring_share=masked_share(ring_for_support, edge_heatmap),
        target_edge_support_share=masked_share(edge_support, edge_heatmap),
        main_support_boundary_probe_overlap=masked_share(boundary_probe_for_support, main_support),
        main_support_outside_ring_overlap=masked_share(ring_for_support, main_support),
        edge_active=bool(edge_active),
    )


def resize_u8(arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR)


def mask_to_u8(mask01: np.ndarray) -> np.ndarray:
    x = np.clip(mask01, 0.0, 1.0)
    return np.repeat((x[..., None] * 255.0).astype(np.uint8), 3, axis=2)


def heat_to_u8(heat01: np.ndarray) -> np.ndarray:
    x = np.clip(heat01, 0.0, 1.0).astype(np.float32)
    out = np.zeros((*x.shape, 3), dtype=np.uint8)
    out[..., 0] = np.clip(255.0 * np.maximum(0.0, x - 0.25) / 0.75, 0.0, 255.0).astype(np.uint8)
    out[..., 1] = np.clip(255.0 * x, 0.0, 255.0).astype(np.uint8)
    out[..., 2] = np.clip(255.0 * (1.0 - x), 0.0, 255.0).astype(np.uint8)
    return out


def build_markdown(
    seq_name: str,
    geom_subdir: str,
    camera: str,
    contract: dict[str, Any],
    stages: list[StageSpec],
    stage_diags: dict[str, list[FrameDiag]],
    boundary_probe_px: int,
) -> str:
    lines: list[str] = []
    lines.append("# H-Family Local Structure Diagnostics")
    lines.append("")
    lines.append(f"- seq: `{seq_name}`")
    lines.append(f"- geom_subdir: `{geom_subdir}`")
    lines.append(f"- camera: `{camera}`")
    lines.append("- fixed triplet: `step000000`, `step000001`, `step000002`")
    lines.append("- note: this is a local-only geometry diagnostic; no cloud run is involved.")
    lines.append(f"- fixed diagnostic boundary probe: `fg - erode(fg,{boundary_probe_px})`")
    lines.append(
        "- precision: "
        f"`tf32={contract.get('tf32', '')}` "
        f"`amp={contract.get('amp', '')}` "
        f"`strict_deterministic={contract.get('strict_deterministic', '')}`"
    )
    lines.append("")
    lines.append("## Stage Summary")
    lines.append("")
    lines.append("| Stage | lambda_edge | warmup | edge_weight_mode | falloff_px | component_bias_mode | front_bias_mode | main_support_cover_mean | main_support_mean_weight | support_density_in_structure | component_count_mean | largest_component_share_mean | back_mode_share_mean | secondary_risk_mean | step000002 edge_share | step000002 component_count | step000002 back_mode_share | step000002 secondary_risk | step000002 boundary_overlap | step000002 outside_ring_overlap | step000002 component_bias_weight_share | step000002 front_bias_weight_share | edge_active_views |")
    lines.append("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stage in stages:
        diags = stage_diags[stage.label]
        support_mean = float(np.mean([diag.main_support_active_ratio for diag in diags]))
        support_weight_mean = float(np.mean([diag.main_support_mean_weight for diag in diags]))
        support_density_mean = float(np.mean([diag.main_support_density_in_structure for diag in diags]))
        component_count_mean = float(np.mean([diag.main_support_component_count for diag in diags]))
        largest_share_mean = float(np.mean([diag.main_support_largest_component_share for diag in diags]))
        back_mode_share_mean = float(np.mean([diag.main_support_back_mode_share for diag in diags]))
        secondary_risk_mean = float(np.mean([diag.main_support_secondary_risk for diag in diags]))
        edge_active = int(sum(1 for diag in diags if diag.edge_active))
        step_focus = diags[min(2, len(diags) - 1)]
        lines.append(
            f"| `{stage.label}` | {stage.lambda_edge:.3f} | {stage.warmup_steps:d} | `{stage.edge_weight_mode}` | {stage.boundary_falloff_px:d} | "
            f"`{stage.component_bias_mode}` | `{stage.front_depth_bias_mode}` | {support_mean:.4f} | {support_weight_mean:.4f} | {support_density_mean:.4f} | "
            f"{component_count_mean:.3f} | {largest_share_mean:.4f} | {back_mode_share_mean:.4f} | {secondary_risk_mean:.4f} | "
            f"{masked_share(step_focus.main_support_mask, step_focus.edge_heatmap):.4f} | {step_focus.main_support_component_count:.3f} | "
            f"{step_focus.main_support_back_mode_share:.4f} | {step_focus.main_support_secondary_risk:.4f} | {step_focus.main_support_boundary_probe_overlap:.4f} | "
            f"{step_focus.main_support_outside_ring_overlap:.4f} | {step_focus.main_support_component_bias_weight_share:.4f} | "
            f"{step_focus.main_support_front_depth_bias_weight_share:.4f} | {edge_active:d} |"
        )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `fg mask`: current foreground mask used by the live lane.")
    lines.append("- `fg bbox`: tight foreground bbox with fixed margin/min-side for H-family structure loss.")
    lines.append("- `structure region`: actual H1 region after bbox and optional foreground-interior intersection.")
    lines.append("- `main support`: the actual weight map used by the main structure loss. For `H1s*` it is a soft target-edge-weighted interior map with optional boundary-distance suppression, optional largest-component bias, and optional front-depth bias.")
    lines.append("- `main support active`: binary view of the effective support after all weighting.")
    lines.append("- `largest component`: dominant connected component extracted from the effective main support.")
    lines.append("- `boundary probe`: fixed-width `fg - erode(fg, boundary_probe_px)` used only for diagnostics.")
    lines.append("- `outside ring`: `dilate(fg) - fg`, kept only as a background-risk probe.")
    lines.append("- `target depth edge`: Sobel magnitude after target-only median/MAD normalization inside the fixed bbox support.")
    lines.append("- `component_count / largest_component_share / top2_share`: fragmentation diagnostics on the final main support, after all weighting and biasing.")
    lines.append("- `back_mode_share / secondary_risk`: depth-bimodality diagnostics inside the final main support; larger means the support is still rewarding a deeper secondary layer.")
    lines.append("- `step000002` is the main review focus for multi-peak rebound, internal fragmentation, and front/back depth ambiguity.")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_canvas(
    seq_name: str,
    geom_subdir: str,
    camera: str,
    contract: dict[str, Any],
    stages: list[StageSpec],
    stage_diags: dict[str, list[FrameDiag]],
    boundary_probe_px: int,
    out_png: str,
) -> None:
    font = ImageFont.load_default()
    frame_count = len(next(iter(stage_diags.values())))
    stage_count = len(stages)
    panel_count = 10
    cell_w = PANEL_W * panel_count + PANEL_GAP * (panel_count - 1)
    width = MARGIN * 2 + frame_count * cell_w + max(0, frame_count - 1) * FRAME_GAP
    height = MARGIN * 2 + 48 + stage_count * (TEXT_H + PANEL_H + ROW_GAP)
    canvas = Image.new("RGB", (width, height), (247, 245, 241))
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, MARGIN), f"H-family local structure diagnostics | {seq_name} | {geom_subdir} | {camera}", fill=(0, 0, 0), font=font)
    draw.text(
        (MARGIN, MARGIN + 16),
        (
            "Review focus: step000002 peak_count risk, torso/arm solidity, boundary-probe spill, outside-ring suppression. "
            f"precision tf32={contract.get('tf32', '')} amp={contract.get('amp', '')} "
            f"strict_det={contract.get('strict_deterministic', '')} boundary_probe_px={boundary_probe_px}"
        ),
        fill=(0, 0, 0),
        font=font,
    )
    y = MARGIN + 48
    for stage in stages:
        diags = stage_diags[stage.label]
        draw.text(
            (MARGIN, y),
            (
                f"{stage.label} | lambda_edge={stage.lambda_edge:.3f} lambda_ring={stage.lambda_ring:.3f} "
                f"bbox_margin={stage.bbox_margin_px} bbox_min_side={stage.bbox_min_side_px} "
                f"region={stage.region_mode} erode={stage.region_erode_px} warmup={stage.warmup_steps} ring_px={stage.ring_px} "
                f"edge_support={stage.edge_support_mode} edge_weight={stage.edge_weight_mode} falloff={stage.boundary_falloff_px} "
                f"component_bias={stage.component_bias_mode} thr={stage.component_bias_threshold_ratio:.2f} other={stage.component_bias_other_scale:.2f} "
                f"front_bias={stage.front_depth_bias_mode} tau={stage.front_depth_bias_tau:.2f} center_q={stage.front_depth_bias_center_quantile:.2f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
        y_panels = y + 16
        x = MARGIN
        for frame_idx, diag in enumerate(diags):
            panels = [
                ("rgb", diag.rgb_u8),
                ("fg mask", mask_to_u8(diag.fg_mask)),
                ("fg bbox", mask_to_u8(diag.bbox_mask)),
                ("structure region", mask_to_u8(diag.structure_mask)),
                ("main support", heat_to_u8(diag.main_support_mask)),
                ("support active", mask_to_u8(diag.main_support_active_mask)),
                ("largest component", mask_to_u8(diag.main_support_largest_component_mask)),
                ("boundary probe", mask_to_u8(diag.boundary_probe_mask)),
                ("outside ring", mask_to_u8(diag.ring_mask)),
                ("target depth edge", heat_to_u8(diag.edge_heatmap)),
            ]
            for panel_idx, (title, arr) in enumerate(panels):
                px = x + panel_idx * (PANEL_W + PANEL_GAP)
                panel = resize_u8(arr, (PANEL_W, PANEL_H))
                canvas.paste(panel, (px, y_panels))
                outline = (160, 0, 0) if frame_idx == 2 else (90, 90, 90)
                draw.rectangle((px - 1, y_panels - 1, px + PANEL_W + 1, y_panels + PANEL_H + 1), outline=outline, width=2)
                draw.text((px, y_panels + PANEL_H + 2), f"{diag.step_label} | {title}", fill=(0, 0, 0), font=font)
            x += cell_w + FRAME_GAP
        y += TEXT_H + PANEL_H + ROW_GAP
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:/vggt")
    ap.add_argument("--contract-json", default="logs/modal_phase5/probe_contract_latest.json")
    ap.add_argument("--candidate-json", default="")
    ap.add_argument("--zju-root", default="F:/datasets/ZJU_MoCap/data/zju_mocap")
    ap.add_argument("--seq-name", default="")
    ap.add_argument("--camera", default="Camera_B1")
    ap.add_argument("--frame-indices", default="0,1,2")
    ap.add_argument("--boundary-probe-px", type=int, default=2)
    ap.add_argument("--stage", action="append", default=[])
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    contract_json = args.contract_json if os.path.isabs(args.contract_json) else os.path.join(repo_dir, args.contract_json)
    contract = read_json(contract_json)
    candidate = None
    if args.candidate_json:
        candidate_json = args.candidate_json if os.path.isabs(args.candidate_json) else os.path.join(repo_dir, args.candidate_json)
        candidate = read_json(candidate_json)
    seq_name = resolve_seq_name(contract=contract, override=args.seq_name)
    geom_subdir = pick_existing_geom_subdir(
        zju_root=os.path.abspath(args.zju_root),
        seq_name=seq_name,
        contract=contract,
        candidate=candidate,
    )
    frame_indices = [int(x.strip()) for x in str(args.frame_indices).split(",") if x.strip()]
    stages = [parse_stage(raw) for raw in (args.stage or [])]
    if not stages:
        stages = [
            StageSpec("H0", 0.0, 0.0, 12, 24, 3, "bbox", 0, 0, "off", 0.0, 32, "uniform", 0, "off", 0.25, 1.0),
            StageSpec("H1s1_core", 0.003, 0.0, 12, 24, 3, "bbox_fg_interior", 3, 80, "off", 0.0, 32, "target_edge_sqrt", 2, "largest_soft", 0.25, 0.35),
            StageSpec("H1s2_core", 0.003, 0.0, 12, 24, 3, "bbox_fg_interior", 3, 80, "off", 0.0, 32, "target_edge_sqrt", 3, "largest_soft", 0.25, 0.35),
            StageSpec("H1sf1", 0.003, 0.0, 12, 24, 3, "bbox_fg_interior", 3, 80, "off", 0.0, 32, "target_edge_sqrt", 2, "largest_soft", 0.25, 0.35, "front_soft", 0.75, 0.55),
            StageSpec("H1sf2", 0.003, 0.0, 12, 24, 3, "bbox_fg_interior", 3, 80, "off", 0.0, 32, "target_edge_sqrt", 3, "largest_soft", 0.25, 0.35, "front_soft", 0.75, 0.55),
        ]

    geom_subdir, npz_paths = resolve_npz_paths(
        zju_root=os.path.abspath(args.zju_root),
        seq_name=seq_name,
        geom_subdir=geom_subdir,
        frame_indices=frame_indices,
    )
    stage_diags: dict[str, list[FrameDiag]] = {}
    for stage in stages:
        stage_diags[stage.label] = [
            load_frame_diag(
                npz_path=npz_path,
                step_idx=step_idx,
                camera=args.camera,
                zju_root=os.path.abspath(args.zju_root),
                seq_name=seq_name,
                stage=stage,
                boundary_probe_px=int(args.boundary_probe_px),
            )
            for step_idx, npz_path in enumerate(npz_paths)
        ]

    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(build_markdown(seq_name=seq_name, geom_subdir=geom_subdir, camera=args.camera, contract=contract, stages=stages, stage_diags=stage_diags, boundary_probe_px=int(args.boundary_probe_px)))
    render_canvas(seq_name=seq_name, geom_subdir=geom_subdir, camera=args.camera, contract=contract, stages=stages, stage_diags=stage_diags, boundary_probe_px=int(args.boundary_probe_px), out_png=out_png)
    print(f"[render-fg-structure-diagnostics] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
