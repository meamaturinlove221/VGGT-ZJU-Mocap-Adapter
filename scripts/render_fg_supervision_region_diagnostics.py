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

from finetune_vggt_pseudo import (
    _build_fg_supervision_boost_mask,
    _infer_mask_path_from_image,
    _normalize_mask_binary_np,
)


PANEL_W = 176
PANEL_H = 132
PANEL_GAP = 8
FRAME_GAP = 20
ROW_GAP = 24
TEXT_H = 34
MARGIN = 18


@dataclass
class StageSpec:
    label: str
    fg_boost: float
    bg_floor: float
    region_mode: str
    region_erode_px: int


@dataclass
class FrameDiag:
    frame_name: str
    rgb_u8: np.ndarray
    fg_mask: np.ndarray
    boost_mask: np.ndarray
    ring_mask: np.ndarray


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_stage(raw: str) -> StageSpec:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 5:
        raise ValueError(f"invalid --stage spec: {raw}")
    return StageSpec(
        label=parts[0],
        fg_boost=float(parts[1]),
        bg_floor=float(parts[2]),
        region_mode=str(parts[3] or "all"),
        region_erode_px=max(0, int(parts[4])),
    )


def resolve_seq_name(args: argparse.Namespace) -> str:
    if args.seq_name:
        return args.seq_name
    if args.contract_json:
        data = read_json(args.contract_json)
        raw = str(data.get("seq_names") or "").strip()
        if raw:
            return raw.replace(",", " ").split()[0]
    return "CoreView_390"


def resolve_frame_names(camera_dir: str, frame_indices: list[int]) -> list[str]:
    files = sorted(
        [
            name
            for name in os.listdir(camera_dir)
            if os.path.isfile(os.path.join(camera_dir, name))
            and os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    out: list[str] = []
    for idx in frame_indices:
        if idx < 0 or idx >= len(files):
            raise IndexError(f"frame index {idx} out of range for {camera_dir}")
        out.append(files[idx])
    return out


def load_frame_diag(
    zju_root: str,
    seq_name: str,
    camera: str,
    frame_name: str,
    stage: StageSpec,
) -> FrameDiag:
    img_path = os.path.join(zju_root, seq_name, camera, frame_name)
    mask_path, _ = _infer_mask_path_from_image(img_path, preferred="mask")
    if mask_path is None or (not os.path.exists(mask_path)):
        raise FileNotFoundError(f"mask not found for image: {img_path}")
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    fg = _normalize_mask_binary_np(np.asarray(Image.open(mask_path)))
    fg_tensor = torch.from_numpy(fg).unsqueeze(0).unsqueeze(0).float()
    boost_tensor, _ = _build_fg_supervision_boost_mask(
        fg_mask01=fg_tensor,
        region_mode=stage.region_mode,
        region_erode_px=stage.region_erode_px,
    )
    if boost_tensor is None:
        boost = np.zeros_like(fg, dtype=np.float32)
    else:
        boost = boost_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    ring = np.clip(fg - boost, 0.0, 1.0).astype(np.float32)
    return FrameDiag(
        frame_name=frame_name,
        rgb_u8=rgb,
        fg_mask=fg.astype(np.float32),
        boost_mask=boost,
        ring_mask=ring,
    )


def resize_u8(arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR)


def mask_to_u8(mask01: np.ndarray) -> np.ndarray:
    x = np.clip(mask01, 0.0, 1.0)
    return np.repeat((x[..., None] * 255.0).astype(np.uint8), 3, axis=2)


def overlay_rgb(rgb_u8: np.ndarray, boost01: np.ndarray, ring01: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    out = rgb.copy()
    green = np.zeros_like(out)
    green[..., 1] = 255.0
    red = np.zeros_like(out)
    red[..., 0] = 255.0
    out = out * (1.0 - 0.35 * boost01[..., None]) + green * (0.35 * boost01[..., None])
    out = out * (1.0 - 0.75 * ring01[..., None]) + red * (0.75 * ring01[..., None])
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(18, 18, 18), font=font)


def build_markdown(
    seq_name: str,
    camera: str,
    frame_names: list[str],
    stages: list[StageSpec],
    stage_diags: dict[str, list[FrameDiag]],
) -> str:
    lines: list[str] = []
    lines.append("# Foreground Supervision Region Diagnostics")
    lines.append("")
    lines.append(
        f"This is a local spatial diagnostic on raw `{seq_name}` / `{camera}` frames "
        f"`{', '.join(frame_names)}`. It visualizes where supervision boost would land. "
        "It is not a cloud metric evaluation."
    )
    lines.append("")
    lines.append("## Stage Summary")
    lines.append("")
    lines.append("| Stage | fg_boost | bg_floor | region_mode | region_erode_px | boost_cover_mean | boost_ratio_in_fg_mean | boundary_ring_cover_mean | boundary_ring_ratio_in_fg_mean |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for stage in stages:
        diags = stage_diags[stage.label]
        fg_cov = np.mean([float(d.fg_mask.mean()) for d in diags])
        boost_cov = np.mean([float(d.boost_mask.mean()) for d in diags])
        ring_cov = np.mean([float(d.ring_mask.mean()) for d in diags])
        boost_ratio = boost_cov / max(fg_cov, 1e-6)
        ring_ratio = ring_cov / max(fg_cov, 1e-6)
        lines.append(
            f"| `{stage.label}` | {stage.fg_boost:.2f} | {stage.bg_floor:.2f} | "
            f"`{stage.region_mode}` | {stage.region_erode_px:d} | {boost_cov:.4f} | "
            f"{boost_ratio:.4f} | {ring_cov:.4f} | {ring_ratio:.4f} |"
        )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `overlay`: green shows the boosted interior, red shows the suppressed boundary ring.")
    lines.append("- `fg mask`: full foreground mask used by the current lane.")
    lines.append("- `boost mask`: where extra foreground supervision would actually apply.")
    lines.append("- `boundary ring`: foreground pixels intentionally left at baseline weight.")
    lines.append("")
    return "\n".join(lines)


def render_canvas(
    seq_name: str,
    camera: str,
    stages: list[StageSpec],
    stage_diags: dict[str, list[FrameDiag]],
    out_png: str,
) -> None:
    font = ImageFont.load_default()
    frame_count = len(next(iter(stage_diags.values())))
    stage_count = len(stages)
    cell_w = PANEL_W * 4 + PANEL_GAP * 3
    width = (
        MARGIN * 2
        + frame_count * cell_w
        + max(0, frame_count - 1) * FRAME_GAP
    )
    height = (
        MARGIN * 2
        + 32
        + stage_count * (TEXT_H + PANEL_H + ROW_GAP)
    )
    canvas = Image.new("RGB", (width, height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (MARGIN, MARGIN), f"Foreground supervision region diagnostics | {seq_name} | {camera}", font)
    y = MARGIN + 32
    for stage in stages:
        diags = stage_diags[stage.label]
        draw_text(
            draw,
            (MARGIN, y),
            (
                f"{stage.label} | boost={stage.fg_boost:.2f} bg_floor={stage.bg_floor:.2f} "
                f"region={stage.region_mode} erode_px={stage.region_erode_px}"
            ),
            font,
        )
        y_panels = y + 16
        x = MARGIN
        for diag in diags:
            overlay = overlay_rgb(diag.rgb_u8, diag.boost_mask, diag.ring_mask)
            panels = [
                ("overlay", overlay),
                ("fg mask", mask_to_u8(diag.fg_mask)),
                ("boost mask", mask_to_u8(diag.boost_mask)),
                ("boundary ring", mask_to_u8(diag.ring_mask)),
            ]
            for panel_idx, (title, arr) in enumerate(panels):
                px = x + panel_idx * (PANEL_W + PANEL_GAP)
                panel = resize_u8(arr, (PANEL_W, PANEL_H))
                canvas.paste(panel, (px, y_panels))
                draw_text(draw, (px, y_panels + PANEL_H + 2), f"{diag.frame_name} | {title}", font)
            x += cell_w + FRAME_GAP
        y += TEXT_H + PANEL_H + ROW_GAP
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:/vggt")
    ap.add_argument("--contract-json", default="logs/modal_phase5/probe_contract_latest.json")
    ap.add_argument("--zju-root", default="F:/datasets/ZJU_MoCap/data/zju_mocap")
    ap.add_argument("--seq-name", default="")
    ap.add_argument("--camera", default="Camera_B1")
    ap.add_argument("--frame-indices", default="0,1,2")
    ap.add_argument("--stage", action="append", default=[])
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    contract_json = args.contract_json
    if contract_json and (not os.path.isabs(contract_json)):
        contract_json = os.path.join(repo_dir, contract_json)
    seq_name = resolve_seq_name(args)
    frame_indices = [int(x.strip()) for x in str(args.frame_indices).split(",") if x.strip()]
    if not args.stage:
        args.stage = [
            "R0|1.0|0.0|all|0",
            "F1-ref|1.5|0.05|all|0",
            "R2|1.3|0.05|interior_only|5",
            "R1|1.5|0.05|interior_only|3",
        ]
    stages = [parse_stage(raw) for raw in args.stage]

    camera_dir = os.path.join(args.zju_root, seq_name, args.camera)
    frame_names = resolve_frame_names(camera_dir, frame_indices)
    stage_diags: dict[str, list[FrameDiag]] = {}
    for stage in stages:
        stage_diags[stage.label] = [
            load_frame_diag(
                zju_root=args.zju_root,
                seq_name=seq_name,
                camera=args.camera,
                frame_name=frame_name,
                stage=stage,
            )
            for frame_name in frame_names
        ]

    out_md = os.path.abspath(args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md))
    out_png = os.path.abspath(args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png))
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(build_markdown(seq_name, args.camera, frame_names, stages, stage_diags))
    render_canvas(seq_name, args.camera, stages, stage_diags, out_png)
    print(f"[render-fg-supervision-region] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
