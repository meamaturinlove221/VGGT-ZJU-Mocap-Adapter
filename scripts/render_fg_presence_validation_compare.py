import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw, ImageFont


F0_GHOST_TOL = 0.08
F0_LUMA_TOL = 0.0015
F0_CONTRAST_TOL = 0.0020
F0_TGT_L1_TOL = 0.0015
PROMOTION_GHOST_MAX = 4.98


@dataclass
class Record:
    label: str
    json_path: str
    run_timestamp: str
    ghost_visual: float
    ghost_mean: float
    fg_luma: float
    fg_contrast: float
    fg_tgt_l1: float
    width_ratio_mean: float
    area_ratio_mean: float
    best_visual_png: str
    triplet_pngs: list[str]
    triplet_metrics: list[dict[str, float]]
    fg_supervision_boost: float
    fg_supervision_bg_floor: float
    fg_supervision_region_mode: str
    fg_supervision_region_erode_px: float
    lambda_fg_conf_presence: float
    fg_conf_presence_target_ratio: float
    fg_conf_presence_enabled: float
    fg_conf_presence_pred_mean: float
    fg_conf_presence_tgt_mean: float
    fg_conf_presence_target_floor: float
    fg_conf_presence_active_ratio: float
    fg_conf_presence_loss: float
    mean_loss_fg_conf_presence: float
    precompute_log: str
    precompute_teacher_forward_mean: float
    precompute_batch_total_mean: float
    precompute_mv_support_mean: float
    precompute_gate_prepare_mean: float


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_text_any(path: str) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def resolve_triplet(repo_dir: str, data: dict[str, Any]) -> tuple[str, list[str]]:
    best_visual_rel = str(data.get("best_visual_png") or "")
    best_visual_abs = ""
    if best_visual_rel:
        best_visual_abs = best_visual_rel if os.path.isabs(best_visual_rel) else os.path.join(repo_dir, best_visual_rel)
    if not best_visual_abs or not os.path.exists(best_visual_abs):
        run_ts = str(data.get("run_timestamp") or "")
        run_tag = str(data.get("run_tag") or "mv_0.001_mvmask_0_default")
        infer_dir = os.path.join(repo_dir, "logs", "modal_phase5", f"_ghost_eval_{run_tag}_{run_ts}")
        cand = os.path.join(infer_dir, "infer_val_e005_cat_fg_mask_pred_tgt_step000001.png")
        if os.path.exists(cand):
            best_visual_abs = cand
    triplet: list[str] = []
    if best_visual_abs and os.path.exists(best_visual_abs):
        directory = os.path.dirname(best_visual_abs)
        for idx in range(3):
            path = os.path.join(directory, f"infer_val_e005_cat_fg_mask_pred_tgt_step{idx:06d}.png")
            if os.path.exists(path):
                triplet.append(path)
    return best_visual_abs, triplet


def resolve_ghost_rows_csv(repo_dir: str, data: dict[str, Any]) -> str:
    ghost_rows_rel = str(data.get("ghost_rows_csv") or "")
    if ghost_rows_rel:
        ghost_rows_abs = ghost_rows_rel if os.path.isabs(ghost_rows_rel) else os.path.join(repo_dir, ghost_rows_rel)
        if os.path.exists(ghost_rows_abs):
            return ghost_rows_abs
    run_ts = str(data.get("run_timestamp") or "")
    run_tag = str(data.get("run_tag") or "")
    if not run_ts or not run_tag:
        return ""
    cand = os.path.join(repo_dir, "logs", "modal_phase5", f"ghost_score_rows_{run_tag}_{run_ts}.csv")
    return cand if os.path.exists(cand) else ""


def load_triplet_metrics(ghost_rows_csv: str) -> list[dict[str, float]]:
    if not ghost_rows_csv or not os.path.exists(ghost_rows_csv):
        return []
    out: list[dict[str, float]] = []
    with open(ghost_rows_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(str(row.get("step") or "").strip()))
            except Exception:
                continue
            if step not in (0, 1, 2):
                continue
            out.append(
                {
                    "step": float(step),
                    "ghost_visual_score": to_float(row.get("ghost_visual_score")),
                    "peak_count": to_float(row.get("peak_count")),
                    "width_ratio": to_float(row.get("width_ratio")),
                    "area_ratio": to_float(row.get("area_ratio")),
                    "fg_pred_luma_mean": to_float(row.get("fg_pred_luma_mean")),
                    "fg_pred_contrast": to_float(row.get("fg_pred_contrast")),
                    "fg_pred_tgt_l1": to_float(row.get("fg_pred_tgt_l1")),
                }
            )
    out.sort(key=lambda item: item["step"])
    return out


def load_record(repo_dir: str, label: str, json_path: str) -> Record:
    path = json_path if os.path.isabs(json_path) else os.path.join(repo_dir, json_path)
    data = read_json(path)
    best_visual_abs, triplet = resolve_triplet(repo_dir, data)
    ghost_rows_csv = resolve_ghost_rows_csv(repo_dir, data)
    timing = summarize_precompute_log(repo_dir, str(data.get("run_timestamp") or ""))
    return Record(
        label=label,
        json_path=path,
        run_timestamp=str(data.get("run_timestamp") or ""),
        ghost_visual=to_float(data.get("ghost_visual_score")),
        ghost_mean=to_float(data.get("ghost_score_mean")),
        fg_luma=to_float(data.get("fg_pred_luma_mean")),
        fg_contrast=to_float(data.get("fg_pred_contrast")),
        fg_tgt_l1=to_float(data.get("fg_pred_tgt_l1")),
        width_ratio_mean=to_float(data.get("width_ratio_mean")),
        area_ratio_mean=to_float(data.get("area_ratio_mean")),
        best_visual_png=best_visual_abs,
        triplet_pngs=triplet,
        triplet_metrics=load_triplet_metrics(ghost_rows_csv),
        fg_supervision_boost=to_float(data.get("fg_supervision_boost"), default=1.0),
        fg_supervision_bg_floor=to_float(data.get("fg_supervision_bg_floor"), default=0.0),
        fg_supervision_region_mode=str(data.get("fg_supervision_region_mode") or "all"),
        fg_supervision_region_erode_px=to_float(data.get("fg_supervision_region_erode_px"), default=0.0),
        lambda_fg_conf_presence=to_float(data.get("lambda_fg_conf_presence"), default=0.0),
        fg_conf_presence_target_ratio=to_float(data.get("fg_conf_presence_target_ratio"), default=0.9),
        fg_conf_presence_enabled=to_float(data.get("fg_conf_presence_enabled"), default=0.0),
        fg_conf_presence_pred_mean=to_float(data.get("fg_conf_presence_pred_mean"), default=0.0),
        fg_conf_presence_tgt_mean=to_float(data.get("fg_conf_presence_tgt_mean"), default=0.0),
        fg_conf_presence_target_floor=to_float(data.get("fg_conf_presence_target_floor"), default=0.0),
        fg_conf_presence_active_ratio=to_float(data.get("fg_conf_presence_active_ratio"), default=0.0),
        fg_conf_presence_loss=to_float(data.get("fg_conf_presence_loss"), default=0.0),
        mean_loss_fg_conf_presence=to_float(data.get("mean_loss_fg_conf_presence"), default=0.0),
        precompute_log=timing["log_path"],
        precompute_teacher_forward_mean=timing["teacher_forward_mean"],
        precompute_batch_total_mean=timing["batch_total_mean"],
        precompute_mv_support_mean=timing["mv_support_mean"],
        precompute_gate_prepare_mean=timing["gate_prepare_mean"],
    )


def find_precompute_log(repo_dir: str, run_timestamp: str) -> str:
    if not run_timestamp:
        return ""
    pattern = os.path.join(
        repo_dir,
        "logs",
        "modal_phase5",
        f"vggt_ft_lr_*_{run_timestamp}.precompute.log",
    )
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    prefix = run_timestamp[:8]
    near_pattern = os.path.join(
        repo_dir,
        "logs",
        "modal_phase5",
        f"vggt_ft_lr_*_{prefix}_*.precompute.log",
    )
    near_matches = sorted(glob.glob(near_pattern))
    if not near_matches:
        return ""
    try:
        target_dt = datetime.strptime(run_timestamp, "%Y%m%d_%H%M%S")
    except Exception:
        return near_matches[-1]
    best_path = ""
    best_delta = None
    ts_regex = re.compile(r"_(\d{8}_\d{6})\.precompute\.log$")
    for path in near_matches:
        name = os.path.basename(path)
        match = ts_regex.search(name)
        if not match:
            continue
        try:
            cand_dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except Exception:
            continue
        delta = abs((cand_dt - target_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = path
    return best_path if best_path else near_matches[-1]


def _collect_number(lines: list[str], marker: str) -> list[float]:
    pattern = rf"{re.escape(marker)}(?:=|\"?\s*:\s*)([0-9]+(?:\.[0-9]+)?)"
    return _collect_regex(lines, pattern)


def _collect_regex(lines: list[str], pattern: str) -> list[float]:
    values: list[float] = []
    regex = re.compile(pattern)
    for line in lines:
        match = regex.search(line)
        if not match:
            continue
        try:
            values.append(float(match.group(1)))
        except Exception:
            continue
    return values


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def summarize_precompute_log(repo_dir: str, run_timestamp: str) -> dict[str, Any]:
    log_path = find_precompute_log(repo_dir, run_timestamp)
    if not log_path or not os.path.exists(log_path):
        return {
            "log_path": "",
            "teacher_forward_mean": float("nan"),
            "batch_total_mean": float("nan"),
            "mv_support_mean": float("nan"),
            "gate_prepare_mean": float("nan"),
        }
    lines = read_text_any(log_path).splitlines()
    return {
        "log_path": log_path,
        "teacher_forward_mean": _mean_or_nan(_collect_number(lines, "teacher_forward_sec")),
        "batch_total_mean": _mean_or_nan(_collect_number(lines, "batch_total_sec")),
        "mv_support_mean": _mean_or_nan(_collect_regex(lines, r"mv_support_done sec=([0-9]+(?:\.[0-9]+)?)")),
        "gate_prepare_mean": _mean_or_nan(_collect_number(lines, "gate_prepare_sec")),
    }


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def delta_fmt(cur: float, base: float, digits: int = 4) -> str:
    if math.isnan(cur) or math.isnan(base):
        return "nan"
    return f"{cur - base:+.{digits}f}"


def f0_metric_gate(rec: Record, baseline: Record) -> tuple[bool, list[str]]:
    checks = [
        ("ghost_visual", abs(rec.ghost_visual - baseline.ghost_visual) <= F0_GHOST_TOL),
        ("fg_luma", abs(rec.fg_luma - baseline.fg_luma) <= F0_LUMA_TOL),
        ("fg_contrast", abs(rec.fg_contrast - baseline.fg_contrast) <= F0_CONTRAST_TOL),
        ("fg_tgt_l1", abs(rec.fg_tgt_l1 - baseline.fg_tgt_l1) <= F0_TGT_L1_TOL),
    ]
    return all(ok for _, ok in checks), [f"{name}={'ok' if ok else 'drift'}" for name, ok in checks]


def promotion_metric_gate(rec: Record, baseline: Record) -> tuple[bool, int]:
    improvements = 0
    if rec.fg_luma > baseline.fg_luma:
        improvements += 1
    if rec.fg_contrast > baseline.fg_contrast:
        improvements += 1
    if rec.fg_tgt_l1 < baseline.fg_tgt_l1:
        improvements += 1
    return rec.ghost_visual <= PROMOTION_GHOST_MAX and improvements >= 2, improvements


def triplet_metric_line(item: dict[str, float]) -> str:
    step = int(item.get("step", float("nan")))
    return (
        f"step{step:06d}: ghost_visual={fmt(item.get('ghost_visual_score', float('nan')))} "
        f"peak_count={fmt(item.get('peak_count', float('nan')), 1)} "
        f"width_ratio={fmt(item.get('width_ratio', float('nan')))} "
        f"area_ratio={fmt(item.get('area_ratio', float('nan')))} "
        f"fg_luma={fmt(item.get('fg_pred_luma_mean', float('nan')))} "
        f"fg_contrast={fmt(item.get('fg_pred_contrast', float('nan')))} "
        f"fg_tgt_l1={fmt(item.get('fg_pred_tgt_l1', float('nan')))}"
    )


def parse_compare_specs(raw_specs: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"invalid compare spec: {raw}")
        label, path = raw.split("=", 1)
        specs.append((label.strip(), path.strip()))
    return specs


def build_markdown(baseline: Record, compares: list[Record]) -> str:
    lines: list[str] = []
    lines.append("# Foreground Presence Validation Compare")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Locked baseline: `{baseline.label}`")
    lines.append(f"- candidate json: `{baseline.json_path}`")
    lines.append(
        f"- metrics: `ghost_visual={fmt(baseline.ghost_visual)}`, `ghost_mean={fmt(baseline.ghost_mean)}`, "
        f"`fg_luma={fmt(baseline.fg_luma)}`, `fg_contrast={fmt(baseline.fg_contrast)}`, `fg_tgt_l1={fmt(baseline.fg_tgt_l1)}`, "
        f"`width_ratio={fmt(baseline.width_ratio_mean)}`, `area_ratio={fmt(baseline.area_ratio_mean)}`"
    )
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Run | ghost_visual | delta_ghost_visual | fg_luma | delta_fg_luma | fg_contrast | delta_fg_contrast | fg_tgt_l1 | delta_fg_tgt_l1 | width_ratio | area_ratio | fg_boost | bg_floor | region_mode | region_erode_px | lambda_presence | target_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for rec in compares:
        lines.append(
            f"| `{rec.label}` | {fmt(rec.ghost_visual)} | {delta_fmt(rec.ghost_visual, baseline.ghost_visual)} | "
            f"{fmt(rec.fg_luma)} | {delta_fmt(rec.fg_luma, baseline.fg_luma)} | "
            f"{fmt(rec.fg_contrast)} | {delta_fmt(rec.fg_contrast, baseline.fg_contrast)} | "
            f"{fmt(rec.fg_tgt_l1)} | {delta_fmt(rec.fg_tgt_l1, baseline.fg_tgt_l1)} | "
            f"{fmt(rec.width_ratio_mean)} | {fmt(rec.area_ratio_mean)} | "
            f"{fmt(rec.fg_supervision_boost, 2)} | {fmt(rec.fg_supervision_bg_floor, 2)} | "
            f"`{rec.fg_supervision_region_mode}` | {fmt(rec.fg_supervision_region_erode_px, 1)} | {fmt(rec.lambda_fg_conf_presence, 3)} | "
            f"{fmt(rec.fg_conf_presence_target_ratio, 2)} |"
        )
    lines.append("")
    lines.append("## Precompute Timing")
    lines.append("")
    lines.append("| Run | teacher_forward_mean_sec | batch_total_mean_sec | mv_support_mean_sec | gate_prepare_mean_sec |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `{baseline.label}` | {fmt(baseline.precompute_teacher_forward_mean, 2)} | "
        f"{fmt(baseline.precompute_batch_total_mean, 2)} | {fmt(baseline.precompute_mv_support_mean, 2)} | "
        f"{fmt(baseline.precompute_gate_prepare_mean, 2)} |"
    )
    for rec in compares:
        lines.append(
            f"| `{rec.label}` | {fmt(rec.precompute_teacher_forward_mean, 2)} | "
            f"{fmt(rec.precompute_batch_total_mean, 2)} | {fmt(rec.precompute_mv_support_mean, 2)} | "
            f"{fmt(rec.precompute_gate_prepare_mean, 2)} |"
        )
    lines.append("")
    lines.append("## Gate Readout")
    lines.append("")
    for rec in compares:
        if rec.label.upper() == "F0":
            ok, details = f0_metric_gate(rec, baseline)
            lines.append(f"- `{rec.label}` metric gate: `{'PASS' if ok else 'FAIL'}` ({', '.join(details)})")
            lines.append("- `F0` visual gate: manual fixed-triplet review required.")
        else:
            ok, improvements = promotion_metric_gate(rec, baseline)
            lines.append(
                f"- `{rec.label}` promotion metric gate: `{'PASS' if ok else 'FAIL'}` "
                f"(ghost<=4.98: {'yes' if rec.ghost_visual <= PROMOTION_GHOST_MAX else 'no'}, improved_metrics={improvements}/3)"
            )
            lines.append(f"- `{rec.label}` visual gate: manual fixed-triplet review required.")
    lines.append("")
    lines.append("## Presence Diagnostics")
    lines.append("")
    for rec in compares:
        lines.append(
            f"- `{rec.label}`: `presence_enabled={fmt(rec.fg_conf_presence_enabled, 1)}`, "
            f"`pred_mean={fmt(rec.fg_conf_presence_pred_mean)}`, `tgt_mean={fmt(rec.fg_conf_presence_tgt_mean)}`, "
            f"`target_floor={fmt(rec.fg_conf_presence_target_floor)}`, `active_ratio={fmt(rec.fg_conf_presence_active_ratio)}`, "
            f"`loss={fmt(rec.fg_conf_presence_loss)}`, `mean_loss={fmt(rec.mean_loss_fg_conf_presence)}`"
        )
    lines.append("")
    lines.append("## Fixed Triplet Ghost Metrics")
    lines.append("")
    lines.append(f"- `{baseline.label}`:")
    for item in baseline.triplet_metrics:
        lines.append(f"  - `{triplet_metric_line(item)}`")
    for rec in compares:
        lines.append(f"- `{rec.label}`:")
        for item in rec.triplet_metrics:
            lines.append(f"  - `{triplet_metric_line(item)}`")
    lines.append("")
    lines.append("## Fixed Triplets")
    lines.append("")
    lines.append(f"- `{baseline.label}` triplet:")
    for path in baseline.triplet_pngs:
        lines.append(f"  - `{path}`")
    for rec in compares:
        lines.append(f"- `{rec.label}` triplet:")
        for path in rec.triplet_pngs:
            lines.append(f"  - `{path}`")
    lines.append("")
    lines.append("## Review Focus")
    lines.append("")
    lines.append("- torso solidity")
    lines.append("- forearm / arm definition")
    lines.append("- body boundary sharpness")
    lines.append("- brighter-but-blurrier failure mode")
    lines.append("- gray shell / glow / ghost rebound")
    lines.append("")
    return "\n".join(lines) + "\n"


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(0, 0, 0)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def resize_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    return image.resize(new_size)


def draw_triplet_row(canvas: Image.Image, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, top: int, rec: Record, baseline: Record, row_w: int) -> int:
    thumb_w = 320
    thumb_h = 220
    gap = 20
    left = 40
    header = (
        f"{rec.label} | ghost_visual={fmt(rec.ghost_visual)} ({delta_fmt(rec.ghost_visual, baseline.ghost_visual)}) | "
        f"fg_luma={fmt(rec.fg_luma)} ({delta_fmt(rec.fg_luma, baseline.fg_luma)}) | "
        f"fg_contrast={fmt(rec.fg_contrast)} ({delta_fmt(rec.fg_contrast, baseline.fg_contrast)}) | "
        f"fg_tgt_l1={fmt(rec.fg_tgt_l1)} ({delta_fmt(rec.fg_tgt_l1, baseline.fg_tgt_l1)})"
    )
    draw_text(draw, (left, top), header, font)
    draw_text(
        draw,
        (left, top + 18),
        (
            f"boost={fmt(rec.fg_supervision_boost, 2)} bg_floor={fmt(rec.fg_supervision_bg_floor, 2)} "
            f"region={rec.fg_supervision_region_mode} erode_px={fmt(rec.fg_supervision_region_erode_px, 1)} "
            f"lambda_presence={fmt(rec.lambda_fg_conf_presence, 3)} target_ratio={fmt(rec.fg_conf_presence_target_ratio, 2)}"
        ),
        font,
    )
    draw_text(
        draw,
        (left, top + 32),
        (
            f"precompute tf_mean={fmt(rec.precompute_teacher_forward_mean, 2)}s "
            f"batch_mean={fmt(rec.precompute_batch_total_mean, 2)}s "
            f"mv_mean={fmt(rec.precompute_mv_support_mean, 2)}s "
            f"gate_mean={fmt(rec.precompute_gate_prepare_mean, 2)}s"
        ),
        font,
    )
    draw_text(
        draw,
        (left, top + 46),
        (
            f"presence enabled={fmt(rec.fg_conf_presence_enabled, 1)} pred={fmt(rec.fg_conf_presence_pred_mean)} "
            f"tgt={fmt(rec.fg_conf_presence_tgt_mean)} floor={fmt(rec.fg_conf_presence_target_floor)} "
            f"loss={fmt(rec.fg_conf_presence_loss)} mean_loss={fmt(rec.mean_loss_fg_conf_presence)}"
        ),
        font,
    )
    y = top + 72
    for idx in range(3):
        x = left + idx * (thumb_w + gap)
        draw.rectangle((x - 2, y - 2, x + thumb_w + 2, y + thumb_h + 2), outline=(80, 80, 80), width=2)
        if idx < len(rec.triplet_pngs) and os.path.exists(rec.triplet_pngs[idx]):
            image = Image.open(rec.triplet_pngs[idx]).convert("RGB")
            thumb = resize_to_fit(image, thumb_w, thumb_h)
            paste_x = x + (thumb_w - thumb.size[0]) // 2
            paste_y = y + (thumb_h - thumb.size[1]) // 2
            canvas.paste(thumb, (paste_x, paste_y))
        label = f"step{idx:06d}"
        if idx < len(rec.triplet_metrics):
            item = rec.triplet_metrics[idx]
            label += (
                f" g={fmt(item.get('ghost_visual_score', float('nan')))} "
                f"pk={fmt(item.get('peak_count', float('nan')), 1)} "
                f"w={fmt(item.get('width_ratio', float('nan')))} "
                f"a={fmt(item.get('area_ratio', float('nan')))}"
            )
        draw_text(draw, (x, y + thumb_h + 6), label, font)
    return y + thumb_h + 40


def build_png(baseline: Record, compares: list[Record], out_path: str) -> None:
    font = ImageFont.load_default()
    width = 1100
    row_height = 344
    height = 210 + row_height * (1 + len(compares))
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)

    draw_text(draw, (40, 24), "Foreground Presence Validation Compare", font)
    draw_text(
        draw,
        (40, 46),
        (
            f"Baseline {baseline.label}: ghost_visual={fmt(baseline.ghost_visual)} "
            f"fg_luma={fmt(baseline.fg_luma)} fg_contrast={fmt(baseline.fg_contrast)} fg_tgt_l1={fmt(baseline.fg_tgt_l1)} "
            f"width_ratio={fmt(baseline.width_ratio_mean)} area_ratio={fmt(baseline.area_ratio_mean)}"
        ),
        font,
    )
    draw_text(draw, (40, 68), "Fixed triplet review uses step000000 / step000001 / step000002 for every run.", font)
    draw_text(
        draw,
        (40, 84),
        (
            f"Baseline precompute: tf_mean={fmt(baseline.precompute_teacher_forward_mean, 2)}s "
            f"batch_mean={fmt(baseline.precompute_batch_total_mean, 2)}s "
            f"mv_mean={fmt(baseline.precompute_mv_support_mean, 2)}s gate_mean={fmt(baseline.precompute_gate_prepare_mean, 2)}s"
        ),
        font,
    )
    draw_text(
        draw,
        (40, 100),
        (
            f"Baseline presence: enabled={fmt(baseline.fg_conf_presence_enabled, 1)} pred={fmt(baseline.fg_conf_presence_pred_mean)} "
            f"tgt={fmt(baseline.fg_conf_presence_tgt_mean)} floor={fmt(baseline.fg_conf_presence_target_floor)} "
            f"loss={fmt(baseline.fg_conf_presence_loss)}"
        ),
        font,
    )

    top = 132
    draw_text(draw, (40, top), "Locked Baseline Triplet", font)
    top = draw_triplet_row(canvas, draw, font, top + 18, baseline, baseline, width - 80)
    for rec in compares:
        top = draw_triplet_row(canvas, draw, font, top + 12, rec, baseline, width - 80)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:\\vggt")
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--baseline-label", default="Locked G0 px=5")
    ap.add_argument("--compare", action="append", default=[])
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    baseline = load_record(repo_dir, args.baseline_label, args.baseline_json)
    compares = [load_record(repo_dir, label, path) for label, path in parse_compare_specs(args.compare)]
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png)

    md = build_markdown(baseline, compares)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    build_png(baseline, compares, out_png)
    print(f"[render-fg-presence-compare] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
