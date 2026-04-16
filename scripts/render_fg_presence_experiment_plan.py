import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass
class CurrentRecord:
    label: str
    json_path: str
    preserve_px: int
    ghost_visual: float
    fg_luma: float
    fg_contrast: float
    fg_tgt_l1: float


@dataclass
class ExperimentSpec:
    label: str
    description: str
    fg_supervision_boost: float
    fg_supervision_bg_floor: float
    lambda_fg_conf_presence: float
    fg_conf_presence_target_ratio: float


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_current_record(repo_dir: str, label: str, rel_path: str) -> CurrentRecord:
    path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_dir, rel_path)
    data = read_json(path)
    return CurrentRecord(
        label=label,
        json_path=path,
        preserve_px=int(float(data.get("precompute_mv_support_fg_preserve_px") or 0)),
        ghost_visual=to_float(data.get("ghost_visual_score")),
        fg_luma=to_float(data.get("fg_pred_luma_mean")),
        fg_contrast=to_float(data.get("fg_pred_contrast")),
        fg_tgt_l1=to_float(data.get("fg_pred_tgt_l1")),
    )


def fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def build_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            label="F0",
            description="Fixed G0 px=5 geometry, no extra foreground-positive supervision.",
            fg_supervision_boost=1.0,
            fg_supervision_bg_floor=0.0,
            lambda_fg_conf_presence=0.0,
            fg_conf_presence_target_ratio=0.9,
        ),
        ExperimentSpec(
            label="F1",
            description="Foreground supervision boost with a weak background floor so the boost actually changes relative weighting.",
            fg_supervision_boost=1.5,
            fg_supervision_bg_floor=0.05,
            lambda_fg_conf_presence=0.0,
            fg_conf_presence_target_ratio=0.9,
        ),
        ExperimentSpec(
            label="F2",
            description="Stronger version of F1; higher foreground emphasis but also higher ghost-rebound risk.",
            fg_supervision_boost=2.0,
            fg_supervision_bg_floor=0.05,
            lambda_fg_conf_presence=0.0,
            fg_conf_presence_target_ratio=0.9,
        ),
        ExperimentSpec(
            label="F3",
            description="F1 plus a weak foreground confidence floor to discourage a thin, low-confidence human shell.",
            fg_supervision_boost=1.5,
            fg_supervision_bg_floor=0.05,
            lambda_fg_conf_presence=0.02,
            fg_conf_presence_target_ratio=0.9,
        ),
    ]


def build_markdown(records: list[CurrentRecord], exps: list[ExperimentSpec]) -> str:
    lines: list[str] = []
    lines.append("# Foreground Presence Local Experiment Plan")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Local-only implementation and validation. No cloud rerun was performed in this step.")
    lines.append("- Fixed geometry contract: `G0 px=5`, `bg_only`, halo preserved.")
    lines.append("- Important repo-specific finding: the active G0 lane is driven by `finetune_vggt_pseudo.py`, not by `train_view_decoder_ablation.py`.")
    lines.append("")
    lines.append("## Current Evidence")
    lines.append("")
    lines.append("| Current Setting | preserve_px | ghost_visual | fg_luma | fg_contrast | fg_tgt_l1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for rec in records:
        lines.append(
            f"| `{rec.label}` | {rec.preserve_px} | {fmt(rec.ghost_visual)} | {fmt(rec.fg_luma)} | {fmt(rec.fg_contrast)} | {fmt(rec.fg_tgt_l1)} |"
        )
    lines.append("")
    lines.append("Interpretation: support-generation fixes moved the ghost metric far more than the foreground-presence metrics. The next local experiment lane should therefore target foreground-positive supervision, not more support-generation churn.")
    lines.append("")
    lines.append("## Implemented Experiment Matrix")
    lines.append("")
    lines.append("| Experiment | fg_supervision_boost | fg_supervision_bg_floor | lambda_fg_conf_presence | fg_conf_presence_target_ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    for exp in exps:
        lines.append(
            f"| `{exp.label}` | {fmt(exp.fg_supervision_boost, 2)} | {fmt(exp.fg_supervision_bg_floor, 2)} | {fmt(exp.lambda_fg_conf_presence, 2)} | {fmt(exp.fg_conf_presence_target_ratio, 2)} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `fg_supervision_boost` is implemented in `finetune_vggt_pseudo.py` on the active supervision weights.")
    lines.append("- `fg_supervision_bg_floor` is necessary in this repo because the current G0 lane hard-gates validity to foreground when `use_fg_mask=on`; without a small background floor, a pure foreground multiplier would mostly cancel under normalized losses.")
    lines.append("- `lambda_fg_conf_presence` is a weak lower-bound regularizer on foreground confidence, not a generic alpha penalty.")
    lines.append("")
    return "\n".join(lines) + "\n"


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(0, 0, 0)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def draw_profile(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], exp: ExperimentSpec, font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    fg_lo = x0 + int(w * 0.32)
    fg_hi = x0 + int(w * 0.68)
    draw.rectangle(box, outline=(120, 120, 120), width=1)
    bg = max(0.0, exp.fg_supervision_bg_floor)
    fg = max(bg, exp.fg_supervision_boost)
    for x in range(x0 + 1, x1):
        in_fg = fg_lo <= x <= fg_hi
        val = fg if in_fg else max(bg, 0.0 if exp.fg_supervision_bg_floor <= 0 else bg)
        y_val = y1 - 6 - int((min(2.1, val) / 2.1) * (h - 18))
        draw.line((x, y1 - 6, x, y_val), fill=(45, 82, 158), width=1)
    draw.rectangle((fg_lo, y0 + 1, fg_hi, y1 - 1), outline=(32, 120, 32), width=2)
    draw_text(draw, (x0 + 8, y0 + 6), exp.label, font)
    draw_text(draw, (x0 + 8, y0 + 24), f"boost={fmt(exp.fg_supervision_boost, 2)} bg_floor={fmt(exp.fg_supervision_bg_floor, 2)}", font)
    draw_text(draw, (x0 + 8, y0 + 42), f"presence={fmt(exp.lambda_fg_conf_presence, 2)} @ ratio={fmt(exp.fg_conf_presence_target_ratio, 2)}", font)


def draw_presence_curve(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], exp: ExperimentSpec, font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(120, 120, 120), width=1)
    tgt = exp.fg_conf_presence_target_ratio
    lam = exp.lambda_fg_conf_presence
    if lam <= 0:
        draw_text(draw, (x0 + 10, y0 + 10), f"{exp.label}: presence regularizer disabled", font)
        return
    prev = None
    for i in range(101):
        pred = i / 100.0
        loss = lam * max(0.0, tgt - pred)
        px = x0 + 10 + int((pred / 1.0) * (x1 - x0 - 20))
        py = y1 - 10 - int((loss / max(1e-6, lam * tgt)) * (y1 - y0 - 20))
        if prev is not None:
            draw.line((prev[0], prev[1], px, py), fill=(170, 45, 45), width=2)
        prev = (px, py)
    target_x = x0 + 10 + int((tgt / 1.0) * (x1 - x0 - 20))
    draw.line((target_x, y0 + 10, target_x, y1 - 10), fill=(32, 120, 32), width=1)
    draw_text(draw, (x0 + 10, y0 + 10), f"{exp.label} presence-loss curve", font)
    draw_text(draw, (x0 + 10, y0 + 28), f"loss = lambda * relu(target_ratio * tgt_fg_conf_mean - pred_fg_conf_mean)", font)


def build_png(records: list[CurrentRecord], exps: list[ExperimentSpec], out_path: str) -> None:
    font = ImageFont.load_default()
    width = 1880
    height = 1320
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)

    draw_text(draw, (40, 28), "Foreground Presence Local Experiment Plan", font)
    draw_text(draw, (40, 52), "English local-only planning artifact. No cloud rerun was triggered here.", font)
    draw_text(draw, (40, 76), "Active G0 lane audit: finetune_vggt_pseudo.py is the live training path for this probe family.", font)

    draw_text(draw, (40, 120), "Current Evidence", font)
    headers = ["Setting", "preserve_px", "ghost_visual", "fg_luma", "fg_contrast", "fg_tgt_l1"]
    col_x = [40, 210, 360, 540, 690, 860]
    top = 150
    for i, header in enumerate(headers):
        draw_text(draw, (col_x[i], top), header, font)
    for row_idx, rec in enumerate(records, start=1):
        y = top + row_idx * 28
        values = [rec.label, str(rec.preserve_px), fmt(rec.ghost_visual), fmt(rec.fg_luma), fmt(rec.fg_contrast), fmt(rec.fg_tgt_l1)]
        for i, value in enumerate(values):
            draw_text(draw, (col_x[i], y), value, font)
    draw_text(draw, (40, 260), "Support-generation fixes moved ghost much more than foreground-presence metrics.", font)
    draw_text(draw, (40, 282), "That is why the next local lane should target foreground-positive supervision, not more support-generation churn.", font)

    draw_text(draw, (40, 338), "F0-F3 Experiment Matrix", font)
    col2_x = [40, 170, 380, 620, 870]
    headers2 = ["Experiment", "fg_boost", "bg_floor", "lambda_fg_conf_presence", "target_ratio"]
    for i, header in enumerate(headers2):
        draw_text(draw, (col2_x[i], 368), header, font)
    for row_idx, exp in enumerate(exps, start=1):
        y = 368 + row_idx * 28
        values = [exp.label, fmt(exp.fg_supervision_boost, 2), fmt(exp.fg_supervision_bg_floor, 2), fmt(exp.lambda_fg_conf_presence, 2), fmt(exp.fg_conf_presence_target_ratio, 2)]
        for i, value in enumerate(values):
            draw_text(draw, (col2_x[i], y), value, font)
        draw_text(draw, (1030, y), exp.description, font)

    draw_text(draw, (40, 540), "Synthetic Supervision Profiles", font)
    box_w = 420
    box_h = 180
    gap_x = 28
    gap_y = 24
    for idx, exp in enumerate(exps):
        col = idx % 2
        row = idx // 2
        x0 = 40 + col * (box_w + gap_x)
        y0 = 570 + row * (box_h + gap_y)
        draw_profile(draw, (x0, y0, x0 + box_w, y0 + box_h), exp, font)

    draw_text(draw, (980, 540), "Foreground Confidence Presence Curve", font)
    draw_presence_curve(draw, (980, 570, 1820, 860), exps[-1], font)

    draw_text(draw, (980, 904), "Interpretation", font)
    draw_text(draw, (980, 928), "F1/F2 implement the repo-compatible version of foreground-positive weighting.", font)
    draw_text(draw, (980, 950), "A weak BG floor is intentional here because the live G0 lane otherwise hard-gates supervision to FG only.", font)
    draw_text(draw, (980, 972), "F3 adds a weak confidence-floor term; it is not the same as the existing alpha penalty in other scripts.", font)
    draw_text(draw, (980, 994), "This step is local-only. The next cloud action should wait until these knobs are locally validated and chosen.", font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:\\vggt")
    ap.add_argument("--px2-json", default="logs/modal_phase5/candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260312_140724.json")
    ap.add_argument("--px3-json", default="logs/modal_phase5/candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260312_120041.json")
    ap.add_argument("--px5-json", default="logs/modal_phase5/candidate_result_latest.json")
    ap.add_argument("--out-md", default="logs/modal_phase5/reports/fg_presence_local_plan_en_latest.md")
    ap.add_argument("--out-png", default="logs/modal_phase5/reports/fg_presence_local_plan_en_latest.png")
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    records = [
        load_current_record(repo_dir, "ghost-first fallback", args.px2_json),
        load_current_record(repo_dir, "old G0 baseline", args.px3_json),
        load_current_record(repo_dir, "current G0 px=5", args.px5_json),
    ]
    exps = build_experiments()
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png)

    md = build_markdown(records, exps)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    build_png(records, exps, out_png)
    print(f"[render-fg-presence-plan] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
