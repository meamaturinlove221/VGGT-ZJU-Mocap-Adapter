import argparse
import csv
import json
import math
import os
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def resolve_triplet(repo_dir: str, data: dict[str, Any]) -> tuple[list[str], str]:
    best_visual_rel = str(data.get("best_visual_png") or "")
    best_visual_abs = ""
    if best_visual_rel:
        best_visual_abs = best_visual_rel if os.path.isabs(best_visual_rel) else os.path.join(repo_dir, best_visual_rel)
    triplet: list[str] = []
    if best_visual_abs and os.path.exists(best_visual_abs):
        directory = os.path.dirname(best_visual_abs)
        for idx in range(3):
            path = os.path.join(directory, f"infer_val_e005_cat_fg_mask_pred_tgt_step{idx:06d}.png")
            if os.path.exists(path):
                triplet.append(path)
    return triplet, best_visual_abs


def resolve_ghost_rows_csv(repo_dir: str, data: dict[str, Any]) -> str:
    rel = str(data.get("ghost_rows_csv") or "")
    if rel:
        abs_path = rel if os.path.isabs(rel) else os.path.join(repo_dir, rel)
        if os.path.exists(abs_path):
            return abs_path
    run_ts = str(data.get("run_timestamp") or "")
    run_tag = str(data.get("run_tag") or "")
    if not run_ts or not run_tag:
        return ""
    path = os.path.join(repo_dir, "logs", "modal_phase5", f"ghost_score_rows_{run_tag}_{run_ts}.csv")
    return path if os.path.exists(path) else ""


def load_triplet_metrics(csv_path: str) -> list[dict[str, float]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    out: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(to_float(row.get("step"), default=-1))
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


def resize_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    return image.resize(new_size)


def build_markdown(
    baseline_label: str,
    baseline_json: str,
    data: dict[str, Any],
    triplet: list[str],
    triplet_metrics: list[dict[str, float]],
    lambda_presence: float,
    target_ratio: float,
) -> str:
    lines: list[str] = []
    lines.append("# Presence-Only P1 Local Diagnostics")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- active baseline: `{baseline_label}`")
    lines.append(f"- candidate json: `{baseline_json}`")
    lines.append(
        "- metrics: "
        f"`ghost_visual={fmt(to_float(data.get('ghost_visual_score')))}`, "
        f"`ghost_mean={fmt(to_float(data.get('ghost_score_mean')))}`, "
        f"`fg_luma={fmt(to_float(data.get('fg_pred_luma_mean')))}`, "
        f"`fg_contrast={fmt(to_float(data.get('fg_pred_contrast')))}`, "
        f"`fg_tgt_l1={fmt(to_float(data.get('fg_pred_tgt_l1')))}`"
    )
    lines.append("")
    lines.append("## Planned P1 Contract")
    lines.append("")
    lines.append("- `fg_supervision_boost = 1.0`")
    lines.append("- `fg_supervision_bg_floor = 0.0`")
    lines.append("- `fg_supervision_region_mode = all`")
    lines.append("- `fg_supervision_region_erode_px = 0`")
    lines.append(f"- `lambda_fg_conf_presence = {lambda_presence:.3f}`")
    lines.append(f"- `fg_conf_presence_target_ratio = {target_ratio:.2f}`")
    lines.append("")
    lines.append("## Baseline Presence Readout")
    lines.append("")
    lines.append(
        "- current baseline fields: "
        f"`presence_enabled={fmt(to_float(data.get('fg_conf_presence_enabled'), default=0.0), 1)}`, "
        f"`pred_mean={fmt(to_float(data.get('fg_conf_presence_pred_mean'), default=0.0))}`, "
        f"`tgt_mean={fmt(to_float(data.get('fg_conf_presence_tgt_mean'), default=0.0))}`, "
        f"`target_floor={fmt(to_float(data.get('fg_conf_presence_target_floor'), default=0.0))}`, "
        f"`active_ratio={fmt(to_float(data.get('fg_conf_presence_active_ratio'), default=0.0))}`, "
        f"`loss={fmt(to_float(data.get('fg_conf_presence_loss'), default=0.0))}`"
    )
    lines.append("")
    lines.append("## Fixed Triplet Ghost Metrics")
    lines.append("")
    for item in triplet_metrics:
        lines.append(f"- `{triplet_metric_line(item)}`")
    lines.append("")
    lines.append("## Fixed Triplets")
    lines.append("")
    for path in triplet:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Review Focus")
    lines.append("")
    lines.append("- confirm P1 is pure presence-only: no boost, no bg floor, no region-only mask")
    lines.append("- compare post-run presence diagnostics against these baseline zeros/non-activity fields")
    lines.append("- focus on step000002 for ghost rebound sensitivity")
    lines.append("- reject brighter-but-blurrier or extra-peak failure modes")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_png(
    baseline_label: str,
    data: dict[str, Any],
    triplet: list[str],
    triplet_metrics: list[dict[str, float]],
    lambda_presence: float,
    target_ratio: float,
    out_png: str,
) -> None:
    font = ImageFont.load_default()
    width = 1100
    height = 430
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), "Presence-Only P1 Local Diagnostics", font=font, fill=(0, 0, 0))
    draw.text(
        (40, 46),
        (
            f"Baseline {baseline_label}: ghost_visual={fmt(to_float(data.get('ghost_visual_score')))} "
            f"fg_luma={fmt(to_float(data.get('fg_pred_luma_mean')))} "
            f"fg_contrast={fmt(to_float(data.get('fg_pred_contrast')))} "
            f"fg_tgt_l1={fmt(to_float(data.get('fg_pred_tgt_l1')))}"
        ),
        font=font,
        fill=(0, 0, 0),
    )
    draw.text(
        (40, 64),
        (
            "Planned P1: boost=1.00 bg_floor=0.00 region=all erode_px=0 "
            f"lambda_presence={lambda_presence:.3f} target_ratio={target_ratio:.2f}"
        ),
        font=font,
        fill=(0, 0, 0),
    )
    draw.text(
        (40, 82),
        (
            f"Baseline presence fields: enabled={fmt(to_float(data.get('fg_conf_presence_enabled'), default=0.0), 1)} "
            f"pred={fmt(to_float(data.get('fg_conf_presence_pred_mean'), default=0.0))} "
            f"tgt={fmt(to_float(data.get('fg_conf_presence_tgt_mean'), default=0.0))} "
            f"floor={fmt(to_float(data.get('fg_conf_presence_target_floor'), default=0.0))} "
            f"loss={fmt(to_float(data.get('fg_conf_presence_loss'), default=0.0))}"
        ),
        font=font,
        fill=(0, 0, 0),
    )
    thumb_w = 320
    thumb_h = 220
    gap = 20
    y = 126
    for idx in range(3):
        x = 40 + idx * (thumb_w + gap)
        draw.rectangle((x - 2, y - 2, x + thumb_w + 2, y + thumb_h + 2), outline=(80, 80, 80), width=2)
        if idx < len(triplet) and os.path.exists(triplet[idx]):
            image = Image.open(triplet[idx]).convert("RGB")
            thumb = resize_to_fit(image, thumb_w, thumb_h)
            canvas.paste(thumb, (x + (thumb_w - thumb.size[0]) // 2, y + (thumb_h - thumb.size[1]) // 2))
        label = f"step{idx:06d}"
        if idx < len(triplet_metrics):
            item = triplet_metrics[idx]
            label += (
                f" g={fmt(item.get('ghost_visual_score', float('nan')))} "
                f"pk={fmt(item.get('peak_count', float('nan')), 1)} "
                f"w={fmt(item.get('width_ratio', float('nan')))} "
                f"a={fmt(item.get('area_ratio', float('nan')))}"
            )
        draw.text((x, y + thumb_h + 8), label, font=font, fill=(0, 0, 0))
    draw.text((40, 386), "Review focus: step000002 ghost sensitivity, torso solidity, no extra peak rebound.", font=font, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:\\vggt")
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--baseline-label", default="Working baseline F0 px=5")
    ap.add_argument("--lambda-presence", type=float, default=0.005)
    ap.add_argument("--target-ratio", type=float, default=0.8)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    baseline_json = args.baseline_json if os.path.isabs(args.baseline_json) else os.path.join(repo_dir, args.baseline_json)
    data = read_json(baseline_json)
    triplet, _ = resolve_triplet(repo_dir, data)
    triplet_metrics = load_triplet_metrics(resolve_ghost_rows_csv(repo_dir, data))
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png)

    md = build_markdown(
        baseline_label=args.baseline_label,
        baseline_json=baseline_json,
        data=data,
        triplet=triplet,
        triplet_metrics=triplet_metrics,
        lambda_presence=float(args.lambda_presence),
        target_ratio=float(args.target_ratio),
    )
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    build_png(
        baseline_label=args.baseline_label,
        data=data,
        triplet=triplet,
        triplet_metrics=triplet_metrics,
        lambda_presence=float(args.lambda_presence),
        target_ratio=float(args.target_ratio),
        out_png=out_png,
    )
    print(f"[render-fg-presence-only-diagnostics] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
