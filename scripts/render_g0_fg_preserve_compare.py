import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass
class RunRecord:
    label: str
    candidate_path: str
    run_timestamp: str
    preserve_px: int
    ghost_visual: float
    ghost_mean: float
    fg_luma: float
    fg_contrast: float
    fg_tgt_l1: float
    depth_conf_fg_preserved_active: int
    depth_conf_fg_after_support_mean: float
    depth_conf_fg_final_mean: float
    best_visual_png: str


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_record(repo_dir: str, label: str, path: str) -> RunRecord:
    data = read_json(path)
    best_visual_rel = str(data.get("best_visual_png") or "")
    run_timestamp = str(data.get("run_timestamp") or "")
    if not best_visual_rel and run_timestamp:
        inferred = os.path.join(
            "logs",
            "modal_phase5",
            f"_ghost_eval_mv_0.001_mvmask_0_default_{run_timestamp}",
            "infer_val_e005_cat_fg_mask_pred_tgt_step000001.png",
        )
        inferred_abs = os.path.join(repo_dir, inferred)
        if os.path.exists(inferred_abs):
            best_visual_rel = inferred
    best_visual_abs = (
        best_visual_rel
        if os.path.isabs(best_visual_rel)
        else os.path.join(repo_dir, best_visual_rel)
    )
    return RunRecord(
        label=label,
        candidate_path=path,
        run_timestamp=run_timestamp,
        preserve_px=int(float(data.get("precompute_mv_support_fg_preserve_px") or 0)),
        ghost_visual=to_float(data.get("ghost_visual_score")),
        ghost_mean=to_float(data.get("ghost_score_mean")),
        fg_luma=to_float(data.get("fg_pred_luma_mean")),
        fg_contrast=to_float(data.get("fg_pred_contrast")),
        fg_tgt_l1=to_float(data.get("fg_pred_tgt_l1")),
        depth_conf_fg_preserved_active=int(float(data.get("depth_conf_fg_preserved_active") or 0)),
        depth_conf_fg_after_support_mean=to_float(data.get("depth_conf_fg_after_support_mean")),
        depth_conf_fg_final_mean=to_float(data.get("depth_conf_fg_final_mean")),
        best_visual_png=best_visual_abs,
    )


def fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def build_markdown(records: list[RunRecord], best_balance: RunRecord, best_ghost: RunRecord) -> str:
    lines: list[str] = []
    lines.append("# G0 Fg Preserve Comparison")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Fixed contract: `G0`")
    lines.append("- Only changed knob: `precompute_mv_support_fg_preserve_px`")
    lines.append("- Compared settings: `2`, `3`, `5`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"- Recommended default: `{best_balance.preserve_px}`. "
        f"It keeps `ghost_visual_score <= 4.98` while improving all three foreground metrics over the old `px=3` baseline."
    )
    lines.append(
        f"- Best pure ghost setting: `{best_ghost.preserve_px}` with `ghost_visual_score={fmt(best_ghost.ghost_visual)}`."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Setting | Run Timestamp | ghost_visual | ghost_mean | fg_luma | fg_contrast | fg_tgt_l1 | halo_active | fg_final>=after_support |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for rec in records:
        halo_ok = rec.depth_conf_fg_final_mean >= rec.depth_conf_fg_after_support_mean
        lines.append(
            f"| `px={rec.preserve_px}` | `{rec.run_timestamp}` | {fmt(rec.ghost_visual)} | {fmt(rec.ghost_mean)} | "
            f"{fmt(rec.fg_luma)} | {fmt(rec.fg_contrast)} | {fmt(rec.fg_tgt_l1)} | "
            f"{rec.depth_conf_fg_preserved_active} | {'yes' if halo_ok else 'no'} |"
        )
    lines.append("")
    lines.append("## Visual Paths")
    lines.append("")
    for rec in records:
        lines.append(f"- `px={rec.preserve_px}` visual: `{rec.best_visual_png}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(0, 0, 0)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def resize_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    return image.resize(new_size)


def build_png(records: list[RunRecord], best_balance: RunRecord, best_ghost: RunRecord, out_path: str) -> None:
    font = ImageFont.load_default()
    width = 1860
    height = 1380
    bg = (248, 248, 246)
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    draw_text(draw, (40, 28), "G0 Fg Preserve Comparison", font)
    draw_text(draw, (40, 54), "English local-only comparison for px=2 / px=3 / px=5", font)
    draw_text(
        draw,
        (40, 88),
        f"Recommended default: px={best_balance.preserve_px} | Best pure ghost: px={best_ghost.preserve_px}",
        font,
    )

    top = 130
    left = 40
    col_x = [left, 420, 800, 1180, 1360, 1540, 1710]
    row_h = 28
    headers = ["Setting", "Run", "ghost_visual", "ghost_mean", "fg_luma", "fg_contrast", "fg_tgt_l1"]
    for idx, header in enumerate(headers):
        draw_text(draw, (col_x[idx], top), header, font)

    for row_idx, rec in enumerate(records, start=1):
        y = top + row_idx * row_h
        fill = (0, 90, 0) if rec.preserve_px == best_balance.preserve_px else (0, 0, 120) if rec.preserve_px == best_ghost.preserve_px else (0, 0, 0)
        values = [
            f"px={rec.preserve_px}",
            rec.run_timestamp,
            fmt(rec.ghost_visual),
            fmt(rec.ghost_mean),
            fmt(rec.fg_luma),
            fmt(rec.fg_contrast),
            fmt(rec.fg_tgt_l1),
        ]
        for idx, value in enumerate(values):
            draw_text(draw, (col_x[idx], y), value, font, fill=fill if idx == 0 else (0, 0, 0))

    draw_text(draw, (40, 260), "Best Visuals", font)

    thumb_top = 300
    thumb_w = 560
    thumb_h = 460
    gap = 40
    for idx, rec in enumerate(records):
        x = 40 + idx * (thumb_w + gap)
        y = thumb_top
        border = (32, 120, 32) if rec.preserve_px == best_balance.preserve_px else (45, 82, 158) if rec.preserve_px == best_ghost.preserve_px else (90, 90, 90)
        draw.rectangle((x - 2, y - 2, x + thumb_w + 2, y + thumb_h + 2), outline=border, width=3)
        if os.path.exists(rec.best_visual_png):
            image = Image.open(rec.best_visual_png).convert("RGB")
            thumb = resize_to_fit(image, thumb_w, thumb_h)
            paste_x = x + (thumb_w - thumb.size[0]) // 2
            paste_y = y + (thumb_h - thumb.size[1]) // 2
            canvas.paste(thumb, (paste_x, paste_y))
        draw_text(draw, (x, y + thumb_h + 12), f"px={rec.preserve_px}  ghost_visual={fmt(rec.ghost_visual)}", font)
        draw_text(draw, (x, y + thumb_h + 34), f"fg_luma={fmt(rec.fg_luma)}  fg_contrast={fmt(rec.fg_contrast)}", font)
        draw_text(draw, (x, y + thumb_h + 56), f"fg_tgt_l1={fmt(rec.fg_tgt_l1)}  halo={rec.depth_conf_fg_preserved_active}", font)

    draw_text(draw, (40, 860), "Decision", font)
    draw_text(
        draw,
        (40, 888),
        (
            f"px={best_balance.preserve_px} is the recommended balance setting because it stays under the ghost target "
            "and improves all foreground-presence metrics over the old px=3 baseline."
        ),
        font,
    )
    draw_text(
        draw,
        (40, 914),
        (
            f"px={best_ghost.preserve_px} remains the ghost-first fallback if the next round prioritizes the strongest possible ghost suppression."
        ),
        font,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:\\vggt")
    ap.add_argument("--px2-json", default="logs/modal_phase5/candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260312_140724.json")
    ap.add_argument("--px3-json", default="logs/modal_phase5/candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260312_120041.json")
    ap.add_argument("--px5-json", default="logs/modal_phase5/candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260312_144819.json")
    ap.add_argument("--out-md", default="logs/modal_phase5/reports/g0_fg_preserve_compare_en_latest.md")
    ap.add_argument("--out-png", default="logs/modal_phase5/reports/g0_fg_preserve_compare_en_latest.png")
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    px2 = os.path.join(repo_dir, args.px2_json) if not os.path.isabs(args.px2_json) else args.px2_json
    px3 = os.path.join(repo_dir, args.px3_json) if not os.path.isabs(args.px3_json) else args.px3_json
    px5 = os.path.join(repo_dir, args.px5_json) if not os.path.isabs(args.px5_json) else args.px5_json
    out_md = os.path.join(repo_dir, args.out_md) if not os.path.isabs(args.out_md) else args.out_md
    out_png = os.path.join(repo_dir, args.out_png) if not os.path.isabs(args.out_png) else args.out_png

    records = [
        load_record(repo_dir, "px2", px2),
        load_record(repo_dir, "px3", px3),
        load_record(repo_dir, "px5", px5),
    ]
    best_ghost = min(records, key=lambda x: x.ghost_visual)
    old_px3 = next(r for r in records if r.preserve_px == 3)
    candidates = [
        r for r in records
        if r.ghost_visual <= 4.98
        and r.fg_luma > old_px3.fg_luma
        and r.fg_contrast > old_px3.fg_contrast
        and r.fg_tgt_l1 < old_px3.fg_tgt_l1
    ]
    best_balance = candidates[0] if candidates else min(records, key=lambda x: x.ghost_visual)

    md = build_markdown(records, best_balance, best_ghost)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    build_png(records, best_balance, best_ghost, out_png)
    print(f"[render-g0-fg-preserve-compare] md={out_md} png={out_png}")


if __name__ == "__main__":
    main()
