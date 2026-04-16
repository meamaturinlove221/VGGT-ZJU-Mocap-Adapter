from __future__ import annotations

import argparse
import csv
import json
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tail_lines(path: Path, n: int = 20) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consolas.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_line(text: str, width: int = 98) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False) or [text]


def render_text_png(out_path: Path, sections: list[tuple[str, list[str]]]) -> None:
    title_font = pick_font(30)
    body_font = pick_font(22)
    line_h = 32
    margin = 40
    lines: list[tuple[str, str]] = []
    for title, body in sections:
        lines.append(("title", title))
        for item in body:
            for wrapped in wrap_line(item):
                lines.append(("body", wrapped))
        lines.append(("body", ""))

    height = margin * 2 + line_h * max(18, len(lines) + 2)
    img = Image.new("RGB", (1800, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for kind, line in lines:
        font = title_font if kind == "title" else body_font
        fill = (0, 0, 0) if kind == "body" else (25, 55, 125)
        draw.text((margin, y), line, font=font, fill=fill)
        y += line_h + (8 if kind == "title" else 0)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--ft-csv", required=True)
    ap.add_argument("--ghost-csv", required=True)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--autoloop-json", required=True)
    ap.add_argument("--watch-json", required=True)
    ap.add_argument("--modal-progress-json", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log = Path(args.run_log).resolve()
    ft_csv = Path(args.ft_csv).resolve()
    ghost_csv = Path(args.ghost_csv).resolve()
    candidate_json = Path(args.candidate_json).resolve()
    autoloop_json = Path(args.autoloop_json).resolve()
    watch_json = Path(args.watch_json).resolve()
    modal_progress_json = Path(args.modal_progress_json).resolve()

    for src in (run_log, ft_csv, ghost_csv, candidate_json, autoloop_json, watch_json, modal_progress_json):
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    ft_rows = read_csv_rows(ft_csv)
    ghost_rows = read_csv_rows(ghost_csv)
    candidate = read_json(candidate_json)
    autoloop = read_json(autoloop_json)
    watch = read_json(watch_json)
    modal_progress = read_json(modal_progress_json)
    log_tail = tail_lines(run_log, n=18)

    ft_last = ft_rows[-1] if ft_rows else {}
    ghost_last = ghost_rows[-1] if ghost_rows else {}

    md_lines = [
        "# P0 Single-Run Validation Report",
        "",
        "## Conclusion",
        "- P0 did not pass.",
        "- Conservative stage2 contract engaged: depth_unproject + mv_support=off + point_target_blend_by_mv_support=off.",
        "- Live heartbeat still did not surface through the manual cloud run path.",
        "- Latest candidate single-source JSON still reflects the last completed stage1 candidate, not the later stage2 timeout.",
        "",
        "## Key Files",
        f"- run_log: {run_log}",
        f"- ft_csv: {ft_csv}",
        f"- ghost_csv: {ghost_csv}",
        f"- candidate_json: {candidate_json}",
        f"- autoloop_json: {autoloop_json}",
        f"- watch_json: {watch_json}",
        f"- modal_progress_json: {modal_progress_json}",
        "",
        "## Fresh FT Row",
        f"- status: {ft_last.get('status', '')}",
        f"- reason: {ft_last.get('reason', '')}",
        f"- eval_num_src_views: {ft_last.get('eval_num_src_views', '')}",
        f"- declared: {ft_last.get('eval_num_src_views_declared', '')}",
        f"- pointmap_source_requested: {ft_last.get('pointmap_source_requested', '')}",
        f"- pointmap_source_resolved: {ft_last.get('pointmap_source_resolved', '')}",
        f"- candidate_invalid_reason: {ft_last.get('candidate_invalid_reason', '')}",
        "",
        "## Latest Candidate JSON",
        f"- source: {candidate.get('source', '')}",
        f"- candidate_family: {candidate.get('candidate_family', '')}",
        f"- ft_failure_reason: {candidate.get('ft_failure_reason', '')}",
        f"- precompute_mv_support_on: {candidate.get('precompute_mv_support_on', '')}",
        f"- point_target_blend_by_mv_support: {candidate.get('point_target_blend_by_mv_support', '')}",
        "",
        "## Frozen Latest State",
        f"- autoloop.current_stage: {autoloop.get('current_stage', '')}",
        f"- autoloop.active_candidate_result_json: {autoloop.get('active_candidate_result_json', '')}",
        f"- watch.active_candidate_result_json: {watch.get('active_candidate_result_json', '')}",
        f"- modal_progress.state: {modal_progress.get('state', '')}",
        f"- modal_progress.note: {modal_progress.get('note', '')}",
        "",
        "## Log Tail",
    ]
    md_lines.extend([f"- {line}" for line in log_tail])
    (out_dir / "p0_single_run_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    sections = [
        ("P0 Single-Run Validation", [
            "结论: P0 未通过；但这轮已经证明问题仍在 P0 观测/执行链，不在算法面。",
            "stage2 保守合同已生效: depth_unproject, precompute_mv_support_on=off, point_target_blend_by_mv_support=off。",
            "live heartbeat 仍未穿透到外层；manual 单轮运行里 hb_source/hb_phase 为空。",
            "candidate_result_latest.json 仍停在最后完成的 stage1 候选，而不是后续 stage2 timeout。"
        ]),
        ("Fresh FT Row", [
            f"status={ft_last.get('status', '')}",
            f"reason={ft_last.get('reason', '')}",
            f"eval_num_src_views={ft_last.get('eval_num_src_views', '')} declared={ft_last.get('eval_num_src_views_declared', '')}",
            f"pointmap_source_requested={ft_last.get('pointmap_source_requested', '')}",
            f"pointmap_source_resolved={ft_last.get('pointmap_source_resolved', '')}",
            f"candidate_invalid_reason={ft_last.get('candidate_invalid_reason', '')}",
        ]),
        ("Latest Candidate JSON", [
            f"candidate_json={candidate_json}",
            f"candidate_family={candidate.get('candidate_family', '')}",
            f"ft_failure_reason={candidate.get('ft_failure_reason', '')}",
            f"precompute_mv_support_on={candidate.get('precompute_mv_support_on', '')}",
            f"point_target_blend_by_mv_support={candidate.get('point_target_blend_by_mv_support', '')}",
        ]),
        ("Open These Local Paths", [
            str(run_log),
            str(ft_csv),
            str(ghost_csv),
            str(candidate_json),
            str(autoloop_json),
            str(watch_json),
            str(modal_progress_json),
        ]),
    ]
    render_text_png(out_dir / "p0_single_run_report.png", sections)


if __name__ == "__main__":
    main()
