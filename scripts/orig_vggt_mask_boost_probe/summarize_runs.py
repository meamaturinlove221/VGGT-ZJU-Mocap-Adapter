from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_mask_boost_probe.common import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_EXTEND_STEPS,
    DEFAULT_NATIVE_STEPCURVE_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_PROFILES,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SHORT_STEPS,
    DEFAULT_TGT_CAMERA,
    GHOST_EPS,
    GHOST_REFERENCE,
    POINT_SUPPORT_REFERENCE,
    SUBJECT_L1_EPS,
    SUBJECT_PSNR_EPS,
    SUPPORT_EPS,
    alpha_step_compare_dir,
    alpha_tag,
    alpha_to_fg_boost,
    load_json,
    manifest_path,
    parse_alphas,
    parse_profiles,
    parse_steps,
    profile_metadata,
    profile_tag,
    profile_summary_json,
    profile_summary_md,
    step_tag,
    to_float,
    to_int,
    write_json,
    write_text,
)


def _image_mad(path_a: Path, path_b: Path, mode: str = "RGB") -> float:
    if (not path_a.is_file()) or (not path_b.is_file()):
        return float("nan")
    with Image.open(str(path_a)).convert(mode) as img_a:
        arr_a = np.asarray(img_a, dtype=np.float32) / 255.0
    with Image.open(str(path_b)).convert(mode) as img_b:
        arr_b = np.asarray(img_b, dtype=np.float32) / 255.0
    if arr_a.shape != arr_b.shape:
        return float("nan")
    return float(np.abs(arr_a - arr_b).mean())


def _normalize_path(value: str | Path) -> str:
    return str(Path(value)).replace("\\", "/")


def _load_exact_ghost_row(stage_dir: Path) -> dict:
    rows_path = stage_dir / "ghost_score_rows.csv"
    if not rows_path.is_file():
        raise RuntimeError(f"ghost_score_rows.csv missing: {rows_path}")
    rows: list[dict] = []
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    target = _normalize_path((stage_dir / "cat_fg_mask_pred_tgt_step000000.png").resolve())
    hits = [row for row in rows if _normalize_path(row.get("path", "")) == target]
    if len(hits) != 1:
        raise RuntimeError(f"ghost exact row match failed for {target}; matched={len(hits)}")
    return hits[0]


def _stage_complete(stage_dir: Path) -> bool:
    required = [
        stage_dir / "report.json",
        stage_dir / "point_support_metrics.json",
        stage_dir / "ghost_score_rows.csv",
        stage_dir / "weight_native.png",
        stage_dir / "cat_weight_pred_tgt_subject_bbox.png",
        stage_dir / "cat_fg_mask_pred_tgt_step000000.png",
    ]
    return all(path.is_file() for path in required)


def _blank_stage(stage_dir: Path, *, stage_status: str = "missing", stage_error: str = "") -> dict:
    nan = float("nan")
    return {
        "stage_dir": str(stage_dir),
        "available": False,
        "stage_status": str(stage_status),
        "stage_error": str(stage_error or ""),
        "weight_native": "",
        "point_triplet": "",
        "ghost_triplet": "",
        "pred_native": "",
        "native_psnr": nan,
        "ghost_visual_score": nan,
        "ghost_peak_count": None,
        "subject_support_share": nan,
        "outside_subject_support_share": nan,
        "largest_component_share": nan,
        "secondary_component_mass": nan,
        "support_peak_count": None,
        "subject_psnr": nan,
        "subject_l1": nan,
        "masked_psnr": nan,
        "masked_mae": nan,
        "subject_mask_coverage_ratio": nan,
        "support_total_mass": nan,
        "support_active_mass": nan,
    }


def _load_task_index(task_state_path: Path) -> dict[str, dict]:
    if not task_state_path.is_file():
        return {}
    payload = load_json(task_state_path)
    tasks = payload.get("tasks", [])
    return {
        str(task.get("name", "")).strip(): dict(task)
        for task in tasks
        if str(task.get("name", "")).strip()
    }


def _stage_task_names(profile: str, *, alpha: int | None, step: int, winner: bool) -> list[str]:
    tag = profile_tag(profile)
    stag = step_tag(step)
    if winner:
        base = f"profile_{tag}_winner_{stag}"
    else:
        if alpha is None:
            return []
        base = f"profile_{tag}_{alpha_tag(alpha)}_{stag}"
    return [
        f"{base}_compare",
        f"{base}_sync_compare",
        f"{base}_score_ghost",
        f"{base}_measure_support",
    ]


def _task_stage_blank(stage_dir: Path, task_index: dict[str, dict], task_names: list[str]) -> dict:
    for name in task_names:
        task = task_index.get(name)
        if not task:
            continue
        status = str(task.get("status", "")).strip() or "missing"
        if status == "completed":
            continue
        message = str(task.get("message", "")).strip()
        return _blank_stage(stage_dir, stage_status=status, stage_error=message)
    return _blank_stage(stage_dir)


def _load_stage(stage_dir: Path) -> dict:
    report = load_json(stage_dir / "report.json")
    point = load_json(stage_dir / "point_support_metrics.json")
    ghost = _load_exact_ghost_row(stage_dir)
    native = report.get("metrics", {}).get("native", {})
    return {
        "stage_dir": str(stage_dir),
        "available": True,
        "stage_status": "completed",
        "stage_error": "",
        "weight_native": str(stage_dir / "weight_native.png"),
        "point_triplet": str(stage_dir / "cat_weight_pred_tgt_subject_bbox.png"),
        "ghost_triplet": str(stage_dir / "cat_fg_mask_pred_tgt_step000000.png"),
        "pred_native": str(stage_dir / "pred_native.png"),
        "native_psnr": to_float(native.get("psnr")),
        "ghost_visual_score": to_float(ghost.get("ghost_visual_score")),
        "ghost_peak_count": to_int(ghost.get("peak_count")),
        "subject_support_share": to_float(point.get("subject_support_share")),
        "outside_subject_support_share": to_float(point.get("outside_subject_support_share")),
        "largest_component_share": to_float(point.get("largest_component_share")),
        "secondary_component_mass": to_float(point.get("secondary_component_mass")),
        "support_peak_count": to_int(point.get("support_peak_count")),
        "subject_psnr": to_float(point.get("subject_psnr", point.get("masked_psnr"))),
        "subject_l1": to_float(point.get("subject_l1", point.get("masked_mae"))),
        "masked_psnr": to_float(point.get("masked_psnr")),
        "masked_mae": to_float(point.get("masked_mae")),
        "subject_mask_coverage_ratio": to_float(point.get("subject_mask_coverage_ratio")),
        "support_total_mass": to_float(point.get("support_total_mass")),
        "support_active_mass": to_float(point.get("support_active_mass")),
    }


def _metric_good_directions(prev: dict, cur: dict, eps: float) -> tuple[int, int]:
    better = 0
    worse = 0
    rules = [
        ("subject_support_share", "up"),
        ("outside_subject_support_share", "down"),
        ("largest_component_share", "up"),
        ("secondary_component_mass", "down"),
    ]
    for key, direction in rules:
        pv = to_float(prev.get(key))
        cv = to_float(cur.get(key))
        if not (math.isfinite(pv) and math.isfinite(cv)):
            continue
        delta = cv - pv
        if direction == "up":
            if delta > eps:
                better += 1
            elif delta < -eps:
                worse += 1
        else:
            if delta < -eps:
                better += 1
            elif delta > eps:
                worse += 1
    return better, worse


def _stage_vs_baseline(baseline: dict, cur: dict) -> dict:
    better, worse = _metric_good_directions(baseline, cur, SUPPORT_EPS)
    d_subject_psnr = to_float(cur.get("subject_psnr")) - to_float(baseline.get("subject_psnr"))
    d_subject_l1 = to_float(cur.get("subject_l1")) - to_float(baseline.get("subject_l1"))
    d_ghost = to_float(cur.get("ghost_visual_score")) - to_float(baseline.get("ghost_visual_score"))
    ghost_ok = bool(d_ghost <= GHOST_EPS)
    subj_psnr_ok = bool(d_subject_psnr > SUBJECT_PSNR_EPS)
    subj_l1_ok = bool(d_subject_l1 < -SUBJECT_L1_EPS)
    score = (
        2.0 * float(better)
        - 2.0 * float(worse)
        + (1.0 if ghost_ok else -1.0)
        + (1.0 if subj_psnr_ok else 0.0)
        + (1.0 if subj_l1_ok else 0.0)
        + 0.10 * float(-d_ghost)
        + 0.05 * float(d_subject_psnr)
        + 10.0 * float(-d_subject_l1)
    )
    return {
        "point_better_count": int(better),
        "point_worse_count": int(worse),
        "delta_subject_psnr": float(d_subject_psnr),
        "delta_subject_l1": float(d_subject_l1),
        "delta_ghost_visual_score": float(d_ghost),
        "ghost_ok_vs_baseline": bool(ghost_ok),
        "subject_psnr_improved": bool(subj_psnr_ok),
        "subject_l1_improved": bool(subj_l1_ok),
        "selection_score": float(score),
    }


def _first_rebound(rows: list[dict]) -> str:
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]
        if not (cur.get("available") and prev.get("available")):
            continue
        _better, worse = _metric_good_directions(prev, cur, SUPPORT_EPS)
        if to_float(cur.get("ghost_visual_score")) > to_float(prev.get("ghost_visual_score")) + GHOST_EPS:
            return step_tag(cur["step"])
        if to_float(cur.get("subject_psnr")) < to_float(prev.get("subject_psnr")) - SUBJECT_PSNR_EPS:
            return step_tag(cur["step"])
        if to_float(cur.get("subject_l1")) > to_float(prev.get("subject_l1")) + SUBJECT_L1_EPS:
            return step_tag(cur["step"])
        if worse >= 2:
            return step_tag(cur["step"])
    return "none_through_step0024"


def _approx_convergence(rows: list[dict]) -> str:
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]
        if not (cur.get("available") and prev.get("available")):
            continue
        d_psnr = abs(to_float(cur.get("subject_psnr")) - to_float(prev.get("subject_psnr")))
        d_l1 = abs(to_float(cur.get("subject_l1")) - to_float(prev.get("subject_l1")))
        d_ghost = abs(to_float(cur.get("ghost_visual_score")) - to_float(prev.get("ghost_visual_score")))
        better, worse = _metric_good_directions(prev, cur, SUPPORT_EPS)
        if d_psnr <= SUBJECT_PSNR_EPS and d_l1 <= SUBJECT_L1_EPS and d_ghost <= GHOST_EPS and worse == 0 and better <= 1:
            return step_tag(cur["step"])
        if to_float(cur.get("ghost_visual_score")) > to_float(prev.get("ghost_visual_score")) + GHOST_EPS:
            return step_tag(prev["step"])
    return "none_through_step0024"


def _compose_grid(image_paths: list[str], labels: list[str], out_path: Path, cols: int = 4) -> None:
    images = []
    for path in image_paths:
        with Image.open(path).convert("RGB") as img:
            images.append(img.copy())
    if not images:
        return
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    cell_h = height + 28
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = Image.new("RGB", (cols * width, rows * cell_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * width
        y = row * cell_h
        canvas.paste(img.resize((width, height)), (x, y))
        draw.text((x + 6, y + height + 6), labels[idx], fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


def _best_short_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            to_float(row.get("vs_baseline", {}).get("selection_score")),
            to_float(row.get("subject_psnr")),
            -to_float(row.get("ghost_visual_score")),
            -to_float(row.get("subject_l1")),
        ),
    )


def _resolve_winner_alpha(manifest: dict, alpha_summaries: list[dict]) -> int | None:
    winner_alpha = manifest.get("winner_alpha")
    if winner_alpha is not None:
        return int(winner_alpha)
    best_alpha_summary = max(
        alpha_summaries,
        key=lambda item: (
            to_float(item.get("best_short_score")),
            to_float(item.get("best_short_subject_psnr")),
            -to_float(item.get("best_short_subject_l1")),
        ),
        default=None,
    )
    if best_alpha_summary is None:
        return None
    return int(best_alpha_summary["alpha"])


def _load_profile_summary(
    out_root: Path,
    native_root: Path,
    profile: str,
    alphas: list[int],
    short_steps: list[int],
    extend_steps: list[int],
) -> dict:
    manifest = load_json(manifest_path(out_root, profile)) if manifest_path(out_root, profile).is_file() else {}
    task_index = _load_task_index(Path(out_root) / "task_state_latest.json")
    baseline_dir = native_root / profile / "step0000" / "compare"
    baseline = _load_stage(baseline_dir) if _stage_complete(baseline_dir) else _blank_stage(baseline_dir)
    baseline["step"] = 0
    baseline["alpha"] = 0

    point_grid_paths: list[str] = []
    point_grid_labels: list[str] = []
    ghost_grid_paths: list[str] = []
    ghost_grid_labels: list[str] = []
    csv_rows: list[dict] = []
    winner_alpha_hint = manifest.get("winner_alpha")

    alpha_summaries: list[dict] = []
    for alpha in alphas:
        all_steps = list(short_steps)
        if winner_alpha_hint is not None and int(alpha) == int(winner_alpha_hint):
            for step in extend_steps:
                if int(step) not in set(all_steps):
                    all_steps.append(int(step))
        step_rows: list[dict] = []
        for step in all_steps:
            stage_dir = alpha_step_compare_dir(out_root, profile, alpha, step)
            is_winner_extend = bool(winner_alpha_hint is not None and int(alpha) == int(winner_alpha_hint) and int(step) in set(extend_steps))
            row = (
                _load_stage(stage_dir)
                if _stage_complete(stage_dir)
                else _task_stage_blank(
                    stage_dir,
                    task_index,
                    _stage_task_names(profile, alpha=int(alpha), step=int(step), winner=is_winner_extend),
                )
            )
            row["step"] = int(step)
            row["alpha"] = int(alpha)
            row["fg_supervision_boost"] = float(alpha_to_fg_boost(alpha))
            if baseline.get("available") and row.get("available"):
                row["vs_baseline"] = _stage_vs_baseline(baseline, row)
                row["weight_mad_vs_step0000"] = _image_mad(Path(baseline["weight_native"]), Path(row["weight_native"]))
                row["pred_mad_vs_step0000"] = _image_mad(Path(baseline["pred_native"]), Path(row["pred_native"]))
            else:
                row["vs_baseline"] = {}
                row["weight_mad_vs_step0000"] = float("nan")
                row["pred_mad_vs_step0000"] = float("nan")
            if row.get("available"):
                point_grid_paths.append(row["point_triplet"])
                point_grid_labels.append(f"{profile} a{alpha} s{step:04d}")
                ghost_grid_paths.append(row["ghost_triplet"])
                ghost_grid_labels.append(f"{profile} a{alpha} s{step:04d}")
            csv_rows.append(
                {
                    "profile": profile,
                    "alpha": int(alpha),
                    "fg_supervision_boost": float(alpha_to_fg_boost(alpha)),
                    "step": int(step),
                    "available": bool(row.get("available")),
                    "ghost_visual_score": to_float(row.get("ghost_visual_score")),
                    "subject_psnr": to_float(row.get("subject_psnr")),
                    "subject_l1": to_float(row.get("subject_l1")),
                    "subject_support_share": to_float(row.get("subject_support_share")),
                    "outside_subject_support_share": to_float(row.get("outside_subject_support_share")),
                    "largest_component_share": to_float(row.get("largest_component_share")),
                    "secondary_component_mass": to_float(row.get("secondary_component_mass")),
                    "support_peak_count": to_int(row.get("support_peak_count"), default=-1),
                    "selection_score": to_float(row.get("vs_baseline", {}).get("selection_score")),
                    "stage_status": str(row.get("stage_status", "")),
                    "stage_error": str(row.get("stage_error", "")),
                }
            )
            step_rows.append(row)

        short_rows = [row for row in step_rows if int(row["step"]) in set(short_steps) and row.get("available")]
        best_short = _best_short_row(short_rows)
        alpha_summaries.append(
            {
                "alpha": int(alpha),
                "fg_supervision_boost": float(alpha_to_fg_boost(alpha)),
                "best_short_step": step_tag(best_short["step"]) if best_short else "",
                "best_short_score": to_float(best_short.get("vs_baseline", {}).get("selection_score")) if best_short else float("nan"),
                "best_short_subject_psnr": to_float(best_short.get("subject_psnr")) if best_short else float("nan"),
                "best_short_subject_l1": to_float(best_short.get("subject_l1")) if best_short else float("nan"),
                "first_rebound_step": _first_rebound(short_rows),
                "steps": step_rows,
            }
        )

    winner_alpha = _resolve_winner_alpha(manifest, alpha_summaries)
    winner_summary = next((row for row in alpha_summaries if int(row["alpha"]) == int(winner_alpha or -1)), None)
    winner_rows = [row for row in (winner_summary or {}).get("steps", []) if row.get("available")]
    approx_convergence = _approx_convergence(winner_rows) if winner_rows else "unknown"
    first_rebound = _first_rebound(winner_rows) if winner_rows else "unknown"
    failed_stage_rows = [
        {
            "alpha": int(alpha_summary["alpha"]),
            "step": int(row["step"]),
            "stage_status": str(row.get("stage_status", "")),
            "stage_error": str(row.get("stage_error", "")),
        }
        for alpha_summary in alpha_summaries
        for row in alpha_summary["steps"]
        if not row.get("available") and str(row.get("stage_status", "")) in {"failed", "blocked"}
    ]

    if baseline.get("available"):
        point_grid_paths.insert(0, baseline["point_triplet"])
        point_grid_labels.insert(0, f"{profile} native step0000")
        ghost_grid_paths.insert(0, baseline["ghost_triplet"])
        ghost_grid_labels.insert(0, f"{profile} native step0000")

    point_state = "MIXED"
    subject_state = "MIXED"
    ghost_state = "MIXED"
    if baseline.get("available") and winner_rows:
        final = winner_rows[-1]
        better0, worse0 = _metric_good_directions(baseline, final, SUPPORT_EPS)
        if better0 >= 2 and to_int(final.get("support_peak_count")) <= to_int(baseline.get("support_peak_count")):
            point_state = "YES"
        elif worse0 >= 2:
            point_state = "NO"

        d_psnr = to_float(final.get("subject_psnr")) - to_float(baseline.get("subject_psnr"))
        d_l1 = to_float(final.get("subject_l1")) - to_float(baseline.get("subject_l1"))
        if d_psnr > SUBJECT_PSNR_EPS and d_l1 < -SUBJECT_L1_EPS:
            subject_state = "YES"
        elif d_psnr < -SUBJECT_PSNR_EPS or d_l1 > SUBJECT_L1_EPS:
            subject_state = "NO"

        monotonic_ghost = all(
            to_float(winner_rows[idx]["ghost_visual_score"]) <= to_float(winner_rows[idx - 1]["ghost_visual_score"]) + GHOST_EPS
            for idx in range(1, len(winner_rows))
        )
        if monotonic_ghost:
            ghost_state = "YES"
        elif first_rebound != "none_through_step0024":
            ghost_state = "NO"

    summary = {
        "profile": profile,
        "profile_meta": profile_metadata(profile, DEFAULT_TGT_CAMERA),
        "baseline_native": baseline,
        "winner_alpha": int(winner_alpha) if winner_alpha is not None else None,
        "winner_fg_supervision_boost": float(alpha_to_fg_boost(winner_alpha)) if winner_alpha is not None else None,
        "approx_convergence_step": str(approx_convergence),
        "first_rebound_step": str(first_rebound),
        "POINT_SUPPORT_SINGLE_SUBJECT": point_state,
        "SUBJECT_RECON_IMPROVES": subject_state,
        "GHOST_CONTINUES_DOWN": ghost_state,
        "failed_stage_count": int(len(failed_stage_rows)),
        "failed_stage_rows": failed_stage_rows,
        "alpha_runs": alpha_summaries,
        "table_rows": csv_rows,
        "_point_grid_paths": point_grid_paths,
        "_point_grid_labels": point_grid_labels,
        "_ghost_grid_paths": ghost_grid_paths,
        "_ghost_grid_labels": ghost_grid_labels,
    }
    return summary


def _reference_payload(native_root: Path) -> dict:
    point_dir = native_root / POINT_SUPPORT_REFERENCE["profile"] / step_tag(POINT_SUPPORT_REFERENCE["step"]) / "compare"
    ghost_dir = native_root / GHOST_REFERENCE["profile"] / step_tag(GHOST_REFERENCE["step"]) / "compare"
    return {
        "point_support_reference": {
            **POINT_SUPPORT_REFERENCE,
            "compare_dir": str(point_dir),
            "summary_source": str(native_root / POINT_SUPPORT_REFERENCE["profile"] / "trend" / "summary.json"),
        },
        "ghost_reference": {
            **GHOST_REFERENCE,
            "compare_dir": str(ghost_dir),
            "summary_source": str(native_root / GHOST_REFERENCE["profile"] / "trend" / "summary.json"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser("orig_vggt_mask_boost_probe_summary")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--native_stepcurve_root", default=str(DEFAULT_NATIVE_STEPCURVE_ROOT))
    ap.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    ap.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    ap.add_argument("--short_steps", default=",".join(str(x) for x in DEFAULT_SHORT_STEPS))
    ap.add_argument("--extend_steps", default=",".join(str(x) for x in DEFAULT_EXTEND_STEPS))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_advisor_md", required=True)
    ap.add_argument("--out_point_grid", required=True)
    ap.add_argument("--out_ghost_grid", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    native_root = Path(args.native_stepcurve_root)
    profiles = parse_profiles(args.profiles)
    alphas = parse_alphas(args.alphas)
    short_steps = parse_steps(args.short_steps, DEFAULT_SHORT_STEPS)
    extend_steps = parse_steps(args.extend_steps, DEFAULT_EXTEND_STEPS)

    payload = {
        "references": _reference_payload(native_root),
        "profiles": [],
    }
    csv_rows: list[dict] = []
    md_lines = [
        "# Native VGGT Mask-Boost Probe Summary",
        "",
        "- point-support first, subject reconstruction second, ghost third.",
        "- `subject_psnr` / `subject_l1` are computed inside the subject mask.",
        "- `weight_native.png` derived metrics are relative normalized shape metrics.",
        "",
        "## Fixed References",
        "",
        f"- point_support_reference=`{payload['references']['point_support_reference']['label']}` dir=`{payload['references']['point_support_reference']['compare_dir']}`",
        f"- ghost_reference=`{payload['references']['ghost_reference']['label']}` dir=`{payload['references']['ghost_reference']['compare_dir']}`",
        "",
        "## Profile Conclusions",
        "",
    ]
    advisor_lines = [
        "# Native VGGT Mask-Boost Advisor Summary",
        "",
        "- This run keeps native VGGT training semantics and only adds subject-interior supervision boost.",
        "- Read point-support first, then subject mask reconstruction, then ghost.",
        "",
    ]
    all_point_paths: list[str] = []
    all_point_labels: list[str] = []
    all_ghost_paths: list[str] = []
    all_ghost_labels: list[str] = []

    for profile in profiles:
        summary = _load_profile_summary(out_root, native_root, profile, alphas, short_steps, extend_steps)
        grid_point_paths = list(summary.pop("_point_grid_paths"))
        grid_point_labels = list(summary.pop("_point_grid_labels"))
        grid_ghost_paths = list(summary.pop("_ghost_grid_paths"))
        grid_ghost_labels = list(summary.pop("_ghost_grid_labels"))
        write_json(profile_summary_json(out_root, profile), summary)

        lines = [
            f"# Native VGGT Mask-Boost Trend: {profile}",
            "",
            "- taxonomy: `weight_native.png` derived metrics are relative normalized shape metrics, not absolute support physics.",
            "- taxonomy: `subject_psnr` / `subject_l1` are mask-inside reconstruction metrics.",
            "",
            f"- winner_alpha = `{summary.get('winner_alpha')}`",
            f"- winner_fg_supervision_boost = `{summary.get('winner_fg_supervision_boost')}`",
            f"- approx_convergence_step = `{summary.get('approx_convergence_step')}`",
            f"- first_rebound_step = `{summary.get('first_rebound_step')}`",
            f"- POINT_SUPPORT_SINGLE_SUBJECT = `{summary.get('POINT_SUPPORT_SINGLE_SUBJECT')}`",
            f"- SUBJECT_RECON_IMPROVES = `{summary.get('SUBJECT_RECON_IMPROVES')}`",
            f"- GHOST_CONTINUES_DOWN = `{summary.get('GHOST_CONTINUES_DOWN')}`",
            f"- failed_stage_count = `{summary.get('failed_stage_count')}`",
            "",
        ]
        for alpha_summary in summary["alpha_runs"]:
            lines.append(f"## {alpha_tag(alpha_summary['alpha'])}")
            lines.append("")
            lines.append(
                f"- fg_boost=`{alpha_summary['fg_supervision_boost']}` "
                f"best_short_step=`{alpha_summary['best_short_step']}` "
                f"best_short_score=`{alpha_summary['best_short_score']:.4f}` "
                f"first_rebound=`{alpha_summary['first_rebound_step']}`"
            )
            for row in alpha_summary["steps"]:
                if row.get("available"):
                    lines.append(
                        f"- `{step_tag(row['step'])}` ghost=`{row['ghost_visual_score']:.6f}` "
                        f"subj_psnr=`{row['subject_psnr']:.6f}` subj_l1=`{row['subject_l1']:.6f}` "
                        f"support_pk=`{row['support_peak_count']}` lc=`{row['largest_component_share']:.6f}` "
                        f"sc=`{row['secondary_component_mass']:.6f}`"
                    )
                else:
                    lines.append(
                        f"- `{step_tag(row['step'])}` unavailable status=`{row['stage_status']}` reason=`{row['stage_error']}`"
                    )
            lines.append("")
        write_text(profile_summary_md(out_root, profile), "\n".join(lines) + "\n")

        payload["profiles"].append(summary)
        md_lines.append(
            f"- `{profile}` winner_alpha=`{summary.get('winner_alpha')}` fg_boost=`{summary.get('winner_fg_supervision_boost')}` "
            f"convergence=`{summary.get('approx_convergence_step')}` rebound=`{summary.get('first_rebound_step')}` "
            f"point_support=`{summary.get('POINT_SUPPORT_SINGLE_SUBJECT')}` subject_recon=`{summary.get('SUBJECT_RECON_IMPROVES')}` "
            f"ghost=`{summary.get('GHOST_CONTINUES_DOWN')}` failed_stages=`{summary.get('failed_stage_count')}`"
        )
        advisor_lines.append(
            f"- {profile}: winner alpha `{summary.get('winner_alpha')}`, boost `{summary.get('winner_fg_supervision_boost')}`, "
            f"approx convergence `{summary.get('approx_convergence_step')}`, first rebound `{summary.get('first_rebound_step')}`."
        )
        csv_rows.extend(summary["table_rows"])

        all_point_paths.extend(grid_point_paths)
        all_point_labels.extend(grid_point_labels)
        all_ghost_paths.extend(grid_ghost_paths)
        all_ghost_labels.extend(grid_ghost_labels)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_advisor_md = Path(args.out_advisor_md)
    write_json(out_json, payload)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "alpha",
                "fg_supervision_boost",
                "step",
                "available",
                "ghost_visual_score",
                "subject_psnr",
                "subject_l1",
                "subject_support_share",
                "outside_subject_support_share",
                "largest_component_share",
                "secondary_component_mass",
                "support_peak_count",
                "selection_score",
                "stage_status",
                "stage_error",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    write_text(out_md, "\n".join(md_lines) + "\n")
    advisor_lines.append("- NO_NONSTOPPED_APPS")
    write_text(out_advisor_md, "\n".join(advisor_lines) + "\n")
    _compose_grid(all_point_paths, all_point_labels, Path(args.out_point_grid), cols=4)
    _compose_grid(all_ghost_paths, all_ghost_labels, Path(args.out_ghost_grid), cols=4)


if __name__ == "__main__":
    main()
