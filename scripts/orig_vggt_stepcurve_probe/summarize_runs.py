from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_stepcurve_probe.common import (  # noqa: E402
    DEFAULT_ALL_STEPS,
    DEFAULT_GHOST_EPS,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORTS_DIR,
    DEFAULT_STEP_HORIZONS,
    DEFAULT_SUPPORT_EPS,
    DEFAULT_TRAIN_PROFILES,
    REFERENCE_PROFILE,
    compare_dir,
    fmt_num,
    load_json,
    load_jsonl,
    normalize_path,
    parse_profiles,
    parse_step_horizons,
    profile_dirs,
    profile_metadata,
    profile_tag,
    step_tag,
    to_float,
    to_int,
    train_dir,
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


def _load_exact_ghost_row(stage_dir: Path) -> dict:
    rows_path = stage_dir / "ghost_score_rows.csv"
    if not rows_path.is_file():
        raise RuntimeError(f"ghost_score_rows.csv missing: {rows_path}")
    rows: list[dict] = []
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    target = normalize_path((stage_dir / "cat_fg_mask_pred_tgt_step000000.png").resolve())
    hits = [row for row in rows if normalize_path(row.get("path", "")) == target]
    if len(hits) != 1:
        raise RuntimeError(f"ghost row exact match failed for {target}; matched={len(hits)}")
    return hits[0]


def _load_stage(stage_dir: Path) -> dict:
    report = load_json(stage_dir / "report.json")
    point = load_json(stage_dir / "point_support_metrics.json")
    ghost = _load_exact_ghost_row(stage_dir)
    render = report.get("render", {})
    native = report.get("metrics", {}).get("native", {})
    return {
        "stage_dir": str(stage_dir),
        "weight_native": str(stage_dir / "weight_native.png"),
        "pred_native": str(stage_dir / "pred_native.png"),
        "ghost_triplet": str(stage_dir / "cat_fg_mask_pred_tgt_step000000.png"),
        "point_triplet": str(stage_dir / "cat_weight_pred_tgt_subject_bbox.png"),
        "available": True,
        "stage_status": "completed",
        "stage_error": "",
        "coverage_ratio": to_float(render.get("coverage_ratio")),
        "mean_conf": to_float(render.get("mean_conf")),
        "valid_contrib": to_int(render.get("valid_contrib")),
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


def _blank_stage(stage_dir: Path, *, stage_status: str = "missing", stage_error: str = "") -> dict:
    nan = float("nan")
    return {
        "stage_dir": str(stage_dir),
        "weight_native": "",
        "pred_native": "",
        "ghost_triplet": "",
        "point_triplet": "",
        "available": False,
        "stage_status": str(stage_status),
        "stage_error": str(stage_error or ""),
        "coverage_ratio": nan,
        "mean_conf": nan,
        "valid_contrib": None,
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


def _stage_complete(stage_dir: Path) -> bool:
    required = [
        stage_dir / "report.json",
        stage_dir / "point_support_metrics.json",
        stage_dir / "ghost_score_rows.csv",
        stage_dir / "cat_fg_mask_pred_tgt_step000000.png",
        stage_dir / "weight_native.png",
        stage_dir / "cat_weight_pred_tgt_subject_bbox.png",
    ]
    return all(path.is_file() for path in required)


def _load_root_task_state(out_root: Path) -> dict:
    path = out_root / "task_state_latest.json"
    if path.is_file():
        return load_json(path)
    return {}


def _task_entry(task_state: dict, name: str) -> dict:
    for task in task_state.get("tasks", []):
        if str(task.get("name")) == str(name):
            return dict(task)
    return {}


def _task_error_text(task: dict) -> str:
    errors = task.get("errors", [])
    if isinstance(errors, list) and errors:
        last = errors[-1]
        if isinstance(last, dict) and str(last.get("message", "")).strip():
            return str(last.get("message", "")).strip()
        if str(last).strip():
            return str(last).strip()
    return str(task.get("message", "")).strip()


def _task_status_and_error(task_state: dict, names: list[str]) -> tuple[str, str]:
    tasks = [_task_entry(task_state, name) for name in names]
    for task in tasks:
        status = str(task.get("status", "")).strip()
        if status == "failed":
            return status, _task_error_text(task)
    for task in tasks:
        status = str(task.get("status", "")).strip()
        if status == "blocked":
            return status, _task_error_text(task)
    for task in reversed(tasks):
        status = str(task.get("status", "")).strip()
        if status:
            return status, _task_error_text(task)
    return "", ""


def _compare_task_names(tag: str, step: int) -> list[str]:
    if int(step) == 0:
        return [
            f"profile_{tag}_step0000_measure_support",
            f"profile_{tag}_step0000_score_ghost",
            f"profile_{tag}_step0000_sync_compare",
            f"profile_{tag}_step0000_compare",
        ]
    stag = step_tag(step)
    return [
        f"profile_{tag}_{stag}_measure_support",
        f"profile_{tag}_{stag}_score_ghost",
        f"profile_{tag}_{stag}_sync_compare",
        f"profile_{tag}_{stag}_compare",
    ]


def _train_rollup(rows: list[dict]) -> dict:
    run_meta = next((row for row in rows if row.get("event") == "run_meta"), {})
    step_eval_rows = [row for row in rows if row.get("event") == "step_eval"]
    epoch_end_rows = [row for row in rows if row.get("event") == "epoch_end"]
    final_step = max((to_int(row.get("step")) for row in step_eval_rows), default=0)
    if final_step <= 0:
        final_step = sum(to_int(row.get("steps")) for row in epoch_end_rows)
    return {
        "run_meta": run_meta,
        "step_eval_count": int(len(step_eval_rows)),
        "epoch_count": int(len(epoch_end_rows)),
        "epoch_end_steps_total": int(sum(to_int(row.get("steps")) for row in epoch_end_rows)),
        "epoch_end_steps_last": int(to_int(epoch_end_rows[-1].get("steps"))) if epoch_end_rows else 0,
        "last_step_eval": int(max((to_int(row.get("step")) for row in step_eval_rows), default=0)),
        "final_step_from_metrics": int(final_step),
    }


def _load_train(train_root: Path) -> dict:
    metrics_path = train_root / "logs" / "finetune_vggt_metrics.jsonl"
    summary_path = train_root / "logs" / "finetune_vggt_summary.json"
    prefix_path = train_root / "train_prefix_trace.json"
    out = {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "prefix_path": str(prefix_path),
        "train_verified": False,
        "final_step": 0,
        "out_last": "",
        "strict_deterministic": False,
        "epoch_end_steps": 0,
        "epoch_end_steps_last": 0,
        "step_eval_count": 0,
        "epoch_count": 0,
        "run_meta_max_steps_per_epoch": 0,
    }
    if not metrics_path.is_file() or not summary_path.is_file():
        return out
    rows = load_jsonl(metrics_path)
    rollup = _train_rollup(rows)
    run_meta = rollup["run_meta"]
    summary = load_json(summary_path)
    out.update(
        {
            "final_step": to_int(summary.get("final_step") or rollup["final_step_from_metrics"]),
            "out_last": str(summary.get("out_last", "")),
            "strict_deterministic": bool(run_meta.get("strict_deterministic", False)),
            "epoch_end_steps": int(summary.get("epoch_end_steps_total", rollup["epoch_end_steps_total"])),
            "epoch_end_steps_last": int(summary.get("epoch_end_steps_last", rollup["epoch_end_steps_last"])),
            "step_eval_count": int(summary.get("step_eval_count", rollup["step_eval_count"])),
            "epoch_count": int(summary.get("epoch_count", rollup["epoch_count"])),
            "run_meta_max_steps_per_epoch": to_int(run_meta.get("max_steps_per_epoch")),
        }
    )
    out["train_verified"] = bool(
        out["strict_deterministic"]
        and out["run_meta_max_steps_per_epoch"] == 1
        and out["step_eval_count"] == out["final_step"]
        and out["epoch_end_steps"] == out["final_step"]
        and out["epoch_count"] == out["final_step"]
        and out["final_step"] == int(rollup["last_step_eval"])
        and str(out["out_last"]).strip()
    )
    if prefix_path.is_file():
        out["prefix"] = load_json(prefix_path)
    return out


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


def _first_rebound(step_rows: list[dict], support_eps: float, ghost_eps: float) -> str:
    for idx in range(1, len(step_rows)):
        prev = step_rows[idx - 1]
        cur = step_rows[idx]
        better, worse = _metric_good_directions(prev, cur, support_eps)
        if to_float(cur.get("ghost_visual_score")) > to_float(prev.get("ghost_visual_score")) + ghost_eps:
            return step_tag(cur["step"])
        if worse >= 2:
            return step_tag(cur["step"])
        if (
            to_int(cur.get("support_peak_count")) > to_int(prev.get("support_peak_count"))
            and to_float(cur.get("largest_component_share")) <= to_float(prev.get("largest_component_share")) + support_eps
        ):
            return step_tag(cur["step"])
        _ = better
    return "none_through_step0016"


def _extend_decision(step_rows: list[dict], support_eps: float, ghost_eps: float) -> bool:
    by_step = {int(row["step"]): row for row in step_rows}
    if 8 not in by_step or 16 not in by_step:
        return False
    s8 = by_step[8]
    s16 = by_step[16]
    better, _worse = _metric_good_directions(s8, s16, support_eps)
    if better < 2:
        return False
    if to_float(s16.get("ghost_visual_score")) > to_float(s8.get("ghost_visual_score")) + ghost_eps:
        return False
    if to_int(s16.get("support_peak_count")) <= to_int(s8.get("support_peak_count")):
        return True
    return bool(
        to_float(s16.get("largest_component_share")) > to_float(s8.get("largest_component_share")) + support_eps
        and to_float(s16.get("secondary_component_mass")) < to_float(s8.get("secondary_component_mass")) - support_eps
    )


def _prefix_summary(profile_root: Path, step_horizons: list[int]) -> tuple[bool, str]:
    longest = max(step_horizons)
    longest_path = train_dir(profile_root.parent, profile_root.name, longest) / "train_prefix_trace.json"
    if not longest_path.is_file():
        return False, f"{step_tag(longest)}_missing"
    longest_payload = load_json(longest_path)
    longest_steps = longest_payload.get("steps", [])
    for step in step_horizons:
        path = train_dir(profile_root.parent, profile_root.name, step) / "train_prefix_trace.json"
        if not path.is_file():
            return False, f"{step_tag(step)}_missing"
        payload = load_json(path)
        steps = payload.get("steps", [])
        if steps != longest_steps[: int(step)]:
            return False, step_tag(step)
    return True, ""


def _compose_grid(image_paths: list[str], labels: list[str], out_path: Path, cols: int = 6) -> None:
    images = []
    for path in image_paths:
        with Image.open(path).convert("RGB") as img:
            images.append(img.copy())
    if not images:
        return
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = Image.new("RGB", (cols * w, rows * (h + 20)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = row * (h + 20)
        canvas.paste(img.resize((w, h)), (x, y))
        draw.text((x + 6, y + 4), labels[idx], fill=(255, 255, 255))
    canvas.save(str(out_path))


def _write_profile_summary(
    out_root: Path,
    profile: str,
    step_rows: list[dict],
    support_eps: float,
    ghost_eps: float,
) -> dict:
    pdirs = profile_dirs(out_root, profile)
    meta = profile_metadata(profile, "Camera_B5")
    baseline = next(row for row in step_rows if int(row["step"]) == 0)
    available_rows = [row for row in step_rows if bool(row.get("available", False))]
    failed_rows = [row for row in step_rows if int(row["step"]) > 0 and not bool(row.get("available", False))]
    first_failed_step = step_tag(failed_rows[0]["step"]) if failed_rows else ""
    first_failed_reason = str(failed_rows[0].get("stage_error", "")).strip() if failed_rows else ""

    metric_rebound = _first_rebound(available_rows, support_eps, ghost_eps) if len(available_rows) >= 2 else "none_through_step0016"
    first_rebound = first_failed_step or metric_rebound
    extend = False if first_failed_step else _extend_decision(available_rows, support_eps, ghost_eps)

    prefix_steps = [
        int(row["step"])
        for row in step_rows
        if int(row["step"]) > 0 and Path(str(row.get("train", {}).get("prefix_path", ""))).is_file()
    ]
    if prefix_steps:
        prefix_ok, prefix_mismatch = _prefix_summary(pdirs["root"], prefix_steps)
    else:
        prefix_ok, prefix_mismatch = False, f"{step_tag(DEFAULT_STEP_HORIZONS[0])}_missing"

    for row in step_rows:
        row["delta_vs_prev"] = {}
        row["weight_mad_vs_step0000"] = float("nan")
        row["pred_mad_vs_step0000"] = float("nan")
    if bool(baseline.get("available", False)):
        baseline["weight_mad_vs_step0000"] = 0.0
        baseline["pred_mad_vs_step0000"] = 0.0
    for row in step_rows:
        if row is baseline:
            continue
        if bool(row.get("available", False)) and bool(baseline.get("available", False)):
            row["weight_mad_vs_step0000"] = _image_mad(Path(row["weight_native"]), Path(baseline["weight_native"]))
            row["pred_mad_vs_step0000"] = _image_mad(Path(row["pred_native"]), Path(baseline["pred_native"]))
    for idx in range(1, len(step_rows)):
        prev = step_rows[idx - 1]
        cur = step_rows[idx]
        if bool(prev.get("available", False)) and bool(cur.get("available", False)):
            cur["delta_vs_prev"] = {
                "ghost_visual_score": to_float(cur["ghost_visual_score"]) - to_float(prev["ghost_visual_score"]),
                "native_psnr": to_float(cur["native_psnr"]) - to_float(prev["native_psnr"]),
                "subject_support_share": to_float(cur["subject_support_share"]) - to_float(prev["subject_support_share"]),
                "outside_subject_support_share": to_float(cur["outside_subject_support_share"]) - to_float(prev["outside_subject_support_share"]),
                "largest_component_share": to_float(cur["largest_component_share"]) - to_float(prev["largest_component_share"]),
                "secondary_component_mass": to_float(cur["secondary_component_mass"]) - to_float(prev["secondary_component_mass"]),
            }
        else:
            cur["delta_vs_prev"] = {
                "ghost_visual_score": float("nan"),
                "native_psnr": float("nan"),
                "subject_support_share": float("nan"),
                "outside_subject_support_share": float("nan"),
                "largest_component_share": float("nan"),
                "secondary_component_mass": float("nan"),
            }

    final = available_rows[-1] if available_rows else baseline
    if first_failed_step:
        point_support_state = "NO"
        ghost_state = "NO"
    elif len(available_rows) < 2:
        point_support_state = "MIXED"
        ghost_state = "MIXED"
    else:
        better0, worse0 = _metric_good_directions(baseline, final, support_eps)
        point_support_state = (
            "YES"
            if better0 >= 2 and to_int(final["support_peak_count"]) <= to_int(baseline["support_peak_count"])
            else ("NO" if worse0 >= 2 else "MIXED")
        )
        monotonic_ghost = all(
            to_float(available_rows[idx]["ghost_visual_score"]) <= to_float(available_rows[idx - 1]["ghost_visual_score"]) + ghost_eps
            for idx in range(1, len(available_rows))
        )
        ghost_state = "YES" if monotonic_ghost else ("NO" if first_rebound != "none_through_step0016" else "MIXED")

    summary = {
        "profile": profile,
        "profile_meta": meta,
        "steps": step_rows,
        "completed_compare_steps": [step_tag(int(row["step"])) for row in step_rows if bool(row.get("available", False))],
        "missing_or_failed_steps": [step_tag(int(row["step"])) for row in step_rows if not bool(row.get("available", False))],
        "prefix_consistent_through_step0016": bool(prefix_ok),
        "first_prefix_mismatch_step": str(prefix_mismatch),
        "first_failed_step": str(first_failed_step),
        "first_failed_reason": str(first_failed_reason),
        "first_rebound_step": str(first_rebound),
        "worth_extending_to_32": bool(extend),
        "POINT_SUPPORT_SINGLE_SUBJECT": point_support_state,
        "GHOST_CONTINUES_DOWN": ghost_state,
        "FIRST_REBOUND_STEP": str(first_rebound),
        "EXTEND_DECISION": "YES_EXTEND_TO_32" if extend else "NO_STOP_AT_16",
    }
    write_json(pdirs["trend_summary_json"], summary)

    md_lines = [
        f"# Native VGGT Step-Curve Trend: {profile}",
        "",
        "- taxonomy: `weight_native.png` derived metrics are relative normalized shape metrics, not absolute support physics.",
        "- taxonomy: `cat_fg_mask_pred_tgt_step000000.png` is a ghost triplet, not point-support.",
        "",
        "## Step Curve",
        "",
    ]
    for row in step_rows:
        if bool(row.get("available", False)):
            md_lines.append(
                f"- `{step_tag(row['step'])}` ghost=`{fmt_num(row['ghost_visual_score'])}` ghost_pk=`{row['ghost_peak_count']}` "
                f"support_pk=`{row['support_peak_count']}` subj=`{fmt_num(row['subject_support_share'])}` "
                f"outside=`{fmt_num(row['outside_subject_support_share'])}` lc=`{fmt_num(row['largest_component_share'])}` "
                f"sc=`{fmt_num(row['secondary_component_mass'])}` psnr=`{fmt_num(row['native_psnr'])}`"
            )
        else:
            md_lines.append(
                f"- `{step_tag(row['step'])}` unavailable status=`{row.get('stage_status', '')}` "
                f"reason=`{str(row.get('stage_error', '')).splitlines()[0] if str(row.get('stage_error', '')).strip() else ''}`"
            )
    md_lines.extend(
        [
            "",
            f"- first_failed_step = `{summary['first_failed_step'] or 'none'}`",
            f"- prefix_consistent_through_step0016 = `{summary['prefix_consistent_through_step0016']}`",
            f"- POINT_SUPPORT_SINGLE_SUBJECT = `{summary['POINT_SUPPORT_SINGLE_SUBJECT']}`",
            f"- GHOST_CONTINUES_DOWN = `{summary['GHOST_CONTINUES_DOWN']}`",
            f"- FIRST_REBOUND_STEP = `{summary['FIRST_REBOUND_STEP']}`",
            f"- EXTEND_DECISION = `{summary['EXTEND_DECISION']}`",
        ]
    )
    write_text(pdirs["trend_summary_md"], "\n".join(md_lines) + "\n")
    return summary


def _collect_profile(out_root: Path, profile: str, steps: list[int]) -> list[dict]:
    rows: list[dict] = []
    task_state = _load_root_task_state(out_root)
    tag = profile_tag(profile)
    for step in steps:
        stage_dir = compare_dir(out_root, profile, step)
        stage_status, stage_error = _task_status_and_error(task_state, _compare_task_names(tag, step))
        stage = _load_stage(stage_dir) if _stage_complete(stage_dir) else _blank_stage(stage_dir, stage_status=stage_status or "missing", stage_error=stage_error)
        row = {"step": int(step), **stage}
        if int(step) > 0:
            train_root = train_dir(out_root, profile, step)
            train_task = _task_entry(task_state, f"profile_{tag}_{step_tag(step)}_train")
            row["train"] = _load_train(train_root)
            row["train_task_status"] = str(train_task.get("status", "")).strip()
            row["train_task_error"] = _task_error_text(train_task)
        rows.append(row)
    return rows


def _load_reference(out_root: Path) -> dict:
    ref_dir = out_root / "reference_23cam_fullset_one_step"
    summary_path = ref_dir / "summary.json"
    if summary_path.is_file():
        return load_json(summary_path)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser("summarize_runs")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--profiles", default=",".join(DEFAULT_TRAIN_PROFILES))
    ap.add_argument("--step_horizons", default=",".join(str(x) for x in DEFAULT_STEP_HORIZONS))
    ap.add_argument("--support_eps", type=float, default=float(DEFAULT_SUPPORT_EPS))
    ap.add_argument("--ghost_eps", type=float, default=float(DEFAULT_GHOST_EPS))
    ap.add_argument("--out_json", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_probe_summary_latest.json"))
    ap.add_argument("--out_csv", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_probe_summary_latest.csv"))
    ap.add_argument("--out_md", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_probe_summary_latest.md"))
    ap.add_argument("--out_advisor_md", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_probe_advisor_latest.md"))
    ap.add_argument("--out_point_grid", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_point_support_grid_latest.png"))
    ap.add_argument("--out_ghost_grid", default=str(DEFAULT_REPORTS_DIR / "orig_vggt_stepcurve_ghost_grid_latest.png"))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    profiles = parse_profiles(args.profiles)
    step_horizons = parse_step_horizons(args.step_horizons)
    steps = [0] + step_horizons

    summaries: list[dict] = []
    csv_rows: list[dict] = []
    point_grid_paths: list[str] = []
    point_grid_labels: list[str] = []
    ghost_grid_paths: list[str] = []
    ghost_grid_labels: list[str] = []

    for profile in profiles:
        step_rows = _collect_profile(out_root, profile, steps)
        summary = _write_profile_summary(out_root, profile, step_rows, float(args.support_eps), float(args.ghost_eps))
        summaries.append(summary)
        for row in summary["steps"]:
            csv_rows.append(
                {
                    "profile": profile,
                    "step": int(row["step"]),
                    "available": bool(row.get("available", False)),
                    "stage_status": str(row.get("stage_status", "")),
                    "ghost_visual_score": row["ghost_visual_score"],
                    "ghost_peak_count": row["ghost_peak_count"],
                    "native_psnr": row["native_psnr"],
                    "subject_support_share": row["subject_support_share"],
                    "outside_subject_support_share": row["outside_subject_support_share"],
                    "largest_component_share": row["largest_component_share"],
                    "secondary_component_mass": row["secondary_component_mass"],
                    "support_peak_count": row["support_peak_count"],
                    "weight_mad_vs_step0000": row.get("weight_mad_vs_step0000", float("nan")),
                    "pred_mad_vs_step0000": row.get("pred_mad_vs_step0000", float("nan")),
                    "stage_error": str(row.get("stage_error", "")).splitlines()[0] if str(row.get("stage_error", "")).strip() else "",
                }
            )
            point_triplet = str(row.get("point_triplet", ""))
            ghost_triplet = str(row.get("ghost_triplet", ""))
            if point_triplet and Path(point_triplet).is_file():
                point_grid_paths.append(point_triplet)
                point_grid_labels.append(f"{profile} {step_tag(row['step'])}")
            if ghost_triplet and Path(ghost_triplet).is_file():
                ghost_grid_paths.append(ghost_triplet)
                ghost_grid_labels.append(f"{profile} {step_tag(row['step'])}")

    reference = _load_reference(out_root)
    payload = {
        "profiles": summaries,
        "reference_23cam_fullset_one_step": reference,
    }
    write_json(args.out_json, payload)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    lines = [
        "# Native VGGT Step-Curve Probe Summary",
        "",
        "- point-support first, ghost second, metrics last.",
        "- `weight_native.png` derived metrics are relative normalized shape metrics.",
        "",
        "## Profile Conclusions",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"- `{summary['profile']}` point_support=`{summary['POINT_SUPPORT_SINGLE_SUBJECT']}` "
                f"ghost=`{summary['GHOST_CONTINUES_DOWN']}` rebound=`{summary['FIRST_REBOUND_STEP']}` "
                f"extend32=`{summary['EXTEND_DECISION']}` prefix_ok=`{summary['prefix_consistent_through_step0016']}` "
                f"first_failed=`{summary.get('first_failed_step', '') or 'none'}`",
            ]
        )
    if reference:
        lines.extend(
            [
                "",
                "## Reference 23src",
                "",
                f"- reference_path=`{reference.get('source_root', '')}` note=`{reference.get('note', '')}`",
            ]
        )
    write_text(args.out_md, "\n".join(lines) + "\n")

    advisor_lines = [
        "# 导师口述版",
        "",
        "先看 point-support，再看 ghost，最后看数值。",
        "",
    ]
    for summary in summaries:
        advisor_lines.append(
            f"- {summary['profile']}: 单主体 `{summary['POINT_SUPPORT_SINGLE_SUBJECT']}`，ghost `{summary['GHOST_CONTINUES_DOWN']}`，首次回弹 `{summary['FIRST_REBOUND_STEP']}`，首个失败步 `{summary.get('first_failed_step', '') or 'none'}`，是否继续到 32 `{summary['EXTEND_DECISION']}`。"
        )
    if reference:
        advisor_lines.append("- 23cam_fullset 仍只保留 one-step 作为 reference，本轮不继续深训。")
    advisor_lines.append("- NO_NONSTOPPED_APPS")
    write_text(args.out_advisor_md, "\n".join(advisor_lines) + "\n")
    advisor_lines = [
        "# 导师口述版",
        "",
        "先看 point-support，再看 ghost，最后看数值。",
        "",
    ]
    for summary in summaries:
        advisor_lines.append(
            f"- {summary['profile']}: 单主体 `{summary['POINT_SUPPORT_SINGLE_SUBJECT']}`，ghost `{summary['GHOST_CONTINUES_DOWN']}`，"
            f"首次回弹 `{summary['FIRST_REBOUND_STEP']}`，首个失败步 `{summary.get('first_failed_step', '') or 'none'}`，"
            f"是否继续到 32 `{summary['EXTEND_DECISION']}`。"
        )
    if reference:
        advisor_lines.append("- 23cam_fullset 仍只保留 one-step 作为 reference，本轮不继续深训。")
    advisor_lines.append("- NO_NONSTOPPED_APPS")
    write_text(args.out_advisor_md, "\n".join(advisor_lines) + "\n")

    _compose_grid(point_grid_paths, point_grid_labels, Path(args.out_point_grid))
    _compose_grid(ghost_grid_paths, ghost_grid_labels, Path(args.out_ghost_grid))
    print(f"[stepcurve-summary] wrote {args.out_json}")


if __name__ == "__main__":
    main()
