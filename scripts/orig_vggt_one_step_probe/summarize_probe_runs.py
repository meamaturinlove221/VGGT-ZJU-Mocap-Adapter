from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_one_step_probe.common import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORTS_DIR,
    fmt_num,
    parse_profiles,
    profile_dirs,
    profile_metadata,
    write_json,
    write_text,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _load_ghost_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = _load_json(path)
    summary = payload.get("summary", [])
    if summary:
        return dict(summary[0])
    return {}


def _image_mad(path_a: Path, path_b: Path, *, mode: str = "RGB") -> float:
    if (not path_a.is_file()) or (not path_b.is_file()):
        return float("nan")
    with Image.open(str(path_a)) as img_a:
        arr_a = np.asarray(img_a.convert(mode), dtype=np.float32) / 255.0
    with Image.open(str(path_b)) as img_b:
        arr_b = np.asarray(img_b.convert(mode), dtype=np.float32) / 255.0
    if arr_a.shape != arr_b.shape:
        return float("nan")
    return float(np.abs(arr_a - arr_b).mean())


def _collect_train_metrics(train_dir: Path) -> dict:
    metrics_path = train_dir / "logs" / "finetune_vggt_metrics.jsonl"
    summary_path = train_dir / "logs" / "finetune_vggt_summary.json"
    out: dict = {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "train_verified": False,
        "run_meta_max_steps_per_epoch": 0,
        "step_eval_count": 0,
        "epoch_end_steps": 0,
        "step_update_ratio": float("nan"),
    }
    if not metrics_path.is_file():
        return out
    rows = _load_jsonl(metrics_path)
    run_meta = next((row for row in rows if row.get("event") == "run_meta"), {})
    step_evals = [row for row in rows if row.get("event") == "step_eval"]
    epoch_end = next((row for row in reversed(rows) if row.get("event") == "epoch_end"), {})
    step_update_ratio = float("nan")
    if step_evals:
        step_update_ratio = _to_float(step_evals[-1].get("step_update_ratio"))
    if not math.isfinite(step_update_ratio):
        step_update_ratio = _to_float(epoch_end.get("mean_step_update_ratio"))
    out.update(
        {
            "run_meta_max_steps_per_epoch": _to_int(run_meta.get("max_steps_per_epoch")),
            "step_eval_count": len(step_evals),
            "epoch_end_steps": _to_int(epoch_end.get("steps")),
            "step_update_ratio": step_update_ratio,
        }
    )
    out["train_verified"] = bool(
        out["run_meta_max_steps_per_epoch"] == 1
        and out["step_eval_count"] == 1
        and out["epoch_end_steps"] == 1
        and math.isfinite(out["step_update_ratio"])
    )
    if summary_path.is_file():
        out["train_summary"] = _load_json(summary_path)
    return out


def _collect_stage(stage_dir: Path) -> dict:
    report_path = stage_dir / "report.json"
    ghost_path = stage_dir / "ghost_score.json"
    report = _load_json(report_path) if report_path.is_file() else {}
    ghost = _load_ghost_summary(ghost_path)
    meta = report.get("meta", {})
    render = report.get("render", {})
    metrics_native = report.get("metrics", {}).get("native", {})
    return {
        "stage_dir": str(stage_dir),
        "report_json": str(report_path),
        "ghost_json": str(ghost_path),
        "weight_native": str(stage_dir / "weight_native.png"),
        "pred_native": str(stage_dir / "pred_native.png"),
        "cat_weight_pred_tgt": str(stage_dir / "cat_weight_pred_tgt.png"),
        "ghost_triplet": str(stage_dir / "cat_fg_mask_pred_tgt_step000000.png"),
        "coverage_ratio": _to_float(render.get("coverage_ratio")),
        "mean_conf": _to_float(render.get("mean_conf")),
        "valid_contrib": _to_int(render.get("valid_contrib")),
        "native_psnr": _to_float(metrics_native.get("psnr")),
        "native_ssim": _to_float(metrics_native.get("ssim")),
        "native_mae": _to_float(metrics_native.get("mae")),
        "ghost_visual_score": _to_float(ghost.get("ghost_visual_score", ghost.get("ghost_visual_score_mean"))),
        "ghost_score": _to_float(ghost.get("ghost_score", ghost.get("ghost_score_mean"))),
        "peak_count": _to_int(ghost.get("peak_count", 0)),
        "width_ratio": _to_float(ghost.get("width_ratio", ghost.get("width_ratio_mean"))),
        "area_ratio": _to_float(ghost.get("area_ratio", ghost.get("area_ratio_mean"))),
        "pred_luma_mean": _to_float(ghost.get("pred_luma_mean", ghost.get("pred_luma_mean_mean"))),
        "cuda_peak_mem_mb": _to_float(meta.get("cuda_peak_mem_mb")),
        "elapsed_sec_total": _to_float(report.get("elapsed_sec_total")),
        "meta": meta,
    }


def _profile_summary(out_root: Path, profile: str, *, tgt_camera: str, pretrained_ckpt: str, objective: str) -> dict:
    dirs = profile_dirs(out_root, profile)
    meta = profile_metadata(profile, tgt_camera)
    train = _collect_train_metrics(dirs["train"])
    pre = _collect_stage(dirs["pre_update"])
    post = _collect_stage(dirs["post_update"])
    summary = {
        "probe_start": str(pretrained_ckpt),
        "objective": str(objective),
        "profile": str(profile),
        "profile_meta": meta,
        "train": train,
        "pre_update": pre,
        "post_update": post,
        "delta": {
            "coverage_ratio": _to_float(post.get("coverage_ratio")) - _to_float(pre.get("coverage_ratio")),
            "mean_conf": _to_float(post.get("mean_conf")) - _to_float(pre.get("mean_conf")),
            "valid_contrib": _to_int(post.get("valid_contrib")) - _to_int(pre.get("valid_contrib")),
            "ghost_visual_score": _to_float(post.get("ghost_visual_score")) - _to_float(pre.get("ghost_visual_score")),
            "peak_count": _to_int(post.get("peak_count")) - _to_int(pre.get("peak_count")),
            "native_psnr": _to_float(post.get("native_psnr")) - _to_float(pre.get("native_psnr")),
            "native_ssim": _to_float(post.get("native_ssim")) - _to_float(pre.get("native_ssim")),
            "weight_mad": _image_mad(dirs["pre_update"] / "weight_native.png", dirs["post_update"] / "weight_native.png"),
            "pred_mad": _image_mad(dirs["pre_update"] / "pred_native.png", dirs["post_update"] / "pred_native.png"),
        },
        "paths": {
            "task_state_json": str(dirs["task_state_json"]),
            "train_dir": str(dirs["train"]),
            "pre_update_dir": str(dirs["pre_update"]),
            "post_update_dir": str(dirs["post_update"]),
            "compare_dir": str(dirs["compare"]),
        },
    }
    return summary


def _write_profile_summary(summary: dict, compare_dir: Path) -> None:
    compare_dir.mkdir(parents=True, exist_ok=True)
    summary_json = compare_dir / "summary.json"
    summary_md = compare_dir / "summary.md"
    write_json(summary_json, summary)
    profile = summary["profile"]
    meta = summary["profile_meta"]
    train = summary["train"]
    pre = summary["pre_update"]
    post = summary["post_update"]
    delta = summary["delta"]
    lines = [
        f"# Original VGGT One-Step Probe: {profile}",
        "",
        f"- probe_start: `{summary['probe_start']}`",
        f"- objective: `{summary['objective']}`",
        f"- fixed_eval: `seq={pre.get('meta', {}).get('seq_name', '')}` / `frame={pre.get('meta', {}).get('frame_id', '')}` / `tgt={pre.get('meta', {}).get('tgt_camera', '')}`",
        "- taxonomy: `weight_native.png` / `cat_weight_pred_tgt*.png` are point-support visuals.",
        "- taxonomy: `cat_fg_mask_pred_tgt_step000000.png` is a ghost triplet, not a point cloud image.",
        "",
        "## Point-Support First",
        "",
        f"- render_src_views: `{meta['render_num_src_views_actual']}`; train_cameras: `{meta['train_num_cameras']}`",
        f"- coverage_ratio: `{fmt_num(pre['coverage_ratio'])} -> {fmt_num(post['coverage_ratio'])}`",
        f"- mean_conf: `{fmt_num(pre['mean_conf'])} -> {fmt_num(post['mean_conf'])}`",
        f"- valid_contrib: `{pre['valid_contrib']} -> {post['valid_contrib']}`",
        f"- weight_mad: `{fmt_num(delta['weight_mad'])}`",
        "",
        "## Ghost Second",
        "",
        f"- ghost_visual_score: `{fmt_num(pre['ghost_visual_score'])} -> {fmt_num(post['ghost_visual_score'])}`",
        f"- peak_count: `{pre['peak_count']} -> {post['peak_count']}`",
        f"- pred_mad: `{fmt_num(delta['pred_mad'])}`",
        f"- native_psnr: `{fmt_num(pre['native_psnr'])} -> {fmt_num(post['native_psnr'])}`",
        "",
        "## Train Sanity",
        "",
        f"- train_verified: `{train['train_verified']}`",
        f"- run_meta.max_steps_per_epoch: `{train['run_meta_max_steps_per_epoch']}`",
        f"- step_eval_count: `{train['step_eval_count']}`",
        f"- epoch_end_steps: `{train['epoch_end_steps']}`",
        f"- step_update_ratio: `{fmt_num(train['step_update_ratio'])}`",
    ]
    write_text(summary_md, "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser("summarize_probe_runs")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--profiles", default="6src_hist,12src_nested,23cam_fullset")
    ap.add_argument("--tgt_camera", default="Camera_B5")
    ap.add_argument("--pretrained_ckpt", default="model.pt")
    ap.add_argument("--objective", default="finetune_defaults")
    ap.add_argument(
        "--out_json",
        default=str(DEFAULT_REPORTS_DIR / "orig_vggt_one_step_probe_summary_latest.json"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(DEFAULT_REPORTS_DIR / "orig_vggt_one_step_probe_summary_latest.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(DEFAULT_REPORTS_DIR / "orig_vggt_one_step_probe_summary_latest.md"),
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    profiles = parse_profiles(args.profiles)

    summaries: list[dict] = []
    for profile in profiles:
        dirs = profile_dirs(out_root, profile)
        if not dirs["pre_update"].is_dir() or not dirs["post_update"].is_dir():
            continue
        summary = _profile_summary(
            out_root,
            profile,
            tgt_camera=args.tgt_camera,
            pretrained_ckpt=args.pretrained_ckpt,
            objective=args.objective,
        )
        _write_profile_summary(summary, dirs["compare"])
        summaries.append(summary)

    rows: list[dict] = []
    for summary in summaries:
        meta = summary["profile_meta"]
        train = summary["train"]
        pre = summary["pre_update"]
        post = summary["post_update"]
        delta = summary["delta"]
        rows.append(
            {
                "profile": summary["profile"],
                "render_num_src_views_actual": meta["render_num_src_views_actual"],
                "train_num_cameras": meta["train_num_cameras"],
                "train_verified": train["train_verified"],
                "step_eval_count": train["step_eval_count"],
                "epoch_end_steps": train["epoch_end_steps"],
                "step_update_ratio": train["step_update_ratio"],
                "pre_coverage_ratio": pre["coverage_ratio"],
                "post_coverage_ratio": post["coverage_ratio"],
                "delta_coverage_ratio": delta["coverage_ratio"],
                "pre_mean_conf": pre["mean_conf"],
                "post_mean_conf": post["mean_conf"],
                "delta_mean_conf": delta["mean_conf"],
                "pre_valid_contrib": pre["valid_contrib"],
                "post_valid_contrib": post["valid_contrib"],
                "delta_valid_contrib": delta["valid_contrib"],
                "pre_ghost_visual_score": pre["ghost_visual_score"],
                "post_ghost_visual_score": post["ghost_visual_score"],
                "delta_ghost_visual_score": delta["ghost_visual_score"],
                "pre_peak_count": pre["peak_count"],
                "post_peak_count": post["peak_count"],
                "delta_peak_count": delta["peak_count"],
                "pre_native_psnr": pre["native_psnr"],
                "post_native_psnr": post["native_psnr"],
                "delta_native_psnr": delta["native_psnr"],
                "weight_mad": delta["weight_mad"],
                "pred_mad": delta["pred_mad"],
                "compare_summary_json": str(profile_dirs(out_root, summary["profile"])["compare"] / "summary.json"),
            }
        )

    payload = {
        "rows": rows,
        "count_profiles": len(rows),
        "taxonomy": {
            "point_support_images": ["weight_native.png", "cat_weight_pred_tgt*.png"],
            "ghost_triplet_images": ["cat_fg_mask_pred_tgt_step000000.png"],
            "note": "This report is for native model.pt one-step probe, not H-family finetune.",
        },
    }
    write_json(args.out_json, payload)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Original VGGT One-Step Probe Summary",
        "",
        "- probe_start: `model.pt`",
        "- fixed_eval: `CoreView_390 / frame 1080 / Camera_B5`",
        "- taxonomy: `weight_native.png` / `cat_weight_pred_tgt*.png` are point-support visuals.",
        "- taxonomy: `cat_fg_mask_pred_tgt_step000000.png` is a ghost triplet, not a point cloud image.",
        "- note: this is the native VGGT one-step probe, not H0/H1 finetune reporting.",
        "",
        "## Point-Support First",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['profile']}` render_src=`{row['render_num_src_views_actual']}` train_cams=`{row['train_num_cameras']}` "
            f"coverage=`{fmt_num(row['pre_coverage_ratio'])} -> {fmt_num(row['post_coverage_ratio'])}` "
            f"mean_conf=`{fmt_num(row['pre_mean_conf'])} -> {fmt_num(row['post_mean_conf'])}` "
            f"weight_mad=`{fmt_num(row['weight_mad'])}`"
        )
    lines.extend(["", "## Ghost Second", ""])
    for row in rows:
        lines.append(
            f"- `{row['profile']}` ghost_visual=`{fmt_num(row['pre_ghost_visual_score'])} -> {fmt_num(row['post_ghost_visual_score'])}` "
            f"peak_count=`{row['pre_peak_count']} -> {row['post_peak_count']}` "
            f"pred_mad=`{fmt_num(row['pred_mad'])}`"
        )
    lines.extend(["", "## Train Sanity", ""])
    for row in rows:
        lines.append(
            f"- `{row['profile']}` verified=`{row['train_verified']}` "
            f"step_eval_count=`{row['step_eval_count']}` "
            f"epoch_end_steps=`{row['epoch_end_steps']}` "
            f"step_update_ratio=`{fmt_num(row['step_update_ratio'])}`"
        )
    write_text(args.out_md, "\n".join(lines) + "\n")
    print(f"[orig-one-step-summary] wrote {args.out_json}")


if __name__ == "__main__":
    main()
