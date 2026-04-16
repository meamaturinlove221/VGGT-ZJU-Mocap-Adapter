import argparse
import ast
import csv
import glob
import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PEAK_MIN_REL = 0.35

FINETUNE_DEFAULTS = {
    "tf32": True,
    "amp": True,
    "strict_deterministic": False,
    "fg_supervision_region_mode": "all",
    "fg_supervision_region_erode_px": 0,
    "lambda_fg_structure_depth_edge": 0.0,
    "fg_structure_bbox_margin_px": 12,
    "fg_structure_bbox_min_side_px": 24,
    "lambda_point_mv_outside_ring": 0.0,
    "point_mv_outside_ring_px": 3,
}

FT_ROW_DEFAULTS = {
    "fg_supervision_region_mode": "all",
    "fg_supervision_region_erode_px": 0.0,
    "fg_structure_depth_edge_active": 0.0,
    "fg_structure_bbox_active_ratio": 0.0,
    "fg_structure_depth_edge_active_views": 0.0,
    "fg_structure_depth_edge_loss": 0.0,
    "fg_structure_depth_edge_pred_mean": 0.0,
    "fg_structure_depth_edge_tgt_mean": 0.0,
    "point_mv_outside_ring_active": 0.0,
    "point_mv_outside_ring_active_views": 0.0,
    "point_mv_outside_ring_hit_ratio": 0.0,
    "point_mv_outside_ring_loss": 0.0,
    "point_mv_outside_ring_base_proj_ratio": 0.0,
    "point_mv_outside_ring_valid_ratio": 0.0,
    "loss_fg_structure_depth_edge": 0.0,
    "loss_contrib_fg_structure_depth_edge": 0.0,
    "mean_loss_fg_structure_depth_edge": 0.0,
    "loss_point_mv_outside_ring": 0.0,
    "loss_contrib_point_mv_outside_ring": 0.0,
    "mean_loss_point_mv_outside_ring": 0.0,
    "tf32": True,
    "amp": True,
    "strict_deterministic": False,
    "runner_tf32": True,
    "runner_amp": True,
    "runner_strict_deterministic": False,
    "precompute_tf32": True,
    "precompute_amp": True,
    "precompute_strict_deterministic": False,
    "teacher_tf32": True,
    "teacher_amp": True,
    "teacher_deterministic": False,
    "lambda_fg_structure_depth_edge": 0.0,
    "fg_structure_bbox_margin_px": 12.0,
    "fg_structure_bbox_min_side_px": 24.0,
    "lambda_point_mv_outside_ring": 0.0,
    "point_mv_outside_ring_px": 3.0,
}

FINETUNE_IGNORE = {"event", "device"}
PRECOMPUTE_IGNORE = {
    "code_dir",
    "mnt_code",
    "mnt_data",
    "mnt_out",
    "precompute_ckpt",
    "geom_subdir",
    "timeout_sec",
    "data_vol",
    "out_vol",
    "archives_dir",
}
EVAL_IGNORE = {"geom"}

FT_COMPARE_KEYS = [
    "pointmap_source_requested",
    "pointmap_source_resolved",
    "point_target_mode",
    "precompute_mv_support_on",
    "precompute_mv_support_region_mode",
    "precompute_mv_support_fg_mask_source",
    "precompute_mv_support_fg_erode_px",
    "precompute_mv_support_fg_preserve_px",
    "fg_supervision_boost",
    "fg_supervision_boost_applied",
    "fg_supervision_bg_floor",
    "fg_supervision_region_mode",
    "fg_supervision_region_erode_px",
    "fg_conf_presence_enabled",
    "fg_conf_presence_target_ratio",
    "lambda_fg_conf_presence",
    "tf32",
    "amp",
    "strict_deterministic",
    "runner_tf32",
    "runner_amp",
    "runner_strict_deterministic",
    "precompute_tf32",
    "precompute_amp",
    "precompute_strict_deterministic",
    "teacher_tf32",
    "teacher_amp",
    "teacher_deterministic",
    "lambda_fg_structure_depth_edge",
    "fg_structure_bbox_margin_px",
    "fg_structure_bbox_min_side_px",
    "lambda_point_mv_outside_ring",
    "point_mv_outside_ring_px",
    "supervision_weight_mode",
    "point_loss_fg_erode_px",
    "point_support_mode",
    "point_mv_depth_support_mode",
    "point_mv_mask_support_mode",
    "support_generation_active",
    "cam_count_used",
    "eval_num_src_views",
    "eval_num_src_views_declared",
    "precompute_source",
]

ACTIVE_KEYS = [
    "fg_structure_depth_edge_active",
    "fg_structure_bbox_active_ratio",
    "fg_structure_depth_edge_active_views",
    "fg_structure_depth_edge_loss",
    "fg_structure_depth_edge_pred_mean",
    "fg_structure_depth_edge_tgt_mean",
    "point_mv_outside_ring_active",
    "point_mv_outside_ring_active_views",
    "point_mv_outside_ring_hit_ratio",
    "point_mv_outside_ring_loss",
    "point_mv_outside_ring_base_proj_ratio",
    "point_mv_outside_ring_valid_ratio",
    "loss_fg_structure_depth_edge",
    "loss_contrib_fg_structure_depth_edge",
    "mean_loss_fg_structure_depth_edge",
    "loss_point_mv_outside_ring",
    "loss_contrib_point_mv_outside_ring",
    "mean_loss_point_mv_outside_ring",
]

PRECISION_KEYS = [
    "runner_tf32",
    "runner_amp",
    "runner_strict_deterministic",
    "teacher_tf32",
    "teacher_amp",
    "teacher_deterministic",
]


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def to_float(value, default=float("nan")):
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def normalize_value(value):
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if abs(float(value) - round(float(value))) < 1e-9:
            return int(round(float(value)))
        return round(float(value), 10)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if math.isnan(num) or math.isinf(num):
            return None
        if abs(num - round(num)) < 1e-9:
            return int(round(num))
        return round(num, 10)
    except Exception:
        return text


def diff_dicts(lhs, rhs, ignore=()):
    ignore_set = {str(x) for x in ignore}
    diffs = []
    for key in sorted((set(lhs.keys()) | set(rhs.keys())) - ignore_set):
        lv = normalize_value(lhs.get(key))
        rv = normalize_value(rhs.get(key))
        if lv != rv:
            diffs.append({"key": key, "baseline": lv, "current": rv})
    return diffs


def first_row(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    raise RuntimeError(f"no rows in csv: {csv_path}")


def fill_defaults(row, defaults):
    out = dict(row)
    for key, value in defaults.items():
        if key not in out or str(out.get(key) or "").strip() == "":
            out[key] = value
    return out


def find_one(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def parse_ft_id_from_ckpt(ft_ckpt):
    text = str(ft_ckpt or "").replace("\\", "/")
    match = re.search(r"/finetune/lr_([^/]+)_(\d{8}_\d{6})/ckpt/model_ft_zju\.pt$", text)
    if not match:
        raise RuntimeError(f"failed to parse ft ckpt timestamp: {ft_ckpt}")
    return match.group(1), match.group(2)


def derive_artifacts(repo_dir, candidate_json):
    candidate = read_json(candidate_json)
    run_timestamp = str(candidate.get("run_timestamp") or "").strip()
    ft_csv = find_one(os.path.join(repo_dir, "logs", "modal_phase5", f"vggt_ft_sweep_ghost_*_{run_timestamp}.csv"))
    ft_row = fill_defaults(first_row(ft_csv), FT_ROW_DEFAULTS)
    ft_lr_label, ft_timestamp = parse_ft_id_from_ckpt(str(ft_row.get("ft_ckpt") or ""))
    return {
        "candidate": candidate,
        "run_timestamp": run_timestamp,
        "ft_csv": ft_csv,
        "ft_row": ft_row,
        "ft_metrics_jsonl": os.path.join(repo_dir, "logs", "modal_phase5", f"ftdebug_{ft_lr_label}_{ft_timestamp}_short_metrics.jsonl"),
        "ft_summary_json": os.path.join(repo_dir, "logs", "modal_phase5", f"ftdebug_{ft_lr_label}_{ft_timestamp}_short_summary.json"),
        "finetune_log": os.path.join(repo_dir, "logs", "modal_phase5", f"vggt_ft_lr_{ft_lr_label}_{ft_timestamp}.finetune.log"),
        "precompute_log": os.path.join(repo_dir, "logs", "modal_phase5", f"vggt_ft_lr_{ft_lr_label}_{ft_timestamp}.precompute.log"),
        "eval_log": os.path.join(repo_dir, "logs", "modal_phase5", f"vggt_ft_lr_{ft_timestamp}.eval_short.log"),
        "ghost_rows_csv": find_one(os.path.join(repo_dir, "logs", "modal_phase5", f"ghost_score_rows_*_{run_timestamp}.csv")),
        "triplet_dir": find_one(os.path.join(repo_dir, "logs", "modal_phase5", f"_ghost_eval_*_{run_timestamp}")),
    }


def load_run_meta(metrics_jsonl):
    run_meta = None
    first_eval = {}
    for row in read_jsonl(metrics_jsonl):
        event = str(row.get("event") or "")
        if event == "run_meta" and run_meta is None:
            run_meta = dict(row)
        elif event in {"step_eval", "step_debug"} and not first_eval:
            first_eval = dict(row)
        if run_meta is not None and first_eval:
            break
    if run_meta is None:
        raise RuntimeError(f"run_meta missing in {metrics_jsonl}")
    merged = dict(FINETUNE_DEFAULTS)
    merged.update(run_meta)
    return merged, first_eval


def parse_cfg_repr(log_path):
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            with open(log_path, "r", encoding=encoding, errors="replace") as f:
                for line in f:
                    if "[local] cfg = Cfg(" not in line:
                        continue
                    expr = line.split("[local] cfg = ", 1)[1].strip()
                    node = ast.parse(expr, mode="eval").body
                    if not isinstance(node, ast.Call):
                        raise RuntimeError(f"unexpected cfg expression in {log_path}")
                    out = {}
                    for kw in node.keywords:
                        if kw.arg:
                            out[kw.arg] = ast.literal_eval(kw.value)
                    return out
        except UnicodeError:
            continue
    raise RuntimeError(f"cfg line missing in {log_path}")


def parse_eval_info(eval_log):
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            with open(eval_log, "r", encoding=encoding, errors="replace") as f:
                for line in f:
                    if line.startswith("[eval] run infer "):
                        match = re.search(r"label=([^\s]+)\s+geom=([^\s]+)", line)
                        if match:
                            return {"label": match.group(1), "geom": match.group(2)}
                        return {}
        except UnicodeError:
            continue
    return {}


def _parse_boolish(value):
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def parse_precompute_precision(log_path, precompute_cfg):
    precision = {
        "runner_tf32": _parse_boolish(precompute_cfg.get("tf32")),
        "runner_amp": _parse_boolish(precompute_cfg.get("amp")),
        "runner_strict_deterministic": _parse_boolish(precompute_cfg.get("strict_deterministic")),
        "teacher_tf32": None,
        "teacher_amp": None,
        "teacher_deterministic": None,
    }
    banner_re = re.compile(
        r"precompute_tf32=\s*(True|False)\s+precompute_amp=\s*(True|False)\s+precompute_strict_deterministic=\s*(True|False)"
    )
    teacher_re = re.compile(
        r"\[VGGTGeomTeacher\].*amp=(True|False).*tf32=(True|False).*deterministic=(True|False)"
    )
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            with open(log_path, "r", encoding=encoding, errors="replace") as f:
                for line in f:
                    if precision["teacher_tf32"] is None:
                        m = teacher_re.search(line)
                        if m:
                            precision["teacher_amp"] = _parse_boolish(m.group(1))
                            precision["teacher_tf32"] = _parse_boolish(m.group(2))
                            precision["teacher_deterministic"] = _parse_boolish(m.group(3))
                    m = banner_re.search(line)
                    if m:
                        precision["runner_tf32"] = _parse_boolish(m.group(1))
                        precision["runner_amp"] = _parse_boolish(m.group(2))
                        precision["runner_strict_deterministic"] = _parse_boolish(m.group(3))
                break
        except UnicodeError:
            continue
    if precision["teacher_tf32"] is None:
        precision["teacher_tf32"] = precision["runner_tf32"]
        precision["teacher_amp"] = precision["runner_amp"]
        precision["teacher_deterministic"] = precision["runner_strict_deterministic"]
    return precision


def collect_numbers(lines, pattern):
    regex = re.compile(pattern)
    out = []
    for line in lines:
        match = regex.search(line)
        if match:
            try:
                out.append(float(match.group(1)))
            except Exception:
                pass
    return out


def summarize_precompute(log_path):
    lines = None
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            with open(log_path, "r", encoding=encoding, errors="replace") as f:
                cand_lines = f.readlines()
            joined = "".join(cand_lines)
            if ("teacher_forward_sec" in joined) or ("batch_gate_ready" in joined) or ("mv_support_done" in joined):
                lines = cand_lines
                break
        except UnicodeError:
            continue
    if lines is None:
        lines = []
    teacher = collect_numbers(lines, r"teacher_forward_sec(?:=|\"?:\s*)([0-9]+(?:\.[0-9]+)?)")
    batch_total = collect_numbers(lines, r"batch_total_sec(?:=|\"?:\s*)([0-9]+(?:\.[0-9]+)?)")
    mv_support = collect_numbers(lines, r"mv_support_sec(?:=|\"?:\s*)([0-9]+(?:\.[0-9]+)?)")
    gate_prepare = collect_numbers(lines, r"gate_prepare_sec(?:=|\"?:\s*)([0-9]+(?:\.[0-9]+)?)")
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "teacher_forward_mean": mean(teacher),
        "batch_total_mean": mean(batch_total),
        "mv_support_mean": mean(mv_support),
        "gate_prepare_mean": mean(gate_prepare),
    }


def load_ghost_rows(csv_path):
    out = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(str(row.get("step") or "").strip()))
            except Exception:
                continue
            out[step] = dict(row)
    return out


def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def smooth_1d(x, k=11):
    kk = max(1, int(k))
    if kk <= 1:
        return x.astype(np.float32)
    pad = kk // 2
    kernel = np.ones((kk,), dtype=np.float32) / float(kk)
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def bbox_from_binary(mask):
    ys, xs = np.where(mask > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def split_triptych(arr):
    h, w = arr.shape[:2]
    p = w // 3
    return arr[:, :p], arr[:, p : 2 * p], arr[:, 2 * p : 3 * p]


def find_peaks(profile, min_rel, min_dist):
    if profile.size < 3:
        return []
    p = profile.astype(np.float32)
    mx = float(p.max())
    if mx <= 1e-8:
        return []
    thr = float(max(0.0, min(1.0, min_rel))) * mx
    peaks = []
    md = max(1, int(min_dist))
    for i in range(1, p.size - 1):
        if p[i] < thr or p[i] < p[i - 1] or p[i] < p[i + 1]:
            continue
        if (not peaks) or (i - peaks[-1] >= md):
            peaks.append(i)
        elif p[i] > p[peaks[-1]]:
            peaks[-1] = i
    return peaks


def analyze_triptych(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    left, mid, _right = split_triptych(arr)
    left_gray = left.mean(axis=2)
    left_bin = left_gray > 128.0
    bbox = bbox_from_binary(left_bin)
    if bbox is None:
        raise RuntimeError(f"mask panel empty: {path}")
    x0, x1, _y0, _y1 = bbox
    mask_w = float(max(1, x1 - x0 + 1))
    g = mid[:, :, 1]
    thr = max(float(np.percentile(g, 84.0)), 10.0)
    mid_bin = g > thr
    prof = smooth_1d(mid_bin.sum(axis=0).astype(np.float32), k=11)
    peaks = find_peaks(prof, min_rel=PEAK_MIN_REL, min_dist=max(6, int(mask_w * 0.12)))
    return {
        "triptych": arr,
        "pred_panel": mid,
        "pred_bin": mid_bin.astype(np.uint8) * 255,
        "peaks": peaks,
        "pred_thr": thr,
    }


def build_png(path, payload, baseline_debug, current_debug):
    font = ImageFont.load_default()
    base_img = Image.fromarray(baseline_debug["triptych"])
    curr_img = Image.fromarray(current_debug["triptych"])
    w = max(base_img.width, curr_img.width)
    scale = min(1.0, 900.0 / float(max(1, w)))
    base_img = base_img.resize((int(base_img.width * scale), int(base_img.height * scale)), Image.Resampling.NEAREST)
    curr_img = curr_img.resize((int(curr_img.width * scale), int(curr_img.height * scale)), Image.Resampling.NEAREST)
    width = max(base_img.width, curr_img.width) + 40
    height = 180 + base_img.height + curr_img.height + 20
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    lines = [
        "H0 Equivalence Debug",
        f"baseline={payload['baseline']['run_timestamp']} current={payload['current']['run_timestamp']}",
        (
            "precision baseline "
            f"runner(tf32={payload['precision']['baseline']['runner_tf32']},"
            f"amp={payload['precision']['baseline']['runner_amp']},"
            f"det={payload['precision']['baseline']['runner_strict_deterministic']}) "
            f"teacher(tf32={payload['precision']['baseline']['teacher_tf32']},"
            f"amp={payload['precision']['baseline']['teacher_amp']},"
            f"det={payload['precision']['baseline']['teacher_deterministic']})"
        ),
        (
            "precision current  "
            f"runner(tf32={payload['precision']['current']['runner_tf32']},"
            f"amp={payload['precision']['current']['runner_amp']},"
            f"det={payload['precision']['current']['runner_strict_deterministic']}) "
            f"teacher(tf32={payload['precision']['current']['teacher_tf32']},"
            f"amp={payload['precision']['current']['teacher_amp']},"
            f"det={payload['precision']['current']['teacher_deterministic']})"
        ),
        f"ghost delta={payload['metric_deltas']['ghost_visual_score']:+.4f}  luma delta={payload['metric_deltas']['fg_pred_luma_mean']:+.6f}",
        f"contrast delta={payload['metric_deltas']['fg_pred_contrast']:+.6f}  tgt_l1 delta={payload['metric_deltas']['fg_pred_tgt_l1']:+.6f}",
        f"step000002 peaks {payload['step2_baseline']['peak_count']} -> {payload['step2_current']['peak_count']}",
        f"first_divergence={payload['first_divergence']}",
    ]
    y = 18
    for line in lines:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 18
    canvas.paste(base_img, (20, y))
    draw.text((20, y - 14), "Baseline step000002 triptych", fill=(0, 0, 0), font=font)
    y2 = y + base_img.height + 20
    canvas.paste(curr_img, (20, y2))
    draw.text((20, y2 - 14), "H0 step000002 triptych", fill=(0, 0, 0), font=font)
    canvas.save(path)


def render_md(path, payload):
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            return f"{v:.6f}"
        return str(v)

    lines = ["# H0 Equivalence Debug", "", "## Summary", ""]
    lines.append(f"- baseline candidate: `{payload['baseline']['candidate_json']}`")
    lines.append(f"- current candidate: `{payload['current']['candidate_json']}`")
    lines.append(f"- first divergence: `{payload['first_divergence']}`")
    lines.extend(["", "## Parity Deltas", ""])
    for key, value in payload["metric_deltas"].items():
        lines.append(f"- `{key}`: `{fmt(value)}`")
    lines.extend(["", "## Precision", ""])
    for side in ("baseline", "current"):
        row = payload["precision"][side]
        lines.append(
            f"- `{side}`: runner(tf32=`{row['runner_tf32']}` amp=`{row['runner_amp']}` strict_deterministic=`{row['runner_strict_deterministic']}`) "
            f"teacher(tf32=`{row['teacher_tf32']}` amp=`{row['teacher_amp']}` deterministic=`{row['teacher_deterministic']}`)"
        )
    lines.extend(["", "## Precision Diff", ""])
    if not payload["precision_diffs"]:
        lines.append("- none")
    else:
        for diff in payload["precision_diffs"]:
            lines.append(f"- `{diff['key']}`: baseline=`{diff['baseline']}` current=`{diff['current']}`")
    for section in ("finetune", "precompute", "eval"):
        lines.extend(["", f"## Effective Config Diff: {section}", ""])
        diffs = payload["effective_config_diffs"][section]
        if not diffs:
            lines.append("- none")
        else:
            for diff in diffs:
                lines.append(f"- `{diff['key']}`: baseline=`{diff['baseline']}` current=`{diff['current']}`")
    lines.extend(["", "## FT Row Config Diff", ""])
    if not payload["ft_row_config_diffs"]:
        lines.append("- none")
    else:
        for diff in payload["ft_row_config_diffs"]:
            lines.append(f"- `{diff['key']}`: baseline=`{diff['baseline']}` current=`{diff['current']}`")
    lines.extend(["", "## Active / Loss-Contrib Diff", ""])
    if not payload["active_loss_diffs"]:
        lines.append("- none")
    else:
        for diff in payload["active_loss_diffs"]:
            lines.append(f"- `{diff['key']}`: baseline=`{diff['baseline']}` current=`{diff['current']}`")
    lines.extend(["", "## Precompute Means", ""])
    for key, row in payload["precompute_means"].items():
        lines.append(f"- `{key}`: baseline=`{fmt(row['baseline'])}` current=`{fmt(row['current'])}` delta=`{fmt(row['delta'])}`")
    lines.extend(["", "## Ghost Rows", ""])
    for row in payload["ghost_rows"]:
        lines.append(
            f"- `step{int(row['step']):06d}`: ghost `{row['baseline']['ghost_visual_score']:.4f} -> {row['current']['ghost_visual_score']:.4f}`, "
            f"peak_count `{int(row['baseline']['peak_count'])} -> {int(row['current']['peak_count'])}`"
        )
    lines.extend(["", "## Image Parity", ""])
    for row in payload["triplet_image_parity"]:
        lines.append(f"- `step{int(row['step']):06d}`: MAD=`{row['mad']:.6f}` PSNR=`{row['psnr']:.4f}`")
    lines.extend(["", "## Step000002 Focus", ""])
    lines.append(f"- baseline peaks: `{payload['step2_baseline']['peak_positions']}`")
    lines.append(f"- current peaks: `{payload['step2_current']['peak_positions']}`")
    lines.append(f"- baseline threshold: `{payload['step2_baseline']['pred_thr']:.4f}`")
    lines.append(f"- current threshold: `{payload['step2_current']['pred_thr']:.4f}`")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser("render_h0_equivalence_debug")
    ap.add_argument("--repo_dir", default=".")
    ap.add_argument("--baseline_candidate_json", required=True)
    ap.add_argument("--current_candidate_json", required=True)
    ap.add_argument("--baseline_contract_json", required=True)
    ap.add_argument("--current_contract_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_baseline_config_json", required=True)
    ap.add_argument("--out_current_config_json", required=True)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    baseline = derive_artifacts(repo_dir, os.path.abspath(args.baseline_candidate_json))
    current = derive_artifacts(repo_dir, os.path.abspath(args.current_candidate_json))
    baseline_contract = read_json(os.path.abspath(args.baseline_contract_json))
    current_contract = read_json(os.path.abspath(args.current_contract_json))
    baseline_run_meta, baseline_first_eval = load_run_meta(baseline["ft_metrics_jsonl"])
    current_run_meta, current_first_eval = load_run_meta(current["ft_metrics_jsonl"])
    baseline_precompute_cfg = parse_cfg_repr(baseline["precompute_log"])
    current_precompute_cfg = parse_cfg_repr(current["precompute_log"])
    baseline_eval_cfg = parse_eval_info(baseline["eval_log"])
    current_eval_cfg = parse_eval_info(current["eval_log"])
    baseline_precision = parse_precompute_precision(baseline["precompute_log"], baseline_precompute_cfg)
    current_precision = parse_precompute_precision(current["precompute_log"], current_precompute_cfg)

    write_json(args.out_baseline_config_json, {"candidate": baseline["candidate"], "contract": baseline_contract, "finetune_run_meta": baseline_run_meta, "precompute_cfg": baseline_precompute_cfg, "precompute_precision": baseline_precision, "eval_cfg": baseline_eval_cfg})
    write_json(args.out_current_config_json, {"candidate": current["candidate"], "contract": current_contract, "finetune_run_meta": current_run_meta, "precompute_cfg": current_precompute_cfg, "precompute_precision": current_precision, "eval_cfg": current_eval_cfg})

    effective_config_diffs = {
        "finetune": diff_dicts(baseline_run_meta, current_run_meta, ignore=FINETUNE_IGNORE),
        "precompute": diff_dicts(baseline_precompute_cfg, current_precompute_cfg, ignore=PRECOMPUTE_IGNORE),
        "eval": diff_dicts(baseline_eval_cfg, current_eval_cfg, ignore=EVAL_IGNORE),
    }
    precision_diffs = diff_dicts(
        {k: baseline_precision.get(k) for k in PRECISION_KEYS},
        {k: current_precision.get(k) for k in PRECISION_KEYS},
    )
    ft_row_config_diffs = diff_dicts({k: baseline["ft_row"].get(k) for k in FT_COMPARE_KEYS}, {k: current["ft_row"].get(k) for k in FT_COMPARE_KEYS})
    active_loss_diffs = diff_dicts({k: baseline["ft_row"].get(k, FT_ROW_DEFAULTS.get(k, 0.0)) for k in ACTIVE_KEYS}, {k: current["ft_row"].get(k, FT_ROW_DEFAULTS.get(k, 0.0)) for k in ACTIVE_KEYS})
    bpre = summarize_precompute(baseline["precompute_log"])
    cpre = summarize_precompute(current["precompute_log"])
    precompute_means = {k: {"baseline": bpre[k], "current": cpre[k], "delta": cpre[k] - bpre[k]} for k in ("teacher_forward_mean", "batch_total_mean", "mv_support_mean", "gate_prepare_mean")}
    baseline_rows = load_ghost_rows(baseline["ghost_rows_csv"])
    current_rows = load_ghost_rows(current["ghost_rows_csv"])
    ghost_rows = []
    for step in (0, 1, 2):
        b = baseline_rows.get(step, {})
        c = current_rows.get(step, {})
        ghost_rows.append({"step": step, "baseline": {"ghost_visual_score": to_float(b.get("ghost_visual_score"), 0.0), "peak_count": to_float(b.get("peak_count"), 0.0)}, "current": {"ghost_visual_score": to_float(c.get("ghost_visual_score"), 0.0), "peak_count": to_float(c.get("peak_count"), 0.0)}})
    triplet_image_parity = []
    for step in (0, 1, 2):
        bpath = os.path.join(baseline["triplet_dir"], f"infer_val_e005_cat_fg_mask_pred_tgt_step{step:06d}.png")
        cpath = os.path.join(current["triplet_dir"], f"infer_val_e005_cat_fg_mask_pred_tgt_step{step:06d}.png")
        bimg = np.asarray(Image.open(bpath).convert("RGB"), dtype=np.uint8)
        cimg = np.asarray(Image.open(cpath).convert("RGB"), dtype=np.uint8)
        triplet_image_parity.append({"step": step, "mad": mean_abs_diff(bimg, cimg), "psnr": psnr(bimg, cimg)})
    baseline_step2 = analyze_triptych(os.path.join(baseline["triplet_dir"], "infer_val_e005_cat_fg_mask_pred_tgt_step000002.png"))
    current_step2 = analyze_triptych(os.path.join(current["triplet_dir"], "infer_val_e005_cat_fg_mask_pred_tgt_step000002.png"))
    metric_deltas = {
        "ghost_visual_score": to_float(current["candidate"].get("ghost_visual_score"), 0.0) - to_float(baseline["candidate"].get("ghost_visual_score"), 0.0),
        "fg_pred_luma_mean": to_float(current["candidate"].get("fg_pred_luma_mean"), 0.0) - to_float(baseline["candidate"].get("fg_pred_luma_mean"), 0.0),
        "fg_pred_contrast": to_float(current["candidate"].get("fg_pred_contrast"), 0.0) - to_float(baseline["candidate"].get("fg_pred_contrast"), 0.0),
        "fg_pred_tgt_l1": to_float(current["candidate"].get("fg_pred_tgt_l1"), 0.0) - to_float(baseline["candidate"].get("fg_pred_tgt_l1"), 0.0),
    }
    first_divergence = "effective config differs"
    if not any(effective_config_diffs.values()):
        first_divergence = "effective config appears equivalent"
        if precision_diffs:
            first_divergence = "precision differs despite otherwise equivalent config"
        elif ft_row_config_diffs:
            first_divergence = "FT row config diverges"
        elif active_loss_diffs:
            first_divergence = "H active/loss fields diverge in FT row"
        elif any(abs(precompute_means[k]["delta"]) > (0.25 if k != "mv_support_mean" else 0.01) for k in precompute_means):
            first_divergence = "precompute timing diverges despite equivalent config"
        elif abs(metric_deltas["ghost_visual_score"]) > 0.08:
            first_divergence = "ghost scorer flips on tiny image drift"

    payload = {
        "baseline": {"candidate_json": os.path.abspath(args.baseline_candidate_json), "contract_json": os.path.abspath(args.baseline_contract_json), "run_timestamp": baseline["run_timestamp"]},
        "current": {"candidate_json": os.path.abspath(args.current_candidate_json), "contract_json": os.path.abspath(args.current_contract_json), "run_timestamp": current["run_timestamp"]},
        "metric_deltas": metric_deltas,
        "precision": {
            "baseline": baseline_precision,
            "current": current_precision,
        },
        "precision_diffs": precision_diffs,
        "effective_config_diffs": effective_config_diffs,
        "ft_row_config_diffs": ft_row_config_diffs,
        "active_loss_diffs": active_loss_diffs,
        "precompute_means": precompute_means,
        "ghost_rows": ghost_rows,
        "triplet_image_parity": triplet_image_parity,
        "step2_baseline": {"peak_positions": baseline_step2["peaks"], "pred_thr": baseline_step2["pred_thr"], "peak_count": len(baseline_step2["peaks"])},
        "step2_current": {"peak_positions": current_step2["peaks"], "pred_thr": current_step2["pred_thr"], "peak_count": len(current_step2["peaks"])},
        "baseline_first_eval_active": {k: normalize_value(baseline_first_eval.get(k, FT_ROW_DEFAULTS.get(k, 0.0))) for k in ACTIVE_KEYS},
        "current_first_eval_active": {k: normalize_value(current_first_eval.get(k, FT_ROW_DEFAULTS.get(k, 0.0))) for k in ACTIVE_KEYS},
        "first_divergence": first_divergence,
    }
    write_json(args.out_json, payload)
    render_md(args.out_md, payload)
    build_png(args.out_png, payload, baseline_step2, current_step2)


if __name__ == "__main__":
    main()
