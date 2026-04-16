import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


PROBE_ORDER = ["T0_smoke", "G0", "S0", "S1", "S2", "S3"]
PROBE_LABELS = {
    "T0_smoke": "T0-smoke",
    "G0": "G0",
    "S0": "S0",
    "S1": "S1",
    "S2": "S2",
    "S3": "S3",
}

MATCH_FIELDS = [
    "precompute_mv_support_on",
    "precompute_mv_support_region_mode",
    "precompute_mv_support_fg_mask_source",
    "precompute_mv_support_fg_erode_px",
    "precompute_mv_support_fg_preserve_px",
    "point_target_mode",
    "point_target_blend_by_mv_support",
    "point_target_blend_mv_region_mode",
    "point_mv_depth_region_mode",
    "use_fg_mask",
    "fg_mask_source",
    "point_support_mode",
    "point_mv_depth_support_mode",
    "point_mv_mask_support_mode",
    "lambda_point_mv_mask",
]

PRIMARY_METRICS = [
    "ghost_score_mean",
    "ghost_visual_score",
    "pred_luma_mean",
    "pred_nonblack_ratio_thr008",
    "fg_pred_luma_mean",
    "fg_pred_nonblack_ratio",
    "fg_pred_contrast",
    "fg_pred_tgt_l1",
]

SUPPORT_METRICS = [
    "mv_support_raw_mean",
    "mv_support_valid_ratio",
    "mv_support_fg_valid_ratio",
    "mv_support_bg_valid_ratio",
    "mv_support_pair_count_eff",
    "mv_support_conf_mean",
    "mv_support_nan_ratio",
    "depth_conf_delta_mean",
    "point_mv_support_mean",
    "point_mv_support_fg_mean",
    "point_mv_support_bg_mean",
    "point_support_eff_mean",
    "point_support_eff_fg_mean",
    "point_support_eff_bg_mean",
    "point_mv_depth_support_eff_mean",
    "point_mv_depth_support_eff_fg_mean",
    "point_mv_depth_support_eff_bg_mean",
    "point_mv_mask_support_eff_mean",
    "point_mv_mask_support_eff_fg_mean",
    "point_mv_mask_support_eff_bg_mean",
    "mv_support_generation_region_mode",
    "mv_support_generation_fg_mask_source",
    "mv_support_fg_mean",
    "mv_support_bg_mean",
    "depth_conf_delta_fg_mean",
    "depth_conf_delta_bg_mean",
    "depth_conf_fg_preserved_active",
    "depth_conf_fg_preserve_px",
    "depth_conf_fg_exact_ratio",
    "depth_conf_fg_preserve_ratio",
    "depth_conf_fg_raw_mean",
    "depth_conf_fg_after_support_mean",
    "depth_conf_fg_final_mean",
]

ACTIVITY_METRICS = [
    "support_generation_active",
    "point_support_path_active",
    "point_mv_depth_support_path_active",
    "point_mv_mask_support_path_active",
    "point_target_blend_mv_support_active",
]


def fill_if_blank(dst: Dict[str, Any], key: str, value: Any) -> None:
    if key not in dst or dst.get(key) in (None, ""):
        dst[key] = value


def infer_activity_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    cand = dict(candidate)
    support_on = normalize_bool_like(cand.get("precompute_mv_support_on")) is True
    point_support_mode = str(cand.get("point_support_mode") or "").strip().lower()
    point_mv_depth_support_mode = str(cand.get("point_mv_depth_support_mode") or "").strip().lower()
    point_mv_mask_support_mode = str(cand.get("point_mv_mask_support_mode") or "").strip().lower()
    blend_by_support = normalize_bool_like(cand.get("point_target_blend_by_mv_support")) is True

    fill_if_blank(cand, "support_generation_active", 1 if support_on else 0)
    fill_if_blank(cand, "point_support_path_active", 1 if point_support_mode not in {"", "off"} else 0)
    fill_if_blank(cand, "point_mv_depth_support_path_active", 1 if point_mv_depth_support_mode not in {"", "off"} else 0)
    fill_if_blank(cand, "point_mv_mask_support_path_active", 1 if point_mv_mask_support_mode not in {"", "off"} else 0)
    fill_if_blank(cand, "point_target_blend_mv_support_active", 1 if blend_by_support else 0)
    return cand


def enrich_candidate(candidate: Dict[str, Any], contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cand = dict(candidate)
    if contract:
        for key in MATCH_FIELDS:
            fill_if_blank(cand, key, contract.get(key))
        for key in (
            "precompute_mv_support_region_mode",
            "precompute_mv_support_fg_mask_source",
            "precompute_mv_support_fg_erode_px",
            "precompute_mv_support_fg_preserve_px",
            "mv_support_generation_region_mode",
            "mv_support_generation_fg_mask_source",
        ):
            fill_if_blank(cand, key, contract.get(key))
    fill_if_blank(cand, "point_support_mode", "off")
    fill_if_blank(cand, "point_mv_depth_support_mode", "off")
    fill_if_blank(cand, "point_mv_mask_support_mode", "off")
    fill_if_blank(cand, "mv_support_generation_region_mode", cand.get("precompute_mv_support_region_mode"))
    fill_if_blank(cand, "mv_support_generation_fg_mask_source", cand.get("precompute_mv_support_fg_mask_source"))
    return infer_activity_fields(cand)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_bool_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes", "y", "on"}:
        return True
    if raw in {"false", "0", "no", "n", "off"}:
        return False
    return str(value).strip()


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        raw = str(value).strip()
        if not raw or raw.lower() == "nan":
            return None
        try:
            val = float(raw)
        except Exception:
            return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def parse_contract_time(contract: Dict[str, Any]) -> Optional[datetime]:
    for key in ("generated_at", "updated_at"):
        raw = contract.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            continue
    return None


def parse_candidate_time(candidate: Dict[str, Any]) -> Optional[datetime]:
    run_ts = candidate.get("run_timestamp")
    if run_ts:
        try:
            return datetime.strptime(str(run_ts), "%Y%m%d_%H%M%S")
        except Exception:
            pass
    updated = candidate.get("updated_at")
    if updated:
        try:
            return datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def get_latest_contracts(status_dir: str) -> Dict[str, str]:
    latest: Dict[str, Tuple[float, str]] = {}
    for name in os.listdir(status_dir):
        if not name.startswith("probe_contract_") or not name.endswith(".json") or name == "probe_contract_latest.json":
            continue
        path = os.path.join(status_dir, name)
        data = read_json(path)
        if not data:
            continue
        probe_id = str(data.get("probe_id") or "")
        if not probe_id:
            continue
        score = os.path.getmtime(path)
        prev = latest.get(probe_id)
        if prev is None or score > prev[0]:
            latest[probe_id] = (score, path)
    return {k: v[1] for k, v in latest.items()}


def load_candidate_pool(repo_dir: str, status_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name in os.listdir(status_dir):
        if not (name.startswith("candidate_result_") and name.endswith(".json")):
            continue
        if name == "candidate_result_latest.json":
            continue
        path = os.path.join(status_dir, name)
        data = read_json(path)
        if data:
            if not data.get("best_visual_png"):
                best_visual = infer_best_visual_from_rows(data, repo_dir, status_dir)
                if best_visual:
                    data["best_visual_png"] = best_visual
            out.append((path, data))
    return out


def infer_best_visual_from_rows(candidate: Dict[str, Any], repo_dir: str, status_dir: str) -> Optional[str]:
    rows_rel = candidate.get("ghost_rows_csv")
    if not rows_rel:
        return None
    rows_rel = str(rows_rel)
    rows_path = rows_rel if os.path.isabs(rows_rel) else os.path.join(repo_dir, rows_rel)
    if not os.path.exists(rows_path):
        rows_path = os.path.join(status_dir, os.path.basename(rows_rel))
    if not os.path.exists(rows_path):
        return None
    best_path = None
    best_score = None
    try:
        with open(rows_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = to_float(row.get("ghost_visual_score"))
                path = row.get("path") or row.get("best_visual_png")
                if not path:
                    continue
                if best_score is None or (score is not None and score < best_score):
                    best_score = score if score is not None else best_score
                    best_path = path
    except Exception:
        return None
    return best_path


def match_candidate(contract: Dict[str, Any], pool: List[Tuple[str, Dict[str, Any]]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    contract_time = parse_contract_time(contract)
    best: Optional[Tuple[int, int, float, str, Dict[str, Any]]] = None
    expected: Dict[str, Any] = {}
    for key in MATCH_FIELDS:
        if key in contract and contract.get(key) not in (None, ""):
            expected[key] = normalize_bool_like(contract.get(key))
    for path, cand in pool:
        score = 0
        total = 0
        for key, exp in expected.items():
            total += 1
            act = normalize_bool_like(cand.get(key))
            if act == exp:
                score += 1
        cand_time = parse_candidate_time(cand)
        time_delta = 1e12
        if contract_time and cand_time:
            ct = contract_time.replace(tzinfo=None) if contract_time.tzinfo else contract_time
            rt = cand_time.replace(tzinfo=None) if cand_time.tzinfo else cand_time
            time_delta = abs((rt - ct).total_seconds())
        if total <= 0:
            continue
        # Prefer candidates produced by the same probe window even if a few fields
        # are still blank in the stamped candidate json. This avoids an older probe
        # with a fully backfilled contract winning over the newest real run.
        exact_time_match = 1 if time_delta <= 300 else 0
        row = (exact_time_match, score, -time_delta, path, cand)
        if best is None or row > best:
            best = row
    if best is None:
        return None
    if best[1] <= 0:
        return None
    # Treat a contract as resolved only when every explicitly expected field matches.
    # This prevents readiness-only contracts such as G0 dry-runs from being shown as if
    # they were live probe results just because an older candidate matched most fields.
    if expected and best[0] <= 0 and best[1] < len(expected):
        return None
    return best[3], best[4]


def fmt_num(value: Any, digits: int = 4) -> str:
    num = to_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def delta_label(delta: Optional[float], lower_is_better: bool = True) -> str:
    if delta is None:
        return ""
    if abs(delta) < 1e-12:
        return "same"
    better = delta < 0 if lower_is_better else delta > 0
    return "better" if better else "worse"


def render_text_png(lines: List[str], out_path: str, width: int = 1600, padding: int = 24, line_h: int = 26) -> None:
    font = ImageFont.load_default()
    height = padding * 2 + line_h * max(1, len(lines))
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    img.save(out_path)


def render_visual_grid(probes: List[Tuple[str, Dict[str, Any]]], repo_dir: str, out_path: str) -> None:
    items = []
    for probe_id, cand in probes:
        rel = cand.get("best_visual_png")
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(repo_dir, rel)
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        items.append((PROBE_LABELS.get(probe_id, probe_id), img))
    if not items:
        return
    target_w = 900
    thumb_h = []
    for _, img in items:
        w, h = img.size
        nh = max(1, int(h * (target_w / float(w))))
        thumb_h.append(nh)
    cols = 2 if len(items) > 1 else 1
    rows = (len(items) + cols - 1) // cols
    cell_w = target_w
    label_h = 28
    pad = 16
    cell_h = max(h + label_h for h in thumb_h)
    canvas = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + pad)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (label, img) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        w, h = img.size
        nh = max(1, int(h * (target_w / float(w))))
        thumb = img.resize((target_w, nh))
        canvas.paste(thumb, (x, y + label_h))
    canvas.save(out_path)


def build_markdown(
    probes: List[Tuple[str, Dict[str, Any]]],
    unresolved: List[Tuple[str, str, Optional[Dict[str, Any]]]],
) -> str:
    lines: List[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"# Support Probe Summary ({today})")
    lines.append("")
    if not probes:
        lines.append("No resolved probe candidates.")
        if unresolved:
            lines.append("")
            lines.append("## Readiness-only Contracts")
            lines.append("")
            for probe_id, contract_path, contract in unresolved:
                region_mode = ""
                if contract:
                    region_mode = str(contract.get("precompute_mv_support_region_mode") or "")
                detail = f", region={region_mode}" if region_mode else ""
                lines.append(f"- {PROBE_LABELS.get(probe_id, probe_id)}: no live candidate matched `{contract_path}`{detail}")
        return "\n".join(lines) + "\n"
    baseline_id, baseline = probes[0]
    lines.append("## Available Probes")
    lines.append("")
    for probe_id, cand in probes:
        lines.append(
            f"- {PROBE_LABELS.get(probe_id, probe_id)}: "
            f"ghost_visual={fmt_num(cand.get('ghost_visual_score'))}, "
            f"fg_luma={fmt_num(cand.get('fg_pred_luma_mean'))}, "
            f"fg_contrast={fmt_num(cand.get('fg_pred_contrast'))}, "
            f"invalid={cand.get('candidate_invalid_reason','')}"
        )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Probe | ghost | ghost_visual | pred_luma | nonblack@008 | fg_luma | fg_nonblack | fg_contrast | fg_tgt_l1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for probe_id, cand in probes:
        lines.append(
            f"| {PROBE_LABELS.get(probe_id, probe_id)} | "
            f"{fmt_num(cand.get('ghost_score_mean'))} | "
            f"{fmt_num(cand.get('ghost_visual_score'))} | "
            f"{fmt_num(cand.get('pred_luma_mean'))} | "
            f"{fmt_num(cand.get('pred_nonblack_ratio_thr008'))} | "
            f"{fmt_num(cand.get('fg_pred_luma_mean'))} | "
            f"{fmt_num(cand.get('fg_pred_nonblack_ratio'))} | "
            f"{fmt_num(cand.get('fg_pred_contrast'))} | "
            f"{fmt_num(cand.get('fg_pred_tgt_l1'))} |"
        )
    lines.append("")
    if len(probes) > 1:
        lines.append(f"## Delta vs {PROBE_LABELS.get(baseline_id, baseline_id)}")
        lines.append("")
        for probe_id, cand in probes[1:]:
            lines.append(f"### {PROBE_LABELS.get(probe_id, probe_id)}")
            lines.append("")
            lines.append("| metric | delta | interpretation |")
            lines.append("|---|---:|---|")
            for key, lower_is_better in [
                ("ghost_score_mean", True),
                ("ghost_visual_score", True),
                ("pred_luma_mean", False),
                ("pred_nonblack_ratio_thr008", False),
                ("fg_pred_luma_mean", False),
                ("fg_pred_nonblack_ratio", False),
                ("fg_pred_contrast", False),
                ("fg_pred_tgt_l1", True),
            ]:
                b = to_float(baseline.get(key))
                c = to_float(cand.get(key))
                delta = None if b is None or c is None else c - b
                lines.append(f"| {key} | {fmt_num(delta)} | {delta_label(delta, lower_is_better)} |")
            lines.append("")
    present_support = [
        name for name in SUPPORT_METRICS
        if any(to_float(c.get(name)) is not None for _, c in probes)
    ]
    if present_support:
        lines.append("## Support Stats")
        lines.append("")
        header = "| Probe | " + " | ".join(present_support) + " |"
        sep = "|" + "---|" * (len(present_support) + 1)
        lines.append(header)
        lines.append(sep)
        for probe_id, cand in probes:
            row = "| " + PROBE_LABELS.get(probe_id, probe_id) + " | " + " | ".join(fmt_num(cand.get(k)) for k in present_support) + " |"
            lines.append(row)
        lines.append("")
    present_activity = [
        name for name in ACTIVITY_METRICS
        if any(to_float(c.get(name)) is not None for _, c in probes)
    ]
    if present_activity:
        lines.append("## Support Path Activity")
        lines.append("")
        header = "| Probe | " + " | ".join(present_activity) + " |"
        sep = "|" + "---|" * (len(present_activity) + 1)
        lines.append(header)
        lines.append(sep)
        for probe_id, cand in probes:
            row = "| " + PROBE_LABELS.get(probe_id, probe_id) + " | " + " | ".join(fmt_num(cand.get(k), 1) for k in present_activity) + " |"
            lines.append(row)
        lines.append("")
    lines.append("## Contracts")
    lines.append("")
    for probe_id, cand in probes:
        lines.append(
            f"- {PROBE_LABELS.get(probe_id, probe_id)}: "
            f"`precompute_mv_support_on={cand.get('precompute_mv_support_on','')}`, "
            f"`point_support_mode={cand.get('point_support_mode','')}`, "
            f"`point_mv_depth_support_mode={cand.get('point_mv_depth_support_mode','')}`, "
            f"`point_mv_mask_support_mode={cand.get('point_mv_mask_support_mode','')}`, "
            f"`point_target_mode={cand.get('point_target_mode','')}`, "
            f"`point_target_blend_by_mv_support={cand.get('point_target_blend_by_mv_support','')}`"
        )
    lines.append("")
    lines.append("## Key Paths")
    lines.append("")
    for probe_id, cand in probes:
        lines.append(f"- {PROBE_LABELS.get(probe_id, probe_id)} candidate json: `{cand.get('candidate_result_json', '')}`")
        lines.append(f"- {PROBE_LABELS.get(probe_id, probe_id)} visual: `{cand.get('best_visual_png', '')}`")
    lines.append("")
    if unresolved:
        lines.append("## Readiness-only Contracts")
        lines.append("")
        for probe_id, contract_path, contract in unresolved:
            region_mode = ""
            if contract:
                region_mode = str(contract.get("precompute_mv_support_region_mode") or "")
            detail = f", region={region_mode}" if region_mode else ""
            lines.append(f"- {PROBE_LABELS.get(probe_id, probe_id)}: no live candidate matched `{contract_path}`{detail}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="F:\\vggt")
    ap.add_argument("--status-dir", default="logs/modal_phase5")
    ap.add_argument("--out-md", default="logs/modal_phase5/human_probe_summary_latest.md")
    ap.add_argument("--out-png", default="logs/modal_phase5/human_probe_summary_latest.png")
    ap.add_argument("--out-grid", default="logs/modal_phase5/human_probe_visual_grid_latest.png")
    ap.add_argument("--probe-order", default=",".join(PROBE_ORDER))
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    status_dir = args.status_dir if os.path.isabs(args.status_dir) else os.path.join(repo_dir, args.status_dir)
    contract_map = get_latest_contracts(status_dir)
    pool = load_candidate_pool(repo_dir, status_dir)

    resolved: List[Tuple[str, Dict[str, Any]]] = []
    unresolved: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    for probe_id in [p.strip() for p in args.probe_order.split(",") if p.strip()]:
        contract_path = contract_map.get(probe_id)
        if not contract_path:
            continue
        contract = read_json(contract_path)
        if not contract:
            continue
        matched = match_candidate(contract, pool)
        if not matched:
            unresolved.append((probe_id, contract_path, contract))
            continue
        _, cand = matched
        resolved.append((probe_id, enrich_candidate(cand, contract)))

    md = build_markdown(resolved, unresolved)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(repo_dir, args.out_md)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(repo_dir, args.out_png)
    out_grid = args.out_grid if os.path.isabs(args.out_grid) else os.path.join(repo_dir, args.out_grid)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    text_lines = [line for line in md.splitlines()]
    render_text_png(text_lines, out_png)
    render_visual_grid(resolved, repo_dir, out_grid)
    print(f"[render-support-probe-summary] probes={len(resolved)} md={out_md} png={out_png} grid={out_grid}")


if __name__ == "__main__":
    main()
