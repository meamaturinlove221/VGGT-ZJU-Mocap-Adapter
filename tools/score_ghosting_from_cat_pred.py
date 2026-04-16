import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class GhostMetrics:
    ghost_score: float
    ghost_soft_score: float
    ghost_visual_score: float
    width_ratio: float
    area_ratio: float
    peak_count: int
    center_offset_ratio: float
    mask_area: float
    pred_area: float
    pred_luma_mean: float
    pred_luma_p90: float
    pred_nonblack_ratio_thr008: float
    pred_nonblack_ratio_thr015: float
    fg_pred_luma_mean: float
    fg_pred_nonblack_ratio: float
    fg_pred_contrast: float
    fg_pred_tgt_l1: float
    dark_penalty: float
    collapse_penalty: float


def _bbox_from_binary(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    kk = max(1, int(k))
    if kk <= 1:
        return x
    pad = kk // 2
    kernel = np.ones((kk,), dtype=np.float32) / float(kk)
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _count_peaks(profile: np.ndarray, min_rel: float, min_dist: int) -> int:
    if profile.size < 3:
        return 0
    p = profile.astype(np.float32)
    mx = float(p.max())
    if mx <= 1e-8:
        return 0
    thr = float(max(0.0, min(1.0, min_rel))) * mx
    peaks: List[int] = []
    md = max(1, int(min_dist))
    for i in range(1, p.size - 1):
        if p[i] < thr:
            continue
        if p[i] < p[i - 1] or p[i] < p[i + 1]:
            continue
        if not peaks:
            peaks.append(i)
            continue
        if i - peaks[-1] >= md:
            peaks.append(i)
            continue
        if p[i] > p[peaks[-1]]:
            peaks[-1] = i
    return len(peaks)


def _split_triptych(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = arr.shape[:2]
    p = w // 3
    if p <= 0:
        raise RuntimeError(f"invalid width for triptych: {w}")
    left = arr[:, :p]
    mid = arr[:, p : 2 * p]
    right = arr[:, 2 * p : 3 * p]
    return left, mid, right


def _extract_step(path: str) -> int:
    m = re.search(r"step(\d+)\.png$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return -1


def _rgb_luma(rgb01: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb01[:, :, 0]
        + 0.7152 * rgb01[:, :, 1]
        + 0.0722 * rgb01[:, :, 2]
    )


def _score_one(path: str, peak_min_rel: float) -> GhostMetrics:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"invalid image shape: {tuple(arr.shape)}")
    left, mid, right = _split_triptych(arr)

    left_gray = left.mean(axis=2)
    left_bin = left_gray > 128.0
    bbox_left = _bbox_from_binary(left_bin)
    if bbox_left is None:
        raise RuntimeError("mask panel has no foreground")
    x0, x1, _y0, _y1 = bbox_left
    mask_w = float(max(1, x1 - x0 + 1))
    mask_area = float(left_bin.sum())
    mask_cx = 0.5 * float(x0 + x1)

    g = mid[:, :, 1]
    thr = max(float(np.percentile(g, 84.0)), 10.0)
    mid_bin = g > thr
    bbox_mid = _bbox_from_binary(mid_bin)
    if bbox_mid is None:
        raise RuntimeError("pred panel has no foreground")

    mx0, mx1, _my0, _my1 = bbox_mid
    pred_w = float(max(1, mx1 - mx0 + 1))
    pred_area = float(mid_bin.sum())
    pred_cx = 0.5 * float(mx0 + mx1)

    width_ratio = pred_w / max(1.0, mask_w)
    area_ratio = pred_area / max(1.0, mask_area)
    center_offset_ratio = abs(pred_cx - mask_cx) / max(1.0, mask_w)

    # Luma-based anti-black diagnostics on prediction panel.
    mid_rgb = mid / 255.0
    right_rgb = right / 255.0
    pred_luma = _rgb_luma(mid_rgb)
    pred_luma_mean = float(pred_luma.mean())
    pred_luma_p90 = float(np.percentile(pred_luma, 90.0))
    pred_nonblack_ratio_thr008 = float((pred_luma > 0.08).mean())
    pred_nonblack_ratio_thr015 = float((pred_luma > 0.15).mean())

    fg_mask = left_bin.astype(bool)
    if fg_mask.any():
        fg_pred_luma = pred_luma[fg_mask]
        fg_pred_luma_mean = float(fg_pred_luma.mean())
        fg_pred_nonblack_ratio = float((fg_pred_luma > 0.08).mean())
        fg_pred_contrast = float(np.percentile(fg_pred_luma, 90.0) - np.percentile(fg_pred_luma, 10.0))
        fg_pred_tgt_l1 = float(np.abs(mid_rgb[fg_mask] - right_rgb[fg_mask]).mean())
    else:
        fg_pred_luma_mean = 0.0
        fg_pred_nonblack_ratio = 0.0
        fg_pred_contrast = 0.0
        fg_pred_tgt_l1 = 1.0

    prof = _smooth_1d(mid_bin.sum(axis=0).astype(np.float32), k=11)
    peak_count = int(_count_peaks(prof, min_rel=peak_min_rel, min_dist=max(6, int(mask_w * 0.12))))

    score = 0.0
    score += max(0.0, width_ratio - 1.10) * 1.00
    score += max(0.0, area_ratio - 1.30) * 0.40
    score += max(0.0, float(peak_count - 1)) * 0.60
    score += max(0.0, center_offset_ratio - 0.22) * 0.50

    # Soft score keeps continuity around hard thresholds for ranking/early-stop.
    soft_score = 0.0
    soft_score += max(0.0, width_ratio - 1.00) * 0.80
    soft_score += max(0.0, area_ratio - 1.10) * 0.25
    soft_score += max(0.0, float(peak_count - 1)) * 0.50
    soft_score += max(0.0, center_offset_ratio - 0.10) * 0.35

    # Hard anti-black / anti-collapse penalties.
    dark_penalty = 0.0
    dark_penalty += max(0.0, 0.045 - pred_luma_mean) * 40.0
    dark_penalty += max(0.0, 0.10 - pred_nonblack_ratio_thr008) * 12.0

    collapse_penalty = 0.0
    collapse_penalty += max(0.0, 0.55 - area_ratio) * 6.0
    collapse_penalty += max(0.0, 0.65 - width_ratio) * 6.0

    ghost_visual_score = float(score + dark_penalty + collapse_penalty)

    return GhostMetrics(
        ghost_score=float(score),
        ghost_soft_score=float(soft_score),
        ghost_visual_score=ghost_visual_score,
        width_ratio=float(width_ratio),
        area_ratio=float(area_ratio),
        peak_count=peak_count,
        center_offset_ratio=float(center_offset_ratio),
        mask_area=mask_area,
        pred_area=pred_area,
        pred_luma_mean=pred_luma_mean,
        pred_luma_p90=pred_luma_p90,
        pred_nonblack_ratio_thr008=pred_nonblack_ratio_thr008,
        pred_nonblack_ratio_thr015=pred_nonblack_ratio_thr015,
        fg_pred_luma_mean=fg_pred_luma_mean,
        fg_pred_nonblack_ratio=fg_pred_nonblack_ratio,
        fg_pred_contrast=fg_pred_contrast,
        fg_pred_tgt_l1=fg_pred_tgt_l1,
        dark_penalty=float(dark_penalty),
        collapse_penalty=float(collapse_penalty),
    )


def _parse_input_specs(raw_specs: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in raw_specs:
        s = str(item).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"invalid --input '{s}', expected label=glob")
        label, pattern = s.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError(f"invalid --input '{s}', expected label=glob")
        out.append((label, pattern))
    if not out:
        raise ValueError("no valid --input provided")
    return out


def main() -> None:
    ap = argparse.ArgumentParser("score_ghosting_from_cat_pred")
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeatable: label=glob_pattern",
    )
    ap.add_argument("--out_csv", default="logs/modal_phase5/ghost_score_rows_latest.csv")
    ap.add_argument("--out_summary_csv", default="logs/modal_phase5/ghost_score_summary_latest.csv")
    ap.add_argument("--out_json", default="logs/modal_phase5/ghost_score_latest.json")
    ap.add_argument("--peak_min_rel", type=float, default=0.35)
    args = ap.parse_args()

    specs = _parse_input_specs(args.input)
    rows: List[Dict[str, object]] = []
    for label, pattern in specs:
        files = sorted(glob.glob(pattern))
        for path in files:
            try:
                m = _score_one(path, peak_min_rel=float(args.peak_min_rel))
                rows.append(
                    {
                        "label": label,
                        "path": path.replace("\\", "/"),
                        "step": _extract_step(path),
                        "ghost_score": m.ghost_score,
                        "ghost_soft_score": m.ghost_soft_score,
                        "ghost_visual_score": m.ghost_visual_score,
                        "width_ratio": m.width_ratio,
                        "area_ratio": m.area_ratio,
                        "peak_count": m.peak_count,
                        "center_offset_ratio": m.center_offset_ratio,
                        "mask_area": m.mask_area,
                        "pred_area": m.pred_area,
                        "pred_luma_mean": m.pred_luma_mean,
                        "pred_luma_p90": m.pred_luma_p90,
                        "pred_nonblack_ratio_thr008": m.pred_nonblack_ratio_thr008,
                        "pred_nonblack_ratio_thr015": m.pred_nonblack_ratio_thr015,
                        "fg_pred_luma_mean": m.fg_pred_luma_mean,
                        "fg_pred_nonblack_ratio": m.fg_pred_nonblack_ratio,
                        "fg_pred_contrast": m.fg_pred_contrast,
                        "fg_pred_tgt_l1": m.fg_pred_tgt_l1,
                        "dark_penalty": m.dark_penalty,
                        "collapse_penalty": m.collapse_penalty,
                    }
                )
            except Exception as ex:
                rows.append(
                    {
                        "label": label,
                        "path": path.replace("\\", "/"),
                        "step": _extract_step(path),
                        "ghost_score": 99.0,
                        "ghost_soft_score": 99.0,
                        "ghost_visual_score": 199.0,
                        "width_ratio": 99.0,
                        "area_ratio": 99.0,
                        "peak_count": 0,
                        "center_offset_ratio": 99.0,
                        "mask_area": 0.0,
                        "pred_area": 0.0,
                        "pred_luma_mean": 0.0,
                        "pred_luma_p90": 0.0,
                        "pred_nonblack_ratio_thr008": 0.0,
                        "pred_nonblack_ratio_thr015": 0.0,
                        "fg_pred_luma_mean": 0.0,
                        "fg_pred_nonblack_ratio": 0.0,
                        "fg_pred_contrast": 0.0,
                        "fg_pred_tgt_l1": 1.0,
                        "dark_penalty": 99.0,
                        "collapse_penalty": 99.0,
                        "error": str(ex),
                    }
                )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        fn = [
            "label",
            "path",
            "step",
            "ghost_score",
            "ghost_soft_score",
            "ghost_visual_score",
            "width_ratio",
            "area_ratio",
            "peak_count",
            "center_offset_ratio",
            "mask_area",
            "pred_area",
            "pred_luma_mean",
            "pred_luma_p90",
            "pred_nonblack_ratio_thr008",
            "pred_nonblack_ratio_thr015",
            "fg_pred_luma_mean",
            "fg_pred_nonblack_ratio",
            "fg_pred_contrast",
            "fg_pred_tgt_l1",
            "dark_penalty",
            "collapse_penalty",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_label: Dict[str, List[Tuple[float, float, float, float, float, float, float, float, float, float, float, float]]] = {}
    for r in rows:
        if str(r.get("error", "")).strip():
            continue
        label = str(r["label"])
        by_label.setdefault(label, []).append(
            (
                float(r["ghost_score"]),
                float(r.get("ghost_soft_score", 99.0)),
                float(r.get("ghost_visual_score", 199.0)),
                float(r.get("pred_luma_mean", 0.0)),
                float(r.get("pred_nonblack_ratio_thr008", 0.0)),
                float(r.get("pred_nonblack_ratio_thr015", 0.0)),
                float(r.get("width_ratio", 99.0)),
                float(r.get("area_ratio", 99.0)),
                float(r.get("fg_pred_luma_mean", 0.0)),
                float(r.get("fg_pred_nonblack_ratio", 0.0)),
                float(r.get("fg_pred_contrast", 0.0)),
                float(r.get("fg_pred_tgt_l1", 1.0)),
            )
        )
    summary_rows: List[Dict[str, object]] = []
    for label, vals in by_label.items():
        hard_arr = np.asarray([v[0] for v in vals], dtype=np.float64)
        soft_arr = np.asarray([v[1] for v in vals], dtype=np.float64)
        visual_arr = np.asarray([v[2] for v in vals], dtype=np.float64)
        luma_mean_arr = np.asarray([v[3] for v in vals], dtype=np.float64)
        nonblack_arr = np.asarray([v[4] for v in vals], dtype=np.float64)
        nonblack015_arr = np.asarray([v[5] for v in vals], dtype=np.float64)
        width_ratio_arr = np.asarray([v[6] for v in vals], dtype=np.float64)
        area_ratio_arr = np.asarray([v[7] for v in vals], dtype=np.float64)
        fg_luma_arr = np.asarray([v[8] for v in vals], dtype=np.float64)
        fg_nonblack_arr = np.asarray([v[9] for v in vals], dtype=np.float64)
        fg_contrast_arr = np.asarray([v[10] for v in vals], dtype=np.float64)
        fg_tgt_l1_arr = np.asarray([v[11] for v in vals], dtype=np.float64)
        summary_rows.append(
            {
                "label": label,
                "count": int(hard_arr.size),
                "ghost_score_mean": float(hard_arr.mean()) if hard_arr.size > 0 else 99.0,
                "ghost_score_p95": float(np.percentile(hard_arr, 95.0)) if hard_arr.size > 0 else 99.0,
                "ghost_score_min": float(hard_arr.min()) if hard_arr.size > 0 else 99.0,
                "ghost_soft_score_mean": float(soft_arr.mean()) if soft_arr.size > 0 else 99.0,
                "ghost_soft_score_p95": float(np.percentile(soft_arr, 95.0)) if soft_arr.size > 0 else 99.0,
                "ghost_soft_score_min": float(soft_arr.min()) if soft_arr.size > 0 else 99.0,
                "ghost_visual_score_mean": float(visual_arr.mean()) if visual_arr.size > 0 else 199.0,
                "ghost_visual_score_p95": float(np.percentile(visual_arr, 95.0)) if visual_arr.size > 0 else 199.0,
                "ghost_visual_score_min": float(visual_arr.min()) if visual_arr.size > 0 else 199.0,
                "pred_luma_mean_mean": float(luma_mean_arr.mean()) if luma_mean_arr.size > 0 else 0.0,
                "pred_nonblack_ratio_thr008_mean": float(nonblack_arr.mean()) if nonblack_arr.size > 0 else 0.0,
                "pred_nonblack_ratio_thr015_mean": float(nonblack015_arr.mean()) if nonblack015_arr.size > 0 else 0.0,
                "width_ratio_mean": float(width_ratio_arr.mean()) if width_ratio_arr.size > 0 else 99.0,
                "area_ratio_mean": float(area_ratio_arr.mean()) if area_ratio_arr.size > 0 else 99.0,
                "fg_pred_luma_mean_mean": float(fg_luma_arr.mean()) if fg_luma_arr.size > 0 else 0.0,
                "fg_pred_nonblack_ratio_mean": float(fg_nonblack_arr.mean()) if fg_nonblack_arr.size > 0 else 0.0,
                "fg_pred_contrast_mean": float(fg_contrast_arr.mean()) if fg_contrast_arr.size > 0 else 0.0,
                "fg_pred_tgt_l1_mean": float(fg_tgt_l1_arr.mean()) if fg_tgt_l1_arr.size > 0 else 1.0,
            }
        )
    summary_rows = sorted(summary_rows, key=lambda x: float(x["ghost_visual_score_mean"]))

    with open(args.out_summary_csv, "w", encoding="utf-8", newline="") as f:
        fn = [
            "label",
            "count",
            "ghost_score_mean",
            "ghost_score_p95",
            "ghost_score_min",
            "ghost_soft_score_mean",
            "ghost_soft_score_p95",
            "ghost_soft_score_min",
            "ghost_visual_score_mean",
            "ghost_visual_score_p95",
            "ghost_visual_score_min",
            "pred_luma_mean_mean",
            "pred_nonblack_ratio_thr008_mean",
            "pred_nonblack_ratio_thr015_mean",
            "width_ratio_mean",
            "area_ratio_mean",
            "fg_pred_luma_mean_mean",
            "fg_pred_nonblack_ratio_mean",
            "fg_pred_contrast_mean",
            "fg_pred_tgt_l1_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    payload = {
        "rows_csv": args.out_csv.replace("\\", "/"),
        "summary_csv": args.out_summary_csv.replace("\\", "/"),
        "summary": summary_rows,
        "count_rows": len(rows),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ghost-score] rows={len(rows)}")
    print(f"[ghost-score] summary={args.out_summary_csv}")


if __name__ == "__main__":
    main()
