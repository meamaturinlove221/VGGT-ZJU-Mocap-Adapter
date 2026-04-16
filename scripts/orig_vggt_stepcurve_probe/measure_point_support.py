from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_stepcurve_probe.common import (  # noqa: E402
    DEFAULT_ZJU_ROOT,
    DEFAULT_SUPPORT_PEAK_MIN_REL,
    DEFAULT_SUPPORT_PEAK_SMOOTH_K,
    DEFAULT_SUPPORT_THRESHOLD_FLOOR,
    fmt_num,
    infer_mask_path,
    load_json,
    normalize_path,
    resolve_local_zju_path,
    to_float,
    write_json,
)
from tools.score_ghosting_from_cat_pred import _count_peaks, _smooth_1d  # noqa: E402


def _normalize_mask01(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(arr.max()) if arr.size > 0 else 0.0
    if maxv <= 1.5:
        pass
    elif maxv > 0.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _weight_support_map(compare_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = compare_dir / "weight_native.png"
    with Image.open(str(path)).convert("RGB") as img:
        rgb = np.asarray(img, dtype=np.float32) / 255.0
    return rgb, rgb[:, :, 1]


def _split_triptych_left_mask(compare_dir: Path) -> np.ndarray:
    path = compare_dir / "cat_fg_mask_pred_tgt_step000000.png"
    with Image.open(str(path)).convert("RGB") as img:
        arr = np.asarray(img, dtype=np.float32) / 255.0
    panel_w = arr.shape[1] // 3
    left = arr[:, :panel_w, :]
    return _normalize_mask01(left.mean(axis=2))


def _load_subject_mask(compare_dir: Path, local_zju_root: Path) -> tuple[np.ndarray, str]:
    report = load_json(compare_dir / "report.json")
    meta = report.get("meta", {})
    candidates: list[Path] = []
    tgt_mask_path = str(meta.get("tgt_mask_path", "")).strip()
    if tgt_mask_path:
        candidates.append(resolve_local_zju_path(tgt_mask_path, local_zju_root))
    tgt_img_path = str(meta.get("tgt_image_path", "")).strip()
    if tgt_img_path:
        local_img = resolve_local_zju_path(tgt_img_path, local_zju_root)
        guessed = infer_mask_path(local_img)
        if guessed is not None:
            candidates.append(guessed)

    for cand in candidates:
        if cand.is_file():
            with Image.open(str(cand)).convert("L") as img:
                mask = _normalize_mask01(np.asarray(img, dtype=np.float32))
            return mask, f"gt_mask:{normalize_path(cand)}"

    return _split_triptych_left_mask(compare_dir), "ghost_triplet_left_panel"


def _resize01(mask01: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(mask01, dtype=np.float32),
        (int(target_hw[1]), int(target_hw[0])),
        interpolation=cv2.INTER_NEAREST,
    )


def _bbox_from_binary(mask01: np.ndarray) -> list[int]:
    ys, xs = np.where(mask01 > 0)
    if xs.size <= 0 or ys.size <= 0:
        raise RuntimeError("subject mask is empty after resize")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _draw_bbox(src: np.ndarray, bbox_xyxy: list[int], note_lines: list[str]) -> np.ndarray:
    img = Image.fromarray(np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox_xyxy
    draw.rectangle((x0, y0, x1, y1), outline=(255, 48, 48), width=3)
    tx = max(6, x0 + 4)
    ty = max(6, y0 + 4)
    for idx, line in enumerate(note_lines):
        draw.text((tx, ty + idx * 14), str(line), fill=(255, 255, 255))
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_rgb01(path: Path) -> np.ndarray:
    with Image.open(str(path)).convert("RGB") as img:
        return np.asarray(img, dtype=np.float32) / 255.0


def _masked_error_metrics(pred01: np.ndarray, tgt01: np.ndarray, mask01: np.ndarray) -> tuple[float, float]:
    if pred01.shape != tgt01.shape:
        raise RuntimeError(f"pred/tgt shape mismatch: {pred01.shape} vs {tgt01.shape}")
    if pred01.shape[:2] != mask01.shape[:2]:
        raise RuntimeError(f"mask/image shape mismatch: {mask01.shape} vs {pred01.shape[:2]}")
    mask = np.clip(np.asarray(mask01, dtype=np.float32), 0.0, 1.0)
    if mask.ndim != 2:
        raise RuntimeError(f"mask must be HxW, got {mask.shape}")
    denom = float(mask.sum()) * float(pred01.shape[2])
    if denom <= 1e-8:
        raise RuntimeError("masked metric denominator is zero")
    diff = pred01 - tgt01
    mse = float(((diff * diff) * mask[:, :, None]).sum() / denom)
    mae = float((np.abs(diff) * mask[:, :, None]).sum() / denom)
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    return psnr, mae


def main() -> None:
    ap = argparse.ArgumentParser("measure_point_support")
    ap.add_argument("--compare_dir", required=True)
    ap.add_argument("--local_zju_root", default=str(DEFAULT_ZJU_ROOT))
    ap.add_argument("--support_threshold_abs", default="auto")
    ap.add_argument("--support_peak_min_rel", type=float, default=float(DEFAULT_SUPPORT_PEAK_MIN_REL))
    ap.add_argument("--support_peak_smooth_k", type=int, default=int(DEFAULT_SUPPORT_PEAK_SMOOTH_K))
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_weight_overlay", default="")
    ap.add_argument("--out_cat_overlay", default="")
    args = ap.parse_args()

    compare_dir = Path(args.compare_dir)
    if not compare_dir.is_absolute():
        compare_dir = (REPO_ROOT / compare_dir).resolve()
    local_zju_root = Path(args.local_zju_root)
    if not local_zju_root.is_absolute():
        local_zju_root = (REPO_ROOT / local_zju_root).resolve()

    weight_rgb, support_map01 = _weight_support_map(compare_dir)
    subject_mask_full, mask_source = _load_subject_mask(compare_dir, local_zju_root)
    subject_mask01 = _resize01(subject_mask_full, support_map01.shape[:2])
    subject_mask_bin = (subject_mask01 > 0.5).astype(np.uint8)
    bbox_xyxy = _bbox_from_binary(subject_mask_bin)
    pred_native = _load_rgb01(compare_dir / "pred_native.png")
    tgt_native = _load_rgb01(compare_dir / "tgt_native.png")
    masked_psnr, masked_mae = _masked_error_metrics(pred_native, tgt_native, subject_mask_bin.astype(np.float32))

    threshold_raw = str(args.support_threshold_abs).strip().lower()
    if threshold_raw in {"", "auto"}:
        nz = support_map01[support_map01 > 0]
        q75 = float(np.quantile(nz, 0.75)) if nz.size > 0 else 0.0
        support_threshold_abs = max(float(DEFAULT_SUPPORT_THRESHOLD_FLOOR), q75)
    else:
        support_threshold_abs = float(args.support_threshold_abs)

    support_total_mass = float(support_map01.sum())
    subject_support_mass = float((support_map01 * subject_mask_bin).sum())
    outside_subject_support_mass = float((support_map01 * (1.0 - subject_mask_bin)).sum())

    active = (support_map01 >= float(support_threshold_abs)).astype(np.uint8)
    active_support_mass = float((support_map01 * active).sum())
    num_labels, labels, _stats, _cent = cv2.connectedComponentsWithStats(active, connectivity=8)
    component_masses: list[float] = []
    for idx in range(1, int(num_labels)):
        component_masses.append(float((support_map01 * (labels == idx)).sum()))
    component_masses = sorted(component_masses, reverse=True)
    largest_component_share = (
        float(component_masses[0] / max(active_support_mass, 1e-8)) if component_masses else 0.0
    )
    secondary_component_mass = (
        float(component_masses[1] / max(active_support_mass, 1e-8)) if len(component_masses) > 1 else 0.0
    )

    x0, y0, x1, y1 = bbox_xyxy
    bbox_mask = np.zeros_like(subject_mask_bin, dtype=np.uint8)
    bbox_mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    active_roi = (active * bbox_mask).astype(np.uint8)
    profile_x = active_roi[y0 : y1 + 1, x0 : x1 + 1].sum(axis=0).astype(np.float32)
    profile_x = _smooth_1d(profile_x, k=max(1, int(args.support_peak_smooth_k)))
    support_peak_count = int(
        _count_peaks(
            profile_x,
            min_rel=float(args.support_peak_min_rel),
            min_dist=max(6, int(max(1, x1 - x0 + 1) * 0.12)),
        )
    )

    metrics = {
        "support_source": "weight_native_green_channel",
        "support_normalization_note": "relative_normalized_shape_metric",
        "mask_source_resolved": str(mask_source),
        "support_threshold_abs": float(support_threshold_abs),
        "subject_support_share": float(subject_support_mass / max(support_total_mass, 1e-8)),
        "outside_subject_support_share": float(outside_subject_support_mass / max(support_total_mass, 1e-8)),
        "largest_component_share": float(largest_component_share),
        "secondary_component_mass": float(secondary_component_mass),
        "support_peak_count": int(support_peak_count),
        "component_count": int(len(component_masses)),
        "roi_bbox_xyxy": [int(v) for v in bbox_xyxy],
        "subject_mask_coverage_ratio": float(subject_mask_bin.mean()),
        "subject_psnr": float(masked_psnr),
        "subject_l1": float(masked_mae),
        "masked_psnr": float(masked_psnr),
        "masked_mae": float(masked_mae),
        "support_total_mass": float(support_total_mass),
        "support_active_mass": float(active_support_mass),
    }

    note_lines = [
        f"pk={metrics['support_peak_count']}",
        f"lc={fmt_num(metrics['largest_component_share'], 3)}",
        f"sc={fmt_num(metrics['secondary_component_mass'], 3)}",
    ]

    weight_overlay_path = Path(args.out_weight_overlay) if str(args.out_weight_overlay).strip() else (compare_dir / "weight_native_subject_bbox.png")
    cat_overlay_path = Path(args.out_cat_overlay) if str(args.out_cat_overlay).strip() else (compare_dir / "cat_weight_pred_tgt_subject_bbox.png")
    out_json = Path(args.out_json) if str(args.out_json).strip() else (compare_dir / "point_support_metrics.json")
    if not weight_overlay_path.is_absolute():
        weight_overlay_path = (REPO_ROOT / weight_overlay_path).resolve()
    if not cat_overlay_path.is_absolute():
        cat_overlay_path = (REPO_ROOT / cat_overlay_path).resolve()
    if not out_json.is_absolute():
        out_json = (REPO_ROOT / out_json).resolve()
    weight_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cat_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    weight_overlay = _draw_bbox(weight_rgb, bbox_xyxy, note_lines)
    Image.fromarray(np.clip(weight_overlay * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(str(weight_overlay_path))

    with Image.open(str(compare_dir / "cat_weight_pred_tgt.png")).convert("RGB") as img:
        cat = np.asarray(img, dtype=np.float32) / 255.0
    panel_w = cat.shape[1] // 3
    sx = float(panel_w) / float(max(1, support_map01.shape[1]))
    sy = float(cat.shape[0]) / float(max(1, support_map01.shape[0]))
    cat_bbox = [
        int(round(bbox_xyxy[0] * sx)),
        int(round(bbox_xyxy[1] * sy)),
        int(round(bbox_xyxy[2] * sx)),
        int(round(bbox_xyxy[3] * sy)),
    ]
    cat_overlay = cat.copy()
    cat_overlay[:, :panel_w, :] = _draw_bbox(cat[:, :panel_w, :], cat_bbox, note_lines)
    Image.fromarray(np.clip(cat_overlay * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(str(cat_overlay_path))

    write_json(out_json, metrics)
    print(json.dumps({"out_json": str(out_json), "support_peak_count": int(metrics["support_peak_count"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
