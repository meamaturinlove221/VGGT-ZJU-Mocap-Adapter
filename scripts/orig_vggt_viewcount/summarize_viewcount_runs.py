from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_viewcount.common import write_json, write_text  # noqa: E402


def _parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --run {item!r}, expected label=path")
        label, path = item.split("=", 1)
        p = Path(path.strip())
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        out.append((label.strip(), p))
    return out


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ghost_row(run_dir: Path) -> dict:
    rows_csv = run_dir / "ghost_score_rows.csv"
    if rows_csv.is_file():
        with rows_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            out: dict[str, float | int | str] = {}
            for key, value in row.items():
                if key in {"label", "path", "error"}:
                    out[key] = value
                    continue
                try:
                    if key == "peak_count":
                        out[key] = int(float(value))
                    else:
                        out[key] = float(value)
                except Exception:
                    out[key] = value
            return out
    summary_json = run_dir / "ghost_score.json"
    if summary_json.is_file():
        payload = _load_json(summary_json)
        summary = payload.get("summary", [])
        if summary:
            return dict(summary[0])
    return {}


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser("summarize_viewcount_runs")
    ap.add_argument("--run", action="append", default=[], help="Repeat: label=run_dir")
    ap.add_argument("--taxonomy_json", default="")
    ap.add_argument("--out_json", default="logs/modal_phase5/reports/orig_vggt_viewcount_summary_latest.json")
    ap.add_argument("--out_csv", default="logs/modal_phase5/reports/orig_vggt_viewcount_summary_latest.csv")
    ap.add_argument("--out_md", default="logs/modal_phase5/reports/orig_vggt_viewcount_summary_latest.md")
    args = ap.parse_args()

    runs = _parse_runs(args.run)
    rows: list[dict] = []
    for label, run_dir in runs:
        report = _load_json(run_dir / "report.json")
        ghost = _load_ghost_row(run_dir)
        meta = report.get("meta", {})
        render = report.get("render", {})
        metrics_native = report.get("metrics", {}).get("native", {})
        metrics_1024 = report.get("metrics", {}).get("upsampled_1024", {})
        sim3 = report.get("sim3", {})
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "view_profile": meta.get("view_profile", label),
                "tgt_camera": meta.get("tgt_camera", ""),
                "frame_id": int(meta.get("frame_id", -1)),
                "num_total_cams": int(meta.get("num_total_cams", 0)),
                "num_src_views_actual": int(meta.get("num_src_views_actual", 0)),
                "coverage_ratio": _to_float(render.get("coverage_ratio")),
                "mean_conf": _to_float(render.get("mean_conf")),
                "valid_contrib": int(render.get("valid_contrib", 0)),
                "native_psnr": _to_float(metrics_native.get("psnr")),
                "native_ssim": _to_float(metrics_native.get("ssim")),
                "native_mae": _to_float(metrics_native.get("mae")),
                "upsampled_psnr": _to_float(metrics_1024.get("psnr")),
                "upsampled_ssim": _to_float(metrics_1024.get("ssim")),
                "upsampled_mae": _to_float(metrics_1024.get("mae")),
                "sim3_rmse_after": _to_float(sim3.get("src_center_rmse_after")),
                "elapsed_sec_total": _to_float(report.get("elapsed_sec_total")),
                "cuda_peak_mem_mb": _to_float(meta.get("cuda_peak_mem_mb")),
                "ghost_visual_score": _to_float(ghost.get("ghost_visual_score", ghost.get("ghost_visual_score_mean"))),
                "ghost_score": _to_float(ghost.get("ghost_score", ghost.get("ghost_score_mean"))),
                "peak_count": int(float(ghost.get("peak_count", 0))) if ghost else 0,
                "width_ratio": _to_float(ghost.get("width_ratio", ghost.get("width_ratio_mean"))),
                "area_ratio": _to_float(ghost.get("area_ratio", ghost.get("area_ratio_mean"))),
                "pred_luma_mean": _to_float(ghost.get("pred_luma_mean", ghost.get("pred_luma_mean_mean"))),
            }
        )

    rows.sort(key=lambda row: row["num_src_views_actual"])
    taxonomy = {}
    if str(args.taxonomy_json).strip():
        taxonomy_path = Path(args.taxonomy_json)
        if not taxonomy_path.is_absolute():
            taxonomy_path = (REPO_ROOT / taxonomy_path).resolve()
        if taxonomy_path.is_file():
            taxonomy = _load_json(taxonomy_path)

    trend = "insufficient_data"
    if len(rows) >= 2:
        ghost_vals = [row["ghost_visual_score"] for row in rows]
        cov_vals = [row["coverage_ratio"] for row in rows]
        if all(ghost_vals[idx] <= ghost_vals[idx - 1] for idx in range(1, len(ghost_vals))):
            trend = "ghost_improves_monotonically"
        elif all(ghost_vals[idx] >= ghost_vals[idx - 1] for idx in range(1, len(ghost_vals))):
            trend = "ghost_worsens_monotonically"
        elif all(cov_vals[idx] >= cov_vals[idx - 1] for idx in range(1, len(cov_vals))):
            trend = "support_coverage_rises_but_ghost_mixed"
        else:
            trend = "mixed"

    payload = {
        "rows": rows,
        "trend_view_count_vs_quality": trend,
        "taxonomy_conclusion": taxonomy.get("conclusions", {}),
    }
    write_json(args.out_json, payload)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    md_lines = [
        "# Original VGGT Viewcount Summary",
        "",
        "## Key Result",
        "",
        f"- target camera is fixed to `{rows[0]['tgt_camera']}` across all runs." if rows else "- no runs loaded.",
        f"- trend_view_count_vs_quality: `{trend}`",
    ]
    if taxonomy:
        md_lines.append(
            "- taxonomy: `cat_fg_mask_pred_tgt*.png` is not point cloud output; `cat_weight_pred_tgt*.png` is the point-support visualization."
        )
    md_lines.extend(["", "## Geometry Support Layer", ""])
    for row in rows:
        md_lines.append(
            f"- `{row['label']}` src_views=`{row['num_src_views_actual']}` coverage=`{row['coverage_ratio']:.6f}` mean_conf=`{row['mean_conf']:.6f}` valid_contrib=`{row['valid_contrib']}`"
        )
    md_lines.extend(["", "## Final Rendering Layer", ""])
    for row in rows:
        md_lines.append(
            f"- `{row['label']}` ghost_visual=`{row['ghost_visual_score']:.6f}` peak_count=`{row['peak_count']}` psnr=`{row['native_psnr']:.6f}` ssim=`{row['native_ssim']:.6f}`"
        )
    write_text(args.out_md, "\n".join(md_lines) + "\n")
    print(f"[viewcount-summary] wrote {args.out_json}")


if __name__ == "__main__":
    main()
