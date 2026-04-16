from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_viewcount.common import write_json, write_text  # noqa: E402


def _classify(path: Path, width: int, height: int) -> dict:
    name = path.name.lower()
    if "cat_weight_pred_tgt" in name:
        return {
            "class_name": "point_support_triplet",
            "is_point_support": True,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "support_weight",
        }
    if name == "weight_native.png":
        return {
            "class_name": "point_support_panel",
            "is_point_support": True,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "support_weight",
        }
    if "cat_fg_mask_pred_tgt" in name:
        return {
            "class_name": "ghost_triplet",
            "is_point_support": False,
            "is_ghost_metric_input": True,
            "left_panel_semantics": "fg_mask",
        }
    if "cat_pred_tgt" in name:
        return {
            "class_name": "pred_tgt_compare",
            "is_point_support": False,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "pred",
        }
    if "gt_with_fg_overlay" in name:
        return {
            "class_name": "gt_overlay",
            "is_point_support": False,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "tgt_with_fg_overlay",
        }
    if ("report" in name or "summary" in name or "board" in name) and width >= max(1600, 2 * height):
        return {
            "class_name": "report_board",
            "is_point_support": False,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "mixed_report",
        }
    if width >= max(2200, 2 * height):
        return {
            "class_name": "report_board",
            "is_point_support": False,
            "is_ghost_metric_input": False,
            "left_panel_semantics": "mixed_report",
        }
    return {
        "class_name": "other_png",
        "is_point_support": False,
        "is_ghost_metric_input": False,
        "left_panel_semantics": "unknown",
    }


def _collect_references(reports_dir: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for ext in ("*.md", "*.json", "*.csv", "*.txt"):
        for path in reports_dir.rglob(ext):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for match in re.findall(r"([A-Za-z0-9_.-]+\.png)", text):
                refs.setdefault(match, []).append(str(path.relative_to(reports_dir)).replace("\\", "/"))
    return refs


def main() -> None:
    ap = argparse.ArgumentParser("audit_visual_taxonomy")
    ap.add_argument("--reports_dir", default="logs/modal_phase5/reports")
    ap.add_argument("--out_json", default="logs/modal_phase5/reports/visual_taxonomy_latest.json")
    ap.add_argument("--out_csv", default="logs/modal_phase5/reports/visual_taxonomy_latest.csv")
    ap.add_argument("--out_md", default="logs/modal_phase5/reports/visual_taxonomy_latest.md")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (REPO_ROOT / reports_dir).resolve()
    refs = _collect_references(reports_dir)
    rows: list[dict] = []
    for path in sorted(reports_dir.rglob("*.png")):
        try:
            with Image.open(str(path)) as img:
                width, height = img.size
            cls = _classify(path, width=width, height=height)
            error = ""
        except Exception as exc:
            width, height = -1, -1
            cls = {
                "class_name": "unreadable_png",
                "is_point_support": False,
                "is_ghost_metric_input": False,
                "left_panel_semantics": "unknown",
            }
            error = str(exc)
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(reports_dir)).replace("\\", "/"),
                "width": width,
                "height": height,
                "class_name": cls["class_name"],
                "is_point_support": cls["is_point_support"],
                "is_ghost_metric_input": cls["is_ghost_metric_input"],
                "left_panel_semantics": cls["left_panel_semantics"],
                "referenced_by_report": refs.get(path.name, []),
                "error": error,
            }
        )

    counts = Counter(row["class_name"] for row in rows)
    ghost_rows = [row for row in rows if row["class_name"] == "ghost_triplet"]
    point_rows = [row for row in rows if row["is_point_support"]]
    payload = {
        "reports_dir": str(reports_dir),
        "count_png": len(rows),
        "count_by_class": dict(counts),
        "conclusions": {
            "ghost_triplets_are_point_clouds": False,
            "ghost_triplets_description": "cat_fg_mask_pred_tgt panels are fg_mask/pred/tgt triplets, not point-support renders.",
            "point_support_description": "cat_weight_pred_tgt and weight_native correspond to raw support/weight visualization.",
        },
        "rows": rows,
    }
    write_json(args.out_json, payload)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "relative_path",
            "class_name",
            "is_point_support",
            "is_ghost_metric_input",
            "left_panel_semantics",
            "width",
            "height",
            "referenced_by_report",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "relative_path": row["relative_path"],
                    "class_name": row["class_name"],
                    "is_point_support": row["is_point_support"],
                    "is_ghost_metric_input": row["is_ghost_metric_input"],
                    "left_panel_semantics": row["left_panel_semantics"],
                    "width": row["width"],
                    "height": row["height"],
                    "referenced_by_report": ";".join(row["referenced_by_report"]),
                    "error": row["error"],
                }
            )

    md_lines = [
        "# Visual Taxonomy Audit",
        "",
        f"- reports_dir: `{reports_dir}`",
        f"- total_png: `{len(rows)}`",
        "",
        "## Conclusions",
        "",
        "- `cat_fg_mask_pred_tgt*.png` is not a point cloud image. It is the ghost scorer input triplet: `fg_mask | pred | tgt`.",
        "- `cat_weight_pred_tgt*.png` and `weight_native.png` are the raw support / weight visualization and do correspond to point-support style outputs.",
        "",
        "## Count By Class",
        "",
    ]
    for name, count in sorted(counts.items()):
        md_lines.append(f"- `{name}`: `{count}`")
    md_lines.extend(["", "## Latest Ghost Triplets", ""])
    for row in ghost_rows[-12:]:
        md_lines.append(
            f"- `{row['relative_path']}` -> class=`{row['class_name']}` point_support=`{row['is_point_support']}`"
        )
    md_lines.extend(["", "## Latest Point Support Images", ""])
    for row in point_rows[-12:]:
        md_lines.append(
            f"- `{row['relative_path']}` -> class=`{row['class_name']}` left=`{row['left_panel_semantics']}`"
        )
    write_text(args.out_md, "\n".join(md_lines) + "\n")
    print(f"[taxonomy] wrote {args.out_json}")
    print(f"[taxonomy] ghost_triplets={len(ghost_rows)} point_support={len(point_rows)}")


if __name__ == "__main__":
    main()
