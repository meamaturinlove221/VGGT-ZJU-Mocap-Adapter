import argparse
import csv
import json
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vggt.utils.geometry import unproject_depth_map_to_point_map


def _split_tokens(raw: str) -> list[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [x for x in re.split(r"[,\s;/|]+", s) if x]


def _to_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 3:
        raise ValueError(f"depth shape must be (V,H,W) or (V,H,W,1), got {d.shape}")
    return d.astype(np.float32, copy=False)


def _to_pointmap(pointmap: np.ndarray) -> np.ndarray:
    p = np.asarray(pointmap)
    if p.ndim != 4 or p.shape[-1] != 3:
        raise ValueError(f"pointmap shape must be (V,H,W,3), got {p.shape}")
    return p.astype(np.float32, copy=False)


def _to_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic)
    if e.ndim == 2:
        e = e[None, ...]
    if e.shape[-2:] == (4, 4):
        e = e[..., :3, :4]
    if e.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic shape must be (V,3,4) or (V,4,4), got {e.shape}")
    return e.astype(np.float64, copy=False)


def _to_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    k = np.asarray(intrinsic)
    if k.ndim == 2:
        k = k[None, ...]
    if k.shape[-2:] == (4, 4):
        k = k[..., :3, :3]
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsic shape must be (V,3,3) or (V,4,4), got {k.shape}")
    return k.astype(np.float64, copy=False)


def _stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "n": int(x.size),
    }


def _project_error_px(
    pointmap_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    eps: float,
    sample_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    V, H, W, _ = pointmap_world.shape
    errs_all: list[np.ndarray] = []
    grid_u, grid_v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

    for vi in range(V):
        pts = pointmap_world[vi].reshape(-1, 3).astype(np.float64)
        d = depth[vi].reshape(-1)
        R = extrinsic_w2c[vi, :3, :3]
        t = extrinsic_w2c[vi, :3, 3]
        K = intrinsic[vi]

        cam = (pts @ R.T) + t[None, :]
        z = cam[:, 2]
        valid = np.isfinite(z) & (z > eps) & np.isfinite(d) & (d > eps)
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        if sample_pixels > 0 and idx.size > sample_pixels:
            idx = rng.choice(idx, size=sample_pixels, replace=False)

        cam_v = cam[idx]
        u = K[0, 0] * (cam_v[:, 0] / cam_v[:, 2]) + K[0, 2]
        v = K[1, 1] * (cam_v[:, 1] / cam_v[:, 2]) + K[1, 2]

        gu = grid_u.reshape(-1)[idx]
        gv = grid_v.reshape(-1)[idx]
        err = np.sqrt((u - gu) ** 2 + (v - gv) ** 2)
        err = err[np.isfinite(err)]
        if err.size > 0:
            errs_all.append(err.astype(np.float64))

    if not errs_all:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(errs_all, axis=0)


def check_one_npz(
    npz_path: str,
    eps: float,
    sample_pixels: int,
    rng: np.random.Generator,
    unproject_impl: str,
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        depth = _to_depth(data["depth"])
        pointmap = _to_pointmap(data["pointmap"])
        extrinsic = _to_extrinsic(data["extrinsic"])
        intrinsic = _to_intrinsic(data["intrinsic"])

    if not (depth.shape[0] == pointmap.shape[0] == extrinsic.shape[0] == intrinsic.shape[0]):
        raise RuntimeError(
            f"view dim mismatch: depth={depth.shape}, pointmap={pointmap.shape}, "
            f"extrinsic={extrinsic.shape}, intrinsic={intrinsic.shape}"
        )

    pointmap_recon = unproject_depth_map_to_point_map(
        depth[..., None],
        extrinsic,
        intrinsic,
        unproject_impl=unproject_impl,
    )
    valid = np.isfinite(depth) & (depth > eps)
    diff = np.linalg.norm(pointmap_recon - pointmap, axis=-1)
    diff = diff[valid]
    point_stats = _stats(diff.astype(np.float64))

    reproj_err = _project_error_px(
        pointmap_world=pointmap,
        extrinsic_w2c=extrinsic,
        intrinsic=intrinsic,
        depth=depth,
        eps=eps,
        sample_pixels=sample_pixels,
        rng=rng,
    )
    reproj_stats = _stats(reproj_err)

    return {
        "point_l2": point_stats,
        "reproj_px": reproj_stats,
        "views": int(depth.shape[0]),
        "height": int(depth.shape[1]),
        "width": int(depth.shape[2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser("check_depth_point_reprojection")
    ap.add_argument("--zju_root", type=str, required=True)
    ap.add_argument("--seq_names", type=str, default="CoreView_390")
    ap.add_argument("--geom_subdir", type=str, default="vggt_geom")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0, help="0 means all")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--sample_pixels", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unproject_impl", type=str, default="legacy")
    ap.add_argument("--out_json", type=str, default="logs/modal_phase5/vggt_ft_reproj_check_latest.json")
    ap.add_argument("--out_csv", type=str, default="logs/modal_phase5/vggt_ft_reproj_check_latest.csv")
    args = ap.parse_args()

    seq_names = _split_tokens(args.seq_names)
    if not seq_names:
        raise RuntimeError("seq_names is empty")

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []

    for seq in seq_names:
        geom_dir = Path(args.zju_root) / seq / args.geom_subdir
        if not geom_dir.is_dir():
            rows.append(
                {
                    "seq": seq,
                    "frame": "",
                    "status": "missing_geom_dir",
                    "point_l2_mean": "",
                    "point_l2_p95": "",
                    "reproj_px_mean": "",
                    "reproj_px_p95": "",
                    "n": "",
                }
            )
            continue

        files = sorted([p for p in geom_dir.glob("*.npz") if p.is_file()])
        stride = max(1, int(args.frame_stride))
        files = files[::stride]
        if int(args.max_frames) > 0:
            files = files[: int(args.max_frames)]

        for p in files:
            try:
                stat = check_one_npz(
                    npz_path=str(p),
                    eps=float(args.eps),
                    sample_pixels=int(args.sample_pixels),
                    rng=rng,
                    unproject_impl=str(args.unproject_impl),
                )
                rows.append(
                    {
                        "seq": seq,
                        "frame": p.stem,
                        "status": "ok",
                        "point_l2_mean": stat["point_l2"]["mean"],
                        "point_l2_p95": stat["point_l2"]["p95"],
                        "reproj_px_mean": stat["reproj_px"]["mean"],
                        "reproj_px_p95": stat["reproj_px"]["p95"],
                        "n": stat["point_l2"]["n"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "seq": seq,
                        "frame": p.stem,
                        "status": f"error: {e}",
                        "point_l2_mean": "",
                        "point_l2_p95": "",
                        "reproj_px_mean": "",
                        "reproj_px_p95": "",
                        "n": "",
                    }
                )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        p_mean = np.asarray([float(r["point_l2_mean"]) for r in ok_rows], dtype=np.float64)
        p_p95 = np.asarray([float(r["point_l2_p95"]) for r in ok_rows], dtype=np.float64)
        r_mean = np.asarray([float(r["reproj_px_mean"]) for r in ok_rows], dtype=np.float64)
        r_p95 = np.asarray([float(r["reproj_px_p95"]) for r in ok_rows], dtype=np.float64)
        summary = {
            "point_l2_mean_mean": float(np.mean(p_mean)),
            "point_l2_p95_mean": float(np.mean(p_p95)),
            "reproj_px_mean_mean": float(np.mean(r_mean)),
            "reproj_px_p95_mean": float(np.mean(r_p95)),
        }
    else:
        summary = {
            "point_l2_mean_mean": float("nan"),
            "point_l2_p95_mean": float("nan"),
            "reproj_px_mean_mean": float("nan"),
            "reproj_px_p95_mean": float("nan"),
        }

    out = {
        "zju_root": str(args.zju_root),
        "seq_names": seq_names,
        "geom_subdir": str(args.geom_subdir),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "eps": float(args.eps),
        "sample_pixels": int(args.sample_pixels),
        "unproject_impl": str(args.unproject_impl),
        "num_rows": len(rows),
        "num_ok": len(ok_rows),
        "summary": summary,
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "seq",
            "frame",
            "status",
            "point_l2_mean",
            "point_l2_p95",
            "reproj_px_mean",
            "reproj_px_p95",
            "n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({"ok": True, "out_json": str(out_json), "out_csv": str(out_csv), "num_ok": len(ok_rows)}))


if __name__ == "__main__":
    main()
