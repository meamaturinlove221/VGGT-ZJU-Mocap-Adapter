import argparse
import csv
import json
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np


def _to_3x4_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic)
    if e.ndim == 2:
        e = e[None, ...]
    if e.shape[-2:] == (4, 4):
        e = e[..., :3, :4]
    if e.shape[-2:] != (3, 4):
        raise ValueError(f"invalid extrinsic shape: {e.shape}")
    return e.astype(np.float64, copy=False)


def _to_3x3_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    k = np.asarray(intrinsic)
    if k.ndim == 2:
        k = k[None, ...]
    if k.shape[-2:] == (4, 4):
        k = k[..., :3, :3]
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"invalid intrinsic shape: {k.shape}")
    return k.astype(np.float64, copy=False)


def _to_pointmap_hw3(pm: np.ndarray) -> np.ndarray:
    p = np.asarray(pm)
    if p.ndim == 3 and p.shape[-1] == 3:
        return p.astype(np.float64, copy=False)
    if p.ndim == 3 and p.shape[0] == 3:
        return np.transpose(p, (1, 2, 0)).astype(np.float64, copy=False)
    raise ValueError(f"invalid pointmap shape: {p.shape}")


def _camera_center_forward_from_w2c(t_3x4: np.ndarray):
    r = t_3x4[:3, :3]
    t = t_3x4[:3, 3]
    c = -r.T @ t
    fwd_world = r.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return c, fwd_world


def _world_to_cam_from_assumption(t_3x4: np.ndarray, assume: str) -> tuple[np.ndarray, np.ndarray]:
    """
    assume:
      - "w2c": T is world->camera directly
      - "c2w": T is camera->world, convert to world->camera
    """
    assume = str(assume).lower().strip()
    r = t_3x4[:3, :3]
    t = t_3x4[:3, 3]
    if assume == "w2c":
        return r, t
    if assume == "c2w":
        rwc = r
        twc = t
        rcw = rwc.T
        tcw = -rwc.T @ twc
        return rcw, tcw
    raise ValueError(f"unknown assume={assume}")


def _project_world_points(points_world: np.ndarray, k_3x3: np.ndarray, t_3x4: np.ndarray, assume: str):
    r, t = _world_to_cam_from_assumption(t_3x4, assume=assume)
    cam = (r @ points_world.T) + t.reshape(3, 1)
    z = cam[2, :]
    x = cam[0, :] / (z + 1e-8)
    y = cam[1, :] / (z + 1e-8)
    u = k_3x3[0, 0] * x + k_3x3[0, 2]
    v = k_3x3[1, 1] * y + k_3x3[1, 2]
    return u, v, z


def _sample_reprojection_metrics(
    pointmap_hw3: np.ndarray,
    k_3x3: np.ndarray,
    t_3x4: np.ndarray,
    assume: str,
    stride: int = 16,
) -> dict:
    h, w, _ = pointmap_hw3.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    take = ((ys % int(max(1, stride))) == 0) & ((xs % int(max(1, stride))) == 0)

    pts = pointmap_hw3[take]
    gt_u = xs[take].astype(np.float64)
    gt_v = ys[take].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return dict(mean=np.nan, median=np.nan, p95=np.nan, max=np.nan, in_bounds=np.nan, z_pos=np.nan)
    pts = pts[valid]
    gt_u = gt_u[valid]
    gt_v = gt_v[valid]
    u, v, z = _project_world_points(pts, k_3x3, t_3x4, assume=assume)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    if not np.any(ok):
        return dict(mean=np.nan, median=np.nan, p95=np.nan, max=np.nan, in_bounds=np.nan, z_pos=np.nan)
    u = u[ok]
    v = v[ok]
    z = z[ok]
    du = u - gt_u[ok]
    dv = v - gt_v[ok]
    err = np.sqrt(du * du + dv * dv)
    in_bounds = (u >= 0.0) & (u <= (w - 1)) & (v >= 0.0) & (v <= (h - 1))
    return dict(
        mean=float(np.mean(err)),
        median=float(np.median(err)),
        p95=float(np.percentile(err, 95)),
        max=float(np.max(err)),
        in_bounds=float(np.mean(in_bounds)),
        z_pos=float(np.mean(z > 0.0)),
    )


def _collect_npz_paths(zju_root: str, seq_names: list[str], geom_subdir: str) -> list[str]:
    paths = []
    for seq in seq_names:
        d = osp.join(zju_root, seq, str(geom_subdir))
        if not osp.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npz"):
                paths.append(osp.join(d, fn))
    return paths


def _safe_mean(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zju_root", type=str, required=True)
    ap.add_argument("--seq_names", type=str, default="CoreView_390")
    ap.add_argument("--geom_subdir", type=str, default="vggt_geom")
    ap.add_argument("--out", type=str, default="camera_sanity_out")
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--point_stride", type=int, default=16)
    ap.add_argument("--ortho_err_thr", type=float, default=1e-3)
    ap.add_argument("--det_dev_thr", type=float, default=1e-3)
    ap.add_argument("--reproj_px_thr", type=float, default=2.0)
    args = ap.parse_args()

    seq_names = [s for s in re.split(r"[,\s]+", str(args.seq_names)) if s]
    if not seq_names:
        raise RuntimeError("seq_names is empty")
    out_dir = str(args.out)
    os.makedirs(out_dir, exist_ok=True)

    paths = _collect_npz_paths(str(args.zju_root), seq_names, geom_subdir=str(args.geom_subdir))
    if not paths:
        raise RuntimeError(
            f"no npz found under {args.zju_root} for seq_names={seq_names} geom_subdir={args.geom_subdir}"
        )

    rng = np.random.RandomState(int(args.seed))
    order = rng.permutation(len(paths))
    take_n = min(int(args.num_frames), len(paths))
    chosen = [paths[int(order[i])] for i in range(take_n)]

    rows = []
    centers_all = []
    fwd_all = []

    for npz_path in chosen:
        d = np.load(npz_path, allow_pickle=True)
        ex = _to_3x4_extrinsic(d["extrinsic"])
        k = _to_3x3_intrinsic(d["intrinsic"])
        pm = np.asarray(d["pointmap"])
        cam_names = d["cam_names"] if "cam_names" in d else None
        if cam_names is not None:
            cam_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in cam_names]

        v = ex.shape[0]
        for i in range(v):
            r = ex[i, :3, :3]
            ortho_err = float(np.linalg.norm(r.T @ r - np.eye(3), ord="fro"))
            det_r = float(np.linalg.det(r))
            c, fwd = _camera_center_forward_from_w2c(ex[i])
            centers_all.append(c)
            fwd_all.append(fwd)

            pm_i = _to_pointmap_hw3(pm[i])
            m_w2c = _sample_reprojection_metrics(
                pointmap_hw3=pm_i,
                k_3x3=k[i],
                t_3x4=ex[i],
                assume="w2c",
                stride=int(args.point_stride),
            )
            m_c2w = _sample_reprojection_metrics(
                pointmap_hw3=pm_i,
                k_3x3=k[i],
                t_3x4=ex[i],
                assume="c2w",
                stride=int(args.point_stride),
            )
            rows.append({
                "npz": npz_path,
                "view_idx": int(i),
                "cam_name": (cam_names[i] if cam_names is not None and i < len(cam_names) else ""),
                "ortho_err": ortho_err,
                "det_r": det_r,
                "center_x": float(c[0]),
                "center_y": float(c[1]),
                "center_z": float(c[2]),
                "fwd_x": float(fwd[0]),
                "fwd_y": float(fwd[1]),
                "fwd_z": float(fwd[2]),
                "reproj_mean_w2c": m_w2c["mean"],
                "reproj_median_w2c": m_w2c["median"],
                "reproj_p95_w2c": m_w2c["p95"],
                "in_bounds_w2c": m_w2c["in_bounds"],
                "z_pos_w2c": m_w2c["z_pos"],
                "reproj_mean_c2w": m_c2w["mean"],
                "reproj_median_c2w": m_c2w["median"],
                "reproj_p95_c2w": m_c2w["p95"],
                "in_bounds_c2w": m_c2w["in_bounds"],
                "z_pos_c2w": m_c2w["z_pos"],
            })

    if not rows:
        raise RuntimeError("no view rows generated")

    csv_path = osp.join(out_dir, "camera_sanity_per_view.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    ortho_vals = [r["ortho_err"] for r in rows]
    det_dev_vals = [abs(r["det_r"] - 1.0) for r in rows]
    reproj_w2c_vals = [r["reproj_mean_w2c"] for r in rows]
    reproj_c2w_vals = [r["reproj_mean_c2w"] for r in rows]
    ratio_better_w2c = float(np.mean(
        np.asarray(reproj_w2c_vals, dtype=np.float64) < np.asarray(reproj_c2w_vals, dtype=np.float64)
    ))

    summary = {
        "num_frames_checked": int(take_n),
        "num_views_checked": int(len(rows)),
        "geom_subdir": str(args.geom_subdir),
        "mean_ortho_err": _safe_mean(ortho_vals),
        "max_ortho_err": float(np.nanmax(np.asarray(ortho_vals, dtype=np.float64))),
        "mean_abs_det_minus_1": _safe_mean(det_dev_vals),
        "max_abs_det_minus_1": float(np.nanmax(np.asarray(det_dev_vals, dtype=np.float64))),
        "mean_reproj_w2c": _safe_mean(reproj_w2c_vals),
        "mean_reproj_c2w": _safe_mean(reproj_c2w_vals),
        "ratio_w2c_better_than_c2w": ratio_better_w2c,
        "likely_extrinsic_type": (
            "w2c"
            if ratio_better_w2c > 0.7
            else ("c2w_or_mismatch" if ratio_better_w2c < 0.3 else "ambiguous")
        ),
        "thresholds": {
            "ortho_err_thr": float(args.ortho_err_thr),
            "det_dev_thr": float(args.det_dev_thr),
            "reproj_px_thr": float(args.reproj_px_thr),
        },
        "checks": {
            "rotation_orthogonality_ok": bool(_safe_mean(ortho_vals) <= float(args.ortho_err_thr)),
            "rotation_det_ok": bool(_safe_mean(det_dev_vals) <= float(args.det_dev_thr)),
            "reprojection_ok": bool(_safe_mean(reproj_w2c_vals) <= float(args.reproj_px_thr)),
        },
        "csv": csv_path,
    }

    with open(osp.join(out_dir, "camera_sanity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    centers_arr = np.asarray(centers_all, dtype=np.float64)
    fwd_arr = np.asarray(fwd_all, dtype=np.float64)
    if centers_arr.ndim == 2 and centers_arr.shape[0] > 0:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(centers_arr[:, 0], centers_arr[:, 1], centers_arr[:, 2], s=8, c="tab:blue")
        step = max(1, centers_arr.shape[0] // 80)
        ax.quiver(
            centers_arr[::step, 0], centers_arr[::step, 1], centers_arr[::step, 2],
            fwd_arr[::step, 0], fwd_arr[::step, 1], fwd_arr[::step, 2],
            length=0.2, normalize=True, color="tab:red"
        )
        ax.set_title("Camera Centers + Forward Vectors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, "camera_axes_3d.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(np.asarray(ortho_vals, dtype=np.float64), bins=40, color="tab:orange")
    ax[0].set_title("R orthogonality error")
    ax[0].set_xlabel("||R^T R - I||_F")
    ax[1].hist(np.asarray(reproj_w2c_vals, dtype=np.float64), bins=40, color="tab:green")
    ax[1].set_title("Reprojection mean px (w2c)")
    ax[1].set_xlabel("pixels")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "camera_metrics_hist.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {csv_path}")


if __name__ == "__main__":
    _main()
