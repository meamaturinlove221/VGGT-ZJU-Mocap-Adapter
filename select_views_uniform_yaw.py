import argparse
import json
import os
import os.path as osp
import re
from typing import Iterable, Optional

import numpy as np


def _to_3x4_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic)
    if e.ndim == 2:
        e = e[None, ...]
    if e.ndim != 3:
        raise ValueError(f"extrinsic must be 2D/3D, got shape={e.shape}")
    if e.shape[-2:] == (4, 4):
        e = e[..., :3, :4]
    elif e.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic must be (...,3,4) or (...,4,4), got shape={e.shape}")
    return e.astype(np.float64, copy=False)


def camera_centers_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """
    Compute camera centers C in world coordinates from OpenCV-style world->camera extrinsics.
    extrinsic format: X_cam = R * X_world + t.
    """
    e = _to_3x4_extrinsic(extrinsic)
    r = e[:, :3, :3]
    t = e[:, :3, 3]
    c = -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)
    return c


def estimate_subject_center(
    pointmap: Optional[np.ndarray],
    camera_centers: Optional[np.ndarray] = None,
    stride: int = 16,
) -> np.ndarray:
    """
    Estimate subject/scene center in world coordinates.
    Priority: robust median from pointmap, fallback to mean(camera centers), else zeros.
    """
    if pointmap is not None:
        pm = np.asarray(pointmap)
        # Accept (V,H,W,3) or (V,3,H,W)
        if pm.ndim == 4 and pm.shape[-1] == 3:
            pm_hw3 = pm
        elif pm.ndim == 4 and pm.shape[1] == 3:
            pm_hw3 = np.transpose(pm, (0, 2, 3, 1))
        else:
            pm_hw3 = None

        if pm_hw3 is not None:
            s = max(int(stride), 1)
            pts = pm_hw3[:, ::s, ::s, :].reshape(-1, 3)
            ok = np.isfinite(pts).all(axis=1)
            if np.any(ok):
                return np.median(pts[ok], axis=0)

    if camera_centers is not None:
        cc = np.asarray(camera_centers)
        if cc.ndim == 2 and cc.shape[-1] == 3 and len(cc) > 0:
            ok = np.isfinite(cc).all(axis=1)
            if np.any(ok):
                return np.mean(cc[ok], axis=0)

    return np.zeros(3, dtype=np.float64)


def yaw_degrees_from_centers(
    camera_centers: np.ndarray,
    subject_center: np.ndarray,
    axis_x: int = 0,
    axis_z: int = 2,
) -> np.ndarray:
    cc = np.asarray(camera_centers, dtype=np.float64)
    sc = np.asarray(subject_center, dtype=np.float64).reshape(1, 3)
    rel = cc - sc
    yaw = np.degrees(np.arctan2(rel[:, int(axis_x)], rel[:, int(axis_z)]))
    return (yaw + 360.0) % 360.0


def circular_distance_deg(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _normalize_idx_list(idx_like: Optional[Iterable[int]]) -> set[int]:
    if idx_like is None:
        return set()
    out = set()
    for v in idx_like:
        try:
            out.add(int(v))
        except Exception:
            continue
    return out


def pick_uniform_yaw_indices(
    yaw_deg: np.ndarray,
    num_select: int,
    rng: np.random.RandomState | np.random.Generator | np.random.mtrand.RandomState,
    exclude_indices: Optional[Iterable[int]] = None,
    jitter_deg: float = 20.0,
    phase_jitter_deg: float = 20.0,
    phase_deg: Optional[float] = None,
) -> list[int]:
    """
    Pick indices whose yaw angles approximately cover 360 degrees uniformly.
    """
    yaw = np.asarray(yaw_deg, dtype=np.float64).reshape(-1)
    n = int(yaw.shape[0])
    k = int(num_select)
    if n <= 0 or k <= 0:
        return []

    excluded = _normalize_idx_list(exclude_indices)
    candidates = [i for i in range(n) if i not in excluded]
    if len(candidates) <= k:
        return list(candidates)

    if phase_deg is None:
        phase_deg = float(rng.uniform(0.0, 360.0))
    phase = (phase_deg + float(rng.uniform(-phase_jitter_deg, phase_jitter_deg))) % 360.0
    bw = 360.0 / float(k)
    jitter = max(float(jitter_deg), 0.0)

    chosen: list[int] = []
    used: set[int] = set()
    for i in range(k):
        target = (phase + i * bw + float(rng.uniform(-jitter, jitter))) % 360.0
        best_idx = None
        best_dist = 1e18
        for idx in candidates:
            if idx in used:
                continue
            d = float(circular_distance_deg(yaw[idx], target))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(int(best_idx))
        used.add(int(best_idx))

    if len(chosen) < k:
        remain = [i for i in candidates if i not in used]
        if remain:
            remain = list(np.asarray(remain)[rng.permutation(len(remain))])
            chosen.extend(int(v) for v in remain[: (k - len(chosen))])
    return chosen[:k]


def _eligible_tgt_indices(
    cam_ids: np.ndarray,
    tgt_view_ids: Optional[Iterable[int]] = None,
    tgt_view_ids_exclude: Optional[Iterable[int]] = None,
) -> list[int]:
    cam_ids = np.asarray(cam_ids).reshape(-1)
    include = None if tgt_view_ids is None else _normalize_idx_list(tgt_view_ids)
    exclude = _normalize_idx_list(tgt_view_ids_exclude)
    out: list[int] = []
    for i, vid in enumerate(cam_ids.tolist()):
        v = int(vid)
        if include is not None and v not in include:
            continue
        if v in exclude:
            continue
        out.append(i)
    return out


def select_src_tgt_uniform_yaw(
    cam_ids: np.ndarray,
    extrinsic: np.ndarray,
    pointmap: Optional[np.ndarray],
    num_src_views: int,
    rng: np.random.RandomState | np.random.Generator | np.random.mtrand.RandomState,
    tgt_view_ids: Optional[Iterable[int]] = None,
    tgt_view_ids_exclude: Optional[Iterable[int]] = None,
    yaw_jitter_deg: float = 20.0,
    yaw_phase_jitter_deg: float = 20.0,
    yaw_axis_x: int = 0,
    yaw_axis_z: int = 2,
    center_mode: str = "pointmap",
) -> tuple[np.ndarray, int, dict]:
    """
    Select target view and source views:
    - target: random from eligible target views
    - source: approximately uniform yaw coverage over 360 degrees
    """
    cam_ids = np.asarray(cam_ids).reshape(-1)
    v = int(cam_ids.shape[0])
    if v < 2:
        raise ValueError(f"need at least 2 views, got {v}")
    num_src = min(int(num_src_views), v - 1)
    if num_src <= 0:
        raise ValueError(f"invalid num_src_views={num_src_views} for V={v}")

    eligible = _eligible_tgt_indices(
        cam_ids=cam_ids,
        tgt_view_ids=tgt_view_ids,
        tgt_view_ids_exclude=tgt_view_ids_exclude,
    )
    if not eligible:
        raise RuntimeError("no eligible target views after holdout filter")

    tgt_idx = int(rng.choice(eligible))

    centers = camera_centers_from_extrinsic(extrinsic)
    center_mode = str(center_mode or "pointmap").lower().strip()
    if center_mode == "camera":
        subject_center = estimate_subject_center(None, camera_centers=centers)
    else:
        subject_center = estimate_subject_center(pointmap, camera_centers=centers)
    yaw = yaw_degrees_from_centers(
        centers, subject_center, axis_x=int(yaw_axis_x), axis_z=int(yaw_axis_z)
    )

    src_idxs = pick_uniform_yaw_indices(
        yaw_deg=yaw,
        num_select=num_src,
        rng=rng,
        exclude_indices=[tgt_idx],
        jitter_deg=float(yaw_jitter_deg),
        phase_jitter_deg=float(yaw_phase_jitter_deg),
    )
    if len(src_idxs) < num_src:
        # Fallback pad from remaining views.
        remain = [i for i in range(v) if i != tgt_idx and i not in src_idxs]
        if remain:
            remain = list(np.asarray(remain)[rng.permutation(len(remain))])
            src_idxs.extend(remain[: (num_src - len(src_idxs))])
    src_idxs = np.asarray(src_idxs[:num_src], dtype=np.int64)

    debug = {
        "tgt_idx": int(tgt_idx),
        "tgt_vid": int(cam_ids[tgt_idx]),
        "src_idxs": [int(x) for x in src_idxs.tolist()],
        "src_vids": [int(cam_ids[x]) for x in src_idxs.tolist()],
        "yaw_deg": yaw.tolist(),
        "subject_center": subject_center.tolist(),
        "camera_centers": centers.tolist(),
    }
    return src_idxs, int(tgt_idx), debug


def _resolve_img_path(path_str: str, zju_root: str) -> str:
    if isinstance(path_str, bytes):
        path_str = path_str.decode("utf-8")
    s = str(path_str).strip().replace("\\", "/")
    if osp.exists(s):
        return s
    if osp.isabs(s) and osp.exists(s):
        return s
    if re.match(r"^[A-Za-z]:/", s):
        key = "/zju_mocap/"
        if key in s:
            s2 = s.split(key, 1)[1]
        else:
            i = s.find("CoreView_")
            s2 = s[i:] if i >= 0 else s
        return osp.join(zju_root, s2.lstrip("/"))
    return osp.join(zju_root, s.lstrip("/"))


def _parse_int_list(raw: str) -> Optional[list[int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out = []
    for p in re.split(r"[,\s;/]+", s):
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out if out else None


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--zju_root", type=str, default="")
    ap.add_argument("--num_src_views", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tgt_view_ids", type=str, default="")
    ap.add_argument("--tgt_view_ids_exclude", type=str, default="")
    ap.add_argument("--yaw_jitter_deg", type=float, default=20.0)
    ap.add_argument("--yaw_phase_jitter_deg", type=float, default=20.0)
    ap.add_argument("--yaw_axis_x", type=int, default=0)
    ap.add_argument("--yaw_axis_z", type=int, default=2)
    ap.add_argument("--yaw_center_mode", type=str, default="pointmap", choices=["pointmap", "camera"])
    ap.add_argument("--dump_json", type=str, default="")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    cam_names = data["cam_names"] if "cam_names" in data else None
    cam_ids = np.arange(len(cam_names) if cam_names is not None else int(data["img_paths"].shape[0]), dtype=np.int64)
    if cam_names is not None:
        cam_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in cam_names]

    extrinsic = data["extrinsic"] if "extrinsic" in data else None
    if extrinsic is None:
        raise RuntimeError("npz missing 'extrinsic'")
    pointmap = data["pointmap"] if "pointmap" in data else None

    rng = np.random.RandomState(int(args.seed))
    src_idxs, tgt_idx, dbg = select_src_tgt_uniform_yaw(
        cam_ids=cam_ids,
        extrinsic=extrinsic,
        pointmap=pointmap,
        num_src_views=int(args.num_src_views),
        rng=rng,
        tgt_view_ids=_parse_int_list(args.tgt_view_ids),
        tgt_view_ids_exclude=_parse_int_list(args.tgt_view_ids_exclude),
        yaw_jitter_deg=float(args.yaw_jitter_deg),
        yaw_phase_jitter_deg=float(args.yaw_phase_jitter_deg),
        yaw_axis_x=int(args.yaw_axis_x),
        yaw_axis_z=int(args.yaw_axis_z),
        center_mode=str(args.yaw_center_mode),
    )

    img_paths = data["img_paths"]
    zju_root = str(args.zju_root or "").strip()
    if zju_root:
        tgt_img = _resolve_img_path(img_paths[int(tgt_idx)], zju_root=zju_root)
        src_imgs = [_resolve_img_path(img_paths[int(i)], zju_root=zju_root) for i in src_idxs.tolist()]
    else:
        tgt_img = str(img_paths[int(tgt_idx)])
        src_imgs = [str(img_paths[int(i)]) for i in src_idxs.tolist()]

    out = {
        "npz": args.npz,
        "tgt_idx": int(tgt_idx),
        "tgt_cam_id": int(cam_ids[int(tgt_idx)]),
        "tgt_cam_name": (cam_names[int(tgt_idx)] if cam_names is not None else None),
        "tgt_img": tgt_img,
        "src_idxs": [int(x) for x in src_idxs.tolist()],
        "src_cam_ids": [int(cam_ids[int(x)]) for x in src_idxs.tolist()],
        "src_cam_names": ([cam_names[int(x)] for x in src_idxs.tolist()] if cam_names is not None else None),
        "src_imgs": src_imgs,
        "yaw_deg": dbg.get("yaw_deg", []),
        "subject_center": dbg.get("subject_center", []),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.dump_json:
        os.makedirs(osp.dirname(args.dump_json) or ".", exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[ok] wrote {args.dump_json}")


if __name__ == "__main__":
    _main()
