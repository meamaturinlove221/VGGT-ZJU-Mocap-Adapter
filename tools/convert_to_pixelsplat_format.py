import argparse
import json
import os
import os.path as osp
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def split_seq_names(raw: str) -> list[str]:
    return [s for s in re.split(r"[,\s]+", str(raw or "").strip()) if s]


def to_3x4_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic)
    if e.ndim == 2:
        e = e[None, ...]
    if e.shape[-2:] == (4, 4):
        e = e[..., :3, :4]
    if e.shape[-2:] != (3, 4):
        raise ValueError(f"invalid extrinsic shape: {e.shape}")
    return e.astype(np.float64, copy=False)


def to_3x3_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    k = np.asarray(intrinsic)
    if k.ndim == 2:
        k = k[None, ...]
    if k.shape[-2:] == (4, 4):
        k = k[..., :3, :3]
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"invalid intrinsic shape: {k.shape}")
    return k.astype(np.float64, copy=False)


def w2c_3x4_to_4x4(w2c_3x4: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :4] = w2c_3x4
    return t


def w2c_3x4_to_c2w_4x4(w2c_3x4: np.ndarray) -> np.ndarray:
    r = w2c_3x4[:3, :3]
    t = w2c_3x4[:3, 3]
    rwc = r.T
    twc = -r.T @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rwc
    out[:3, 3] = twc
    return out


def safe_name(raw: str) -> str:
    s = str(raw)
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._") or "view"


def resolve_img_path(path_str: Any, zju_root: str, seq_names: list[str]) -> str:
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
            s = s.split(key, 1)[1]
        else:
            parts = s.split("/")
            cut = None
            for i, p in enumerate(parts):
                if p.startswith("CoreView_"):
                    cut = i
                    break
            if cut is not None:
                s = "/".join(parts[cut:])
            else:
                for seq in seq_names:
                    if seq in s:
                        s = seq + s.split(seq, 1)[1]
                        break
    return osp.join(zju_root, s.lstrip("/"))


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def maybe_link_or_copy(src: str, dst: str, mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode == "none":
        return src
    ensure_dir(str(Path(dst).parent))
    if osp.exists(dst):
        return dst
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"unsupported copy_mode: {mode}")
    return dst


@dataclass
class ViewRecord:
    view_index: int
    cam_name: str
    image_src: str
    image_path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w_opencv: list[list[float]]
    w2c_opencv: list[list[float]]


def process_one_npz(
    npz_path: str,
    zju_root: str,
    seq_names: list[str],
    out_frame_dir: str,
    copy_mode: str,
    max_views: int,
) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    img_paths = data["img_paths"]
    extrinsic = to_3x4_extrinsic(data["extrinsic"])
    intrinsic = to_3x3_intrinsic(data["intrinsic"])

    cam_names = data["cam_names"] if "cam_names" in data else None
    if cam_names is None:
        cam_names = [f"view_{i:03d}" for i in range(extrinsic.shape[0])]
    else:
        cam_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in cam_names
        ]

    v = int(extrinsic.shape[0])
    if max_views > 0:
        v = min(v, int(max_views))

    images_dir = osp.join(out_frame_dir, "images")
    ensure_dir(images_dir)

    views: list[ViewRecord] = []
    for i in range(v):
        src_path = resolve_img_path(img_paths[i], zju_root=zju_root, seq_names=seq_names)
        if not osp.exists(src_path):
            raise FileNotFoundError(f"image not found: {src_path}")

        with Image.open(src_path) as im:
            w, h = im.size

        cam = safe_name(cam_names[i] if i < len(cam_names) else f"view_{i:03d}")
        dst_name = f"{i:03d}_{cam}{Path(src_path).suffix.lower()}"
        dst_path = osp.join(images_dir, dst_name)
        out_img_path = maybe_link_or_copy(src=src_path, dst=dst_path, mode=copy_mode)

        k = intrinsic[i]
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])

        w2c = w2c_3x4_to_4x4(extrinsic[i])
        c2w = w2c_3x4_to_c2w_4x4(extrinsic[i])
        rec = ViewRecord(
            view_index=i,
            cam_name=cam,
            image_src=src_path.replace("\\", "/"),
            image_path=str(out_img_path).replace("\\", "/"),
            width=int(w),
            height=int(h),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            c2w_opencv=c2w.tolist(),
            w2c_opencv=w2c.tolist(),
        )
        views.append(rec)

    frame_payload = {
        "coord_convention": "opencv_c2w",
        "frame_npz": npz_path.replace("\\", "/"),
        "num_views": len(views),
        "views": [asdict(vr) for vr in views],
    }
    with open(osp.join(out_frame_dir, "cameras_opencv_c2w.json"), "w", encoding="utf-8") as f:
        json.dump(frame_payload, f, ensure_ascii=False, indent=2)
    return frame_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zju_root", type=str, required=True)
    ap.add_argument("--seq_names", type=str, default="CoreView_390")
    ap.add_argument("--geom_subdir", type=str, default="vggt_geom_ft_20260208_044454")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0, help="0 means all")
    ap.add_argument("--max_views", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--copy_mode",
        type=str,
        default="none",
        choices=["none", "copy", "hardlink", "symlink"],
    )
    args = ap.parse_args()

    seq_names = split_seq_names(args.seq_names)
    if not seq_names:
        raise RuntimeError("seq_names is empty")

    out_root = osp.abspath(args.out_root)
    ensure_dir(out_root)

    manifest: dict[str, Any] = {
        "coord_convention": "opencv_c2w",
        "zju_root": osp.abspath(args.zju_root).replace("\\", "/"),
        "seq_names": seq_names,
        "geom_subdir": args.geom_subdir,
        "copy_mode": args.copy_mode,
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "max_views": int(args.max_views),
        "sequences": {},
    }

    for seq in seq_names:
        geom_dir = osp.join(args.zju_root, seq, args.geom_subdir)
        if not osp.isdir(geom_dir):
            raise FileNotFoundError(f"geom dir not found: {geom_dir}")
        npz_files = sorted(
            [osp.join(geom_dir, fn) for fn in os.listdir(geom_dir) if fn.endswith(".npz")]
        )
        if not npz_files:
            raise RuntimeError(f"no npz in: {geom_dir}")

        step = max(int(args.frame_stride), 1)
        npz_files = npz_files[::step]
        if int(args.max_frames) > 0:
            npz_files = npz_files[: int(args.max_frames)]

        seq_out = osp.join(out_root, seq)
        ensure_dir(seq_out)
        seq_frames = []
        for npz_path in npz_files:
            frame_id = Path(npz_path).stem
            frame_out = osp.join(seq_out, f"frame_{frame_id}")
            ensure_dir(frame_out)
            payload = process_one_npz(
                npz_path=npz_path,
                zju_root=args.zju_root,
                seq_names=seq_names,
                out_frame_dir=frame_out,
                copy_mode=args.copy_mode,
                max_views=int(args.max_views),
            )
            seq_frames.append(
                {
                    "frame_id": frame_id,
                    "frame_dir": frame_out.replace("\\", "/"),
                    "num_views": int(payload["num_views"]),
                }
            )

        manifest["sequences"][seq] = {
            "geom_dir": geom_dir.replace("\\", "/"),
            "num_frames": len(seq_frames),
            "frames": seq_frames,
        }

    manifest_path = osp.join(out_root, "pixelsplat_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, "manifest": manifest_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
