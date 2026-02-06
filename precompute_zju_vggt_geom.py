import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vggt_geom import VGGTGeomTeacher
from zju_multiview import ZJUMocapSeq


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


ZJU_ROOT = r"F:\datasets\ZJU_MoCap\data\zju_mocap"
SEQ_LIST = [
    "CoreView_390",
    # "CoreView_313",
    # "CoreView_377",
]
SELECT_CAMERAS = [
    "Camera_B1",
    "Camera_B5",
    "Camera_B10",
    "Camera_B14",
    "Camera_B19",
    "Camera_B23",
]
CKPT_PATH = "model.pt"


def _split_list(raw: str) -> list[str]:
    raw = (raw or "").replace(",", " ").replace(";", " ").replace("|", " ")
    return [x for x in raw.split() if x]


def _resolve_ckpt_path() -> str:
    # Prefer explicit envs from Modal launcher.
    env_ckpt = os.environ.get("VGGT_CKPT", "").strip()
    if not env_ckpt:
        env_ckpt = os.environ.get("VGGT_PRECOMPUTE_CKPT", "").strip()
    if env_ckpt:
        return env_ckpt

    # Portable default: model.pt next to this script.
    local_default = Path(__file__).resolve().with_name("model.pt")
    if local_default.exists():
        return str(local_default)

    # Final fallback for legacy local runs.
    return CKPT_PATH


def main():
    # _ENV_VARS_VGGT_
    zju_root = os.environ.get("VGGT_ZJU_ROOT", "").strip() or ZJU_ROOT
    ckpt_path = _resolve_ckpt_path()
    out_dir = os.environ.get("VGGT_OUT_DIR", "").strip() or "vggt_geom"

    seq_list = _split_list(os.environ.get("VGGT_SEQ_NAMES", "").strip()) or list(SEQ_LIST)
    cam_list = _split_list(os.environ.get("VGGT_CAM_NAMES", "").strip()) or list(SELECT_CAMERAS)

    max_raw = os.environ.get("VGGT_MAX_FRAMES", "").strip()
    max_frames = int(max_raw) if max_raw.isdigit() else 0

    device_env = os.environ.get("VGGT_DEVICE", "").strip().lower()
    if device_env in {"", "auto", "none"}:
        device_env = ""
    use_cuda = torch.cuda.is_available() and (torch.cuda.device_count() > 0)
    device = device_env or ("cuda" if use_cuda else "cpu")
    if str(device).startswith("cuda") and torch.cuda.device_count() == 0:
        device = "cpu"

    print("[dev]", "resolved_device=", device, "cuda_count=", torch.cuda.device_count())
    teacher = VGGTGeomTeacher(ckpt_path, device=device)

    for seq_name in seq_list:
        seq_root = osp.join(zju_root, seq_name)
        if not osp.isdir(seq_root):
            print("[warn] seq not found:", seq_root)
            continue

        out_root = osp.join(seq_root, out_dir)
        os.makedirs(out_root, exist_ok=True)

        seq = ZJUMocapSeq(seq_root, cam_names=(cam_list if len(cam_list) > 0 else None))
        print(f"[{seq_name}] frames={seq.num_frames()} cams={seq.num_cams()}")

        num_frames = seq.num_frames()
        if max_frames > 0:
            num_frames = min(num_frames, int(max_frames))

        for frame_idx in tqdm(range(num_frames), desc=seq_name):
            fid = seq.get_frame_id(frame_idx)
            out_npz = osp.join(out_root, f"frame_{fid:06d}.npz")
            if osp.exists(out_npz):
                continue

            cam2path = seq.get_frame_paths(frame_idx)
            if not cam2path:
                continue

            cam_names = sorted(cam2path.keys())
            img_paths = [osp.relpath(cam2path[c], start=zju_root).replace("\\", "/") for c in cam_names]

            with torch.no_grad():
                # Resolve relative image paths to absolute paths for teacher inference.
                img_paths2 = []
                for p in img_paths:
                    s = str(p)
                    if osp.exists(s):
                        img_paths2.append(s)
                        continue

                    s2 = s.replace("\\", "/")
                    if len(s2) >= 3 and s2[1] == ":" and s2[2] == "/":
                        s2 = s2[2:]
                    ix = s2.find("CoreView_")
                    if ix != -1:
                        s2 = s2[ix:]
                    s2 = s2.lstrip("/")
                    if s2.startswith("CoreView_"):
                        cand = osp.join(zju_root, s2)
                    else:
                        cand = osp.join(seq_root, s2)
                    img_paths2.append(cand)
                img_paths = img_paths2

                geom = teacher(img_paths)

            np.savez_compressed(
                out_npz,
                cam_names=np.array(cam_names),
                img_paths=np.array(img_paths),
                depth=to_numpy(geom["depth"]),
                depth_conf=to_numpy(geom["depth_conf"]),
                pointmap=to_numpy(geom["pointmap"]),
                extrinsic=to_numpy(geom["extrinsic"]),
                intrinsic=to_numpy(geom["intrinsic"]),
            )

        print(f"[{seq_name}] done, saved to {out_root}")


if __name__ == "__main__":
    main()
