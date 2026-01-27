# F:\vggt\precompute_zju_vggt_geom.py

import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch

from zju_multiview import ZJUMocapSeq
from vggt_geom import VGGTGeomTeacher


def to_numpy(x):
    """兼容 torch.Tensor 和 numpy.ndarray，统一转成 numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        # 比如已经是 numpy，或者 list 之类
        return np.asarray(x)


# 1) ZJU-MoCap 根目录（照你现在的结构）
ZJU_ROOT = r"F:\datasets\ZJU_MoCap\data\zju_mocap"

# 2) 先挑几个 sequence 做实验，确认没问题再加
SEQ_LIST = [
    "CoreView_390",
    # "CoreView_313",
    # "CoreView_377",
]

# 3) 可选：限定使用哪些 Camera_*（不写就是全部）
SELECT_CAMERAS = [
    "Camera_B1",
    "Camera_B5",
    "Camera_B9",
    "Camera_B13",
]

# 4) VGGT 权重
CKPT_PATH = r"F:\vggt\model.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = VGGTGeomTeacher(CKPT_PATH, device=device)

    for seq_name in SEQ_LIST:
        seq_root = osp.join(ZJU_ROOT, seq_name)
        if not osp.isdir(seq_root):
            print("[warn] seq not found:", seq_root)
            continue

        # 输出目录：直接放在 CoreView_xxx/vggt_geom
        out_root = osp.join(seq_root, "vggt_geom")
        os.makedirs(out_root, exist_ok=True)

        # 构建 sequence 读帧
        seq = ZJUMocapSeq(seq_root, cam_names=SELECT_CAMERAS)

        print(f"[{seq_name}] frames={seq.num_frames()} cams={seq.num_cams()}")

        for frame_idx in tqdm(range(seq.num_frames()), desc=seq_name):
            fid = seq.get_frame_id(frame_idx)  # 比如 0,1,2,...
            out_npz = osp.join(out_root, f"frame_{fid:06d}.npz")
            if osp.exists(out_npz):
                continue

            cam2path = seq.get_frame_paths(frame_idx)
            if not cam2path:
                continue

            cam_names = sorted(cam2path.keys())
            img_paths = [osp.relpath(cam2path[c], start=ZJU_ROOT).replace(
                "\\", "/") for c in cam_names]

            with torch.no_grad():
                geom = teacher(img_paths)

            # 存 numpy
            np.savez_compressed(
                out_npz,
                cam_names=np.array(cam_names),
                img_paths=np.array(img_paths),
                depth=to_numpy(geom["depth"]),           # (V,H,W)
                depth_conf=to_numpy(geom["depth_conf"]),  # (V,H,W)
                pointmap=to_numpy(geom["pointmap"]),     # (V,H,W,3)
                extrinsic=to_numpy(geom["extrinsic"]),   # (V,4,4)
                intrinsic=to_numpy(geom["intrinsic"]),   # (V,3,3)
            )

        print(f"[{seq_name}] done, saved to {out_root}")


if __name__ == "__main__":
    main()
