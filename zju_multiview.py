# F:\vggt\zju_multiview.py

import os
import os.path as osp
from glob import glob
from typing import List, Dict


class ZJUMocapSeq:
    """
    适配当前 ZJU-MoCap 目录结构：

        CoreView_xxx/
            Camera_B1/
                000000.jpg
                000001.jpg
                ...
            Camera_B2/
                ...
            ...
            annots/
            keypoints2d/
            mask/
            new_params/
            params/
            vertices/
            annots.npy
            intri.yaml
            extri.yaml
            ...

    功能：
      - 自动找到所有 Camera_* 文件夹
      - 用第一个 camera 的文件名推断 frame id（例如 000000.jpg -> 0）
      - 按 frame_idx 返回 {camera_name: image_path}
    """

    def __init__(
        self,
        seq_root: str,
        cam_names: List[str] | None = None,
        img_pattern: str = "*.*",  # 允许 jpg/png 等
    ):
        self.seq_root = seq_root

        # 1) 找 Camera_* 目录（只看 CoreView_xxx 这一层）
        if cam_names is None:
            all_dirs = [
                d for d in os.listdir(seq_root)
                if osp.isdir(osp.join(seq_root, d))
            ]
            self.cam_names = sorted(
                d for d in all_dirs
                if d.startswith("Camera_")
            )
        else:
            self.cam_names = cam_names

        if not self.cam_names:
            raise RuntimeError(f"No Camera_* folders found in {seq_root}")

        # 2) 用第一个 camera 的文件名来定义 frame id 列表
        first_cam_dir = osp.join(seq_root, self.cam_names[0])

        img_files = sorted(glob(osp.join(first_cam_dir, img_pattern)))
        if not img_files:
            raise RuntimeError(f"No images found in {first_cam_dir}")

        self.frame_ids: list[int] = []
        self.fname_by_fid: dict[int, str] = {}  # fid -> "000000.jpg"

        for p in img_files:
            base = osp.basename(p)          # "000000.jpg"
            stem, ext = osp.splitext(base)  # "000000", ".jpg"
            try:
                fid = int(stem)
            except ValueError:
                # 如果不是纯数字（比如 "000000_00"），以后再特殊处理
                continue
            self.frame_ids.append(fid)
            self.fname_by_fid[fid] = base

        self.frame_ids = sorted(self.frame_ids)

        print(
            f"[ZJUMocapSeq] {osp.basename(seq_root)} | "
            f"cameras: {len(self.cam_names)} | frames: {len(self.frame_ids)}"
        )

    # ------------- 一些便捷方法 -------------

    def num_frames(self) -> int:
        return len(self.frame_ids)

    def num_cams(self) -> int:
        return len(self.cam_names)

    def get_frame_id(self, frame_idx: int) -> int:
        return self.frame_ids[frame_idx]

    def get_frame_paths(self, frame_idx: int) -> Dict[str, str]:
        """
        给定 frame_idx（0 ~ num_frames-1）返回这一帧的所有视角：
            { "Camera_B1": ".../Camera_B1/000000.jpg",
              "Camera_B2": ".../Camera_B2/000000.jpg",
              ... }
        没有该帧的相机会自动跳过。
        """
        fid = self.frame_ids[frame_idx]
        fname = self.fname_by_fid[fid]

        paths: Dict[str, str] = {}
        for cam in self.cam_names:
            p = osp.join(self.seq_root, cam, fname)
            if osp.exists(p):
                paths[cam] = p
        return paths
