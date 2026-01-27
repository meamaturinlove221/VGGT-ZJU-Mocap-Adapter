import torch
from pathlib import Path

# 从本地源码包导入
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1. 准备几张图片路径（同一场景，或者先随便两张也行）
    image_paths = [
        r"F:\your_images\img1.png",
        r"F:\your_images\img2.png",
    ]
    image_paths = [Path(p) for p in image_paths]

    # 2. 预处理成张量 [V, C, H, W]
    images = load_and_preprocess_images(image_paths).to(device)
    print("images.shape:", images.shape)

    # 3. 构建 VGGT 模型骨架
    model = VGGT().to(device)

    # 4. 从本地加载预训练权重（你刚才下载的 model.pt）
    state_dict_path = r"F:\vggt\model.pt"
    print("loading weights from:", state_dict_path)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. 推理：先跑 aggregator 拿 token
    # [V, C, H, W] -> [1, V, C, H, W]
    images_batched = images[None]

    with torch.no_grad():
        if device == "cuda":
            # 2.3.1 支持 bfloat16 / fp16 的混合精度，这里用 fp16 就行
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = torch.no_grad()

        with ctx:
            aggregated_tokens_list, ps_idx = model.aggregator(images_batched)

        # 6. 相机外参 / 内参
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_batched.shape[-2:]
        )
        print("extrinsic.shape:", extrinsic.shape)  # [B, V, 4, 4]
        print("intrinsic.shape:", intrinsic.shape)  # [B, V, 3, 3]

        # 7. 深度图
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_batched, ps_idx
        )
        print("depth_map.shape:", depth_map.shape)  # [B, V, H, W]

        # 8. 用深度 + 相机反投影出 3D 点
        point_map_by_unproj = unproject_depth_map_to_point_map(
            depth_map.squeeze(0),      # 去掉 batch 维 -> [V, H, W]
            extrinsic.squeeze(0),      # -> [V, 4, 4]
            intrinsic.squeeze(0),      # -> [V, 3, 3]
        )
        print("point_map_by_unproj.shape:", point_map_by_unproj.shape)  # [V, H, W, 3]


if __name__ == "__main__":
    main()
