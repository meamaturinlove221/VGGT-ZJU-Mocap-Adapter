# overfit_one.py
import torch
from torch.utils.data import DataLoader
from zju_dataset_view import ZJUViewSynthDataset
from view_decoder import ViewDecoder  # 你当前用的 decoder 类
from train_view_decoder import compute_fg_bg_loss  # 或者直接复制那段函数

device = "cuda"


def main():
    dataset = ZJUViewSynthDataset(
        root="F:/datasets/ZJU_MoCap/data/zju_mocap",
        seq_name="CoreView_390",
        geom_root="F:/datasets/ZJU_MoCap/zju_geom/CoreView_390",
        num_src_views=2
    )
    # 只取前 1 个样本
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    src_imgs = batch["src_imgs"].to(device)
    src_depth = batch["src_depth"].to(device)
    tgt_img = batch["tgt_img"].to(device)
    tgt_depth = batch["tgt_depth"].to(device)

    B, S, C, H, W = src_imgs.shape

    model = ViewDecoder().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1000):
        optim.zero_grad()
        pred = model(src_imgs, src_depth)  # (B,3,H,W)
        loss = compute_fg_bg_loss(pred, tgt_img, tgt_depth)
        loss.backward()
        optim.step()

        if step % 50 == 0:
            print(f"step {step}, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
