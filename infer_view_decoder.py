# infer_view_decoder.py
import torch
from torchvision.utils import save_image
from zju_dataset_view import ZJUViewSynthDataset
from view_decoder import GeomViewDecoder
CKPT = "checkpoints_view_decoder/viewdec_epoch05.pth"


def main():
    dataset = ZJUViewSynthDataset(
        root=ZJU_ROOT,
        seq_names=["CoreView_390"],
        num_src_views=3,
        split="val",  # 你可以在 dataset 里简单加一个 split 控制
    )

    model = GeomViewDecoder().cuda()
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    os.makedirs("vis_baseline", exist_ok=True)

    with torch.no_grad():
        for i in range(0, min(len(dataset), 50)):
            batch = dataset[i]
            # 单张样本自己加个 batch 维
            src_imgs = batch["src_imgs"].to(device)
            src_depth = batch["src_depth"].to(device)
            src_depth_conf = batch["src_depth_conf"].to(device)
            src_pointmap = batch["src_pointmap"].to(device)
            tgt_img = batch["tgt_img"].to(device)

            pred_rgb, pred_conf = model(
                src_imgs, src_depth, src_depth_conf, src_pointmap)

            pair = torch.cat([pred, tgt_img], dim=3)  # (1,3,H,2W)
            save_image(pair, f"vis_baseline/frame_{i:04d}.png")

    pred = pred_rgb  # 为了不动后面其它代码


if __name__ == "__main__":
    main()
