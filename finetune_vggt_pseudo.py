import argparse
import json
import os
import os.path as osp
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _split_tokens(raw: str) -> List[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [x for x in re.split(r"[,\s;/|]+", s) if x]


def _normalize_seq_names(seq_names) -> List[str]:
    if seq_names is None:
        return []
    if isinstance(seq_names, str):
        return _split_tokens(seq_names)
    if isinstance(seq_names, (list, tuple)):
        out = []
        for x in seq_names:
            out.extend(_split_tokens(str(x)))
        return out
    return _split_tokens(str(seq_names))


def _resolve_img_path(path_str: str, zju_root: str, seq_names: Sequence[str]) -> str:
    s = str(path_str).strip().replace("\\", "/")
    if osp.exists(s):
        return s
    if osp.isabs(s):
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
    s = s.lstrip("/")
    return osp.join(zju_root, s)


def _to_depth01(conf_like: torch.Tensor) -> torch.Tensor:
    x = conf_like.float()
    mx = float(x.max().item()) if x.numel() > 0 else 0.0
    if mx <= 1.5:
        return x.clamp(0.0, 1.0)
    if mx <= 32.0:
        return (x / (mx + 1e-8)).clamp(0.0, 1.0)
    if mx <= 255.0 + 1e-3:
        return (x / 255.0).clamp(0.0, 1.0)
    return (x / (mx + 1e-8)).clamp(0.0, 1.0)


def _safe_resize_like(x: torch.Tensor, ref_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref_hw:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref_hw, mode="nearest")
    return F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)


def _augment_images(imgs: torch.Tensor, jitter: float, noise_std: float) -> torch.Tensor:
    # imgs: (B, V, 3, H, W) in [0,1]
    if jitter <= 0 and noise_std <= 0:
        return imgs
    b, v = imgs.shape[:2]
    out = imgs.clone()
    for bi in range(b):
        for vi in range(v):
            x = out[bi, vi]
            if jitter > 0:
                alpha = 1.0 + random.uniform(-jitter, jitter)  # contrast
                beta = random.uniform(-jitter, jitter)  # brightness
                x = x * alpha + beta
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
            out[bi, vi] = x.clamp(0.0, 1.0)
    return out


@dataclass
class Sample:
    img_paths: List[str]
    depth: np.ndarray
    depth_conf: np.ndarray
    pointmap: np.ndarray


class PseudoGeomDataset(Dataset):
    def __init__(
        self,
        zju_root: str,
        seq_names: Sequence[str],
        cam_names: Optional[Sequence[str]] = None,
        max_frames: int = 0,
        geom_subdir: str = "vggt_geom",
    ):
        self.zju_root = str(zju_root)
        self.seq_names = [str(s) for s in seq_names]
        self.cam_names = set([str(c) for c in (cam_names or [])])
        self.items: List[tuple[str, str]] = []
        for seq in self.seq_names:
            gdir = Path(self.zju_root) / seq / geom_subdir
            if not gdir.is_dir():
                continue
            files = sorted([p for p in gdir.glob("*.npz") if p.is_file()])
            if max_frames > 0:
                files = files[: max_frames]
            for p in files:
                self.items.append((seq, str(p)))
        if not self.items:
            raise RuntimeError(
                f"no pseudo geometry found under {self.zju_root} for seq_names={self.seq_names}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        seq, path = self.items[idx]
        with np.load(path, allow_pickle=True) as data:
            img_paths = [str(x) for x in data["img_paths"].tolist()]
            cam_names = [str(x) for x in data["cam_names"].tolist()] if "cam_names" in data else []
            depth = np.asarray(data["depth"])
            depth_conf = np.asarray(data["depth_conf"])
            pointmap = np.asarray(data["pointmap"])

        if self.cam_names and cam_names:
            keep = [i for i, c in enumerate(cam_names) if c in self.cam_names]
            if len(keep) >= 2:
                img_paths = [img_paths[i] for i in keep]
                depth = depth[keep]
                depth_conf = depth_conf[keep]
                pointmap = pointmap[keep]

        img_paths = [
            _resolve_img_path(p, self.zju_root, self.seq_names) for p in img_paths
        ]
        return Sample(
            img_paths=img_paths,
            depth=depth,
            depth_conf=depth_conf,
            pointmap=pointmap,
        )


def _sample_to_tensors(sample: Sample, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = load_and_preprocess_images(sample.img_paths).unsqueeze(0).to(device)  # (1,V,3,H,W)

    depth = torch.as_tensor(sample.depth, device=device).float()
    if depth.ndim == 3:
        depth = depth[..., None]
    depth = depth.unsqueeze(0)  # (1,V,H,W,1)

    conf = torch.as_tensor(sample.depth_conf, device=device).float()
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    conf = conf.unsqueeze(0)  # (1,V,H,W)

    point = torch.as_tensor(sample.pointmap, device=device).float()
    if point.ndim != 4 or point.shape[-1] != 3:
        raise RuntimeError(f"unexpected pointmap shape: {tuple(point.shape)}")
    point = point.unsqueeze(0)  # (1,V,H,W,3)

    return imgs, depth, conf, point


def parse_args():
    parser = argparse.ArgumentParser("finetune_vggt_pseudo")
    parser.add_argument("--zju_root", type=str, default=os.environ.get("VGGT_ZJU_ROOT", "/mnt/data/zju_mocap"))
    parser.add_argument("--seq_names", type=str, default=os.environ.get("VGGT_SEQ_NAMES", "CoreView_390"))
    parser.add_argument("--cam_names", type=str, default=os.environ.get("VGGT_CAM_NAMES", ""))
    parser.add_argument("--pretrained_ckpt", type=str, default=os.environ.get("VGGT_CKPT", "model.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=int(os.environ.get("VGGT_MAX_FRAMES", "0") or 0))
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_depth", type=float, default=1.0)
    parser.add_argument("--lambda_point", type=float, default=0.5)
    parser.add_argument("--lambda_conf", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=0.12)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--geom_subdir", type=str, default="vggt_geom")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=os.environ.get("VGGT_DEVICE", "auto"))
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[info] ignored unknown args: {unknown}")
    return args


def _resolve_device(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"", "auto", "none"}:
        return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    if s.startswith("cuda") and torch.cuda.device_count() <= 0:
        return "cpu"
    return s


def _load_state_dict(ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


def _strip_prefix_if_present(sd: dict, prefix: str) -> dict:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(str(k).startswith(prefix) for k in keys):
        return {str(k)[len(prefix):]: v for k, v in sd.items()}
    return sd


def _load_model_compat(model: torch.nn.Module, ckpt_path: str) -> None:
    sd = _load_state_dict(ckpt_path)
    if not isinstance(sd, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(sd)}")

    sd = _strip_prefix_if_present(sd, "module.")
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(str(k) for k in sd.keys())
    matched = len(model_keys & ckpt_keys)
    if matched <= 0:
        raise RuntimeError(
            f"no matching keys between checkpoint and model, ckpt={ckpt_path}"
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        "[finetune] load_state_dict strict=False",
        f"matched={matched}",
        f"missing={len(missing)}",
        f"unexpected={len(unexpected)}",
    )
    if unexpected:
        print("[finetune] unexpected(sample)=", unexpected[:8])
    if missing:
        print("[finetune] missing(sample)=", missing[:8])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    print(f"[finetune] device={device}")

    seq_names = _normalize_seq_names(args.seq_names)
    if not seq_names:
        raise RuntimeError("seq_names is empty")
    cam_names = _split_tokens(args.cam_names)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_path = Path(args.log_dir) / "finetune_vggt_metrics.jsonl"

    ds = PseudoGeomDataset(
        zju_root=args.zju_root,
        seq_names=seq_names,
        cam_names=cam_names,
        max_frames=int(args.max_frames),
        geom_subdir=str(args.geom_subdir),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda b: b[0])
    print(f"[finetune] samples={len(ds)} seq={seq_names}")

    model = VGGT(enable_track=False).to(device)
    _load_model_compat(model, args.pretrained_ckpt)

    # Freeze aggregator and camera head; tune depth/point heads.
    for p in model.parameters():
        p.requires_grad = False
    if model.depth_head is not None:
        for p in model.depth_head.parameters():
            p.requires_grad = True
    if model.point_head is not None:
        for p in model.point_head.parameters():
            p.requires_grad = True
    if model.aggregator is not None:
        model.aggregator.eval()
    if model.camera_head is not None:
        model.camera_head.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable params found")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    step = 0
    for epoch in range(int(args.epochs)):
        model.train()
        if model.aggregator is not None:
            model.aggregator.eval()
        if model.camera_head is not None:
            model.camera_head.eval()

        loss_sum = 0.0
        count = 0
        for sample in dl:
            imgs, depth_tgt, conf_tgt_raw, point_tgt = _sample_to_tensors(sample, device=device)
            imgs_aug = _augment_images(imgs, jitter=float(args.jitter), noise_std=float(args.noise_std))

            agg_tokens_list, ps_idx = model.aggregator(imgs_aug)
            depth_pred, conf_pred_raw = model.depth_head(agg_tokens_list, imgs_aug, ps_idx)
            point_pred, _ = model.point_head(agg_tokens_list, imgs_aug, ps_idx)

            # Align pseudo target resolution if needed.
            if depth_pred.shape[-3:-1] != depth_tgt.shape[-3:-1]:
                dt = depth_tgt.permute(0, 1, 4, 2, 3).reshape(-1, 1, depth_tgt.shape[2], depth_tgt.shape[3])
                dt = _safe_resize_like(dt, depth_pred.shape[-3:-1], mode="bilinear")
                depth_tgt = dt.reshape(depth_tgt.shape[0], depth_tgt.shape[1], 1, dt.shape[-2], dt.shape[-1]).permute(0, 1, 3, 4, 2)

            if conf_tgt_raw.shape[-2:] != conf_pred_raw.shape[-2:]:
                ct = conf_tgt_raw.reshape(-1, 1, conf_tgt_raw.shape[-2], conf_tgt_raw.shape[-1])
                ct = _safe_resize_like(ct, conf_pred_raw.shape[-2:], mode="nearest")
                conf_tgt_raw = ct.reshape(conf_tgt_raw.shape[0], conf_tgt_raw.shape[1], ct.shape[-2], ct.shape[-1])

            if point_pred.shape[-3:-1] != point_tgt.shape[-3:-1]:
                pt = point_tgt.permute(0, 1, 4, 2, 3).reshape(-1, 3, point_tgt.shape[2], point_tgt.shape[3])
                pt = _safe_resize_like(pt, point_pred.shape[-3:-1], mode="bilinear")
                point_tgt = pt.reshape(point_tgt.shape[0], point_tgt.shape[1], 3, pt.shape[-2], pt.shape[-1]).permute(0, 1, 3, 4, 2)

            conf_tgt = _to_depth01(conf_tgt_raw)
            conf_pred = _to_depth01(conf_pred_raw)
            valid = (depth_tgt[..., 0] > 1e-6).float()
            w = (valid * conf_tgt).detach()

            depth_abs = (depth_pred - depth_tgt).abs()[..., 0]
            point_abs = (point_pred - point_tgt).abs().mean(dim=-1)
            conf_abs = (conf_pred - conf_tgt).abs()

            denom = w.sum() + 1e-6
            loss_depth = (depth_abs * w).sum() / denom
            loss_point = (point_abs * w).sum() / denom
            loss_conf = (conf_abs * valid).sum() / (valid.sum() + 1e-6)

            loss = (
                float(args.lambda_depth) * loss_depth
                + float(args.lambda_point) * loss_point
                + float(args.lambda_conf) * loss_conf
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            count += 1
            loss_sum += float(loss.item())
            if (step % 20) == 0:
                msg = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_depth": float(loss_depth.item()),
                    "loss_point": float(loss_point.item()),
                    "loss_conf": float(loss_conf.item()),
                    "weight_mean": float(w.mean().item()),
                }
                print("[finetune]", msg)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        mean_loss = loss_sum / max(1, count)
        print(f"[finetune] epoch={epoch} mean_loss={mean_loss:.6f}")
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "mean_loss": mean_loss}, ensure_ascii=False) + "\n")

    out_last = Path(args.ckpt_dir) / "model_ft_zju_last.pt"
    out_best = Path(args.ckpt_dir) / "model_ft_zju.pt"
    torch.save(model.state_dict(), out_last)
    torch.save(model.state_dict(), out_best)
    print(f"[finetune] saved: {out_best}")


if __name__ == "__main__":
    main()
