import argparse
import json
import os
import os.path as osp
import re
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from train_view_decoder_ablation import (
    GeomViewDecoderAblation,
    autocast_ctx,
    build_masks_from_batch,
    load_checkpoint,
    load_state_dict_fuzzy,
    normalize_pred_conf,
    save_debug_pack,
)


def _decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _split_csv(s):
    if not s:
        return []
    return [x for x in re.split(r"[,;\s]+", str(s).strip()) if x]


def _resolve_img_path(root, path_str, seq_names):
    s = _decode_name(path_str).strip().replace("\\", "/")
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
        return osp.join(root, s)
    return osp.join(root, s.lstrip("/"))


def _infer_mask_path(img_path):
    if not img_path:
        return None
    s = str(img_path).replace("\\", "/")
    parts = s.split("/")
    image_tokens = {"images", "images_512", "images_1024", "imgs", "img"}
    mask_tokens = ["mask", "masks", "mask_cihp", "masks_cihp"]

    def _is_cam_token(token):
        t = str(token).strip()
        if not t:
            return False
        tl = t.lower()
        if tl.startswith("camera_"):
            return True
        return re.fullmatch(r"\d+", t) is not None

    idx_image = None
    for i, p in enumerate(parts):
        if str(p).lower() in image_tokens:
            idx_image = i
            break

    idx_cam = None
    for i, p in enumerate(parts):
        if _is_cam_token(p):
            idx_cam = i
            break

    if idx_image is None and idx_cam is None:
        return None

    cands = []
    if idx_image is not None:
        prefix = list(parts[:idx_image])
        suffix = list(parts[idx_image + 1:])
        for t in mask_tokens:
            cands.append(prefix + [t] + suffix)
    if idx_cam is not None:
        prefix = list(parts[:idx_cam])
        suffix = list(parts[idx_cam:])
        for t in mask_tokens:
            cands.append(prefix + [t] + suffix)

    seen = set()
    for p2 in cands:
        out = "/".join(p2)
        base, _ = osp.splitext(out)
        cand = base + ".png"
        if cand in seen:
            continue
        seen.add(cand)
        if osp.isfile(cand):
            return cand
    return None


def _normalize_conf(conf):
    conf = conf.astype(np.float32, copy=False)
    if conf.size == 0:
        return conf
    if not np.isfinite(conf).all():
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(conf.max())
    if maxv <= 1.5:
        pass
    elif maxv <= 32.0:
        conf = conf / (maxv + 1e-8)
    elif maxv <= 255.0 + 1e-3:
        conf = conf / 255.0
    else:
        conf = conf / (maxv + 1e-8)
    return np.clip(conf, 0.0, 1.0)


def _normalize_mask(mask):
    m = mask.astype(np.float32, copy=False)
    if m.size == 0:
        return m
    if not np.isfinite(m).all():
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(m.max())
    if maxv <= 1.5:
        pass
    elif maxv <= 32.0:
        m = m / (maxv + 1e-8)
    elif maxv <= 255.0 + 1e-3:
        m = m / 255.0
    else:
        m = m / (maxv + 1e-8)
    return np.clip(m, 0.0, 1.0)


def _pick_state_dict(ckpt, prefer_ema=False):
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format invalid: expected dict.")
    has_model = ckpt.get("model", None) is not None
    has_ema = ckpt.get("ema", None) is not None
    if prefer_ema:
        if has_ema:
            return ckpt["ema"]
        if has_model:
            return ckpt["model"]
    else:
        if has_model:
            return ckpt["model"]
        if has_ema:
            return ckpt["ema"]
    raise RuntimeError("Checkpoint has neither model nor ema weights.")


def _cfg_get(ckpt_args, key, default):
    v = ckpt_args.get(key, default)
    return default if v is None else v


def _safe_tag(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _copy_if_exists(src, dst):
    if osp.isfile(src):
        shutil.copy2(src, dst)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        "infer_single_case_compare (fixed frame + fixed cameras)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="infer_single_case")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--zju_root", type=str, default=None)
    parser.add_argument("--seq_names", type=str, nargs="*", default=None)
    parser.add_argument("--geom_subdir", type=str, default=None)
    parser.add_argument("--seq_name", type=str, default="CoreView_390")
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--tgt_camera", type=str, required=True)
    parser.add_argument("--src_cameras", type=str, required=True)
    parser.add_argument("--conf_thr", type=float, default=None)
    parser.add_argument("--conf_temp", type=float, default=None)
    parser.add_argument("--vis_conf_min", type=float, default=0.0)
    parser.add_argument("--vis_conf_max", type=float, default=1.0)
    parser.add_argument("--vis_depth_min", type=float, default=0.0)
    parser.add_argument("--vis_depth_max", type=float, default=5.0)
    parser.add_argument("--vis_mask_min", type=float, default=0.0)
    parser.add_argument("--vis_mask_max", type=float, default=1.0)
    parser.add_argument("--vis_weight_min", type=float, default=0.0)
    parser.add_argument("--vis_weight_max", type=float, default=1.0)
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema", action="store_true")
    ema_group.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    zju_root = args.zju_root or _cfg_get(ckpt_args, "zju_root", "/mnt/data/zju_mocap")
    geom_subdir = args.geom_subdir or _cfg_get(ckpt_args, "geom_subdir", "vggt_geom")
    seq_name = args.seq_name
    seq_names = list(args.seq_names) if args.seq_names else [seq_name]
    seq_root = osp.join(zju_root, seq_name)
    npz_path = osp.join(seq_root, geom_subdir, f"frame_{int(args.frame_id):06d}.npz")
    if not osp.isfile(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        img_paths = data["img_paths"]
        depth = data["depth"]
        depth_conf = data["depth_conf"]
        pointmap = data["pointmap"]
        cam_names_np = data["cam_names"] if "cam_names" in data else None

    if cam_names_np is None:
        raise RuntimeError("npz missing cam_names; cannot map fixed cameras.")
    cam_names = [_decode_name(x) for x in cam_names_np]
    cam_to_idx = {name: i for i, name in enumerate(cam_names)}

    tgt_cam = args.tgt_camera.strip()
    src_cams = _split_csv(args.src_cameras)
    if len(src_cams) == 0:
        raise ValueError("src_cameras is empty.")
    if tgt_cam not in cam_to_idx:
        raise KeyError(f"target camera not found in npz cam_names: {tgt_cam}")
    missing_src = [c for c in src_cams if c not in cam_to_idx]
    if missing_src:
        raise KeyError(f"source cameras not found in npz cam_names: {missing_src}")

    tgt_idx = int(cam_to_idx[tgt_cam])
    src_idxs = [int(cam_to_idx[c]) for c in src_cams]

    src_imgs_l = []
    src_depth_l = []
    src_conf_l = []
    src_pm_l = []
    src_vids = []
    for idx in src_idxs:
        p = _resolve_img_path(zju_root, img_paths[idx], seq_names)
        img = Image.open(p).convert("RGB")
        src_imgs_l.append(TF.to_tensor(img))
        d = depth[idx]
        c = depth_conf[idx]
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]
        c = _normalize_conf(c)
        pm = pointmap[idx]
        src_depth_l.append(torch.from_numpy(d).float().unsqueeze(0))
        src_conf_l.append(torch.from_numpy(c).float().unsqueeze(0))
        src_pm_l.append(torch.from_numpy(pm).permute(2, 0, 1).float())
        src_vids.append(int(idx))

    tgt_img_path = _resolve_img_path(zju_root, img_paths[tgt_idx], seq_names)
    tgt_img_pil = Image.open(tgt_img_path).convert("RGB")
    tgt_img = TF.to_tensor(tgt_img_pil)
    d = depth[tgt_idx]
    c = depth_conf[tgt_idx]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if c.ndim == 3 and c.shape[-1] == 1:
        c = c[..., 0]
    c = _normalize_conf(c)
    pm = pointmap[tgt_idx]
    tgt_depth = torch.from_numpy(d).float().unsqueeze(0)
    tgt_conf = torch.from_numpy(c).float().unsqueeze(0)
    tgt_pm = torch.from_numpy(pm).permute(2, 0, 1).float()

    mask_path = _infer_mask_path(tgt_img_path)
    if mask_path is None:
        raise FileNotFoundError(f"mask not found for tgt image: {tgt_img_path}")
    m = np.array(Image.open(mask_path).convert("L"))
    m = _normalize_mask(m)
    tgt_fg = torch.from_numpy((m > 0.5).astype(np.float32))

    src_imgs = torch.stack(src_imgs_l, dim=0).unsqueeze(0).to(device)
    src_depth_t = torch.stack(src_depth_l, dim=0).unsqueeze(0).to(device)
    src_conf_t = torch.stack(src_conf_l, dim=0).unsqueeze(0).to(device)
    src_pm_t = torch.stack(src_pm_l, dim=0).unsqueeze(0).to(device)
    tgt_img_t = tgt_img.unsqueeze(0).to(device)
    tgt_depth_t = tgt_depth.unsqueeze(0).to(device)
    tgt_conf_t = tgt_conf.unsqueeze(0).to(device)
    tgt_pm_t = tgt_pm.unsqueeze(0).to(device)
    tgt_fg_t = tgt_fg.unsqueeze(0).to(device)
    tgt_vid_t = torch.tensor([int(tgt_idx)], dtype=torch.long, device=device)
    src_vids_t = torch.tensor([src_vids], dtype=torch.long, device=device)

    model = GeomViewDecoderAblation(
        ref_mode=str(_cfg_get(ckpt_args, "ref_mode", "first")),
        use_conf_gate=bool(_cfg_get(ckpt_args, "use_conf_gate", True)),
        use_tone=bool(_cfg_get(ckpt_args, "use_tone", True)),
        init_alpha=float(_cfg_get(ckpt_args, "init_alpha", 0.12)),
        use_view_cond=bool(_cfg_get(ckpt_args, "use_view_cond", False)),
        num_views=int(_cfg_get(ckpt_args, "num_views", 0)),
        view_dim=int(_cfg_get(ckpt_args, "view_dim", 16)),
        view_affine_strength=float(_cfg_get(ckpt_args, "view_affine_strength", 1.0)),
        view_cond_mode=str(_cfg_get(ckpt_args, "view_cond_mode", "tgt")),
        rgb_sigmoid_temp=float(_cfg_get(ckpt_args, "rgb_sigmoid_temp", 1.0)),
        conf_sigmoid_temp=float(_cfg_get(ckpt_args, "conf_sigmoid_temp", 1.0)),
        split_conf_head=bool(_cfg_get(ckpt_args, "split_conf_head", False)),
        conf_gate_detach=bool(_cfg_get(ckpt_args, "conf_gate_detach", False)),
        conf_gate_floor=float(_cfg_get(ckpt_args, "conf_gate_floor", 0.0)),
        logit_clip=float(_cfg_get(ckpt_args, "logit_clip", 0.0)),
        use_depth_head=bool(_cfg_get(ckpt_args, "use_depth_head", False)),
    )
    model.conf_thr = float(_cfg_get(ckpt_args, "conf_thr", 0.2))
    model.conf_temp = float(_cfg_get(ckpt_args, "conf_temp", 0.06))
    prefer_ema = bool(args.use_ema)
    sd = _pick_state_dict(ckpt, prefer_ema=prefer_ema)
    load_state_dict_fuzzy(model, sd)
    model.to(device)
    model.eval()

    use_amp = bool(_cfg_get(ckpt_args, "amp", False) or _cfg_get(ckpt_args, "use_amp", False))
    with torch.no_grad():
        with autocast_ctx(device=device, enabled=use_amp):
            pred_rgb, pred_conf, aux_pred = model(
                src_imgs, src_depth_t, src_conf_t, src_pm_t,
                tgt_vid=tgt_vid_t, src_vids=src_vids_t, return_aux=True
            )
            pred_conf_safe, _ = normalize_pred_conf(pred_conf)
            H, W = pred_rgb.shape[-2:]
            conf_thr = float(model.conf_thr if args.conf_thr is None else args.conf_thr)
            conf_temp = float(model.conf_temp if args.conf_temp is None else args.conf_temp)
            batch = {
                "tgt_img": tgt_img_t,
                "tgt_depth": tgt_depth_t,
                "tgt_depth_conf": tgt_conf_t,
                "tgt_conf": tgt_conf_t,
                "tgt_pointmap": tgt_pm_t,
                "tgt_fg": tgt_fg_t,
                "tgt_vid": tgt_vid_t,
                "src_vids": src_vids_t,
                "tgt_img_path": [tgt_img_path],
                "tgt_mask_path": [mask_path],
            }
            train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf, aux_masks = build_masks_from_batch(
                batch=batch,
                pred_hw=(H, W),
                device=device,
                conf_thr=conf_thr,
                conf_temp=conf_temp,
                train_min_cover=float(_cfg_get(ckpt_args, "train_min_cover", 0.10)),
                fg_thr=float(_cfg_get(ckpt_args, "fg_thr", 0.5)),
                fg_min_cover=float(_cfg_get(ckpt_args, "fg_min_cover", 0.05)),
                fg_dilate_k=int(_cfg_get(ckpt_args, "fg_dilate_k", 7)),
                fg_keep_largest_cc=bool(_cfg_get(ckpt_args, "fg_keep_largest_cc", True)),
                fg_lcc_min_pixels=int(_cfg_get(ckpt_args, "fg_lcc_min_pixels", 32)),
                fg_drop_ground=bool(_cfg_get(ckpt_args, "fg_drop_ground", False)),
                fg_ground_axis=int(_cfg_get(ckpt_args, "fg_ground_axis", 1)),
                fg_ground_q=float(_cfg_get(ckpt_args, "fg_ground_q", 0.05)),
                fg_ground_margin=float(_cfg_get(ckpt_args, "fg_ground_margin", 0.02)),
                fg_ground_min_points=int(_cfg_get(ckpt_args, "fg_ground_min_points", 64)),
                valid_min_cover=float(_cfg_get(ckpt_args, "valid_min_cover", 0.10)),
                valid_dilate_k=int(_cfg_get(ckpt_args, "valid_dilate_k", 7)),
                valid_k_max=int(_cfg_get(ckpt_args, "valid_k_max", 31)),
                bg_weight=float(_cfg_get(ckpt_args, "bg_weight", 0.05)),
                conf_raw_min=float(_cfg_get(ckpt_args, "conf_raw_min", 1.0)),
                conf_raw_max=float(_cfg_get(ckpt_args, "conf_raw_max", 8.0)),
                conf_auto_norm=bool(_cfg_get(ckpt_args, "conf_auto_norm", True)),
                conf_use_quantile=bool(_cfg_get(ckpt_args, "conf_use_quantile", True)),
                conf_qlo=float(_cfg_get(ckpt_args, "conf_qlo", 0.05)),
                conf_qhi=float(_cfg_get(ckpt_args, "conf_qhi", 0.95)),
                use_conf_in_train_mask=bool(_cfg_get(ckpt_args, "use_conf_loss_gate", True)),
                train_mask_mode=str(_cfg_get(ckpt_args, "train_mask_mode", "fg_conf")),
                pred_conf_gate=pred_conf,
                use_conf_gate=bool(_cfg_get(ckpt_args, "use_conf_gate", True)),
                conf_gate_detach=bool(_cfg_get(ckpt_args, "conf_gate_detach", False)),
                conf_gate_floor=float(_cfg_get(ckpt_args, "conf_gate_floor", 0.0)),
                conf_gate_gamma=float(_cfg_get(ckpt_args, "conf_gate_gamma", 1.0)),
                recon_gate_floor=float(_cfg_get(ckpt_args, "recon_gate_floor", _cfg_get(ckpt_args, "conf_gate_floor", 0.0))),
                recon_mask_mode=str(_cfg_get(ckpt_args, "recon_mask_mode", "fg")),
                recon_weight_renorm=bool(_cfg_get(ckpt_args, "recon_weight_renorm", False)),
                recon_weight_clip_max=float(_cfg_get(ckpt_args, "recon_weight_clip_max", 1.0)),
                require_tgt_fg=bool(_cfg_get(ckpt_args, "require_tgt_fg", True)),
                allow_fg_from_conf=bool(_cfg_get(ckpt_args, "allow_fg_from_conf", False)),
                log_mask_stats=False,
            )

    aux_dbg = {
        "pred_conf": pred_conf_safe.detach(),
        "tgt_depth_conf": aux_masks.get("tgt_depth_conf", None),
        "tgt_depth_conf_raw": aux_masks.get("tgt_depth_conf_raw", None),
        "tgt_depth": tgt_depth_t.detach(),
        "tgt_fg": tgt_fg_t.detach(),
        "tgt_img_path": [tgt_img_path],
        "tgt_mask_path": [mask_path],
        "tgt_vid": tgt_vid_t.detach(),
        "source_fg_key": aux_masks.get("source_fg_key", None),
        "fg_mask": fg_mask.detach(),
        "train_mask": train_mask.detach(),
        "recon_weight": recon_weight.detach(),
        "valid_mask": valid_mask.detach(),
    }
    if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
        aux_dbg["gate"] = aux_pred.get("gate")
    if isinstance(aux_pred, dict) and aux_pred.get("pred_depth", None) is not None:
        aux_dbg["pred_depth"] = aux_pred.get("pred_depth")

    prefix = f"single_{_safe_tag(seq_name)}_{int(args.frame_id):06d}_{_safe_tag(tgt_cam)}_{_safe_tag(geom_subdir)}"
    fixed_ranges = {
        "conf": (float(args.vis_conf_min), float(args.vis_conf_max)),
        "depth": (float(args.vis_depth_min), float(args.vis_depth_max)),
        "mask": (float(args.vis_mask_min), float(args.vis_mask_max)),
        "weight": (float(args.vis_weight_min), float(args.vis_weight_max)),
    }
    save_debug_pack(
        pred_rgb.detach(),
        tgt_img_t.detach(),
        aux_dbg,
        step=0,
        out_dir=args.out_dir,
        prefix=prefix,
        split_cat_panels=True,
        fixed_ranges=fixed_ranges,
    )

    # Compatibility aliases with old naming.
    _copy_if_exists(
        osp.join(args.out_dir, f"{prefix}_recon_weight_step000000.png"),
        osp.join(args.out_dir, "weight_native.png"),
    )
    _copy_if_exists(
        osp.join(args.out_dir, f"{prefix}_pred_raw_step000000.png"),
        osp.join(args.out_dir, "pred_native.png"),
    )
    _copy_if_exists(
        osp.join(args.out_dir, f"{prefix}_tgt_step000000.png"),
        osp.join(args.out_dir, "tgt_native.png"),
    )
    _copy_if_exists(
        osp.join(args.out_dir, f"{prefix}_cat_recon_weight_pred_tgt_step000000.png"),
        osp.join(args.out_dir, "cat_weight_pred_tgt_native.png"),
    )

    report = {
        "seq_name": seq_name,
        "frame_id": int(args.frame_id),
        "geom_subdir": str(geom_subdir),
        "tgt_camera": tgt_cam,
        "src_cameras": src_cams,
        "npz_path": npz_path,
        "tgt_img_path": tgt_img_path,
        "tgt_mask_path": mask_path,
        "ckpt": args.ckpt,
        "prefer_ema": bool(prefer_ema),
        "device": str(device),
        "paths": {
            "weight_native": osp.join(args.out_dir, "weight_native.png"),
            "pred_native": osp.join(args.out_dir, "pred_native.png"),
            "tgt_native": osp.join(args.out_dir, "tgt_native.png"),
            "cat_weight_pred_tgt_native": osp.join(args.out_dir, "cat_weight_pred_tgt_native.png"),
            "cat_pred_tgt": osp.join(args.out_dir, f"{prefix}_cat_pred_tgt_step000000.png"),
            "cat_recon_weight_pred_tgt": osp.join(args.out_dir, f"{prefix}_cat_recon_weight_pred_tgt_step000000.png"),
            "cat_gate_pred_tgt": osp.join(args.out_dir, f"{prefix}_cat_gate_pred_tgt_step000000.png"),
            "overlay": osp.join(args.out_dir, f"{prefix}_gt_with_fg_overlay_step000000.png"),
        },
    }
    with open(osp.join(args.out_dir, "single_case_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
