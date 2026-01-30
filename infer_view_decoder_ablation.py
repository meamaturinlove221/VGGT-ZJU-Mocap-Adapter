import os
import sys
import json
import csv
import re
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# --- make sure project root (this file's dir) is importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _as_namespace(d: Optional[Dict[str, Any]]) -> SimpleNamespace:
    d = d or {}
    return SimpleNamespace(**d)


def _pick_state_dict(ckpt: Dict[str, Any], prefer_ema: bool = False) -> Dict[str, Any]:
    """
    Train script saves payload:
      {"model": state_dict, "ema": state_dict or None, "args": dict, ...}
    Some other scripts might save pure state_dict.
    """
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        if prefer_ema:
            if ckpt.get("ema") is not None:
                return ckpt["ema"]
            if ckpt.get("model") is not None:
                return ckpt["model"]
        else:
            if ckpt.get("model") is not None:
                return ckpt["model"]
            if ckpt.get("ema") is not None:
                return ckpt["ema"]
    # fallback: assume it's already a state_dict
    return ckpt


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def load_state_dict_fuzzy(model: torch.nn.Module, sd: Dict[str, Any]) -> None:
    """
    Handle common wrappers: 'module.' from DDP, etc.
    """
    if not isinstance(sd, dict):
        raise TypeError(f"state_dict must be dict, got {type(sd)}")

    # try direct
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) == 0 and len(unexpected) == 0:
        return

    # try stripping 'module.'
    sd2 = _strip_prefix(sd, "module.")
    missing2, unexpected2 = model.load_state_dict(sd2, strict=False)
    # If still mismatching heavily, user will see it in logs anyway.


def build_cfg_from_ckpt_and_cli(ckpt_args: Dict[str, Any], cli: Any) -> SimpleNamespace:
    """
    We reuse train-time functions (normalize_pred_conf/build_masks_from_batch/masked_l1/save_debug_pack),
    so cfg must contain the same attributes as training args.
    Best way: start from ckpt['args'] dict, then override a few from CLI.
    """
    cfg = _as_namespace(ckpt_args)

    # --- inference-only knobs ---
    # (these may not exist in training args)
    cfg.split = getattr(cli, "split", getattr(cfg, "split", "val"))
    cfg.zju_root = cli.zju_root if cli.zju_root is not None else getattr(
        cfg, "zju_root", None)
    if cli.seq_names is not None and len(cli.seq_names) > 0:
        cfg.seq_names = cli.seq_names

    # dataset knobs (override if provided)
    if cli.num_src_views is not None:
        cfg.num_src_views = int(cli.num_src_views)
    if cli.frame_subsample is not None:
        cfg.frame_subsample = int(cli.frame_subsample)
    if cli.train_ratio is not None:
        cfg.train_ratio = float(cli.train_ratio)
    if cli.split_seed is not None:
        cfg.split_seed = int(cli.split_seed)
    if cli.view_seed is not None:
        cfg.view_seed = int(cli.view_seed)

    # model knobs (override if provided)
    if cli.ref_mode is not None:
        cfg.ref_mode = cli.ref_mode
    if cli.use_conf_gate is not None:
        cfg.use_conf_gate = bool(cli.use_conf_gate)
    if cli.use_tone is not None:
        cfg.use_tone = bool(cli.use_tone)
    if cli.init_alpha is not None:
        cfg.init_alpha = float(cli.init_alpha)
    if cli.conf_thr is not None:
        cfg.conf_thr = float(cli.conf_thr)
    if cli.conf_temp is not None:
        cfg.conf_temp = float(cli.conf_temp)
    if getattr(cli, "fg_keep_largest_cc", None) is not None:
        cfg.fg_keep_largest_cc = bool(cli.fg_keep_largest_cc)
    if getattr(cli, "fg_lcc_min_pixels", None) is not None:
        cfg.fg_lcc_min_pixels = int(cli.fg_lcc_min_pixels)
    if getattr(cli, "fg_drop_ground", None) is not None:
        cfg.fg_drop_ground = bool(cli.fg_drop_ground)
    if getattr(cli, "fg_ground_axis", None) is not None:
        cfg.fg_ground_axis = int(cli.fg_ground_axis)
    if getattr(cli, "fg_ground_q", None) is not None:
        cfg.fg_ground_q = float(cli.fg_ground_q)
    if getattr(cli, "fg_ground_margin", None) is not None:
        cfg.fg_ground_margin = float(cli.fg_ground_margin)
    if getattr(cli, "fg_ground_min_points", None) is not None:
        cfg.fg_ground_min_points = int(cli.fg_ground_min_points)
    if getattr(cli, "use_view_cond", None) is not None:
        cfg.use_view_cond = bool(cli.use_view_cond)
    if getattr(cli, "num_views", None) is not None:
        cfg.num_views = int(cli.num_views)
    if getattr(cli, "view_dim", None) is not None:
        cfg.view_dim = int(cli.view_dim)
    if getattr(cli, "view_affine_strength", None) is not None:
        cfg.view_affine_strength = float(cli.view_affine_strength)
    if getattr(cli, "view_cond_mode", None) is not None:
        cfg.view_cond_mode = str(cli.view_cond_mode)

    # safety defaults if missing (should usually be present in ckpt['args'])
    if not hasattr(cfg, "use_amp"):
        cfg.use_amp = False
    if not hasattr(cfg, "ref_mode"):
        cfg.ref_mode = "first"
    if not hasattr(cfg, "use_conf_gate"):
        cfg.use_conf_gate = True
    if not hasattr(cfg, "use_tone"):
        cfg.use_tone = True
    if not hasattr(cfg, "init_alpha"):
        cfg.init_alpha = 0.12
    if not hasattr(cfg, "conf_thr"):
        cfg.conf_thr = 0.2
    if not hasattr(cfg, "conf_temp"):
        cfg.conf_temp = 0.06
    if not hasattr(cfg, "num_src_views"):
        cfg.num_src_views = 4
    if not hasattr(cfg, "frame_subsample"):
        cfg.frame_subsample = 1
    if not hasattr(cfg, "train_ratio"):
        cfg.train_ratio = 0.9
    if not hasattr(cfg, "split_seed"):
        cfg.split_seed = 0
    if not hasattr(cfg, "view_seed"):
        cfg.view_seed = 0
    if not hasattr(cfg, "train_min_cover"):
        cfg.train_min_cover = 0.10
    if not hasattr(cfg, "fg_thr"):
        cfg.fg_thr = 0.5
    if not hasattr(cfg, "fg_min_cover"):
        cfg.fg_min_cover = 0.05
    if not hasattr(cfg, "fg_dilate_k"):
        cfg.fg_dilate_k = 7
    if not hasattr(cfg, "fg_keep_largest_cc"):
        cfg.fg_keep_largest_cc = True
    if not hasattr(cfg, "fg_lcc_min_pixels"):
        cfg.fg_lcc_min_pixels = 32
    if not hasattr(cfg, "fg_drop_ground"):
        cfg.fg_drop_ground = False
    if not hasattr(cfg, "fg_ground_axis"):
        cfg.fg_ground_axis = 1
    if not hasattr(cfg, "fg_ground_q"):
        cfg.fg_ground_q = 0.05
    if not hasattr(cfg, "fg_ground_margin"):
        cfg.fg_ground_margin = 0.02
    if not hasattr(cfg, "fg_ground_min_points"):
        cfg.fg_ground_min_points = 64
    if not hasattr(cfg, "valid_min_cover"):
        cfg.valid_min_cover = 0.10
    if not hasattr(cfg, "valid_dilate_k"):
        cfg.valid_dilate_k = 7
    if not hasattr(cfg, "valid_k_max"):
        cfg.valid_k_max = 31
    if not hasattr(cfg, "bg_weight"):
        cfg.bg_weight = 0.05
    if not hasattr(cfg, "conf_raw_min"):
        cfg.conf_raw_min = 1.0
    if not hasattr(cfg, "conf_raw_max"):
        cfg.conf_raw_max = 8.0
    if not hasattr(cfg, "conf_auto_norm"):
        cfg.conf_auto_norm = True
    if not hasattr(cfg, "conf_use_quantile"):
        cfg.conf_use_quantile = True
    if not hasattr(cfg, "conf_qlo"):
        cfg.conf_qlo = 0.05
    if not hasattr(cfg, "conf_qhi"):
        cfg.conf_qhi = 0.95
    if not hasattr(cfg, "use_view_cond"):
        cfg.use_view_cond = False
    if not hasattr(cfg, "num_views"):
        cfg.num_views = 0
    if not hasattr(cfg, "view_dim"):
        cfg.view_dim = 16
    if not hasattr(cfg, "view_affine_strength"):
        cfg.view_affine_strength = 1.0
    if not hasattr(cfg, "view_cond_mode"):
        cfg.view_cond_mode = "tgt"

    cfg.seq_names = _normalize_seq_names(getattr(cfg, "seq_names", None))
    return cfg


def _normalize_seq_names(seq_names):
    if seq_names is None:
        return None
    if isinstance(seq_names, str):
        s = seq_names.strip()
        if not s:
            return []
        return [p for p in re.split(r"[,\s]+", s) if p]
    if isinstance(seq_names, (list, tuple)):
        out = []
        for item in seq_names:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                out.extend([p for p in re.split(r"[,\s]+", s) if p])
            else:
                out.append(str(item))
        return out
    return [str(seq_names)]


def _parse_only_steps_env(env_key: str = "ONLY_STEPS") -> set:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return set()
    parts = re.split(r"[,\s;/]+", raw)
    out = set()
    for p in parts:
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _conf_stats_1d(
    x: Optional[torch.Tensor],
    qs: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99),
    max_samples: int = 200000,
) -> Optional[Dict[str, float]]:
    if x is None or not torch.is_tensor(x):
        return None
    t = x.detach().float().reshape(-1)
    if t.numel() == 0:
        return None
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return None
    if t.numel() > max_samples:
        idx = torch.randperm(t.numel(), device=t.device)[:max_samples]
        t = t[idx]
    t_cpu = t.cpu()
    stats = {
        "min": float(t_cpu.min().item()),
        "max": float(t_cpu.max().item()),
    }
    try:
        q_tensor = torch.tensor(qs, dtype=torch.float32)
        q_vals = torch.quantile(t_cpu, q_tensor).tolist()
        for q, v in zip(qs, q_vals):
            stats[f"q{int(q * 100):02d}"] = float(v)
    except Exception:
        pass
    return stats


def _adjust_conf_cfg_from_stats(cfg: SimpleNamespace, stats: Dict[str, float]) -> Dict[str, Any]:
    raw_min = float(stats.get("min", 0.0))
    raw_max = float(stats.get("max", 1.0))
    q05 = float(stats.get("q05", raw_min))
    q95 = float(stats.get("q95", raw_max))
    if q95 <= q05:
        q05, q95 = raw_min, raw_max

    base_conf_thr = float(getattr(cfg, "conf_thr", 0.2))
    base_fg_thr = float(getattr(cfg, "fg_thr", 0.5))

    if raw_max <= 1.2 and raw_min >= -0.2:
        scale_mode = "01"
        new_conf_raw_min = 0.0
        new_conf_raw_max = 1.0
        new_fg_thr = base_fg_thr
        new_conf_thr = base_conf_thr
    elif raw_max > 16.0 and raw_min >= -0.2:
        scale_mode = "0255"
        new_conf_raw_min = 0.0
        new_conf_raw_max = 255.0
        new_fg_thr = base_fg_thr
        new_conf_thr = base_conf_thr
    else:
        scale_mode = "raw"
        new_conf_raw_min = q05
        new_conf_raw_max = q95
        new_fg_thr = base_fg_thr
        if base_fg_thr <= 1.2:
            new_fg_thr = new_conf_raw_min + base_fg_thr * (
                new_conf_raw_max - new_conf_raw_min
            )
        new_conf_thr = base_conf_thr
        if base_conf_thr > 1.2:
            denom = max(new_conf_raw_max - new_conf_raw_min, 1e-6)
            new_conf_thr = (base_conf_thr - new_conf_raw_min) / denom
            new_conf_thr = max(0.0, min(1.0, new_conf_thr))

    changed = False

    def _set(name: str, value: float):
        nonlocal changed
        cur = getattr(cfg, name, None)
        if cur is None or abs(float(cur) - float(value)) > 1e-6:
            setattr(cfg, name, value)
            changed = True

    _set("conf_raw_min", new_conf_raw_min)
    _set("conf_raw_max", new_conf_raw_max)
    _set("fg_thr", new_fg_thr)
    _set("conf_thr", new_conf_thr)

    conf_thr_raw = new_conf_raw_min + new_conf_thr * (
        new_conf_raw_max - new_conf_raw_min
    )
    return {
        "scale_mode": scale_mode,
        "changed": changed,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "q05": q05,
        "q95": q95,
        "conf_raw_min": new_conf_raw_min,
        "conf_raw_max": new_conf_raw_max,
        "fg_thr": new_fg_thr,
        "conf_thr": new_conf_thr,
        "conf_thr_raw": conf_thr_raw,
    }


def _filter_existing_seq_names(seq_names, zju_root):
    existing = []
    missing = []
    for name in seq_names:
        geom_dir = os.path.join(zju_root, name, "vggt_geom")
        if os.path.isdir(geom_dir):
            existing.append(name)
        else:
            missing.append((name, geom_dir))
    return existing, missing


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "infer_view_decoder_ablation (matches train_view_decoder_ablation.py)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="infer_vis")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_samples", type=int,
                        default=50, help="-1 means all")

    # default to raw weights; opt-in EMA when explicitly requested
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema", action="store_true",
                           help="use EMA weights if ckpt has them")
    ema_group.add_argument("--no_ema", action="store_true",
                           help="(deprecated) disable EMA even if ckpt has it")

    # optional overrides (otherwise taken from ckpt['args'])
    parser.add_argument("--zju_root", type=str, default=None)
    parser.add_argument("--seq_names", type=str, nargs="*", default=None)

    parser.add_argument("--num_src_views", type=int, default=None)
    parser.add_argument("--frame_subsample", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--view_seed", type=int, default=None)

    parser.add_argument("--ref_mode", type=str,
                        default=None, choices=["first", "mean"])
    parser.add_argument("--use_conf_gate", type=int,
                        default=None, choices=[0, 1])
    parser.add_argument("--use_tone", type=int, default=None, choices=[0, 1])
    parser.add_argument("--init_alpha", type=float, default=None)
    parser.add_argument("--conf_thr", type=float, default=None)
    parser.add_argument("--conf_temp", type=float, default=None)
    parser.add_argument("--fg_keep_largest_cc", type=int, default=None, choices=[0, 1])
    parser.add_argument("--fg_lcc_min_pixels", type=int, default=None)
    parser.add_argument("--fg_drop_ground", type=int, default=None, choices=[0, 1],
                        help="Remove ground from fg mask using pointmap height quantile")
    parser.add_argument("--fg_ground_axis", type=int, default=None,
                        help="Vertical axis index in pointmap (0=x,1=y,2=z)")
    parser.add_argument("--fg_ground_q", type=float, default=None,
                        help="Ground height quantile within fg&valid region")
    parser.add_argument("--fg_ground_margin", type=float, default=None,
                        help="Margin above ground height to keep")
    parser.add_argument("--fg_ground_min_points", type=int, default=None,
                        help="Min fg points to estimate ground height per sample")
    parser.add_argument("--use_view_cond", nargs="?", const=1,
                        default=None, type=int, choices=[0, 1])
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--view_dim", type=int, default=None)
    parser.add_argument("--view_affine_strength", type=float, default=None)
    parser.add_argument("--view_cond_mode", type=str, default=None,
                        choices=["tgt", "tgt_src_mean"])

    args = parser.parse_args()

    # --- import train-time helpers (exact same metric/mask/vis code) ---
    from train_view_decoder_ablation import (
        load_checkpoint,
        autocast_ctx,
        normalize_pred_conf,
        build_masks_from_batch,
        masked_l1,
        psnr,
        ssim,
        save_debug_pack,
    )
    from zju_dataset_view import ZJUViewSynthDataset
    from view_decoder_ablation import GeomViewDecoderAblation

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- load ckpt ---
    # 或者直接 load_checkpoint(args.ckpt, "cpu")
    ckpt = load_checkpoint(args.ckpt, device="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    cfg = build_cfg_from_ckpt_and_cli(ckpt_args, args)

    if cfg.zju_root is None:
        raise ValueError(
            "zju_root is None. Provide --zju_root or make sure ckpt['args']['zju_root'] exists.")

    if not hasattr(cfg, "seq_names") or cfg.seq_names is None or len(cfg.seq_names) == 0:
        raise ValueError(
            "seq_names is empty. Provide --seq_names ... or make sure ckpt['args']['seq_names'] exists.")

    seq_names = list(cfg.seq_names)
    existing, missing = _filter_existing_seq_names(seq_names, cfg.zju_root)
    if missing:
        if len(existing) == 0:
            missing_str = ", ".join(
                [f"{n} ({p})" for n, p in missing])
            raise FileNotFoundError(
                f"[ZJUViewSynthDataset] geom dir not found: {missing_str}")
        if len(seq_names) > 1:
            missing_str = ", ".join(
                [f"{n} ({p})" for n, p in missing])
            print(
                f"[warn] skip missing sequences (geom dir not found): {missing_str}")
            cfg.seq_names = existing

    # deterministic views for val/test to be stable
    deterministic_views = (cfg.split != "train")

    ds = ZJUViewSynthDataset(
        root=cfg.zju_root,
        seq_names=cfg.seq_names,
        split=cfg.split,
        num_src_views=int(cfg.num_src_views),
        frame_subsample=int(cfg.frame_subsample),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        deterministic_views=deterministic_views,
        view_seed=int(cfg.view_seed),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    if cfg.use_view_cond and int(getattr(cfg, "num_views", 0)) <= 0:
        cfg.num_views = int(getattr(ds, "num_views", 0))
        if cfg.num_views <= 0:
            raise ValueError(
                "Cannot auto-infer num_views from dataset. Please set --num_views explicitly."
            )
    if cfg.use_view_cond:
        print(f"[info] num_views = {cfg.num_views}")

    # --- build model exactly like training ---
    model = GeomViewDecoderAblation(
        ref_mode=getattr(cfg, "ref_mode", "first"),
        use_conf_gate=bool(getattr(cfg, "use_conf_gate", True)),
        use_tone=bool(getattr(cfg, "use_tone", True)),
        init_alpha=float(getattr(cfg, "init_alpha", 0.12)),
        use_view_cond=bool(getattr(cfg, "use_view_cond", False)),
        num_views=int(getattr(cfg, "num_views", 0)),
        view_dim=int(getattr(cfg, "view_dim", 16)),
        view_affine_strength=float(getattr(cfg, "view_affine_strength", 1.0)),
        view_cond_mode=str(getattr(cfg, "view_cond_mode", "tgt")),
        rgb_sigmoid_temp=float(getattr(cfg, "rgb_sigmoid_temp", 1.0)),
        conf_sigmoid_temp=float(getattr(cfg, "conf_sigmoid_temp", 1.0)),
        split_conf_head=bool(getattr(cfg, "split_conf_head", False)),
        conf_gate_detach=bool(getattr(cfg, "conf_gate_detach", False)),
        conf_gate_floor=float(getattr(cfg, "conf_gate_floor", 0.0)),
        logit_clip=float(getattr(cfg, "logit_clip", 0.0)),
    )
    # these are used inside model forward
    model.conf_thr = float(getattr(cfg, "conf_thr", 0.6))
    model.conf_temp = float(getattr(cfg, "conf_temp", 1.0))

    prefer_ema = bool(getattr(args, "use_ema", False))
    if (not prefer_ema) and isinstance(ckpt, dict):
        if ckpt.get("model") is None and ckpt.get("ema") is not None:
            print("[warn] ckpt has no raw weights; falling back to EMA.")
    sd = _pick_state_dict(ckpt, prefer_ema=prefer_ema)
    load_state_dict_fuzzy(model, sd)

    model.to(device)
    model.eval()

    # --- meta for naming ---
    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    global_step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0

    # --- eval loop (same as training val loop) ---
    sum_wl1 = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    n = 0

    per_frame_jsonl = os.path.join(
        args.out_dir, f"metrics_{cfg.split}_per_frame.jsonl")
    per_frame_csv = os.path.join(
        args.out_dir, f"metrics_{cfg.split}_per_frame.csv")
    per_frame_idx = 0
    by_view = {}

    def _acc_view(vid, l1v, ps):
        v = int(vid)
        entry = by_view.get(v)
        if entry is None:
            entry = {"sum_l1": 0.0, "sum_psnr": 0.0, "n": 0}
            by_view[v] = entry
        entry["sum_l1"] += float(l1v)
        entry["sum_psnr"] += float(ps)
        entry["n"] += 1

    max_items_to_save = (1000000000 if getattr((globals().get('cfg') or globals().get('args')),'split','val') != 'train' else 2)

    only_steps = _parse_only_steps_env()
    if only_steps:
        print(f"[info] ONLY_STEPS (infer debug pack) = {sorted(only_steps)}")
    saved = 0
    conf_stats_done = False

    with open(per_frame_jsonl, "w", encoding="utf-8") as jf, open(
        per_frame_csv, "w", encoding="utf-8", newline=""
    ) as cf, torch.no_grad():
        fieldnames = ["sample_idx", "batch_idx",
                      "batch_pos", "wl1", "psnr", "ssim"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for i, batch in enumerate(tqdm(dl, desc=f"infer[{cfg.split}]")):
            if args.num_samples >= 0 and n >= args.num_samples:
                break

            # move tensors
            def to_dev(x):
                return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

            batch = {k: to_dev(v) for k, v in batch.items()}

            src_imgs = batch["src_imgs"]
            src_depth = batch["src_depth"]
            src_depth_conf = batch["src_depth_conf"]
            src_pointmap = batch["src_pointmap"]
            tgt_img = batch["tgt_img"]
            tgt_vid = batch.get("tgt_vid", None)
            src_vids = batch.get("src_vids", None)

            use_amp = bool(getattr(cfg, "amp", False)
                           or getattr(cfg, "use_amp", False))
            with autocast_ctx(device=device, enabled=use_amp):
                pred_rgb, pred_conf, aux_pred = model(
                    src_imgs,
                    src_depth,
                    src_depth_conf,
                    src_pointmap,
                    tgt_vid=tgt_vid,
                    src_vids=src_vids,
                    return_aux=True,
                )

                pred_conf_safe, _ = normalize_pred_conf(pred_conf)
                H, W = pred_rgb.shape[-2:]
                train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf, aux_masks = build_masks_from_batch(
                    batch=batch,
                    pred_hw=(H, W),
                    device=device,
                    conf_thr=float(cfg.conf_thr),
                    conf_temp=float(cfg.conf_temp),
                    train_min_cover=float(cfg.train_min_cover),
                    fg_thr=float(cfg.fg_thr),
                    fg_min_cover=float(cfg.fg_min_cover),
                    fg_dilate_k=int(cfg.fg_dilate_k),
                    fg_keep_largest_cc=bool(cfg.fg_keep_largest_cc),
                    fg_lcc_min_pixels=int(cfg.fg_lcc_min_pixels),
                    fg_drop_ground=bool(getattr(cfg, "fg_drop_ground", False)),
                    fg_ground_axis=int(getattr(cfg, "fg_ground_axis", 1)),
                    fg_ground_q=float(getattr(cfg, "fg_ground_q", 0.05)),
                    fg_ground_margin=float(getattr(cfg, "fg_ground_margin", 0.02)),
                    fg_ground_min_points=int(getattr(cfg, "fg_ground_min_points", 64)),
                    valid_min_cover=float(cfg.valid_min_cover),
                    valid_dilate_k=int(cfg.valid_dilate_k),
                    valid_k_max=int(cfg.valid_k_max),
                    bg_weight=float(cfg.bg_weight),
                    conf_raw_min=float(cfg.conf_raw_min),
                    conf_raw_max=float(cfg.conf_raw_max),
                    conf_auto_norm=bool(cfg.conf_auto_norm),
                    conf_use_quantile=bool(cfg.conf_use_quantile),
                    conf_qlo=float(cfg.conf_qlo),
                    conf_qhi=float(cfg.conf_qhi),
                    use_conf_in_train_mask=bool(
                        getattr(cfg, "use_conf_loss_gate", True)),
                    train_mask_mode=str(
                        getattr(cfg, "train_mask_mode", "fg_conf")),
                    pred_conf_gate=pred_conf,
                    use_conf_gate=bool(getattr(cfg, "use_conf_gate", True)),
                    conf_gate_detach=bool(
                        getattr(cfg, "conf_gate_detach", False)),
                    conf_gate_floor=float(
                        getattr(cfg, "conf_gate_floor", 0.0)),
                    conf_gate_gamma=float(
                        getattr(cfg, "conf_gate_gamma", 1.0)),
                    recon_gate_floor=float(
                        getattr(cfg, "recon_gate_floor",
                                getattr(cfg, "conf_gate_floor", 0.0))),
                    recon_mask_mode=str(
                        getattr(cfg, "recon_mask_mode", "fg")),
                    recon_weight_renorm=bool(
                        getattr(cfg, "recon_weight_renorm", False)),
                    recon_weight_clip_max=float(
                        getattr(cfg, "recon_weight_clip_max", 1.0)),
                )

                if not conf_stats_done:
                    conf_stats_done = True
                    stats = _conf_stats_1d(
                        aux_masks.get("tgt_depth_conf_raw", None))
                    if stats is not None:
                        q01 = stats.get("q01", float("nan"))
                        q05 = stats.get("q05", float("nan"))
                        q50 = stats.get("q50", float("nan"))
                        q95 = stats.get("q95", float("nan"))
                        q99 = stats.get("q99", float("nan"))
                        print(
                            "[tgt_depth_conf_raw] "
                            f"min={stats['min']:.6f} max={stats['max']:.6f} "
                            f"q01={q01:.6f} q05={q05:.6f} q50={q50:.6f} "
                            f"q95={q95:.6f} q99={q99:.6f}"
                        )
                        adj = _adjust_conf_cfg_from_stats(cfg, stats)
                        print(
                            "[conf_scale] "
                            f"mode={adj['scale_mode']} "
                            f"conf_raw_min={adj['conf_raw_min']:.6f} "
                            f"conf_raw_max={adj['conf_raw_max']:.6f} "
                            f"fg_thr={adj['fg_thr']:.6f} "
                            f"conf_thr={adj['conf_thr']:.6f} "
                            f"conf_thr_raw={adj['conf_thr_raw']:.6f}"
                        )
                        if adj["changed"]:
                            train_mask, valid_mask, fg_mask, recon_weight, tgt_depth_conf, aux_masks = build_masks_from_batch(
                                batch=batch,
                                pred_hw=(H, W),
                                device=device,
                                conf_thr=float(cfg.conf_thr),
                                conf_temp=float(cfg.conf_temp),
                                train_min_cover=float(cfg.train_min_cover),
                                fg_thr=float(cfg.fg_thr),
                                fg_min_cover=float(cfg.fg_min_cover),
                                fg_dilate_k=int(cfg.fg_dilate_k),
                                fg_keep_largest_cc=bool(cfg.fg_keep_largest_cc),
                                fg_lcc_min_pixels=int(cfg.fg_lcc_min_pixels),
                                fg_drop_ground=bool(getattr(cfg, "fg_drop_ground", False)),
                                fg_ground_axis=int(getattr(cfg, "fg_ground_axis", 1)),
                                fg_ground_q=float(getattr(cfg, "fg_ground_q", 0.05)),
                                fg_ground_margin=float(getattr(cfg, "fg_ground_margin", 0.02)),
                                fg_ground_min_points=int(getattr(cfg, "fg_ground_min_points", 64)),
                                valid_min_cover=float(cfg.valid_min_cover),
                                valid_dilate_k=int(cfg.valid_dilate_k),
                                valid_k_max=int(cfg.valid_k_max),
                                bg_weight=float(cfg.bg_weight),
                                conf_raw_min=float(cfg.conf_raw_min),
                                conf_raw_max=float(cfg.conf_raw_max),
                                conf_auto_norm=bool(cfg.conf_auto_norm),
                                conf_use_quantile=bool(cfg.conf_use_quantile),
                                conf_qlo=float(cfg.conf_qlo),
                                conf_qhi=float(cfg.conf_qhi),
                                use_conf_in_train_mask=bool(
                                    getattr(cfg, "use_conf_loss_gate", True)),
                                train_mask_mode=str(
                                    getattr(cfg, "train_mask_mode", "fg_conf")),
                                pred_conf_gate=pred_conf,
                                use_conf_gate=bool(
                                    getattr(cfg, "use_conf_gate", True)),
                                conf_gate_detach=bool(
                                    getattr(cfg, "conf_gate_detach", False)),
                                conf_gate_floor=float(
                                    getattr(cfg, "conf_gate_floor", 0.0)),
                                conf_gate_gamma=float(
                                    getattr(cfg, "conf_gate_gamma", 1.0)),
                                recon_gate_floor=float(
                                    getattr(cfg, "recon_gate_floor",
                                            getattr(cfg, "conf_gate_floor", 0.0))),
                                recon_mask_mode=str(
                                    getattr(cfg, "recon_mask_mode", "fg")),
                                recon_weight_renorm=bool(
                                    getattr(cfg, "recon_weight_renorm", False)),
                                recon_weight_clip_max=float(
                                    getattr(cfg, "recon_weight_clip_max", 1.0)),
                            )

                bsz = int(pred_rgb.shape[0])
                for b in range(bsz):
                    if args.num_samples >= 0 and n >= args.num_samples:
                        break
                    pred_b = pred_rgb[b:b + 1]
                    tgt_b = tgt_img[b:b + 1]
                    w_b = recon_weight[b:b + 1]

                    wl1_b = float(masked_l1(pred_b, tgt_b, w_b).item())
                    p_b = float(psnr(pred_b, tgt_b).item())
                    s_b = float(ssim(pred_b, tgt_b).item())
                    if tgt_vid is not None:
                        vid = int(tgt_vid.view(-1)[b].item())
                        _acc_view(vid, wl1_b, p_b)

                    row = {
                        "sample_idx": per_frame_idx,
                        "batch_idx": i,
                        "batch_pos": b,
                        "wl1": wl1_b,
                        "psnr": p_b,
                        "ssim": s_b,
                    }
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    writer.writerow(row)
                    per_frame_idx += 1

                    sum_wl1 += wl1_b
                    sum_psnr += p_b
                    sum_ssim += s_b
                    n += 1

            # save a few debug packs
            step = locals().get('step', locals().get('sample_idx', locals().get('batch_idx', saved)))
            _os=__import__('os')
            raw_only_steps=_os.environ.get('ONLY_STEPS','')
            only_steps=set(int(x) for x in raw_only_steps.replace(';',',').split(',') if x.strip().isdigit())
            sid=int(locals().get('sample_idx', locals().get('batch_idx', locals().get('idx', locals().get('i', 0)))))
            if ((only_steps and (sid in only_steps)) or ((not only_steps) and (saved < max_items_to_save))):
                aux_dbg = {
                    "pred_conf": pred_conf_safe.detach(),
                    "tgt_depth_conf": aux_masks.get("tgt_depth_conf", None),
                    "tgt_depth_conf_raw": aux_masks.get("tgt_depth_conf_raw", None),
                    "fg_mask": fg_mask.detach(),
                    "train_mask": train_mask.detach(),
                    "recon_weight": recon_weight.detach(),
                    "valid_mask": valid_mask.detach(),
                }
                if isinstance(aux_pred, dict) and aux_pred.get("gate", None) is not None:
                    aux_dbg["gate"] = aux_pred.get("gate")
                save_debug_pack(
                    pred_rgb,
                    tgt_img,
                    aux_dbg,
                    global_step + i,
                    out_dir=args.out_dir,
                    prefix=f"infer_{cfg.split}_e{epoch:03d}",
                )
                saved += 1

    mean_wl1 = sum_wl1 / max(n, 1)
    mean_psnr = sum_psnr / max(n, 1)
    mean_ssim = sum_ssim / max(n, 1)

    print(
        f"[infer:{cfg.split}] N={n} mean weighted-L1={mean_wl1:.6f} PSNR={mean_psnr:.2f} SSIM={mean_ssim:.4f}")
    if len(by_view) > 0:
        rows = []
        for vid, s in by_view.items():
            n_v = max(1, s["n"])
            rows.append((vid, s["sum_l1"] / n_v, s["sum_psnr"] / n_v, n_v))
        rows.sort(key=lambda x: x[1], reverse=True)
        print("[by_view] worst->best (vid, meanL1, meanPSNR, N):")
        for row in rows:
            print(f"  {row}")
        by_view_csv = os.path.join(
            args.out_dir, f"metrics_{cfg.split}_by_view.csv")
        with open(by_view_csv, "w", encoding="utf-8", newline="") as bvf:
            writer = csv.writer(bvf)
            writer.writerow(["vid", "mean_l1", "mean_psnr", "n"])
            for row in rows:
                writer.writerow(row)
    summary = {
        "split": cfg.split,
        "ckpt": args.ckpt,
        "prefer_ema": prefer_ema,
        "epoch_in_ckpt": epoch,
        "step_in_ckpt": global_step,
        "N": n,
        "mean_weighted_L1": mean_wl1,
        "mean_PSNR": mean_psnr,
        "mean_SSIM": mean_ssim,
        "zju_root": cfg.zju_root,
        "seq_names": list(cfg.seq_names),
        "num_src_views": int(cfg.num_src_views),
        "frame_subsample": int(cfg.frame_subsample),
        "train_ratio": float(cfg.train_ratio),
        "split_seed": int(cfg.split_seed),
        "view_seed": int(cfg.view_seed),
    }
    with open(os.path.join(args.out_dir, f"metrics_{cfg.split}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
