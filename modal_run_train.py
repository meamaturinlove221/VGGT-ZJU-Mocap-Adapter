import os
import sys
import json
import time
import shlex
import shutil
import inspect
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import modal

# --------------------------
# Constants (container paths)
# --------------------------
MNT_DATA = "/mnt/data"   # data volume mount
MNT_OUT = "/mnt/out"    # output volume mount
MNT_CODE = "/mnt/code"   # mounted repo code

DEFAULT_DATA_VOL = "vggt-zju-data"
DEFAULT_OUT_VOL = "vggt-out"

# Your screenshots show:
#   vggt-zju-data / zju_mocap / CoreView_390 / ...
DEFAULT_ZJU_ROOT = f"{MNT_DATA}/zju_mocap"
DEFAULT_ARCHIVES = f"{MNT_DATA}/archives"


# --------------------------
# Env helpers
# --------------------------
def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None else v


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _split_seq_names(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    # allow comma/space separated
    parts = []
    for chunk in s.replace(",", " ").split():
        if chunk.strip():
            parts.append(chunk.strip())
    return parts


# --------------------------
# GPU parsing (robust alias)
# --------------------------
def _make_gpu_from_spec(spec: str):
    spec0 = (spec or "").strip()
    if not spec0:
        return None
    s = spec0.strip().lower().replace("_", "-")

    if s in ("none", "cpu", "0"):
        return None

    # normalize common aliases
    alias = {
        "a100-80g": "a100-80gb",
        "a100-80": "a100-80gb",
        "a100-80gb": "a100-80gb",
        "a100-40g": "a100-40gb",
        "a100-40": "a100-40gb",
        "a100-40gb": "a100-40gb",
        "a100": "a100",
        "h100": "h100",
        "l40s": "l40s",
        "l40": "l40s",
    }
    s = alias.get(s, s)

    # Try to construct modal.gpu.* objects if available
    try:
        gpu_mod = modal.gpu
    except Exception:
        gpu_mod = None

    def _ctor(cls, **kwargs):
        try:
            sig = inspect.signature(cls)
            ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return cls(**ok)
        except Exception:
            try:
                return cls()
            except Exception:
                return None

    if gpu_mod is not None:
        if s == "a100-80gb":
            cls = getattr(gpu_mod, "A100", None)
            if cls is not None:
                # Modal historically used size="80GB"
                g = _ctor(cls, size="80GB")
                if g is None:
                    g = _ctor(cls)
                return g
        if s == "a100-40gb":
            cls = getattr(gpu_mod, "A100", None)
            if cls is not None:
                g = _ctor(cls, size="40GB")
                if g is None:
                    g = _ctor(cls)
                return g
        if s == "a100":
            cls = getattr(gpu_mod, "A100", None)
            if cls is not None:
                g = _ctor(cls)
                return g
        if s == "h100":
            cls = getattr(gpu_mod, "H100", None)
            if cls is not None:
                g = _ctor(cls)
                return g
        if s == "l40s":
            cls = getattr(gpu_mod, "L40S", None)
            if cls is not None:
                g = _ctor(cls)
                return g

    # Fallback: pass through string (some Modal versions accept it)
    # Prefer canonical string for A100-80GB:
    if s == "a100-80gb":
        return "A100-80GB"
    if s == "a100-40gb":
        return "A100-40GB"
    if s == "a100":
        return "A100"
    if s == "h100":
        return "H100"
    if s == "l40s":
        return "L40S"

    return spec0


# --------------------------
# Config
# --------------------------
@dataclass
class Cfg:
    # volumes & paths
    data_vol: str
    out_vol: str
    code_dir: str
    zju_root: str
    archives_dir: str

    # run
    seq_names: list[str]
    gpu_spec: str
    timeout_sec: int

    # training script
    train_script: str
    train_args_extra: str

    # hyperparams
    epochs: int
    batch_size: int
    accum_steps: int
    lr: float
    lr_schedule: str
    warmup_steps: int
    min_lr_ratio: float
    early_stop: int

    # misc toggles
    best_by: str
    use_ema: bool
    ema_decay: float
    ema_start_step: int

    split_conf_head: bool
    conf_head_lr_mult: float

    bg_weight: float
    train_mask_mode: str
    recon_mask_mode: str
    recon_weight_renorm: bool
    recon_weight_clip_max: float

    conf_use_quantile: bool
    conf_qlo: float
    conf_qhi: float

    amp: bool
    tf32: bool

    debug_train_every: int
    debug_val_every_epoch: int
    split_cat_panels: bool

    @staticmethod
    def from_env() -> "Cfg":
        data_vol = _env("VGGT_DATA_VOL", DEFAULT_DATA_VOL)
        out_vol = _env("VGGT_OUT_VOL",  DEFAULT_OUT_VOL)
        code_dir = _env("VGGT_CODE_DIR", os.getcwd())

        # IMPORTANT: based on your volume screenshots:
        # vggt-zju-data / zju_mocap / CoreView_390 / ...
        zju_root = _env("VGGT_ZJU_ROOT", DEFAULT_ZJU_ROOT).strip()
        if zju_root == "":
            zju_root = DEFAULT_ZJU_ROOT

        archives_dir = _env("VGGT_ARCHIVES_DIR", DEFAULT_ARCHIVES)

        seq_env = _env("VGGT_SEQ", "CoreView_390")
        seq_names = _split_seq_names(_env("VGGT_SEQ_NAMES", seq_env))

        gpu_spec = _env("VGGT_GPU", "A100-80GB")
        timeout_sec = _env_int("VGGT_TIMEOUT_SEC", 24 * 3600)

        train_script = _env("TRAIN_SCRIPT", "train_view_decoder_ablation.py")
        train_args_extra = _env("TRAIN_ARGS", "")

        return Cfg(
            data_vol=data_vol,
            out_vol=out_vol,
            code_dir=code_dir,
            zju_root=zju_root,
            archives_dir=archives_dir,
            seq_names=seq_names,
            gpu_spec=gpu_spec,
            timeout_sec=timeout_sec,
            train_script=train_script,
            train_args_extra=train_args_extra,
            epochs=_env_int("VGGT_EPOCHS", 50),
            batch_size=_env_int("VGGT_BATCH", 3),
            accum_steps=_env_int("VGGT_ACCUM", 1),
            lr=_env_float("VGGT_LR", 5e-5),
            lr_schedule=_env("VGGT_LR_SCHEDULE", "cosine"),
            warmup_steps=_env_int("VGGT_WARMUP", 400),
            min_lr_ratio=_env_float("VGGT_MIN_LR_RATIO", 0.1),
            early_stop=_env_int("VGGT_EARLY_STOP", 15),
            best_by=_env("VGGT_BEST_BY", "raw_psnr"),
            use_ema=_env_bool("VGGT_USE_EMA", True),
            ema_decay=_env_float("VGGT_EMA_DECAY", 0.99),
            ema_start_step=_env_int("VGGT_EMA_START_STEP", 0),
            split_conf_head=_env_bool("VGGT_SPLIT_CONF_HEAD", True),
            conf_head_lr_mult=_env_float("VGGT_CONF_HEAD_LR_MULT", 2.0),
            bg_weight=_env_float("VGGT_BG_WEIGHT", 0.05),
            train_mask_mode=_env("VGGT_TRAIN_MASK_MODE", "fg_conf"),
            recon_mask_mode=_env("VGGT_RECON_MASK_MODE", "valid"),
            recon_weight_renorm=_env_bool("VGGT_RECON_WEIGHT_RENORM", True),
            recon_weight_clip_max=_env_float(
                "VGGT_RECON_WEIGHT_CLIP_MAX", 1.0),
            conf_use_quantile=_env_bool("VGGT_CONF_USE_QUANTILE", True),
            conf_qlo=_env_float("VGGT_CONF_QLO", 0.05),
            conf_qhi=_env_float("VGGT_CONF_QHI", 0.95),
            amp=_env_bool("VGGT_AMP", True),
            tf32=_env_bool("VGGT_TF32", True),
            debug_train_every=_env_int("VGGT_DEBUG_TRAIN_EVERY", 200),
            debug_val_every_epoch=_env_int("VGGT_DEBUG_VAL_EVERY_EPOCH", 1),
            split_cat_panels=_env_bool("VGGT_SPLIT_CAT_PANELS", True),
        )


# --------------------------
# Modal setup (read env at import time)
# --------------------------
CFG_IMPORT = Cfg.from_env()

DATA_VOL = modal.Volume.from_name(CFG_IMPORT.data_vol, create_if_missing=False)
OUT_VOL = modal.Volume.from_name(CFG_IMPORT.out_vol,  create_if_missing=False)

GPU_OBJ = _make_gpu_from_spec(CFG_IMPORT.gpu_spec)

# Keep image simple; your repo typically installs torch in its own way.
# If你已经在你自己的 modal image 里装好了依赖，也可以把 pip_install 删掉。
IMAGE = (
    modal.Image.from_registry('pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime')
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "pyyaml",
        "tqdm",
        "pillow",
        "matplotlib",
        "scipy",
    )
)

APP_NAME = _env("VGGT_APP_NAME", "vggt-train")
app = modal.App(APP_NAME)


def _print_tree(root: Path, max_items: int = 200):
    root = Path(root)
    if not root.exists():
        print(f"[tree] missing: {root}")
        return
    items = []
    for x in sorted(root.iterdir(), key=lambda p: p.name):
        items.append(f"{x.name}{'/' if x.is_dir() else ''}")
        if len(items) >= max_items:
            items.append("... (truncated)")
            break
    print(f"[tree] {root} -> {len(items)} entries")
    for it in items[:30]:
        print("  -", it)
    if len(items) > 30:
        print("  - ...")


def _find_seq_dir(zju_root: Path, seq: str) -> Path | None:
    zju_root = Path(zju_root)
    cand = zju_root / seq
    if (cand / "vggt_geom").exists():
        return cand
    # fallback: maybe zju_root itself is seq dir
    if (zju_root / "vggt_geom").exists():
        return zju_root
    return None


def _ensure_dataset(cfg: Cfg):
    """
    Ensure that zju_root/seq/vggt_geom exists.
    Based on your screenshot, correct layout is:
      /mnt/data/zju_mocap/CoreView_390/vggt_geom
    """
    zju_root = Path(cfg.zju_root)
    if not zju_root.exists():
        print(f"[data] zju_root not found: {zju_root}")
        print(f"[data] listing {Path(MNT_DATA)}:")
        _print_tree(Path(MNT_DATA), max_items=200)

    missing = []
    for seq in cfg.seq_names:
        seq_dir = _find_seq_dir(zju_root, seq)
        if seq_dir is None or not (seq_dir / "vggt_geom").exists():
            missing.append(seq)

    if not missing:
        print(f"[data] ok: zju_root={zju_root} has all seqs {cfg.seq_names}")
        return

    # If missing, try extraction from archives (only if archives exist)
    archives = Path(cfg.archives_dir)
    print(f"[data] missing seq(s): {missing}")
    print(f"[data] trying archives at: {archives}")
    _print_tree(archives, max_items=200)

    for seq in missing:
        tar_path = archives / f"{seq}.tar"
        if not tar_path.exists():
            # try assembling parts
            parts = sorted(archives.glob(f"{seq}.tar.part*"))
            if parts:
                tar_path = archives / f"{seq}.tar"
                if not tar_path.exists():
                    print(
                        f"[data] assembling {tar_path} from {len(parts)} parts...")
                    with open(tar_path, "wb") as w:
                        for p in parts:
                            print("  [part]", p.name)
                            with open(p, "rb") as r:
                                shutil.copyfileobj(
                                    r, w, length=16 * 1024 * 1024)
                    print(f"[data] assembled: {tar_path}")
            else:
                print(f"[data] no tar or parts found for {seq} in {archives}")
                continue

        # Extract into zju_root (expects it contains seq folder)
        print(f"[data] extracting {tar_path} -> {zju_root}")
        zju_root.mkdir(parents=True, exist_ok=True)

        # system tar is faster than python tarfile for huge archives
        subprocess.check_call(
            ["tar", "xf", str(tar_path), "-C", str(zju_root)])

        # Verify layout; if tar extracted files directly (no seq folder),
        # create a symlink at parent to satisfy expected zju_root/seq layout
        seq_dir = _find_seq_dir(zju_root, seq)
        if seq_dir is None:
            # Search one level deep for vggt_geom
            geom_hits = list(zju_root.glob("**/vggt_geom"))
            if geom_hits:
                real_seq_dir = geom_hits[0].parent
                print(f"[data] found vggt_geom at: {real_seq_dir}")
                parent = real_seq_dir.parent
                link = parent / seq
                if not link.exists() and str(link) != str(real_seq_dir):
                    try:
                        link.symlink_to(real_seq_dir, target_is_directory=True)
                        print(f"[data] symlinked {link} -> {real_seq_dir}")
                    except Exception as e:
                        print(f"[data] symlink failed: {e}")
            else:
                print("[data] extraction done but still can't find vggt_geom")

    # persist changes
    try:
        DATA_VOL.commit()
        print("[data] committed data volume")
    except Exception as e:
        print(f"[data] commit skipped/failed: {e}")


def _build_train_cmd(cfg: Cfg) -> list[str]:
    """
    Map env->train_view_decoder_ablation.py args.
    """
    tag = time.strftime("%Y%m%d_%H%M%S")
    seq_tag = cfg.seq_names[0] if cfg.seq_names else "SEQ"
    log_dir = f"{MNT_OUT}/vggt/runs/{seq_tag}_{tag}"
    ckpt_dir = f"{MNT_OUT}/vggt/ckpt/{seq_tag}_{tag}"

    cmd = [
        sys.executable,
        str(Path(MNT_CODE) / cfg.train_script),
        "--zju_root", cfg.zju_root,
        "--seq_names", ",".join(cfg.seq_names),
        "--epochs", str(cfg.epochs),
        "--batch_size", str(cfg.batch_size),
        "--accum_steps", str(cfg.accum_steps),
        "--lr", str(cfg.lr),
        "--lr_schedule", cfg.lr_schedule,
        "--warmup_steps", str(cfg.warmup_steps),
        "--min_lr_ratio", str(cfg.min_lr_ratio),
        "--early_stop", str(cfg.early_stop),
        "--best_by", cfg.best_by,
        "--ema_decay", str(cfg.ema_decay),
        "--ema_start_step", str(cfg.ema_start_step),
        "--conf_head_lr_mult", str(cfg.conf_head_lr_mult),
        "--bg_weight", str(cfg.bg_weight),
        "--train_mask_mode", cfg.train_mask_mode,
        "--recon_mask_mode", cfg.recon_mask_mode,
        "--recon_weight_clip_max", str(cfg.recon_weight_clip_max),
        "--conf_qlo", str(cfg.conf_qlo),
        "--conf_qhi", str(cfg.conf_qhi),
        "--debug_train_every", str(cfg.debug_train_every),
        "--debug_val_every_epoch", str(cfg.debug_val_every_epoch),
        "--log_dir", log_dir,
        "--ckpt_dir", ckpt_dir,
    ]

    if cfg.use_ema:
        # default in script is True, but explicit is fine
        cmd.append("--use_ema")
    else:
        cmd.append("--no_use_ema")

    if cfg.split_conf_head:
        cmd.append("--split_conf_head")

    if cfg.recon_weight_renorm:
        cmd.append("--recon_weight_renorm")
    else:
        cmd.append("--no_recon_weight_renorm")

    if cfg.conf_use_quantile:
        cmd.append("--conf_use_quantile")
    else:
        cmd.append("--no_conf_use_quantile")

    if cfg.amp:
        cmd.append("--amp")
    else:
        cmd.append("--no_amp")

    if cfg.tf32:
        cmd.append("--tf32")
    else:
        cmd.append("--no_tf32")

    if cfg.split_cat_panels:
        cmd.append("--split_cat_panels")
    else:
        cmd.append("--no_split_cat_panels")

    extra = (cfg.train_args_extra or "").strip()
    if extra:
        cmd += shlex.split(extra)

    return cmd


IMAGE = IMAGE.add_local_dir(os.environ.get('VGGT_CODE_DIR','.'), remote_path='/mnt/code')
@app.function(
    image=IMAGE,
    gpu=GPU_OBJ,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={MNT_DATA: DATA_VOL, MNT_OUT: OUT_VOL},
)
def run_remote(cfg_json: str):
    cfg = Cfg(**json.loads(cfg_json))

    print("[remote] cfg:")
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")

    # Ensure code exists
    code_dir = Path(MNT_CODE)
    if not code_dir.exists():
        # [auto] code mount fallback begin
        code_src = os.path.dirname(__file__)
        try:
            if not os.path.exists(MNT_CODE):
                os.makedirs(os.path.dirname(MNT_CODE), exist_ok=True)
                try:
                    os.symlink(code_src, MNT_CODE)
                    print('[remote] created symlink: ' + str(MNT_CODE) + ' -> ' + str(code_src))
                except Exception as e:
                    print('[remote] cannot symlink ' + str(MNT_CODE) + ' : ' + str(e) + ' ; will use code_src=' + str(code_src))
        except Exception as e:
            print('[remote] code mount fallback failed: ' + str(e) + ' ; continue')
        try:
            import sys as _sys
            if code_src not in _sys.path: _sys.path.insert(0, code_src)
            if os.path.exists(MNT_CODE) and MNT_CODE not in _sys.path: _sys.path.insert(0, MNT_CODE)
        except Exception as e:
            print('[remote] sys.path fallback failed: ' + str(e))
        print('[remote] warning: missing code mount; fallback applied; continue')
        # [auto] code mount fallback end

    os.chdir(code_dir)
    print(f"[remote] cwd={Path.cwd()}")

    # Validate dataset path based on your screenshots:
    # /mnt/data/zju_mocap/CoreView_390/...
    _ensure_dataset(cfg)

    # Show dataset tree head
    zju_root = Path(cfg.zju_root)
    _print_tree(zju_root, max_items=120)
    for seq in cfg.seq_names[:2]:
        _print_tree(zju_root / seq, max_items=80)

    cmd = _build_train_cmd(cfg)
    print("[remote] running cmd:")
    print(" ", " ".join(cmd))

    # [auto] fix: debug_val_every_epoch is a flag
    # remove accidental value after --debug_val_every_epoch (it is a flag)
    if '--debug_val_every_epoch' in cmd:
        i = cmd.index('--debug_val_every_epoch')
        if i+1 < len(cmd) and (not str(cmd[i+1]).startswith('-')):
            cmd.pop(i+1)

    subprocess.check_call(cmd)

    # Persist outputs
    try:
        OUT_VOL.commit()
        print("[remote] committed out volume")
    except Exception as e:
        print(f"[remote] out commit skipped/failed: {e}")


@app.local_entrypoint()
def main():
    cfg = Cfg.from_env()
    if not cfg.seq_names:
        raise SystemExit(
            "VGGT_SEQ/VGGT_SEQ_NAMES is empty; set at least one seq like CoreView_390")

    # Print minimal sanity based on your volume screenshots
    print("[local] NOTE: dataset layout (from your screenshots) is:")
    print("        /mnt/data/zju_mocap/CoreView_390/...")

    print("[local] cfg summary:")
    for k, v in asdict(cfg).items():
        if k in ("train_args_extra",):
            continue
        print(f"  {k} = {v}")
    if cfg.train_args_extra.strip():
        print("  train_args_extra =", cfg.train_args_extra)

    run_remote.remote(json.dumps(asdict(cfg)))