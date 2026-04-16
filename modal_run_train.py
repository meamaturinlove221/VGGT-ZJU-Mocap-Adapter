import os
import sys
import json
import time
import shlex
import select
import subprocess
import zipfile
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path

import modal

# --------------------------
# Env helpers
# --------------------------


def _env(key: str, default: str | None = None) -> str:
    v = os.environ.get(key)
    if v is None:
        return "" if default is None else default
    return v


def _env_is_set(key: str) -> bool:
    v = os.environ.get(key)
    return (v is not None) and (str(v).strip() != "")


def _env_int(key: str, default: int) -> int:
    s = _env(key, str(default)).strip()
    try:
        return int(s)
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    s = _env(key, str(default)).strip()
    try:
        return float(s)
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    s = _env(key, str(default)).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _split_seq_names(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return parts


def _split_csv_list(s: str) -> list[str]:
    """Split env var like 'a,b c' into ['a','b','c'] (comma/space supported)."""
    s = (s or "").strip()
    if not s:
        return []
    parts: list[str] = []
    for chunk in s.replace(",", " ").split():
        if chunk.strip():
            parts.append(chunk.strip())
    return parts


def _sanitize_env_path(value: str, key: str) -> str:
    """
    Sanitize path-like env values.
    This guards against accidental control characters introduced by nested
    Python string parsing (e.g. '\\v' becoming VT 0x0b).
    """
    s = (value or "").strip()
    if not s:
        return s

    if not any(ord(ch) < 32 for ch in s):
        return s

    repaired: list[str] = []
    for ch in s:
        oc = ord(ch)
        if oc >= 32:
            repaired.append(ch)
            continue
        # Map common escape control chars back to textual form.
        if ch == "\x0b":  # vertical tab (\v)
            repaired.append("\\v")
        elif ch == "\t":
            repaired.append("\\t")
        elif ch == "\n":
            repaired.append("\\n")
        elif ch == "\r":
            repaired.append("\\r")
        elif ch == "\f":
            repaired.append("\\f")
        elif ch == "\a":
            repaired.append("\\a")
        elif ch == "\b":
            repaired.append("\\b")
        # Unknown control chars are dropped.

    out = "".join(repaired).strip()
    print(f"[local] [warn] {key} had control characters; sanitized to: {out!r}")
    return out


def _resolve_local_code_dir() -> str:
    raw = _sanitize_env_path(_env("VGGT_CODE_DIR", "."), "VGGT_CODE_DIR")
    if not raw:
        raw = "."
    p = Path(raw).expanduser()
    if p.exists():
        return str(p)
    cwd = Path.cwd()
    print(
        f"[local] [warn] VGGT_CODE_DIR not found: {raw!r}; fallback to cwd: {cwd}")
    return str(cwd)


def _apply_profile_defaults(cfg: "Cfg") -> "Cfg":
    profile = _env("VGGT_PROFILE", "").strip().lower()
    if not profile:
        return cfg

    if profile != "phase5_final":
        print(
            f"[local] [warn] unknown VGGT_PROFILE={profile!r}; ignore profile defaults")
        return cfg

    print("[local] apply profile defaults: VGGT_PROFILE=phase5_final")
    if not _env_is_set("VGGT_MODE"):
        cfg.mode = "infer"
    if not _env_is_set("VGGT_SEQ_NAMES"):
        cfg.seq_names = ["CoreView_390"]
    if not _env_is_set("VGGT_GEOM_SUBDIR"):
        cfg.geom_subdir = "vggt_geom_ft_20260208_044454"
    if not _env_is_set("VGGT_INFER_CKPT"):
        cfg.infer_ckpt = (
            "/mnt/out/viewdec_ablation/"
            "CoreView_390_20260208_110514/ckpt/viewdec_ablation_best.pth"
        )
    if not _env_is_set("VGGT_INFER_SPLIT"):
        cfg.infer_split = "val"
    if not _env_is_set("VGGT_INFER_USE_EMA"):
        cfg.infer_use_ema = False
    if not _env_is_set("VGGT_INFER_BATCH_SIZE"):
        cfg.infer_batch_size = 8
    if not _env_is_set("VGGT_INFER_NUM_WORKERS"):
        cfg.infer_num_workers = 8
    if not _env_is_set("VGGT_INFER_NUM_SAMPLES"):
        cfg.infer_num_samples = -1
    if not _env_is_set("VGGT_INFER_OUT_DIR"):
        cfg.infer_out_dir = "/mnt/out/infer_viewdec/CoreView_390_phase5_final_release"

    return cfg


def _normalize_gpu_spec(value: str | None) -> str | None:
    spec = (value or "").strip()
    if not spec:
        return None
    if spec.lower() in ("none", "cpu", "false", "0"):
        return None
    return spec


# --------------------------
# Config
# --------------------------


@dataclass
class Cfg:
    # code + mounts
    code_dir: str
    mnt_code: str
    mnt_data: str
    mnt_out: str

    # data
    zju_root: str
    seq_names: list[str]

    # --- pipeline mode ---
    mode: str = "train"  # "precompute", "train", or "infer"
    precompute_script: str = "precompute_zju_vggt_geom.py"
    # optional; if set, will be copied to {mnt_code}/model.pt if needed
    precompute_ckpt: str = ""
    precompute_args_extra: str = ""
    infer_script: str = "infer_view_decoder_ablation.py"
    infer_ckpt: str = ""
    infer_split: str = "val"
    infer_out_dir: str = ""
    infer_num_samples: int = -1
    infer_batch_size: int = 8
    infer_num_workers: int = 8
    infer_use_ema: bool = True
    infer_args_extra: str = ""
    geom_subdir: str = "vggt_geom"
    cam_names: list[str] = field(default_factory=list)
    max_frames: int = 0  # 0=all
    pointmap_source: str = "depth_unproject"
    point_head_frame: str = "auto"
    precompute_unproject_impl: str = "legacy"
    precompute_mv_support_on: bool = False
    precompute_mv_support_tol_abs: float = 0.06
    precompute_mv_support_tol_rel: float = 0.10
    precompute_mv_support_stride: int = 2
    precompute_mv_support_mode: str = "linear"
    precompute_mv_support_floor: float = 0.05
    precompute_mv_support_gamma: float = 1.0
    precompute_mv_support_clip_thr: float = 0.20
    precompute_mv_support_clip_floor: float = 0.30
    precompute_mv_support_hard_thr: float = -1.0
    precompute_mv_conf_valid_floor: float = 0.02
    precompute_mv_support_save: bool = False
    precompute_mv_support_save_raw_conf: bool = False
    precompute_mv_support_region_mode: str = "auto"
    precompute_mv_support_fg_mask_source: str = "auto"
    precompute_mv_support_fg_erode_px: int = 5
    precompute_mv_support_fg_preserve_px: int = 5
    precompute_batch_frames: int = 6
    precompute_mv_support_view_chunk: int = 8
    precompute_image_workers: int = 16
    precompute_save_workers: int = 4
    precompute_max_pending_saves: int = 8
    precompute_save_compressed: bool = False

    # run (modal)
    gpu_spec: str = "A100-80GB"
    gpu_spec_train: str = "A100-80GB"
    gpu_spec_precompute: str = "A100-40GB"
    gpu_spec_infer: str = "A100-40GB"
    precompute_cpu: float = 16.0
    precompute_memory_mb: int = 65536
    train_cpu: float = 16.0
    train_memory_mb: int = 65536
    timeout_sec: int = 24 * 60 * 60

    # training
    train_script: str = "train_view_decoder_ablation.py"
    train_args_extra: str = ""
    epochs: int = 50
    batch_size: int = 3
    accum_steps: int = 1
    lr: float = 5e-5
    lr_schedule: str = "cosine"
    warmup_steps: int = 400
    min_lr_ratio: float = 0.1
    bg_weight: float = 0.05
    train_mask_mode: str = "fg_conf"
    recon_mask_mode: str = "fg"
    require_tgt_fg: bool = True
    allow_fg_from_conf: bool = False
    mask_cover_min: float = 0.01
    mask_cover_max: float = 0.80
    mask_sanity_mode: str = "warn"
    conf_use_quantile_mask: bool = True
    conf_qlo: float = 0.05
    conf_qhi: float = 0.95
    conf_sup_use_quantile: bool = False
    conf_sup_gamma: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    best_by: str = "ema"
    use_view_cond: bool = False
    view_cond_mode: str = "tgt"
    view_select_mode: str = "uniform_yaw"
    yaw_jitter_deg: float = 20.0
    yaw_phase_jitter_deg: float = 20.0
    yaw_axis_x: int = 0
    yaw_axis_z: int = 2
    yaw_center_mode: str = "pointmap"
    mosaic_every_steps: int = 200
    mosaic_num_targets: int = 3
    mosaic_num_src_views: int = 6
    mosaic_tile_size: int = 300
    mosaic_point_stride: int = 24
    mosaic_seed: int = 2026
    tf32: bool = True
    amp: bool = True
    strict_deterministic: bool = False
    min_improve: float = 1e-4
    early_stop: int = 0

    # volumes
    data_vol: str = "vggt-zju-data"
    out_vol: str = "vggt-out"
    archives_dir: str = "/mnt/data/archives"

    # misc
    nan_check_every: int = 200
    debug_fixed_batch: bool = False
    debug_fixed_index: int = 0

    @staticmethod
    def from_env() -> "Cfg":
        seq_names = _split_seq_names(_env("VGGT_SEQ_NAMES", "CoreView_390"))
        if not seq_names:
            raise RuntimeError("VGGT_SEQ_NAMES is empty")

        legacy_gpu_spec = _env("VGGT_GPU_SPEC", "").strip()
        gpu_spec_train = _env(
            "VGGT_GPU_SPEC_TRAIN",
            legacy_gpu_spec or "A100-80GB",
        ).strip()
        gpu_spec_precompute = _env(
            "VGGT_GPU_SPEC_PRECOMPUTE",
            legacy_gpu_spec or "A100-40GB",
        ).strip()
        gpu_spec_infer = _env(
            "VGGT_GPU_SPEC_INFER",
            legacy_gpu_spec or "A100-40GB",
        ).strip()

        cfg = Cfg(
            # mounts
            code_dir=_resolve_local_code_dir(),
            mnt_code=_env("VGGT_MNT_CODE", "/mnt/code"),
            mnt_data=_env("VGGT_MNT_DATA", "/mnt/data"),
            mnt_out=_env("VGGT_MNT_OUT", "/mnt/out"),
            # data
            zju_root=_env("VGGT_ZJU_ROOT", "/mnt/data/zju_mocap"),
            seq_names=seq_names,
            mode=_env("VGGT_MODE", "train").strip() or "train",
            precompute_script=_env(
                "VGGT_PRECOMPUTE_SCRIPT", "precompute_zju_vggt_geom.py").strip()
            or "precompute_zju_vggt_geom.py",
            precompute_ckpt=_env("VGGT_PRECOMPUTE_CKPT", "").strip(),
            precompute_args_extra=_env("VGGT_PRECOMPUTE_ARGS_EXTRA", "").strip(),
            infer_script=_env("VGGT_INFER_SCRIPT",
                              "infer_view_decoder_ablation.py"),
            infer_ckpt=_env("VGGT_INFER_CKPT", "").strip(),
            infer_split=(_env("VGGT_INFER_SPLIT", "val").strip() or "val"),
            infer_out_dir=_env("VGGT_INFER_OUT_DIR", "").strip(),
            infer_num_samples=_env_int("VGGT_INFER_NUM_SAMPLES", -1),
            infer_batch_size=_env_int("VGGT_INFER_BATCH_SIZE", 8),
            infer_num_workers=_env_int("VGGT_INFER_NUM_WORKERS", 8),
            infer_use_ema=_env_bool("VGGT_INFER_USE_EMA", True),
            infer_args_extra=_env("VGGT_INFER_ARGS_EXTRA", ""),
            geom_subdir=(_env("VGGT_GEOM_SUBDIR", "vggt_geom").strip() or "vggt_geom"),
            cam_names=_split_csv_list(_env("VGGT_CAM_NAMES", "")),
            max_frames=_env_int("VGGT_MAX_FRAMES", 0),
            pointmap_source=(_env("VGGT_POINTMAP_SOURCE", "depth_unproject").strip() or "depth_unproject"),
            point_head_frame=(_env("VGGT_POINT_HEAD_FRAME", "auto").strip() or "auto"),
            precompute_unproject_impl=(_env("VGGT_UNPROJECT_IMPL", "legacy").strip() or "legacy"),
            precompute_mv_support_on=_env_bool("VGGT_MV_SUPPORT_ON", False),
            precompute_mv_support_tol_abs=_env_float("VGGT_MV_SUPPORT_TOL_ABS", 0.06),
            precompute_mv_support_tol_rel=_env_float("VGGT_MV_SUPPORT_TOL_REL", 0.10),
            precompute_mv_support_stride=_env_int("VGGT_MV_SUPPORT_STRIDE", 2),
            precompute_mv_support_mode=(_env("VGGT_MV_SUPPORT_MODE", "linear").strip().lower() or "linear"),
            precompute_mv_support_floor=_env_float("VGGT_MV_SUPPORT_FLOOR", 0.05),
            precompute_mv_support_gamma=_env_float("VGGT_MV_SUPPORT_GAMMA", 1.0),
            precompute_mv_support_clip_thr=_env_float("VGGT_MV_SUPPORT_CLIP_THR", 0.20),
            precompute_mv_support_clip_floor=_env_float("VGGT_MV_SUPPORT_CLIP_FLOOR", 0.30),
            precompute_mv_support_hard_thr=_env_float("VGGT_MV_SUPPORT_HARD_THR", -1.0),
            precompute_mv_conf_valid_floor=_env_float("VGGT_MV_CONF_VALID_FLOOR", 0.02),
            precompute_mv_support_save=_env_bool("VGGT_MV_SUPPORT_SAVE", False),
            precompute_mv_support_save_raw_conf=_env_bool("VGGT_MV_SUPPORT_SAVE_RAW_CONF", False),
            precompute_mv_support_region_mode=(_env("VGGT_MV_SUPPORT_REGION_MODE", "auto").strip().lower() or "auto"),
            precompute_mv_support_fg_mask_source=(_env("VGGT_MV_SUPPORT_FG_MASK_SOURCE", "auto").strip().lower() or "auto"),
            precompute_mv_support_fg_erode_px=_env_int("VGGT_MV_SUPPORT_FG_ERODE_PX", 5),
            precompute_mv_support_fg_preserve_px=_env_int("VGGT_MV_SUPPORT_FG_PRESERVE_PX", 5),
            precompute_batch_frames=_env_int("VGGT_PRECOMPUTE_BATCH_FRAMES", 6),
            precompute_mv_support_view_chunk=_env_int("VGGT_PRECOMPUTE_MV_SUPPORT_VIEW_CHUNK", 8),
            precompute_image_workers=_env_int("VGGT_PRECOMPUTE_IMAGE_WORKERS", 16),
            precompute_save_workers=_env_int("VGGT_PRECOMPUTE_SAVE_WORKERS", 4),
            precompute_max_pending_saves=_env_int("VGGT_PRECOMPUTE_MAX_PENDING_SAVES", 8),
            precompute_save_compressed=_env_bool("VGGT_PRECOMPUTE_SAVE_COMPRESSED", False),
            # run
            gpu_spec=(legacy_gpu_spec or gpu_spec_train or "A100-80GB"),
            gpu_spec_train=(gpu_spec_train or "A100-80GB"),
            gpu_spec_precompute=(gpu_spec_precompute or "A100-40GB"),
            gpu_spec_infer=(gpu_spec_infer or "A100-40GB"),
            precompute_cpu=_env_float("VGGT_PRECOMPUTE_CPU", 16.0),
            precompute_memory_mb=_env_int("VGGT_PRECOMPUTE_MEMORY_MB", 65536),
            train_cpu=_env_float("VGGT_TRAIN_CPU", 16.0),
            train_memory_mb=_env_int("VGGT_TRAIN_MEMORY_MB", 65536),
            timeout_sec=_env_int("VGGT_TIMEOUT_SEC", 24 * 60 * 60),
            # training
            train_script=_env("VGGT_TRAIN_SCRIPT",
                              "train_view_decoder_ablation.py"),
            train_args_extra=_env("VGGT_TRAIN_ARGS_EXTRA", ""),
            epochs=_env_int("VGGT_EPOCHS", 50),
            batch_size=_env_int("VGGT_BATCH_SIZE", 3),
            accum_steps=_env_int("VGGT_ACCUM_STEPS", 1),
            lr=_env_float("VGGT_LR", 5e-5),
            lr_schedule=_env("VGGT_LR_SCHEDULE", "cosine"),
            warmup_steps=_env_int("VGGT_WARMUP_STEPS", 400),
            min_lr_ratio=_env_float("VGGT_MIN_LR_RATIO", 0.1),
            bg_weight=_env_float("VGGT_BG_WEIGHT", 0.05),
            train_mask_mode=_env("VGGT_TRAIN_MASK_MODE", "fg_conf"),
            recon_mask_mode=_env("VGGT_RECON_MASK_MODE", "fg"),
            require_tgt_fg=_env_bool("VGGT_REQUIRE_TGT_FG", True),
            allow_fg_from_conf=_env_bool("VGGT_ALLOW_FG_FROM_CONF", False),
            mask_cover_min=_env_float("VGGT_MASK_COVER_MIN", 0.01),
            mask_cover_max=_env_float("VGGT_MASK_COVER_MAX", 0.80),
            mask_sanity_mode=(_env("VGGT_MASK_SANITY_MODE", "warn").strip().lower() or "warn"),
            conf_use_quantile_mask=_env_bool(
                "VGGT_CONF_USE_QUANTILE_MASK", True),
            conf_qlo=_env_float("VGGT_CONF_QLO", 0.05),
            conf_qhi=_env_float("VGGT_CONF_QHI", 0.95),
            conf_sup_use_quantile=_env_bool(
                "VGGT_CONF_SUP_USE_QUANTILE", False),
            conf_sup_gamma=_env_float("VGGT_CONF_SUP_GAMMA", 1.0),
            use_ema=_env_bool("VGGT_USE_EMA", True),
            ema_decay=_env_float("VGGT_EMA_DECAY", 0.999),
            best_by=_env("VGGT_BEST_BY", "ema"),
            use_view_cond=_env_bool("VGGT_USE_VIEW_COND", False),
            view_cond_mode=_env("VGGT_VIEW_COND_MODE", "tgt"),
            view_select_mode=_env("VGGT_VIEW_SELECT_MODE", "uniform_yaw"),
            yaw_jitter_deg=_env_float("VGGT_YAW_JITTER_DEG", 20.0),
            yaw_phase_jitter_deg=_env_float("VGGT_YAW_PHASE_JITTER_DEG", 20.0),
            yaw_axis_x=_env_int("VGGT_YAW_AXIS_X", 0),
            yaw_axis_z=_env_int("VGGT_YAW_AXIS_Z", 2),
            yaw_center_mode=_env("VGGT_YAW_CENTER_MODE", "pointmap"),
            mosaic_every_steps=_env_int("VGGT_MOSAIC_EVERY_STEPS", 200),
            mosaic_num_targets=_env_int("VGGT_MOSAIC_NUM_TARGETS", 3),
            mosaic_num_src_views=_env_int("VGGT_MOSAIC_NUM_SRC_VIEWS", 6),
            mosaic_tile_size=_env_int("VGGT_MOSAIC_TILE_SIZE", 300),
            mosaic_point_stride=_env_int("VGGT_MOSAIC_POINT_STRIDE", 24),
            mosaic_seed=_env_int("VGGT_MOSAIC_SEED", 2026),
            tf32=_env_bool("VGGT_TF32", True),
            amp=_env_bool("VGGT_AMP", True),
            strict_deterministic=_env_bool("VGGT_STRICT_DETERMINISTIC", False),
            min_improve=_env_float("VGGT_MIN_IMPROVE", 1e-4),
            early_stop=_env_int("VGGT_EARLY_STOP", 0),
            # volumes
            data_vol=_env("VGGT_DATA_VOL", "vggt-zju-data"),
            out_vol=_env("VGGT_OUT_VOL", "vggt-out"),
            archives_dir=_env("VGGT_ARCHIVES_DIR", "/mnt/data/archives"),
            # misc
            nan_check_every=_env_int("VGGT_NAN_CHECK_EVERY", 200),
            debug_fixed_batch=_env_bool("VGGT_DEBUG_FIXED_BATCH", False),
            debug_fixed_index=_env_int("VGGT_DEBUG_FIXED_INDEX", 0),
        )
        return _apply_profile_defaults(cfg)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Cfg":
        return Cfg(**json.loads(s))


# --------------------------
# Volumes + mounts
# --------------------------

CFG_IMPORT = Cfg.from_env()

DATA_VOL = modal.Volume.from_name(CFG_IMPORT.data_vol, create_if_missing=False)
OUT_VOL = modal.Volume.from_name(CFG_IMPORT.out_vol, create_if_missing=False)

MNT_CODE = CFG_IMPORT.mnt_code
MNT_DATA = CFG_IMPORT.mnt_data
MNT_OUT = CFG_IMPORT.mnt_out

# Exclude volatile/local artifacts from code upload to avoid Modal build races.
CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    ".github/**",
    "__pycache__/**",
    "*.pyc",
    "logs/**",
    "infer_out/**",
]

# Use string GPU spec (avoid deprecated modal.gpu.A100(...))
GPU_ARG_TRAIN = _normalize_gpu_spec(CFG_IMPORT.gpu_spec_train or CFG_IMPORT.gpu_spec)
GPU_ARG_PRECOMPUTE = _normalize_gpu_spec(
    CFG_IMPORT.gpu_spec_precompute or CFG_IMPORT.gpu_spec
)
GPU_ARG_INFER = _normalize_gpu_spec(CFG_IMPORT.gpu_spec_infer or CFG_IMPORT.gpu_spec)

# Image: include deps for both precompute + training
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
    )
    .pip_install(
        "torch",
        "torchvision",
    )
    .pip_install(
        'huggingface_hub',
        "numpy",
        "tqdm",
        "einops",
        "pillow",
        "opencv-python-headless",
        "matplotlib",
        "scipy",
        # pixelSplat runtime deps
        "hydra-core",
        "lightning",
        "jaxtyping",
        "beartype",
        "wandb",
        "colorama",
        "scikit-image",
        "colorspacious",
        "moviepy",
        "imageio",
        "timm",
        "dacite",
        "lpips",
        "e3nn",
        "plyfile",
        "tabulate",
        "svg.py",
    )
)

PIXELSPLAT_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
    )
    .pip_install(
        "wheel",
        "torch",
        "torchvision",
    )
    .pip_install(
        'huggingface_hub',
        "numpy",
        "tqdm",
        "einops",
        "pillow",
        "opencv-python-headless",
        "matplotlib",
        "scipy",
        "hydra-core",
        "lightning",
        "jaxtyping",
        "beartype",
        "wandb",
        "colorama",
        "scikit-image",
        "colorspacious",
        "moviepy",
        "imageio",
        "timm",
        "dacite",
        "lpips",
        "e3nn",
        "plyfile",
        "tabulate",
        "svg.py",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "CC": "gcc",
        "CXX": "g++",
    })
    .pip_install(
        "git+https://github.com/dcharatan/diff-gaussian-rasterization-modified",
    )
)

app = modal.App("vggt-zju-runner")


# --------------------------
# Utilities
# --------------------------


def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[remote] $", " ".join(cmd))
    t0 = time.time()
    tail = deque(maxlen=120)
    run_env = dict(env or os.environ.copy())
    run_env["PYTHONUNBUFFERED"] = "1"
    heartbeat_sec = max(5.0, float(run_env.get("VGGT_REMOTE_CMD_HEARTBEAT_SEC", "30") or 30.0))
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    last_emit = time.time()
    stream = p.stdout
    while True:
        ready, _, _ = select.select([stream], [], [], heartbeat_sec)
        now = time.time()
        if ready:
            line = stream.readline()
            if line == "":
                if p.poll() is not None:
                    break
                continue
            line = line.rstrip("\n")
            tail.append(line)
            print(line, flush=True)
            last_emit = now
            continue
        if p.poll() is not None:
            break
        print(
            f"[remote] [alive] child_pid={p.pid} elapsed={now - t0:.1f}s "
            f"idle={now - last_emit:.1f}s cmd={' '.join(cmd[:3])}...",
            flush=True,
        )
        last_emit = now
    rc = p.wait()
    dt = time.time() - t0
    if rc != 0:
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"command failed (rc={rc}, {dt:.1f}s): {' '.join(cmd)}\n"
            f"----- remote stdout tail -----\n{tail_text}")
    print(f"[remote] [ok] finished in {dt:.1f}s")


def _print_tree(root: Path, max_items: int = 80) -> None:
    print(f"[remote] dataset sample tree: {root}")
    if not root.exists():
        print("[remote] [warn] path does not exist")
        return
    cnt = 0
    for p in root.rglob("*"):
        if cnt >= max_items:
            print(f"[remote] ... (truncated after {max_items} items)")
            break
        rel = p.relative_to(root)
        if p.is_dir():
            print(f"[remote] [d] {rel}/")
        else:
            try:
                sz = p.stat().st_size
            except Exception:
                sz = -1
            print(f"[remote] [f] {rel} ({sz} bytes)")
        cnt += 1


def _find_seq_dir(zju_root: Path, seq: str) -> Path:
    # ZJU mocap: /mnt/data/zju_mocap/CoreView_390
    p = zju_root / seq
    if p.exists():
        return p
    raise RuntimeError(f"[remote] seq dir not found: {p}")


def _ensure_dataset(cfg: Cfg) -> None:
    """
    Training requires {zju_root}/{seq}/{geom_subdir} to exist.
    If not, try to extract from {archives_dir}/{seq}/... if present.
    """
    zju_root = Path(cfg.zju_root)
    archives_dir = Path(cfg.archives_dir)
    geom_subdir = str(cfg.geom_subdir or "vggt_geom").strip() or "vggt_geom"

    for seq in cfg.seq_names:
        seq_dir = _find_seq_dir(zju_root, seq)
        geom_dir = seq_dir / geom_subdir
        if geom_dir.exists():
            print(f"[remote] [ok] found {geom_subdir}: {geom_dir}")
            continue

        # Attempt extraction from archives
        arc_seq = archives_dir / seq
        if not arc_seq.exists():
            raise RuntimeError(
                f"[remote] missing {geom_subdir} and no archives found.\n"
                f"  need: {geom_dir}\n"
                f"  archives_dir: {arc_seq} (not found)"
            )

        # Extract all .tar / .tar.gz / .tgz under arc_seq into seq_dir
        print(
            f"[remote] [warn] {geom_subdir} missing, extracting archives from: {arc_seq}")
        seq_dir.mkdir(parents=True, exist_ok=True)
        extracted_any = False
        for tar_path in sorted(arc_seq.rglob("*.tar*")):
            extracted_any = True
            _run_cmd(["tar", "-xf", str(tar_path), "-C", str(seq_dir)])
        if not extracted_any:
            raise RuntimeError(
                f"[remote] archives dir exists but no tar files: {arc_seq}")

        if not geom_dir.exists():
            raise RuntimeError(
                f"[remote] after extraction, still missing: {geom_dir}")

        print(f"[remote] [ok] extracted {geom_subdir}: {geom_dir}")


def _build_train_cmd(cfg: Cfg) -> list[str]:
    # Call the repo train script with env + args.
    code_dir = Path(MNT_CODE)
    script = code_dir / cfg.train_script
    if not script.exists():
        raise RuntimeError(f"[remote] train_script not found: {script}")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    seq_tag = cfg.seq_names[0] if cfg.seq_names else "seq"
    run_root = Path(MNT_OUT) / "viewdec_ablation" / f"{seq_tag}_{run_tag}"
    log_dir = run_root / "logs"
    ckpt_dir = run_root / "ckpt"

    script_name = Path(cfg.train_script).name.lower()
    if "finetune_vggt_pseudo" in script_name:
        args = [
            f"--zju_root={cfg.zju_root}",
            f"--seq_names={','.join(cfg.seq_names)}",
            f"--geom_subdir={cfg.geom_subdir}",
            f"--log_dir={str(log_dir)}",
            f"--ckpt_dir={str(ckpt_dir)}",
            f"--epochs={cfg.epochs}",
            f"--max_frames={cfg.max_frames}",
            f"--lr={cfg.lr}",
            f"--min_improve={cfg.min_improve}",
            f"--early_stop_patience={int(cfg.early_stop)}",
        ]
        if cfg.cam_names:
            args.append(f"--cam_names={','.join(cfg.cam_names)}")
        if cfg.tf32:
            args.append("--tf32")
        else:
            args.append("--no-tf32")
        if cfg.amp:
            args.append("--amp")
        else:
            args.append("--no-amp")
    else:
        args = [
            f"--zju_root={cfg.zju_root}",
            f"--seq_names={','.join(cfg.seq_names)}",
            f"--geom_subdir={cfg.geom_subdir}",
            f"--log_dir={str(log_dir)}",
            f"--ckpt_dir={str(ckpt_dir)}",
            f"--epochs={cfg.epochs}",
            f"--batch_size={cfg.batch_size}",
            f"--accum_steps={cfg.accum_steps}",
            f"--lr={cfg.lr}",
            f"--lr_schedule={cfg.lr_schedule}",
            f"--warmup_steps={cfg.warmup_steps}",
            f"--min_lr_ratio={cfg.min_lr_ratio}",
            f"--bg_weight={cfg.bg_weight}",
            f"--conf_qlo={cfg.conf_qlo}",
            f"--conf_qhi={cfg.conf_qhi}",
            f"--ema_decay={cfg.ema_decay}",
            f"--nan_check_every={cfg.nan_check_every}",
            f"--min_improve={cfg.min_improve}",
            f"--early_stop={int(cfg.early_stop)}",
        ]
        if cfg.conf_use_quantile_mask:
            args.append("--conf_use_quantile")
        else:
            args.append("--no_conf_use_quantile")
        if cfg.conf_sup_use_quantile:
            args.append("--conf_sup_use_quantile")
            args.append(f"--conf_sup_gamma={cfg.conf_sup_gamma}")
        if cfg.use_ema:
            args.append("--use_ema")
        else:
            args.append("--no_use_ema")
        if cfg.best_by:
            args.append(f"--best_by={cfg.best_by}")
        if cfg.use_view_cond:
            args.append("--use_view_cond")
            args.append(f"--view_cond_mode={cfg.view_cond_mode}")
        if cfg.tf32:
            args.append("--tf32")
        else:
            args.append("--no_tf32")
        if cfg.amp:
            args.append("--amp")
        else:
            args.append("--no_amp")
        if cfg.debug_fixed_batch:
            args.append("--debug_fixed_batch")
        else:
            args.append("--no_debug_fixed_batch")
        args.append(f"--debug_fixed_index={cfg.debug_fixed_index}")

    if "train_view_decoder_ablation" in script_name:
        mask_sanity_mode = str(cfg.mask_sanity_mode or "warn").strip().lower()
        if mask_sanity_mode not in ("warn", "raise", "off"):
            mask_sanity_mode = "warn"
        args.extend([
            f"--view_select_mode={cfg.view_select_mode}",
            f"--yaw_jitter_deg={cfg.yaw_jitter_deg}",
            f"--yaw_phase_jitter_deg={cfg.yaw_phase_jitter_deg}",
            f"--yaw_axis_x={int(cfg.yaw_axis_x)}",
            f"--yaw_axis_z={int(cfg.yaw_axis_z)}",
            f"--yaw_center_mode={cfg.yaw_center_mode}",
            f"--train_mask_mode={cfg.train_mask_mode}",
            f"--recon_mask_mode={cfg.recon_mask_mode}",
            f"--mask_cover_min={cfg.mask_cover_min}",
            f"--mask_cover_max={cfg.mask_cover_max}",
            f"--mask_sanity_mode={mask_sanity_mode}",
            f"--mosaic_every_steps={int(cfg.mosaic_every_steps)}",
            f"--mosaic_num_targets={int(cfg.mosaic_num_targets)}",
            f"--mosaic_num_src_views={int(cfg.mosaic_num_src_views)}",
            f"--mosaic_tile_size={int(cfg.mosaic_tile_size)}",
            f"--mosaic_point_stride={int(cfg.mosaic_point_stride)}",
            f"--mosaic_seed={int(cfg.mosaic_seed)}",
        ])
        if cfg.require_tgt_fg:
            args.append("--require_tgt_fg")
        else:
            args.append("--no_require_tgt_fg")
        if cfg.allow_fg_from_conf:
            args.append("--allow_fg_from_conf")
        else:
            args.append("--no_allow_fg_from_conf")

    # Extra user-provided args
    if cfg.train_args_extra.strip():
        args.extend(shlex.split(cfg.train_args_extra))

    return [sys.executable, "-u", str(script)] + args


def _resolve_script_path(code_dir: Path, script: str) -> Path:
    """Resolve script relative to code_dir; fallback to recursive search."""
    p = Path(script)
    if p.is_absolute() and p.exists():
        return p
    cand = (code_dir / script)
    if cand.exists():
        return cand
    # fallback: search by basename
    base = Path(script).name
    for q in code_dir.rglob(base):
        if q.is_file():
            return q
    return cand


def _basename_cross_platform(path_str: str) -> str:
    s = (path_str or "").strip().replace("\\", "/").rstrip("/")
    if not s:
        return ""
    return s.split("/")[-1]


def _sibling_tmp_ckpt(path: Path) -> Path:
    return Path(str(path) + ".tmp")


def _probe_checkpoint_readable(path: Path) -> tuple[bool, str]:
    try:
        st = path.stat()
    except Exception as e:
        return False, f"stat_failed:{type(e).__name__}:{e}"
    if st.st_size <= 0:
        return False, "empty_file"

    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if len(names) <= 0:
                    return False, "zip_empty"
            return True, "zip_ok"
        except Exception as e:
            return False, f"zip_probe_failed:{type(e).__name__}:{e}"

    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth", ".ckpt", ".bin"):
        try:
            import torch

            try:
                _ = torch.load(str(path), map_location="cpu", weights_only=False)
            except TypeError:
                _ = torch.load(str(path), map_location="cpu")
            return True, "torch_load_ok"
        except Exception as e:
            return False, f"torch_probe_failed:{type(e).__name__}:{e}"

    return False, "unknown_nonzip_checkpoint_format"


def _select_readable_checkpoint_path(path: Path, label: str) -> tuple[Path, str]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for cand in (path, _sibling_tmp_ckpt(path)):
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    failures: list[str] = []
    for idx, cand in enumerate(candidates):
        if not cand.exists():
            failures.append(f"{cand}:missing")
            continue
        ok, note = _probe_checkpoint_readable(cand)
        if ok:
            if idx == 0:
                return cand, "primary"
            print(
                f"[remote] [warn] {label} primary checkpoint unreadable; "
                f"using sibling tmp checkpoint: {cand}",
                flush=True,
            )
            return cand, "tmp_fallback"
        failures.append(f"{cand}:{note}")

    fail_blob = "; ".join(failures)
    raise RuntimeError(
        f"[remote] ckpt_handoff_failed for {label}. "
        f"no readable checkpoint candidate found. {fail_blob}"
    )


def _resolve_precompute_ckpt_path(code_dir: Path, ckpt: str) -> Path:
    """
    Resolve checkpoint path inside container.
    Accepts Linux absolute paths, code_dir-relative paths, and Windows-style
    paths by mapping them to code_dir/<basename>.
    """
    raw = (ckpt or "").strip()
    default_ckpt = code_dir / "model.pt"

    if not raw:
        return default_ckpt

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    if not p.is_absolute():
        rel = code_dir / raw
        if rel.exists():
            return rel

    base = _basename_cross_platform(raw)
    if base:
        by_base = code_dir / base
        if by_base.exists():
            return by_base

    # Preserve absolute path intent even if currently missing, for clearer errors.
    if p.is_absolute():
        return p
    return default_ckpt


def _resolve_infer_ckpt_path(code_dir: Path, cfg: Cfg) -> Path:
    raw = (cfg.infer_ckpt or "").strip()
    if not raw:
        seq_tag = cfg.seq_names[0] if cfg.seq_names else "CoreView_390"
        base = Path(MNT_OUT) / "viewdec_ablation"
        cands = sorted(base.glob(f"{seq_tag}_*/ckpt/viewdec_ablation_best.pth"))
        if cands:
            return cands[-1]
        raise RuntimeError(
            "[remote] infer ckpt not provided and no auto-discovered best checkpoint.\n"
            f"  searched: {base}/{seq_tag}_*/ckpt/viewdec_ablation_best.pth\n"
            "  set VGGT_INFER_CKPT explicitly."
        )

    p = Path(raw)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path(MNT_OUT) / raw)
        candidates.append(code_dir / raw)

    if "/" not in raw and "\\" not in raw:
        candidates.append(
            Path(MNT_OUT) / "viewdec_ablation" / raw / "ckpt" / "viewdec_ablation_best.pth")
        candidates.append(Path(MNT_OUT) / "viewdec_ablation" /
                          raw / "ckpt" / "viewdec_ablation_last.pth")

    for cand in candidates:
        if cand.exists():
            return cand

    if p.is_absolute():
        return p
    return candidates[0]


def _resolve_infer_out_dir(cfg: Cfg) -> Path:
    raw = (cfg.infer_out_dir or "").strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return Path(MNT_OUT) / p

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    seq_tag = cfg.seq_names[0] if cfg.seq_names else "seq"
    return Path(MNT_OUT) / "infer_viewdec" / f"{seq_tag}_{run_tag}"


def _build_infer_cmd(cfg: Cfg) -> list[str]:
    code_dir = Path(MNT_CODE)
    script = _resolve_script_path(code_dir, cfg.infer_script)
    if not script.exists():
        raise RuntimeError(f"[remote] infer_script not found: {script}")

    ckpt_path = _resolve_infer_ckpt_path(code_dir, cfg)
    if not ckpt_path.exists():
        raise RuntimeError(
            "[remote] checkpoint not found for infer.\n"
            f"  requested: {cfg.infer_ckpt!r}\n"
            f"  resolved: {ckpt_path}"
        )
    ckpt_path, ckpt_resolve_reason = _select_readable_checkpoint_path(
        ckpt_path, "infer"
    )
    out_dir = _resolve_infer_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        f"--ckpt={str(ckpt_path)}",
        f"--split={str(cfg.infer_split or 'val')}",
        f"--out_dir={str(out_dir)}",
        f"--zju_root={cfg.zju_root}",
        f"--geom_subdir={str(cfg.geom_subdir or 'vggt_geom')}",
        f"--batch_size={int(cfg.infer_batch_size)}",
        f"--num_workers={int(cfg.infer_num_workers)}",
        f"--num_samples={int(cfg.infer_num_samples)}",
    ]
    if cfg.seq_names:
        args.append("--seq_names")
        args.extend(cfg.seq_names)
    if cfg.infer_use_ema:
        args.append("--use_ema")
    else:
        args.append("--no_ema")
    if cfg.infer_args_extra.strip():
        args.extend(shlex.split(cfg.infer_args_extra))

    print(f"[remote] infer ckpt = {ckpt_path}")
    print(f"[remote] infer ckpt_resolve_reason = {ckpt_resolve_reason}")
    print(f"[remote] infer out_dir = {out_dir}")
    return [sys.executable, "-u", str(script)] + args


def _run_precompute(cfg: Cfg, code_dir: Path) -> None:
    script_path = _resolve_script_path(code_dir, cfg.precompute_script)
    if not script_path.exists():
        # Provide a helpful directory listing for debugging.
        try:
            items = sorted([p.name for p in code_dir.iterdir()])
            print("[remote] [dbg] code_dir listing (top-level):", items[:200])
        except Exception:
            pass
        raise RuntimeError(
            f"[remote] precompute_script not found: {script_path}")
    ckpt_path = _resolve_precompute_ckpt_path(code_dir, cfg.precompute_ckpt)
    if not ckpt_path.exists():
        try:
            top = sorted([p.name for p in code_dir.iterdir() if p.is_file()])
        except Exception:
            top = []
        raise RuntimeError(
            "[remote] checkpoint not found for precompute.\n"
            f"  requested: {cfg.precompute_ckpt!r}\n"
            f"  resolved: {ckpt_path}\n"
            f"  code_dir: {code_dir}\n"
            f"  top-level files sample: {top[:80]}"
        )
    ckpt_path, ckpt_resolve_reason = _select_readable_checkpoint_path(
        ckpt_path, "precompute"
    )
    print(f"[remote] precompute ckpt = {ckpt_path}")
    print(f"[remote] precompute ckpt_resolve_reason = {ckpt_resolve_reason}")
    print(f"[remote] precompute out_dir = {cfg.geom_subdir or 'vggt_geom'}")

    env = os.environ.copy()
    env["VGGT_ZJU_ROOT"] = str(cfg.zju_root)
    env["VGGT_SEQ_NAMES"] = ",".join(cfg.seq_names)
    env["VGGT_CKPT"] = str(ckpt_path)
    env["VGGT_PRECOMPUTE_CKPT"] = str(ckpt_path)
    env["VGGT_OUT_DIR"] = str(cfg.geom_subdir or "vggt_geom")
    env["VGGT_POINTMAP_SOURCE"] = str(cfg.pointmap_source or "depth_unproject")
    env["VGGT_POINT_HEAD_FRAME"] = str(cfg.point_head_frame or "auto")
    env["VGGT_UNPROJECT_IMPL"] = str(cfg.precompute_unproject_impl or "legacy")
    env["VGGT_TF32"] = "1" if bool(cfg.tf32) else "0"
    env["VGGT_AMP"] = "1" if bool(cfg.amp) else "0"
    if "VGGT_STRICT_DETERMINISTIC" in os.environ:
        env["VGGT_STRICT_DETERMINISTIC"] = str(os.environ.get("VGGT_STRICT_DETERMINISTIC", "0"))
    env["VGGT_MV_SUPPORT_ON"] = "1" if bool(cfg.precompute_mv_support_on) else "0"
    env["VGGT_MV_SUPPORT_TOL_ABS"] = str(float(cfg.precompute_mv_support_tol_abs))
    env["VGGT_MV_SUPPORT_TOL_REL"] = str(float(cfg.precompute_mv_support_tol_rel))
    env["VGGT_MV_SUPPORT_STRIDE"] = str(int(cfg.precompute_mv_support_stride))
    env["VGGT_MV_SUPPORT_MODE"] = str(cfg.precompute_mv_support_mode or "linear")
    env["VGGT_MV_SUPPORT_FLOOR"] = str(float(cfg.precompute_mv_support_floor))
    env["VGGT_MV_SUPPORT_GAMMA"] = str(float(cfg.precompute_mv_support_gamma))
    env["VGGT_MV_SUPPORT_CLIP_THR"] = str(float(cfg.precompute_mv_support_clip_thr))
    env["VGGT_MV_SUPPORT_CLIP_FLOOR"] = str(float(cfg.precompute_mv_support_clip_floor))
    env["VGGT_MV_SUPPORT_HARD_THR"] = str(float(cfg.precompute_mv_support_hard_thr))
    env["VGGT_MV_CONF_VALID_FLOOR"] = str(float(cfg.precompute_mv_conf_valid_floor))
    env["VGGT_MV_SUPPORT_SAVE"] = "1" if bool(cfg.precompute_mv_support_save) else "0"
    env["VGGT_MV_SUPPORT_SAVE_RAW_CONF"] = "1" if bool(cfg.precompute_mv_support_save_raw_conf) else "0"
    env["VGGT_MV_SUPPORT_REGION_MODE"] = str(cfg.precompute_mv_support_region_mode or "auto")
    env["VGGT_MV_SUPPORT_FG_MASK_SOURCE"] = str(cfg.precompute_mv_support_fg_mask_source or "auto")
    env["VGGT_MV_SUPPORT_FG_ERODE_PX"] = str(int(cfg.precompute_mv_support_fg_erode_px))
    env["VGGT_MV_SUPPORT_FG_PRESERVE_PX"] = str(int(cfg.precompute_mv_support_fg_preserve_px))
    env["VGGT_PRECOMPUTE_BATCH_FRAMES"] = str(max(1, int(cfg.precompute_batch_frames)))
    env["VGGT_PRECOMPUTE_MV_SUPPORT_VIEW_CHUNK"] = str(max(1, int(cfg.precompute_mv_support_view_chunk)))
    env["VGGT_PRECOMPUTE_IMAGE_WORKERS"] = str(max(1, int(cfg.precompute_image_workers)))
    env["VGGT_PRECOMPUTE_SAVE_WORKERS"] = str(max(0, int(cfg.precompute_save_workers)))
    env["VGGT_PRECOMPUTE_MAX_PENDING_SAVES"] = str(max(1, int(cfg.precompute_max_pending_saves)))
    env["VGGT_PRECOMPUTE_SAVE_COMPRESSED"] = "1" if bool(cfg.precompute_save_compressed) else "0"
    if cfg.cam_names:
        env["VGGT_CAM_NAMES"] = ",".join(cfg.cam_names)
    if int(cfg.max_frames) > 0:
        env["VGGT_MAX_FRAMES"] = str(int(cfg.max_frames))
    # Make sure code is importable
    env["PYTHONPATH"] = str(
        code_dir) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, "-u", str(script_path)]
    if str(cfg.precompute_args_extra).strip():
        cmd.extend(shlex.split(str(cfg.precompute_args_extra).strip()))
    print("[remote] precompute cmd:", " ".join(cmd))
    _run_cmd(cmd, cwd=code_dir, env=env)


# --------------------------
# Modal remote entry
# --------------------------


def _run_remote_impl(cfg_json: str) -> None:
    cfg = Cfg.from_json(cfg_json)
    print("[remote] cfg.seq_names =", cfg.seq_names)
    print("[remote] cfg.zju_root =", cfg.zju_root)
    print("[remote] cfg.mode =", cfg.mode)

    code_dir = Path(MNT_CODE)
    if not code_dir.exists():
        raise RuntimeError(f"[remote] code dir missing: {code_dir}")

    # Make sure the invoked module dir is the repo root.
    os.chdir(code_dir)

    # Show dataset tree head
    zju_root = Path(cfg.zju_root)
    # show one seq head for sanity
    _print_tree(zju_root / cfg.seq_names[0], max_items=80)

    if str(cfg.mode).lower().startswith("pre"):
        print("[remote] mode = precompute")
        _run_precompute(cfg, code_dir)
        try:
            DATA_VOL.commit()
            print("[remote] committed data volume")
        except Exception as e:
            print(f"[remote] commit skipped/failed: {e}")
        return

    # infer/train paths: ensure geometry directory exists (or extract)
    _ensure_dataset(cfg)

    mode = str(cfg.mode).lower().strip()
    if mode.startswith("infer"):
        print("[remote] mode = infer")
        cmd = _build_infer_cmd(cfg)
    else:
        print("[remote] mode = train")
        cmd = _build_train_cmd(cfg)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(
        code_dir) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    _run_cmd(cmd, cwd=code_dir, env=env)

    # persist out volume
    try:
        OUT_VOL.commit()
        print("[remote] committed out volume")
    except Exception as e:
        print(f"[remote] out commit skipped/failed: {e}")


@app.function(
    image=IMAGE.add_local_dir(
        str(Path(CFG_IMPORT.code_dir).resolve()),
        remote_path=str(CFG_IMPORT.mnt_code),
        ignore=CODE_SYNC_IGNORE,
    ),
    gpu=GPU_ARG_PRECOMPUTE,
    cpu=CFG_IMPORT.precompute_cpu,
    memory=CFG_IMPORT.precompute_memory_mb,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={
        MNT_DATA: DATA_VOL,
        MNT_OUT: OUT_VOL,
    },
)
def run_remote_precompute(cfg_json: str) -> None:
    _run_remote_impl(cfg_json)


@app.function(
    image=PIXELSPLAT_IMAGE.add_local_dir(
        str(Path(CFG_IMPORT.code_dir).resolve()),
        remote_path=str(CFG_IMPORT.mnt_code),
        ignore=CODE_SYNC_IGNORE,
    ),
    gpu=GPU_ARG_PRECOMPUTE,
    cpu=CFG_IMPORT.precompute_cpu,
    memory=CFG_IMPORT.precompute_memory_mb,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={
        MNT_DATA: DATA_VOL,
        MNT_OUT: OUT_VOL,
    },
)
def run_remote_pixelsplat_precompute(cfg_json: str) -> None:
    _run_remote_impl(cfg_json)


@app.function(
    image=IMAGE.add_local_dir(
        str(Path(CFG_IMPORT.code_dir).resolve()),
        remote_path=str(CFG_IMPORT.mnt_code),
        ignore=CODE_SYNC_IGNORE,
    ),
    gpu=GPU_ARG_TRAIN,
    cpu=CFG_IMPORT.train_cpu,
    memory=CFG_IMPORT.train_memory_mb,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={
        MNT_DATA: DATA_VOL,
        MNT_OUT: OUT_VOL,
    },
)
def run_remote_train(cfg_json: str) -> None:
    _run_remote_impl(cfg_json)


@app.function(
    image=IMAGE.add_local_dir(
        str(Path(CFG_IMPORT.code_dir).resolve()),
        remote_path=str(CFG_IMPORT.mnt_code),
        ignore=CODE_SYNC_IGNORE,
    ),
    gpu=GPU_ARG_INFER,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={
        MNT_DATA: DATA_VOL,
        MNT_OUT: OUT_VOL,
    },
)
def run_remote_infer(cfg_json: str) -> None:
    _run_remote_impl(cfg_json)


def _should_use_pixelsplat_image(cfg: Cfg) -> bool:
    mode = str(cfg.mode).strip().lower()
    if not mode.startswith("pre"):
        return False
    script = str(cfg.precompute_script or "").strip().lower()
    extra = str(cfg.precompute_args_extra or "").strip().lower()
    return ("pixelsplat" in script) or ("pixelsplat" in extra)


@app.local_entrypoint()
def main() -> None:
    cfg = Cfg.from_env()
    print("[local] cfg =", cfg)
    mode = str(cfg.mode).strip().lower()
    if _should_use_pixelsplat_image(cfg):
        print(f"[local] route to run_remote_pixelsplat_precompute gpu={GPU_ARG_PRECOMPUTE}")
        run_remote_pixelsplat_precompute.remote(cfg.to_json())
    elif mode.startswith("pre"):
        print(f"[local] route to run_remote_precompute gpu={GPU_ARG_PRECOMPUTE}")
        run_remote_precompute.remote(cfg.to_json())
    elif mode.startswith("infer"):
        print(f"[local] route to run_remote_infer gpu={GPU_ARG_INFER}")
        run_remote_infer.remote(cfg.to_json())
    else:
        print(f"[local] route to run_remote_train gpu={GPU_ARG_TRAIN}")
        run_remote_train.remote(cfg.to_json())


if __name__ == "__main__":
    with app.run():
        main()
