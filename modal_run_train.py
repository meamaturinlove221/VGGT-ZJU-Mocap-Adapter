import os
import sys
import json
import time
import shlex
import subprocess
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
    mode: str = "train"  # "precompute" or "train"
    precompute_script: str = "precompute_zju_vggt_geom.py"
    # optional; if set, will be copied to {mnt_code}/model.pt if needed
    precompute_ckpt: str = ""
    cam_names: list[str] = field(default_factory=list)
    max_frames: int = 0  # 0=all

    # run (modal)
    gpu_spec: str = "A100-80GB"
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
    tf32: bool = True
    amp: bool = True

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

        return Cfg(
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
            cam_names=_split_csv_list(_env("VGGT_CAM_NAMES", "")),
            max_frames=_env_int("VGGT_MAX_FRAMES", 0),
            # run
            gpu_spec=_env("VGGT_GPU_SPEC", "A100-80GB"),
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
            tf32=_env_bool("VGGT_TF32", True),
            amp=_env_bool("VGGT_AMP", True),
            # volumes
            data_vol=_env("VGGT_DATA_VOL", "vggt-zju-data"),
            out_vol=_env("VGGT_OUT_VOL", "vggt-out"),
            archives_dir=_env("VGGT_ARCHIVES_DIR", "/mnt/data/archives"),
            # misc
            nan_check_every=_env_int("VGGT_NAN_CHECK_EVERY", 200),
            debug_fixed_batch=_env_bool("VGGT_DEBUG_FIXED_BATCH", False),
            debug_fixed_index=_env_int("VGGT_DEBUG_FIXED_INDEX", 0),
        )

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

# Use string GPU spec (avoid deprecated modal.gpu.A100(...))
GPU_ARG = (CFG_IMPORT.gpu_spec.strip() or None)
if GPU_ARG and GPU_ARG.lower() in ("none", "cpu", "false", "0"):
    GPU_ARG = None

# Image: include deps for both precompute + training
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install('huggingface_hub','torch','torchvision',"numpy",
        "tqdm",
        "einops",
        "pillow",
        "opencv-python-headless",
        "matplotlib",
        "scipy",
    )
)

app = modal.App("vggt-zju-runner")


# --------------------------
# Utilities
# --------------------------


def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[remote] $", " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line.rstrip("\n"))
    rc = p.wait()
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(
            f"command failed (rc={rc}, {dt:.1f}s): {' '.join(cmd)}")
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
    Training requires {zju_root}/{seq}/vggt_geom to exist.
    If not, try to extract from {archives_dir}/{seq}/... if present.
    """
    zju_root = Path(cfg.zju_root)
    archives_dir = Path(cfg.archives_dir)

    for seq in cfg.seq_names:
        seq_dir = _find_seq_dir(zju_root, seq)
        geom_dir = seq_dir / "vggt_geom"
        if geom_dir.exists():
            print(f"[remote] [ok] found vggt_geom: {geom_dir}")
            continue

        # Attempt extraction from archives
        arc_seq = archives_dir / seq
        if not arc_seq.exists():
            raise RuntimeError(
                f"[remote] missing vggt_geom and no archives found.\n"
                f"  need: {geom_dir}\n"
                f"  archives_dir: {arc_seq} (not found)"
            )

        # Extract all .tar / .tar.gz / .tgz under arc_seq into seq_dir
        print(
            f"[remote] [warn] vggt_geom missing, extracting archives from: {arc_seq}")
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

        print(f"[remote] [ok] extracted vggt_geom: {geom_dir}")


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

    args = [
        f"--zju_root={cfg.zju_root}",
        f"--seq_names={','.join(cfg.seq_names)}",
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

    # Extra user-provided args
    if cfg.train_args_extra.strip():
        args.extend(shlex.split(cfg.train_args_extra))

    return [sys.executable, str(script)] + args


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
    print(f"[remote] precompute ckpt = {ckpt_path}")

    env = os.environ.copy()
    env["VGGT_ZJU_ROOT"] = str(cfg.zju_root)
    env["VGGT_SEQ_NAMES"] = ",".join(cfg.seq_names)
    env["VGGT_CKPT"] = str(ckpt_path)
    env["VGGT_PRECOMPUTE_CKPT"] = str(ckpt_path)
    if cfg.cam_names:
        env["VGGT_CAM_NAMES"] = ",".join(cfg.cam_names)
    if int(cfg.max_frames) > 0:
        env["VGGT_MAX_FRAMES"] = str(int(cfg.max_frames))
    # Make sure code is importable
    env["PYTHONPATH"] = str(
        code_dir) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    # Run via runpy so the script can stay argument-free.
    parts = [
        "import os,runpy; ",
        f"os.environ.setdefault('VGGT_ZJU_ROOT',{cfg.zju_root!r}); ",
        f"os.environ.setdefault('VGGT_SEQ_NAMES',{','.join(cfg.seq_names)!r}); ",
        f"os.environ.setdefault('VGGT_CKPT',{str(ckpt_path)!r}); ",
        f"os.environ.setdefault('VGGT_PRECOMPUTE_CKPT',{str(ckpt_path)!r}); ",
    ]
    if cfg.cam_names:
        parts.append(
            f"os.environ.setdefault('VGGT_CAM_NAMES',{','.join(cfg.cam_names)!r}); ")
    if int(cfg.max_frames) > 0:
        parts.append(
            f"os.environ.setdefault('VGGT_MAX_FRAMES',{str(int(cfg.max_frames))!r}); ")
    parts.append(f"runpy.run_path({str(script_path)!r}, run_name='__main__')")
    py = "".join(parts)
    cmd = [sys.executable, "-c", py]
    print("[remote] precompute cmd:", " ".join(cmd))
    _run_cmd(cmd, cwd=code_dir, env=env)


# --------------------------
# Modal remote entry
# --------------------------


@app.function(
    image=IMAGE.add_local_dir(
        str(Path(CFG_IMPORT.code_dir).resolve()), remote_path=str(CFG_IMPORT.mnt_code)),
    gpu=GPU_ARG,
    timeout=CFG_IMPORT.timeout_sec,
    volumes={
        MNT_DATA: DATA_VOL,
        MNT_OUT: OUT_VOL,
    },
)
def run_remote(cfg_json: str) -> None:
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

    # training path: ensure vggt_geom exists (or extract)
    _ensure_dataset(cfg)

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


@app.local_entrypoint()
def main() -> None:
    cfg = Cfg.from_env()
    print("[local] cfg =", cfg)
    run_remote.remote(cfg.to_json())


if __name__ == "__main__":
    with app.run():
        main()
