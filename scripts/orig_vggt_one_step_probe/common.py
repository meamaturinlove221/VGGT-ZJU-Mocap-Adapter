from __future__ import annotations

import math
from pathlib import Path

from scripts.orig_vggt_viewcount.common import (
    DEFAULT_ZJU_ROOT,
    REPO_ROOT,
    VIEW_PROFILES,
    resolve_view_spec,
    sanitize_tag,
    split_tokens,
    write_json,
    write_text,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_one_step_probe"
DEFAULT_REPORTS_DIR = REPO_ROOT / "logs" / "modal_phase5" / "reports"
DEFAULT_MODAL_SCRIPT = REPO_ROOT / "modal_run_train.py"
DEFAULT_REMOTE_ZJU_ROOT = "/mnt/data/zju_mocap"
DEFAULT_REMOTE_OUT_ROOT = "/mnt/out/logs/modal_phase5/orig_vggt_one_step_probe"
DEFAULT_MODAL_OUT_VOLUME = "vggt-out"
DEFAULT_MODAL_MAX_RETRIES = 3
DEFAULT_MODAL_RETRY_SLEEP_SEC = 10
DEFAULT_GHOST_PEAK_MIN_REL = 0.35
TASK_NAMES = [
    "prepare_profile",
    "precompute_geom",
    "one_step_train_remote",
    "sync_train_local",
    "pre_update_compare_remote",
    "sync_pre_update_local",
    "score_pre_update_ghost",
    "post_update_compare_remote",
    "sync_post_update_local",
    "score_post_update_ghost",
    "compare_summary",
]


def parse_profiles(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    names = split_tokens(raw)
    if not names:
        return list(VIEW_PROFILES.keys())
    bad = [name for name in names if name not in VIEW_PROFILES]
    if bad:
        raise ValueError(
            f"unknown profiles={bad!r}; expected subset of {sorted(VIEW_PROFILES.keys())}"
        )
    return names


def parse_on_off(raw: str | None, *, default: bool = True) -> bool:
    text = str(raw if raw is not None else ("on" if default else "off")).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected on/off style value, got {raw!r}")


def frame_tag(frame_id: int, tgt_camera: str) -> str:
    return f"frame_{int(frame_id):06d}_{str(tgt_camera).strip()}"


def remote_geom_subdir(profile: str) -> str:
    return f"vggt_geom_orig_probe_{sanitize_tag(profile)}"


def train_camera_names(profile: str, tgt_camera: str) -> list[str]:
    view_spec = resolve_view_spec(view_profile=profile, tgt_camera=tgt_camera)
    cameras = list(dict.fromkeys(list(view_spec["src_cameras"]) + [str(tgt_camera).strip()]))
    return cameras


def profile_metadata(profile: str, tgt_camera: str) -> dict:
    view_spec = resolve_view_spec(view_profile=profile, tgt_camera=tgt_camera)
    train_cams = train_camera_names(profile, tgt_camera)
    return {
        "profile": profile,
        "view_profile": view_spec["view_profile"],
        "profile_kind": view_spec["profile_kind"],
        "src_cameras_render": list(view_spec["src_cameras"]),
        "render_num_src_views_actual": int(view_spec["num_src_views_actual"]),
        "render_num_total_cams": int(view_spec["num_total_cams"]),
        "train_cameras": train_cams,
        "train_num_cameras": int(len(train_cams)),
    }


def profile_dirs(out_root: Path, profile: str) -> dict[str, Path]:
    root = Path(out_root) / str(profile)
    return {
        "root": root,
        "task_state_json": root / "task_state.json",
        "task_state_md": root / "task_state.md",
        "train": root / "train",
        "pre_update": root / "pre_update",
        "post_update": root / "post_update",
        "compare": root / "compare",
    }


def ensure_parent(path: str | Path) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst


def to_volume_path(remote_path: str | Path) -> str:
    text = str(remote_path).replace("\\", "/").strip()
    if text.startswith("/mnt/out/"):
        return "/" + text[len("/mnt/out/") :].lstrip("/")
    if text.startswith("mnt/out/"):
        return "/" + text[len("mnt/out/") :].lstrip("/")
    if text.startswith("/"):
        return text
    return "/" + text.lstrip("/")


def fmt_num(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "NaN"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return "NaN"
    return f"{val:.{digits}f}"


__all__ = [
    "DEFAULT_GHOST_PEAK_MIN_REL",
    "DEFAULT_MODAL_MAX_RETRIES",
    "DEFAULT_MODAL_OUT_VOLUME",
    "DEFAULT_MODAL_RETRY_SLEEP_SEC",
    "DEFAULT_MODAL_SCRIPT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_REMOTE_OUT_ROOT",
    "DEFAULT_REMOTE_ZJU_ROOT",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_ZJU_ROOT",
    "REPO_ROOT",
    "TASK_NAMES",
    "ensure_parent",
    "fmt_num",
    "frame_tag",
    "parse_on_off",
    "parse_profiles",
    "profile_dirs",
    "profile_metadata",
    "remote_geom_subdir",
    "sanitize_tag",
    "to_volume_path",
    "train_camera_names",
    "write_json",
    "write_text",
]
