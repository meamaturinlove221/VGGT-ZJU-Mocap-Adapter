from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scripts.orig_vggt_stepcurve_probe.common import (
    DEFAULT_MODAL_MAX_RETRIES,
    DEFAULT_MODAL_OUT_VOLUME,
    DEFAULT_MODAL_RETRY_SLEEP_SEC,
    DEFAULT_MODAL_SCRIPT,
    DEFAULT_REMOTE_ZJU_ROOT,
    DEFAULT_ZJU_ROOT,
    REPO_ROOT,
    load_json,
    parse_on_off,
    profile_metadata,
    sanitize_tag,
    split_tokens,
    to_float,
    to_int,
    to_volume_path,
    write_json,
    write_text,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_mask_boost_probe"
DEFAULT_REPORTS_DIR = REPO_ROOT / "logs" / "modal_phase5" / "reports"
DEFAULT_REMOTE_OUT_ROOT = "/mnt/out/logs/modal_phase5/orig_vggt_mask_boost_probe"
DEFAULT_NATIVE_STEPCURVE_ROOT = REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_stepcurve_probe"

# Backward-compat alias for earlier draft imports.
DEFAULT_NATIVE_STEPcurve_ROOT = DEFAULT_NATIVE_STEPCURVE_ROOT

DEFAULT_PROFILES = ["12src_nested", "6src_hist"]
DEFAULT_SHORT_STEPS = [1, 2, 4, 8]
DEFAULT_EXTEND_STEPS = [12, 16, 24]
DEFAULT_ALPHAS = [1, 2, 4]
DEFAULT_SEQ_NAME = "CoreView_390"
DEFAULT_FRAME_ID = 1080
DEFAULT_TGT_CAMERA = "Camera_B5"
DEFAULT_PRETRAINED_CKPT = "model.pt"
DEFAULT_MAX_FRAMES = 400
DEFAULT_SEED = 0

DEFAULT_MASK_ERODE_PX = 2
DEFAULT_USE_FG_MASK = "on"
DEFAULT_FG_MASK_SOURCE = "mask"
DEFAULT_FG_SUPERVISION_REGION_MODE = "interior_only"

POINT_SUPPORT_REFERENCE = {
    "label": "6src_hist step0001",
    "profile": "6src_hist",
    "step": 1,
}

GHOST_REFERENCE = {
    "label": "12src_nested step0004",
    "profile": "12src_nested",
    "step": 4,
}

SUBJECT_PSNR_EPS = 0.02
SUBJECT_L1_EPS = 1e-4
SUPPORT_EPS = 0.002
GHOST_EPS = 0.02


def parse_profiles(raw: str | Iterable[str] | None) -> list[str]:
    vals = split_tokens(raw)
    return vals or list(DEFAULT_PROFILES)


def parse_steps(raw: str | Iterable[str] | None, default_values: list[int]) -> list[int]:
    vals = split_tokens(raw)
    if not vals:
        return list(default_values)
    out = sorted(dict.fromkeys(int(v) for v in vals))
    if not out:
        raise ValueError("resolved empty step list")
    return out


def parse_alphas(raw: str | Iterable[str] | None) -> list[int]:
    vals = split_tokens(raw)
    if not vals:
        return list(DEFAULT_ALPHAS)
    out = sorted(dict.fromkeys(int(v) for v in vals))
    if not out:
        raise ValueError("resolved empty alpha list")
    return out


def profile_tag(profile: str) -> str:
    mapping = {
        "12src_nested": "12src",
        "6src_hist": "6src",
        "23cam_fullset": "23src",
    }
    return mapping.get(str(profile), sanitize_tag(str(profile)))


def alpha_tag(alpha: int) -> str:
    return f"alpha{int(alpha)}"


def step_tag(step: int) -> str:
    return f"step{int(step):04d}"


def alpha_to_fg_boost(alpha: int) -> float:
    # finetune_vggt_pseudo uses an internal 1 + (boost-1) * mask profile.
    return float(1.0 + float(alpha))


def profile_root(out_root: Path, profile: str) -> Path:
    return Path(out_root) / str(profile)


def manifest_path(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "profile_manifest.json"


def alpha_root(out_root: Path, profile: str, alpha: int) -> Path:
    return profile_root(out_root, profile) / alpha_tag(alpha)


def alpha_step_root(out_root: Path, profile: str, alpha: int, step: int) -> Path:
    return alpha_root(out_root, profile, alpha) / step_tag(step)


def alpha_step_compare_dir(out_root: Path, profile: str, alpha: int, step: int) -> Path:
    return alpha_step_root(out_root, profile, alpha, step) / "compare"


def alpha_step_train_dir(out_root: Path, profile: str, alpha: int, step: int) -> Path:
    return alpha_step_root(out_root, profile, alpha, step) / "train"


def profile_summary_json(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "trend" / "summary.json"


def profile_summary_md(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "trend" / "summary.md"


def build_task_specs(
    profiles: list[str],
    alphas: list[int],
    short_steps: list[int],
    extend_steps: list[int],
) -> list[dict]:
    tasks: list[dict] = []

    def add(name: str, depends_on: list[str] | None = None, max_attempts: int = 1) -> None:
        tasks.append(
            {
                "name": str(name),
                "status": "pending",
                "depends_on": list(depends_on or []),
                "attempt": 0,
                "max_attempts": int(max_attempts),
                "started_at": "",
                "updated_at": "",
                "message": "",
                "outputs": {},
                "errors": [],
            }
        )

    add("bootstrap_preflight")
    add("native_reference_refresh_support_metrics", ["bootstrap_preflight"], max_attempts=2)
    add("native_reference_refresh_summary", ["native_reference_refresh_support_metrics"], max_attempts=2)

    profile_trend_tasks: list[str] = []
    for profile in profiles:
        tag = profile_tag(profile)
        add(f"profile_{tag}_prepare", ["native_reference_refresh_summary"])
        add(f"profile_{tag}_precompute_geom", [f"profile_{tag}_prepare"], max_attempts=2)

        short_terminal_tasks: list[str] = []
        for alpha in alphas:
            atag = alpha_tag(alpha)
            prev_dep = f"profile_{tag}_precompute_geom"
            for step in short_steps:
                stag = step_tag(step)
                add(f"profile_{tag}_{atag}_{stag}_train", [prev_dep], max_attempts=2)
                add(f"profile_{tag}_{atag}_{stag}_sync_train", [f"profile_{tag}_{atag}_{stag}_train"], max_attempts=3)
                add(f"profile_{tag}_{atag}_{stag}_prefix_audit", [f"profile_{tag}_{atag}_{stag}_sync_train"], max_attempts=2)
                add(f"profile_{tag}_{atag}_{stag}_compare", [f"profile_{tag}_{atag}_{stag}_prefix_audit"], max_attempts=2)
                add(f"profile_{tag}_{atag}_{stag}_sync_compare", [f"profile_{tag}_{atag}_{stag}_compare"], max_attempts=2)
                add(f"profile_{tag}_{atag}_{stag}_score_ghost", [f"profile_{tag}_{atag}_{stag}_sync_compare"], max_attempts=2)
                add(f"profile_{tag}_{atag}_{stag}_measure_support", [f"profile_{tag}_{atag}_{stag}_score_ghost"], max_attempts=2)
                prev_dep = f"profile_{tag}_{atag}_{stag}_measure_support"
            add(f"profile_{tag}_{atag}_short_summary", [prev_dep], max_attempts=2)
            short_terminal_tasks.append(f"profile_{tag}_{atag}_short_summary")

        add(f"profile_{tag}_select_winner", short_terminal_tasks, max_attempts=2)

        prev_dep = f"profile_{tag}_select_winner"
        for step in extend_steps:
            stag = step_tag(step)
            add(f"profile_{tag}_winner_{stag}_train", [prev_dep], max_attempts=2)
            add(f"profile_{tag}_winner_{stag}_sync_train", [f"profile_{tag}_winner_{stag}_train"], max_attempts=3)
            add(f"profile_{tag}_winner_{stag}_prefix_audit", [f"profile_{tag}_winner_{stag}_sync_train"], max_attempts=2)
            add(f"profile_{tag}_winner_{stag}_compare", [f"profile_{tag}_winner_{stag}_prefix_audit"], max_attempts=2)
            add(f"profile_{tag}_winner_{stag}_sync_compare", [f"profile_{tag}_winner_{stag}_compare"], max_attempts=2)
            add(f"profile_{tag}_winner_{stag}_score_ghost", [f"profile_{tag}_winner_{stag}_sync_compare"], max_attempts=2)
            add(f"profile_{tag}_winner_{stag}_measure_support", [f"profile_{tag}_winner_{stag}_score_ghost"], max_attempts=2)
            prev_dep = f"profile_{tag}_winner_{stag}_measure_support"

        add(f"profile_{tag}_trend_summary", [prev_dep], max_attempts=2)
        profile_trend_tasks.append(f"profile_{tag}_trend_summary")

    add("global_summary_refresh", profile_trend_tasks, max_attempts=2)
    add("modal_cleanup_stop_nonstopped_apps", ["global_summary_refresh"], max_attempts=2)
    add("modal_cleanup_verify_clean", ["modal_cleanup_stop_nonstopped_apps"], max_attempts=2)
    return tasks


__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_EXTEND_STEPS",
    "DEFAULT_FG_MASK_SOURCE",
    "DEFAULT_FG_SUPERVISION_REGION_MODE",
    "DEFAULT_FRAME_ID",
    "DEFAULT_MASK_ERODE_PX",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MODAL_MAX_RETRIES",
    "DEFAULT_MODAL_OUT_VOLUME",
    "DEFAULT_MODAL_RETRY_SLEEP_SEC",
    "DEFAULT_MODAL_SCRIPT",
    "DEFAULT_NATIVE_STEPCURVE_ROOT",
    "DEFAULT_NATIVE_STEPcurve_ROOT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_PRETRAINED_CKPT",
    "DEFAULT_PROFILES",
    "DEFAULT_REMOTE_OUT_ROOT",
    "DEFAULT_REMOTE_ZJU_ROOT",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_SEED",
    "DEFAULT_SEQ_NAME",
    "DEFAULT_SHORT_STEPS",
    "DEFAULT_TGT_CAMERA",
    "DEFAULT_USE_FG_MASK",
    "DEFAULT_ZJU_ROOT",
    "GHOST_EPS",
    "GHOST_REFERENCE",
    "POINT_SUPPORT_REFERENCE",
    "SUBJECT_L1_EPS",
    "SUBJECT_PSNR_EPS",
    "SUPPORT_EPS",
    "alpha_root",
    "alpha_step_compare_dir",
    "alpha_step_root",
    "alpha_step_train_dir",
    "alpha_tag",
    "alpha_to_fg_boost",
    "build_task_specs",
    "load_json",
    "manifest_path",
    "parse_alphas",
    "parse_on_off",
    "parse_profiles",
    "parse_steps",
    "profile_metadata",
    "profile_root",
    "profile_summary_json",
    "profile_summary_md",
    "profile_tag",
    "sanitize_tag",
    "step_tag",
    "to_float",
    "to_int",
    "to_volume_path",
    "write_json",
    "write_text",
]
