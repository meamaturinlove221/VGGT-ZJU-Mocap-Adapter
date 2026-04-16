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
    to_volume_path,
    write_json,
    write_text,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_hardcase_probe"
DEFAULT_REPORTS_DIR = REPO_ROOT / "logs" / "modal_phase5" / "reports"
DEFAULT_REMOTE_OUT_ROOT = "/mnt/out/logs/modal_phase5/orig_vggt_hardcase_probe"

DEFAULT_PROFILES = ["12src_nested", "6src_hist"]
DEFAULT_SEQ_NAME = "CoreView_390"
DEFAULT_PRETRAINED_CKPT = "model.pt"
DEFAULT_MAX_FRAMES = 400
DEFAULT_SEED = 0
DEFAULT_TARGET_STEP = 1
DEFAULT_STRICT_DETERMINISTIC = True

# Evenly spread non-source cameras that keep both 6src_hist and 12src_nested
# render semantics stable.
DEFAULT_SCOUT_CAMERAS = [
    "Camera_B2",
    "Camera_B5",
    "Camera_B8",
    "Camera_B13",
    "Camera_B17",
    "Camera_B22",
]
DEFAULT_SCOUT_FRAMES = [960, 1020, 1080, 1140, 1200]
DEFAULT_SELECTED_CAMERA_COUNT = 2
DEFAULT_SELECTED_FRAMES_PER_CAMERA = 2

POINT_SUPPORT_REFERENCE = {
    "profile": "6src_hist",
    "step": 1,
    "label": "6src_hist_step0001",
    "kind": "point_support_reference",
}
GHOST_REFERENCE = {
    "profile": "12src_nested",
    "step": 4,
    "label": "12src_nested_step0004",
    "kind": "ghost_reference",
}
REFERENCE_23SRC = {
    "profile": "23cam_fullset",
    "source_root": REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_one_step_probe" / "23cam_fullset",
}

GHOST_EPS = 0.02
SUPPORT_EPS = 0.002


def parse_profiles(raw: str | Iterable[str] | None) -> list[str]:
    vals = split_tokens(raw)
    return vals or list(DEFAULT_PROFILES)


def parse_frames(raw: str | Iterable[str] | None) -> list[int]:
    vals = split_tokens(raw)
    if not vals:
        return list(DEFAULT_SCOUT_FRAMES)
    out = sorted(dict.fromkeys(int(v) for v in vals))
    if not out:
        raise ValueError("resolved empty scout frames")
    return out


def parse_cameras(raw: str | Iterable[str] | None) -> list[str]:
    vals = split_tokens(raw)
    return vals or list(DEFAULT_SCOUT_CAMERAS)


def profile_tag(profile: str) -> str:
    mapping = {
        "12src_nested": "12src",
        "6src_hist": "6src",
        "23cam_fullset": "23src",
    }
    return mapping.get(str(profile), sanitize_tag(str(profile)))


def case_id(seq_name: str, frame_id: int, tgt_camera: str) -> str:
    return f"{sanitize_tag(seq_name)}__frame_{int(frame_id):06d}__{sanitize_tag(tgt_camera)}"


def case_stub(seq_name: str, frame_id: int, tgt_camera: str) -> dict:
    cid = case_id(seq_name, frame_id, tgt_camera)
    return {
        "case_id": cid,
        "seq_name": str(seq_name),
        "frame_id": int(frame_id),
        "tgt_camera": str(tgt_camera),
        "case_tag": f"{seq_name}/frame_{int(frame_id):06d}/{tgt_camera}",
    }


def profile_root(out_root: Path, profile: str) -> Path:
    return Path(out_root) / str(profile)


def manifest_path(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "profile_manifest.json"


def scout_compare_dir(out_root: Path, profile: str, seq_name: str, frame_id: int, tgt_camera: str) -> Path:
    stub = case_stub(seq_name, frame_id, tgt_camera)
    return profile_root(out_root, profile) / "scout_cases" / stub["case_id"] / "step0000" / "compare"


def selected_compare_dir(out_root: Path, profile: str, seq_name: str, frame_id: int, tgt_camera: str) -> Path:
    stub = case_stub(seq_name, frame_id, tgt_camera)
    return profile_root(out_root, profile) / "selected_cases" / stub["case_id"] / "step0001" / "compare"


def camera_train_root(out_root: Path, profile: str, tgt_camera: str) -> Path:
    return profile_root(out_root, profile) / "camera_runs" / sanitize_tag(tgt_camera) / "step0001" / "train"


def camera_geom_subdir(profile: str, tgt_camera: str) -> str:
    return f"vggt_geom_orig_probe_{sanitize_tag(profile)}_hardcase_{sanitize_tag(tgt_camera)}"


def validation_summary_json(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "validation" / "summary.json"


def validation_summary_md(out_root: Path, profile: str) -> Path:
    return profile_root(out_root, profile) / "validation" / "summary.md"


def reference_root(out_root: Path) -> Path:
    return Path(out_root) / "references"


def build_task_specs(profiles: list[str]) -> list[dict]:
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
    add("reference_ingest_fixed_winners", ["bootstrap_preflight"])
    profile_summary_tasks: list[str] = []
    for profile in profiles:
        tag = profile_tag(profile)
        add(f"profile_{tag}_scout_step0000", ["reference_ingest_fixed_winners"], max_attempts=2)
        add(f"profile_{tag}_select_hardcases", [f"profile_{tag}_scout_step0000"])
        add(f"profile_{tag}_train_selected_step0001", [f"profile_{tag}_select_hardcases"], max_attempts=2)
        add(f"profile_{tag}_compare_selected_step0001", [f"profile_{tag}_train_selected_step0001"], max_attempts=2)
        add(f"profile_{tag}_validation_summary", [f"profile_{tag}_compare_selected_step0001"])
        profile_summary_tasks.append(f"profile_{tag}_validation_summary")
    add("global_summary_refresh", profile_summary_tasks)
    add("modal_cleanup_stop_nonstopped_apps", ["global_summary_refresh"], max_attempts=2)
    add("modal_cleanup_verify_clean", ["modal_cleanup_stop_nonstopped_apps"], max_attempts=2)
    return tasks


__all__ = [
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MODAL_MAX_RETRIES",
    "DEFAULT_MODAL_OUT_VOLUME",
    "DEFAULT_MODAL_RETRY_SLEEP_SEC",
    "DEFAULT_MODAL_SCRIPT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_PRETRAINED_CKPT",
    "DEFAULT_PROFILES",
    "DEFAULT_REMOTE_OUT_ROOT",
    "DEFAULT_REMOTE_ZJU_ROOT",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_SCOUT_CAMERAS",
    "DEFAULT_SCOUT_FRAMES",
    "DEFAULT_SEED",
    "DEFAULT_SELECTED_CAMERA_COUNT",
    "DEFAULT_SELECTED_FRAMES_PER_CAMERA",
    "DEFAULT_SEQ_NAME",
    "DEFAULT_STRICT_DETERMINISTIC",
    "DEFAULT_TARGET_STEP",
    "DEFAULT_ZJU_ROOT",
    "GHOST_EPS",
    "GHOST_REFERENCE",
    "POINT_SUPPORT_REFERENCE",
    "REFERENCE_23SRC",
    "REPO_ROOT",
    "SUPPORT_EPS",
    "build_task_specs",
    "camera_geom_subdir",
    "camera_train_root",
    "case_id",
    "case_stub",
    "load_json",
    "manifest_path",
    "parse_cameras",
    "parse_frames",
    "parse_on_off",
    "parse_profiles",
    "profile_metadata",
    "profile_root",
    "profile_tag",
    "reference_root",
    "sanitize_tag",
    "scout_compare_dir",
    "selected_compare_dir",
    "to_volume_path",
    "validation_summary_json",
    "validation_summary_md",
    "write_json",
    "write_text",
]
