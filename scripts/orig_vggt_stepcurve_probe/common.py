from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from scripts.orig_vggt_one_step_probe.common import (
    DEFAULT_GHOST_PEAK_MIN_REL,
    DEFAULT_MODAL_MAX_RETRIES,
    DEFAULT_MODAL_OUT_VOLUME,
    DEFAULT_MODAL_RETRY_SLEEP_SEC,
    DEFAULT_MODAL_SCRIPT,
    DEFAULT_REMOTE_ZJU_ROOT,
    REPO_ROOT,
    frame_tag,
    parse_on_off,
    profile_metadata,
    remote_geom_subdir,
    to_volume_path,
)
from scripts.orig_vggt_viewcount.common import (
    DEFAULT_ZJU_ROOT,
    VIEW_PROFILES,
    infer_mask_path,
    sanitize_tag,
    split_tokens,
    write_json,
    write_text,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_stepcurve_probe"
DEFAULT_REPORTS_DIR = REPO_ROOT / "logs" / "modal_phase5" / "reports"
DEFAULT_REMOTE_OUT_ROOT = "/mnt/out/logs/modal_phase5/orig_vggt_stepcurve_probe"
DEFAULT_TRAIN_PROFILES = ["12src_nested", "6src_hist"]
REFERENCE_PROFILE = "23cam_fullset"
DEFAULT_STEP_HORIZONS = [1, 2, 4, 8, 16]
DEFAULT_ALL_STEPS = [0, 1, 2, 4, 8, 16]
TASK_STATUS_VALUES = ["pending", "in_progress", "completed", "failed", "blocked"]

DEFAULT_SUPPORT_THRESHOLD_FLOOR = 0.15
DEFAULT_SUPPORT_PEAK_MIN_REL = 0.35
DEFAULT_SUPPORT_PEAK_SMOOTH_K = 11
DEFAULT_SUPPORT_EPS = 0.002
DEFAULT_GHOST_EPS = 0.02


def parse_profiles(raw: str | Iterable[str] | None) -> list[str]:
    names = split_tokens(raw)
    if not names:
        return list(DEFAULT_TRAIN_PROFILES)
    bad = [name for name in names if name not in VIEW_PROFILES]
    if bad:
        raise ValueError(
            f"unknown profiles={bad!r}; expected subset of {sorted(VIEW_PROFILES.keys())}"
        )
    return names


def parse_step_horizons(raw: str | Iterable[str] | None) -> list[int]:
    vals = split_tokens(raw)
    if not vals:
        return list(DEFAULT_STEP_HORIZONS)
    out: list[int] = []
    for item in vals:
        val = int(item)
        if val <= 0:
            raise ValueError(f"step horizon must be positive, got {item!r}")
        out.append(val)
    out = sorted(dict.fromkeys(out))
    return out


def profile_tag(profile: str) -> str:
    mapping = {
        "12src_nested": "12src",
        "6src_hist": "6src",
        "23cam_fullset": "23src",
    }
    return mapping.get(str(profile), sanitize_tag(str(profile)))


def step_tag(step: int) -> str:
    return f"step{int(step):04d}"


def step_dirs(out_root: Path, profile: str, step: int) -> dict[str, Path]:
    profile_root = Path(out_root) / str(profile)
    tag = step_tag(step)
    dirs = {
        "root": profile_root / tag,
        "compare": profile_root / tag / "compare",
    }
    if int(step) > 0:
        dirs["train"] = profile_root / tag / "train"
    return dirs


def profile_dirs(out_root: Path, profile: str) -> dict[str, Path]:
    root = Path(out_root) / str(profile)
    return {
        "root": root,
        "profile_manifest_json": root / "profile_manifest.json",
        "trend_dir": root / "trend",
        "trend_summary_json": root / "trend" / "summary.json",
        "trend_summary_md": root / "trend" / "summary.md",
    }


def compare_dir(out_root: Path, profile: str, step: int) -> Path:
    return step_dirs(out_root, profile, step)["compare"]


def train_dir(out_root: Path, profile: str, step: int) -> Path:
    return step_dirs(out_root, profile, step)["train"]


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


def ensure_parent(path: str | Path) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def to_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def normalize_path(value: str | Path) -> str:
    return str(Path(value)).replace("\\", "/")


def resolve_local_zju_path(remote_path: str | Path, local_zju_root: str | Path) -> Path:
    text = normalize_path(remote_path).strip()
    local_root = Path(local_zju_root)
    prefixes = ["/mnt/data/zju_mocap/", "mnt/data/zju_mocap/", "/mnt/data/", "mnt/data/"]
    for prefix in prefixes:
        if text.startswith(prefix):
            suffix = text[len(prefix) :].lstrip("/")
            if prefix.startswith("/mnt/data/zju_mocap") or prefix.startswith("mnt/data/zju_mocap"):
                return (local_root / suffix).resolve()
            return (local_root.parent / suffix).resolve()
    return Path(text)


def stable_hash(payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_task_specs(profiles: list[str], step_horizons: list[int]) -> list[dict]:
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
    profile_trend_tasks: list[str] = []
    for profile in profiles:
        tag = profile_tag(profile)
        add(f"profile_{tag}_prepare", ["bootstrap_preflight"])
        add(f"profile_{tag}_precompute_geom", [f"profile_{tag}_prepare"])
        add(f"profile_{tag}_step0000_compare", [f"profile_{tag}_precompute_geom"])
        add(f"profile_{tag}_step0000_sync_compare", [f"profile_{tag}_step0000_compare"], max_attempts=2)
        add(f"profile_{tag}_step0000_score_ghost", [f"profile_{tag}_step0000_sync_compare"])
        add(f"profile_{tag}_step0000_measure_support", [f"profile_{tag}_step0000_score_ghost"])
        baseline_tail = f"profile_{tag}_step0000_measure_support"
        for step in step_horizons:
            stag = step_tag(step)
            add(f"profile_{tag}_{stag}_train", [baseline_tail], max_attempts=2)
            add(f"profile_{tag}_{stag}_sync_train", [f"profile_{tag}_{stag}_train"], max_attempts=3)
            add(f"profile_{tag}_{stag}_prefix_audit", [f"profile_{tag}_{stag}_sync_train"])
            add(f"profile_{tag}_{stag}_compare", [f"profile_{tag}_{stag}_prefix_audit"], max_attempts=2)
            add(f"profile_{tag}_{stag}_sync_compare", [f"profile_{tag}_{stag}_compare"], max_attempts=2)
            add(f"profile_{tag}_{stag}_score_ghost", [f"profile_{tag}_{stag}_sync_compare"])
            add(f"profile_{tag}_{stag}_measure_support", [f"profile_{tag}_{stag}_score_ghost"])
        add(f"profile_{tag}_trend_summary", [baseline_tail])
        profile_trend_tasks.append(f"profile_{tag}_trend_summary")

    add("reference_23src_ingest_existing_onestep", ["bootstrap_preflight"])
    add("global_summary_refresh", profile_trend_tasks + ["reference_23src_ingest_existing_onestep"])
    add("modal_cleanup_stop_nonstopped_apps", ["global_summary_refresh"], max_attempts=2)
    add("modal_cleanup_verify_clean", ["modal_cleanup_stop_nonstopped_apps"], max_attempts=2)
    return tasks


__all__ = [
    "DEFAULT_ALL_STEPS",
    "DEFAULT_GHOST_EPS",
    "DEFAULT_GHOST_PEAK_MIN_REL",
    "DEFAULT_MODAL_MAX_RETRIES",
    "DEFAULT_MODAL_OUT_VOLUME",
    "DEFAULT_MODAL_RETRY_SLEEP_SEC",
    "DEFAULT_MODAL_SCRIPT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_REMOTE_OUT_ROOT",
    "DEFAULT_REMOTE_ZJU_ROOT",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_STEP_HORIZONS",
    "DEFAULT_SUPPORT_EPS",
    "DEFAULT_SUPPORT_PEAK_MIN_REL",
    "DEFAULT_SUPPORT_PEAK_SMOOTH_K",
    "DEFAULT_SUPPORT_THRESHOLD_FLOOR",
    "DEFAULT_TRAIN_PROFILES",
    "DEFAULT_ZJU_ROOT",
    "REFERENCE_PROFILE",
    "REPO_ROOT",
    "TASK_STATUS_VALUES",
    "build_task_specs",
    "compare_dir",
    "ensure_parent",
    "fmt_num",
    "frame_tag",
    "infer_mask_path",
    "load_json",
    "load_jsonl",
    "normalize_path",
    "parse_on_off",
    "parse_profiles",
    "parse_step_horizons",
    "profile_dirs",
    "profile_metadata",
    "profile_tag",
    "remote_geom_subdir",
    "resolve_local_zju_path",
    "sanitize_tag",
    "stable_hash",
    "step_dirs",
    "step_tag",
    "to_float",
    "to_int",
    "to_volume_path",
    "train_dir",
    "write_json",
    "write_text",
]
