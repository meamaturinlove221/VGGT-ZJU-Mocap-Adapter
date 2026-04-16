import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SUPPORT_KEYS = [
    "mv_support_raw_mean",
    "mv_support_valid_ratio",
    "mv_support_fg_valid_ratio",
    "mv_support_bg_valid_ratio",
    "mv_support_pair_count_eff",
    "mv_support_conf_mean",
    "mv_support_nan_ratio",
    "depth_conf_delta_mean",
    "mv_support_fg_mean",
    "mv_support_bg_mean",
    "depth_conf_delta_fg_mean",
    "depth_conf_delta_bg_mean",
    "depth_conf_fg_preserved_active",
    "depth_conf_fg_preserve_px",
    "depth_conf_fg_exact_ratio",
    "depth_conf_fg_preserve_ratio",
    "depth_conf_fg_raw_mean",
    "depth_conf_fg_after_support_mean",
    "depth_conf_fg_final_mean",
    "mv_support_generation_region_mode",
    "mv_support_generation_fg_mask_source",
]


def _run_modal_json(args: list[str]) -> list[dict]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        return []
    try:
        text = proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        text = proc.stdout.decode("utf-8", errors="ignore")
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    flat: list[dict] = []
    queue = list(parsed if isinstance(parsed, list) else [parsed])
    while queue:
        item = queue.pop(0)
        if item is None:
            continue
        if isinstance(item, list):
            queue[:0] = item
            continue
        if isinstance(item, dict):
            flat.append(item)
    return flat


def _get_volume_file_names(items: list[dict]) -> list[str]:
    out: list[str] = []
    for obj in items:
        type_value = str(obj.get("Type", "") or obj.get("type", "")).strip()
        if type_value and type_value != "file":
            continue
        for key in ("Filename", "filename", "Path", "path", "Name", "name"):
            value = str(obj.get(key, "")).strip()
            if value:
                out.append(value)
                break
    seen = set()
    uniq: list[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)
    return uniq


def _resolve_remote_file_path(remote_dir: str, listed_path: str) -> str:
    remote_dir_norm = str(remote_dir or "").strip().replace("\\", "/").rstrip("/")
    listed_path_norm = str(listed_path or "").strip().replace("\\", "/")
    if not listed_path_norm:
        return ""
    if listed_path_norm.startswith("/"):
        return listed_path_norm
    if not remote_dir_norm:
        return "/" + listed_path_norm.lstrip("/")
    remote_rel = remote_dir_norm.lstrip("/")
    if listed_path_norm == remote_rel or listed_path_norm.startswith(remote_rel + "/"):
        return "/" + listed_path_norm
    return remote_dir_norm + "/" + listed_path_norm.lstrip("/")


def _download_volume_file(volume_name: str, remote_path: str, local_path: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", volume_name, remote_path, str(local_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
    )
    return proc.returncode == 0 and local_path.exists()


def _scalarize(value):
    arr = np.asarray(value).reshape(-1)
    if arr.size <= 0:
        return None
    item = arr[0]
    if np.issubdtype(arr.dtype, np.number):
        num = float(item)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    text = str(item)
    return text if text else None


def _extract_support_stats_from_npz(path: Path) -> dict:
    out: dict = {}
    with np.load(path, allow_pickle=True) as z:
        for key in SUPPORT_KEYS:
            if key not in z:
                continue
            try:
                value = _scalarize(z[key])
            except Exception:
                continue
            if value is None:
                continue
            out[key] = value
    return out


def _extract_support_stats_from_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict = {}
    for key in SUPPORT_KEYS:
        if key not in obj:
            continue
        value = obj.get(key)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            continue
        if value is None:
            continue
        out[key] = value
    return out


def _pick_first_match(file_names: list[str], suffix_pattern: str) -> str:
    for name in file_names:
        if name.lower().endswith(suffix_pattern.lower()):
            return name
    return ""


def fetch_support_stats(seq_names: str, geom_subdir: str, volume_name: str, remote_root: str) -> dict:
    seq_tokens = [tok.strip() for tok in str(seq_names or "").replace(",", " ").replace(";", " ").split() if tok.strip()]
    if not seq_tokens or not str(geom_subdir or "").strip():
        return {}

    with tempfile.TemporaryDirectory(prefix="precompute_support_stats_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for seq in seq_tokens:
            remote_dir = f"{remote_root.rstrip('/')}/{seq}/{geom_subdir}"
            items = _run_modal_json(["modal", "volume", "ls", "--json", volume_name, remote_dir])
            file_names = _get_volume_file_names(items)
            if not file_names:
                continue

            sidecar_name = _pick_first_match(file_names, ".support_stats.json")
            if sidecar_name:
                remote_file = _resolve_remote_file_path(remote_dir, sidecar_name)
                local_file = tmp_root / f"{seq}_support_stats.json"
                if _download_volume_file(volume_name, remote_file, local_file):
                    stats = _extract_support_stats_from_json(local_file)
                    if stats:
                        return stats

            npz_name = _pick_first_match(file_names, ".npz")
            if npz_name:
                remote_file = _resolve_remote_file_path(remote_dir, npz_name)
                local_file = tmp_root / f"{seq}_support_stats.npz"
                if _download_volume_file(volume_name, remote_file, local_file):
                    stats = _extract_support_stats_from_npz(local_file)
                    if stats:
                        return stats
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-names", required=True)
    parser.add_argument("--geom-subdir", required=True)
    parser.add_argument("--volume-name", default="vggt-zju-data")
    parser.add_argument("--remote-root", default="/zju_mocap")
    args = parser.parse_args()

    stats = fetch_support_stats(
        seq_names=args.seq_names,
        geom_subdir=args.geom_subdir,
        volume_name=args.volume_name,
        remote_root=args.remote_root,
    )
    sys.stdout.write(json.dumps(stats, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
