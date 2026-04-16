from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_one_step_probe.common import (  # noqa: E402
    DEFAULT_GHOST_PEAK_MIN_REL,
    DEFAULT_MODAL_MAX_RETRIES,
    DEFAULT_MODAL_OUT_VOLUME,
    DEFAULT_MODAL_RETRY_SLEEP_SEC,
    DEFAULT_MODAL_SCRIPT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REMOTE_OUT_ROOT,
    DEFAULT_REMOTE_ZJU_ROOT,
    DEFAULT_REPORTS_DIR,
    TASK_NAMES,
    frame_tag,
    parse_on_off,
    parse_profiles,
    profile_dirs,
    profile_metadata,
    remote_geom_subdir,
    to_volume_path,
    write_json,
    write_text,
)


TRANSIENT_MODAL_PATTERNS = [
    "Connection lost",
    "WinError 10053",
    "WinError 10054",
    "SSL shutdown timed out",
    "Deadline exceeded",
    "heartbeat failed",
    "modal.exception.ConnectionError",
    "timed out waiting for final app logs",
    "Could not connect to the Modal server",
    "Cannot connect to host",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _quote_args(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _safe_print_text(text: str) -> None:
    if not text:
        return
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = str(text).encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
    print(safe, end="" if safe.endswith("\n") else "\n", flush=True)


def _with_utf8_env(src_env: dict[str, str] | None) -> dict[str, str]:
    env = {str(k): str(v) for k, v in dict(src_env or os.environ).items()}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _strip_vggt_env(src_env: dict[str, str]) -> dict[str, str]:
    dst = {}
    for key, value in _with_utf8_env(src_env).items():
        if str(key).startswith("VGGT_"):
            continue
        dst[str(key)] = str(value)
    return dst


def _run_stream(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    label: str,
) -> tuple[int, list[str]]:
    print(f"[{label}] cmd: {' '.join(str(x) for x in cmd)}", flush=True)
    run_env = _with_utf8_env(env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        lines.append(line)
        _safe_print_text(line)
    code = int(proc.wait())
    return code, lines


def _run_capture(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    label: str,
) -> str:
    print(f"[{label}] cmd: {' '.join(str(x) for x in cmd)}", flush=True)
    run_env = _with_utf8_env(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=run_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.stdout:
        _safe_print_text(proc.stdout)
    if proc.stderr:
        _safe_print_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed rc={proc.returncode}")
    return str(proc.stdout or "")


def _extract_modal_failure_reason(lines: list[str]) -> str:
    preferred_prefixes = (
        "RuntimeError:",
        "ValueError:",
        "AssertionError:",
        "FileNotFoundError:",
        "KeyError:",
        "IndexError:",
        "TypeError:",
        "Exception:",
    )
    for raw in reversed(list(lines or [])):
        line = str(raw or "").strip()
        if not line:
            continue
        if line.startswith(preferred_prefixes):
            return line
    for raw in reversed(list(lines or [])):
        line = str(raw or "").strip()
        if line and ("error" in line.lower() or "exception" in line.lower()):
            return line
    for raw in reversed(list(lines or [])):
        line = str(raw or "").strip()
        if line:
            return line
    return ""


def _modal_run(
    *,
    modal_script: str,
    env_updates: dict[str, str],
    code_dir: Path,
    max_retries: int,
    retry_sleep_sec: int,
    label: str,
) -> list[str]:
    env = _strip_vggt_env(os.environ)
    env.update({str(k): str(v) for k, v in env_updates.items()})
    for attempt in range(1, max(1, int(max_retries)) + 1):
        rc, lines = _run_stream(
            ["modal", "run", "-q", str(modal_script)],
            cwd=code_dir,
            env=env,
            label=f"{label}[attempt={attempt}]",
        )
        if rc == 0:
            return lines
        blob = "\n".join(lines)
        is_transient = any(pattern in blob for pattern in TRANSIENT_MODAL_PATTERNS)
        if is_transient and attempt < max_retries:
            print(
                f"[{label}] transient modal failure detected; retry after {int(retry_sleep_sec)} sec",
                flush=True,
            )
            time.sleep(max(1, int(retry_sleep_sec)))
            continue
        detail = _extract_modal_failure_reason(lines)
        if detail:
            raise RuntimeError(f"{label} failed rc={rc}: {detail}")
        raise RuntimeError(f"{label} failed rc={rc}")
    raise RuntimeError(f"{label} exhausted retries")


def _modal_ls(volume: str, remote_dir: str) -> list[dict]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, int(DEFAULT_MODAL_MAX_RETRIES)) + 1):
        try:
            text = _run_capture(
                ["modal", "volume", "ls", "--json", str(volume), str(remote_dir)],
                cwd=REPO_ROOT,
                label=f"volume_ls:{remote_dir}",
            ).strip()
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= int(DEFAULT_MODAL_MAX_RETRIES):
                raise
            print(
                f"[volume_ls:{remote_dir}] retry after failure ({attempt}/{int(DEFAULT_MODAL_MAX_RETRIES)}): {exc}",
                flush=True,
            )
            time.sleep(max(1, int(DEFAULT_MODAL_RETRY_SLEEP_SEC)))
    else:
        raise RuntimeError(f"volume_ls exhausted retries: {remote_dir}") from last_exc
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return list(payload["items"])
    raise RuntimeError(f"unexpected modal volume ls payload for {remote_dir}")


def _modal_get(volume: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, int(DEFAULT_MODAL_MAX_RETRIES)) + 1):
        try:
            _run_capture(
                ["modal", "volume", "get", str(volume), str(remote_path), str(local_path)],
                cwd=REPO_ROOT,
                label=f"volume_get:{remote_path}",
            )
            return
        except Exception as exc:
            last_exc = exc
            try:
                if local_path.is_file():
                    local_path.unlink()
            except Exception:
                pass
            if attempt >= int(DEFAULT_MODAL_MAX_RETRIES):
                raise
            print(
                f"[volume_get:{remote_path}] retry after failure ({attempt}/{int(DEFAULT_MODAL_MAX_RETRIES)}): {exc}",
                flush=True,
            )
            time.sleep(max(1, int(DEFAULT_MODAL_RETRY_SLEEP_SEC)))
    raise RuntimeError(f"volume_get exhausted retries: {remote_path}") from last_exc


def _item_type(item: dict) -> str:
    for key in ("Type", "type"):
        if key in item:
            return str(item[key]).strip().lower()
    return ""


def _item_name(item: dict) -> str:
    for key in ("Filename", "filename", "Path", "path", "Name", "name"):
        if key in item and str(item[key]).strip():
            return str(item[key]).strip()
    return ""


def _remote_child_path(parent: str, raw_name: str) -> str:
    parent_norm = to_volume_path(parent).rstrip("/")
    name = str(raw_name).replace("\\", "/").strip()
    if not name:
        return parent_norm
    if name.startswith("/"):
        return name
    if name.startswith(parent_norm.lstrip("/") + "/"):
        return "/" + name.lstrip("/")
    return f"{parent_norm}/{name.lstrip('/')}"


def _download_volume_tree(
    *,
    volume: str,
    remote_dir: str,
    local_dir: Path,
    flatten_root: bool = False,
) -> None:
    entries = _modal_ls(volume, remote_dir)
    if not entries:
        raise RuntimeError(f"remote dir is empty: {remote_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in entries:
        item_type = _item_type(item)
        item_name = _item_name(item)
        if not item_name:
            continue
        remote_child = _remote_child_path(remote_dir, item_name)
        leaf = Path(str(item_name).replace("\\", "/")).name
        if item_type == "dir":
            child_local = local_dir if flatten_root else (local_dir / leaf)
            _download_volume_tree(
                volume=volume,
                remote_dir=remote_child,
                local_dir=child_local,
                flatten_root=False,
            )
            continue
        if item_type not in {"", "file"}:
            continue
        _modal_get(volume, remote_child, local_dir / leaf)


def _extract_tagged_line(lines: list[str], prefix: str) -> str:
    for line in lines:
        text = str(line).strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""


def _latest_raw_compare_run_dir(
    *,
    remote_stage_root: str,
    seq_name: str,
    frame_id: int,
    tgt_camera: str,
    volume: str,
) -> str:
    remote_frame_dir = to_volume_path(
        f"{str(remote_stage_root).rstrip('/')}/{seq_name}/{frame_tag(frame_id, tgt_camera)}"
    )
    entries = _modal_ls(volume, remote_frame_dir)
    dir_leaves = []
    for item in entries:
        if _item_type(item) != "dir":
            continue
        name = _item_name(item)
        if not name:
            continue
        dir_leaves.append(Path(str(name).replace("\\", "/")).name)
    if not dir_leaves:
        raise RuntimeError(f"no remote raw-compare runs found under {remote_frame_dir}")
    return f"/mnt/out{remote_frame_dir}/{sorted(dir_leaves)[-1]}"


def _clear_local_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _stage_ready(stage_dir: Path) -> bool:
    required = [
        stage_dir / "weight_native.png",
        stage_dir / "pred_native.png",
        stage_dir / "report.json",
        stage_dir / "cat_fg_mask_pred_tgt_step000000.png",
    ]
    return all(path.is_file() for path in required)


def _train_ready(train_dir: Path) -> bool:
    required = [
        train_dir / "logs" / "finetune_vggt_metrics.jsonl",
        train_dir / "logs" / "finetune_vggt_summary.json",
    ]
    return all(path.is_file() for path in required)


class ProfileTaskState:
    def __init__(self, *, path_json: Path, path_md: Path, context: dict):
        self.path_json = path_json
        self.path_md = path_md
        if path_json.is_file():
            with path_json.open("r", encoding="utf-8") as f:
                self.payload = json.load(f)
        else:
            self.payload = {
                "profile": context["profile"],
                "status": "pending",
                "current_task": "",
                "message": "",
                "updated_at": _now_iso(),
                "seq_name": context["seq_name"],
                "frame_id": int(context["frame_id"]),
                "tgt_camera": context["tgt_camera"],
                "profile_meta": context["profile_meta"],
                "paths": context["paths"],
                "tasks": [],
                "artifacts": {},
                "errors": [],
            }
        existing = {str(item.get("name")) for item in self.payload.get("tasks", [])}
        for name in TASK_NAMES:
            if name not in existing:
                self.payload.setdefault("tasks", []).append(
                    {
                        "name": name,
                        "status": "pending",
                        "updated_at": "",
                        "message": "",
                        "outputs": {},
                    }
                )
        self.save()

    def _task(self, name: str) -> dict:
        for item in self.payload.get("tasks", []):
            if str(item.get("name")) == str(name):
                return item
        raise KeyError(name)

    def is_completed(self, name: str) -> bool:
        return str(self._task(name).get("status")) == "completed"

    def outputs(self, name: str) -> dict:
        return dict(self._task(name).get("outputs", {}))

    def start(self, name: str, message: str) -> None:
        task = self._task(name)
        task["status"] = "in_progress"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        self.payload["status"] = "running"
        self.payload["current_task"] = str(name)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def complete(self, name: str, message: str, *, outputs: dict | None = None) -> None:
        task = self._task(name)
        task["status"] = "completed"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        if outputs:
            task.setdefault("outputs", {}).update(outputs)
            self.payload.setdefault("artifacts", {}).update(outputs)
        self.payload["updated_at"] = _now_iso()
        self.payload["message"] = str(message)
        self.payload["current_task"] = str(name)
        self.save()

    def fail(self, name: str, message: str) -> None:
        task = self._task(name)
        task["status"] = "failed"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        self.payload["status"] = "failed"
        self.payload["current_task"] = str(name)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.payload.setdefault("errors", []).append(
            {"task": str(name), "message": str(message), "updated_at": _now_iso()}
        )
        self.save()

    def finalize(self, *, success: bool, message: str) -> None:
        self.payload["status"] = "completed" if success else "failed"
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def save(self) -> None:
        write_json(self.path_json, self.payload)
        lines = [
            f"# Native VGGT One-Step Probe: {self.payload.get('profile', '')}",
            "",
            f"- status: `{self.payload.get('status', '')}`",
            f"- current_task: `{self.payload.get('current_task', '')}`",
            f"- message: {self.payload.get('message', '')}",
            f"- updated_at: `{self.payload.get('updated_at', '')}`",
            f"- fixed_eval: `seq={self.payload.get('seq_name', '')}` / `frame={self.payload.get('frame_id', '')}` / `tgt={self.payload.get('tgt_camera', '')}`",
            "",
            "## Tasks",
            "",
        ]
        for item in self.payload.get("tasks", []):
            lines.append(
                f"- `{item.get('name', '')}` status=`{item.get('status', '')}` updated_at=`{item.get('updated_at', '')}`"
            )
            if str(item.get("message", "")).strip():
                lines.append(f"- note: {item.get('message', '')}")
        lines.extend(["", "## Artifacts", ""])
        for key, value in sorted(self.payload.get("artifacts", {}).items()):
            lines.append(f"- `{key}`: `{value}`")
        write_text(self.path_md, "\n".join(lines) + "\n")


def _write_root_state(
    *,
    out_root: Path,
    profiles: list[str],
    reports_dir: Path,
    status: str,
    message: str,
) -> None:
    rows = []
    for profile in profiles:
        dirs = profile_dirs(out_root, profile)
        payload = {}
        if dirs["task_state_json"].is_file():
            with dirs["task_state_json"].open("r", encoding="utf-8") as f:
                payload = json.load(f)
        rows.append(
            {
                "profile": profile,
                "status": payload.get("status", "pending"),
                "current_task": payload.get("current_task", ""),
                "task_state_json": str(dirs["task_state_json"]),
                "compare_summary_json": str(dirs["compare"] / "summary.json"),
            }
        )
    payload = {
        "updated_at": _now_iso(),
        "status": str(status),
        "message": str(message),
        "profiles": rows,
        "reports": {
            "summary_json": str(reports_dir / "orig_vggt_one_step_probe_summary_latest.json"),
            "summary_csv": str(reports_dir / "orig_vggt_one_step_probe_summary_latest.csv"),
            "summary_md": str(reports_dir / "orig_vggt_one_step_probe_summary_latest.md"),
        },
    }
    write_json(out_root / "task_state_latest.json", payload)
    lines = [
        "# Native VGGT One-Step Probe Task State",
        "",
        f"- status: `{status}`",
        f"- message: {message}",
        f"- updated_at: `{payload['updated_at']}`",
        "",
        "## Profiles",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['profile']}` status=`{row['status']}` current_task=`{row['current_task']}` task_state=`{row['task_state_json']}`"
        )
    write_text(out_root / "task_state_latest.md", "\n".join(lines) + "\n")


def _run_ghost_score(stage_dir: Path, profile: str, peak_min_rel: float) -> None:
    input_png = stage_dir / "cat_fg_mask_pred_tgt_step000000.png"
    if not input_png.is_file():
        raise RuntimeError(f"ghost input missing: {input_png}")
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tools" / "score_ghosting_from_cat_pred.py"),
        "--input",
        f"{profile}={str(input_png)}",
        "--out_csv",
        str(stage_dir / "ghost_score_rows.csv"),
        "--out_summary_csv",
        str(stage_dir / "ghost_score_summary.csv"),
        "--out_json",
        str(stage_dir / "ghost_score.json"),
        "--peak_min_rel",
        str(float(peak_min_rel)),
    ]
    rc, _ = _run_stream(cmd, cwd=REPO_ROOT, env=None, label=f"ghost_score:{profile}:{stage_dir.name}")
    if rc != 0:
        raise RuntimeError(f"ghost scoring failed for {stage_dir}")


def _run_summary(
    *,
    out_root: Path,
    profiles: list[str],
    tgt_camera: str,
    pretrained_ckpt: str,
    reports_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "orig_vggt_one_step_probe" / "summarize_probe_runs.py"),
        "--out_root",
        str(out_root),
        "--profiles",
        ",".join(profiles),
        "--tgt_camera",
        str(tgt_camera),
        "--pretrained_ckpt",
        str(pretrained_ckpt),
        "--out_json",
        str(reports_dir / "orig_vggt_one_step_probe_summary_latest.json"),
        "--out_csv",
        str(reports_dir / "orig_vggt_one_step_probe_summary_latest.csv"),
        "--out_md",
        str(reports_dir / "orig_vggt_one_step_probe_summary_latest.md"),
    ]
    rc, _ = _run_stream(cmd, cwd=REPO_ROOT, env=None, label="summary")
    if rc != 0:
        raise RuntimeError("summary script failed")


def main() -> None:
    ap = argparse.ArgumentParser("orig_vggt_one_step_probe")
    ap.add_argument("--profiles", default="6src_hist,12src_nested,23cam_fullset")
    ap.add_argument("--seq_name", default="CoreView_390")
    ap.add_argument("--frame_id", type=int, default=1080)
    ap.add_argument("--tgt_camera", default="Camera_B5")
    ap.add_argument("--pretrained_ckpt", default="model.pt")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps_per_epoch", type=int, default=1)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--reports_dir", default=str(DEFAULT_REPORTS_DIR))
    ap.add_argument("--code_dir", default=str(REPO_ROOT))
    ap.add_argument("--remote_zju_root", default=str(DEFAULT_REMOTE_ZJU_ROOT))
    ap.add_argument("--remote_out_root", default=str(DEFAULT_REMOTE_OUT_ROOT))
    ap.add_argument("--modal_script", default=str(DEFAULT_MODAL_SCRIPT.name))
    ap.add_argument("--modal_out_volume", default=str(DEFAULT_MODAL_OUT_VOLUME))
    ap.add_argument("--modal_max_retries", type=int, default=int(DEFAULT_MODAL_MAX_RETRIES))
    ap.add_argument("--modal_retry_sleep_sec", type=int, default=int(DEFAULT_MODAL_RETRY_SLEEP_SEC))
    ap.add_argument("--gpu_spec_precompute", default="A100-80GB")
    ap.add_argument("--resume_task_state", default="on")
    ap.add_argument("--continue_on_profile_error", default="on")
    ap.add_argument("--peak_min_rel", type=float, default=float(DEFAULT_GHOST_PEAK_MIN_REL))
    args = ap.parse_args()

    profiles = parse_profiles(args.profiles)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (REPO_ROOT / reports_dir).resolve()
    code_dir = Path(args.code_dir)
    if not code_dir.is_absolute():
        code_dir = (REPO_ROOT / code_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    resume_on = parse_on_off(args.resume_task_state, default=True)
    continue_on_error = parse_on_off(args.continue_on_profile_error, default=True)
    _write_root_state(
        out_root=out_root,
        profiles=profiles,
        reports_dir=reports_dir,
        status="running",
        message="starting native VGGT one-step probe tasks",
    )

    failures: list[dict] = []
    completed_profiles: list[str] = []

    for profile in profiles:
        meta = profile_metadata(profile, args.tgt_camera)
        dirs = profile_dirs(out_root, profile)
        remote_profile_root = f"{str(args.remote_out_root).rstrip('/')}/{profile}"
        remote_pre_root = f"{remote_profile_root}/pre_update"
        remote_post_root = f"{remote_profile_root}/post_update"
        geom_subdir = remote_geom_subdir(profile)
        context = {
            "profile": profile,
            "seq_name": args.seq_name,
            "frame_id": int(args.frame_id),
            "tgt_camera": args.tgt_camera,
            "profile_meta": meta,
            "paths": {
                "local_root": str(dirs["root"]),
                "local_train_dir": str(dirs["train"]),
                "local_pre_update_dir": str(dirs["pre_update"]),
                "local_post_update_dir": str(dirs["post_update"]),
                "local_compare_dir": str(dirs["compare"]),
                "remote_profile_root": remote_profile_root,
                "remote_pre_update_root": remote_pre_root,
                "remote_post_update_root": remote_post_root,
                "remote_geom_subdir": geom_subdir,
            },
        }
        state = ProfileTaskState(
            path_json=dirs["task_state_json"],
            path_md=dirs["task_state_md"],
            context=context,
        )
        try:
            train_outputs = state.outputs("one_step_train_remote")
            remote_train_run_root = str(train_outputs.get("remote_train_run_root", "")).strip()
            remote_train_best_ckpt = str(train_outputs.get("remote_train_best_ckpt", "")).strip()

            if not (resume_on and state.is_completed("prepare_profile")):
                state.start("prepare_profile", "create local dirs and persist profile metadata")
                for key in ("root", "train", "pre_update", "post_update", "compare"):
                    dirs[key].mkdir(parents=True, exist_ok=True)
                state.complete(
                    "prepare_profile",
                    "profile directories ready",
                    outputs={
                        "profile_root": str(dirs["root"]),
                        "train_cameras_csv": ",".join(meta["train_cameras"]),
                        "render_src_cameras_csv": ",".join(meta["src_cameras_render"]),
                        "remote_geom_subdir": geom_subdir,
                    },
                )

            if not (resume_on and state.is_completed("precompute_geom")):
                state.start("precompute_geom", f"remote precompute for geom_subdir={geom_subdir}")
                _modal_run(
                    modal_script=args.modal_script,
                    env_updates={
                        "VGGT_CODE_DIR": str(code_dir),
                        "VGGT_MODE": "precompute",
                        "VGGT_GPU_SPEC_PRECOMPUTE": str(args.gpu_spec_precompute),
                        "VGGT_PRECOMPUTE_SCRIPT": "precompute_zju_vggt_geom.py",
                        "VGGT_PRECOMPUTE_CKPT": str(args.pretrained_ckpt),
                        "VGGT_ZJU_ROOT": str(args.remote_zju_root),
                        "VGGT_SEQ_NAMES": str(args.seq_name),
                        "VGGT_CAM_NAMES": ",".join(meta["train_cameras"]),
                        "VGGT_MAX_FRAMES": str(int(args.max_frames)),
                        "VGGT_GEOM_SUBDIR": geom_subdir,
                    },
                    code_dir=code_dir,
                    max_retries=int(args.modal_max_retries),
                    retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    label=f"{profile}:precompute_geom",
                )
                state.complete(
                    "precompute_geom",
                    "remote precompute finished",
                    outputs={"remote_geom_subdir": geom_subdir},
                )

            if not (resume_on and state.is_completed("one_step_train_remote") and remote_train_best_ckpt):
                remote_train_run_root = (
                    f"{remote_profile_root}/train_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                remote_train_best_ckpt = f"{remote_train_run_root}/ckpt/model_ft_zju.pt"
                train_args = [
                    "--zju_root",
                    str(args.remote_zju_root),
                    "--seq_names",
                    str(args.seq_name),
                    "--cam_names",
                    ",".join(meta["train_cameras"]),
                    "--pretrained_ckpt",
                    str(args.pretrained_ckpt),
                    "--resume_ckpt=",
                    "--epochs",
                    str(int(args.epochs)),
                    "--max_frames",
                    str(int(args.max_frames)),
                    "--max_steps_per_epoch",
                    str(int(args.max_steps_per_epoch)),
                    "--eval_every_steps",
                    "1",
                    "--debug_metrics_every_steps",
                    "1",
                    "--debug_vis_every_steps",
                    "1",
                    "--debug_vis_max_steps",
                    "1",
                    "--debug_vis_views",
                    "1",
                    "--geom_subdir",
                    geom_subdir,
                    "--log_dir",
                    f"{remote_train_run_root}/logs",
                    "--ckpt_dir",
                    f"{remote_train_run_root}/ckpt",
                ]
                state.start(
                    "one_step_train_remote",
                    f"remote one-step finetune -> {remote_train_run_root}",
                )
                _modal_run(
                    modal_script=args.modal_script,
                    env_updates={
                        "VGGT_CODE_DIR": str(code_dir),
                        "VGGT_MODE": "precompute",
                        "VGGT_GPU_SPEC_PRECOMPUTE": str(args.gpu_spec_precompute),
                        "VGGT_PRECOMPUTE_SCRIPT": "finetune_vggt_pseudo.py",
                        "VGGT_PRECOMPUTE_CKPT": str(args.pretrained_ckpt),
                        "VGGT_ZJU_ROOT": str(args.remote_zju_root),
                        "VGGT_SEQ_NAMES": str(args.seq_name),
                        "VGGT_CAM_NAMES": ",".join(meta["train_cameras"]),
                        "VGGT_MAX_FRAMES": str(int(args.max_frames)),
                        "VGGT_GEOM_SUBDIR": geom_subdir,
                        "VGGT_PRECOMPUTE_ARGS_EXTRA": _quote_args(train_args),
                    },
                    code_dir=code_dir,
                    max_retries=int(args.modal_max_retries),
                    retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    label=f"{profile}:one_step_train_remote",
                )
                state.complete(
                    "one_step_train_remote",
                    "remote one-step finetune finished",
                    outputs={
                        "remote_train_run_root": remote_train_run_root,
                        "remote_train_best_ckpt": remote_train_best_ckpt,
                    },
                )

            if not (resume_on and state.is_completed("sync_train_local") and _train_ready(dirs["train"])):
                if not remote_train_run_root:
                    remote_train_run_root = str(state.outputs("one_step_train_remote").get("remote_train_run_root", ""))
                state.start("sync_train_local", f"sync remote train dir {remote_train_run_root}")
                _clear_local_dir(dirs["train"])
                _download_volume_tree(
                    volume=str(args.modal_out_volume),
                    remote_dir=to_volume_path(f"{str(remote_train_run_root).rstrip('/')}/logs"),
                    local_dir=dirs["train"] / "logs",
                    flatten_root=True,
                )
                state.complete(
                    "sync_train_local",
                    "local train artifacts synced (logs/debug vis only; best ckpt stays remote)",
                    outputs={
                        "local_train_dir": str(dirs["train"]),
                        "local_train_sync_scope": "logs_only_remote_ckpt_retained",
                    },
                )

            if not (resume_on and state.is_completed("pre_update_compare_remote")):
                raw_args = [
                    "--seq_name",
                    str(args.seq_name),
                    "--frame_id",
                    str(int(args.frame_id)),
                    "--tgt_camera",
                    str(args.tgt_camera),
                    "--view_profile",
                    str(profile),
                    "--zju_root",
                    str(args.remote_zju_root),
                    "--out_dir",
                    str(remote_pre_root),
                ]
                state.start(
                    "pre_update_compare_remote",
                    f"remote raw compare from {args.pretrained_ckpt}",
                )
                lines = _modal_run(
                    modal_script=args.modal_script,
                    env_updates={
                        "VGGT_CODE_DIR": str(code_dir),
                        "VGGT_MODE": "precompute",
                        "VGGT_GPU_SPEC_PRECOMPUTE": str(args.gpu_spec_precompute),
                        "VGGT_PRECOMPUTE_SCRIPT": "scripts/orig_vggt_viewcount/render_raw_compare.py",
                        "VGGT_PRECOMPUTE_CKPT": str(args.pretrained_ckpt),
                        "VGGT_ZJU_ROOT": str(args.remote_zju_root),
                        "VGGT_SEQ_NAMES": str(args.seq_name),
                        "VGGT_GEOM_SUBDIR": str(remote_pre_root),
                        "VGGT_PRECOMPUTE_ARGS_EXTRA": _quote_args(raw_args),
                    },
                    code_dir=code_dir,
                    max_retries=int(args.modal_max_retries),
                    retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    label=f"{profile}:pre_update_compare_remote",
                )
                remote_pre_run_dir = _extract_tagged_line(lines, "RUN_DIR:")
                if not remote_pre_run_dir:
                    remote_pre_run_dir = _latest_raw_compare_run_dir(
                        remote_stage_root=remote_pre_root,
                        seq_name=args.seq_name,
                        frame_id=int(args.frame_id),
                        tgt_camera=args.tgt_camera,
                        volume=str(args.modal_out_volume),
                    )
                state.complete(
                    "pre_update_compare_remote",
                    "remote pre-update raw compare finished",
                    outputs={"remote_pre_update_run_dir": remote_pre_run_dir},
                )

            if not (resume_on and state.is_completed("sync_pre_update_local") and _stage_ready(dirs["pre_update"])):
                remote_pre_run_dir = str(
                    state.outputs("pre_update_compare_remote").get("remote_pre_update_run_dir", "")
                )
                state.start("sync_pre_update_local", f"sync {remote_pre_run_dir}")
                _clear_local_dir(dirs["pre_update"])
                _download_volume_tree(
                    volume=str(args.modal_out_volume),
                    remote_dir=to_volume_path(remote_pre_run_dir),
                    local_dir=dirs["pre_update"],
                    flatten_root=True,
                )
                state.complete(
                    "sync_pre_update_local",
                    "local pre-update stage synced",
                    outputs={"local_pre_update_dir": str(dirs["pre_update"])},
                )

            if not (
                resume_on
                and state.is_completed("score_pre_update_ghost")
                and (dirs["pre_update"] / "ghost_score.json").is_file()
            ):
                state.start("score_pre_update_ghost", "score pre-update ghost triplet locally")
                _run_ghost_score(dirs["pre_update"], profile, float(args.peak_min_rel))
                state.complete(
                    "score_pre_update_ghost",
                    "pre-update ghost scoring finished",
                    outputs={"pre_update_ghost_json": str(dirs["pre_update"] / "ghost_score.json")},
                )

            if not (resume_on and state.is_completed("post_update_compare_remote")):
                remote_train_best_ckpt = str(
                    state.outputs("one_step_train_remote").get("remote_train_best_ckpt", remote_train_best_ckpt)
                )
                raw_args = [
                    "--seq_name",
                    str(args.seq_name),
                    "--frame_id",
                    str(int(args.frame_id)),
                    "--tgt_camera",
                    str(args.tgt_camera),
                    "--view_profile",
                    str(profile),
                    "--zju_root",
                    str(args.remote_zju_root),
                    "--out_dir",
                    str(remote_post_root),
                ]
                state.start(
                    "post_update_compare_remote",
                    f"remote raw compare from {remote_train_best_ckpt}",
                )
                lines = _modal_run(
                    modal_script=args.modal_script,
                    env_updates={
                        "VGGT_CODE_DIR": str(code_dir),
                        "VGGT_MODE": "precompute",
                        "VGGT_GPU_SPEC_PRECOMPUTE": str(args.gpu_spec_precompute),
                        "VGGT_PRECOMPUTE_SCRIPT": "scripts/orig_vggt_viewcount/render_raw_compare.py",
                        "VGGT_PRECOMPUTE_CKPT": str(remote_train_best_ckpt),
                        "VGGT_ZJU_ROOT": str(args.remote_zju_root),
                        "VGGT_SEQ_NAMES": str(args.seq_name),
                        "VGGT_GEOM_SUBDIR": str(remote_post_root),
                        "VGGT_PRECOMPUTE_ARGS_EXTRA": _quote_args(raw_args),
                    },
                    code_dir=code_dir,
                    max_retries=int(args.modal_max_retries),
                    retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    label=f"{profile}:post_update_compare_remote",
                )
                remote_post_run_dir = _extract_tagged_line(lines, "RUN_DIR:")
                if not remote_post_run_dir:
                    remote_post_run_dir = _latest_raw_compare_run_dir(
                        remote_stage_root=remote_post_root,
                        seq_name=args.seq_name,
                        frame_id=int(args.frame_id),
                        tgt_camera=args.tgt_camera,
                        volume=str(args.modal_out_volume),
                    )
                state.complete(
                    "post_update_compare_remote",
                    "remote post-update raw compare finished",
                    outputs={"remote_post_update_run_dir": remote_post_run_dir},
                )

            if not (resume_on and state.is_completed("sync_post_update_local") and _stage_ready(dirs["post_update"])):
                remote_post_run_dir = str(
                    state.outputs("post_update_compare_remote").get("remote_post_update_run_dir", "")
                )
                state.start("sync_post_update_local", f"sync {remote_post_run_dir}")
                _clear_local_dir(dirs["post_update"])
                _download_volume_tree(
                    volume=str(args.modal_out_volume),
                    remote_dir=to_volume_path(remote_post_run_dir),
                    local_dir=dirs["post_update"],
                    flatten_root=True,
                )
                state.complete(
                    "sync_post_update_local",
                    "local post-update stage synced",
                    outputs={"local_post_update_dir": str(dirs["post_update"])},
                )

            if not (
                resume_on
                and state.is_completed("score_post_update_ghost")
                and (dirs["post_update"] / "ghost_score.json").is_file()
            ):
                state.start("score_post_update_ghost", "score post-update ghost triplet locally")
                _run_ghost_score(dirs["post_update"], profile, float(args.peak_min_rel))
                state.complete(
                    "score_post_update_ghost",
                    "post-update ghost scoring finished",
                    outputs={"post_update_ghost_json": str(dirs["post_update"] / "ghost_score.json")},
                )

            if not (
                resume_on
                and state.is_completed("compare_summary")
                and (dirs["compare"] / "summary.json").is_file()
            ):
                state.start("compare_summary", "refresh compare summary/report outputs")
                _run_summary(
                    out_root=out_root,
                    profiles=profiles,
                    tgt_camera=args.tgt_camera,
                    pretrained_ckpt=args.pretrained_ckpt,
                    reports_dir=reports_dir,
                )
                state.complete(
                    "compare_summary",
                    "compare summary refreshed",
                    outputs={"compare_summary_json": str(dirs["compare"] / "summary.json")},
                )

            state.finalize(success=True, message="all profile tasks completed")
            completed_profiles.append(profile)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            task_name = state.payload.get("current_task", "unknown_task")
            state.fail(str(task_name), msg + "\n" + traceback.format_exc())
            state.finalize(success=False, message=msg)
            failures.append({"profile": profile, "error": msg})
            if not continue_on_error:
                break
        finally:
            _write_root_state(
                out_root=out_root,
                profiles=profiles,
                reports_dir=reports_dir,
                status="running" if not failures else "degraded",
                message=f"completed_profiles={completed_profiles} failures={len(failures)}",
            )

    if completed_profiles:
        _run_summary(
            out_root=out_root,
            profiles=profiles,
            tgt_camera=args.tgt_camera,
            pretrained_ckpt=args.pretrained_ckpt,
            reports_dir=reports_dir,
        )

    final_status = "completed" if not failures else "failed"
    final_message = (
        "all requested profiles completed"
        if not failures
        else f"completed={len(completed_profiles)} failed={len(failures)}"
    )
    _write_root_state(
        out_root=out_root,
        profiles=profiles,
        reports_dir=reports_dir,
        status=final_status,
        message=final_message,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
