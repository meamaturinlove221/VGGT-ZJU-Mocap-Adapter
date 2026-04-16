from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_one_step_probe.run_task import (  # noqa: E402
    _clear_local_dir,
    _download_volume_tree,
    _extract_tagged_line,
    _item_name,
    _item_type,
    _latest_raw_compare_run_dir,
    _modal_get,
    _modal_ls,
    _modal_run,
    _quote_args,
    _run_capture,
    _run_stream,
)
from scripts.orig_vggt_stepcurve_probe.common import (  # noqa: E402
    DEFAULT_GHOST_PEAK_MIN_REL,
    DEFAULT_MODAL_MAX_RETRIES,
    DEFAULT_MODAL_OUT_VOLUME,
    DEFAULT_MODAL_RETRY_SLEEP_SEC,
    DEFAULT_MODAL_SCRIPT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REMOTE_OUT_ROOT,
    DEFAULT_REMOTE_ZJU_ROOT,
    DEFAULT_REPORTS_DIR,
    DEFAULT_STEP_HORIZONS,
    DEFAULT_TRAIN_PROFILES,
    DEFAULT_ZJU_ROOT,
    REFERENCE_PROFILE,
    build_task_specs,
    compare_dir,
    load_json,
    load_jsonl,
    parse_on_off,
    parse_profiles,
    parse_step_horizons,
    profile_dirs,
    profile_metadata,
    profile_tag,
    remote_geom_subdir,
    step_tag,
    to_int,
    to_volume_path,
    train_dir,
    write_json,
    write_text,
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _task_names_for_profile(profile: str, step_horizons: list[int]) -> list[str]:
    tag = profile_tag(profile)
    names = [
        f"profile_{tag}_prepare",
        f"profile_{tag}_precompute_geom",
        f"profile_{tag}_step0000_compare",
        f"profile_{tag}_step0000_sync_compare",
        f"profile_{tag}_step0000_score_ghost",
        f"profile_{tag}_step0000_measure_support",
    ]
    for step in step_horizons:
        stag = step_tag(step)
        names.extend(
            [
                f"profile_{tag}_{stag}_train",
                f"profile_{tag}_{stag}_sync_train",
                f"profile_{tag}_{stag}_prefix_audit",
                f"profile_{tag}_{stag}_compare",
                f"profile_{tag}_{stag}_sync_compare",
                f"profile_{tag}_{stag}_score_ghost",
                f"profile_{tag}_{stag}_measure_support",
            ]
        )
    names.append(f"profile_{tag}_trend_summary")
    return names


class RootTaskState:
    def __init__(self, path_json: Path, path_md: Path, payload: dict, *, resume: bool):
        self.path_json = path_json
        self.path_md = path_md
        if resume and path_json.is_file():
            self.payload = load_json(path_json)
        else:
            self.payload = payload
        self.payload.setdefault("tasks", [])
        existing = {task["name"]: task for task in self.payload["tasks"]}
        for spec in payload["tasks"]:
            if spec["name"] not in existing:
                self.payload["tasks"].append(spec)
            elif str(existing[spec["name"]].get("status", "")) == "in_progress":
                existing[spec["name"]]["status"] = "pending"
        self.payload["updated_at"] = _now_iso()
        self.save()

    def _task(self, name: str) -> dict:
        for task in self.payload["tasks"]:
            if str(task.get("name")) == str(name):
                return task
        raise KeyError(name)

    def status(self, name: str) -> str:
        return str(self._task(name).get("status", "pending"))

    def outputs(self, name: str) -> dict:
        return dict(self._task(name).get("outputs", {}))

    def deps_failed(self, name: str) -> bool:
        task = self._task(name)
        return any(self.status(dep) in {"failed", "blocked"} for dep in task.get("depends_on", []))

    def deps_ready(self, name: str) -> bool:
        task = self._task(name)
        return all(self.status(dep) == "completed" for dep in task.get("depends_on", []))

    def start(self, name: str, message: str) -> None:
        task = self._task(name)
        task["attempt"] = int(task.get("attempt", 0)) + 1
        task["status"] = "in_progress"
        task["started_at"] = task.get("started_at") or _now_iso()
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        self.payload["status"] = "running"
        self.payload["current_task"] = str(name)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def complete(self, name: str, message: str, outputs: dict | None = None) -> None:
        task = self._task(name)
        task["status"] = "completed"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        if outputs:
            task.setdefault("outputs", {}).update(outputs)
        self.payload["current_task"] = str(name)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def fail(self, name: str, message: str) -> None:
        task = self._task(name)
        task["status"] = "failed"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        task.setdefault("errors", []).append({"at": _now_iso(), "message": str(message)})
        self.payload["current_task"] = str(name)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def block(self, name: str, message: str) -> None:
        task = self._task(name)
        if self.status(name) == "completed":
            return
        task["status"] = "blocked"
        task["updated_at"] = _now_iso()
        task["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def finalize(self, status: str, message: str) -> None:
        self.payload["status"] = str(status)
        self.payload["message"] = str(message)
        self.payload["updated_at"] = _now_iso()
        self.save()

    def save(self) -> None:
        write_json(self.path_json, self.payload)
        lines = [
            "# Native VGGT Step-Curve Probe Task State",
            "",
            f"- status: `{self.payload.get('status', '')}`",
            f"- current_task: `{self.payload.get('current_task', '')}`",
            f"- message: {self.payload.get('message', '')}",
            f"- updated_at: `{self.payload.get('updated_at', '')}`",
            "",
            "## Tasks",
            "",
        ]
        for task in self.payload.get("tasks", []):
            lines.append(
                f"- `{task.get('name', '')}` status=`{task.get('status', '')}` "
                f"attempt=`{task.get('attempt', 0)}` updated_at=`{task.get('updated_at', '')}`"
            )
            if str(task.get("message", "")).strip():
                lines.append(f"- note: {task.get('message', '')}")
        write_text(self.path_md, "\n".join(lines) + "\n")


def _stage_ready(stage_dir: Path) -> bool:
    required = [
        stage_dir / "weight_native.png",
        stage_dir / "pred_native.png",
        stage_dir / "report.json",
        stage_dir / "cat_fg_mask_pred_tgt_step000000.png",
    ]
    return all(path.is_file() for path in required)


def _train_logs_ready(train_root: Path) -> bool:
    required = [
        train_root / "logs" / "finetune_vggt_metrics.jsonl",
        train_root / "logs" / "finetune_vggt_summary.json",
    ]
    return all(path.is_file() for path in required)


def _remote_train_artifacts(volume: str, remote_train_run_root: str) -> dict:
    root = str(remote_train_run_root).strip()
    if not root:
        return {}
    logs_entries = _modal_ls(volume, to_volume_path(f"{root.rstrip('/')}/logs"))
    ckpt_entries = _modal_ls(volume, to_volume_path(f"{root.rstrip('/')}/ckpt"))
    log_names = {
        Path(_item_name(item)).name
        for item in logs_entries
        if _item_type(item) in {"", "file"}
    }
    ckpt_names = {
        Path(_item_name(item)).name
        for item in ckpt_entries
        if _item_type(item) in {"", "file"}
    }

    def _remote_ckpt(name: str) -> str:
        return f"{root.rstrip('/')}/ckpt/{name}"

    return {
        "log_names": sorted(log_names),
        "ckpt_names": sorted(ckpt_names),
        "has_metrics": "finetune_vggt_metrics.jsonl" in log_names,
        "has_summary": "finetune_vggt_summary.json" in log_names,
        "has_out_last": "model_ft_zju_last.pt" in ckpt_names,
        "has_out_last_tmp": "model_ft_zju_last.pt.tmp" in ckpt_names,
        "has_out_best": "model_ft_zju.pt" in ckpt_names,
        "remote_metrics": f"{root.rstrip('/')}/logs/finetune_vggt_metrics.jsonl"
        if "finetune_vggt_metrics.jsonl" in log_names
        else "",
        "remote_summary": f"{root.rstrip('/')}/logs/finetune_vggt_summary.json"
        if "finetune_vggt_summary.json" in log_names
        else "",
        "remote_out_last": _remote_ckpt("model_ft_zju_last.pt")
        if "model_ft_zju_last.pt" in ckpt_names
        else (
            _remote_ckpt("model_ft_zju_last.pt.tmp")
            if "model_ft_zju_last.pt.tmp" in ckpt_names
            else ""
        ),
        "remote_out_best": _remote_ckpt("model_ft_zju.pt") if "model_ft_zju.pt" in ckpt_names else "",
        "has_debug_vis": "ft_debug_vis" in {
            Path(_item_name(item)).name for item in logs_entries if _item_type(item) == "dir"
        },
    }


def _remote_train_ready(volume: str, remote_train_run_root: str) -> bool:
    artifacts = _remote_train_artifacts(volume, remote_train_run_root)
    return bool(artifacts.get("has_metrics") and artifacts.get("remote_out_last"))


def _train_metrics_rollup(rows: list[dict]) -> dict:
    run_meta = next((row for row in rows if row.get("event") == "run_meta"), {})
    step_eval_rows = [row for row in rows if row.get("event") == "step_eval"]
    epoch_end_rows = [row for row in rows if row.get("event") == "epoch_end"]
    last_epoch = epoch_end_rows[-1] if epoch_end_rows else {}
    final_step = max((to_int(row.get("step")) for row in step_eval_rows), default=0)
    if final_step <= 0:
        final_step = sum(to_int(row.get("steps")) for row in epoch_end_rows)
    return {
        "run_meta": run_meta,
        "step_eval_rows": step_eval_rows,
        "epoch_end_rows": epoch_end_rows,
        "step_eval_count": int(len(step_eval_rows)),
        "epoch_count": int(len(epoch_end_rows)),
        "epoch_end_steps_total": int(sum(to_int(row.get("steps")) for row in epoch_end_rows)),
        "epoch_end_steps_last": int(to_int(last_epoch.get("steps"))) if epoch_end_rows else 0,
        "final_step_from_metrics": int(final_step),
        "last_step_eval": int(max((to_int(row.get("step")) for row in step_eval_rows), default=0)),
        "best_epoch": int(to_int(last_epoch.get("best_epoch"), default=-1)) if epoch_end_rows else -1,
        "best_loss": float(last_epoch.get("best_loss", float("nan"))) if epoch_end_rows else float("nan"),
        "final_epoch": int(to_int(last_epoch.get("epoch"), default=-1)) if epoch_end_rows else -1,
    }


def _write_local_train_summary(
    train_root: Path,
    rows: list[dict],
    remote_artifacts: dict,
    *,
    existing_summary: dict | None = None,
) -> Path:
    logs_root = train_root / "logs"
    summary_path = logs_root / "finetune_vggt_summary.json"
    rollup = _train_metrics_rollup(rows)
    payload = dict(existing_summary or {})
    payload.setdefault("summary_source", "local_sync_remote_copy" if existing_summary else "local_sync_synthesized")
    payload["best_epoch"] = int(payload.get("best_epoch", rollup["best_epoch"]))
    payload["best_loss"] = float(payload.get("best_loss", rollup["best_loss"]))
    payload["final_epoch"] = int(payload.get("final_epoch", rollup["final_epoch"]))
    payload["final_step"] = int(payload.get("final_step", rollup["final_step_from_metrics"]))
    payload["out_best"] = str(payload.get("out_best") or remote_artifacts.get("remote_out_best", ""))
    payload["out_last"] = str(payload.get("out_last") or remote_artifacts.get("remote_out_last", ""))
    payload["step_eval_count"] = int(rollup["step_eval_count"])
    payload["epoch_count"] = int(rollup["epoch_count"])
    payload["epoch_end_steps_total"] = int(rollup["epoch_end_steps_total"])
    payload["epoch_end_steps_last"] = int(rollup["epoch_end_steps_last"])
    write_json(summary_path, payload)
    return summary_path


def _sync_train_logs(volume: str, remote_train_run_root: str, local_train_root: Path) -> dict:
    logs_root = local_train_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict = {}
    for _ in range(24):
        artifacts = _remote_train_artifacts(volume, remote_train_run_root)
        if artifacts.get("has_metrics") and artifacts.get("remote_out_last"):
            break
        time.sleep(10)
    else:
        raise RuntimeError(
            f"remote train artifacts not ready under {remote_train_run_root}; "
            f"artifacts={json.dumps(artifacts, ensure_ascii=False)}"
        )

    _modal_get(
        volume,
        to_volume_path(str(artifacts.get("remote_metrics", ""))),
        logs_root / "finetune_vggt_metrics.jsonl",
    )
    existing_summary: dict | None = None
    if str(artifacts.get("remote_summary", "")).strip():
        _modal_get(
            volume,
            to_volume_path(str(artifacts.get("remote_summary", ""))),
            logs_root / "finetune_vggt_summary.json",
        )
        existing_summary = load_json(logs_root / "finetune_vggt_summary.json")
    rows = load_jsonl(logs_root / "finetune_vggt_metrics.jsonl")
    _write_local_train_summary(local_train_root, rows, artifacts, existing_summary=existing_summary)

    if bool(artifacts.get("has_debug_vis")):
        _download_volume_tree(
            volume=volume,
            remote_dir=to_volume_path(f"{str(remote_train_run_root).rstrip('/')}/logs/ft_debug_vis"),
            local_dir=logs_root,
            flatten_root=True,
        )
    return {
        "remote_out_last_ckpt": str(artifacts.get("remote_out_last", "")),
        "remote_out_best_ckpt": str(artifacts.get("remote_out_best", "")),
        "remote_summary_present": bool(artifacts.get("has_summary")),
    }


def _sync_compare_stage(volume: str, remote_run_dir: str, local_stage_dir: Path) -> None:
    remote_root = str(remote_run_dir).strip()
    if not remote_root:
        raise RuntimeError("remote compare run dir missing")
    required = {
        "weight_native.png",
        "pred_native.png",
        "tgt_native.png",
        "report.json",
        "cat_weight_pred_tgt.png",
        "cat_fg_mask_pred_tgt_step000000.png",
    }
    entries: list[dict] = []
    for _ in range(18):
        entries = _modal_ls(volume, to_volume_path(remote_root))
        file_names = {
            Path(_item_name(item)).name
            for item in entries
            if _item_type(item) in {"", "file"}
        }
        if required.issubset(file_names):
            break
        time.sleep(10)
    else:
        raise RuntimeError(
            f"remote compare outputs not ready under {remote_root}; "
            f"missing={sorted(required - file_names)}"
        )

    _clear_local_dir(local_stage_dir)
    _download_volume_tree(
        volume=volume,
        remote_dir=to_volume_path(remote_root),
        local_dir=local_stage_dir,
        flatten_root=True,
    )


def _sync_prefix_trace(volume: str, remote_trace_path: str, local_trace_path: Path) -> None:
    remote_path = to_volume_path(remote_trace_path)
    target_name = Path(remote_path).name
    parent_dir = str(Path(remote_path).parent).replace("\\", "/")
    entries: list[dict] = []
    for _ in range(18):
        entries = _modal_ls(volume, parent_dir)
        file_names = {
            Path(_item_name(item)).name
            for item in entries
            if _item_type(item) in {"", "file"}
        }
        if target_name in file_names:
            break
        time.sleep(10)
    else:
        raise RuntimeError(f"remote prefix trace not ready under {parent_dir}: {target_name}")
    _modal_get(volume, remote_path, local_trace_path)


def _prefix_ready(train_root: Path) -> bool:
    return (train_root / "train_prefix_trace.json").is_file()


def _support_ready(stage_dir: Path) -> bool:
    required = [
        stage_dir / "point_support_metrics.json",
        stage_dir / "weight_native_subject_bbox.png",
        stage_dir / "cat_weight_pred_tgt_subject_bbox.png",
    ]
    return all(path.is_file() for path in required)


def _validate_train_logs(train_root: Path, horizon: int) -> dict:
    rows = load_jsonl(train_root / "logs" / "finetune_vggt_metrics.jsonl")
    summary = load_json(train_root / "logs" / "finetune_vggt_summary.json")
    rollup = _train_metrics_rollup(rows)
    run_meta = rollup["run_meta"]
    strict = bool(run_meta.get("strict_deterministic", False))
    if not strict:
        raise RuntimeError("run_meta.strict_deterministic != true")
    if to_int(run_meta.get("max_steps_per_epoch")) != 1:
        raise RuntimeError("run_meta.max_steps_per_epoch != 1")
    if int(rollup["step_eval_count"]) != int(horizon):
        raise RuntimeError(f"step_eval_count != {int(horizon)}")
    if int(rollup["epoch_end_steps_total"]) != int(horizon):
        raise RuntimeError(f"epoch_end_steps_total != {int(horizon)}")
    if int(rollup["last_step_eval"]) != int(horizon):
        raise RuntimeError(f"last_step_eval != {int(horizon)}")
    if to_int(summary.get("final_step")) != int(horizon):
        raise RuntimeError(f"train_summary.final_step != {int(horizon)}")
    out_last = str(summary.get("out_last", "")).strip()
    if not out_last:
        raise RuntimeError("out_last missing in finetune summary")
    return {
        "out_last": out_last,
        "strict_deterministic": strict,
        "step_eval_count": int(rollup["step_eval_count"]),
        "epoch_end_steps": int(rollup["epoch_end_steps_total"]),
        "epoch_end_steps_last": int(rollup["epoch_end_steps_last"]),
        "epoch_count": int(rollup["epoch_count"]),
    }


def _sync_and_validate_train(
    *,
    volume: str,
    remote_train_run_root: str,
    local_train_root: Path,
    horizon: int,
) -> tuple[str, dict]:
    _clear_local_dir(local_train_root)
    sync_info = _sync_train_logs(volume, remote_train_run_root, local_train_root)
    train_info = _validate_train_logs(local_train_root, horizon)
    outputs = {"local_train_dir": str(local_train_root)}
    outputs.update(sync_info)
    outputs.update(train_info)
    return "train logs synced and validated", outputs


def _run_python(script: Path, args: list[str], label: str) -> None:
    rc, _ = _run_stream(
        [sys.executable, "-u", str(script), *args],
        cwd=REPO_ROOT,
        env=None,
        label=label,
    )
    if rc != 0:
        raise RuntimeError(f"{label} failed rc={rc}")


def _load_profile_manifest(manifest_path: Path) -> dict:
    if manifest_path.is_file():
        return load_json(manifest_path)
    return {}


def _write_profile_manifest(manifest_path: Path, payload: dict) -> None:
    write_json(manifest_path, payload)


def _validate_ghost_exact_row(stage_dir: Path) -> None:
    import csv

    rows_path = stage_dir / "ghost_score_rows.csv"
    if not rows_path.is_file():
        raise RuntimeError(f"ghost rows missing: {rows_path}")
    target = str((stage_dir / "cat_fg_mask_pred_tgt_step000000.png").resolve()).replace("\\", "/")
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    hits = [row for row in rows if str(Path(str(row.get("path", ""))).resolve()).replace("\\", "/") == target]
    if len(hits) != 1:
        raise RuntimeError(f"ghost peak exact row match failed for {target}; matched={len(hits)}")


def _list_nonstopped_apps() -> list[dict]:
    text = _run_capture(["modal", "app", "list", "--json"], cwd=REPO_ROOT, env=None, label="modal_app_list").strip()
    payload = json.loads(text or "[]")
    apps = payload if isinstance(payload, list) else list(payload.get("items", []))
    out: list[dict] = []
    for app in apps:
        name = str(app.get("name", app.get("Name", "")))
        state = str(app.get("state", app.get("State", app.get("status", app.get("Status", ""))))).lower()
        app_id = str(app.get("app_id", app.get("App ID", app.get("id", app.get("Id", "")))))
        if state and state != "stopped":
            out.append({"app_id": app_id, "name": name, "state": state})
    return out


def _stop_apps(apps: list[dict]) -> None:
    for app in apps:
        app_id = str(app.get("app_id", "")).strip()
        if not app_id:
            continue
        _run_capture(["modal", "app", "stop", app_id], cwd=REPO_ROOT, env=None, label=f"modal_app_stop:{app_id}")


def _measure_support(compare_stage_dir: Path, local_zju_root: Path, support_threshold_abs: str) -> dict:
    out_json = compare_stage_dir / "point_support_metrics.json"
    _run_python(
        REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "measure_point_support.py",
        [
            "--compare_dir",
            str(compare_stage_dir),
            "--local_zju_root",
            str(local_zju_root),
            "--support_threshold_abs",
            str(support_threshold_abs),
            "--out_json",
            str(out_json),
            "--out_weight_overlay",
            str(compare_stage_dir / "weight_native_subject_bbox.png"),
            "--out_cat_overlay",
            str(compare_stage_dir / "cat_weight_pred_tgt_subject_bbox.png"),
        ],
        label=f"measure_support:{compare_stage_dir}",
    )
    return load_json(out_json)


def _score_ghost(compare_stage_dir: Path, profile: str) -> None:
    _run_python(
        REPO_ROOT / "tools" / "score_ghosting_from_cat_pred.py",
        [
            "--input",
            f"{profile}={str((compare_stage_dir / 'cat_fg_mask_pred_tgt_step000000.png').resolve())}",
            "--out_csv",
            str(compare_stage_dir / "ghost_score_rows.csv"),
            "--out_summary_csv",
            str(compare_stage_dir / "ghost_score_summary.csv"),
            "--out_json",
            str(compare_stage_dir / "ghost_score.json"),
            "--peak_min_rel",
            str(float(DEFAULT_GHOST_PEAK_MIN_REL)),
        ],
        label=f"score_ghost:{profile}:{compare_stage_dir.name}",
    )
    _validate_ghost_exact_row(compare_stage_dir)


def _modal_run_script(
    *,
    modal_script: str,
    code_dir: Path,
    gpu_spec_precompute: str,
    remote_zju_root: str,
    seq_name: str,
    script_path: str,
    ckpt_path: str,
    quoted_args: str,
    label: str,
    modal_max_retries: int,
    modal_retry_sleep_sec: int,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    env_updates = {
        "VGGT_CODE_DIR": str(code_dir),
        "VGGT_MODE": "precompute",
        "VGGT_GPU_SPEC_PRECOMPUTE": str(gpu_spec_precompute),
        "VGGT_PRECOMPUTE_SCRIPT": str(script_path),
        "VGGT_PRECOMPUTE_CKPT": str(ckpt_path),
        "VGGT_ZJU_ROOT": str(remote_zju_root),
        "VGGT_SEQ_NAMES": str(seq_name),
        "VGGT_PRECOMPUTE_ARGS_EXTRA": str(quoted_args),
    }
    if extra_env:
        env_updates.update({str(k): str(v) for k, v in extra_env.items()})
    return _modal_run(
        modal_script=modal_script,
        env_updates=env_updates,
        code_dir=code_dir,
        max_retries=int(modal_max_retries),
        retry_sleep_sec=int(modal_retry_sleep_sec),
        label=label,
    )


def _modal_run_train_script(
    *,
    modal_script: str,
    code_dir: Path,
    gpu_spec_train: str,
    remote_zju_root: str,
    seq_name: str,
    script_path: str,
    forwarded_args: list[str],
    label: str,
    modal_max_retries: int,
    modal_retry_sleep_sec: int,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    forwarded_blob = base64.b64encode(
        json.dumps([str(x) for x in forwarded_args], ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    env_updates = {
        "VGGT_CODE_DIR": str(code_dir),
        "VGGT_MODE": "train",
        "VGGT_GPU_SPEC_TRAIN": str(gpu_spec_train),
        "VGGT_TRAIN_SCRIPT": str(script_path),
        "VGGT_TRAIN_ARGS_EXTRA": f"--stepcurve_forward_b64={forwarded_blob}",
        "VGGT_ZJU_ROOT": str(remote_zju_root),
        "VGGT_SEQ_NAMES": str(seq_name),
    }
    if extra_env:
        env_updates.update({str(k): str(v) for k, v in extra_env.items()})
    return _modal_run(
        modal_script=modal_script,
        env_updates=env_updates,
        code_dir=code_dir,
        max_retries=int(modal_max_retries),
        retry_sleep_sec=int(modal_retry_sleep_sec),
        label=label,
    )


def main() -> None:
    ap = argparse.ArgumentParser("orig_vggt_stepcurve_probe")
    ap.add_argument("--profiles", default=",".join(DEFAULT_TRAIN_PROFILES))
    ap.add_argument("--step_horizons", default=",".join(str(x) for x in DEFAULT_STEP_HORIZONS))
    ap.add_argument("--seq_name", default="CoreView_390")
    ap.add_argument("--frame_id", type=int, default=1080)
    ap.add_argument("--tgt_camera", default="Camera_B5")
    ap.add_argument("--pretrained_ckpt", default="model.pt")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict_deterministic", default="on")
    ap.add_argument("--gpu_spec_precompute", default="A100-80GB")
    ap.add_argument("--resume_task_state", default="on")
    ap.add_argument("--continue_on_profile_error", default="on")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--reports_dir", default=str(DEFAULT_REPORTS_DIR))
    ap.add_argument("--code_dir", default=str(REPO_ROOT))
    ap.add_argument("--remote_zju_root", default=str(DEFAULT_REMOTE_ZJU_ROOT))
    ap.add_argument("--remote_out_root", default=str(DEFAULT_REMOTE_OUT_ROOT))
    ap.add_argument("--modal_script", default=str(DEFAULT_MODAL_SCRIPT.name))
    ap.add_argument("--modal_out_volume", default=str(DEFAULT_MODAL_OUT_VOLUME))
    ap.add_argument("--modal_max_retries", type=int, default=int(DEFAULT_MODAL_MAX_RETRIES))
    ap.add_argument("--modal_retry_sleep_sec", type=int, default=int(DEFAULT_MODAL_RETRY_SLEEP_SEC))
    ap.add_argument("--local_zju_root", default=str(DEFAULT_ZJU_ROOT))
    args = ap.parse_args()

    profiles = parse_profiles(args.profiles)
    step_horizons = parse_step_horizons(args.step_horizons)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (REPO_ROOT / reports_dir).resolve()
    code_dir = Path(args.code_dir)
    if not code_dir.is_absolute():
        code_dir = (REPO_ROOT / code_dir).resolve()
    local_zju_root = Path(args.local_zju_root)
    if not local_zju_root.is_absolute():
        local_zju_root = (REPO_ROOT / local_zju_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "pending",
        "current_task": "",
        "message": "",
        "updated_at": _now_iso(),
        "profiles": profiles,
        "step_horizons": step_horizons,
        "tasks": build_task_specs(profiles, step_horizons),
    }
    resume_on = parse_on_off(args.resume_task_state, default=True)
    state = RootTaskState(
        out_root / "task_state_latest.json",
        out_root / "task_state_latest.md",
        payload,
        resume=resume_on,
    )
    continue_on_profile_error = parse_on_off(args.continue_on_profile_error, default=True)
    strict_deterministic_flag = parse_on_off(args.strict_deterministic, default=True)
    failures: list[str] = []
    validator_wait_rounds = max(1, int(args.modal_retry_sleep_sec))
    validator_wait_sleep_sec = max(2, int(args.modal_retry_sleep_sec))

    def _validator_ok(fn) -> bool:
        try:
            return bool(fn())
        except Exception:
            return False

    def execute_task(
        name: str,
        *,
        validator,
        runner,
        ignore_failed_deps: bool = False,
    ) -> bool:
        if state.status(name) == "completed" and _validator_ok(validator):
            return True
        if state.deps_failed(name) and not ignore_failed_deps:
            state.block(name, "dependency failed or blocked")
            return False
        if not state.deps_ready(name) and not ignore_failed_deps:
            return False
        task = state._task(name)
        while int(task.get("attempt", 0)) < int(task.get("max_attempts", 1)):
            state.start(name, "running")
            try:
                message, outputs = runner()
                if outputs:
                    task.setdefault("outputs", {}).update(outputs)
                    task["updated_at"] = _now_iso()
                    state.payload["updated_at"] = _now_iso()
                    state.save()
                validator_ok = _validator_ok(validator)
                if not validator_ok:
                    for _ in range(validator_wait_rounds):
                        time.sleep(validator_wait_sleep_sec)
                        validator_ok = _validator_ok(validator)
                        if validator_ok:
                            break
                if not validator_ok:
                    raise RuntimeError("validator check failed after task run")
                state.complete(name, message, outputs)
                return True
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if int(task.get("attempt", 0)) >= int(task.get("max_attempts", 1)):
                    state.fail(name, msg + "\n" + traceback.format_exc())
                    failures.append(f"{name}: {msg}")
                    return False
                state._task(name).setdefault("errors", []).append({"at": _now_iso(), "message": msg})
                state._task(name)["status"] = "pending"
                state._task(name)["updated_at"] = _now_iso()
                state.save()
                time.sleep(5)
        return False

    def remote_stage_root(profile: str, step: int) -> str:
        return f"{str(args.remote_out_root).rstrip('/')}/{profile}/{step_tag(step)}/compare"

    def remote_train_run_root(profile: str, step: int) -> str:
        return f"{str(args.remote_out_root).rstrip('/')}/{profile}/{step_tag(step)}/train_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    execute_task(
        "bootstrap_preflight",
        validator=lambda: True,
        runner=lambda: (
            (
                lambda apps_before, apps_after, modal_version: (
                    "bootstrap preflight ok",
                    {
                        "modal_version": modal_version,
                        "apps_stopped_count": len(apps_before),
                        "apps_remaining_after_preflight": len(apps_after),
                        "reference_source": str((REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_one_step_probe" / REFERENCE_PROFILE).resolve()),
                    },
                )
            )(
                (lambda apps: (_stop_apps(apps), apps)[1])(_list_nonstopped_apps()),
                _list_nonstopped_apps(),
                _run_capture(["modal", "--version"], cwd=REPO_ROOT, env=None, label="modal_version").strip(),
            )
        ),
        ignore_failed_deps=True,
    )

    for profile in profiles:
        tag = profile_tag(profile)
        meta = profile_metadata(profile, args.tgt_camera)
        pdirs = profile_dirs(out_root, profile)
        geom_subdir = f"{remote_geom_subdir(profile)}_stepcurve"
        manifest_path = pdirs["profile_manifest_json"]

        execute_task(
            f"profile_{tag}_prepare",
            validator=lambda pdirs=pdirs: pdirs["root"].is_dir(),
            runner=lambda pdirs=pdirs, meta=meta, geom_subdir=geom_subdir: (
                pdirs["root"].mkdir(parents=True, exist_ok=True),
                pdirs["trend_dir"].mkdir(parents=True, exist_ok=True),
                ("profile directories ready", {"profile_root": str(pdirs["root"]), "geom_subdir": geom_subdir, "train_cameras_csv": ",".join(meta["train_cameras"])}),
            )[-1],
        )

        execute_task(
            f"profile_{tag}_precompute_geom",
            validator=lambda: True,
            runner=lambda profile=profile, meta=meta, geom_subdir=geom_subdir: (
                _modal_run_script(
                    modal_script=args.modal_script,
                    code_dir=code_dir,
                    gpu_spec_precompute=args.gpu_spec_precompute,
                    remote_zju_root=args.remote_zju_root,
                    seq_name=args.seq_name,
                    script_path="precompute_zju_vggt_geom.py",
                    ckpt_path=args.pretrained_ckpt,
                    quoted_args="",
                    label=f"{profile}:precompute_geom",
                    modal_max_retries=int(args.modal_max_retries),
                    modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    extra_env={
                        "VGGT_CAM_NAMES": ",".join(meta["train_cameras"]),
                        "VGGT_MAX_FRAMES": str(int(args.max_frames)),
                        "VGGT_GEOM_SUBDIR": geom_subdir,
                    },
                ),
                ("remote precompute finished", {"geom_subdir": geom_subdir}),
            )[-1],
        )

        step0_compare_dir = compare_dir(out_root, profile, 0)
        execute_task(
            f"profile_{tag}_step0000_compare",
            validator=lambda: bool(state.outputs(f"profile_{tag}_step0000_compare").get("remote_run_dir")),
            runner=lambda profile=profile: (
                (
                    lambda lines: (
                        "remote step0000 compare finished",
                        {
                            "remote_run_dir": _extract_tagged_line(lines, "RUN_DIR:")
                            or _latest_raw_compare_run_dir(
                                remote_stage_root=remote_stage_root(profile, 0),
                                seq_name=args.seq_name,
                                frame_id=int(args.frame_id),
                                tgt_camera=args.tgt_camera,
                                volume=str(args.modal_out_volume),
                            )
                        },
                    )
                )(
                    _modal_run_script(
                        modal_script=args.modal_script,
                        code_dir=code_dir,
                        gpu_spec_precompute=args.gpu_spec_precompute,
                        remote_zju_root=args.remote_zju_root,
                        seq_name=args.seq_name,
                        script_path="scripts/orig_vggt_viewcount/render_raw_compare.py",
                        ckpt_path=args.pretrained_ckpt,
                        quoted_args=_quote_args(
                            [
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
                                str(remote_stage_root(profile, 0)),
                            ]
                        ),
                        label=f"{profile}:step0000_compare",
                        modal_max_retries=int(args.modal_max_retries),
                        modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                    )
                )
            ),
        )

        execute_task(
            f"profile_{tag}_step0000_sync_compare",
            validator=lambda d=step0_compare_dir: _stage_ready(d),
            runner=lambda d=step0_compare_dir: (
                _sync_compare_stage(
                    volume=str(args.modal_out_volume),
                    remote_run_dir=str(state.outputs(f"profile_{tag}_step0000_compare").get("remote_run_dir", "")),
                    local_stage_dir=d,
                ),
                ("step0000 compare synced", {"local_compare_dir": str(d)}),
            )[-1],
        )

        execute_task(
            f"profile_{tag}_step0000_score_ghost",
            validator=lambda d=step0_compare_dir: (d / "ghost_score_rows.csv").is_file() and (d / "ghost_score.json").is_file(),
            runner=lambda d=step0_compare_dir, profile=profile: (_score_ghost(d, profile), ("step0000 ghost scored", {"ghost_json": str(d / "ghost_score.json")}))[-1],
        )

        execute_task(
            f"profile_{tag}_step0000_measure_support",
            validator=lambda d=step0_compare_dir: _support_ready(d),
            runner=lambda d=step0_compare_dir, profile=profile, meta=meta: (
                (
                    lambda metrics: (
                        _write_profile_manifest(
                            manifest_path,
                            {
                                "profile": profile,
                                "seq_name": args.seq_name,
                                "frame_id": int(args.frame_id),
                                "tgt_camera": args.tgt_camera,
                                "geom_subdir": geom_subdir,
                                "support_threshold_abs": float(metrics["support_threshold_abs"]),
                                "mask_source_resolved": str(metrics["mask_source_resolved"]),
                                "train_cameras": meta["train_cameras"],
                            },
                        ),
                        (
                            "step0000 point-support measured",
                            {"support_threshold_abs": float(metrics["support_threshold_abs"])},
                        ),
                    )[-1]
                )(_measure_support(d, local_zju_root, "auto"))
            ),
        )

        for step in step_horizons:
            stag = step_tag(step)
            train_root = train_dir(out_root, profile, step)
            step_compare_dir = compare_dir(out_root, profile, step)
            remote_run_root = remote_train_run_root(profile, step)
            remote_out_last = f"{remote_run_root}/ckpt/model_ft_zju_last.pt"

            execute_task(
                f"profile_{tag}_{stag}_train",
                validator=lambda name=f"profile_{tag}_{stag}_train", remote_run_root=remote_run_root: bool(
                    str(state.outputs(name).get("remote_train_run_root", remote_run_root)).strip()
                ),
                runner=lambda profile=profile, meta=meta, step=step, remote_run_root=remote_run_root: (
                    _modal_run_train_script(
                        modal_script=args.modal_script,
                        code_dir=code_dir,
                        gpu_spec_train=args.gpu_spec_precompute,
                        remote_zju_root=args.remote_zju_root,
                        seq_name=args.seq_name,
                        script_path="scripts/orig_vggt_stepcurve_probe/run_finetune_strict.py",
                        forwarded_args=[
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
                            str(int(step)),
                            "--max_frames",
                            str(int(args.max_frames)),
                            "--max_steps_per_epoch",
                            "1",
                            "--eval_every_steps",
                            "1",
                            "--debug_metrics_every_steps",
                            "1",
                            "--debug_vis_every_steps",
                            "1",
                            "--debug_vis_max_steps",
                            str(int(step)),
                            "--debug_vis_views",
                            "1",
                            "--seed",
                            str(int(args.seed)),
                            "--geom_subdir",
                            geom_subdir,
                            "--log_dir",
                            f"{remote_run_root}/logs",
                            "--ckpt_dir",
                            f"{remote_run_root}/ckpt",
                        ],
                        label=f"{profile}:{stag}_train",
                        modal_max_retries=int(args.modal_max_retries),
                        modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                        extra_env={"VGGT_STRICT_DETERMINISTIC": "1" if strict_deterministic_flag else "0"},
                    ),
                    ("remote train finished", {"remote_train_run_root": remote_run_root, "remote_out_last_ckpt": remote_out_last}),
                )[-1],
            )

            execute_task(
                f"profile_{tag}_{stag}_sync_train",
                validator=lambda tr=train_root, step=step: _train_logs_ready(tr) and bool(_validate_train_logs(tr, step)),
                runner=lambda tr=train_root, name=f"profile_{tag}_{stag}_train", step=step: _sync_and_validate_train(
                    volume=str(args.modal_out_volume),
                    remote_train_run_root=str(state.outputs(name).get("remote_train_run_root", remote_run_root)),
                    local_train_root=tr,
                    horizon=int(step),
                ),
            )

            execute_task(
                f"profile_{tag}_{stag}_prefix_audit",
                validator=lambda tr=train_root: _prefix_ready(tr),
                runner=lambda tr=train_root, profile=profile, meta=meta, step=step: (
                    (
                        lambda remote_trace_path: (
                            _modal_run_script(
                                modal_script=args.modal_script,
                                code_dir=code_dir,
                                gpu_spec_precompute=args.gpu_spec_precompute,
                                remote_zju_root=args.remote_zju_root,
                                seq_name=args.seq_name,
                                script_path="scripts/orig_vggt_stepcurve_probe/audit_prefix.py",
                                ckpt_path=args.pretrained_ckpt,
                                quoted_args=_quote_args(
                                    [
                                        "--profile",
                                        str(profile),
                                        "--zju_root",
                                        str(args.remote_zju_root),
                                        "--seq_name",
                                        str(args.seq_name),
                                        "--cam_names",
                                        ",".join(meta["train_cameras"]),
                                        "--geom_subdir",
                                        geom_subdir,
                                        "--max_frames",
                                        str(int(args.max_frames)),
                                        "--seed",
                                        str(int(args.seed)),
                                        "--step_horizon",
                                        str(int(step)),
                                        "--tgt_camera",
                                        str(args.tgt_camera),
                                        "--strict_deterministic",
                                        "on" if strict_deterministic_flag else "off",
                                        "--out_json",
                                        remote_trace_path,
                                    ]
                                ),
                                label=f"{profile}:{stag}_prefix_audit",
                                modal_max_retries=int(args.modal_max_retries),
                                modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                            ),
                            _sync_prefix_trace(
                                str(args.modal_out_volume),
                                remote_trace_path,
                                tr / "train_prefix_trace.json",
                            ),
                            ("prefix audit synced", {"prefix_trace_json": str(tr / 'train_prefix_trace.json')}),
                        )[-1]
                    )(
                        f"{str(args.remote_out_root).rstrip('/')}/{profile}/{stag}/train/train_prefix_trace.json"
                    ),
                )[-1],
            )

            execute_task(
                f"profile_{tag}_{stag}_compare",
                validator=lambda name=f"profile_{tag}_{stag}_compare": bool(state.outputs(name).get("remote_run_dir")),
                runner=lambda profile=profile, step=step, train_name=f"profile_{tag}_{stag}_train", sync_name=f"profile_{tag}_{stag}_sync_train": (
                    (
                        lambda lines: (
                            "remote compare finished",
                            {
                                "remote_run_dir": _extract_tagged_line(lines, "RUN_DIR:")
                                or _latest_raw_compare_run_dir(
                                    remote_stage_root=remote_stage_root(profile, step),
                                    seq_name=args.seq_name,
                                    frame_id=int(args.frame_id),
                                    tgt_camera=args.tgt_camera,
                                    volume=str(args.modal_out_volume),
                                )
                            },
                        )
                    )(
                        _modal_run_script(
                            modal_script=args.modal_script,
                            code_dir=code_dir,
                            gpu_spec_precompute=args.gpu_spec_precompute,
                            remote_zju_root=args.remote_zju_root,
                            seq_name=args.seq_name,
                            script_path="scripts/orig_vggt_viewcount/render_raw_compare.py",
                            ckpt_path=str(
                                state.outputs(sync_name).get("remote_out_last_ckpt")
                                or state.outputs(train_name).get("remote_out_last_ckpt", remote_out_last)
                            ),
                            quoted_args=_quote_args(
                                [
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
                                    str(remote_stage_root(profile, step)),
                                ]
                            ),
                            label=f"{profile}:{stag}_compare",
                            modal_max_retries=int(args.modal_max_retries),
                            modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                        )
                    )
                ),
            )

            execute_task(
                f"profile_{tag}_{stag}_sync_compare",
                validator=lambda d=step_compare_dir: _stage_ready(d),
                runner=lambda d=step_compare_dir, name=f"profile_{tag}_{stag}_compare": (
                    _sync_compare_stage(
                        volume=str(args.modal_out_volume),
                        remote_run_dir=str(state.outputs(name).get("remote_run_dir", "")),
                        local_stage_dir=d,
                    ),
                    ("compare synced", {"local_compare_dir": str(d)}),
                )[-1],
            )

            execute_task(
                f"profile_{tag}_{stag}_score_ghost",
                validator=lambda d=step_compare_dir: (d / "ghost_score_rows.csv").is_file() and (d / "ghost_score.json").is_file(),
                runner=lambda d=step_compare_dir, profile=profile: (_score_ghost(d, profile), ("ghost scored", {"ghost_json": str(d / 'ghost_score.json')}))[-1],
            )

            execute_task(
                f"profile_{tag}_{stag}_measure_support",
                validator=lambda d=step_compare_dir: _support_ready(d),
                runner=lambda d=step_compare_dir: (
                    (
                        lambda manifest, metrics: (
                            "point-support measured",
                            {
                                "point_support_json": str(d / "point_support_metrics.json"),
                                "support_threshold_abs": float(metrics["support_threshold_abs"]),
                            },
                        )
                    )(
                        _load_profile_manifest(manifest_path),
                        _measure_support(
                            d,
                            local_zju_root,
                            str(_load_profile_manifest(manifest_path).get("support_threshold_abs", "auto")),
                        ),
                    )
                ),
            )

        execute_task(
            f"profile_{tag}_trend_summary",
            validator=lambda pdirs=pdirs: pdirs["trend_summary_json"].is_file() and pdirs["trend_summary_md"].is_file(),
            runner=lambda profile=profile: (
                _run_python(
                    REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "summarize_runs.py",
                    [
                        "--out_root",
                        str(out_root),
                        "--profiles",
                        str(profile),
                        "--step_horizons",
                        ",".join(str(x) for x in step_horizons),
                        "--out_json",
                        str(reports_dir / f"_{profile}_stepcurve_summary.json"),
                        "--out_csv",
                        str(reports_dir / f"_{profile}_stepcurve_summary.csv"),
                        "--out_md",
                        str(reports_dir / f"_{profile}_stepcurve_summary.md"),
                        "--out_advisor_md",
                        str(reports_dir / f"_{profile}_stepcurve_advisor.md"),
                        "--out_point_grid",
                        str(reports_dir / f"_{profile}_point_grid.png"),
                        "--out_ghost_grid",
                        str(reports_dir / f"_{profile}_ghost_grid.png"),
                    ],
                    label=f"{profile}:trend_summary",
                ),
                ("profile trend summary refreshed", {"trend_summary_json": str(pdirs['trend_summary_json'])}),
            )[-1],
            ignore_failed_deps=True,
        )

        if failures and not continue_on_profile_error:
            break

    execute_task(
        "reference_23src_ingest_existing_onestep",
        validator=lambda: (out_root / "reference_23cam_fullset_one_step" / "summary.json").is_file(),
        runner=lambda: (
            (
                lambda ref_root, src_root, pre_json, post_json: (
                    ref_root.mkdir(parents=True, exist_ok=True),
                    _run_python(
                        REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "measure_point_support.py",
                        [
                            "--compare_dir",
                            str(src_root / "pre_update"),
                            "--local_zju_root",
                            str(local_zju_root),
                            "--support_threshold_abs",
                            "auto",
                            "--out_json",
                            str(pre_json),
                            "--out_weight_overlay",
                            str(ref_root / "pre_update_weight_native_subject_bbox.png"),
                            "--out_cat_overlay",
                            str(ref_root / "pre_update_cat_weight_pred_tgt_subject_bbox.png"),
                        ],
                        label="reference23:pre_support",
                    ),
                    _run_python(
                        REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "measure_point_support.py",
                        [
                            "--compare_dir",
                            str(src_root / "post_update"),
                            "--local_zju_root",
                            str(local_zju_root),
                            "--support_threshold_abs",
                            str(load_json(pre_json)["support_threshold_abs"]),
                            "--out_json",
                            str(post_json),
                            "--out_weight_overlay",
                            str(ref_root / "post_update_weight_native_subject_bbox.png"),
                            "--out_cat_overlay",
                            str(ref_root / "post_update_cat_weight_pred_tgt_subject_bbox.png"),
                        ],
                        label="reference23:post_support",
                    ),
                    write_json(
                        ref_root / "summary.json",
                        {
                            "source_root": str(src_root),
                            "note": "existing one-step reference only; no new stepcurve training",
                            "one_step_compare_summary": load_json(src_root / "compare" / "summary.json"),
                            "pre_update_point_support": load_json(pre_json),
                            "post_update_point_support": load_json(post_json),
                        },
                    ),
                    write_text(ref_root / "summary.md", "# 23cam Fullset Reference\n\n- existing one-step only\n"),
                    ("reference 23src ingested", {"reference_summary_json": str(ref_root / "summary.json")}),
                )[-1]
            )(
                out_root / "reference_23cam_fullset_one_step",
                REPO_ROOT / "logs" / "modal_phase5" / "orig_vggt_one_step_probe" / REFERENCE_PROFILE,
                out_root / "reference_23cam_fullset_one_step" / "pre_update_point_support_metrics.json",
                out_root / "reference_23cam_fullset_one_step" / "post_update_point_support_metrics.json",
            )
        ),
        ignore_failed_deps=True,
    )

    def _ready_profiles_for_global() -> list[str]:
        ready: list[str] = []
        for profile in profiles:
            tag = profile_tag(profile)
            if state.status(f"profile_{tag}_trend_summary") == "completed":
                ready.append(profile)
        return ready

    execute_task(
        "global_summary_refresh",
        validator=lambda: (reports_dir / "orig_vggt_stepcurve_probe_summary_latest.json").is_file(),
        runner=lambda: (
            (
                lambda ready: (
                    (
                        _run_python(
                            REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "summarize_runs.py",
                            [
                                "--out_root",
                                str(out_root),
                                "--profiles",
                                ",".join(ready),
                                "--step_horizons",
                                ",".join(str(x) for x in step_horizons),
                                "--out_json",
                                str(reports_dir / "orig_vggt_stepcurve_probe_summary_latest.json"),
                                "--out_csv",
                                str(reports_dir / "orig_vggt_stepcurve_probe_summary_latest.csv"),
                                "--out_md",
                                str(reports_dir / "orig_vggt_stepcurve_probe_summary_latest.md"),
                                "--out_advisor_md",
                                str(reports_dir / "orig_vggt_stepcurve_probe_advisor_latest.md"),
                                "--out_point_grid",
                                str(reports_dir / "orig_vggt_stepcurve_point_support_grid_latest.png"),
                                "--out_ghost_grid",
                                str(reports_dir / "orig_vggt_stepcurve_ghost_grid_latest.png"),
                            ],
                            label="global_summary_refresh",
                        )
                        if ready
                        else (
                            write_json(
                                reports_dir / "orig_vggt_stepcurve_probe_summary_latest.json",
                                {
                                    "profiles": [],
                                    "reference_23cam_fullset_one_step": load_json(
                                        out_root / "reference_23cam_fullset_one_step" / "summary.json"
                                    )
                                    if (out_root / "reference_23cam_fullset_one_step" / "summary.json").is_file()
                                    else {},
                                    "note": "no ready training profiles yet",
                                },
                            ),
                            write_text(
                                reports_dir / "orig_vggt_stepcurve_probe_summary_latest.csv",
                                "",
                            ),
                            write_text(
                                reports_dir / "orig_vggt_stepcurve_probe_summary_latest.md",
                                "# Native VGGT Step-Curve Probe Summary\n\n- no ready training profiles yet.\n",
                            ),
                            write_text(
                                reports_dir / "orig_vggt_stepcurve_probe_advisor_latest.md",
                                "# 导师口述版\n\n- 训练 profile 还没有 ready summary；当前输出只包含 reference 或失败快照。\n- NO_NONSTOPPED_APPS\n",
                            ),
                        )
                    ),
                    ("global summary refreshed", {"ready_profiles": ready}),
                )[-1]
            )(_ready_profiles_for_global())
        ),
        ignore_failed_deps=True,
    )

    execute_task(
        "modal_cleanup_stop_nonstopped_apps",
        validator=lambda: True,
        runner=lambda: (
            (
                lambda apps: (
                    _stop_apps(apps),
                    ("requested modal stop on non-stopped apps", {"stopped_count": len(apps)}),
                )[-1]
            )(_list_nonstopped_apps())
        ),
        ignore_failed_deps=True,
    )

    def _verify_clean_runner():
        rounds = 5
        for _idx in range(rounds):
            apps = _list_nonstopped_apps()
            if not apps:
                advisor_path = reports_dir / "orig_vggt_stepcurve_probe_advisor_latest.md"
                if advisor_path.is_file():
                    text = advisor_path.read_text(encoding="utf-8")
                    if "NO_NONSTOPPED_APPS" not in text:
                        advisor_path.write_text(text.rstrip() + "\n- NO_NONSTOPPED_APPS\n", encoding="utf-8")
                return "modal cleanup verified clean", {"nonstopped_apps": 0}
            time.sleep(30)
        raise RuntimeError("non-stopped modal apps still present after verification rounds")

    execute_task(
        "modal_cleanup_verify_clean",
        validator=lambda: True,
        runner=_verify_clean_runner,
        ignore_failed_deps=True,
    )

    final_status = "completed" if not failures else "failed"
    state.finalize(final_status, "all tasks reached terminal state")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
