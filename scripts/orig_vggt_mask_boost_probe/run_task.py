from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.orig_vggt_mask_boost_probe.common import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_EXTEND_STEPS,
    DEFAULT_FRAME_ID,
    DEFAULT_FG_MASK_SOURCE,
    DEFAULT_FG_SUPERVISION_REGION_MODE,
    DEFAULT_MASK_ERODE_PX,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MODAL_MAX_RETRIES,
    DEFAULT_MODAL_OUT_VOLUME,
    DEFAULT_MODAL_RETRY_SLEEP_SEC,
    DEFAULT_MODAL_SCRIPT,
    DEFAULT_NATIVE_STEPCURVE_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_PRETRAINED_CKPT,
    DEFAULT_PROFILES,
    DEFAULT_REMOTE_OUT_ROOT,
    DEFAULT_REMOTE_ZJU_ROOT,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SEED,
    DEFAULT_SEQ_NAME,
    DEFAULT_SHORT_STEPS,
    DEFAULT_TGT_CAMERA,
    DEFAULT_USE_FG_MASK,
    DEFAULT_ZJU_ROOT,
    alpha_root,
    alpha_step_compare_dir,
    alpha_step_train_dir,
    alpha_tag,
    alpha_to_fg_boost,
    build_task_specs,
    load_json,
    manifest_path,
    parse_alphas,
    parse_on_off,
    parse_profiles,
    parse_steps,
    profile_metadata,
    profile_root,
    profile_summary_json,
    profile_summary_md,
    profile_tag,
    step_tag,
    write_json,
    write_text,
)
from scripts.orig_vggt_mask_boost_probe.summarize_runs import (  # noqa: E402
    _best_short_row,
    _load_stage as _load_summary_stage,
    _stage_complete as _summary_stage_complete,
    _stage_vs_baseline,
)
from scripts.orig_vggt_stepcurve_probe.common import remote_geom_subdir  # noqa: E402
from scripts.orig_vggt_stepcurve_probe.run_task import (  # noqa: E402
    RootTaskState,
    _extract_tagged_line,
    _latest_raw_compare_run_dir,
    _list_nonstopped_apps,
    _load_profile_manifest,
    _measure_support,
    _modal_run_script,
    _modal_run_train_script,
    _prefix_ready,
    _quote_args,
    _run_capture,
    _run_python,
    _score_ghost,
    _stage_ready,
    _stop_apps,
    _support_ready,
    _sync_and_validate_train,
    _sync_compare_stage,
    _sync_prefix_trace,
    _validate_train_logs,
    _write_profile_manifest,
)


def _collect_native_compare_dirs(native_root: Path, profile: str) -> list[Path]:
    root = native_root / profile
    if not root.is_dir():
        return []
    out: list[Path] = []
    for step_dir in sorted(root.glob("step*/compare")):
        if step_dir.is_dir():
            out.append(step_dir)
    return out


def _native_baseline_stage(native_root: Path, profile: str) -> dict:
    baseline_dir = native_root / profile / "step0000" / "compare"
    if not _summary_stage_complete(baseline_dir):
        raise RuntimeError(f"native baseline compare incomplete: {baseline_dir}")
    stage = _load_summary_stage(baseline_dir)
    stage["step"] = 0
    return stage


def _alpha_short_rows(out_root: Path, profile: str, alpha: int, short_steps: list[int], baseline: dict) -> list[dict]:
    rows: list[dict] = []
    for step in short_steps:
        stage_dir = alpha_step_compare_dir(out_root, profile, alpha, step)
        if not _summary_stage_complete(stage_dir):
            continue
        row = _load_summary_stage(stage_dir)
        row["step"] = int(step)
        row["alpha"] = int(alpha)
        row["fg_supervision_boost"] = float(alpha_to_fg_boost(alpha))
        row["vs_baseline"] = _stage_vs_baseline(baseline, row)
        rows.append(row)
    return rows


def _write_alpha_short_summary(
    out_root: Path,
    profile: str,
    alpha: int,
    short_steps: list[int],
    baseline: dict,
) -> dict:
    rows = _alpha_short_rows(out_root, profile, alpha, short_steps, baseline)
    best_row = _best_short_row(rows)
    payload = {
        "profile": str(profile),
        "alpha": int(alpha),
        "fg_supervision_boost": float(alpha_to_fg_boost(alpha)),
        "available_steps": [step_tag(int(row["step"])) for row in rows],
        "best_short_step": step_tag(best_row["step"]) if best_row else "",
        "best_short_score": _safe_float(best_row.get("vs_baseline", {}).get("selection_score")) if best_row else float("nan"),
        "best_short_subject_psnr": _safe_float(best_row.get("subject_psnr")) if best_row else float("nan"),
        "best_short_subject_l1": _safe_float(best_row.get("subject_l1")) if best_row else float("nan"),
        "rows": rows,
    }
    out_dir = alpha_root(out_root, profile, alpha)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "short_summary.json", payload)
    lines = [
        f"# Mask Boost Short Sweep: {profile} {alpha_tag(alpha)}",
        "",
        f"- fg_supervision_boost = `{payload['fg_supervision_boost']}`",
        f"- best_short_step = `{payload['best_short_step'] or 'none'}`",
        f"- best_short_score = `{payload['best_short_score']}`",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{step_tag(row['step'])}` score=`{_safe_float(row['vs_baseline'].get('selection_score'))}` "
            f"ghost=`{_safe_float(row.get('ghost_visual_score'))}` subj_psnr=`{_safe_float(row.get('subject_psnr'))}` "
            f"subj_l1=`{_safe_float(row.get('subject_l1'))}`"
        )
    write_text(out_dir / "short_summary.md", "\n".join(lines) + "\n")
    return payload


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _select_winner_alpha(out_root: Path, profile: str, alphas: list[int], short_steps: list[int], baseline: dict) -> dict:
    candidates: list[dict] = []
    for alpha in alphas:
        summary_path = alpha_root(out_root, profile, alpha) / "short_summary.json"
        if summary_path.is_file():
            payload = load_json(summary_path)
        else:
            payload = _write_alpha_short_summary(out_root, profile, alpha, short_steps, baseline)
        if str(payload.get("best_short_step", "")).strip():
            candidates.append(payload)
    if not candidates:
        raise RuntimeError(f"no valid short sweep candidates for {profile}")
    winner = max(
        candidates,
        key=lambda row: (
            _safe_float(row.get("best_short_score")),
            _safe_float(row.get("best_short_subject_psnr")),
            -_safe_float(row.get("best_short_subject_l1")),
        ),
    )
    winner_alpha = int(winner["alpha"])
    manifest = _load_profile_manifest(manifest_path(out_root, profile))
    manifest["winner_alpha"] = winner_alpha
    manifest["winner_fg_supervision_boost"] = float(alpha_to_fg_boost(winner_alpha))
    manifest["winner_best_short_step"] = str(winner.get("best_short_step", ""))
    manifest["winner_selection_score"] = _safe_float(winner.get("best_short_score"))
    manifest["alpha_short_candidates"] = [
        {
            "alpha": int(item["alpha"]),
            "best_short_step": str(item.get("best_short_step", "")),
            "best_short_score": _safe_float(item.get("best_short_score")),
        }
        for item in candidates
    ]
    _write_profile_manifest(manifest_path(out_root, profile), manifest)
    return {
        "winner_alpha": winner_alpha,
        "winner_fg_supervision_boost": float(alpha_to_fg_boost(winner_alpha)),
        "winner_best_short_step": str(winner.get("best_short_step", "")),
        "winner_selection_score": _safe_float(winner.get("best_short_score")),
    }


def _winner_alpha_from_manifest(out_root: Path, profile: str) -> int:
    manifest = _load_profile_manifest(manifest_path(out_root, profile))
    alpha = manifest.get("winner_alpha")
    if alpha is None:
        raise RuntimeError(f"winner_alpha missing in manifest for {profile}")
    return int(alpha)


def _train_args(
    *,
    remote_zju_root: str,
    seq_name: str,
    train_cameras: list[str],
    pretrained_ckpt: str,
    step: int,
    max_frames: int,
    seed: int,
    geom_subdir: str,
    remote_train_run_root: str,
    alpha: int,
) -> list[str]:
    return [
        "--zju_root",
        str(remote_zju_root),
        "--seq_names",
        str(seq_name),
        "--cam_names",
        ",".join(train_cameras),
        "--pretrained_ckpt",
        str(pretrained_ckpt),
        "--resume_ckpt=",
        "--epochs",
        str(int(step)),
        "--max_frames",
        str(int(max_frames)),
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
        str(int(seed)),
        "--geom_subdir",
        str(geom_subdir),
        "--use_fg_mask",
        str(DEFAULT_USE_FG_MASK),
        "--fg_mask_source",
        str(DEFAULT_FG_MASK_SOURCE),
        "--fg_supervision_boost",
        str(float(alpha_to_fg_boost(alpha))),
        "--fg_supervision_region_mode",
        str(DEFAULT_FG_SUPERVISION_REGION_MODE),
        "--fg_supervision_region_erode_px",
        str(int(DEFAULT_MASK_ERODE_PX)),
        "--log_dir",
        f"{remote_train_run_root}/logs",
        "--ckpt_dir",
        f"{remote_train_run_root}/ckpt",
    ]


def _refresh_native_profile_metrics(native_root: Path, profile: str, local_zju_root: Path) -> dict:
    compare_dirs = _collect_native_compare_dirs(native_root, profile)
    if not compare_dirs:
        raise RuntimeError(f"no native compare dirs found for {profile}")
    manifest_file = native_root / profile / "profile_manifest.json"
    native_manifest = _load_profile_manifest(manifest_file)
    baseline_dir = native_root / profile / "step0000" / "compare"
    if baseline_dir not in compare_dirs:
        raise RuntimeError(f"native baseline compare missing for {profile}: {baseline_dir}")
    threshold = native_manifest.get("support_threshold_abs")
    baseline_metrics = _measure_support(
        baseline_dir,
        local_zju_root,
        str(threshold if threshold is not None else "auto"),
    )
    threshold = float(baseline_metrics["support_threshold_abs"])
    for compare_dir in compare_dirs:
        if compare_dir == baseline_dir:
            continue
        _measure_support(compare_dir, local_zju_root, str(threshold))
    _write_profile_manifest(
        manifest_file,
        {
            **native_manifest,
            "profile": profile,
            "support_threshold_abs": float(threshold),
            "mask_source_resolved": str(baseline_metrics.get("mask_source_resolved", "")),
        },
    )
    return {
        "profile": profile,
        "support_threshold_abs": float(threshold),
        "compare_count": len(compare_dirs),
    }


def _run_task_validator_loop(state: RootTaskState, failures: list[str], validator_wait_rounds: int, validator_wait_sleep_sec: int):
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
                    task["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    state.payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    state.save()
                validator_ok = _validator_ok(validator)
                if not validator_ok:
                    for _ in range(int(validator_wait_rounds)):
                        time.sleep(int(validator_wait_sleep_sec))
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
                state._task(name).setdefault("errors", []).append(
                    {"at": datetime.now().isoformat(timespec="seconds"), "message": msg}
                )
                state._task(name)["status"] = "pending"
                state._task(name)["updated_at"] = datetime.now().isoformat(timespec="seconds")
                state.save()
                time.sleep(5)
        return False

    return execute_task


def main() -> None:
    ap = argparse.ArgumentParser("orig_vggt_mask_boost_probe")
    ap.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    ap.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    ap.add_argument("--short_steps", default=",".join(str(x) for x in DEFAULT_SHORT_STEPS))
    ap.add_argument("--extend_steps", default=",".join(str(x) for x in DEFAULT_EXTEND_STEPS))
    ap.add_argument("--seq_name", default=str(DEFAULT_SEQ_NAME))
    ap.add_argument("--frame_id", type=int, default=int(DEFAULT_FRAME_ID))
    ap.add_argument("--tgt_camera", default=str(DEFAULT_TGT_CAMERA))
    ap.add_argument("--pretrained_ckpt", default=str(DEFAULT_PRETRAINED_CKPT))
    ap.add_argument("--max_frames", type=int, default=int(DEFAULT_MAX_FRAMES))
    ap.add_argument("--seed", type=int, default=int(DEFAULT_SEED))
    ap.add_argument("--strict_deterministic", default="on")
    ap.add_argument("--gpu_spec_precompute", default="A100-80GB")
    ap.add_argument("--resume_task_state", default="on")
    ap.add_argument("--continue_on_profile_error", default="on")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--reports_dir", default=str(DEFAULT_REPORTS_DIR))
    ap.add_argument("--native_stepcurve_root", default=str(DEFAULT_NATIVE_STEPCURVE_ROOT))
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
    alphas = parse_alphas(args.alphas)
    short_steps = parse_steps(args.short_steps, DEFAULT_SHORT_STEPS)
    extend_steps = parse_steps(args.extend_steps, DEFAULT_EXTEND_STEPS)

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (REPO_ROOT / reports_dir).resolve()
    native_root = Path(args.native_stepcurve_root)
    if not native_root.is_absolute():
        native_root = (REPO_ROOT / native_root).resolve()
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
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "profiles": profiles,
        "alphas": alphas,
        "short_steps": short_steps,
        "extend_steps": extend_steps,
        "tasks": build_task_specs(profiles, alphas, short_steps, extend_steps),
    }
    state = RootTaskState(
        out_root / "task_state_latest.json",
        out_root / "task_state_latest.md",
        payload,
        resume=parse_on_off(args.resume_task_state, default=True),
    )
    continue_on_profile_error = parse_on_off(args.continue_on_profile_error, default=True)
    strict_deterministic_flag = parse_on_off(args.strict_deterministic, default=True)
    failures: list[str] = []
    execute_task = _run_task_validator_loop(
        state,
        failures,
        validator_wait_rounds=max(1, int(args.modal_retry_sleep_sec)),
        validator_wait_sleep_sec=max(2, int(args.modal_retry_sleep_sec)),
    )

    def remote_alpha_compare_root(profile: str, alpha: int, step: int) -> str:
        return f"{str(args.remote_out_root).rstrip('/')}/{profile}/{alpha_tag(alpha)}/{step_tag(step)}/compare"

    def remote_alpha_train_root(profile: str, alpha: int, step: int) -> str:
        return (
            f"{str(args.remote_out_root).rstrip('/')}/{profile}/{alpha_tag(alpha)}/{step_tag(step)}/train_runs/"
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def remote_alpha_prefix_trace(profile: str, alpha: int, step: int) -> str:
        return f"{str(args.remote_out_root).rstrip('/')}/{profile}/{alpha_tag(alpha)}/{step_tag(step)}/train/train_prefix_trace.json"

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
                        "native_stepcurve_root": str(native_root),
                        "profiles_checked": profiles,
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

    execute_task(
        "native_reference_refresh_support_metrics",
        validator=lambda: all(
            (native_root / profile / "profile_manifest.json").is_file() for profile in profiles
        ),
        runner=lambda: (
            (
                lambda refreshed: (
                    "native reference support metrics refreshed",
                    {"profiles_refreshed": refreshed},
                )
            )([_refresh_native_profile_metrics(native_root, profile, local_zju_root) for profile in profiles])
        ),
    )

    execute_task(
        "native_reference_refresh_summary",
        validator=lambda: (reports_dir / "orig_vggt_stepcurve_probe_summary_latest.json").is_file(),
        runner=lambda: (
            _run_python(
                REPO_ROOT / "scripts" / "orig_vggt_stepcurve_probe" / "summarize_runs.py",
                [
                    "--out_root",
                    str(native_root),
                    "--profiles",
                    ",".join(profiles),
                    "--step_horizons",
                    "1,2,4,8,16",
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
                label="native_reference_refresh_summary",
            ),
            ("native stepcurve summary refreshed", {"summary_json": str(reports_dir / "orig_vggt_stepcurve_probe_summary_latest.json")}),
        )[-1],
    )

    for profile in profiles:
        tag = profile_tag(profile)
        meta = profile_metadata(profile, args.tgt_camera)
        profile_dir = profile_root(out_root, profile)
        trend_json = profile_summary_json(out_root, profile)
        trend_md = profile_summary_md(out_root, profile)
        geom_subdir = f"{remote_geom_subdir(profile)}_maskboost"
        native_manifest = _load_profile_manifest(native_root / profile / "profile_manifest.json")
        baseline = _native_baseline_stage(native_root, profile)
        profile_manifest_file = manifest_path(out_root, profile)

        execute_task(
            f"profile_{tag}_prepare",
            validator=lambda p=profile_manifest_file: p.is_file(),
            runner=lambda profile=profile, meta=meta, geom_subdir=geom_subdir, native_manifest=native_manifest: (
                profile_dir.mkdir(parents=True, exist_ok=True),
                (profile_dir / "trend").mkdir(parents=True, exist_ok=True),
                _write_profile_manifest(
                    profile_manifest_file,
                    {
                        **_load_profile_manifest(profile_manifest_file),
                        "profile": profile,
                        "seq_name": args.seq_name,
                        "frame_id": int(args.frame_id),
                        "tgt_camera": str(args.tgt_camera),
                        "geom_subdir": str(geom_subdir),
                        "train_cameras": meta["train_cameras"],
                        "alphas": [int(x) for x in alphas],
                        "short_steps": [int(x) for x in short_steps],
                        "extend_steps": [int(x) for x in extend_steps],
                        "native_stepcurve_root": str(native_root),
                        "native_support_threshold_abs": native_manifest.get("support_threshold_abs"),
                    },
                ),
                ("profile manifest ready", {"profile_root": str(profile_dir), "geom_subdir": str(geom_subdir)}),
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

        for alpha in alphas:
            atag = alpha_tag(alpha)
            for step in short_steps:
                stag = step_tag(step)
                local_train_root = alpha_step_train_dir(out_root, profile, alpha, step)
                local_compare_dir = alpha_step_compare_dir(out_root, profile, alpha, step)
                remote_train_root_default = remote_alpha_train_root(profile, alpha, step)
                remote_out_last = f"{remote_train_root_default}/ckpt/model_ft_zju_last.pt"
                task_prefix = f"profile_{tag}_{atag}_{stag}"

                execute_task(
                    f"{task_prefix}_train",
                    validator=lambda name=f"{task_prefix}_train", default_root=remote_train_root_default: bool(
                        str(state.outputs(name).get("remote_train_run_root", default_root)).strip()
                    ),
                    runner=lambda profile=profile, alpha=alpha, step=step, default_root=remote_train_root_default: (
                        _modal_run_train_script(
                            modal_script=args.modal_script,
                            code_dir=code_dir,
                            gpu_spec_train=args.gpu_spec_precompute,
                            remote_zju_root=args.remote_zju_root,
                            seq_name=args.seq_name,
                            script_path="scripts/orig_vggt_stepcurve_probe/run_finetune_strict.py",
                            forwarded_args=_train_args(
                                remote_zju_root=str(args.remote_zju_root),
                                seq_name=str(args.seq_name),
                                train_cameras=meta["train_cameras"],
                                pretrained_ckpt=str(args.pretrained_ckpt),
                                step=int(step),
                                max_frames=int(args.max_frames),
                                seed=int(args.seed),
                                geom_subdir=str(geom_subdir),
                                remote_train_run_root=str(default_root),
                                alpha=int(alpha),
                            ),
                            label=f"{profile}:{atag}:{stag}_train",
                            modal_max_retries=int(args.modal_max_retries),
                            modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                            extra_env={"VGGT_STRICT_DETERMINISTIC": "1" if strict_deterministic_flag else "0"},
                        ),
                        (
                            "remote train finished",
                            {
                                "remote_train_run_root": str(default_root),
                                "remote_out_last_ckpt": str(remote_out_last),
                                "alpha": int(alpha),
                                "step": int(step),
                            },
                        ),
                    )[-1],
                )

                execute_task(
                    f"{task_prefix}_sync_train",
                    validator=lambda tr=local_train_root, step=step: tr.is_dir() and bool(_validate_train_logs(tr, step)),
                    runner=lambda tr=local_train_root, name=f"{task_prefix}_train", step=step: _sync_and_validate_train(
                        volume=str(args.modal_out_volume),
                        remote_train_run_root=str(state.outputs(name).get("remote_train_run_root", remote_train_root_default)),
                        local_train_root=tr,
                        horizon=int(step),
                    ),
                )

                execute_task(
                    f"{task_prefix}_prefix_audit",
                    validator=lambda tr=local_train_root: _prefix_ready(tr),
                    runner=lambda tr=local_train_root, profile=profile, alpha=alpha, step=step: (
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
                                            str(geom_subdir),
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
                                            str(remote_trace_path),
                                        ]
                                    ),
                                    label=f"{profile}:{atag}:{stag}_prefix_audit",
                                    modal_max_retries=int(args.modal_max_retries),
                                    modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                                ),
                                _sync_prefix_trace(
                                    str(args.modal_out_volume),
                                    str(remote_trace_path),
                                    tr / "train_prefix_trace.json",
                                ),
                                ("prefix audit synced", {"prefix_trace_json": str(tr / 'train_prefix_trace.json')}),
                            )[-1]
                        )(remote_alpha_prefix_trace(profile, alpha, step))
                    ),
                )

                execute_task(
                    f"{task_prefix}_compare",
                    validator=lambda name=f"{task_prefix}_compare": bool(state.outputs(name).get("remote_run_dir")),
                    runner=lambda profile=profile, alpha=alpha, step=step, train_name=f"{task_prefix}_train", sync_name=f"{task_prefix}_sync_train": (
                        (
                            lambda lines: (
                                "remote compare finished",
                                {
                                    "remote_run_dir": _extract_tagged_line(lines, "RUN_DIR:")
                                    or _latest_raw_compare_run_dir(
                                        remote_stage_root=remote_alpha_compare_root(profile, alpha, step),
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
                                        str(remote_alpha_compare_root(profile, alpha, step)),
                                    ]
                                ),
                                label=f"{profile}:{atag}:{stag}_compare",
                                modal_max_retries=int(args.modal_max_retries),
                                modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                            )
                        )
                    ),
                )

                execute_task(
                    f"{task_prefix}_sync_compare",
                    validator=lambda d=local_compare_dir: _stage_ready(d),
                    runner=lambda d=local_compare_dir, name=f"{task_prefix}_compare": (
                        _sync_compare_stage(
                            volume=str(args.modal_out_volume),
                            remote_run_dir=str(state.outputs(name).get("remote_run_dir", "")),
                            local_stage_dir=d,
                        ),
                        ("compare synced", {"local_compare_dir": str(d)}),
                    )[-1],
                )

                execute_task(
                    f"{task_prefix}_score_ghost",
                    validator=lambda d=local_compare_dir: (d / "ghost_score_rows.csv").is_file() and (d / "ghost_score.json").is_file(),
                    runner=lambda d=local_compare_dir, profile=profile: (
                        _score_ghost(d, profile),
                        ("ghost scored", {"ghost_json": str(d / 'ghost_score.json')}),
                    )[-1],
                )

                execute_task(
                    f"{task_prefix}_measure_support",
                    validator=lambda d=local_compare_dir: _support_ready(d),
                    runner=lambda d=local_compare_dir, profile=profile: (
                        (
                            lambda manifest, metrics: (
                                "point-support measured",
                                {
                                    "point_support_json": str(d / "point_support_metrics.json"),
                                    "support_threshold_abs": float(metrics["support_threshold_abs"]),
                                    "alpha": int(alpha),
                                    "step": int(step),
                                },
                            )
                        )(
                            _load_profile_manifest(manifest_path(out_root, profile)),
                            _measure_support(
                                d,
                                local_zju_root,
                                str(
                                    _load_profile_manifest(manifest_path(out_root, profile)).get(
                                        "native_support_threshold_abs",
                                        "auto",
                                    )
                                ),
                            ),
                        )
                    ),
                )

            execute_task(
                f"profile_{tag}_{atag}_short_summary",
                validator=lambda aroot=alpha_root(out_root, profile, alpha): (aroot / "short_summary.json").is_file(),
                runner=lambda profile=profile, alpha=alpha: (
                    (
                        lambda payload: (
                            "alpha short summary refreshed",
                            {
                                "alpha": int(alpha),
                                "best_short_step": str(payload.get("best_short_step", "")),
                                "best_short_score": _safe_float(payload.get("best_short_score")),
                            },
                        )
                    )(_write_alpha_short_summary(out_root, profile, alpha, short_steps, baseline))
                ),
                ignore_failed_deps=True,
            )

        execute_task(
            f"profile_{tag}_select_winner",
            validator=lambda p=profile_manifest_file: p.is_file() and ("winner_alpha" in _load_profile_manifest(p)),
            runner=lambda profile=profile: (
                (
                    lambda result: (
                        "winner alpha selected",
                        result,
                    )
                )(_select_winner_alpha(out_root, profile, alphas, short_steps, baseline))
            ),
            ignore_failed_deps=True,
        )

        for step in extend_steps:
            stag = step_tag(step)
            task_prefix = f"profile_{tag}_winner_{stag}"

            execute_task(
                f"{task_prefix}_train",
                validator=lambda name=f"{task_prefix}_train": bool(state.outputs(name).get("remote_train_run_root")),
                runner=lambda profile=profile, step=step: (
                    (
                        lambda winner_alpha, remote_train_root_default: (
                            _modal_run_train_script(
                                modal_script=args.modal_script,
                                code_dir=code_dir,
                                gpu_spec_train=args.gpu_spec_precompute,
                                remote_zju_root=args.remote_zju_root,
                                seq_name=args.seq_name,
                                script_path="scripts/orig_vggt_stepcurve_probe/run_finetune_strict.py",
                                forwarded_args=_train_args(
                                    remote_zju_root=str(args.remote_zju_root),
                                    seq_name=str(args.seq_name),
                                    train_cameras=meta["train_cameras"],
                                    pretrained_ckpt=str(args.pretrained_ckpt),
                                    step=int(step),
                                    max_frames=int(args.max_frames),
                                    seed=int(args.seed),
                                    geom_subdir=str(geom_subdir),
                                    remote_train_run_root=str(remote_train_root_default),
                                    alpha=int(winner_alpha),
                                ),
                                label=f"{profile}:winner:{step_tag(step)}_train",
                                modal_max_retries=int(args.modal_max_retries),
                                modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                                extra_env={"VGGT_STRICT_DETERMINISTIC": "1" if strict_deterministic_flag else "0"},
                            ),
                            (
                                "winner extend train finished",
                                {
                                    "winner_alpha": int(winner_alpha),
                                    "remote_train_run_root": str(remote_train_root_default),
                                    "remote_out_last_ckpt": f"{remote_train_root_default}/ckpt/model_ft_zju_last.pt",
                                },
                            ),
                        )[-1]
                    )(
                        _winner_alpha_from_manifest(out_root, profile),
                        remote_alpha_train_root(profile, _winner_alpha_from_manifest(out_root, profile), step),
                    )
                ),
            )

            execute_task(
                f"{task_prefix}_sync_train",
                validator=lambda profile=profile, step=step: (
                    lambda winner_alpha: alpha_step_train_dir(out_root, profile, winner_alpha, step).is_dir()
                    and bool(_validate_train_logs(alpha_step_train_dir(out_root, profile, winner_alpha, step), step))
                )(_winner_alpha_from_manifest(out_root, profile)),
                runner=lambda profile=profile, step=step, name=f"{task_prefix}_train": (
                    (
                        lambda winner_alpha, tr: _sync_and_validate_train(
                            volume=str(args.modal_out_volume),
                            remote_train_run_root=str(state.outputs(name).get("remote_train_run_root", "")),
                            local_train_root=tr,
                            horizon=int(step),
                        )
                    )(
                        _winner_alpha_from_manifest(out_root, profile),
                        alpha_step_train_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step),
                    )
                ),
            )

            execute_task(
                f"{task_prefix}_prefix_audit",
                validator=lambda profile=profile, step=step: _prefix_ready(
                    alpha_step_train_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step)
                ),
                runner=lambda profile=profile, step=step: (
                    (
                        lambda winner_alpha, tr, remote_trace_path: (
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
                                        str(geom_subdir),
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
                                        str(remote_trace_path),
                                    ]
                                ),
                                label=f"{profile}:winner:{step_tag(step)}_prefix_audit",
                                modal_max_retries=int(args.modal_max_retries),
                                modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                            ),
                            _sync_prefix_trace(str(args.modal_out_volume), str(remote_trace_path), tr / "train_prefix_trace.json"),
                            ("winner prefix audit synced", {"prefix_trace_json": str(tr / 'train_prefix_trace.json')}),
                        )[-1]
                    )(
                        _winner_alpha_from_manifest(out_root, profile),
                        alpha_step_train_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step),
                        remote_alpha_prefix_trace(profile, _winner_alpha_from_manifest(out_root, profile), step),
                    )
                ),
            )

            execute_task(
                f"{task_prefix}_compare",
                validator=lambda name=f"{task_prefix}_compare": bool(state.outputs(name).get("remote_run_dir")),
                runner=lambda profile=profile, step=step, train_name=f"{task_prefix}_train", sync_name=f"{task_prefix}_sync_train": (
                    (
                        lambda winner_alpha, lines: (
                            "winner extend compare finished",
                            {
                                "winner_alpha": int(winner_alpha),
                                "remote_run_dir": _extract_tagged_line(lines, "RUN_DIR:")
                                or _latest_raw_compare_run_dir(
                                    remote_stage_root=remote_alpha_compare_root(profile, winner_alpha, step),
                                    seq_name=args.seq_name,
                                    frame_id=int(args.frame_id),
                                    tgt_camera=args.tgt_camera,
                                    volume=str(args.modal_out_volume),
                                ),
                            },
                        )
                    )(
                        _winner_alpha_from_manifest(out_root, profile),
                        _modal_run_script(
                            modal_script=args.modal_script,
                            code_dir=code_dir,
                            gpu_spec_precompute=args.gpu_spec_precompute,
                            remote_zju_root=args.remote_zju_root,
                            seq_name=args.seq_name,
                            script_path="scripts/orig_vggt_viewcount/render_raw_compare.py",
                            ckpt_path=str(
                                state.outputs(sync_name).get("remote_out_last_ckpt")
                                or state.outputs(train_name).get("remote_out_last_ckpt", "")
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
                                    str(remote_alpha_compare_root(profile, _winner_alpha_from_manifest(out_root, profile), step)),
                                ]
                            ),
                            label=f"{profile}:winner:{step_tag(step)}_compare",
                            modal_max_retries=int(args.modal_max_retries),
                            modal_retry_sleep_sec=int(args.modal_retry_sleep_sec),
                        ),
                    )
                ),
            )

            execute_task(
                f"{task_prefix}_sync_compare",
                validator=lambda profile=profile, step=step: _stage_ready(
                    alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step)
                ),
                runner=lambda profile=profile, step=step, name=f"{task_prefix}_compare": (
                    (
                        lambda winner_alpha, d: (
                            _sync_compare_stage(
                                volume=str(args.modal_out_volume),
                                remote_run_dir=str(state.outputs(name).get("remote_run_dir", "")),
                                local_stage_dir=d,
                            ),
                            ("winner compare synced", {"local_compare_dir": str(d), "winner_alpha": int(winner_alpha)}),
                        )[-1]
                    )(
                        _winner_alpha_from_manifest(out_root, profile),
                        alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step),
                    )
                ),
            )

            execute_task(
                f"{task_prefix}_score_ghost",
                validator=lambda profile=profile, step=step: (
                    lambda d: (d / "ghost_score_rows.csv").is_file() and (d / "ghost_score.json").is_file()
                )(alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step)),
                runner=lambda profile=profile, step=step: (
                    (
                        lambda d: (
                            _score_ghost(d, profile),
                            ("winner ghost scored", {"ghost_json": str(d / 'ghost_score.json')}),
                        )[-1]
                    )(alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step))
                ),
            )

            execute_task(
                f"{task_prefix}_measure_support",
                validator=lambda profile=profile, step=step: _support_ready(
                    alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step)
                ),
                runner=lambda profile=profile, step=step: (
                    (
                        lambda d, metrics: (
                            "winner point-support measured",
                            {
                                "winner_alpha": int(_winner_alpha_from_manifest(out_root, profile)),
                                "point_support_json": str(d / "point_support_metrics.json"),
                                "support_threshold_abs": float(metrics["support_threshold_abs"]),
                            },
                        )
                    )(
                        alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step),
                        _measure_support(
                            alpha_step_compare_dir(out_root, profile, _winner_alpha_from_manifest(out_root, profile), step),
                            local_zju_root,
                            str(_load_profile_manifest(manifest_path(out_root, profile)).get("native_support_threshold_abs", "auto")),
                        ),
                    )
                ),
            )

        execute_task(
            f"profile_{tag}_trend_summary",
            validator=lambda j=trend_json, m=trend_md: j.is_file() and m.is_file(),
            runner=lambda profile=profile: (
                _run_python(
                    REPO_ROOT / "scripts" / "orig_vggt_mask_boost_probe" / "summarize_runs.py",
                    [
                        "--out_root",
                        str(out_root),
                        "--native_stepcurve_root",
                        str(native_root),
                        "--profiles",
                        str(profile),
                        "--alphas",
                        ",".join(str(x) for x in alphas),
                        "--short_steps",
                        ",".join(str(x) for x in short_steps),
                        "--extend_steps",
                        ",".join(str(x) for x in extend_steps),
                        "--out_json",
                        str(reports_dir / f"_{profile}_mask_boost_summary.json"),
                        "--out_csv",
                        str(reports_dir / f"_{profile}_mask_boost_summary.csv"),
                        "--out_md",
                        str(reports_dir / f"_{profile}_mask_boost_summary.md"),
                        "--out_advisor_md",
                        str(reports_dir / f"_{profile}_mask_boost_advisor.md"),
                        "--out_point_grid",
                        str(reports_dir / f"_{profile}_mask_boost_point_grid.png"),
                        "--out_ghost_grid",
                        str(reports_dir / f"_{profile}_mask_boost_ghost_grid.png"),
                    ],
                    label=f"{profile}:mask_boost_trend_summary",
                ),
                ("profile trend summary refreshed", {"trend_summary_json": str(trend_json)}),
            )[-1],
            ignore_failed_deps=True,
        )

        if failures and not continue_on_profile_error:
            break

    def _ready_profiles_for_global() -> list[str]:
        ready: list[str] = []
        for profile in profiles:
            tag = profile_tag(profile)
            if state.status(f"profile_{tag}_trend_summary") == "completed":
                ready.append(profile)
        return ready

    execute_task(
        "global_summary_refresh",
        validator=lambda: (reports_dir / "orig_vggt_mask_boost_probe_summary_latest.json").is_file(),
        runner=lambda: (
            (
                lambda ready: (
                    (
                        _run_python(
                            REPO_ROOT / "scripts" / "orig_vggt_mask_boost_probe" / "summarize_runs.py",
                            [
                                "--out_root",
                                str(out_root),
                                "--native_stepcurve_root",
                                str(native_root),
                                "--profiles",
                                ",".join(ready),
                                "--alphas",
                                ",".join(str(x) for x in alphas),
                                "--short_steps",
                                ",".join(str(x) for x in short_steps),
                                "--extend_steps",
                                ",".join(str(x) for x in extend_steps),
                                "--out_json",
                                str(reports_dir / "orig_vggt_mask_boost_probe_summary_latest.json"),
                                "--out_csv",
                                str(reports_dir / "orig_vggt_mask_boost_probe_summary_latest.csv"),
                                "--out_md",
                                str(reports_dir / "orig_vggt_mask_boost_probe_summary_latest.md"),
                                "--out_advisor_md",
                                str(reports_dir / "orig_vggt_mask_boost_probe_advisor_latest.md"),
                                "--out_point_grid",
                                str(reports_dir / "orig_vggt_mask_boost_point_support_grid_latest.png"),
                                "--out_ghost_grid",
                                str(reports_dir / "orig_vggt_mask_boost_ghost_grid_latest.png"),
                            ],
                            label="mask_boost_global_summary",
                        )
                        if ready
                        else (
                            write_json(
                                reports_dir / "orig_vggt_mask_boost_probe_summary_latest.json",
                                {
                                    "profiles": [],
                                    "note": "no ready mask-boost profiles yet",
                                },
                            ),
                            write_text(reports_dir / "orig_vggt_mask_boost_probe_summary_latest.csv", ""),
                            write_text(
                                reports_dir / "orig_vggt_mask_boost_probe_summary_latest.md",
                                "# Native VGGT Mask-Boost Probe Summary\n\n- no ready profiles yet.\n",
                            ),
                            write_text(
                                reports_dir / "orig_vggt_mask_boost_probe_advisor_latest.md",
                                "# Native VGGT Mask-Boost Advisor Summary\n\n- no ready profiles yet.\n",
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
                advisor_path = reports_dir / "orig_vggt_mask_boost_probe_advisor_latest.md"
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
