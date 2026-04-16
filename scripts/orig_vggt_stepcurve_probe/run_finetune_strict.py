from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _parse_paths(argv: list[str]):
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    parsed, _unknown = ap.parse_known_args(argv)
    return parsed


def _extract_forwarded_args() -> list[str]:
    env_args = str(os.environ.get("STEPCURVE_FINETUNE_ARGS_JSON", "")).strip()
    if env_args:
        return list(json.loads(env_args))
    argv = list(sys.argv[1:])
    b64 = ""
    for idx, arg in enumerate(argv):
        text = str(arg).strip()
        if text.startswith("--stepcurve_forward_b64="):
            b64 = text.split("=", 1)[1].strip()
            break
        if text == "--stepcurve_forward_b64" and idx + 1 < len(argv):
            b64 = str(argv[idx + 1]).strip()
            break
    if b64:
        decoded = base64.b64decode(b64.encode("ascii")).decode("utf-8")
        return list(json.loads(decoded))
    return argv


def _safe_sync_path(path: Path) -> None:
    try:
        if path.is_file():
            with path.open("rb") as f:
                os.fsync(f.fileno())
        elif path.is_dir():
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except Exception as exc:
        print(f"[run_finetune_strict] sync skipped for {path}: {exc}", flush=True)


def _promote_tmp_ckpt(final_path: Path, *, timeout_sec: int = 90, interval_sec: int = 5) -> Path:
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    deadline = time.time() + max(0, int(timeout_sec))
    last_err = ""
    while time.time() <= deadline:
        if final_path.is_file():
            return final_path
        if tmp_path.is_file():
            try:
                _ = torch.load(str(tmp_path), map_location="cpu")
                os.replace(str(tmp_path), str(final_path))
                _safe_sync_path(final_path)
                _safe_sync_path(final_path.parent)
                print(f"[run_finetune_strict] promoted tmp checkpoint -> {final_path}", flush=True)
                return final_path
            except Exception as exc:
                last_err = str(exc)
        time.sleep(max(1, int(interval_sec)))
    raise RuntimeError(
        f"failed to materialize final checkpoint {final_path} from tmp; "
        f"tmp_exists={tmp_path.is_file()} last_err={last_err}"
    )


def _write_summary_if_needed(log_dir: Path, ckpt_dir: Path) -> Path:
    metrics_path = log_dir / "finetune_vggt_metrics.jsonl"
    summary_path = log_dir / "finetune_vggt_summary.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"metrics jsonl missing: {metrics_path}")
    rows = _load_jsonl(metrics_path)
    step_eval_rows = [row for row in rows if row.get("event") == "step_eval"]
    epoch_end_rows = [row for row in rows if row.get("event") == "epoch_end"]
    last_epoch = epoch_end_rows[-1] if epoch_end_rows else {}
    final_step = 0
    if step_eval_rows:
        final_step = max(int(row.get("step", 0)) for row in step_eval_rows)
    if final_step <= 0:
        final_step = sum(int(row.get("steps", 0)) for row in epoch_end_rows)
    payload = {
        "best_epoch": int(last_epoch.get("best_epoch", -1)),
        "best_loss": float(last_epoch.get("best_loss", float("nan"))),
        "final_epoch": int(last_epoch.get("epoch", -1)),
        "final_step": int(final_step),
        "out_best": str(ckpt_dir / "model_ft_zju.pt"),
        "out_last": str(ckpt_dir / "model_ft_zju_last.pt"),
        "epoch_end_steps_total": int(sum(int(row.get("steps", 0)) for row in epoch_end_rows)),
        "step_eval_count": int(len(step_eval_rows)),
        "summary_source": "run_finetune_strict_repair",
    }
    if summary_path.is_file():
        try:
            current = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        merged = dict(current)
        merged.update({k: v for k, v in payload.items() if v not in ("", None)})
        payload = merged
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _safe_sync_path(summary_path)
    _safe_sync_path(summary_path.parent)
    print(f"[run_finetune_strict] ensured summary: {summary_path}", flush=True)
    return summary_path


def _settle_outputs(log_dir: Path, ckpt_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model_ft_zju.pt"
    last_path = ckpt_dir / "model_ft_zju_last.pt"
    if not best_path.is_file():
        best_tmp = best_path.with_name(best_path.name + ".tmp")
        if best_tmp.is_file():
            _ = torch.load(str(best_tmp), map_location="cpu")
            os.replace(str(best_tmp), str(best_path))
            print(f"[run_finetune_strict] promoted tmp best checkpoint -> {best_path}", flush=True)
    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint missing after train: {best_path}")
    _promote_tmp_ckpt(last_path)
    _write_summary_if_needed(log_dir, ckpt_dir)
    _safe_sync_path(best_path)
    _safe_sync_path(last_path)
    _safe_sync_path(ckpt_dir)
    if hasattr(os, "sync"):
        try:
            os.sync()
        except Exception as exc:
            print(f"[run_finetune_strict] os.sync skipped: {exc}", flush=True)
    time.sleep(5)


def main() -> None:
    forwarded_args = _extract_forwarded_args()
    parsed = _parse_paths(forwarded_args)
    env = os.environ.copy()
    env["VGGT_STRICT_DETERMINISTIC"] = "1"
    cmd = [sys.executable, "-u", str(REPO_ROOT / "finetune_vggt_pseudo.py"), *forwarded_args]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    if int(proc.returncode) != 0:
        raise SystemExit(int(proc.returncode))
    _settle_outputs(Path(parsed.log_dir), Path(parsed.ckpt_dir))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
