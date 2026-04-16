import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import modal
import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reliable Modal volume transfer helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download-dir", help="Download files from a Modal volume directory")
    dl.add_argument("--volume", required=True)
    dl.add_argument("--remote-dir", required=True)
    dl.add_argument("--local-dir", required=True)
    dl.add_argument("--pattern", default="*.npz")
    dl.add_argument("--workers", type=int, default=2)
    dl.add_argument("--verify-npz", action="store_true")
    dl.add_argument("--force", action="store_true")

    ul = sub.add_parser("upload-dir", help="Upload a local directory to a Modal volume")
    ul.add_argument("--volume", required=True)
    ul.add_argument("--local-dir", required=True)
    ul.add_argument("--remote-dir", required=True)
    ul.add_argument("--pattern", default="*.npz")
    ul.add_argument("--force", action="store_true")

    return ap.parse_args()


def _volume(name: str) -> modal.Volume:
    return modal.Volume.from_name(name)


def _iter_remote_files(volume_name: str, remote_dir: str, pattern: str) -> list[str]:
    vol = _volume(volume_name)
    items = vol.listdir(remote_dir)
    out: list[str] = []
    for item in items:
        path = str(getattr(item, "path", "") or getattr(item, "filename", "") or "")
        type_raw = getattr(item, "type", "") or getattr(item, "Type", "")
        if not path:
            continue
        is_file = False
        if isinstance(type_raw, int):
            is_file = int(type_raw) == 1
        else:
            type_name = str(type_raw)
            is_file = type_name.lower() == "file" or "fileentrytype.file" in type_name.lower()
        if not is_file:
            continue
        leaf = Path(path).name
        if Path(leaf).match(pattern):
            if not path.startswith("/"):
                path = "/" + path.lstrip("/")
            out.append(path)
    out.sort()
    return out


def _verify_npz(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        _ = data.files
        if "pointmap" in data.files:
            _ = np.asarray(data["pointmap"], dtype=np.float32)
        if "depth" in data.files:
            _ = np.asarray(data["depth"], dtype=np.float32)
        if "depth_conf" in data.files:
            _ = np.asarray(data["depth_conf"], dtype=np.float32)
        if "prior_pointmap" in data.files:
            _ = np.asarray(data["prior_pointmap"], dtype=np.float32)
        if "prior_valid_mask" in data.files:
            _ = np.asarray(data["prior_valid_mask"], dtype=np.float32)


def _download_one(volume_name: str, remote_path: str, local_path: Path, verify_npz: bool, force: bool) -> str:
    if local_path.exists() and not force:
        if not verify_npz:
            return "skip"
        try:
            _verify_npz(local_path)
            return "skip"
        except Exception:
            pass
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    vol = _volume(volume_name)
    with tmp_path.open("wb") as f:
        vol.read_file_into_fileobj(remote_path, f)
    if verify_npz:
        _verify_npz(tmp_path)
    os.replace(tmp_path, local_path)
    return "ok"


def _download_dir(args: argparse.Namespace) -> int:
    remote_files = _iter_remote_files(args.volume, args.remote_dir, args.pattern)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _download_one,
                args.volume,
                remote_path,
                local_dir / Path(remote_path).name,
                bool(args.verify_npz),
                bool(args.force),
            ): remote_path
            for remote_path in remote_files
        }
        for fut in as_completed(futures):
            remote_path = futures[fut]
            result = fut.result()
            done += 1
            print(f"[modal-transfer] {done}/{len(remote_files)} {result} {remote_path}")
    return 0


def _iter_local_files(local_dir: Path, pattern: str) -> list[Path]:
    out = sorted([p for p in local_dir.glob(pattern) if p.is_file()])
    return out


def _upload_dir(args: argparse.Namespace) -> int:
    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.is_dir():
        raise RuntimeError(f"local dir not found: {local_dir}")
    files = _iter_local_files(local_dir, args.pattern)
    vol = _volume(args.volume)
    with vol.batch_upload(force=bool(args.force)) as batch:
        for file_path in files:
            remote_path = str(Path(args.remote_dir).as_posix().rstrip("/") + "/" + file_path.name)
            batch.put_file(str(file_path), remote_path)
            print(f"[modal-transfer] queued upload {file_path.name} -> {remote_path}")
    print(f"[modal-transfer] uploaded {len(files)} files to {args.remote_dir}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.cmd == "download-dir":
        return _download_dir(args)
    if args.cmd == "upload-dir":
        return _upload_dir(args)
    raise RuntimeError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
