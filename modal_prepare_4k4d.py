from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import modal


APP_NAME = os.environ.get("VGGT_MODAL_APP_NAME_4K4D_PREP", "vggt-prepare-4k4d")
DATA_VOLUME_NAME = os.environ.get("VGGT_DATA_VOL", "vggt-zju-data")
REMOTE_DATA_DIR = PurePosixPath(os.environ.get("VGGT_MNT_DATA", "/mnt/data"))

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
verify_image = modal.Image.debian_slim(python_version="3.11")


def _normalize_remote_subdir(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subdir.")
    return cleaned


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Local bridge dir not found: {local_dir}")
    remote_subdir = _normalize_remote_subdir(remote_subdir)
    print(f"[modal-4k4d] upload: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}")
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


@app.function(
    image=verify_image,
    volumes={REMOTE_DATA_DIR.as_posix(): data_volume},
    timeout=60 * 60,
)
def verify_remote_tree(remote_subdir: str, max_items: int = 80) -> None:
    root = Path(str(REMOTE_DATA_DIR / _normalize_remote_subdir(remote_subdir)))
    print(f"[modal-4k4d] remote tree root: {root}")
    if not root.exists():
        raise FileNotFoundError(f"Remote bridge root not found: {root}")
    count = 0
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            print(f"[remote] [d] {rel}/")
        else:
            print(f"[remote] [f] {rel} ({path.stat().st_size} bytes)")
        count += 1
        if count >= max_items:
            print(f"[remote] ... truncated after {max_items} items")
            break
    try:
        data_volume.commit()
        print("[modal-4k4d] committed data volume")
    except Exception as exc:
        print(f"[modal-4k4d] commit skipped/failed: {exc}")


@app.local_entrypoint()
def main(
    local_bridge_root: str,
    remote_root: str = "4k4d_bridge",
    verify: bool = True,
) -> None:
    local_dir = Path(local_bridge_root).expanduser().resolve()
    remote_subdir = f"{_normalize_remote_subdir(remote_root)}/{local_dir.name}"
    uploaded = _upload_dir(local_dir, remote_subdir)
    if verify:
        verify_remote_tree.remote(uploaded)
    else:
        print(f"[modal-4k4d] uploaded without verification: {DATA_VOLUME_NAME}:{uploaded}")
