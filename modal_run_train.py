# modal_run_train.py
from __future__ import annotations

import os
import glob
import subprocess
from pathlib import Path

import modal

APP_NAME = "vggt-viewdecoder-train"

vol_data = modal.Volume.from_name("vggt-zju-data", create_if_missing=True)
vol_out = modal.Volume.from_name("vggt-out", create_if_missing=True)

ROOT = Path(__file__).resolve().parent  # 本地 F:\vggt

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .workdir("/workspace/vggt")  # ✅ 放在 add_local_dir 之前
    .add_local_dir(              # ✅ 必须最后
        str(ROOT),
        remote_path="/workspace/vggt",
        ignore=[
            "ckpt", "runs", "infer_vis", "out_vis",
            "debug_vis", "debug_viewdec", "debug_viewdec_ablation",
            "__pycache__", ".git",
            "model.pt",
        ],
    )
)


app = modal.App(APP_NAME, image=image)


def _cat_parts_to_file(parts_glob: str, out_path: str):
    parts = sorted(glob.glob(parts_glob))
    if not parts:
        raise RuntimeError(f"No parts found for glob: {parts_glob}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 用 bash 的 cat 拼回去
    subprocess.run(
        ["bash", "-lc", f"cat {parts_glob} > {out_path}"], check=True)


def _ensure_model_pt():
    dst = "/workspace/vggt/model.pt"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[ok] model.pt already exists: {dst}")
        return
    print("[prep] reconstructing model.pt from /mnt/out/weights ...")
    _cat_parts_to_file("/mnt/out/weights/model.pt.part*", dst)
    print("[ok] reconstructed model.pt")


def _ensure_coreview390():
    target_dir = "/tmp/zju_mocap/CoreView_390"
    if os.path.isdir(target_dir):
        print(f"[ok] CoreView_390 already extracted: {target_dir}")
        return "/tmp/zju_mocap"

    os.makedirs("/tmp/zju_mocap", exist_ok=True)
    tar_path = "/tmp/CoreView_390.tar"

    print("[prep] reconstructing CoreView_390.tar from /mnt/zju/archives ...")
    _cat_parts_to_file("/mnt/zju/archives/CoreView_390.tar.part*", tar_path)

    print("[prep] extracting CoreView_390.tar to /tmp/zju_mocap ...")
    subprocess.run(
        ["bash", "-lc", f"tar -xf {tar_path} -C /tmp/zju_mocap"], check=True)

    if not os.path.isdir(target_dir):
        raise RuntimeError(f"Extraction failed, not found: {target_dir}")

    print("[ok] extracted CoreView_390")
    return "/tmp/zju_mocap"


@app.function(
    gpu=["H100", "A100"],
    timeout=60 * 60 * 24,
    volumes={
        "/mnt/zju": vol_data,  # 数据卷
        "/mnt/out": vol_out,   # 输出卷（也放权重分片）
    },
)
def train_one(
    seq_names: str = "CoreView_390",
    split: str = "train",
    device: str = "cuda",
):
    # 1) 权重重组
    _ensure_model_pt()

    # 2) 数据重组 + 解包（到本地临时盘 /tmp 更快）
    zju_root = _ensure_coreview390()

    out_root = "/mnt/out/vggt"
    os.makedirs(out_root, exist_ok=True)

    cmd = [
        "python", "train_view_decoder_ablation.py",
        "--zju_root", zju_root,
        "--seq_names", seq_names,
        # 注意：是 log_dir，不是 out_dir
        "--log_dir", f"{out_root}/runs/{seq_names}",
        "--ckpt_dir", f"{out_root}/ckpt/{seq_names}",
        # 可选：为了复现实验的固定划分
        # "--split_seed", "0",
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(seq_names: str = "CoreView_390", split: str = "train"):
    train_one.remote(seq_names=seq_names, split=split)
