# modal_run_train.py
from __future__ import annotations

import os
import glob
import time
import subprocess
from pathlib import Path

import modal

APP_NAME = "vggt-viewdecoder-train"

VOL_DATA = "vggt-zju-data"
VOL_OUT = "vggt-out"

vol_data = modal.Volume.from_name(VOL_DATA, create_if_missing=True)
vol_out = modal.Volume.from_name(VOL_OUT,  create_if_missing=True)

ROOT = Path(__file__).resolve().parent  # 你的本地 vggt 工程目录

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .workdir("/workspace/vggt")
    .add_local_dir(
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


def _ensure_seq_tar_extracted(seq_name: str) -> str:
    """
    期望卷里存在：/mnt/zju/archives/{seq_name}.tar.part*
    解包到：/tmp/zju_mocap/{seq_name}
    返回 zju_root = /tmp/zju_mocap
    """
    zju_root = "/tmp/zju_mocap"
    target_dir = f"{zju_root}/{seq_name}"
    if os.path.isdir(target_dir):
        print(f"[ok] {seq_name} already extracted: {target_dir}")
        return zju_root

    os.makedirs(zju_root, exist_ok=True)
    tar_path = f"/tmp/{seq_name}.tar"

    print(f"[prep] reconstructing {seq_name}.tar from /mnt/zju/archives ...")
    _cat_parts_to_file(f"/mnt/zju/archives/{seq_name}.tar.part*", tar_path)

    print(f"[prep] extracting {seq_name}.tar to {zju_root} ...")
    subprocess.run(
        ["bash", "-lc", f"tar -xf {tar_path} -C {zju_root}"], check=True)

    if not os.path.isdir(target_dir):
        raise RuntimeError(f"Extraction failed, not found: {target_dir}")

    print(f"[ok] extracted {seq_name}")
    return zju_root


def _mk_run_name(run_name: str | None) -> str:
    return run_name if (run_name and run_name.strip()) else time.strftime("run_%Y%m%d_%H%M%S")


@app.function(
    gpu="A100",
    timeout=86400,  # 7 天；你要更久就自己改大
    volumes={
        "/mnt/zju": vol_data,  # 数据卷
        "/mnt/out": vol_out,   # 输出卷（也放权重分片 /mnt/out/weights）
    },
)
def train_one(
    seq_names: str = "CoreView_390",
    run_name: str = "",
    # 默认 epochs 给超大值，相当于“没有上限”，直到你手动停 / 或到 timeout
    epochs: int = 999999,
    batch_size: int = 8,
    accum_steps: int = 1,
    num_workers_train: int = 8,
    num_workers_val: int = 4,
    # 为了避免你这次在 keep_largest_cc 上卡死，默认关掉
    fg_keep_largest_cc: int = 0,
    # 这些你现在代码里有（你也可以不改默认，只当透传开关）
    train_mask_mode: str = "",
    recon_mask_mode: str = "",
    best_by: str = "",
):
    # 1) 权重重组
    _ensure_model_pt()

    # 2) 数据解包到 /tmp（更快）
    zju_root = _ensure_seq_tar_extracted(seq_names)

    out_root = "/mnt/out/vggt"
    os.makedirs(out_root, exist_ok=True)

    run_name2 = _mk_run_name(run_name)
    log_dir = f"{out_root}/runs/{seq_names}/{run_name2}"
    ckpt_dir = f"{out_root}/ckpt/{seq_names}/{run_name2}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3) 让 vgg16 / torch hub 缓存写进 volume（避免每次都下载）
    env = os.environ.copy()
    env["TORCH_HOME"] = f"{out_root}/torch_cache"
    env["XDG_CACHE_HOME"] = f"{out_root}/cache"
    os.makedirs(env["TORCH_HOME"], exist_ok=True)
    os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)

    cmd = [
        "python", "train_view_decoder_ablation.py",
        "--zju_root", zju_root,
        "--seq_names", seq_names,
        "--log_dir", log_dir,
        "--ckpt_dir", ckpt_dir,
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--accum_steps", str(int(accum_steps)),
        "--num_workers_train", str(int(num_workers_train)),
        "--num_workers_val", str(int(num_workers_val)),
        "--fg_keep_largest_cc", str(int(fg_keep_largest_cc)),
    ]

    # 可选：你传了才加（避免参数名不匹配导致直接挂）
    if train_mask_mode.strip():
        cmd += ["--train_mask_mode", train_mask_mode.strip()]
    if recon_mask_mode.strip():
        cmd += ["--recon_mask_mode", recon_mask_mode.strip()]
    if best_by.strip():
        cmd += ["--best_by", best_by.strip()]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # 4) 把结果落盘到 volume
    vol_out.commit()
    print("[ok] committed volume:", VOL_OUT)
    print("[ok] log_dir =", log_dir)
    print("[ok] ckpt_dir=", ckpt_dir)


@app.local_entrypoint()
def main(
    seq_names: str = "CoreView_390",
    run_name: str = "",
    epochs: int = 999999,
    batch_size: int = 8,
    accum_steps: int = 1,
    num_workers_train: int = 8,
    num_workers_val: int = 4,
    fg_keep_largest_cc: int = 0,
    train_mask_mode: str = "",
    recon_mask_mode: str = "",
    best_by: str = "",
):
    train_one.remote(
        seq_names=seq_names,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        accum_steps=accum_steps,
        num_workers_train=num_workers_train,
        num_workers_val=num_workers_val,
        fg_keep_largest_cc=fg_keep_largest_cc,
        train_mask_mode=train_mask_mode,
        recon_mask_mode=recon_mask_mode,
        best_by=best_by,
    )
