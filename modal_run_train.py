# modal_run_train.py
from __future__ import annotations

import os
import glob
import shlex
import subprocess
from pathlib import Path

import modal

APP_NAME = "vggt-viewdecoder-train"

vol_data = modal.Volume.from_name("vggt-zju-data", create_if_missing=True)
vol_out = modal.Volume.from_name("vggt-out", create_if_missing=True)

ROOT = Path(__file__).resolve().parent  # 你的本地工程根目录（例如 F:\vggt）

# 关键：装 opencv-python-headless，让连通域/最大连通块走更快的实现（避免卡在 while stack）
# 另外补齐 libgl1/libglib2.0-0 等常见运行依赖（opencv/可视化库有时会用到）
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .pip_install("opencv-python-headless==4.10.0.84")
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


def _ensure_coreview390(seq_name: str):
    target_dir = f"/tmp/zju_mocap/{seq_name}"
    if os.path.isdir(target_dir):
        print(f"[ok] {seq_name} already extracted: {target_dir}")
        return "/tmp/zju_mocap"

    os.makedirs("/tmp/zju_mocap", exist_ok=True)
    tar_path = f"/tmp/{seq_name}.tar"

    print(f"[prep] reconstructing {seq_name}.tar from /mnt/zju/archives ...")
    _cat_parts_to_file(f"/mnt/zju/archives/{seq_name}.tar.part*", tar_path)

    print(f"[prep] extracting {seq_name}.tar to /tmp/zju_mocap ...")
    subprocess.run(
        ["bash", "-lc", f"tar -xf {tar_path} -C /tmp/zju_mocap"], check=True)

    if not os.path.isdir(target_dir):
        raise RuntimeError(f"Extraction failed, not found: {target_dir}")

    print(f"[ok] extracted {seq_name}")
    return "/tmp/zju_mocap"


@app.function(
    gpu=["H100", "A100"],
    # “不设步数/epoch 上限”通常意味着：不在命令里固定 --epochs/--max_steps；
    # 但 Modal 仍需要一个超时。这里给到 7 天，基本等于“不会被 24h 卡死”。
    timeout=60 * 60 * 24 * 7,
    volumes={
        "/mnt/zju": vol_data,  # 数据卷：/mnt/zju/archives/*.tar.part*
        "/mnt/out": vol_out,   # 输出卷：/mnt/out/vggt/... 以及 /mnt/out/weights/model.pt.part*
    },
)
def train_one(
    seq_names: str = "CoreView_390",
    # 下面这些都“可选”：None 就不往命令里塞，从而不在 modal 侧限制 epoch/step
    epochs: int | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    accum_steps: int | None = None,
    # 训练策略参数（你可以改默认，也可以在 modal run 时覆盖）
    train_mask_mode: str | None = "fg_conf",
    recon_mask_mode: str | None = "valid",
    best_by: str | None = "raw_psnr",
    conf_head_lr_mult: float | None = None,
    # 额外透传参数：例如 "--lr 5e-5 --warmup_steps 1000"
    extra_args: str = "",
):
    # 1) 权重重组
    _ensure_model_pt()

    # 2) 数据解包到 /tmp（更快）
    zju_root = _ensure_coreview390(seq_names)

    out_root = "/mnt/out/vggt"
    os.makedirs(out_root, exist_ok=True)

    cmd: list[str] = [
        "python", "train_view_decoder_ablation.py",
        "--zju_root", zju_root,
        "--seq_names", seq_names,
        "--log_dir", f"{out_root}/runs/{seq_names}",
        "--ckpt_dir", f"{out_root}/ckpt/{seq_names}",
    ]

    # 关键：不固定 epochs/max_steps（除非你显式传了）
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if max_steps is not None:
        cmd += ["--max_steps", str(max_steps)]

    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if accum_steps is not None:
        cmd += ["--accum_steps", str(accum_steps)]

    if train_mask_mode:
        cmd += ["--train_mask_mode", str(train_mask_mode)]
    if recon_mask_mode:
        cmd += ["--recon_mask_mode", str(recon_mask_mode)]
    if best_by:
        cmd += ["--best_by", str(best_by)]
    if conf_head_lr_mult is not None:
        cmd += ["--conf_head_lr_mult", str(conf_head_lr_mult)]

    if extra_args.strip():
        cmd += shlex.split(extra_args)

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(
    seq_names: str = "CoreView_390",
    epochs: int | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    accum_steps: int | None = None,
    train_mask_mode: str | None = "fg_conf",
    recon_mask_mode: str | None = "valid",
    best_by: str | None = "raw_psnr",
    conf_head_lr_mult: float | None = None,
    extra_args: str = "",
):
    train_one.remote(
        seq_names=seq_names,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        accum_steps=accum_steps,
        train_mask_mode=train_mask_mode,
        recon_mask_mode=recon_mask_mode,
        best_by=best_by,
        conf_head_lr_mult=conf_head_lr_mult,
        extra_args=extra_args,
    )
