# Phase5 Final Release Report

## Release
- model_id: `phase5_final_r20260208_noema_rawpsnr`
- date: `2026-02-08`
- selected_ckpt: `/mnt/out/viewdec_ablation/CoreView_390_20260208_110514/ckpt/viewdec_ablation_best.pth`
- geom_subdir: `vggt_geom_ft_20260208_044454`
- release_verify_modal_url: <https://modal.com/apps/shimakaze22333/main/ap-gL7hBqCg0QnBZb17AUzxkA>

## Metrics (val, no_ema)
- N: 118
- weighted-L1: 0.0599219092815104
- PSNR: 21.1999153686782
- SSIM: 0.863973554918322

## Comparison
- vs phase4_c_best_noema PSNR delta: -0.191688489105744
- vs phase4_c_best_noema SSIM delta: 0.0122249954837864
- vs phase4_c_best_noema wL1 delta: -0.00205419900811325

## Validation Checks
- N==118: True
- source_fg_key==tgt_fg: True
- tgt_mask_path present: True
- overlay_applied: True

## Default Run Command
```powershell
$env:VGGT_CODE_DIR='F:\vggt'
$env:VGGT_PROFILE='phase5_final'
modal run modal_run_train.py
```

Optional wrapper:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_infer_phase5_final.ps1 -CodeDir F:\vggt
```

## Rollback Anchors
- `/mnt/out/viewdec_ablation/CoreView_390_20260207_185302/ckpt/viewdec_ablation_best.pth`
- `/mnt/out/viewdec_ablation/CoreView_390_20260208_065946/ckpt/viewdec_ablation_best.pth`