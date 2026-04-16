# Human Prior Modal Pipeline

## Goal

This pipeline does all of the following in one flow:

1. launch `finetune_vggt_pseudo.py` on Modal A100
2. fetch remote logs and checkpoints back to local
3. export high-resolution fused point clouds as `.ply`
4. export high-resolution fixed-view preview contact sheets
5. export high-resolution target-view reprojection images
6. stop lingering Modal apps after the run

## Main Entry

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_4k4d_human_prior_end_to_end.ps1
```

Dry-run only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_4k4d_human_prior_end_to_end.ps1 -DryRun
```

## Important Scripts

- `scripts/run_modal_4k4d_human_prior_finetune.ps1`
  Launches Modal finetune and now checks human-prior sidecars before wasting cloud time.
- `scripts/fetch_modal_4k4d_outputs_highres.ps1`
  Fetches remote run outputs, exports high-resolution local artifacts, and verifies cleanup.
- `scripts/run_modal_4k4d_human_prior_end_to_end.ps1`
  One-shot wrapper for training + fetching + high-res export + cleanup verification.

## Returned Artifacts

The return root is created under:

```text
out_vis/modal_returns/<seq>_<timestamp>/
```

Expected returned files include:

- remote `finetune_vggt_summary.json`
- remote `finetune_vggt_metrics.jsonl`
- remote `model_ft_zju.pt`
- remote `model_ft_zju_last.pt`
- local high-resolution `.ply`
- local high-resolution point-cloud preview `.png`
- local high-resolution reprojection `.png`
- local `return_manifest.json`
- local `return_manifest.md`

## High-Resolution Settings

Current defaults:

- `DepthUpsampleFactor = 2.0`
- `PreviewTileSize = 1400`
- `MaxPoints = 180000`
- `PreviewPoints = 90000`

These are exposed as script parameters and can be increased further if needed.

## Cleanup Guarantee

`scripts/fetch_modal_4k4d_outputs_highres.ps1` now:

- calls `modal app list`
- stops non-stopped apps matching `vggt-zju-runner`
- records cleanup status in the return manifest
- fails when cleanup is required but active apps still remain

## Current Hard Blocker

Strict human-prior mode now fails fast if sidecars are missing.

In the current workspace/session, the precheck reports:

```text
modal volume ls --json vggt-zju-data /4k4d_bridge/0012_11/human_prior
```

and that path does not currently contain the expected `.npz` sidecars.

So:

- code path is ready
- Modal path is ready
- return/high-res/cleanup path is ready
- the remaining blocker is the human-prior sidecar data itself
