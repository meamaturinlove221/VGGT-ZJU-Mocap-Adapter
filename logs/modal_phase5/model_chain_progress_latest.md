# Model Chain Progress

## Scope
- Route: `PixelSplat -> MVSplat -> MVSplat360 -> DepthSplat -> AnySplat`
- Baseline gate model: `phase5_final_r20260208_noema_rawpsnr`

## Current Status
- Phase 1.5: in progress
  - `phase5_final` release manifest exists.
  - Gate script exists: `scripts/release_gate_phase5_final.ps1`.
  - Cross-seq smoke script exists: `scripts/cross_seq_smoke_phase5.ps1`.
  - Gate run: pass (`logs/modal_phase5/release_gate_result_latest.json`).
  - Cross-seq smoke: skipped due to missing eligible sequence except `CoreView_390`.
- Phase 2: ready to start
  - Converter exists: `tools/convert_to_pixelsplat_format.py`.
  - Pending: generate PixelSplat-format subset and run minimal loop.
- Phase 3-6: not started

## Phase Gates
- Phase 1.5 gate:
  - `N == 118`
  - `PSNR >= 21.15`
  - `SSIM >= 0.860`
  - `wL1 <= 0.061`
  - `source_fg_key == tgt_fg`
  - `tgt_mask_path` present
- Cross-seq smoke gate:
  - 2 sequences (excluding `CoreView_390`)
  - each with `num_samples=30`
  - `PSNR >= 19.5`, `SSIM >= 0.82`, `wL1 <= 0.085`

## Artifacts
- Gate latest JSON: `logs/modal_phase5/release_gate_result_latest.json`
- Cross-seq latest CSV: `logs/modal_phase5/cross_seq_smoke_latest.csv`
- Cross-seq latest JSON: `logs/modal_phase5/cross_seq_smoke_latest.json`
- Final report latest: `logs/modal_phase5/final_release_report_latest.md`

## Notes
- Current dataset availability in Modal volume appears limited to `CoreView_390` for `vggt_geom_ft_20260208_044454`.
- To enable true cross-seq smoke, first precompute and upload matching geom subdir for at least two additional sequences.
