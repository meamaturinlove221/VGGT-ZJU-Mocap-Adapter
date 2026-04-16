# Human Transparency Probe Runbook

## Scope

This runbook defines the next support localization phase after the completed `T0/T1/T2/T4` round.
It uses the current codebase and current findings:

- `T1` worse than `T0-smoke`
- `T2 ~= T1`
- `T4` does not rescue foreground presence

Implication:

- the next target is the **support path itself**
- the next phase must separate upstream precompute support from each training-time support consumer

Non-goals:

- no `dyn_proxy` tuning
- no BA / reprojection branch
- no backbone edits
- no overnight autoloop during probe runs
- no parallel probes

## Current Findings To Preserve

1. `T0-smoke` is a live-chain smoke test, not a causal baseline.
2. `T1` already showed harm with `blend` off, so support toxicity is upstream of `blend`.
3. `T2` did not materially worsen `T1`, so `point_target_blend_by_mv_support` is demoted from primary suspect.
4. `T4` made foreground presence worse, so `point_mv_depth_region_mode = bg_only` is not the current default direction.

## Code Reality

Support enters the system in two distinct layers.

### Layer A: Upstream precompute support

`PrecomputeMvSupportOn` computes multiview support during geometry precompute and changes saved pseudo-geometry / confidence before training starts.

Current local code default has been hardened:

- when support generation is enabled and no explicit region override is provided,
- `PrecomputeMvSupportRegionMode` now defaults to `auto`,
- and `auto` resolves to `bg_only` when a foreground mask can be loaded, otherwise falls back to `all`.

This is a safety default, not yet a promoted validated baseline.
`G0` was later run once as a real single-window cloud validation and came out clearly better than `S0`, while still not fully replacing the current `T0-smoke`-like default.
Any local dry-run contract for `G0` should still be treated as readiness-only until a matching live candidate/result exists.
Use `scripts/prepare_g0_single_window_local.ps1` to snapshot the frozen `S0` baseline and refresh all local readiness materials before giving `G0` any future confirmation window.

### Layer B: Training-time support consumers

Current real consumers are:

- `PointSupportMode`
- `PointMvDepthSupportMode`
- `PointMvMaskSupportMode`
- `PointTargetMode=blend` + `PointTargetBlendByMvSupport=on`

The next probe phase should isolate Layer A first, then each Layer B consumer one by one.

## Phase 1: Smoke Gate

### T0-smoke

Purpose:

- verify the live chain still reaches honest stage2 evaluation
- verify heartbeat / fg metrics / candidate single-source JSON are alive
- do **not** use this run to draw algorithm conclusions

Required success split:

1. Execution success
   - at least one valid `stage2` row with:
     - `exit_code = 0`
     - `ghost_score_mean` non-empty
     - `ghost_visual_score` non-empty
     - `pred_luma_mean` non-empty
     - `pred_nonblack_ratio_thr008` non-empty
     - `fg_pred_luma_mean` non-empty
     - `fg_pred_nonblack_ratio` non-empty
     - `fg_pred_contrast` non-empty
     - `fg_pred_tgt_l1` non-empty
     - `eval_num_src_views_mismatch = false`

2. Audit success
   - `candidate_result_latest.json`
   - `overnight_ghost_autoloop_latest.json`
   - `watch_ghost_outputs_latest.json`
   stay consistent on the active candidate fields

Only if both pass should support localization probes start.

## Phase 2: Support Localization Probes

Before consumer-path localization, there is one more upstream generation-region baseline.

### G0 Upstream Support Generated On Background Only

Contract:

- `PointmapSource = depth_unproject`
- `PointTargetMode = depth_unproject`
- `PrecomputeMvSupportOn = on`
- `PrecomputeMvSupportRegionMode = bg_only`
- `PrecomputeMvSupportFgMaskSource = mask`
- `PointSupportMode = off`
- `PointMvDepthSupportMode = off`
- `PointMvMaskSupportMode = off`
- `PointTargetBlendByMvSupport = off`

Purpose:

- answer whether the generation-stage support poison is largely caused by letting support see the human foreground at all
- compare directly against `S0`, where support generation is still `all`

All probes must share:

- same data window
- same seed
- same `eval_num_src_views_actual`
- same resume ckpt / bootstrap source

### S0 Upstream Precompute Support Only

Contract:

- `PointmapSource = depth_unproject`
- `PointTargetMode = depth_unproject`
- `PrecomputeMvSupportOn = on`
- `PointSupportMode = off`
- `PointMvDepthSupportMode = off`
- `PointMvMaskSupportMode = off`
- `PointTargetBlendByMvSupport = off`
- `PointTargetBlendMvRegionMode = all`
- `PointMvDepthRegionMode = all`
- `UseFgMask = on`
- `FgMaskSource = mask`
- `LambdaPointMvMask = 0`

Purpose:

- answer whether upstream precompute support alone already poisons the pseudo-geometry path

### S1 Add Point Support Consumer Only

Contract:

- same as `S0`
- only change: `PointSupportMode = direct`

Purpose:

- isolate whether the main point/target support consumer is a toxin

### S2 Add mv-depth Support Consumer Only

Contract:

- same as `S0`
- only change: `PointMvDepthSupportMode = direct`

Purpose:

- isolate whether `mv_depth` support weighting is a toxin

### S3 Add mv-mask Support Consumer Only

Contract:

- same as `S0`
- only change: `PointMvMaskSupportMode = inverse`

Purpose:

- isolate whether `mv_mask` support weighting is a toxin

## Readout

Primary:

- `ghost_visual_score`
- `ghost_score_mean`
- `pred_luma_mean`
- `pred_nonblack_ratio_thr008`

Foreground presence:

- `fg_pred_luma_mean`
- `fg_pred_nonblack_ratio`
- `fg_pred_contrast`
- `fg_pred_tgt_l1`

Support-path readout from row / candidate JSON:

- `precompute_mv_support_on`
- `point_support_mode`
- `point_mv_depth_support_mode`
- `point_mv_mask_support_mode`
- `point_target_mode`
- `point_target_blend_by_mv_support`
- `point_mv_support_mean / fg_mean / bg_mean`
- `point_support_eff_mean / fg_mean / bg_mean`
- `point_mv_depth_support_eff_mean / fg_mean / bg_mean`
- `point_mv_mask_support_eff_mean / fg_mean / bg_mean`

Interpretation:

- `S0` worse than `T0-smoke`
  - upstream precompute support itself is harmful

- `S1` worse than `S0`
  - `PointSupportMode=direct` is a main toxin

- `S2` worse than `S0`
  - `PointMvDepthSupportMode=direct` is a main toxin

- `S3` worse than `S0`
  - `PointMvMaskSupportMode=inverse` is a main toxin

## Recommended Order

Current local evidence has already upgraded the priority order:

- `S0` is a clean generation-only probe
- `S0` was worse than `T0-smoke`
- multi-frame local diagnosis shows `bg_only` sharply reduces foreground `depth_conf` suppression

So the next support-enabled baseline was no longer `S1`; it was `G0`, and that single validation run has now already been completed.

1. keep `T0-smoke`-like settings as the temporary default for the dynamic-human lane
2. treat `G0` as the validated next support-enabled candidate
3. only if `G0` still fails to explain enough, resume `S1`
4. then `S2`
5. then `S3`

Reason:

- `T0-smoke` validates the live chain
- `G0` directly tests the strongest current hypothesis: support generation should not see dynamic foreground
- `S1/S2/S3` are now second-layer consumer probes, not the immediate next step
- blend remains intentionally excluded from this phase because the previous round already demoted it

## Current Local Evidence

See:

- `logs/modal_phase5/reports/support_generation_diagnosis_20260310.md`
- `logs/modal_phase5/reports/support_generation_multiframe_latest.md`
- `logs/modal_phase5/reports/support_generation_gallery_latest.png`
- `logs/modal_phase5/reports/support_generation_g0_readiness_latest.md`
- `logs/modal_phase5/reports/support_generation_g0_readiness_latest.png`
- `logs/modal_phase5/reports/support_generation_g0_verification_latest.md`
- `logs/modal_phase5/reports/support_generation_g0_verification_latest.png`

What matters:

- in `S0`, support generation alone already hurt the result relative to `T0-smoke`
- in local multi-frame diagnosis, `bg_only` materially reduces foreground `depth_conf` damage

This is why `G0` became the preferred next cloud baseline, and why it remains the correct support-enabled candidate to compare against `T0-smoke` and `S0`.

## Temporary Default Until Next Probe Round

For the human-dynamic lane, keep the temporary default close to `T0-smoke`:

- `PrecomputeMvSupportOn = off`
- `PointTargetBlendByMvSupport = off`
- `PointTargetMode = depth_unproject`
- `PointMvDepthRegionMode = all`
- `UseFgMask = on`

Do **not** promote `PointMvDepthRegionMode = bg_only` as the new default.
`T4` already argued against that.

## Execution Helpers

Do not hand-type the probe commands.

Use:

- `scripts/emit_human_transparency_probe_commands.ps1`

For direct execution of one probe after `T0-smoke` passes:

- `scripts/run_human_transparency_probe_once.ps1`

For the smoke gate itself:

- `scripts/run_p0_single_resume_once.ps1`

To refresh the latest probe summary and comparison grid locally:

- `scripts/refresh_support_probe_summary.ps1`

To refresh the local multi-frame support-generation diagnosis:

- `scripts/refresh_support_generation_multiframe.ps1`

To verify the exact `G0` contract locally without starting cloud execution:

- `scripts/verify_g0_contract_local.ps1`
