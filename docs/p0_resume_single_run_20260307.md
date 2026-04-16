# P0 Single-Run Resume Runbook

## Goal

Resume exactly one cloud run whose only purpose is to answer:

1. Can `stage2` produce valid rows under the reduced-load `P0` path?
2. If not, does the failure now localize to `teacher.forward_batch`, postprocess/save, or still to outer no-output logic?

This runbook is intentionally narrower than the full overnight plan.
Do not mix `P1`, `P2`, BA, or new algorithm changes into this run.

## Preconditions

Before resuming cloud execution, keep all of the following true:

1. No local autoloop / watcher / `modal run` process is alive.
2. No live Modal app exists.
3. Current code includes the local-only `P0` hardening already added:
   - `stage2` forced to `depth_unproject` while `P0` is pending
   - `stage2 precompute_mv_support_on=off`
   - `stage2 point_target_blend_by_mv_support=off`
   - precompute segmented heartbeat / timing
   - stale modal-progress handling
   - fixed CSV bool parsing
4. Preferred local-only maintenance entrypoint before any cloud resume:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_p0_local_maintenance.ps1`
5. If running validators separately, use:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_paused_state.ps1`
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_local_readiness.ps1`
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_source_contract.ps1`
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_candidate_result_consistency.ps1`
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preflight_p0_resume_local.ps1`
6. Optionally snapshot the frozen state before resuming:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/snapshot_p0_state.ps1`
7. Optionally emit the machine-readable manifest:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/emit_p0_resume_manifest.ps1`

## Fixed Rules

These rules are non-negotiable for the next resume:

1. Single main chain only.
2. No parallel second training chain.
3. No `P1` AnySplat six-pack.
4. No `P2 dyn_proxy` enablement in conclusions.
5. No BA / pose-refine side path.
6. No increase of `precompute_batch_frames` above the current stable value.

## Required Runtime Shape

The resumed run must satisfy all of these conditions.

### Stage1

Use `stage1_strong` only as fast screening:

- `no_output_timeout ~= 240s`
- fast-fail enabled
- no repeated same-candidate loops

Expected result:

- candidates either fail quickly with `candidate_invalid_reason=no_output_timeout`
- or produce a ckpt and hand off to `stage2`

### Stage2

While `p0_gate_pass=false`, `stage2` must run under the stabilized route:

- `pointmap_source=depth_unproject`
- `precompute_mv_support_on=off`
- `point_target_blend_by_mv_support=off`
- dual-lane off
- post-rescue off

These values should be visible in:

- `logs/modal_phase5/overnight_ghost_autoloop_latest.json`
- `logs/modal_phase5/overnight_ghost_autoloop_latest.md`
- `logs/modal_phase5/watch_ghost_outputs_latest.json`
- `logs/modal_phase5/watch_ghost_outputs_latest.md`
- `logs/modal_phase5/ghost_mvdepth_sweep_latest.csv`

## What Must Be Visible In Logs

The resumed run is only useful if the new precompute instrumentation appears.

### Expected `precompute_zju_vggt_geom.py` markers

- `[precompute-heartbeat] {"phase": ...}`
- `batch_begin`
- `batch_teacher_done`
- `frame=... save_done`
- `batch_done`

If `mv_support` is turned back off as planned, `mv_support_done` may be absent. That is expected.

### Expected `VGGTGeomTeacher` marker

- `[VGGTGeomTeacher] batch=... resolve=... load=... transfer=... agg=... cam=... depth=... point=... total=...`

If these markers do not appear at all, the run is still too opaque to trust.

## P0 Acceptance

`P0` passes only if, within 90 minutes from resume:

1. `stage2` produces at least `3` rows with `exit_code=0`
2. each row has all of:
   - `ghost_score_mean`
   - `ghost_visual_score`
   - `pred_luma_mean`
   - `pred_nonblack_ratio_thr008`
3. `visual_guard_blocked=false`
4. `quality_guard_blocked=false`
5. `eval_num_src_views_mismatch=false`

Anything short of this is still `P0` failure.

## Failure Triage

Use this exact split when the next run fails.

### Case A: `stage1` still dies at 240s

Interpretation:

- cloud-side training visibility is still not reliable
- do not blame geometry/precompute yet

Next focus:

- modal wrapper heartbeat
- child stdout forwarding
- short finetune quiet/stall logic

### Case B: `stage2` reaches precompute, and `batch_teacher_done` never appears

Interpretation:

- failure is likely before or inside `teacher.forward_batch`

Next focus:

- `VGGTGeomTeacher.forward_batch`
- input resolution / load / transfer
- model forward path

### Case C: `batch_teacher_done` appears, but `save_done` never appears

Interpretation:

- failure is after model forward
- likely postprocess, world-point packaging, or write path

Next focus:

- postprocess in `_process_frame_batch(...)`
- serialization / `np.savez_compressed`
- per-frame output splitting

### Case D: `stage2` succeeds only after turning off `mv_support`

Interpretation:

- current `P0` blocker was indeed support-side overhead
- do not immediately turn it back on

Next focus:

- keep `P0` stable first
- only later run a daytime A/B for `precompute_mv_support_on`

### Case E: `stage2` still times out with `mv_support=off`

Interpretation:

- blocker is deeper than support
- likely in model forward, transfer, or save path

Next focus:

- inspect new timing buckets from `VGGTGeomTeacher` and `precompute_zju_vggt_geom.py`

## Data To Save After The Run

After the single run, preserve these artifacts before any further changes:

1. `logs/modal_phase5/overnight_ghost_autoloop_latest.json`
2. `logs/modal_phase5/overnight_ghost_autoloop_latest.md`
3. `logs/modal_phase5/watch_ghost_outputs_latest.json`
4. `logs/modal_phase5/vggt_ft_sweep_latest.csv`
5. `logs/modal_phase5/ghost_mvdepth_sweep_latest.csv`
6. the current autoloop `.out.log`

## Explicitly Out Of Scope

Do not add any of the following to the next resume run:

1. `P1` six-pack
2. `P2` enablement as a judged improvement
3. BA / COLMAP refine
4. new reprojection loss
5. higher `precompute_batch_frames`
6. alternate GPU experiments

## Expected Decision After One Run

At the end of the next single resume run, only decide one of these:

1. `P0 passed`
2. `P0 still blocked in teacher.forward_batch`
3. `P0 still blocked after teacher.forward_batch`
4. `P0 still blocked in stage1 visibility/stall logic`

Do not make a ghost-algorithm conclusion from that run.
