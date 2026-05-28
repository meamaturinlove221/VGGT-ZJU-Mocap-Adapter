<h1 align="center">VGGT-ZJU-MoCap Adapter</h1>

<p align="center">
  <b>A dataset bridge for running VGGT-style human-prior experiments on ZJU-MoCap cases.</b>
</p>

<p align="center">
  Multi-view RGB · camera binding · mask audit · SMPL/SMPL-X prior alignment · VGGT-ready case export
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#why-this-repo-exists">Why</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#evidence-boundary">Evidence Boundary</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/VGGT-human--prior-blue" alt="VGGT human prior" />
  <img src="https://img.shields.io/badge/ZJU--MoCap-dataset--adapter-0f766e" alt="ZJU-MoCap dataset adapter" />
  <img src="https://img.shields.io/badge/status-research--utility-orange" alt="research utility" />
  <img src="https://img.shields.io/badge/policy-failure--closed-red" alt="failure closed" />
</p>

<p align="center">
  <img src="docs/figures/vggt_zju_mocap_adapter_architecture.svg" alt="VGGT-ZJU-MoCap Adapter architecture" width="100%" />
</p>

---

## What this repo is

`VGGT-ZJU-MoCap-Adapter` is the data reliability layer of the VGGT + human-prior project.

The job of this repository is simple and strict: turn a ZJU-MoCap-style case into a VGGT-ready package whose cameras, frames, masks, and body-prior projections can be trusted before any downstream model claims an improvement.

It does not try to be the final reconstruction model. It prepares the case, writes the evidence, and makes later comparison work less ambiguous.

---

## Why this repo exists

In multi-view human reconstruction, many failures look like model failures at first glance.

In practice, the real cause is often earlier in the chain:

- RGB frames and camera files are not bound to the same frame.
- Intrinsics are used after resizing without being updated.
- World-to-camera and camera-to-world conventions are mixed.
- Human masks keep too much background or cut away body parts.
- SMPL / SMPL-X priors project to the wrong place.
- A teacher/reference artifact is later mistaken for a student model output.

This repository keeps those problems visible. If the case is not geometrically reliable, it should fail at the data layer instead of being silently pushed into training.

---

## Where it sits in the project

```text
ZJU-MoCap-style data
        │
        ▼
VGGT-ZJU-MoCap-Adapter
  ├─ frame / path normalization
  ├─ camera audit
  ├─ mask audit
  ├─ body-prior projection check
  ├─ diagnostics and controls
  └─ VGGT-ready case package
        │
        ├─ vanilla VGGT baseline
        │
        └─ VGGT-SMPL-X Human Prior Adapter
              │
              ▼
      human-main full-scene RGB point-cloud evidence
```

Related routes:

- `VGGT-SMPL-X-Human-Prior-Adapter`: model-side prior injection and training route.
- `vggt_for_4k_4d`: full-scene evidence organization and report-side comparison route.

---

## Pipeline

### 1. Collect the case

The adapter starts from a ZJU-MoCap-style case: multi-view RGB frames, camera parameters, masks, subject metadata, and available body-prior annotations.

### 2. Normalize frame binding

The first pass makes the case explicit: subject, sequence, camera id, frame id, file path, and image size should be traceable from the exported manifest.

### 3. Audit cameras

The camera layer checks whether intrinsics and extrinsics match the exported image size and the chosen coordinate convention. The output should make it clear which transform is used and where it is used.

### 4. Audit masks

Masks are treated as geometry inputs, not decoration. A bad mask can make a correct prior look wrong or make a wrong prior look plausible. The export should keep mask diagnostics available for later review.

### 5. Align body priors

SMPL / SMPL-X priors are projected through the audited camera path. Projection overlays are used here as diagnostics: they help check alignment, but they are not final reconstruction evidence.

### 6. Export a VGGT-ready package

A clean case package can be passed to vanilla VGGT as a baseline or to the downstream human-prior adapter for model-side experiments.

---

## Output package

A typical export is expected to contain:

- multi-view RGB frames;
- camera intrinsics and extrinsics;
- human / foreground masks;
- SMPL or SMPL-X body-prior data when available;
- projection overlays for inspection;
- reference depth / point artifacts when needed;
- source manifests;
- audit and failure reports.

The exact file layout can evolve, but the rule should stay the same: every exported artifact must have a source and a reason to exist.

---

## Evidence boundary

This repository uses a strict boundary between four kinds of artifacts.

| Artifact type | Role | Can be used as final student evidence? |
| --- | --- | --- |
| Teacher / reference | Checks geometry and provides supervision clues | No |
| Vanilla VGGT baseline | Baseline comparison | No |
| Diagnostics | Finds camera, mask, or projection problems | No |
| Student candidate | Downstream model output after consuming the audited case | Only if it passes visual and comparison gates |

A projection overlay can prove that a prior lands in the right image region. It cannot prove that the 3D point cloud is good.

A depth or point reference can help debug the case. It cannot be promoted as the model result.

The final target remains a human-main, full-scene RGB point cloud: the person should be the subject of the view, and enough environment should remain to prove that the result still lives in the scene rather than in an isolated crop.

---

## Failure-closed rule

If the case does not pass alignment checks, the pipeline should stop.

The correct output in that situation is not a polished figure. It is a short failure note that records what broke:

```text
camera convention mismatch
mask and projected body prior do not overlap
frame index cannot be traced
resized image uses stale intrinsics
teacher/reference artifact is being confused with student output
```

Failing early is useful. It protects the downstream experiment from spending hours on a case that was already broken before training began.

---

## What makes this useful

This repository is built for the part of research that usually gets hidden in a short paper paragraph: data binding, audits, failed cases, controls, and evidence packaging.

It is useful when the goal is not only to run a demo, but to answer a harder question:

> If the model improves, can we prove that the improvement came from the intended human-prior route rather than from a cleaner crop, a lucky mask, or a reference artifact?

That is the purpose of the adapter.

---

## Current status

This repository should be read as a research utility and dataset-bridge route. It is not presented as a final point-cloud reconstruction benchmark.

The current focus is to make ZJU-MoCap-style cases auditable and reusable inside the wider VGGT + SMPL-X experiment stack.

---

## Figure

The architecture figure is stored at:

```text
docs/figures/vggt_zju_mocap_adapter_architecture.svg
```

---

## Data note

This repository is designed to work around ZJU-MoCap-style local data and body-prior assets. Restricted datasets, private RGB frames, masks, camera files, and body model files should stay outside the public repository unless their license explicitly allows redistribution.
