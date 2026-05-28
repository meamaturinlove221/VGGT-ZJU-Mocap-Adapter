# VGGT-ZJU-MoCap Adapter

<p align="center">
  <b>A VGGT human-prior experiment data bridge for ZJU-MoCap-style data.</b>
</p>

<p align="center">
  Multi-view RGB · camera binding · mask audit · SMPL/SMPL-X prior alignment · VGGT-ready case export
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#what-problem-this-repo-solves">What problem this repo solves</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#project-value">Project value</a>
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

It does not package final results, and it does not treat diagnostic figures as achievements. This repository does one thing: it organizes ZJU-MoCap-style data into a trusted case that can be consumed by VGGT or a human-prior branch, while making the key parts clear: cameras, frame indices, masks, and body-prior projections.

Whether the downstream model has truly improved must be judged from a clean data entry point. This repository handles that entry point.

---

## What problem this repo solves

In multi-view human reconstruction, many issues that look like model problems are eventually traced back to data binding problems.

Common cases include:

- RGB frames and camera files do not match;
- intrinsics are not updated after image resizing;
- world-to-camera and camera-to-world conventions are mixed;
- unstable mask boundaries affect the human region and scene context;
- SMPL / SMPL-X priors are projected to the wrong image location;
- teacher/reference artifacts are mistaken for student model outputs;
- projection overlays look correct, while the 3D point cloud still has not formed a reliable human shape.

This repository exposes these issues early. If the data layer is not trustworthy, the pipeline should stop here instead of continuing training, taking more screenshots, and pushing the problem further downstream.

---

## Project position

```text
ZJU-MoCap-style data
        │
        ▼
VGGT-ZJU-MoCap-Adapter
  ├─ path and frame-index normalization
  ├─ camera-parameter audit
  ├─ mask audit
  ├─ body-prior projection check
  ├─ diagnostic figures and control packages
  └─ VGGT-ready case package
        │
        ├─ vanilla VGGT baseline
        │
        └─ VGGT-SMPL-X Human Prior Adapter
              │
              ▼
      human-main full-scene RGB point-cloud evidence
```

Relation to other repositories:

- `VGGT-SMPL-X-Human-Prior-Adapter`: the model-side route for human-prior injection and training.
- `vggt_for_4k_4d`: the route for full-scene evidence, control results, and report-side comparisons.

This repository sits earlier in the chain. Its role is to turn ZJU-MoCap-style data into an input that can actually be discussed.

---

## Pipeline

### 1. Collect the case

The input usually contains multi-view RGB, camera parameters, masks, subject / sequence / frame information, and available SMPL or SMPL-X human-prior data.

### 2. Normalize frame indices and paths

Every exported sample should be traceable back to its original subject, sequence, camera id, frame id, and file path. The worst case here is that the pipeline runs, but no one can tell which frame was actually used.

### 3. Check cameras

Camera parameters are the foundation of multi-view geometry. Before export, intrinsics, extrinsics, coordinate conventions, image size, and resizing relationships need to be explicit. If this part is wrong, projection, depth, and point clouds will all be wrong afterward.

### 4. Check masks

Masks are not decorative images. They affect the human region, background retention, projection checks, and downstream supervision. If the mask itself is unreliable, even a visually pleasing model result should be treated carefully.

### 5. Align the human prior

SMPL / SMPL-X priors need to be projected back to the image through the same camera chain. Projection overlays are only used here as diagnostic figures to check whether the prior lands in a reasonable place. They cannot replace final 3D point-cloud evidence.

### 6. Export the VGGT-ready package

A case that passes the checks can enter the downstream routes: vanilla VGGT as the baseline, and the VGGT-SMPL-X human-prior branch for model-side experiments.

---

## Project value

The work based on the ZJU-MoCap dataset gave us a useful stepping stone for getting familiar with VGGT, and it also gave us experience for exploring human feed-forward prior experiments. Along the way, we ran into many typical engineering problems, including but not limited to repeatedly hitting walls and gradually making human point-cloud hole filling the first priority.

There were also problems in the evaluation method: the point-cloud figures were not clear enough, and the human subject did not show real precision improvement. Dataset background issues also caused VGGT to confuse the background with the human subject during reconstruction, making the human modeling less clear. What was really missing was a model representation, a training objective, local detail generation, and a 3D main-figure evaluation system.

This means the route must move toward:

- real 3D learned residual;
- multi-view detail supervision;
- baseline high-confidence detail preservation;
- SMPL feature-conditioned local geometry branch;
- human-main full-scene visual gate.

Following the advisor's suggestion, we introduced the 4K4D dataset and SMPL-X, and achieved better results in the new project.

---

## Current status

This repository is currently positioned as a research utility / dataset bridge, not as a final point-cloud reconstruction benchmark.

The current focus is to make ZJU-MoCap-style cases auditable and reusable, and to let them enter the more complete VGGT + SMPL-X human-prior experiment stack.

---

## Data note

This repository is designed around local ZJU-MoCap-style data and human model assets. Restricted datasets, RGB frames, masks, camera files, and SMPL/SMPL-X body model files should not be placed directly in the public repository unless their licenses explicitly allow redistribution.

---

## Architecture figure

The architecture figure is stored at:

```text
docs/figures/vggt_zju_mocap_adapter_architecture.svg
```
