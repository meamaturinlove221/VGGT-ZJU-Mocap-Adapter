# 4K4D Bridge

This repo's active precompute/train/infer path is still organized around a ZJU-style multi-view sequence layout:

```text
<data_root>/<seq>/
  Camera_00/000000.png
  Camera_01/000000.png
  ...
  mask/Camera_00/000000.png
  mask/Camera_01/000000.png
  ...
```

The 4K4D raw subset can be bridged into that shape without rewriting the downstream pipeline.

## 1. Inspect the raw subset

```powershell
python tools/dna_4k4d.py inventory `
  --dataset-path "G:\数据集\datasets" `
  --json-out "reports\dna_4k4d_inventory.json"
```

Build a one-sequence manifest:

```powershell
python tools/dna_4k4d.py manifest `
  --dataset-path "G:\数据集\datasets" `
  --seq 0012_11 `
  --frame 0 `
  --target-camera 00 `
  --auto-sources 6 `
  --output-dir "reports\dna_case_probe"
```

## 2. Export a ZJU-style bridge sequence

Example: export one 7-view subset for sequence `0012_11`, frames `[0, 60)` with step 10.

```powershell
python tools/export_4k4d_zju_bridge.py `
  --dataset-root "G:\数据集\datasets" `
  --seq 0012_11 `
  --output-root "G:\bridge_4k4d" `
  --target-camera 00 `
  --auto-sources 6 `
  --frame-start 0 `
  --frame-stop 60 `
  --frame-step 10
```

The exporter writes:

```text
G:\bridge_4k4d\0012_11\
  Camera_00\000000.png
  Camera_01\000000.png
  ...
  mask\Camera_00\000000.png
  mask\Camera_01\000000.png
  bridge_manifest.json
```

If you already know the exact cameras, prefer explicit ids:

```powershell
python tools/export_4k4d_zju_bridge.py `
  --dataset-root "G:\数据集\datasets" `
  --seq 0012_11 `
  --output-root "G:\bridge_4k4d" `
  --camera-ids 00 01 10 19 28 37 46 `
  --frames 0 10 20 30
```

## 3. Upload the bridged sequence to Modal

```powershell
modal run modal_prepare_4k4d.py `
  --local-bridge-root "G:\bridge_4k4d\0012_11" `
  --remote-root "4k4d_bridge"
```

That makes the sequence available under:

```text
/mnt/data/4k4d_bridge/0012_11
```

## 4. Reuse the existing Modal pipeline

Once the bridged sequence is in the data volume, the existing scripts can keep using the current env contract.

Point the existing `VGGT_ZJU_ROOT` at the bridge root:

```powershell
$env:VGGT_ZJU_ROOT = "/mnt/data/4k4d_bridge"
$env:VGGT_SEQ_NAMES = "0012_11"
$env:VGGT_CAM_NAMES = "Camera_00,Camera_01,Camera_10,Camera_19,Camera_28,Camera_37,Camera_46"
```

Then run the same `modal_run_train.py` / precompute / finetune flow as before.

## Notes

- Keeping the bridge layout ZJU-like means `precompute_zju_vggt_geom.py`, `zju_dataset_view.py`, `infer_view_decoder_ablation.py`, and `finetune_vggt_pseudo.py` can continue to work unchanged.
- The bridge exporter is intentionally frame/camera selective so we do not explode storage by materializing every 4K4D image up front.
