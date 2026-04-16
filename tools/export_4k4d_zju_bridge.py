from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dna_4k4d as dna  # noqa: E402


def decode_encoded_image(buffer: np.ndarray) -> Image.Image:
    decoded = cv2.imdecode(np.asarray(buffer), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode encoded image bytes.")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_rgb_frame(main_handle: h5py.File, camera_id: str, frame_id: str) -> Image.Image:
    cam_num = int(camera_id)
    group_name = dna.camera_group_name(camera_id)
    frame_key = str(int(frame_id))
    buffer = main_handle[group_name][str(cam_num)]["color"][frame_key][()]
    return decode_encoded_image(buffer)


def load_mask_frame(ann_handle: h5py.File, camera_id: str, frame_id: str) -> Image.Image:
    cam_key = str(int(camera_id))
    frame_key = str(int(frame_id))
    buffer = ann_handle["Mask"][cam_key]["mask"][frame_key][()]
    rgb = decode_encoded_image(buffer)
    gray = np.max(np.asarray(rgb), axis=2).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export 4K4D frames into a ZJU-style bridge sequence so the existing "
            "precompute/train/infer pipeline can reuse it."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Extracted data_used_in_4K4D root or its parent folder.",
    )
    parser.add_argument("--seq", required=True, help="Sequence id such as 0012_11.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output parent directory. The exporter writes <output-root>/<seq>/...",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="*",
        default=[],
        help="Explicit camera ids to export. Example: 00 01 10 19 28 37 46",
    )
    parser.add_argument(
        "--target-camera",
        help="Optional target camera. Used only when --camera-ids is omitted.",
    )
    parser.add_argument(
        "--source-cameras",
        nargs="*",
        default=[],
        help="Optional source cameras. Used only when --camera-ids is omitted.",
    )
    parser.add_argument(
        "--auto-sources",
        type=int,
        default=6,
        help="Auto-pick N source cameras when target camera is set and source cameras are omitted.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=[],
        help="Explicit frame ids. If omitted, frame slicing arguments are used.",
    )
    parser.add_argument("--frame-start", type=int, default=0, help="Inclusive frame start.")
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=-1,
        help="Exclusive frame stop. -1 means to the end of the common frame list.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap after slicing. 0 means no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs. By default existing files are skipped.",
    )
    return parser.parse_args()


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def choose_cameras(args: argparse.Namespace, available_cameras: list[str]) -> list[str]:
    explicit = [dna.normalize_camera_id(camera) for camera in args.camera_ids]
    if explicit:
        return _unique_keep_order(explicit)

    target_camera = (
        dna.normalize_camera_id(args.target_camera)
        if args.target_camera is not None and str(args.target_camera).strip() != ""
        else None
    )
    source_cameras = [dna.normalize_camera_id(camera) for camera in args.source_cameras]
    if target_camera is not None:
        if not source_cameras:
            source_cameras = dna.auto_pick_sources(available_cameras, target_camera, args.auto_sources)
        return _unique_keep_order([target_camera, *source_cameras])
    return list(available_cameras)


def _frame_ids_for_camera(main_handle: h5py.File, ann_handle: h5py.File, camera_id: str) -> set[str]:
    cam_key = str(int(camera_id))
    group_name = dna.camera_group_name(camera_id)
    rgb_ids = set(main_handle[group_name][cam_key]["color"].keys())
    mask_ids = set(ann_handle["Mask"][cam_key]["mask"].keys())
    return rgb_ids & mask_ids


def choose_frames(
    args: argparse.Namespace,
    main_handle: h5py.File,
    ann_handle: h5py.File,
    camera_ids: list[str],
) -> list[str]:
    common_frame_ids: set[str] | None = None
    for camera_id in camera_ids:
        camera_frames = _frame_ids_for_camera(main_handle, ann_handle, camera_id)
        common_frame_ids = camera_frames if common_frame_ids is None else (common_frame_ids & camera_frames)
    ordered = dna.sort_numeric(common_frame_ids or [])
    if args.frames:
        requested = [str(int(frame)) for frame in args.frames]
        ordered_set = set(ordered)
        filtered = [frame for frame in requested if frame in ordered_set]
    else:
        start = max(0, int(args.frame_start))
        stop = None if int(args.frame_stop) < 0 else max(start, int(args.frame_stop))
        step = max(1, int(args.frame_step))
        filtered = []
        for frame in ordered:
            frame_num = int(frame)
            if frame_num < start:
                continue
            if stop is not None and frame_num >= stop:
                continue
            if ((frame_num - start) % step) != 0:
                continue
            filtered.append(frame)
    if int(args.max_frames) > 0:
        filtered = filtered[: int(args.max_frames)]
    return filtered


def _camera_dir_name(camera_id: str) -> str:
    return f"Camera_{dna.normalize_camera_id(camera_id)}"


def _frame_file_name(frame_id: str) -> str:
    return f"{int(frame_id):06d}.png"


def export_bridge(args: argparse.Namespace) -> Path:
    context = dna.build_context(Path(args.dataset_root), dna.SUBSET_NAME)
    output_root = Path(args.output_root).resolve()
    seq_root = output_root / args.seq
    seq_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dna_4k4d_bridge_") as temp_name:
        temp_dir = Path(temp_name)
        main_smc = dna.resolve_required_file(
            context,
            f"{dna.SUBSET_NAME}/main/{args.seq}.smc",
            temp_dir,
        )
        ann_smc = dna.resolve_required_file(
            context,
            f"{dna.SUBSET_NAME}/annotations/{args.seq}_annots.smc",
            temp_dir,
        )
        rgb_cams_smc, rgb_cams_source = dna.materialize_rgb_cams_smc(context, args.seq, temp_dir)
        if rgb_cams_smc is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams SMC for {args.seq}")
        camera_summary = dna.load_camera_summary(rgb_cams_smc)
        available_cameras = list(camera_summary["camera_ids"])
        camera_ids = choose_cameras(args, available_cameras)
        invalid = [camera for camera in camera_ids if camera not in available_cameras]
        if invalid:
            raise ValueError(f"Unknown camera ids for {args.seq}: {', '.join(invalid)}")

        exported = 0
        skipped = 0
        with h5py.File(main_smc, "r") as main_handle, h5py.File(ann_smc, "r") as ann_handle:
            frame_ids = choose_frames(args, main_handle, ann_handle, camera_ids)
            if not frame_ids:
                raise RuntimeError("No common frame ids matched the requested selection.")
            for camera_id in camera_ids:
                camera_dir = seq_root / _camera_dir_name(camera_id)
                mask_dir = seq_root / "mask" / _camera_dir_name(camera_id)
                camera_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)
                for frame_id in frame_ids:
                    file_name = _frame_file_name(frame_id)
                    rgb_path = camera_dir / file_name
                    mask_path = mask_dir / file_name
                    if not args.overwrite and rgb_path.exists() and mask_path.exists():
                        skipped += 2
                        continue
                    rgb_image = load_rgb_frame(main_handle, camera_id, frame_id)
                    mask_image = load_mask_frame(ann_handle, camera_id, frame_id)
                    rgb_image.save(rgb_path)
                    mask_image.save(mask_path)
                    exported += 2

        manifest = {
            "dataset_root": str(context.subset_roots[0] if context.subset_roots else context.dataset_path),
            "seq_id": args.seq,
            "output_seq_root": str(seq_root),
            "camera_ids": camera_ids,
            "frame_ids": [str(int(frame_id)) for frame_id in frame_ids],
            "camera_summary": camera_summary,
            "rgb_cams_source": rgb_cams_source,
            "main_smc": str(main_smc),
            "annotations_smc": str(ann_smc),
            "exported_file_count": int(exported),
            "skipped_file_count": int(skipped),
            "layout": {
                "image_dir_pattern": f"{args.seq}/Camera_XX/{_frame_file_name('0')}",
                "mask_dir_pattern": f"{args.seq}/mask/Camera_XX/{_frame_file_name('0')}",
            },
        }
        manifest_path = seq_root / "bridge_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Bridge manifest written to {manifest_path}")
        print(
            f"Exported bridge seq={args.seq} cameras={len(camera_ids)} "
            f"frames={len(frame_ids)} files_written={exported} files_skipped={skipped}"
        )
    return seq_root


def main() -> int:
    args = parse_args()
    export_bridge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
