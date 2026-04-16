import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zju_multiview import ZJUMocapSeq


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export fused 4K4D point clouds and preview renders from a bridged sequence.")
    ap.add_argument("--bridge-root", default="", help="Root containing bridged sequences, e.g. out_vis/bridge_4k4d_med96")
    ap.add_argument("--seq", default="", help="Sequence id, e.g. 0012_11")
    ap.add_argument("--seq-root", default="", help="Direct sequence root; used by npz export mode")
    ap.add_argument("--output-dir", required=True, help="Directory to write PLY and preview outputs")
    ap.add_argument("--frames", type=int, nargs="+", default=None, help="Frame indices in bridged sequence order")
    ap.add_argument("--npz-paths", nargs="*", default=None, help="Existing npz files to export without rerunning VGGT")
    ap.add_argument("--cam-names", nargs="*", default=None, help="Optional camera names override")
    ap.add_argument("--device", default="cpu", help="VGGT device, e.g. cpu or cuda")
    ap.add_argument("--ckpt", default="model.pt", help="Path to VGGT checkpoint")
    ap.add_argument("--pointmap-source", default="depth_unproject", choices=["depth_unproject", "point_head", "auto"])
    ap.add_argument(
        "--rebuild-from-depth",
        action="store_true",
        help="Rebuild world points from stored depth/intrinsic/extrinsic instead of replaying saved pointmap.",
    )
    ap.add_argument(
        "--depth-upsample-factor",
        type=float,
        default=1.0,
        help="Optional scale factor applied to stored depth/conf before depth unprojection. Example: 2.0 for 518->1036.",
    )
    ap.add_argument("--conf-percentile", type=float, default=75.0, help="Keep points above this confidence percentile")
    ap.add_argument("--max-points", type=int, default=120000, help="Max points to save in each PLY")
    ap.add_argument("--preview-points", type=int, default=40000, help="Max points to draw in each preview")
    ap.add_argument("--preview-tile-size", type=int, default=900, help="Per-view tile size for contact-sheet preview rendering")
    ap.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold in [0,255]")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for point subsampling")
    return ap.parse_args()


def _load_mask(mask_path: Path, width: int, height: int, threshold: int) -> np.ndarray:
    if not mask_path.is_file():
        return np.ones((height, width), dtype=bool)
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask) > int(threshold)


def _preprocess_image_to_uint8(image_path: str, mode: str = "crop", target_size: int = 518) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    width, height = img.size
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        arr = arr[start_y : start_y + target_size, :, :]

    if mode == "pad":
        h_pad = target_size - arr.shape[0]
        w_pad = target_size - arr.shape[1]
        if h_pad > 0 or w_pad > 0:
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            arr = np.pad(
                arr,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )
    return arr


def _resize_float_map(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    img = img.resize((int(out_w), int(out_h)), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _resolve_pixel_center_offset(unproject_impl: str | None) -> float:
    mode = str(unproject_impl or "legacy").strip().lower()
    if mode in {"upstream433", "pixel_center", "center_0.5"}:
        return 0.5
    return 0.0


def _scale_intrinsic(intrinsic: np.ndarray, scale_x: float, scale_y: float, pixel_center_offset: float) -> np.ndarray:
    k = np.asarray(intrinsic, dtype=np.float32).copy()
    k[0, 0] *= float(scale_x)
    k[1, 1] *= float(scale_y)
    if abs(float(pixel_center_offset) - 0.5) < 1e-6:
        k[0, 2] = (k[0, 2] + 0.5) * float(scale_x) - 0.5
        k[1, 2] = (k[1, 2] + 0.5) * float(scale_y) - 0.5
    else:
        k[0, 2] *= float(scale_x)
        k[1, 2] *= float(scale_y)
    return k


def _unproject_depth_to_world(
    depth_map: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    pixel_center_offset: float,
) -> np.ndarray:
    height, width = depth_map.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32) + float(pixel_center_offset),
        np.arange(width, dtype=np.float32) + float(pixel_center_offset),
        indexing="ij",
    )
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = np.asarray(depth_map, dtype=np.float32)
    x_cam = (xs - cx) * z / max(fx, 1e-8)
    y_cam = (ys - cy) * z / max(fy, 1e-8)
    cam = np.stack((x_cam, y_cam, z), axis=-1)

    r = np.asarray(extrinsic_w2c[:3, :3], dtype=np.float32)
    t = np.asarray(extrinsic_w2c[:3, 3], dtype=np.float32)
    world = np.einsum("ij,hwj->hwi", r.T, cam - t[None, None, :])
    return world.astype(np.float32, copy=False)


def _as_scalar_string(value: object, default: str = "legacy") -> str:
    if value is None:
        return str(default)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.tolist())
        if value.size == 1:
            return str(value.reshape(-1)[0])
        return str(default)
    return str(value)


def _rebuild_from_depth_bundle(
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    upsample_factor: float,
    unproject_impl: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    views, base_h, base_w = depth.shape
    scale = max(float(upsample_factor), 1.0)
    out_h = int(round(base_h * scale))
    out_w = int(round(base_w * scale))
    pixel_center_offset = _resolve_pixel_center_offset(unproject_impl)
    scale_x = float(out_w) / float(base_w)
    scale_y = float(out_h) / float(base_h)

    if out_h != base_h or out_w != base_w:
        depth = np.stack([_resize_float_map(depth[i], out_w=out_w, out_h=out_h) for i in range(views)], axis=0)
        conf = np.stack([_resize_float_map(conf[i], out_w=out_w, out_h=out_h) for i in range(views)], axis=0)
        intrinsic = np.stack(
            [
                _scale_intrinsic(intrinsic[i], scale_x=scale_x, scale_y=scale_y, pixel_center_offset=pixel_center_offset)
                for i in range(views)
            ],
            axis=0,
        )

    pointmap = np.stack(
        [
            _unproject_depth_to_world(
                depth_map=depth[i],
                extrinsic_w2c=extrinsic[i],
                intrinsic=intrinsic[i],
                pixel_center_offset=pixel_center_offset,
            )
            for i in range(views)
        ],
        axis=0,
    )
    meta = {
        "export_geometry_source": "depth_unproject_rebuilt",
        "depth_upsample_factor": scale,
        "base_resolution": [int(base_h), int(base_w)],
        "output_resolution": [int(out_h), int(out_w)],
        "unproject_impl": str(unproject_impl),
    }
    return pointmap, conf, meta


def _sample_rows(points: np.ndarray, colors: np.ndarray, keep: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= keep:
        return points, colors
    ids = list(range(points.shape[0]))
    rng.shuffle(ids)
    ids = np.asarray(ids[:keep], dtype=np.int64)
    return points[ids], colors[ids]


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        vertex.tofile(f)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = np.percentile(points, 5.0, axis=0)
    maxs = np.percentile(points, 95.0, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(float(elev_deg))
    azim = np.deg2rad(float(azim_deg))
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]], dtype=np.float32)
    rot_z = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_x @ rot_z


def _render_view(points: np.ndarray, colors: np.ndarray, elev_deg: float, azim_deg: float, size: int = 900) -> Image.Image:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    rot = _rotation_matrix(elev_deg, azim_deg)
    xyz = points @ rot.T

    xy = xyz[:, :2]
    z = xyz[:, 2]
    mins = np.percentile(xy, 2.0, axis=0)
    maxs = np.percentile(xy, 98.0, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1e-3)

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    xy_norm = (xy - center[None, :]) / radius
    px = np.clip(((xy_norm[:, 0] * 0.5) + 0.5) * (size - 1), 0, size - 1).astype(np.int32)
    py = np.clip((1.0 - ((xy_norm[:, 1] * 0.5) + 0.5)) * (size - 1), 0, size - 1).astype(np.int32)

    order = np.argsort(z)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx = np.clip(px[order] + dx, 0, size - 1)
        yy = np.clip(py[order] + dy, 0, size - 1)
        canvas[yy, xx] = colors[order]
    return Image.fromarray(canvas, mode="RGB")


def _render_contact_sheet(path: Path, points: np.ndarray, colors: np.ndarray, title: str, tile_size: int = 900) -> None:
    views = [
        ("View A", 18, 35),
        ("View B", 15, 125),
        ("View C", 72, 25),
    ]
    tiles = []
    for label, elev, azim in views:
        tile = _render_view(points=points, colors=colors, elev_deg=elev, azim_deg=azim, size=int(tile_size))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((18, 18, 180, 68), fill=(255, 255, 255))
        draw.text((30, 30), label, fill=(0, 0, 0))
        tiles.append(tile)

    title_h = 72
    total_w = sum(tile.width for tile in tiles)
    total_h = title_h + max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), title, fill=(0, 0, 0))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, title_h))
        x += tile.width
        tile.close()
    canvas.save(path)


def _make_overview(out_path: Path, sheet_paths: list[Path]) -> None:
    if not sheet_paths:
        return
    images = [Image.open(path).convert("RGB") for path in sheet_paths]
    try:
        target_h = max(img.height for img in images)
        resized = []
        for img in images:
            new_w = max(1, int(round(img.width * (target_h / img.height))))
            resized.append(img.resize((new_w, target_h), Image.Resampling.BICUBIC))
        total_w = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += img.width
        canvas.save(out_path)
    finally:
        for img in images:
            img.close()


def _load_colors_from_paths(img_paths: list[str], target_size: int) -> np.ndarray:
    colors = [_preprocess_image_to_uint8(path, mode="crop", target_size=int(target_size)) for path in img_paths]
    return np.stack(colors, axis=0)


def _export_pred_bundle(
    *,
    seq_name: str,
    seq_root: Path,
    frame_idx: int | None,
    frame_id: int,
    cam_names: list[str],
    img_paths: list[str],
    pointmap: np.ndarray,
    conf: np.ndarray,
    out_dir: Path,
    conf_percentile: float,
    max_points: int,
    preview_points: int,
    preview_tile_size: int,
    mask_threshold: int,
    rng: random.Random,
    extra_meta: dict | None = None,
) -> dict:
    _, height, width, _ = pointmap.shape
    colors = _load_colors_from_paths(img_paths, target_size=width)
    _, height, width, _ = pointmap.shape
    mask_stack = []
    for cam_name, img_path in zip(cam_names, img_paths):
        img_name = Path(img_path).name
        mask_path = seq_root / "mask" / cam_name / img_name
        mask_stack.append(_load_mask(mask_path, width=width, height=height, threshold=mask_threshold))
    mask = np.stack(mask_stack, axis=0)

    valid = np.isfinite(pointmap).all(axis=-1) & np.isfinite(conf) & (conf > 1e-6) & mask
    conf_valid = conf[valid]
    if conf_valid.size <= 0:
        raise RuntimeError(f"no valid points survived filtering for frame_id={frame_id}")
    conf_thr = float(np.percentile(conf_valid, conf_percentile))
    keep = valid & (conf >= conf_thr)

    points = pointmap[keep]
    point_colors = colors[keep]
    points, point_colors = _sample_rows(points, point_colors, keep=max_points, rng=rng)

    centered = points - np.median(points, axis=0, keepdims=True)
    preview_pts, preview_colors = _sample_rows(centered, point_colors, keep=preview_points, rng=rng)

    stem = f"frame_{frame_id:06d}"
    ply_path = out_dir / f"{stem}_fused_pointcloud.ply"
    png_path = out_dir / f"{stem}_preview_contact_sheet.png"
    meta_path = out_dir / f"{stem}_preview_meta.json"

    _write_binary_ply(ply_path, points=points, colors=point_colors)
    _render_contact_sheet(
        png_path,
        points=preview_pts,
        colors=preview_colors,
        title=f"{seq_name} | frame {frame_id:06d} | {len(cam_names)}-view fused point cloud",
        tile_size=int(preview_tile_size),
    )

    meta = {
        "seq": seq_name,
        "frame_idx": frame_idx,
        "frame_id": frame_id,
        "num_views": len(cam_names),
        "cam_names": cam_names,
        "conf_percentile": conf_percentile,
        "conf_threshold": conf_thr,
        "points_saved": int(points.shape[0]),
        "preview_points": int(preview_pts.shape[0]),
        "preview_tile_size": int(preview_tile_size),
        "ply_path": str(ply_path),
        "preview_path": str(png_path),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _export_frame(
    *,
    teacher,
    seq: ZJUMocapSeq,
    seq_root: Path,
    frame_idx: int,
    out_dir: Path,
    conf_percentile: float,
    max_points: int,
    preview_points: int,
    preview_tile_size: int,
    mask_threshold: int,
    rng: random.Random,
    rebuild_from_depth: bool,
    depth_upsample_factor: float,
) -> dict:
    frame_id = int(seq.get_frame_id(frame_idx))
    cam2path = seq.get_frame_paths(frame_idx)
    cam_names = sorted(cam2path.keys())
    img_paths = [cam2path[name] for name in cam_names]

    prepared = teacher.prepare_batch_inputs([img_paths])
    pred = teacher.forward_prepared_batch(prepared)[0]
    pointmap = pred["pointmap"].detach().cpu().numpy()
    conf = pred["depth_conf"].detach().cpu().numpy()
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    extra_meta = {
        "export_geometry_source": "saved_pointmap",
        "depth_upsample_factor": 1.0,
        "base_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "output_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "unproject_impl": str(pred.get("unproject_impl", "legacy")),
    }
    if rebuild_from_depth or float(depth_upsample_factor) != 1.0:
        pointmap, conf, extra_meta = _rebuild_from_depth_bundle(
            depth=pred["depth"].detach().cpu().numpy(),
            conf=pred["depth_conf"].detach().cpu().numpy(),
            extrinsic=pred["extrinsic"].detach().cpu().numpy(),
            intrinsic=pred["intrinsic"].detach().cpu().numpy(),
            upsample_factor=float(depth_upsample_factor),
            unproject_impl=str(pred.get("unproject_impl", "legacy")),
        )
    return _export_pred_bundle(
        seq_name=seq_root.name,
        seq_root=seq_root,
        frame_idx=frame_idx,
        frame_id=frame_id,
        cam_names=cam_names,
        img_paths=img_paths,
        pointmap=pointmap,
        conf=conf,
        out_dir=out_dir,
        conf_percentile=conf_percentile,
        max_points=max_points,
        preview_points=preview_points,
        preview_tile_size=preview_tile_size,
        mask_threshold=mask_threshold,
        rng=rng,
        extra_meta=extra_meta,
    )

def _export_npz(
    *,
    npz_path: Path,
    seq_root: Path,
    out_dir: Path,
    conf_percentile: float,
    max_points: int,
    preview_points: int,
    preview_tile_size: int,
    mask_threshold: int,
    rng: random.Random,
    rebuild_from_depth: bool,
    depth_upsample_factor: float,
) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        pointmap = np.asarray(data["pointmap"])
        conf = np.asarray(data["depth_conf"])
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        depth = np.asarray(data["depth"]) if "depth" in data.files else None
        extrinsic = np.asarray(data["extrinsic"]) if "extrinsic" in data.files else None
        intrinsic = np.asarray(data["intrinsic"]) if "intrinsic" in data.files else None
        unproject_impl = _as_scalar_string(data["unproject_impl"], default="legacy") if "unproject_impl" in data.files else "legacy"
        cam_names = [str(x) for x in data["cam_names"].tolist()]
        rel_paths = [str(x) for x in data["img_paths"].tolist()]

    img_paths = []
    for rel_path in rel_paths:
        p = Path(rel_path)
        img_paths.append(str(p if p.is_absolute() else (seq_root / rel_path)))

    name_match = npz_path.stem
    frame_id = int(name_match.split("_")[-1])
    extra_meta = {
        "export_geometry_source": "saved_pointmap",
        "depth_upsample_factor": 1.0,
        "base_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "output_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "unproject_impl": str(unproject_impl),
    }
    if rebuild_from_depth or float(depth_upsample_factor) != 1.0:
        if depth is None or extrinsic is None or intrinsic is None:
            raise RuntimeError(f"npz missing depth/extrinsic/intrinsic required for depth rebuild: {npz_path}")
        pointmap, conf, extra_meta = _rebuild_from_depth_bundle(
            depth=depth,
            conf=conf,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            upsample_factor=float(depth_upsample_factor),
            unproject_impl=unproject_impl,
        )
    return _export_pred_bundle(
        seq_name=seq_root.name,
        seq_root=seq_root,
        frame_idx=None,
        frame_id=frame_id,
        cam_names=cam_names,
        img_paths=img_paths,
        pointmap=pointmap,
        conf=conf,
        out_dir=out_dir,
        conf_percentile=conf_percentile,
        max_points=max_points,
        preview_points=preview_points,
        preview_tile_size=preview_tile_size,
        mask_threshold=mask_threshold,
        rng=rng,
        extra_meta=extra_meta,
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    results = []
    sheet_paths: list[Path] = []

    if args.npz_paths:
        seq_root = Path(args.seq_root).resolve() if args.seq_root else Path(args.bridge_root).resolve() / args.seq
        if not seq_root.is_dir():
            raise RuntimeError(f"seq_root not found for npz export mode: {seq_root}")
        for raw_path in args.npz_paths:
            meta = _export_npz(
                npz_path=Path(raw_path).resolve(),
                seq_root=seq_root,
                out_dir=out_dir,
                conf_percentile=float(args.conf_percentile),
                max_points=int(args.max_points),
                preview_points=int(args.preview_points),
                preview_tile_size=int(args.preview_tile_size),
                mask_threshold=int(args.mask_threshold),
                rng=rng,
                rebuild_from_depth=bool(args.rebuild_from_depth),
                depth_upsample_factor=float(args.depth_upsample_factor),
            )
            results.append(meta)
            sheet_paths.append(Path(meta["preview_path"]))
        seq_name = seq_root.name
        view_setup = f"{results[0]['num_views']}-view" if results else "unknown"
        frame_list = [int(item["frame_id"]) for item in results]
    else:
        if not args.bridge_root or not args.seq or not args.frames:
            raise RuntimeError("bridge export mode requires --bridge-root, --seq, and --frames")
        from vggt_geom import VGGTGeomTeacher

        bridge_root = Path(args.bridge_root).resolve()
        seq_root = bridge_root / args.seq
        seq = ZJUMocapSeq(str(seq_root), cam_names=args.cam_names if args.cam_names else None)
        teacher = VGGTGeomTeacher(
            ckpt_path=str(Path(args.ckpt).resolve()),
            device=args.device,
            pointmap_source=args.pointmap_source,
            amp=False,
        )
        try:
            for frame_idx in args.frames:
                meta = _export_frame(
                    teacher=teacher,
                    seq=seq,
                    seq_root=seq_root,
                    frame_idx=int(frame_idx),
                    out_dir=out_dir,
                    conf_percentile=float(args.conf_percentile),
                    max_points=int(args.max_points),
                    preview_points=int(args.preview_points),
                    preview_tile_size=int(args.preview_tile_size),
                    mask_threshold=int(args.mask_threshold),
                    rng=rng,
                    rebuild_from_depth=bool(args.rebuild_from_depth),
                    depth_upsample_factor=float(args.depth_upsample_factor),
                )
                results.append(meta)
                sheet_paths.append(Path(meta["preview_path"]))
        finally:
            executor = getattr(teacher, "_image_executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                teacher._image_executor = None
        seq_name = args.seq
        view_setup = f"{len(seq.cam_names)}-view"
        frame_list = [int(x) for x in args.frames]

    overview_path = out_dir / f"{seq_name}_overview_contact_sheet.png"
    _make_overview(overview_path, sheet_paths)
    summary = {
        "seq": seq_name,
        "view_setup": view_setup,
        "frames": frame_list,
        "device": args.device,
        "pointmap_source": args.pointmap_source,
        "rebuild_from_depth": bool(args.rebuild_from_depth),
        "depth_upsample_factor": float(args.depth_upsample_factor),
        "outputs": results,
        "overview_contact_sheet": str(overview_path),
    }
    summary_path = out_dir / f"{seq_name}_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
