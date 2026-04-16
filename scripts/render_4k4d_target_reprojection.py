import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render a target-view reprojection preview from an existing 4K4D VGGT npz."
    )
    ap.add_argument("--npz-path", required=True, help="Existing frame_XXXXXX.npz file")
    ap.add_argument("--seq-root", required=True, help="Bridged sequence root, e.g. out_vis/bridge_4k4d_med96/0012_11")
    ap.add_argument("--output-dir", required=True, help="Directory for reprojection outputs")
    ap.add_argument("--target-view", default="", help="Camera name or 0-based index; default is the first view")
    ap.add_argument("--conf-percentile", type=float, default=60.0, help="Keep points above this confidence percentile")
    ap.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold in [0,255]")
    ap.add_argument("--splat-radius", type=int, default=1, help="Square splat radius in pixels")
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
    ap.add_argument(
        "--render-original-resolution",
        action="store_true",
        help="Render the reprojection onto the original bridged image resolution instead of the VGGT forward resolution.",
    )
    ap.add_argument("--background", default="white", choices=["white", "black"], help="Canvas background color")
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


def _load_aligned_rgb(img_paths: list[str], target_size: int) -> np.ndarray:
    colors = [_preprocess_image_to_uint8(path, mode="crop", target_size=int(target_size)) for path in img_paths]
    return np.stack(colors, axis=0)


def _load_original_rgb(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
    return pointmap, conf, intrinsic, meta


def _resolve_target_index(target_view: str, cam_names: list[str]) -> int:
    token = str(target_view or "").strip()
    if token == "":
        return 0
    if token in cam_names:
        return cam_names.index(token)
    try:
        idx = int(token)
    except ValueError as exc:
        raise ValueError(f"unknown target_view={target_view!r}") from exc
    if idx < 0 or idx >= len(cam_names):
        raise ValueError(f"target_view index out of range: {idx}")
    return idx


def _filter_points(
    *,
    pointmap: np.ndarray,
    conf: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    valid = np.isfinite(pointmap).all(axis=-1) & np.isfinite(conf) & (conf > 1e-6) & mask
    conf_valid = conf[valid]
    if conf_valid.size <= 0:
        raise RuntimeError("no valid points survived mask/conf filtering")
    conf_thr = float(np.percentile(conf_valid, float(conf_percentile)))
    keep = valid & (conf >= conf_thr)
    return pointmap[keep], colors[keep], conf[keep], conf_thr


def _project_world_to_view(
    points_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(extrinsic_w2c[:3, :3], dtype=np.float32)
    t = np.asarray(extrinsic_w2c[:3, 3], dtype=np.float32)
    k = np.asarray(intrinsic[:3, :3], dtype=np.float32)

    cam = points_world @ r.T + t[None, :]
    z = cam[:, 2]
    u = k[0, 0] * (cam[:, 0] / (z + 1e-8)) + k[0, 2]
    v = k[1, 1] * (cam[:, 1] / (z + 1e-8)) + k[1, 2]
    return u, v, z


def _recover_original_resolution_intrinsic(
    intrinsic_processed: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
) -> np.ndarray:
    scale = float(processed_width) / float(max(original_width, 1))
    resized_height = int(round(float(original_height) * scale / 14.0) * 14.0)
    crop_top = max((resized_height - int(processed_height)) // 2, 0)

    intrinsic_original = np.asarray(intrinsic_processed, dtype=np.float32).copy()
    intrinsic_original[0, 0] /= max(scale, 1e-8)
    intrinsic_original[1, 1] /= max(scale, 1e-8)
    intrinsic_original[0, 2] /= max(scale, 1e-8)
    intrinsic_original[1, 2] = (intrinsic_original[1, 2] + float(crop_top)) / max(scale, 1e-8)
    return intrinsic_original


def _rasterize(
    *,
    points_world: np.ndarray,
    point_colors: np.ndarray,
    point_conf: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    splat_radius: int,
    background: str,
) -> tuple[np.ndarray, np.ndarray]:
    bg = 255 if background == "white" else 0
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=np.uint8)
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)

    u, v, z = _project_world_to_view(points_world, extrinsic_w2c=extrinsic_w2c, intrinsic=intrinsic)
    px = np.rint(u).astype(np.int32)
    py = np.rint(v).astype(np.int32)
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(z)
        & (z > 1e-6)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    if not np.any(valid):
        return canvas, coverage

    px = px[valid]
    py = py[valid]
    z = z[valid]
    point_colors = point_colors[valid]
    point_conf = point_conf[valid]

    # Prefer nearer points and, secondarily, higher confidence.
    order = np.lexsort((-point_conf.astype(np.float32), z.astype(np.float32)))
    px = px[order]
    py = py[order]
    z = z[order]
    point_colors = point_colors[order]

    for dy in range(-int(splat_radius), int(splat_radius) + 1):
        yy = py + dy
        valid_y = (yy >= 0) & (yy < height)
        if not np.any(valid_y):
            continue
        yy = yy[valid_y]
        xx_base = px[valid_y]
        zz = z[valid_y]
        cc = point_colors[valid_y]
        for dx in range(-int(splat_radius), int(splat_radius) + 1):
            xx = xx_base + dx
            valid_xy = (xx >= 0) & (xx < width)
            if not np.any(valid_xy):
                continue
            flat_y = yy[valid_xy]
            flat_x = xx[valid_xy]
            flat_z = zz[valid_xy]
            flat_c = cc[valid_xy]
            for idx in range(flat_z.shape[0]):
                y = int(flat_y[idx])
                x = int(flat_x[idx])
                z_here = float(flat_z[idx])
                if z_here >= float(depth_buf[y, x]):
                    continue
                depth_buf[y, x] = z_here
                canvas[y, x] = flat_c[idx]
                coverage[y, x] = 255
    return canvas, coverage


def _with_label(img: Image.Image, label: str) -> Image.Image:
    title_h = 40
    out = Image.new("RGB", (img.width, img.height + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((16, 12), label, fill=(0, 0, 0))
    out.paste(img, (0, title_h))
    return out


def _stack_contact_sheet(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    labelled = [_with_label(img, label) for label, img in images]
    try:
        total_w = sum(img.width for img in labelled)
        total_h = max(img.height for img in labelled)
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x = 0
        for img in labelled:
            canvas.paste(img, (x, 0))
            x += img.width
        canvas.save(out_path)
    finally:
        for img in labelled:
            img.close()


def main() -> None:
    args = _parse_args()
    npz_path = Path(args.npz_path).resolve()
    seq_root = Path(args.seq_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        pointmap = np.asarray(data["pointmap"], dtype=np.float32)
        conf = np.asarray(data["depth_conf"], dtype=np.float32)
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        depth = np.asarray(data["depth"], dtype=np.float32) if "depth" in data else None
        extrinsic = np.asarray(data["extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
        cam_names = [str(x) for x in data["cam_names"].tolist()]
        rel_paths = [str(x) for x in data["img_paths"].tolist()]
        unproject_impl = _as_scalar_string(data["unproject_impl"], default="legacy") if "unproject_impl" in data else "legacy"

    img_paths = []
    for rel_path in rel_paths:
        p = Path(rel_path)
        img_paths.append(str(p if p.is_absolute() else (seq_root / rel_path)))

    extra_meta = {
        "export_geometry_source": "saved_pointmap",
        "depth_upsample_factor": 1.0,
        "base_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "output_resolution": [int(pointmap.shape[1]), int(pointmap.shape[2])],
        "unproject_impl": str(unproject_impl),
    }
    if bool(args.rebuild_from_depth) or float(args.depth_upsample_factor) != 1.0:
        if depth is None:
            raise RuntimeError(f"depth bundle missing in npz: {npz_path}")
        pointmap, conf, intrinsic, extra_meta = _rebuild_from_depth_bundle(
            depth=depth,
            conf=conf,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            upsample_factor=float(args.depth_upsample_factor),
            unproject_impl=str(unproject_impl),
        )

    _, height, width, _ = pointmap.shape
    colors = _load_aligned_rgb(img_paths, target_size=width)
    mask = []
    for cam_name, img_path in zip(cam_names, img_paths):
        img_name = Path(img_path).name
        mask_path = seq_root / "mask" / cam_name / img_name
        mask.append(_load_mask(mask_path, width=width, height=height, threshold=int(args.mask_threshold)))
    mask = np.stack(mask, axis=0)

    points_world, point_colors, point_conf, conf_thr = _filter_points(
        pointmap=pointmap,
        conf=conf,
        colors=colors,
        mask=mask,
        conf_percentile=float(args.conf_percentile),
    )

    target_idx = _resolve_target_index(args.target_view, cam_names)
    target_cam = cam_names[target_idx]
    frame_id = int(npz_path.stem.split("_")[-1])
    render_height = height
    render_width = width
    render_intrinsic = intrinsic[target_idx]
    render_resolution_mode = "model"
    if bool(args.render_original_resolution):
        img_target = _load_original_rgb(img_paths[target_idx])
        render_width, render_height = img_target.size
        render_intrinsic = _recover_original_resolution_intrinsic(
            intrinsic[target_idx],
            original_width=render_width,
            original_height=render_height,
            processed_width=width,
            processed_height=height,
        )
        render_resolution_mode = "original"
    else:
        img_target = Image.fromarray(colors[target_idx], mode="RGB")

    render_all, cover_all = _rasterize(
        points_world=points_world,
        point_colors=point_colors,
        point_conf=point_conf,
        extrinsic_w2c=extrinsic[target_idx],
        intrinsic=render_intrinsic,
        height=render_height,
        width=render_width,
        splat_radius=int(args.splat_radius),
        background=args.background,
    )

    keep_sources = []
    offset = 0
    valid_all = np.isfinite(pointmap).all(axis=-1) & np.isfinite(conf) & (conf > 1e-6) & mask & (conf >= conf_thr)
    for view_idx in range(len(cam_names)):
        count = int(valid_all[view_idx].sum())
        if view_idx != target_idx and count > 0:
            keep_sources.append((offset, offset + count))
        offset += count
    if keep_sources:
        src_idx = np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in keep_sources], axis=0)
        points_src = points_world[src_idx]
        colors_src = point_colors[src_idx]
        conf_src = point_conf[src_idx]
    else:
        points_src = np.zeros((0, 3), dtype=np.float32)
        colors_src = np.zeros((0, 3), dtype=np.uint8)
        conf_src = np.zeros((0,), dtype=np.float32)

    render_sources, cover_sources = _rasterize(
        points_world=points_src,
        point_colors=colors_src,
        point_conf=conf_src,
        extrinsic_w2c=extrinsic[target_idx],
        intrinsic=render_intrinsic,
        height=render_height,
        width=render_width,
        splat_radius=int(args.splat_radius),
        background=args.background,
    )

    stem = f"frame_{frame_id:06d}_target_{target_cam}"
    gt_path = out_dir / f"{stem}_gt.png"
    all_path = out_dir / f"{stem}_reprojection_all_views.png"
    src_path = out_dir / f"{stem}_reprojection_source_only.png"
    sheet_path = out_dir / f"{stem}_reprojection_contact_sheet.png"
    meta_path = out_dir / f"{stem}_reprojection_meta.json"

    img_target.save(gt_path)
    Image.fromarray(render_all, mode="RGB").save(all_path)
    Image.fromarray(render_sources, mode="RGB").save(src_path)
    _stack_contact_sheet(
        [
            (f"Target GT | {target_cam}", img_target.copy()),
            (f"Reprojection | all {len(cam_names)} views", Image.fromarray(render_all, mode="RGB")),
            (f"Reprojection | source-only {max(len(cam_names) - 1, 0)} views", Image.fromarray(render_sources, mode="RGB")),
        ],
        sheet_path,
    )

    meta = {
        "seq": seq_root.name,
        "frame_id": frame_id,
        "target_cam": target_cam,
        "target_view_index": target_idx,
        "num_views": len(cam_names),
        "cam_names": cam_names,
        "conf_percentile": float(args.conf_percentile),
        "conf_threshold": conf_thr,
        "points_after_filter": int(points_world.shape[0]),
        "points_after_filter_source_only": int(points_src.shape[0]),
        "coverage_all_views": float(np.mean(cover_all > 0)),
        "coverage_source_only": float(np.mean(cover_sources > 0)),
        "output_resolution": [int(render_height), int(render_width)],
        "geometry_resolution": [int(height), int(width)],
        "render_resolution_mode": render_resolution_mode,
        "gt_path": str(gt_path),
        "reprojection_all_views_path": str(all_path),
        "reprojection_source_only_path": str(src_path),
        "contact_sheet_path": str(sheet_path),
    }
    meta.update(extra_meta)
    meta["output_resolution"] = [int(render_height), int(render_width)]
    meta["geometry_resolution"] = [int(height), int(width)]
    meta["render_resolution_mode"] = render_resolution_mode
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
