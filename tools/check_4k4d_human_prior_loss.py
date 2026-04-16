import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_vggt_pseudo import (  # noqa: E402
    PseudoGeomDataset,
    _apply_region_weight_boost,
    _build_conf_weight,
    _cam_to_world_point_map_torch,
    _resolve_human_prior_region_mask,
    _resolve_point_frame_auto,
    _robust_abs,
    _safe_resize_like,
    _sample_to_tensors,
    _to_depth01,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Local human_prior loss verifier for 4K4D geom caches."
    )
    ap.add_argument("--seq-root", required=True, help="Local bridged sequence root, e.g. out_vis/bridge_4k4d_med96/0012_11")
    ap.add_argument("--geom-root", required=True, help="Local geom cache dir under seq-root or elsewhere")
    ap.add_argument("--human-prior-subdir", default="human_prior", help="Sidecar subdir under seq-root")
    ap.add_argument("--frames", nargs="*", type=int, default=None, help="Optional explicit frame ids")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap after sorting selected frames")
    ap.add_argument("--conf-weight-thr", type=float, default=0.0)
    ap.add_argument("--conf-weight-gamma", type=float, default=1.0)
    ap.add_argument("--human-prior-weight-boost", type=float, default=1.50)
    ap.add_argument("--human-prior-weight-region", type=str, default="body")
    ap.add_argument("--human-prior-loss-region", type=str, default="head_face")
    ap.add_argument("--human-prior-complete-weight", type=float, default=0.35)
    ap.add_argument("--human-prior-complete-region", type=str, default="body")
    ap.add_argument("--human-prior-head-fallback-top-ratio", type=float, default=0.32)
    ap.add_argument("--human-prior-face-fallback-top-ratio", type=float, default=0.18)
    ap.add_argument("--robust-l1-eps", type=float, default=0.0)
    ap.add_argument("--use-fg-mask", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--fg-mask-source", type=str, default="mask", choices=["auto", "mask", "mask_cihp"])
    return ap.parse_args()


def _frame_id_from_geom_path(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def _resolve_world_pointmap(point_map: torch.Tensor, extrinsic_w2c: torch.Tensor, intrinsic: torch.Tensor, declared_frame: str) -> tuple[torch.Tensor, str, Dict[str, float]]:
    frame = str(declared_frame or "").strip().lower()
    if frame == "world":
        return point_map, "world", {}
    if frame == "camera":
        return _cam_to_world_point_map_torch(point_map, extrinsic_w2c), "camera", {}
    resolved, info = _resolve_point_frame_auto(point_map, extrinsic_w2c, intrinsic)
    if resolved == "camera":
        return _cam_to_world_point_map_torch(point_map, extrinsic_w2c), "camera", info
    return point_map, "world", info


def _select_indices(dataset: PseudoGeomDataset, frames: Optional[Sequence[int]], max_frames: int) -> list[int]:
    indexed = []
    wanted = {int(x) for x in frames} if frames else None
    for idx, (_, geom_path) in enumerate(dataset.items):
        frame_id = _frame_id_from_geom_path(geom_path)
        if wanted is not None and frame_id not in wanted:
            continue
        indexed.append(idx)
    if max_frames > 0:
        indexed = indexed[: int(max_frames)]
    if not indexed:
        raise RuntimeError("no dataset items selected for verification")
    return indexed


def _rewrite_sample_img_paths_local(sample: Any, seq_root: Path, frame_id: int) -> None:
    frame_name = f"{int(frame_id):06d}.png"
    rewritten: list[str] = []
    for cam_name in sample.cam_names:
        local_path = (seq_root / cam_name / frame_name).resolve()
        if not local_path.is_file():
            raise RuntimeError(f"local bridged RGB missing: {local_path}")
        rewritten.append(str(local_path))
    sample.img_paths = rewritten


def main() -> int:
    args = _parse_args()
    seq_root = Path(args.seq_root).expanduser().resolve()
    geom_root = Path(args.geom_root).expanduser().resolve()
    if not geom_root.is_dir():
        raise RuntimeError(f"geom root not found: {geom_root}")
    zju_root = seq_root.parent
    seq_name = seq_root.name
    geom_subdir = geom_root.name
    dataset = PseudoGeomDataset(
        zju_root=str(zju_root),
        seq_names=[seq_name],
        cam_names=None,
        max_frames=0,
        geom_subdir=geom_subdir,
        human_prior_enable=True,
        human_prior_subdir=str(args.human_prior_subdir),
        human_prior_strict=True,
    )
    selected_indices = _select_indices(dataset=dataset, frames=args.frames, max_frames=int(args.max_frames))
    rows: list[dict[str, Any]] = []

    for idx in selected_indices:
        sample = dataset[idx]
        frame_id = _frame_id_from_geom_path(dataset.items[idx][1])
        _rewrite_sample_img_paths_local(sample=sample, seq_root=seq_root, frame_id=frame_id)
        (
            _imgs,
            depth_tgt,
            conf_tgt,
            point_tgt,
            extrinsic_tgt,
            intrinsic_tgt,
            _point_source,
            point_frame,
            fg_mask_tgt,
            fg_mask_source,
            human_prior_point,
            human_prior_valid_mask,
            human_prior_body_mask,
            human_prior_head_mask,
            human_prior_face_mask,
            human_prior_source,
            human_prior_pointmap_frame,
        ) = _sample_to_tensors(
            sample=sample,
            device="cpu",
            use_fg_mask=(str(args.use_fg_mask).lower() == "on"),
            fg_mask_source=str(args.fg_mask_source),
            use_human_prior=True,
        )

        point_world, point_world_frame, point_world_info = _resolve_world_pointmap(
            point_map=point_tgt,
            extrinsic_w2c=extrinsic_tgt,
            intrinsic=intrinsic_tgt,
            declared_frame=point_frame,
        )
        if human_prior_point is None:
            raise RuntimeError(f"human prior missing for frame={frame_id:06d}")
        human_prior_world, prior_world_frame, prior_world_info = _resolve_world_pointmap(
            point_map=human_prior_point,
            extrinsic_w2c=extrinsic_tgt,
            intrinsic=intrinsic_tgt,
            declared_frame=human_prior_pointmap_frame,
        )

        conf_hw = point_world.shape[-3:-1]
        if fg_mask_tgt is not None and fg_mask_tgt.shape[-2:] != conf_hw:
            mt = fg_mask_tgt.reshape(-1, 1, fg_mask_tgt.shape[-2], fg_mask_tgt.shape[-1])
            mt = _safe_resize_like(mt, conf_hw, mode="nearest")
            fg_mask_tgt = mt.reshape(fg_mask_tgt.shape[0], fg_mask_tgt.shape[1], mt.shape[-2], mt.shape[-1])
        if human_prior_valid_mask is not None and human_prior_valid_mask.shape[-2:] != conf_hw:
            hm = human_prior_valid_mask.reshape(-1, 1, human_prior_valid_mask.shape[-2], human_prior_valid_mask.shape[-1])
            hm = _safe_resize_like(hm, conf_hw, mode="nearest")
            human_prior_valid_mask = hm.reshape(human_prior_valid_mask.shape[0], human_prior_valid_mask.shape[1], hm.shape[-2], hm.shape[-1])
        if human_prior_body_mask is not None and human_prior_body_mask.shape[-2:] != conf_hw:
            hm = human_prior_body_mask.reshape(-1, 1, human_prior_body_mask.shape[-2], human_prior_body_mask.shape[-1])
            hm = _safe_resize_like(hm, conf_hw, mode="nearest")
            human_prior_body_mask = hm.reshape(human_prior_body_mask.shape[0], human_prior_body_mask.shape[1], hm.shape[-2], hm.shape[-1])
        if human_prior_head_mask is not None and human_prior_head_mask.shape[-2:] != conf_hw:
            hm = human_prior_head_mask.reshape(-1, 1, human_prior_head_mask.shape[-2], human_prior_head_mask.shape[-1])
            hm = _safe_resize_like(hm, conf_hw, mode="nearest")
            human_prior_head_mask = hm.reshape(human_prior_head_mask.shape[0], human_prior_head_mask.shape[1], hm.shape[-2], hm.shape[-1])
        if human_prior_face_mask is not None and human_prior_face_mask.shape[-2:] != conf_hw:
            hm = human_prior_face_mask.reshape(-1, 1, human_prior_face_mask.shape[-2], human_prior_face_mask.shape[-1])
            hm = _safe_resize_like(hm, conf_hw, mode="nearest")
            human_prior_face_mask = hm.reshape(human_prior_face_mask.shape[0], human_prior_face_mask.shape[1], hm.shape[-2], hm.shape[-1])

        conf_tgt = _to_depth01(conf_tgt)
        valid_all = (depth_tgt[..., 0] > 1e-6).float()
        valid = valid_all
        if fg_mask_tgt is not None and str(args.use_fg_mask).lower() == "on":
            valid = valid_all * fg_mask_tgt.clamp(0.0, 1.0)
        supervision_valid = valid

        if human_prior_valid_mask is None:
            human_prior_valid_mask = torch.isfinite(human_prior_world).all(dim=-1).float()
        if human_prior_body_mask is None:
            human_prior_body_mask = human_prior_valid_mask.detach().clone()

        human_prior_weight_mask = _resolve_human_prior_region_mask(
            region_mode=str(args.human_prior_weight_region),
            valid_mask01=human_prior_valid_mask,
            body_mask01=human_prior_body_mask,
            head_mask01=human_prior_head_mask,
            face_mask01=human_prior_face_mask,
            head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
            face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
        )
        human_prior_loss_mask = _resolve_human_prior_region_mask(
            region_mode=str(args.human_prior_loss_region),
            valid_mask01=human_prior_valid_mask,
            body_mask01=human_prior_body_mask,
            head_mask01=human_prior_head_mask,
            face_mask01=human_prior_face_mask,
            head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
            face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
        )
        human_prior_complete_mask = _resolve_human_prior_region_mask(
            region_mode=str(args.human_prior_complete_region),
            valid_mask01=human_prior_valid_mask,
            body_mask01=human_prior_body_mask,
            head_mask01=human_prior_head_mask,
            face_mask01=human_prior_face_mask,
            head_fallback_top_ratio=float(args.human_prior_head_fallback_top_ratio),
            face_fallback_top_ratio=float(args.human_prior_face_fallback_top_ratio),
        )
        if human_prior_weight_mask is not None:
            human_prior_weight_mask = (human_prior_weight_mask * human_prior_valid_mask).clamp(0.0, 1.0)
        if human_prior_loss_mask is not None:
            human_prior_loss_mask = (human_prior_loss_mask * human_prior_valid_mask).clamp(0.0, 1.0)
        if human_prior_complete_mask is not None:
            human_prior_complete_mask = (human_prior_complete_mask * human_prior_valid_mask).clamp(0.0, 1.0)
            human_prior_complete_mask = (human_prior_complete_mask * (1.0 - valid_all)).clamp(0.0, 1.0)

        conf_weight = _build_conf_weight(
            conf01=conf_tgt,
            valid01=supervision_valid,
            thr=float(args.conf_weight_thr),
            gamma=float(args.conf_weight_gamma),
        )
        weight_base, _ = _apply_region_weight_boost(
            base_weight01=conf_weight,
            region_mask01=human_prior_weight_mask,
            boost=float(args.human_prior_weight_boost),
            info_prefix="human_prior_weight",
            fg_stats_mask01=(fg_mask_tgt if fg_mask_tgt is not None else None),
        )
        human_prior_finite_mask = torch.isfinite(human_prior_world).all(dim=-1).float()
        point_world_finite_mask = torch.isfinite(point_world).all(dim=-1).float()
        w_point_prior = (weight_base * human_prior_loss_mask * human_prior_finite_mask * point_world_finite_mask).detach()
        if human_prior_complete_mask is not None and float(args.human_prior_complete_weight) > 0.0:
            completion_weight_map = (human_prior_complete_mask * float(args.human_prior_complete_weight)).detach()
            w_point_prior = torch.maximum(w_point_prior, completion_weight_map)
        denom_point_prior = float((w_point_prior.sum() + 1e-6).item())
        point_world_safe = torch.where(
            point_world_finite_mask[..., None] > 0.5,
            point_world,
            torch.zeros_like(point_world),
        )
        human_prior_world_safe = torch.where(
            human_prior_finite_mask[..., None] > 0.5,
            human_prior_world,
            torch.zeros_like(human_prior_world),
        )
        point_prior_abs = _robust_abs(
            point_world_safe - human_prior_world_safe,
            float(args.robust_l1_eps),
        ).mean(dim=-1)
        loss_point_prior = float(((point_prior_abs * w_point_prior).sum() / (w_point_prior.sum() + 1e-6)).item())

        row = {
            "frame_id": int(frame_id),
            "cam_count": int(len(sample.cam_names)),
            "prior_source": str(human_prior_source or ""),
            "fg_mask_source": str(fg_mask_source or ""),
            "point_world_frame": str(point_world_frame),
            "prior_world_frame": str(prior_world_frame),
            "point_frame_err_world": float(point_world_info.get("point_frame_err_world", 0.0)),
            "point_frame_err_camera": float(point_world_info.get("point_frame_err_camera", 0.0)),
            "prior_frame_err_world": float(prior_world_info.get("point_frame_err_world", 0.0)),
            "prior_frame_err_camera": float(prior_world_info.get("point_frame_err_camera", 0.0)),
            "supervision_valid_cover": float(supervision_valid.mean().item()),
            "prior_valid_cover": float(human_prior_valid_mask.mean().item()),
            "prior_body_cover": float(human_prior_body_mask.mean().item()),
            "prior_head_cover": float(human_prior_head_mask.mean().item()) if human_prior_head_mask is not None else 0.0,
            "prior_face_cover": float(human_prior_face_mask.mean().item()) if human_prior_face_mask is not None else 0.0,
            "prior_loss_cover": float(human_prior_loss_mask.mean().item()) if human_prior_loss_mask is not None else 0.0,
            "prior_complete_cover": float(human_prior_complete_mask.mean().item()) if human_prior_complete_mask is not None else 0.0,
            "prior_loss_weight_nonzero": float((w_point_prior > 0.0).float().mean().item()),
            "prior_loss_weight_mean": float(w_point_prior.mean().item()),
            "prior_loss_proxy": float(loss_point_prior),
            "prior_loss_proxy_active": 1.0 if denom_point_prior > 1e-6 else 0.0,
        }
        rows.append(row)
        print(
            f"[human-prior-check] frame={frame_id:06d} "
            f"weight_nonzero={row['prior_loss_weight_nonzero']:.6f} "
            f"loss_proxy={row['prior_loss_proxy']:.6f} "
            f"prior_cover={row['prior_valid_cover']:.6f}"
        )

    out_root = (seq_root / str(args.human_prior_subdir)).resolve()
    summary_json = out_root / "human_prior_loss_check_summary.json"
    summary_csv = out_root / "human_prior_loss_check_summary.csv"
    blob = {
        "seq_root": str(seq_root),
        "geom_root": str(geom_root),
        "human_prior_subdir": str(args.human_prior_subdir),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_fields = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[human-prior-check] summary_json={summary_json}")
    print(f"[human-prior-check] summary_csv={summary_csv}")
    if not all(float(row["prior_loss_weight_nonzero"]) > 0.0 and float(row["prior_loss_proxy"]) > 0.0 for row in rows):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
