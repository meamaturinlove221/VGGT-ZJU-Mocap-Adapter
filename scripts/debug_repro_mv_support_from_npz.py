import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import precompute_zju_vggt_geom as pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seq-root", default=str(Path(pc.ZJU_ROOT) / "CoreView_390"))
    ap.add_argument("--zju-root", default=pc.ZJU_ROOT)
    ap.add_argument("--region-mode", default="bg_only")
    ap.add_argument("--fg-mask-source", default="mask")
    ap.add_argument("--fg-erode-px", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tol-abs", type=float, default=0.06)
    ap.add_argument("--tol-rel", type=float, default=0.1)
    ap.add_argument("--conf-valid-floor", type=float, default=0.02)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    z = np.load(npz_path, allow_pickle=True)
    pointmap = torch.from_numpy(z["pointmap"]).float()
    depth = torch.from_numpy(z["depth"]).float()
    depth_conf = torch.from_numpy(z["depth_conf"]).float()
    extrinsic = torch.from_numpy(z["extrinsic"]).float()
    intrinsic = torch.from_numpy(z["intrinsic"]).float()
    img_paths = [str(x) for x in z["img_paths"].tolist()]
    device = pointmap.device

    print(
        json.dumps(
            {
                "pointmap_shape": list(pointmap.shape),
                "depth_shape": list(depth.shape),
                "depth_conf_shape": list(depth_conf.shape),
                "extrinsic_shape": list(extrinsic.shape),
                "intrinsic_shape": list(intrinsic.shape),
                "img_count": len(img_paths),
                "sample_img_path": img_paths[0] if img_paths else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    depth3, _ = pc._depth_like_to_3d(depth)
    target_hw = (int(depth3.shape[1]), int(depth3.shape[2]))
    gate, fg_mask, resolved_src, resolved_mode = pc._build_support_generation_gate(
        img_paths,
        zju_root=args.zju_root,
        seq_root=args.seq_root,
        target_hw=target_hw,
        region_mode=args.region_mode,
        fg_mask_source=args.fg_mask_source,
        fg_erode_px=args.fg_erode_px,
        device=device,
    )
    print(
        json.dumps(
            {
                "target_hw": list(target_hw),
                "gate_shape": None if gate is None else list(gate.shape),
                "fg_mask_shape": None if fg_mask is None else list(fg_mask.shape),
                "resolved_src": resolved_src,
                "resolved_mode": resolved_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    out = pc._build_multiview_support(
        point_world=pointmap,
        depth=depth,
        depth_conf=depth_conf,
        extrinsic_w2c=extrinsic,
        intrinsic=intrinsic,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        stride=args.stride,
        conf_valid_floor=args.conf_valid_floor,
        generation_gate=gate,
        return_diag=True,
    )
    print(
        json.dumps(
            {
                "support_shape": list(out["support"].shape),
                "cover_shape": list(out["cover"].shape),
                "valid_shape": list(out["valid"].shape),
                "gate_shape_after": None if out["gate"] is None else list(out["gate"].shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
