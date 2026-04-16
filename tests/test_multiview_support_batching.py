import unittest

import torch

from precompute_zju_vggt_geom import (
    _build_multiview_support,
    _build_multiview_support_batched,
    _compute_mv_support_diag_batch_adaptive,
)


def _make_intrinsic(batch: int, views: int, height: int, width: int) -> torch.Tensor:
    intrinsic = torch.zeros(batch, views, 3, 3, dtype=torch.float32)
    intrinsic[..., 0, 0] = 7.5
    intrinsic[..., 1, 1] = 8.0
    intrinsic[..., 0, 2] = (width - 1) * 0.5
    intrinsic[..., 1, 2] = (height - 1) * 0.5
    intrinsic[..., 2, 2] = 1.0
    return intrinsic


def _make_extrinsic(batch: int, views: int) -> torch.Tensor:
    extrinsic = torch.zeros(batch, views, 3, 4, dtype=torch.float32)
    extrinsic[..., 0, 0] = 1.0
    extrinsic[..., 1, 1] = 1.0
    extrinsic[..., 2, 2] = 1.0
    return extrinsic


def _make_case(batch: int = 2, views: int = 3, height: int = 6, width: int = 8):
    intrinsic = _make_intrinsic(batch, views, height, width)
    extrinsic = _make_extrinsic(batch, views)
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    xs = xs.view(1, 1, height, width)
    ys = ys.view(1, 1, height, width)
    depth = (
        1.2
        + 0.05 * torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1)
        + 0.03 * torch.arange(views, dtype=torch.float32).view(1, views, 1, 1)
        + 0.01 * ys
        + 0.02 * xs
    )
    depth_conf = torch.full((batch, views, height, width), 0.9, dtype=torch.float32)
    depth_conf[:, :, 0, 0] = 0.0
    depth_conf[:, :, -1, -1] = 0.01
    fx = intrinsic[..., 0, 0].view(batch, views, 1, 1)
    fy = intrinsic[..., 1, 1].view(batch, views, 1, 1)
    cx = intrinsic[..., 0, 2].view(batch, views, 1, 1)
    cy = intrinsic[..., 1, 2].view(batch, views, 1, 1)
    x_cam = (xs - cx) * depth / fx
    y_cam = (ys - cy) * depth / fy
    point_world = torch.stack((x_cam, y_cam, depth), dim=-1)
    generation_gate = torch.ones(batch, views, height, width, dtype=torch.float32)
    generation_gate[0, :, 1:3, 2:5] = 0.0
    generation_gate[1, :, 3:, :2] = 0.0
    return point_world, depth, depth_conf, extrinsic, intrinsic, generation_gate


class TestMultiviewSupportBatching(unittest.TestCase):
    def test_batched_support_matches_single_frame_wrapper(self):
        case = _make_case()
        point_world, depth, depth_conf, extrinsic, intrinsic, generation_gate = case
        diag_batch = _build_multiview_support_batched(
            point_world=point_world,
            depth=depth,
            depth_conf=depth_conf,
            extrinsic_w2c=extrinsic,
            intrinsic=intrinsic,
            tol_abs=0.06,
            tol_rel=0.1,
            stride=2,
            conf_valid_floor=0.02,
            generation_gate=generation_gate,
            return_diag=True,
        )
        for batch_idx in range(point_world.shape[0]):
            diag_single = _build_multiview_support(
                point_world=point_world[batch_idx],
                depth=depth[batch_idx],
                depth_conf=depth_conf[batch_idx],
                extrinsic_w2c=extrinsic[batch_idx],
                intrinsic=intrinsic[batch_idx],
                tol_abs=0.06,
                tol_rel=0.1,
                stride=2,
                conf_valid_floor=0.02,
                generation_gate=generation_gate[batch_idx],
                return_diag=True,
            )
            torch.testing.assert_close(diag_batch["support"][batch_idx], diag_single["support"])
            torch.testing.assert_close(diag_batch["cover"][batch_idx], diag_single["cover"])
            torch.testing.assert_close(diag_batch["valid"][batch_idx], diag_single["valid"])
            torch.testing.assert_close(diag_batch["gate"][batch_idx], diag_single["gate"])

    def test_adaptive_batch_helper_matches_per_frame_results(self):
        case = _make_case(batch=3, views=4, height=5, width=7)
        point_world, depth, depth_conf, extrinsic, intrinsic, generation_gate = case
        geoms = []
        gates = []
        for batch_idx in range(point_world.shape[0]):
            geoms.append(
                {
                    "pointmap": point_world[batch_idx],
                    "depth": depth[batch_idx],
                    "depth_conf": depth_conf[batch_idx],
                    "extrinsic": extrinsic[batch_idx],
                    "intrinsic": intrinsic[batch_idx],
                }
            )
            gates.append(generation_gate[batch_idx])
        diag_list = _compute_mv_support_diag_batch_adaptive(
            geoms,
            generation_gates=gates,
            tol_abs=0.05,
            tol_rel=0.08,
            stride=1,
            conf_valid_floor=0.02,
            heartbeat_sec=0.0,
            heartbeat_prefix="[test_mv_support]",
        )
        self.assertEqual(len(diag_list), len(geoms))
        for batch_idx, diag in enumerate(diag_list):
            diag_single = _build_multiview_support(
                point_world=point_world[batch_idx],
                depth=depth[batch_idx],
                depth_conf=depth_conf[batch_idx],
                extrinsic_w2c=extrinsic[batch_idx],
                intrinsic=intrinsic[batch_idx],
                tol_abs=0.05,
                tol_rel=0.08,
                stride=1,
                conf_valid_floor=0.02,
                generation_gate=generation_gate[batch_idx],
                return_diag=True,
            )
            torch.testing.assert_close(diag["support"], diag_single["support"])
            torch.testing.assert_close(diag["cover"], diag_single["cover"])
            torch.testing.assert_close(diag["valid"], diag_single["valid"])
            torch.testing.assert_close(diag["gate"], diag_single["gate"])


if __name__ == "__main__":
    unittest.main()
