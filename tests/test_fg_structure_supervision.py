import json
import shutil
import subprocess
import tempfile
import unittest
import re
from pathlib import Path
from unittest import mock

import torch

from finetune_vggt_pseudo import (
    _build_fg_boundary_band_mask,
    _build_fg_bbox_mask,
    _build_fg_outside_ring_mask,
    _build_front_depth_soft_bias,
    _build_fg_structure_region_mask,
    _build_fg_structure_target_edge_support_mask,
    _build_inside_distance_weight,
    _build_largest_component_soft_bias,
    _fg_structure_depth_edge_loss,
    _point_mv_outside_ring_loss,
    _summarize_main_support_depth_modes,
    _summarize_main_support_components,
)
from vggt_geom import VGGTGeomTeacher


def _make_identity_projection(h: int, w: int, v: int = 2):
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    point_world = points.unsqueeze(0).unsqueeze(0).repeat(1, v, 1, 1, 1)
    extrinsic = torch.zeros(1, v, 3, 4, dtype=torch.float32)
    intrinsic = torch.zeros(1, v, 3, 3, dtype=torch.float32)
    for vi in range(v):
        extrinsic[0, vi, :, :3] = torch.eye(3)
        intrinsic[0, vi, 0, 0] = 1.0
        intrinsic[0, vi, 1, 1] = 1.0
        intrinsic[0, vi, 2, 2] = 1.0
    return point_world, extrinsic, intrinsic


class TestFgStructureSupervision(unittest.TestCase):
    def test_vggt_geom_teacher_respects_amp_autocast_gate(self):
        class DummyCameraHead(torch.nn.Module):
            def forward(self, agg_tokens_list):
                token = agg_tokens_list[-1]
                b, v = int(token.shape[0]), int(token.shape[1])
                return [torch.zeros(b, v, 8, dtype=torch.float32)]

        class DummyDepthHead(torch.nn.Module):
            def forward(self, agg_tokens_list, imgs, ps_idx):
                b, v, _, h, w = imgs.shape
                depth = torch.ones(b, v, h, w, 1, dtype=torch.float32)
                conf = torch.ones(b, v, h, w, 1, dtype=torch.float32)
                return depth, conf

        class DummyVGGT(torch.nn.Module):
            def __init__(self, enable_track=False):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(0.0))
                self.camera_head = DummyCameraHead()
                self.depth_head = DummyDepthHead()
                self.point_head = None

            def aggregator(self, imgs):
                b, v = int(imgs.shape[0]), int(imgs.shape[1])
                return [torch.zeros(b, v, 4, dtype=torch.float32)], None

        def fake_pose_encoding_to_extri_intri(pose_enc, image_hw):
            b, v = int(pose_enc.shape[0]), int(pose_enc.shape[1])
            extrinsic = torch.zeros(b, v, 3, 4, dtype=torch.float32)
            intrinsic = torch.zeros(b, v, 3, 3, dtype=torch.float32)
            extrinsic[..., :3, :3] = torch.eye(3, dtype=torch.float32)
            intrinsic[..., 0, 0] = 1.0
            intrinsic[..., 1, 1] = 1.0
            intrinsic[..., 2, 2] = 1.0
            return extrinsic, intrinsic

        prepared = {
            "imgs": torch.zeros(1, 2, 3, 4, 4, dtype=torch.float32),
            "batch_size": 1,
            "view_count": 2,
            "resolve_dt": 0.0,
            "load_dt": 0.0,
        }

        def run_case(amp_enabled: bool) -> int:
            autocast_calls = {"count": 0}

            class FakeAutocast:
                def __enter__(self):
                    autocast_calls["count"] += 1
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            with mock.patch("vggt_geom.VGGT", DummyVGGT), \
                 mock.patch("vggt_geom.pose_encoding_to_extri_intri", side_effect=fake_pose_encoding_to_extri_intri), \
                 mock.patch("vggt_geom._resolve_device", side_effect=lambda device: str(device)), \
                 mock.patch("torch.load", return_value={"dummy": torch.tensor(0.0)}), \
                 mock.patch.object(torch.Tensor, "pin_memory", lambda self: self), \
                 mock.patch.object(torch.Tensor, "to", lambda self, *args, **kwargs: self), \
                 mock.patch("torch.cuda.amp.autocast", side_effect=lambda *args, **kwargs: FakeAutocast()):
                teacher = VGGTGeomTeacher(
                    ckpt_path="dummy.pt",
                    device="cuda",
                    amp=amp_enabled,
                    tf32=True,
                    deterministic=False,
                )
                teacher.forward_prepared_batch(prepared)
            return int(autocast_calls["count"])

        self.assertEqual(run_case(False), 0)
        self.assertEqual(run_case(True), 1)

    def test_build_fg_bbox_mask_respects_margin_and_min_side(self):
        fg = torch.zeros(1, 1, 9, 9, dtype=torch.float32)
        fg[0, 0, 4, 4] = 1.0
        bbox = _build_fg_bbox_mask(fg_mask01=fg, margin_px=1, min_side_px=4)
        self.assertEqual(tuple(bbox.shape), (1, 1, 9, 9))
        self.assertEqual(int((bbox > 0.5).sum().item()), 16)
        self.assertEqual(float(bbox[0, 0, 4, 4].item()), 1.0)

    def test_build_fg_outside_ring_mask_creates_dilate_minus_fg(self):
        fg = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
        fg[0, 0, 1:4, 1:4] = 1.0
        ring = _build_fg_outside_ring_mask(fg_mask01=fg, ring_px=1)
        self.assertEqual(int((ring > 0.5).sum().item()), 16)
        self.assertEqual(float(ring[0, 0, 2, 2].item()), 0.0)

    def test_build_fg_structure_region_mask_intersects_bbox_with_eroded_fg(self):
        fg = torch.zeros(1, 1, 9, 9, dtype=torch.float32)
        fg[0, 0, 2:7, 2:7] = 1.0
        bbox = _build_fg_bbox_mask(fg_mask01=fg, margin_px=1, min_side_px=5)
        region = _build_fg_structure_region_mask(
            fg_mask01=fg,
            fg_bbox_mask01=bbox,
            region_mode="bbox_fg_interior",
            region_erode_px=1,
        )
        self.assertEqual(tuple(region.shape), (1, 1, 9, 9))
        self.assertEqual(float(region[0, 0, 4, 4].item()), 1.0)
        self.assertEqual(float(region[0, 0, 2, 2].item()), 0.0)
        self.assertEqual(float(region[0, 0, 1, 1].item()), 0.0)

    def test_build_fg_boundary_band_mask_is_fg_minus_eroded_fg(self):
        fg = torch.zeros(1, 1, 7, 7, dtype=torch.float32)
        fg[0, 0, 1:6, 1:6] = 1.0
        band = _build_fg_boundary_band_mask(fg_mask01=fg, erode_px=1)
        self.assertEqual(float(band[0, 0, 3, 3].item()), 0.0)
        self.assertEqual(float(band[0, 0, 1, 3].item()), 1.0)
        self.assertGreater(int((band > 0.5).sum().item()), 0)

    def test_fg_structure_depth_edge_loss_near_zero_when_pred_equals_target(self):
        depth = torch.linspace(0.1, 1.0, 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16, 1)
        valid = torch.ones(1, 1, 16, 16, dtype=torch.float32)
        fg = torch.ones(1, 1, 16, 16, dtype=torch.float32)
        bbox = _build_fg_bbox_mask(fg_mask01=fg, margin_px=0, min_side_px=16)
        boundary = _build_fg_boundary_band_mask(fg_mask01=fg, erode_px=1)
        loss, info = _fg_structure_depth_edge_loss(
            depth_pred=depth,
            depth_tgt=depth,
            valid01=valid,
            fg_bbox_mask01=bbox,
            fg_structure_region_mask01=bbox,
            boundary_probe_mask01=boundary,
            min_active_px=64,
        )
        self.assertLess(float(loss.item()), 1e-5)
        self.assertEqual(info["fg_structure_depth_edge_active_views"], 1.0)
        self.assertAlmostEqual(info["fg_structure_bbox_cover"], 1.0, places=6)
        self.assertAlmostEqual(info["fg_structure_region_cover"], 1.0, places=6)
        self.assertAlmostEqual(info["fg_structure_depth_edge_loss_interior"], info["fg_structure_depth_edge_loss"], places=6)

    def test_fg_structure_depth_edge_loss_inactive_when_bbox_pixels_too_few(self):
        depth = torch.ones(1, 1, 8, 8, 1, dtype=torch.float32)
        valid = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        valid[0, 0, 0:2, 0:2] = 1.0
        bbox = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        bbox[0, 0, 0:2, 0:2] = 1.0
        loss, info = _fg_structure_depth_edge_loss(
            depth_pred=depth,
            depth_tgt=depth,
            valid01=valid,
            fg_bbox_mask01=bbox,
            fg_structure_region_mask01=bbox,
            min_active_px=64,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(info["fg_structure_depth_edge_active_views"], 0.0)

    def test_build_fg_structure_target_edge_support_mask_selects_top_quantile(self):
        edge = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
        edge[0, 0, 2, 2] = 1.0
        edge[0, 0, 2, 1] = 0.8
        edge[0, 0, 1, 2] = 0.7
        valid = torch.ones(1, 1, 5, 5, dtype=torch.float32)
        region = torch.ones(1, 1, 5, 5, dtype=torch.float32)
        support, info = _build_fg_structure_target_edge_support_mask(
            target_edge01=edge,
            valid01=valid,
            fg_structure_region_mask01=region,
            view_active01=torch.ones(1, 1, dtype=torch.float32),
            mode="target_edge_quantile",
            quantile=0.9,
            min_support_px=2,
        )
        self.assertIsNotNone(support)
        self.assertEqual(info["fg_structure_target_edge_support_active"], 1.0)
        self.assertEqual(info["fg_structure_target_edge_support_views"], 1.0)
        self.assertGreaterEqual(int((support > 0.5).sum().item()), 2)
        self.assertEqual(float(support[0, 0, 2, 2].item()), 1.0)

    def test_build_inside_distance_weight_suppresses_boundary_shells(self):
        fg = torch.zeros(1, 1, 7, 7, dtype=torch.float32)
        fg[0, 0, 1:6, 1:6] = 1.0
        weight = _build_inside_distance_weight(mask01=fg, falloff_px=2)
        self.assertIsNotNone(weight)
        center = float(weight[0, 0, 3, 3].item())
        inner_shell = float(weight[0, 0, 2, 3].item())
        boundary_shell = float(weight[0, 0, 1, 3].item())
        self.assertGreater(center, inner_shell)
        self.assertGreater(inner_shell, boundary_shell)
        self.assertEqual(float(weight[0, 0, 0, 0].item()), 0.0)

    def test_summarize_main_support_components_detects_fragmentation(self):
        support = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        support[0, 0, 1:4, 1:4] = 1.0
        support[0, 0, 5:7, 5:7] = 0.9
        info, largest = _summarize_main_support_components(
            weight_map01=support,
            threshold_ratio=0.25,
        )
        self.assertEqual(info["main_support_component_active_views"], 1.0)
        self.assertEqual(info["main_support_component_count"], 2.0)
        self.assertGreater(info["main_support_largest_component_share"], 0.65)
        self.assertGreater(info["main_support_top2_component_share"], 0.99)
        self.assertGreaterEqual(info["main_support_centroid_distance_mean"], 0.0)
        self.assertIsNotNone(largest)
        self.assertEqual(int((largest > 0.5).sum().item()), 9)

    def test_build_largest_component_soft_bias_downweights_secondary_islands(self):
        support = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        support[0, 0, 1:4, 1:4] = 1.0
        support[0, 0, 5:7, 5:7] = 0.8
        bias, info = _build_largest_component_soft_bias(
            weight_map01=support,
            threshold_ratio=0.25,
            other_scale=0.35,
        )
        self.assertIsNotNone(bias)
        self.assertLess(info["main_support_component_bias_weight_share"], 1.0)
        self.assertEqual(float(bias[0, 0, 2, 2].item()), 1.0)
        self.assertAlmostEqual(float(bias[0, 0, 5, 5].item()), 0.35, places=6)

    def test_build_front_depth_soft_bias_prefers_front_layer_over_back_layer(self):
        depth = torch.full((1, 1, 8, 8), 0.4, dtype=torch.float32)
        depth[0, 0, 6:, :] = 1.2
        weight = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        bbox = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        bias, info = _build_front_depth_soft_bias(
            depth_tgt01=depth,
            weight_map01=weight,
            bbox_active01=bbox,
            mode="front_soft",
            tau=0.75,
            center_quantile=0.55,
            min_active_px=8,
        )
        self.assertIsNotNone(bias)
        front_mean = float(bias[0, 0, :6, :].mean().item())
        back_mean = float(bias[0, 0, 6:, :].mean().item())
        self.assertGreater(front_mean, back_mean)
        self.assertEqual(info["fg_structure_front_depth_bias_active_views"], 1.0)
        self.assertLess(info["fg_structure_front_depth_bias_weight_share"], 1.0)

    def test_summarize_main_support_depth_modes_detects_bimodal_depth_risk(self):
        depth = torch.full((1, 1, 8, 8), 0.35, dtype=torch.float32)
        depth[0, 0, :, 4:] = 0.95
        support = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        bbox = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        info = _summarize_main_support_depth_modes(
            depth_tgt01=depth,
            weight_map01=support,
            bbox_active01=bbox,
            min_active_px=16,
        )
        self.assertEqual(info["main_support_depth_mode_active_views"], 1.0)
        self.assertGreaterEqual(info["main_support_depth_mode_count"], 2.0)
        self.assertGreater(info["main_support_back_mode_share"], 0.0)
        self.assertGreater(info["main_support_depth_hist_peak_ratio"], 0.0)
        self.assertGreater(info["main_support_secondary_risk"], 0.0)

    def test_point_mv_outside_ring_loss_inactive_on_tiny_coverage(self):
        point_world, extrinsic, intrinsic = _make_identity_projection(h=8, w=8, v=2)
        ring = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
        ring[0, 0, 1:3, 1:3] = 1.0
        ring[0, 1, 1:3, 1:3] = 1.0
        valid = torch.ones(1, 2, 8, 8, dtype=torch.float32)
        loss, info = _point_mv_outside_ring_loss(
            point_world=point_world,
            outside_ring_mask_tgt=ring,
            extrinsic_w2c=extrinsic,
            intrinsic=intrinsic,
            src_valid_mask=valid,
            support_weight=None,
            robust_eps=0.0,
            stride=1,
            min_active_ring_px=32,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(info["point_mv_outside_ring_active_views"], 0.0)

    def test_h0_dry_run_contract_is_strict_noop(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "pointmap_source": "depth_unproject",
                "point_target_mode": "depth_unproject",
                "precompute_mv_support_on": "on",
                "precompute_mv_support_region_mode": "bg_only",
                "precompute_mv_support_fg_mask_source": "mask",
                "precompute_mv_support_fg_erode_px": "5",
                "precompute_mv_support_fg_preserve_px": "5",
                "point_support_mode": "off",
                "point_mv_depth_support_mode": "off",
                "point_mv_mask_support_mode": "off",
                "point_target_blend_by_mv_support": "off",
                "point_target_blend_mv_region_mode": "all",
                "point_mv_depth_region_mode": "all",
                "use_fg_mask": "on",
                "fg_mask_source": "mask",
                "lambda_point_mv_mask": "0",
                "fg_supervision_boost": "1.0",
                "fg_supervision_bg_floor": "0.0",
                "fg_supervision_region_mode": "all",
                "fg_supervision_region_erode_px": "0",
                "lambda_fg_conf_presence": "0.0",
                "fg_conf_presence_target_ratio": "0.9",
                "tf32": "0",
                "amp": "1",
                "strict_deterministic": "1",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H0",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_supervision_boost"]), "1")
                self.assertEqual(str(latest["fg_supervision_bg_floor"]), "0")
                self.assertEqual(str(latest["lambda_fg_conf_presence"]), "0")
                self.assertEqual(str(latest["lambda_fg_structure_depth_edge"]), "0")
                self.assertEqual(str(latest["lambda_point_mv_outside_ring"]), "0")
                self.assertEqual(str(latest["fg_structure_bbox_margin_px"]), "12")
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox")
                self.assertEqual(str(latest["fg_structure_region_erode_px"]), "0")
                self.assertEqual(str(latest["fg_structure_depth_edge_warmup_steps"]), "0")
                self.assertEqual(str(latest["fg_structure_boundary_probe_px"]), "2")
                self.assertEqual(str(latest["fg_structure_edge_support_mode"]), "off")
                self.assertEqual(str(latest["fg_structure_edge_support_quantile"]), "0")
                self.assertEqual(str(latest["fg_structure_edge_support_min_px"]), "32")
                self.assertEqual(str(latest["point_mv_outside_ring_px"]), "3")
                self.assertEqual(str(latest["tf32"]), "0")
                self.assertEqual(str(latest["amp"]), "1")
                self.assertEqual(str(latest["strict_deterministic"]), "1")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h_family_strict_bypass_guard_present(self):
        repo_dir = Path(__file__).resolve().parents[1]
        source = (repo_dir / "finetune_vggt_pseudo.py").read_text(encoding="utf-8")
        self.assertIn(
            "h_family_enabled = (lambda_fg_structure_depth_edge > 0.0) or (lambda_point_mv_outside_ring > 0.0)",
            source,
        )
        self.assertRegex(
            source,
            re.compile(
                r"loss_fg_structure_depth_edge = torch\.zeros\(\[\], device=device, dtype=torch\.float32\)\s+"
                r"loss_point_mv_outside_ring = torch\.zeros\(\[\], device=device, dtype=torch\.float32\)\s+"
                r"if h_family_enabled:",
                re.MULTILINE,
            ),
        )

    def test_h1a_dry_run_contract_uses_interior_region_and_warmup(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "seq_names": "CoreView_390",
                "resume_ckpt": "dummy.pt",
                "pseudo_geom_subdir": "dummy_geom",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H1a",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox_fg_interior")
                self.assertEqual(str(latest["fg_structure_region_erode_px"]), "3")
                self.assertEqual(str(latest["fg_structure_depth_edge_warmup_steps"]), "80")
                self.assertEqual(str(latest["fg_structure_boundary_probe_px"]), "2")
                self.assertEqual(str(latest["fg_structure_edge_support_mode"]), "off")
                self.assertEqual(str(latest["fg_structure_edge_support_quantile"]), "0")
                self.assertEqual(str(latest["fg_structure_edge_support_min_px"]), "32")
                self.assertEqual(str(latest["lambda_fg_structure_depth_edge"]), "0.003")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h1d_dry_run_contract_uses_target_edge_support(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "seq_names": "CoreView_390",
                "resume_ckpt": "dummy.pt",
                "pseudo_geom_subdir": "dummy_geom",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H1d",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox_fg_interior")
                self.assertEqual(str(latest["fg_structure_region_erode_px"]), "3")
                self.assertEqual(str(latest["fg_structure_edge_support_mode"]), "target_edge_quantile")
                self.assertEqual(str(latest["fg_structure_edge_support_quantile"]), "0.75")
                self.assertEqual(str(latest["fg_structure_edge_support_min_px"]), "32")
                self.assertEqual(str(latest["lambda_fg_structure_depth_edge"]), "0.003")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h1s1_dry_run_contract_uses_soft_edge_weighting(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "seq_names": "CoreView_390",
                "resume_ckpt": "dummy.pt",
                "pseudo_geom_subdir": "dummy_geom",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H1s1",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox_fg_interior")
                self.assertEqual(str(latest["fg_structure_edge_support_mode"]), "off")
                self.assertEqual(str(latest["fg_structure_edge_weight_mode"]), "target_edge_sqrt")
                self.assertEqual(str(latest["fg_structure_boundary_falloff_px"]), "2")
                self.assertEqual(str(latest["fg_structure_component_bias_mode"]), "off")
                self.assertEqual(str(latest["lambda_fg_structure_depth_edge"]), "0.003")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h1s1_core_dry_run_contract_uses_largest_component_bias(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "seq_names": "CoreView_390",
                "resume_ckpt": "dummy.pt",
                "pseudo_geom_subdir": "dummy_geom",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H1s1_core",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox_fg_interior")
                self.assertEqual(str(latest["fg_structure_edge_weight_mode"]), "target_edge_sqrt")
                self.assertEqual(str(latest["fg_structure_boundary_falloff_px"]), "2")
                self.assertEqual(str(latest["fg_structure_component_bias_mode"]), "largest_soft")
                self.assertEqual(str(latest["fg_structure_component_bias_threshold_ratio"]), "0.25")
                self.assertEqual(str(latest["fg_structure_component_bias_other_scale"]), "0.35")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h1sf1_dry_run_contract_enables_front_depth_bias(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        latest_contract = repo_dir / "logs" / "modal_phase5" / "probe_contract_latest.json"
        backup_text = latest_contract.read_text(encoding="utf-8-sig") if latest_contract.exists() else None
        stamped_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            inherit_path = Path(tmpdir) / "inherit_contract.json"
            inherit = {
                "seq_names": "CoreView_390",
                "resume_ckpt": "dummy.pt",
                "pseudo_geom_subdir": "dummy_geom",
            }
            inherit_path.write_text(json.dumps(inherit), encoding="utf-8")
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(script),
                        "-RepoDir",
                        str(repo_dir),
                        "-ProbeId",
                        "H1sf1",
                        "-InheritContractPath",
                        str(inherit_path),
                        "-SeqNames",
                        "CoreView_390",
                        "-ResumeCkpt",
                        "dummy.pt",
                        "-PseudoGeomSubdir",
                        "dummy_geom",
                        "-EvalNumSrcViews",
                        "8",
                        "-LambdaPointMvDepth",
                        "0.001",
                        "-PrecomputeMvSupportFgPreservePx",
                        "5",
                        "-DryRun",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "contract_stamped=" in line:
                        stamped_path = line.split("contract_stamped=", 1)[1].strip()
                latest = json.loads(latest_contract.read_text(encoding="utf-8-sig"))
                self.assertEqual(str(latest["fg_structure_region_mode"]), "bbox_fg_interior")
                self.assertEqual(str(latest["fg_structure_edge_weight_mode"]), "target_edge_sqrt")
                self.assertEqual(str(latest["fg_structure_component_bias_mode"]), "largest_soft")
                self.assertEqual(str(latest["fg_structure_front_depth_bias_mode"]), "front_soft")
                self.assertEqual(str(latest["fg_structure_front_depth_bias_tau"]), "0.75")
                self.assertEqual(str(latest["fg_structure_front_depth_bias_center_quantile"]), "0.55")
            finally:
                if backup_text is None:
                    if latest_contract.exists():
                        latest_contract.unlink()
                else:
                    latest_contract.write_text(backup_text, encoding="utf-8")
                if stamped_path:
                    stamped = Path(stamped_path)
                    if stamped.exists():
                        stamped.unlink()

    def test_h1_probe_id_is_rejected(self):
        repo_dir = Path(__file__).resolve().parents[1]
        script = repo_dir / "scripts" / "run_human_transparency_probe_once.ps1"
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-RepoDir",
                str(repo_dir),
                "-ProbeId",
                "H1",
                "-DryRun",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deprecated", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
