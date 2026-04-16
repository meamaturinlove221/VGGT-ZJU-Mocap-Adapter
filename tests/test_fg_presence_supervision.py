import unittest

import torch

from finetune_vggt_pseudo import (
    _apply_fg_supervision_boost,
    _build_fg_supervision_boost_mask,
    _fg_conf_presence_floor_loss,
)


class TestFgPresenceSupervision(unittest.TestCase):
    def test_fg_supervision_boost_scales_only_foreground(self):
        base = torch.ones(1, 1, 2, 4, dtype=torch.float32)
        fg = torch.tensor(
            [[[[1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        out, info = _apply_fg_supervision_boost(
            base_weight01=base,
            fg_mask01=fg,
            fg_boost=1.5,
        )
        expected = torch.tensor(
            [[[[1.5, 1.5, 1.0, 1.0],
               [1.5, 1.5, 1.0, 1.0]]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(info["fg_supervision_boost_applied"], 1.0)
        self.assertAlmostEqual(info["fg_supervision_profile_fg_mean"], 1.5, places=6)
        self.assertAlmostEqual(info["fg_supervision_profile_bg_mean"], 1.0, places=6)

    def test_fg_supervision_boost_mask_defaults_to_full_foreground(self):
        fg = torch.tensor(
            [[[[1.0, 1.0, 0.0],
               [1.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        boost_mask, info = _build_fg_supervision_boost_mask(
            fg_mask01=fg,
            region_mode="all",
            region_erode_px=5,
        )
        self.assertTrue(torch.allclose(boost_mask, fg))
        self.assertEqual(info["fg_supervision_region_mode"], "all")
        self.assertAlmostEqual(info["fg_supervision_boundary_ring_cover"], 0.0, places=6)

    def test_fg_supervision_boost_mask_erodes_only_boost_region(self):
        fg = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 1.0, 1.0, 0.0],
               [0.0, 1.0, 1.0, 1.0, 0.0],
               [0.0, 1.0, 1.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        boost_mask, info = _build_fg_supervision_boost_mask(
            fg_mask01=fg,
            region_mode="interior_only",
            region_erode_px=1,
        )
        expected = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(boost_mask, expected))
        self.assertEqual(info["fg_supervision_region_mode"], "interior_only")
        self.assertAlmostEqual(info["fg_supervision_boost_cover_ratio_in_fg"], 1.0 / 9.0, places=6)
        self.assertAlmostEqual(info["fg_supervision_boundary_ring_ratio_in_fg"], 8.0 / 9.0, places=6)

    def test_fg_supervision_boost_uses_original_fg_for_stats_when_requested(self):
        base = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        fg_all = torch.ones_like(base)
        fg_interior = torch.tensor(
            [[[[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        out, info = _apply_fg_supervision_boost(
            base_weight01=base,
            fg_mask01=fg_interior,
            fg_boost=1.5,
            fg_stats_mask01=fg_all,
        )
        self.assertAlmostEqual(float(out[0, 0, 1, 1].item()), 1.5, places=6)
        self.assertAlmostEqual(float(out[0, 0, 0, 0].item()), 1.0, places=6)
        self.assertAlmostEqual(info["fg_supervision_profile_fg_mean"], (1.5 + 8.0 * 1.0) / 9.0, places=6)

    def test_fg_conf_presence_loss_zero_when_prediction_meets_floor(self):
        pred = torch.full((1, 1, 2, 2), 0.9, dtype=torch.float32)
        tgt = torch.ones_like(pred)
        fg = torch.ones_like(pred)
        loss, info = _fg_conf_presence_floor_loss(
            pred_conf01=pred,
            tgt_conf01=tgt,
            fg_mask01=fg,
            valid01=None,
            target_ratio=0.9,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(info["fg_conf_presence_enabled"], 1.0)
        self.assertAlmostEqual(info["fg_conf_presence_target_floor"], 0.9, places=6)

    def test_fg_conf_presence_loss_positive_when_prediction_below_floor(self):
        pred = torch.tensor([[[[0.3, 0.3], [0.2, 0.2]]]], dtype=torch.float32)
        tgt = torch.tensor([[[[0.9, 0.9], [0.8, 0.8]]]], dtype=torch.float32)
        fg = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
        valid = torch.ones_like(fg)
        loss, info = _fg_conf_presence_floor_loss(
            pred_conf01=pred,
            tgt_conf01=tgt,
            fg_mask01=fg,
            valid01=valid,
            target_ratio=0.9,
        )
        self.assertGreater(float(loss.item()), 0.0)
        self.assertAlmostEqual(info["fg_conf_presence_pred_mean"], 0.3, places=6)
        self.assertAlmostEqual(info["fg_conf_presence_tgt_mean"], 0.9, places=6)
        self.assertAlmostEqual(info["fg_conf_presence_target_floor"], 0.81, places=6)

    def test_fg_conf_presence_loss_respects_valid_mask(self):
        pred = torch.tensor([[[[0.2, 0.8], [0.2, 0.8]]]], dtype=torch.float32)
        tgt = torch.tensor([[[[1.0, 1.0], [0.2, 0.2]]]], dtype=torch.float32)
        fg = torch.ones_like(pred)
        valid = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
        loss, info = _fg_conf_presence_floor_loss(
            pred_conf01=pred,
            tgt_conf01=tgt,
            fg_mask01=fg,
            valid01=valid,
            target_ratio=0.8,
        )
        self.assertGreater(float(loss.item()), 0.0)
        self.assertAlmostEqual(info["fg_conf_presence_active_ratio"], 0.5, places=6)
        self.assertAlmostEqual(info["fg_conf_presence_pred_mean"], 0.2, places=6)
        self.assertAlmostEqual(info["fg_conf_presence_tgt_mean"], 0.6, places=6)
        self.assertAlmostEqual(info["fg_conf_presence_target_floor"], 0.48, places=6)

    def test_fg_conf_presence_loss_is_zero_when_fg_mask_is_empty(self):
        pred = torch.full((1, 1, 2, 2), 0.4, dtype=torch.float32)
        tgt = torch.full((1, 1, 2, 2), 0.9, dtype=torch.float32)
        fg = torch.zeros_like(pred)
        loss, info = _fg_conf_presence_floor_loss(
            pred_conf01=pred,
            tgt_conf01=tgt,
            fg_mask01=fg,
            valid01=None,
            target_ratio=0.8,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(info["fg_conf_presence_enabled"], 0.0)
        self.assertAlmostEqual(info["fg_conf_presence_active_ratio"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
