import unittest

import torch

from train_view_decoder_ablation import build_masks_from_batch


class TestBuildMasksFromBatch(unittest.TestCase):
    def _base_batch(self):
        return {
            "tgt_depth_conf": torch.ones(1, 1, 4, 4, dtype=torch.float32),
        }

    def test_require_tgt_fg_raises_when_missing(self):
        batch = self._base_batch()
        batch["tgt_conf"] = torch.ones(1, 1, 4, 4, dtype=torch.float32)
        with self.assertRaises(KeyError):
            build_masks_from_batch(
                batch=batch,
                pred_hw=(4, 4),
                device="cpu",
                fg_keep_largest_cc=False,
            )

    def test_allow_fg_from_conf_enables_conf_fallback(self):
        batch = self._base_batch()
        batch["tgt_conf"] = torch.ones(1, 1, 4, 4, dtype=torch.float32)
        _, _, _, _, _, aux = build_masks_from_batch(
            batch=batch,
            pred_hw=(4, 4),
            device="cpu",
            fg_keep_largest_cc=False,
            require_tgt_fg=True,
            allow_fg_from_conf=True,
        )
        self.assertEqual(aux["source_fg_key"], "tgt_conf")

    def test_default_uses_tgt_fg(self):
        batch = self._base_batch()
        batch["tgt_fg"] = torch.ones(1, 1, 4, 4, dtype=torch.float32)
        _, _, _, _, _, aux = build_masks_from_batch(
            batch=batch,
            pred_hw=(4, 4),
            device="cpu",
            fg_keep_largest_cc=False,
        )
        self.assertEqual(aux["source_fg_key"], "tgt_fg")


if __name__ == "__main__":
    unittest.main()
