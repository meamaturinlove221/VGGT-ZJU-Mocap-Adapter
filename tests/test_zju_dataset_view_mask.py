import os
import tempfile
import unittest

import numpy as np

from zju_dataset_view import ZJUViewSynthDataset


class TestZJUMaskPathAndNormalize(unittest.TestCase):
    def setUp(self):
        self.ds = ZJUViewSynthDataset.__new__(ZJUViewSynthDataset)

    def _touch(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"")

    def test_infer_mask_path_prefers_mask_over_mask_cihp(self):
        with tempfile.TemporaryDirectory() as td:
            img = os.path.join(td, "CoreView_390", "images", "Camera_B1", "000001.jpg")
            p_mask = os.path.join(td, "CoreView_390", "mask", "Camera_B1", "000001.png")
            p_cihp = os.path.join(td, "CoreView_390", "mask_cihp", "Camera_B1", "000001.png")
            self._touch(img)
            self._touch(p_mask)
            self._touch(p_cihp)
            got = self.ds._infer_mask_path(img)
            self.assertEqual(os.path.normpath(got), os.path.normpath(p_mask))

    def test_infer_mask_path_supports_alias_and_camera_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            img1 = os.path.join(td, "CoreView_390", "images_512", "camera_02", "000010.jpg")
            msk1 = os.path.join(td, "CoreView_390", "masks", "camera_02", "000010.png")
            self._touch(img1)
            self._touch(msk1)
            got1 = self.ds._infer_mask_path(img1)
            self.assertEqual(os.path.normpath(got1), os.path.normpath(msk1))

            img2 = os.path.join(td, "CoreView_390", "img", "12", "000111.jpg")
            msk2 = os.path.join(td, "CoreView_390", "masks_cihp", "12", "000111.png")
            self._touch(img2)
            self._touch(msk2)
            got2 = self.ds._infer_mask_path(img2)
            self.assertEqual(os.path.normpath(got2), os.path.normpath(msk2))

            img3 = os.path.join(td, "CoreView_390", "Camera_04", "000200.jpg")
            msk3 = os.path.join(td, "CoreView_390", "mask", "Camera_04", "000200.png")
            self._touch(img3)
            self._touch(msk3)
            got3 = self.ds._infer_mask_path(img3)
            self.assertEqual(os.path.normpath(got3), os.path.normpath(msk3))

    def test_normalize_mask_supports_01_255_and_label_maps(self):
        m01 = np.array([[0.0, 1.0]], dtype=np.float32)
        n01 = self.ds._normalize_mask(m01)
        self.assertTrue(np.allclose(n01, m01))

        m255 = np.array([[0, 128, 255]], dtype=np.uint8)
        n255 = self.ds._normalize_mask(m255)
        self.assertAlmostEqual(float(n255[0, 1]), 128.0 / 255.0, places=5)
        self.assertAlmostEqual(float(n255.max()), 1.0, places=6)

        m19 = np.array([[0, 10, 19]], dtype=np.uint8)
        n19 = self.ds._normalize_mask(m19)
        self.assertAlmostEqual(float(n19[0, 1]), 10.0 / 19.0, places=5)
        self.assertAlmostEqual(float(n19.max()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
