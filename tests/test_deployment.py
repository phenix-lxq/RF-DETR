"""Compare standalone deployment math against RF-DETR's PyTorch reference."""

import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from rfdetr.models.postprocess import PostProcess

from infer_segmentation import decode, preprocess


class DeploymentParityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.metadata = {
            "input": {"shape": [1, 3, 312, 312], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "classes": ["first", "second"], "num_select": 7,
        }

    def test_rgb_preprocessing_matches_upscale_and_downscale(self):
        for height, width in ((23, 37), (480, 640), (720, 1280)):
            with self.subTest(size=(height, width)):
                rgb = self.rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                reference = F.normalize(
                    F.resize(F.to_tensor(Image.fromarray(rgb)), [312, 312], antialias=False),
                    self.metadata["input"]["mean"], self.metadata["input"]["std"],
                ).numpy()
                actual = preprocess(rgb, self.metadata)
                self.assertEqual(actual.shape, (1, 3, 312, 312))
                self.assertTrue(actual.flags.c_contiguous)
                np.testing.assert_allclose(actual[0], reference, rtol=0, atol=3e-5)

    def test_query_class_selection_and_resized_masks_match_reference(self):
        raw = (
            self.rng.random((1, 5, 4), dtype=np.float32),
            np.array([[[4, 3, 9], [6, -4, -9], [2, 5, -8], [-7, -8, -9], [1, 2, -5]]], dtype=np.float32),
            self.rng.standard_normal((1, 5, 7, 9)).astype(np.float32),
        )
        for threshold in (0.5, 1.0):
            with self.subTest(threshold=threshold):
                actual = decode(raw, (53, 31), self.metadata, threshold)
                reference = PostProcess(num_select=7)(
                    dict(zip(("pred_boxes", "pred_logits", "pred_masks"), (torch.from_numpy(value) for value in raw))),
                    torch.tensor([[31, 53]]), score_threshold=threshold,
                )[0]
                keep = reference["labels"] < len(self.metadata["classes"])
                np.testing.assert_array_equal(actual.class_ids, reference["labels"][keep].numpy())
                np.testing.assert_allclose(actual.scores, reference["scores"][keep].numpy(), atol=1e-6)
                np.testing.assert_allclose(actual.boxes, reference["boxes"][keep].numpy(), atol=1e-5)
                np.testing.assert_array_equal(actual.masks, reference["masks"][keep].squeeze(1).numpy())
                if threshold == 1.0:
                    self.assertEqual(actual.masks.shape, (0, 31, 53))

    def test_equal_score_order_matches_pytorch_reference(self):
        raw = (
            self.rng.random((1, 3, 4), dtype=np.float32),
            np.zeros((1, 3, 3), dtype=np.float32),
            self.rng.standard_normal((1, 3, 2, 2)).astype(np.float32),
        )
        actual = decode(raw, (11, 7), self.metadata, 0.49)
        reference = PostProcess(num_select=7)(
            dict(zip(("pred_boxes", "pred_logits", "pred_masks"), (torch.from_numpy(value) for value in raw))),
            torch.tensor([[7, 11]]), score_threshold=0.49,
        )[0]
        keep = reference["labels"] < 2
        np.testing.assert_array_equal(actual.class_ids, reference["labels"][keep].numpy())
        np.testing.assert_array_equal(actual.masks, reference["masks"][keep].squeeze(1).numpy())

    def test_class_metadata_mismatch_is_rejected(self):
        raw = (np.zeros((1, 5, 4), np.float32), np.zeros((1, 5, 91), np.float32),
               np.zeros((1, 5, 7, 9), np.float32))
        with self.assertRaisesRegex(ValueError, "class slots"):
            decode(raw, (53, 31), self.metadata, 0.5)


if __name__ == "__main__":
    unittest.main()
