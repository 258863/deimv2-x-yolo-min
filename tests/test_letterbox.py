import unittest

import torch
from PIL import Image

from engine.data.transforms.letterbox import LetterBox
from engine.deim.postprocessor import PostProcessor


class LetterBoxTests(unittest.TestCase):
    def test_landscape_and_portrait_use_actual_rounded_gains(self):
        for wh in ((7, 4), (4, 7)):
            image = Image.new("RGB", wh, (0, 0, 0))
            target = {"boxes": torch.tensor([[1., 1., wh[0] - 1., wh[1] - 1.]])}
            result, transformed = LetterBox(10)(image, target)
            self.assertEqual(result.size, (10, 10))
            gain = transformed["letterbox_gain"]
            pad = transformed["letterbox_pad"]
            expected = target["boxes"].clone()
            expected[:, 0::2] *= gain[0]
            expected[:, 1::2] *= gain[1]
            expected[:, 0::2] += pad[0]
            expected[:, 1::2] += pad[1]
            torch.testing.assert_close(transformed["boxes"], expected)

    def test_rgb_padding_is_filled_on_every_channel(self):
        # PIL's Image.new() only broadcasts a single int fill to every band for single-band modes
        # (e.g. 'L'); for multi-band modes such as 'RGB' it used to write the value into the first
        # band only and leave the rest at 0, tinting the padding instead of filling it gray.
        image = Image.new("RGB", (7, 4), (200, 200, 200))
        result, _ = LetterBox(10, fill=114)(image, {"boxes": torch.empty((0, 4))})
        self.assertEqual(result.getpixel((0, 0)), (114, 114, 114))
        self.assertEqual(result.getpixel((9, 9)), (114, 114, 114))

    def test_single_band_padding_still_works(self):
        image = Image.new("L", (7, 4), 200)
        result, _ = LetterBox(10, fill=114)(image, {"boxes": torch.empty((0, 4))})
        self.assertEqual(result.getpixel((0, 0)), 114)

    def test_odd_padding_and_empty_boxes(self):
        image = Image.new("RGB", (4, 3), (0, 0, 0))
        result, target = LetterBox((8, 8))(image, {"boxes": torch.empty((0, 4))})
        self.assertEqual(result.size, (8, 8))
        self.assertEqual(target["letterbox_pad"].tolist(), [0., 1.])
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))

    def test_postprocessor_letterbox_roundtrip_and_legacy_path(self):
        processor = PostProcessor(num_classes=1, num_top_queries=1)
        output = {"pred_logits": torch.tensor([[[10.]]]), "pred_boxes": torch.tensor([[[.5, .5, .2, .2]]])}
        # Canvas 10x10; original image 7x4 rounds to 10x6, with vertical pad 2.
        pred = processor(output, torch.tensor([[10, 10]]))[0]["boxes"]
        gain = torch.tensor([10 / 7, 6 / 4])
        pred[:, 0::2] = (pred[:, 0::2] - 0) / gain[0]
        pred[:, 1::2] = (pred[:, 1::2] - 2) / gain[1]
        torch.testing.assert_close(pred, torch.tensor([[2.8, 4 / 3, 4.2, 8 / 3]]))
        legacy = processor(output, torch.tensor([[7, 4]]))[0]["boxes"]
        torch.testing.assert_close(legacy, torch.tensor([[2.8, 1.6, 4.2, 2.4]]))


if __name__ == "__main__":
    unittest.main()
