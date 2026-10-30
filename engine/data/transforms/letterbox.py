import math

import torch
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import Transform
from PIL import Image

from ...core import register


@register()
class LetterBox(Transform):
    """Resize while preserving aspect ratio, then center-pad with YOLO gray."""

    def __init__(self, size=640, fill=114):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.fill = fill

    def forward(self, image, target=None):
        original_w, original_h = image.size if isinstance(image, Image.Image) else (image.shape[-1], image.shape[-2])
        target_w, target_h = self.size
        gain = min(target_w / original_w, target_h / original_h)
        resized_w = int(round(original_w * gain))
        resized_h = int(round(original_h * gain))
        pad_w, pad_h = target_w - resized_w, target_h - resized_h
        left, top = pad_w // 2, pad_h // 2
        if isinstance(image, Image.Image):
            image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
            # Image.new() only broadcasts a single int to every band for 'L'/'1' images; for
            # multi-band modes (RGB, RGBA, ...) it writes the value into the first band only and
            # zero-fills the rest, which silently tinted the padding instead of filling it gray.
            bands = len(image.getbands())
            fill = self.fill if bands == 1 else (self.fill,) * bands
            canvas = Image.new(image.mode, (target_w, target_h), color=fill)
            canvas.paste(image, (left, top))
            image = canvas
        else:
            image = F.resize(image, [resized_h, resized_w])
            image = F.pad(image, [left, top, pad_w - left, pad_h - top], fill=self.fill)
        if target is None:
            return image
        target = dict(target)
        if 'boxes' in target:
            boxes = target['boxes'].clone()
            boxes[..., 0::2] *= resized_w / original_w
            boxes[..., 1::2] *= resized_h / original_h
            boxes[..., 0::2] += left
            boxes[..., 1::2] += top
            target['boxes'] = boxes
        target['letterbox_gain'] = torch.tensor([resized_w / original_w, resized_h / original_h], dtype=torch.float32)
        target['letterbox_pad'] = torch.tensor([left, top], dtype=torch.float32)
        target['size'] = torch.tensor([target_w, target_h])
        return image, target
