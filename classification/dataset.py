"""Dataset and spectrogram-specific transforms for DIAT-uSAT."""

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


CLASS_NAMES = (
    "long_blade_rotor",
    "short_blade_rotor",
    "bird_mini_helicopter",
    "bird",
    "drone",
    "rc_plane",
)


class CropWhiteBorder:
    """Remove the white MATLAB figure margin without using label-dependent sizes."""

    def __init__(self, threshold: int = 245) -> None:
        self.threshold = threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        tensor = F.pil_to_tensor(image)
        foreground = tensor.amin(dim=0) < self.threshold
        rows = foreground.any(dim=1).nonzero(as_tuple=False).flatten()
        cols = foreground.any(dim=0).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0 or cols.numel() == 0:
            return image
        top, bottom = rows[[0, -1]].tolist()
        left, right = cols[[0, -1]].tolist()
        return F.crop(image, top, left, bottom - top + 1, right - left + 1)


class RandomTimeFrequencyMask:
    """Mask short time/frequency bands while preserving the spectrum geometry."""

    def __init__(self, probability: float = 0.5, max_fraction: float = 0.08) -> None:
        self.probability = probability
        self.max_fraction = max_fraction

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image

        _, height, width = image.shape
        output = image.clone()
        fill = output.mean(dim=(1, 2), keepdim=True)
        time_width = random.randint(0, max(1, int(width * self.max_fraction)))
        freq_height = random.randint(0, max(1, int(height * self.max_fraction)))
        if time_width:
            start = random.randint(0, width - time_width)
            output[:, :, start : start + time_width] = fill
        if freq_height:
            start = random.randint(0, height - freq_height)
            output[:, start : start + freq_height, :] = fill
        return output


def build_transform(image_size: tuple[int, int], training: bool, preprocessing: str = "diat"):
    if preprocessing not in ("diat", "rgb"):
        raise ValueError(f"Unknown image preprocessing: {preprocessing}")
    height, width = image_size
    operations = [CropWhiteBorder()] if preprocessing == "diat" else []
    operations.extend([
        v2.Resize((height, width), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    if training and preprocessing == "diat":
        operations.append(RandomTimeFrequencyMask())
    operations.append(
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    return v2.Compose(operations)


class DIATUSATDataset(Dataset):
    """Read a prepared DIAT-uSAT CSV manifest."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path = "manifest.csv",
        split: str = "train",
        image_size: tuple[int, int] = (288, 384),
        class_names: tuple[str, ...] = CLASS_NAMES,
        preprocessing: str = "diat",
    ) -> None:
        self.root = Path(root)
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_diat_usat.py first."
            )

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.samples = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.samples:
            raise ValueError(f"No samples for split '{split}' in {manifest_path}")

        self.split = split
        self.class_names = tuple(class_names)
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Class names must contain at least two unique names")
        if any(not 0 <= int(sample["label"]) < len(self.class_names) for sample in self.samples):
            raise ValueError("Manifest labels must match the configured class names")
        self.transform = build_transform(image_size, training=split == "train", preprocessing=preprocessing)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        path = self.root / sample["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(sample["label"])
