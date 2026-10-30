"""Classification components built on the DEIMv2 backbone."""

from .dataset import CLASS_NAMES, DIATUSATDataset
from .model import DEIMv2SpectrogramClassifier

__all__ = ["CLASS_NAMES", "DIATUSATDataset", "DEIMv2SpectrogramClassifier"]
