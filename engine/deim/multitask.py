"""
Detection + image classification sharing one backbone.

    image -> DINOv3STAs (shared) -> 3 feature maps -+-> HybridEncoder -> DEIMTransformer -> boxes
                                                    +-> global pool + MLP               -> image class

Nothing in the detection path changes: a DEIM checkpoint loads into DEIMMultiTask with
only `image_head.*` left at its initial values, and exporting the detection outputs works
exactly as before.

Image labels travel in each target dict as `image_label` (LongTensor, shape [1]).
-1 means "no classification label for this image" and is ignored by the loss, so
detection-only and labelled images can be mixed in one dataset.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from ..data.image_labels import image_labels_from_targets, merge_image_labels  # noqa: F401 (re-exported)
from .deim import DEIM


__all__ = ['DEIMMultiTask', 'ImageClassificationHead', 'ImageClassificationMeter', 'MultiTaskCriterion',
           'image_labels_from_targets', 'merge_image_labels']



class ImageClassificationHead(nn.Module):
    """Same head as classification/model.py: mean-pool each level, concat, LayerNorm, MLP."""

    def __init__(self, in_channels: Sequence[int], num_classes: int, hidden_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        feature_dim = int(sum(in_channels))
        layers: List[nn.Module] = [nn.LayerNorm(feature_dim), nn.Dropout(dropout)]
        if hidden_dim:
            layers += [nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            feature_dim = hidden_dim
        layers.append(nn.Linear(feature_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        nn.init.trunc_normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat([feature.mean(dim=(-2, -1)) for feature in features], dim=1)
        return self.mlp(pooled)


@register()
class DEIMMultiTask(DEIM):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    __share__ = ['num_image_classes', ]

    def __init__(self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_image_classes: int = 0,
        cls_in_channels: Sequence[int] = (256, 256, 256),
        cls_hidden_dim: int = 0,
        cls_dropout: float = 0.2,
        cls_source: str = 'backbone',
    ):
        super().__init__(backbone, encoder, decoder)
        if not num_image_classes or num_image_classes < 2:
            raise ValueError('DEIMMultiTask needs num_image_classes >= 2 '
                             '(set image_names in data.yaml or num_image_classes in the config)')
        if cls_source not in ('backbone', 'encoder'):
            raise ValueError("cls_source must be 'backbone' or 'encoder'")
        self.num_image_classes = int(num_image_classes)
        self.cls_source = cls_source
        self.image_head = ImageClassificationHead(cls_in_channels, self.num_image_classes,
                                                  cls_hidden_dim, cls_dropout)

    def forward(self, x, targets=None):
        features = self.backbone(x)
        encoded = self.encoder(features)
        outputs = self.decoder(encoded, targets)
        # Added last so `next(iter(outputs.values()))` in the criterion is still a detection tensor.
        outputs['image_logits'] = self.image_head(features if self.cls_source == 'backbone' else encoded)
        return outputs


@register()
class MultiTaskCriterion(nn.Module):
    """Detection losses from the wrapped criterion + weighted cross-entropy on image labels."""
    __inject__ = ['det_criterion', ]
    __share__ = ['num_image_classes', ]

    def __init__(self, det_criterion=None, image_cls_weight: float = 1.0, label_smoothing: float = 0.0,
                 class_weights=None, num_image_classes: int = 0):
        super().__init__()
        if det_criterion is None:
            raise ValueError('MultiTaskCriterion needs det_criterion (e.g. DEIMCriterion)')
        self.det_criterion = det_criterion
        self.image_cls_weight = float(image_cls_weight)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if num_image_classes and class_weights.numel() != num_image_classes:
                raise ValueError('class_weights must have one value per image class')
            self.register_buffer('class_weights', class_weights, persistent=False)
        else:
            self.class_weights = None

    def forward(self, outputs, targets, **kwargs):
        losses = self.det_criterion(outputs, targets, **kwargs)
        logits = outputs.get('image_logits')
        if logits is None:
            return losses
        logits = logits.float()
        labels = image_labels_from_targets(targets, logits.device)
        valid = labels >= 0
        if valid.any():
            loss = F.cross_entropy(logits[valid], labels[valid], weight=self.class_weights,
                                   label_smoothing=self.label_smoothing)
        else:
            loss = logits.sum() * 0.0  # keeps the head in the graph (DDP) when a batch has no labels
        losses['loss_image_cls'] = loss * self.image_cls_weight
        return losses


class ImageClassificationMeter:
    """Confusion matrix over images that have a label; summed across processes."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, targets) -> None:
        labels = image_labels_from_targets(targets, 'cpu')
        predictions = logits.detach().float().argmax(dim=-1).cpu()
        valid = labels >= 0
        index = labels[valid] * self.num_classes + predictions[valid]
        self.matrix += torch.bincount(index, minlength=self.num_classes ** 2).reshape(self.num_classes, -1)

    def synchronize(self) -> None:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            matrix = self.matrix.cuda() if torch.cuda.is_available() else self.matrix
            dist.all_reduce(matrix)
            self.matrix = matrix.cpu()

    def summary(self) -> dict:
        matrix = self.matrix.double()
        total = matrix.sum()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        correct = matrix.diag()
        recall = torch.where(support > 0, correct / support.clamp(min=1), torch.zeros_like(correct))
        precision = torch.where(predicted > 0, correct / predicted.clamp(min=1), torch.zeros_like(correct))
        f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp(min=1e-12),
                         torch.zeros_like(correct))
        present = support > 0
        return {
            'images': int(total.item()),
            'accuracy': float(correct.sum() / total) if total > 0 else 0.0,
            'macro_f1': float(f1[present].mean()) if present.any() else 0.0,
            'per_class_recall': [float(v) for v in recall],
            'confusion_matrix': self.matrix.tolist(),
        }
