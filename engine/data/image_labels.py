"""Image-level labels for multi-task (detection + classification) training.

Each target dict may carry `image_label`: LongTensor of shape [1], -1 = no label.
Mosaic concatenates target tensors, so a mosaic carries several values; MixUp and
CopyBlend call `merge_image_labels`.  An image keeps its class only when every image
blended into it has the same class -- otherwise it is ignored by the classification loss.
"""

import torch

IGNORE = -1


def merge_image_labels(*labels: torch.Tensor) -> torch.Tensor:
    values = torch.cat([label.reshape(-1) for label in labels])
    if values.numel() and bool((values >= 0).all()) and bool((values == values[0]).all()):
        return values[:1].clone()
    return torch.tensor([IGNORE], dtype=torch.long)


def image_labels_from_targets(targets, device=None) -> torch.Tensor:
    """One label per image (-1 where there is none or the sources disagree)."""
    labels = []
    for target in targets:
        value = target.get('image_label')
        if value is None:
            labels.append(IGNORE)
            continue
        labels.append(int(merge_image_labels(value)[0]))
    return torch.tensor(labels, dtype=torch.long, device=device)
