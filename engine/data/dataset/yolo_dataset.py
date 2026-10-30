import math
from pathlib import Path
import warnings

from faster_coco_eval import COCO
from PIL import Image
import torchvision
import yaml

from ...core import register
from .coco_dataset import CocoDetection, ConvertCocoPolysToMask


def read_yolo_metadata(data_yaml):
    yaml_path = Path(data_yaml).expanduser().resolve()
    with yaml_path.open(encoding='utf-8') as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f'{yaml_path}: expected a dataset mapping')
    names = metadata.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError(f'{yaml_path}: names must use consecutive integer keys starting at 0')
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f'{yaml_path}: names must contain nonempty class names')
    if metadata.get('nc', len(names)) != len(names):
        raise ValueError(f'{yaml_path}: nc does not match names')
    root = Path(metadata.get('path') or '.').expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return metadata, names, root.resolve()


def yolo_image_paths(metadata, root, split):
    sources = metadata.get(split)
    if not sources:
        raise ValueError(f'YOLO data.yaml has no {split!r} split')
    sources = sources if isinstance(sources, list) else [sources]
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.mpo'}
    for source in sources:
        path = Path(source).expanduser()
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(sorted(item.resolve() for item in path.rglob('*') if item.is_file() and item.suffix.lower() in extensions))
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                line = line.strip()
                if not line:
                    continue
                image = Path(line).expanduser()
                images.append((image if image.is_absolute() else path.parent / image).resolve())
        else:
            raise FileNotFoundError(f'YOLO split must be an image directory or image-list TXT: {path}')
    images = list(dict.fromkeys(images))
    if not images:
        raise ValueError(f'No images found in YOLO split {split!r}')
    for image in images:
        if not image.is_file() or image.suffix.lower() not in extensions:
            raise FileNotFoundError(f'Invalid YOLO image path: {image}')
    return images


def yolo_label_path(image):
    parts = list(image.parts)
    if 'images' in parts:
        index = len(parts) - 1 - parts[::-1].index('images')
        parts[index] = 'labels'
    return Path(*parts).with_suffix('.txt')


@register()
class YoloDetection(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['num_classes', 'remap_mscoco_category']

    def __init__(self, data_yaml, split='train', transforms=None, num_classes=None,
                 remap_mscoco_category=False):
        metadata, names, root = read_yolo_metadata(data_yaml)
        if remap_mscoco_category:
            raise ValueError('YOLO datasets require remap_mscoco_category=False')
        if num_classes is not None and num_classes != len(names):
            raise ValueError(f'num_classes={num_classes} does not match {len(names)} YOLO classes')
        torchvision.datasets.VisionDataset.__init__(self, str(root))
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)
        self.img_folder = str(root)
        self.ann_file = str(data_yaml)
        self.return_masks = False
        self.remap_mscoco_category = False
        self.class_names = names
        images = []
        annotations = []
        missing = 0
        for image_id, image_path in enumerate(yolo_image_paths(metadata, root, split), start=1):
            with Image.open(image_path) as image:
                width, height = image.size
            images.append(dict(id=image_id, file_name=str(image_path), width=width, height=height))
            label_path = yolo_label_path(image_path)
            if not label_path.exists():
                missing += 1
                continue
            for line_number, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(), start=1):
                if not line.strip():
                    continue
                location = f'{label_path}:{line_number}'
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'{location}: expected class cx cy width height; segmentation/OBB/pose are not supported')
                try:
                    category, center_x, center_y, box_width, box_height = map(float, fields)
                except ValueError as error:
                    raise ValueError(f'{location}: labels must be numeric') from error
                values = (category, center_x, center_y, box_width, box_height)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f'{location}: labels must be finite')
                if not category.is_integer() or not 0 <= category < len(names):
                    raise ValueError(f'{location}: class ID must be in 0..{len(names) - 1}')
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
                    raise ValueError(f'{location}: expected normalized coordinates and positive box sizes')
                left = max(0., (center_x - box_width / 2) * width)
                top = max(0., (center_y - box_height / 2) * height)
                right = min(float(width), (center_x + box_width / 2) * width)
                bottom = min(float(height), (center_y + box_height / 2) * height)
                annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                                        category_id=int(category), bbox=[left, top, right - left, bottom - top],
                                        area=(right - left) * (bottom - top), iscrowd=0))
        if missing:
            warnings.warn(f'YOLO {split}: {missing}/{len(images)} images have no label TXT; treated as background', stacklevel=2)
        self.coco = COCO()
        self.coco.dataset = dict(images=images, annotations=annotations,
                                 categories=[dict(id=index, name=name) for index, name in enumerate(names)],
                                 info={})
        self.coco.createIndex()
        self.ids = sorted(self.coco.imgs)
