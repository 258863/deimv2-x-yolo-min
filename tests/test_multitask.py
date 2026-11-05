"""Detection + image classification with one shared backbone (configs/deimv2/*_multitask.yml)."""
from pathlib import Path
import tempfile
import unittest
import warnings

from PIL import Image
import torch
import torch.nn as nn
import yaml

from engine.data.dataset.yolo_dataset import YoloMultiTask
from engine.data.image_labels import image_labels_from_targets, merge_image_labels
from engine.deim.multitask import DEIMMultiTask, ImageClassificationMeter, MultiTaskCriterion


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deimv2/deimv2_dinov3_x_yolo_multitask.yml'
IMAGE_NAMES = ['OK', 'shift', 'tombstone']


def make_dataset(root: Path, labels='csv'):
    """Two splits; per split: ok.jpg (no boxes), shift.jpg (1 box), unlabeled.jpg (box, no class)."""
    rows = ['image,label']
    for split in ('train', 'val'):
        for name, box, cls in (('ok', None, 'OK'), ('shift', '0 0.5 0.5 0.4 0.5', 'shift'), ('unlabeled', '0 0.3 0.3 0.2 0.2', None)):
            sub = (cls or 'misc') if labels == 'folder' else ''
            folder = root / 'images' / split / sub
            label_dir = root / 'labels' / split / sub
            folder.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (100, 80), (90, 90, 90)).save(folder / f'{name}.jpg')
            (label_dir / f'{name}.txt').write_text((box + '\n') if box else '')
            if cls:
                rows.append(f'images/{split}/{sub + "/" if sub else ""}{name}.jpg,{cls}')
    (root / 'image_labels.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    data = dict(path='.', train='images/train', val='images/val', names=['defect'],
                image_names=IMAGE_NAMES, image_labels='folder' if labels == 'folder' else 'image_labels.csv')
    (root / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return root / 'data.yaml'


class LabelRulesTests(unittest.TestCase):
    def test_merge_keeps_class_only_when_sources_agree(self):
        t = lambda v: torch.tensor([v])
        self.assertEqual(merge_image_labels(t(2), t(2)).tolist(), [2])
        self.assertEqual(merge_image_labels(t(2), t(1)).tolist(), [-1])
        self.assertEqual(merge_image_labels(t(2), t(-1)).tolist(), [-1])

    def test_mosaic_concatenated_labels(self):
        targets = [{'image_label': torch.tensor([1, 1, 1, 1])}, {'image_label': torch.tensor([1, 0, 1, 1])},
                   {'image_label': torch.tensor([0])}, {}]
        self.assertEqual(image_labels_from_targets(targets).tolist(), [1, -1, 0, -1])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def labels_by_name(self, dataset):
        result = {}
        for index in range(len(dataset)):
            _, target = dataset.load_item(index)
            result[Path(dataset.coco.imgs[dataset.ids[index]]['file_name']).stem] = target['image_label'].tolist()
        return result

    def test_csv_labels(self):
        data_yaml = make_dataset(self.root, 'csv')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dataset = YoloMultiTask(data_yaml, split='train')
        self.assertEqual(dataset.image_class_names, IMAGE_NAMES)
        self.assertEqual(self.labels_by_name(dataset), {'ok': [0], 'shift': [1], 'unlabeled': [-1]})
        self.assertTrue(any('detection only' in str(w.message) for w in caught))

    def test_folder_labels(self):
        data_yaml = make_dataset(self.root, 'folder')
        dataset = YoloMultiTask(data_yaml, split='val')
        self.assertEqual(self.labels_by_name(dataset), {'ok': [0], 'shift': [1], 'unlabeled': [-1]})
        _, target = dataset.load_item([i for i in range(len(dataset))
                                       if 'shift' in dataset.coco.imgs[dataset.ids[i]]['file_name']][0])
        self.assertEqual(len(target['boxes']), 1)

    def test_crop_that_removes_every_box_drops_the_class(self):
        data_yaml = make_dataset(self.root, 'csv')

        def crop_everything_away(image, target, dataset):
            target = dict(target, boxes=target['boxes'][:0], labels=target['labels'][:0])
            return image, target, dataset

        dataset = YoloMultiTask(data_yaml, split='train', transforms=crop_everything_away)
        by_name = {Path(dataset.coco.imgs[dataset.ids[i]]['file_name']).stem: dataset[i][1]['image_label'].tolist()
                   for i in range(len(dataset))}
        self.assertEqual(by_name['shift'], [-1])  # had a box, lost it
        self.assertEqual(by_name['ok'], [0])      # never had boxes: class kept

    def test_unknown_label_rejected(self):
        data_yaml = make_dataset(self.root, 'csv')
        (self.root / 'image_labels.csv').write_text('image,label\nimages/train/ok.jpg,nonsense\n')
        with self.assertRaisesRegex(ValueError, 'unknown image label'):
            YoloMultiTask(data_yaml, split='train')


class FakeDetCriterion(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return {'loss_det': outputs['pred_logits'].sum() * 0 + 1.0}


class CriterionTests(unittest.TestCase):
    def test_cross_entropy_ignores_unlabeled_images(self):
        criterion = MultiTaskCriterion(FakeDetCriterion(), image_cls_weight=2.0, num_image_classes=3)
        logits = torch.zeros(3, 3, requires_grad=True)
        outputs = {'pred_logits': torch.zeros(3, 1, 1), 'image_logits': logits}
        targets = [{'image_label': torch.tensor([0])}, {'image_label': torch.tensor([-1])}, {'image_label': torch.tensor([2])}]
        losses = criterion(outputs, targets)
        self.assertAlmostEqual(losses['loss_image_cls'].item(), 2.0 * torch.log(torch.tensor(3.0)).item(), places=5)
        sum(losses.values()).backward()
        self.assertTrue(torch.all(logits.grad[1] == 0))

    def test_batch_without_labels_still_has_a_loss(self):
        criterion = MultiTaskCriterion(FakeDetCriterion(), num_image_classes=3)
        logits = torch.zeros(2, 3, requires_grad=True)
        losses = criterion({'pred_logits': torch.zeros(2, 1, 1), 'image_logits': logits},
                           [{'image_label': torch.tensor([-1])}, {}])
        self.assertEqual(losses['loss_image_cls'].item(), 0.0)
        losses['loss_image_cls'].backward()  # graph exists (needed for DDP)


class MeterTests(unittest.TestCase):
    def test_accuracy_and_macro_f1(self):
        meter = ImageClassificationMeter(3)
        logits = torch.eye(3)[[0, 1, 1, 2]]
        meter.update(logits, [{'image_label': torch.tensor([v])} for v in (0, 1, 2, -1)])
        summary = meter.summary()
        self.assertEqual(summary['images'], 3)
        self.assertAlmostEqual(summary['accuracy'], 2 / 3)
        self.assertEqual(summary['confusion_matrix'], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])


class TinyModelTests(unittest.TestCase):
    def test_head_sees_backbone_features(self):
        class Backbone(nn.Module):
            def forward(self, x):
                return [torch.ones(x.shape[0], 4, 2, 2) * (i + 1) for i in range(3)]

        class Passthrough(nn.Module):
            def forward(self, x, targets=None):
                return x

        class Decoder(nn.Module):
            def forward(self, feats, targets=None):
                return {'pred_logits': feats[0].new_zeros(feats[0].shape[0], 5, 2), 'pred_boxes': feats[0].new_zeros(feats[0].shape[0], 5, 4)}

        model = DEIMMultiTask(Backbone(), Passthrough(), Decoder(), num_image_classes=3, cls_in_channels=[4, 4, 4])
        outputs = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(list(outputs), ['pred_logits', 'pred_boxes', 'image_logits'])
        self.assertEqual(tuple(outputs['image_logits'].shape), (2, 3))
        with self.assertRaises(ValueError):
            DEIMMultiTask(Backbone(), Passthrough(), Decoder(), num_image_classes=0)


class RealConfigTests(unittest.TestCase):
    """Builds the real X model from the multi-task config (CPU, random weights) and runs one step."""

    def test_config_builds_and_trains_one_step(self):
        from engine.core import YAMLConfig
        with tempfile.TemporaryDirectory() as folder:
            data_yaml = make_dataset(Path(folder), 'csv')
            cfg = YAMLConfig(str(CONFIG), yolo_data=str(data_yaml), DINOv3STAs={'weights_path': None})
            self.assertEqual(cfg.yaml_cfg['num_image_classes'], 3)
            model, criterion = cfg.model, cfg.criterion
            self.assertIsInstance(model, DEIMMultiTask)
            self.assertIsInstance(criterion, MultiTaskCriterion)
            self.assertEqual(model.image_head.mlp[-1].out_features, 3)
            self.assertEqual(cfg.yaml_cfg['train_dataloader']['dataset']['type'], 'YoloMultiTask')

            model.train()
            images = torch.rand(2, 3, 640, 640)
            targets = [
                {'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]), 'image_label': torch.tensor([1])},
                {'labels': torch.zeros(0, dtype=torch.long), 'boxes': torch.zeros(0, 4), 'image_label': torch.tensor([0])},
            ]
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
            self.assertIn('loss_image_cls', losses)
            self.assertIn('loss_mal', losses)
            sum(losses.values()).backward()
            self.assertIsNotNone(model.image_head.mlp[-1].weight.grad)

            model.eval()
            with torch.no_grad():
                outputs = model(images[:1])
            self.assertEqual(tuple(outputs['image_logits'].shape), (1, 3))

    def test_plain_detection_checkpoint_loads(self):
        """A DEIM checkpoint loads into the multi-task model; only image_head is new."""
        from engine.core import YAMLConfig
        with tempfile.TemporaryDirectory() as folder:
            data_yaml = make_dataset(Path(folder), 'csv')
            multitask = YAMLConfig(str(CONFIG), yolo_data=str(data_yaml), DINOv3STAs={'weights_path': None}).model
            plain = YAMLConfig(str(ROOT / 'configs/deimv2/deimv2_dinov3_x_yolo.yml'), yolo_data=str(data_yaml),
                               DINOv3STAs={'weights_path': None}).model
            missing, unexpected = multitask.load_state_dict(plain.state_dict(), strict=False)
            self.assertEqual(unexpected, [])
            self.assertTrue(missing and all(key.startswith('image_head.') for key in missing))


if __name__ == '__main__':
    unittest.main()
