from pathlib import Path
import gc
import tempfile
import unittest

from PIL import Image
import torch
import yaml

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator
from engine.data.dataset.coco_utils import get_coco_api_from_dataset
from engine.data.dataset.yolo_dataset import YoloDetection
from engine.data.transforms import Compose


ROOT = Path(__file__).resolve().parents[1]


class YoloDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for split in ('train', 'val'):
            (self.root / 'images' / split).mkdir(parents=True)
            (self.root / 'labels' / split).mkdir(parents=True)
            for name in ('object', 'empty'):
                Image.new('RGB', (100, 80)).save(self.root / 'images' / split / f'{name}.jpg')
            (self.root / 'labels' / split / 'object.txt').write_text('1 0.5 0.5 0.4 0.5\n')
            (self.root / 'labels' / split / 'empty.txt').write_text('')
        self.data_yaml = self.root / 'data.yaml'
        self.metadata = dict(path='.', train='images/train', val='images/val', names=['first', 'second'])
        self.write_metadata()

    def write_metadata(self):
        self.data_yaml.write_text(yaml.safe_dump(self.metadata))

    def test_boxes_background_and_category_ids(self):
        dataset = YoloDetection(self.data_yaml, split='val')
        _, empty = dataset[0]
        _, target = dataset[1]
        torch.testing.assert_close(target['boxes'], torch.tensor([[30., 20., 70., 60.]]))
        self.assertEqual(target['labels'].tolist(), [1])
        self.assertEqual(target['area'].tolist(), [1600.])
        self.assertEqual(empty['boxes'].shape, (0, 4))
        self.assertEqual(dataset.category2name, {0: 'first', 1: 'second'})
        self.assertIs(get_coco_api_from_dataset(dataset), dataset.coco)
        self.assertEqual(list(self.root.rglob('*.json')), [])

    def test_list_paths_names_mapping_and_missing_labels(self):
        (self.root / 'train.txt').write_text('./images/train/object.jpg\n./images/train/empty.jpg\n')
        self.metadata.update(train=['train.txt', 'images/train'], names={0: 'first', 1: 'second'})
        self.write_metadata()
        (self.root / 'labels/train/empty.txt').unlink()
        with self.assertWarnsRegex(UserWarning, 'treated as background'):
            dataset = YoloDetection(self.data_yaml)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][1]['labels'].tolist(), [1])

    def test_invalid_labels_fail_with_filename(self):
        for label in ('2 0.5 0.5 0.4 0.5', '0.5 0.5 0.5 0.4 0.5',
                      '1 nan 0.5 0.4 0.5', '1 2 0.5 0.4 0.5',
                      '1 0.5 0.5 0 0.5', '1 0 0 1 1 0 1'):
            with self.subTest(label=label):
                (self.root / 'labels/train/object.txt').write_text(label)
                with self.assertRaisesRegex(ValueError, 'object.txt:1'):
                    YoloDetection(self.data_yaml)

    def test_metadata_and_class_count_validation(self):
        with self.assertRaisesRegex(ValueError, 'num_classes'):
            YoloDetection(self.data_yaml, num_classes=3)
        with self.assertRaisesRegex(ValueError, 'remap_mscoco_category'):
            YoloDetection(self.data_yaml, remap_mscoco_category=True)
        self.metadata['nc'] = 3
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'nc does not match'):
            YoloDetection(self.data_yaml)

    def test_mosaic_uses_raw_yolo_samples(self):
        transforms = Compose(ops=[dict(type='Mosaic', output_size=64, use_cache=False),
                                  dict(type='Resize', size=[64, 64]),
                                  dict(type='ConvertPILImage', dtype='float32', scale=True),
                                  dict(type='ConvertBoxes', fmt='cxcywh', normalize=True)])
        dataset = YoloDetection(self.data_yaml, transforms=transforms)
        image, target = dataset[1]
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertTrue(torch.isfinite(target['boxes']).all())
        self.assertTrue((target['labels'] == 1).all())

    def test_perfect_predictions_score_one_with_background_images(self):
        dataset = YoloDetection(self.data_yaml, split='val')
        evaluator = CocoEvaluator(get_coco_api_from_dataset(dataset), ['bbox'])
        predictions = {}
        for index in range(len(dataset)):
            _, target = dataset.load_item(index)
            predictions[target['image_id'].item()] = dict(
                boxes=target['boxes'].as_subclass(torch.Tensor), labels=target['labels'],
                scores=torch.ones(len(target['labels'])))
        evaluator.update(predictions)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        self.assertAlmostEqual(evaluator.coco_eval['bbox'].stats[0], 1., places=5)

    def test_l_x_configs_build_loaders_and_custom_heads_without_leaking_to_coco(self):
        for size in ('l', 'x'):
            with self.subTest(size=size):
                config = YAMLConfig(str(ROOT / f'configs/deimv2/deimv2_dinov3_{size}_yolo.yml'),
                                    yolo_data=str(self.data_yaml))
                self.assertEqual(config.yaml_cfg['num_classes'], 2)
                self.assertFalse(config.yaml_cfg['remap_mscoco_category'])
                config.yaml_cfg['DINOv3STAs']['weights_path'] = None
                for split in ('train', 'val'):
                    config.yaml_cfg[f'{split}_dataloader']['num_workers'] = 0
                model = config.model
                self.assertEqual(model.decoder.enc_score_head.out_features, 2)
                self.assertEqual(config.postprocessor.num_classes, 2)
                loader = config.train_dataloader
                loader.set_epoch(0)
                images, targets = next(iter(loader))
                self.assertEqual(images.shape[0], 2)
                self.assertTrue(all(target['labels'].numel() == 0 or target['labels'].max() < 2 for target in targets))
                self.assertTrue(all(((target['boxes'] >= 0) & (target['boxes'] <= 1)).all() for target in targets))
                validation = config.val_dataloader
                images, targets = next(iter(validation))
                self.assertEqual(images.shape, (2, 3, 640, 640))
                self.assertEqual(len(config.evaluator.coco_gt.imgs), 2)
                del model, config, loader, validation
                gc.collect()
        coco_config = YAMLConfig(str(ROOT / 'configs/deimv2/deimv2_dinov3_l_coco.yml'))
        self.assertNotIn('yolo_data', coco_config.yaml_cfg)
        self.assertEqual(coco_config.yaml_cfg['num_classes'], 80)
        self.assertEqual(coco_config.yaml_cfg['train_dataloader']['dataset']['type'], 'CocoDetection')


if __name__ == '__main__':
    unittest.main()
