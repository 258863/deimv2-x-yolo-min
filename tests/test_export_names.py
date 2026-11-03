"""Check export class overrides against the real YOLO configuration loader."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deployment_export', ROOT / 'tools/deployment/deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class ExportNamesTests(unittest.TestCase):
    def test_names_bypass_placeholder_dataset(self):
        from engine.core import YAMLConfig
        for names, count in [('capacitor', 1), ('hd,lx,sxbcj', 3)]:
            cfg = YAMLConfig(str(ROOT / 'configs/deimv2/deimv2_dinov3_x_yolo.yml'),
                             **deploy.export_options(names=names))
            self.assertIsNone(cfg.yaml_cfg['yolo_data'])
            self.assertEqual(cfg.yaml_cfg['num_classes'], count)
            self.assertFalse(cfg.yaml_cfg['remap_mscoco_category'])

    def test_metadata_names_resolution(self):
        self.assertEqual(deploy.resolve_names(stored=['capacitor']), ['capacitor'])
        self.assertEqual(deploy.resolve_names('hd,lx', ['hd', 'lx']), ['hd', 'lx'])
        self.assertIsNone(deploy.resolve_names())
        with self.assertRaisesRegex(ValueError, 'conflict'):
            deploy.resolve_names('lx,hd', ['hd', 'lx'])
        for value in [[], [0], [''], {'0': 'capacitor'}]:
            with self.assertRaises(ValueError):
                deploy.resolve_names(stored=value)

    def test_invalid_names_rejected(self):
        for names in ['', ' ', 'hd,', 'hd, hd']:
            with self.assertRaises(ValueError):
                deploy.export_options(names=names)

    def test_conflicting_sources_rejected(self):
        with self.assertRaises(ValueError):
            deploy.export_options(data='data.yaml', names='capacitor')

    def test_existing_data_override_preserved(self):
        self.assertEqual(deploy.export_options(data='data.yaml'),
                         {'yolo_data': str(Path('data.yaml').resolve())})
        self.assertEqual(deploy.export_options(), {})
