"""Checkpoint class metadata and resume semantics without training a model."""
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from engine.solver._solver import BaseSolver


class CheckpointNamesTests(unittest.TestCase):
    def solver(self, names=None):
        solver = BaseSolver.__new__(BaseSolver)
        solver.last_epoch = 2
        solver.model = torch.nn.Linear(2, 1)
        solver.ema = torch.nn.Linear(2, 1)
        if names is not None:
            solver.class_names = names
        return solver

    def test_training_reads_names_from_actual_loader(self):
        dataset = SimpleNamespace(class_names=['lx', 'hd'])
        loader = SimpleNamespace(dataset=dataset, shuffle=False)
        cfg = SimpleNamespace(optimizer=None, lr_scheduler=None,
                              lr_warmup_scheduler=None, train_dataloader=loader,
                              val_dataloader=loader, evaluator=None, resume=None)
        solver = BaseSolver(cfg)
        with patch.object(solver, '_setup'), patch(
                'engine.solver._solver.dist_utils.warp_loader', side_effect=lambda x, **kw: x):
            solver.train()
        dataset.class_names[0] = 'changed_after_setup'
        self.assertEqual(solver.class_names, ['lx', 'hd'])

    def test_serialized_checkpoint_retains_names_and_ema(self):
        solver = self.solver(['capacitor'])
        state = solver.state_dict()
        solver.class_names[0] = 'changed_after_save'
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        self.assertEqual(loaded['names'], ['capacitor'])
        self.assertEqual(loaded['num_classes'], 1)
        self.assertIn('model', loaded)
        self.assertIn('ema', loaded)

    def test_reordered_classes_rejected_before_loading(self):
        solver = self.solver(['hd', 'lx'])
        with self.assertRaisesRegex(ValueError, 'class names/order'):
            solver.load_state_dict({'names': ['lx', 'hd'], 'last_epoch': 99})
        self.assertEqual(solver.last_epoch, 2)

    def test_old_checkpoints_still_load(self):
        solver = self.solver(['capacitor'])
        state = solver.state_dict()
        del state['names'], state['num_classes']
        solver.load_state_dict(state)
        self.assertEqual(solver.class_names, ['capacitor'])

    def test_model_name_is_separate_from_class_names(self):
        solver = self.solver(['capacitor'])
        solver.model_name = '电容检测模型_v1'
        state = solver.state_dict()
        self.assertEqual(state['model_name'], '电容检测模型_v1')
        self.assertEqual(state['names'], ['capacitor'])
        resumed = self.solver(['capacitor'])
        resumed.load_state_dict(state)
        self.assertEqual(resumed.model_name, '电容检测模型_v1')
        resumed.model_name = '显式指定的新名称'
        resumed.load_state_dict(state)
        self.assertEqual(resumed.model_name, '显式指定的新名称')

    def test_model_name_config_validation(self):
        self.assertEqual(BaseSolver(SimpleNamespace(yaml_cfg={'model_name': ' 电容模型 '})).model_name,
                         '电容模型')
        for name in ['', ' ', 123]:
            with self.assertRaises(ValueError):
                BaseSolver(SimpleNamespace(yaml_cfg={'model_name': name}))

    def test_no_names_are_invented_for_other_datasets(self):
        self.assertNotIn('names', self.solver().state_dict())
