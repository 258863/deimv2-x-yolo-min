"""CPU-only checks for deployment's fail-closed graph contract."""
import importlib.util
from pathlib import Path
import unittest

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, 'Optional deployment dependencies not installed')
class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'tools/deployment/layout.py'
        spec = importlib.util.spec_from_file_location('deployment_layout_test', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def model(self, size=640, input_name='images'):
        graph = helper.make_graph([], 'invalid',
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1,3,size,size])],
            [helper.make_tensor_value_info('pred_logits',TensorProto.FLOAT,[1,300,3]),
             helper.make_tensor_value_info('pred_boxes',TensorProto.FLOAT,[1,300,4])])
        return helper.make_model(graph)

    def test_wrong_shape_rejected_before_running_graph(self):
        with self.assertRaisesRegex(ValueError, '640'):
            self.module.rewrite_layout(self.model(size=480))

    def test_unknown_graph_rejected_instead_of_guessed(self):
        model = self.model()
        original = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, 'Slice'):
            self.module.rewrite_layout(model)
        self.assertEqual(model.SerializeToString(), original)

    def test_wrong_input_name_rejected(self):
        with self.assertRaisesRegex(ValueError, 'images'):
            self.module.rewrite_layout(self.model(input_name='input'))


if __name__ == '__main__':
    unittest.main()
