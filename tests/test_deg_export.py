"""DEG histogram export must preserve counts without requiring scatter plugins."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = ort = None


MODULE_PATH = Path(__file__).resolve().parents[1] / 'classification/modules/deg_axis_adapter.py'
spec = importlib.util.spec_from_file_location('deg_histogram_under_test', MODULE_PATH)
deg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deg)


class Histogram(torch.nn.Module):
    def forward(self, theta):
        return deg.cell_histogram(theta)


class DegExportTests(unittest.TestCase):
    def test_export_counts_match_scatter_with_padding_and_boundaries(self):
        generator = torch.Generator().manual_seed(42)
        for shape, bins, cell in [((1, 1, 80, 80), 36, 8), ((2, 1, 13, 17), 36, 8),
                                  ((1, 1, 1, 1), 36, 8), ((1, 1, 9, 7), 1, 4)]:
            with self.subTest(shape=shape, bins=bins, cell=cell):
                theta = torch.rand(shape, generator=generator) * torch.pi
                theta.flatten()[0] = 0
                theta.flatten()[-1] = torch.pi
                expected = deg.cell_histogram(theta, bins, cell)
                with patch('torch.onnx.is_in_onnx_export', return_value=True):
                    actual = deg.cell_histogram(theta, bins, cell)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # The final cell can contain only padding plus one valid pixel.
                self.assertTrue(torch.isfinite(actual).all())

    def test_padded_pixels_do_not_count_as_zero_angle(self):
        theta = torch.full((1, 1, 1, 1), torch.pi)
        with patch('torch.onnx.is_in_onnx_export', return_value=True):
            actual = deg.cell_histogram(theta)
        self.assertEqual(torch.count_nonzero(actual).item(), 1)
        self.assertEqual(actual[0, 0, 0, 0].item(), 0)
        self.assertGreater(actual[0, -1, 0, 0].item(), 3)

    @unittest.skipIf(onnx is None or ort is None, 'Optional ONNX dependencies not installed')
    def test_onnx_runtime_matches_without_scatter_nodes(self):
        theta = torch.rand((2, 1, 13, 17), generator=torch.Generator().manual_seed(7)) * torch.pi
        theta[0, 0, 0, 0] = 0
        theta[1, 0, 0, 0] = torch.pi
        expected = Histogram()(theta)
        buffer = io.BytesIO()
        torch.onnx.export(Histogram(), (theta,), buffer, opset_version=17,
                          input_names=['theta'], output_names=['histogram'], dynamo=False)
        graph = onnx.load_model_from_string(buffer.getvalue())
        onnx.checker.check_model(graph)
        self.assertFalse(any(node.op_type.startswith('Scatter') for node in graph.graph.node))
        session = ort.InferenceSession(buffer.getvalue(), providers=['CPUExecutionProvider'])
        actual = torch.from_numpy(session.run(None, {'theta': theta.numpy()})[0])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
