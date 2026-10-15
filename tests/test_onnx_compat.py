"""In-memory opset-17 -> 16 conversion for the Jetson Nano's ONNX Runtime 1.11."""
import importlib.util
import unittest

import numpy as np

from inspection.local.onnx_compat import runtime_max_opset

HAVE_ONNX = importlib.util.find_spec('onnx') is not None


class RuntimeOpsetTests(unittest.TestCase):
    def test_runtimes_before_1_12_stop_at_opset_16(self):
        for version, opset in (('1.11.0', 16), ('1.11.1', 16), ('1.12.0', 17), ('1.16.3', 17),
                               ('1.31.0', 17), ('1.11.0+cu102', 16)):
            self.assertEqual(runtime_max_opset(version), opset, version)


@unittest.skipUnless(HAVE_ONNX, 'the onnx package is only installed on the Jetson')
class LoweringTests(unittest.TestCase):
    def model(self, bias=True, axis=-1, epsilon=1e-5):
        from onnx import helper, numpy_helper, TensorProto
        rng = np.random.RandomState(0)
        inputs = ['x', 'scale'] + (['bias'] if bias else [])
        init = [numpy_helper.from_array(rng.rand(8).astype(np.float32) + .5, 'scale')]
        if bias:
            init.append(numpy_helper.from_array(rng.randn(8).astype(np.float32), 'bias'))
        graph = helper.make_graph(
            [helper.make_node('LayerNormalization', inputs, ['y'], axis=axis, epsilon=epsilon),
             helper.make_node('Relu', ['y'], ['out'])],
            'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5, 8])],
            [helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 5, 8])], init)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)

    def run_model(self, model_bytes, x):
        import onnxruntime as ort
        session = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
        return session.run(None, {'x': x})[0]

    def convert(self, model, version='1.11.0'):
        import tempfile, os, onnx
        from inspection.local.onnx_compat import compatible_model
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'm.onnx')
            onnx.save(model, path)
            return compatible_model(path, version)

    def test_layer_norm_is_written_out_and_gives_the_same_result(self):
        import onnx
        x = np.random.RandomState(1).randn(2, 5, 8).astype(np.float32) * 3
        for bias in (True, False):
            model = self.model(bias=bias, epsilon=1e-3)
            converted = self.convert(model)
            lowered = onnx.load_from_string(converted)
            self.assertNotIn('LayerNormalization', {n.op_type for n in lowered.graph.node})
            self.assertEqual([o.version for o in lowered.opset_import if o.domain == ''], [16])
            np.testing.assert_allclose(self.run_model(converted, x),
                                       self.run_model(model.SerializeToString(), x),
                                       rtol=1e-5, atol=1e-5)

    def test_a_runtime_that_reads_opset_17_gets_the_file_unchanged(self):
        self.assertIsNone(self.convert(self.model(), '1.16.3'))

    def test_a_positive_axis_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            self.convert(self.model(axis=2))


if __name__ == '__main__':
    unittest.main()
