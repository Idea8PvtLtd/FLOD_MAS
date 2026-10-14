"""Where the local models run: local_inspection.execution_provider."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from config_validation import validate_config
from inspection.local import ort_providers
from inspection.local.ort_providers import provider_list

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config.json').read_text())


def names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]


class ProviderListTests(unittest.TestCase):
    def test_the_default_is_the_cpu(self):
        self.assertEqual(CONFIG['local_inspection']['execution_provider'], 'cpu')
        self.assertEqual(names(provider_list()), ['CPUExecutionProvider'])

    def test_cuda_falls_back_to_the_cpu(self):
        self.assertEqual(names(provider_list('cuda')), ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def test_tensorrt_keeps_full_precision_and_caches_engines(self):
        providers = provider_list('tensorrt', '/cache')
        self.assertEqual(names(providers),
                         ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        options = providers[0][1]
        self.assertFalse(options['trt_fp16_enable'])
        self.assertTrue(options['trt_engine_cache_enable'])
        self.assertEqual(options['trt_engine_cache_path'], '/cache')

    def test_an_unknown_provider_is_refused(self):
        with self.assertRaises(ValueError):
            provider_list('gpu')

    def test_providers_the_runtime_lacks_are_left_out(self):
        class Runtime:
            @staticmethod
            def get_available_providers():
                return ['CPUExecutionProvider']

            @staticmethod
            def preload_dlls():
                pass

            @staticmethod
            def InferenceSession(path, providers):
                return providers
        with patch.dict('sys.modules', onnxruntime=Runtime):
            self.assertEqual(ort_providers.create_session('model.onnx', 'cuda'), ['CPUExecutionProvider'])


class ValidationTests(unittest.TestCase):
    def test_each_known_provider_is_accepted(self):
        for kind in ('cpu', 'cuda', 'tensorrt'):
            config = copy.deepcopy(CONFIG)
            config['local_inspection']['execution_provider'] = kind
            validate_config(config)

    def test_an_unknown_provider_is_rejected(self):
        config = copy.deepcopy(CONFIG)
        config['local_inspection']['execution_provider'] = 'gpu'
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()
