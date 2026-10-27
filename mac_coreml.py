"""MAC ONLY: run a model through its Core ML package when one is present.

convert_models_mac.py writes <model>.mlpackage next to each model file. On
macOS these adapters load that package so Core ML can use the Apple GPU and
Neural Engine; everywhere else, or without a package, the original model is
used exactly as before. FLOD_COREML=0 forces the originals for comparison.
"""
import os
import sys
import threading
from pathlib import Path
import numpy as np


def coreml_package(model_path):
    """The Core ML package for model_path, or None when it must not be used."""
    if sys.platform != 'darwin' or os.environ.get('FLOD_COREML', '1') == '0':
        return None
    package = Path(model_path).with_suffix('.mlpackage')
    if not package.is_dir():
        return None
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print(f'[coreml] coremltools missing; {Path(model_path).name} stays on the CPU')
        return None
    return package


def load_package(package):
    import coremltools as ct
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    shape = tuple(model.get_spec().description.input[0].type.multiArrayType.shape)
    print(f'[coreml] {package.name} loaded for input {shape}')
    return model, shape


class _Value:
    """Enough of onnxruntime's NodeArg for the callers that inspect inputs."""
    def __init__(self, name, shape, type='tensor(float)'):
        self.name, self.shape, self.type = name, list(shape), type


class CoreMLSession:
    """onnxruntime.InferenceSession's run/get_inputs over a fixed-shape package.

    Inputs of any other shape go to `fallback()`, an ONNX Runtime session that
    is only built when first needed, so an unexpected frame size still works.
    """

    def __init__(self, package, onnx_session_factory):
        self.model, self.shape = load_package(package)
        self.fallback_factory = onnx_session_factory
        self.fallback_session = None
        self.lock = threading.Lock()
        self.input_name, self.output_name = 'input', 'output'

    def get_inputs(self):
        return [_Value(self.input_name, self.shape)]

    def get_outputs(self):
        return [_Value(self.output_name, ())]

    def get_providers(self):
        return ['CoreML']

    def fallback(self):
        with self.lock:
            if self.fallback_session is None:
                self.fallback_session = self.fallback_factory()
            return self.fallback_session

    def run(self, output_names, feeds):
        (x,) = feeds.values()
        if tuple(x.shape) != self.shape:
            print(f'[coreml] input {tuple(x.shape)} is not {self.shape}; using the original model')
            session = self.fallback()
            return session.run(None, {session.get_inputs()[0].name: x})
        with self.lock:
            scores = self.model.predict({self.input_name: x})['output']
        return [np.asarray(scores, np.float32)]


class CoreMLKeras:
    """Callable like the Keras model: model(batch, training=False) -> scores."""

    def __init__(self, package):
        self.model, self.shape = load_package(package)
        self.lock = threading.Lock()

    def __call__(self, batch, training=False):
        with self.lock:
            return self.model.predict({'input': np.asarray(batch, np.float32)})['output']
