"""MAC ONLY: convert the Windows model files to Core ML packages (.mlpackage).

    python convert_models_mac.py                # every model, default precisions
    python convert_models_mac.py --check        # also compare against the originals

Each package is written next to the model it came from, with the same stem
(strip_unet_resnet34.onnx -> strip_unet_resnet34.mlpackage). mac_coreml.py
picks a package up automatically on macOS, so Core ML runs the model on the
Apple GPU / Neural Engine. Remove a package, or set FLOD_COREML=0, to run the
original file on the CPU again. The original files are never modified.

Core ML needs fixed input shapes, so each package is built for the shape the
application feeds it. A different shape falls back to the original model.

Needs (conversion only): coremltools onnx onnx2torch torch tensorflow tf-keras.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
LOCAL = CONFIG['local_inspection']


def source_shape():
    """The padded source-model input for the first camera's frame size."""
    camera = CONFIG['cameras'][0]
    width, height = int(camera['width']), int(camera['height'])
    if int(camera.get('rotation', 0)) in (90, 270):
        width, height = height, width
    train_width = int(LOCAL['source_width'])
    scaled = round(height * train_width / width)
    pad = lambda value: value + (32 - value % 32) % 32
    return (1, 3, pad(scaled), pad(train_width))


def glue_shape():
    # SegformerOnnx's defaults; the exported model itself only accepts 64 x 1024.
    return (1, 3, 64, 1024)


def onnx_to_coreml(onnx_path, shape, precision):
    """ONNX -> fixed-shape ONNX -> PyTorch -> traced -> Core ML ML program."""
    import coremltools as ct
    import onnx
    import onnx2torch
    import onnxruntime as ort
    import torch
    from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

    model = onnx.load(str(onnx_path))
    input_name = model.graph.input[0].name
    make_input_shape_fixed(model.graph, input_name, list(shape))
    fix_output_shapes(model)
    with tempfile.TemporaryDirectory() as folder:
        fixed, folded = Path(folder) / 'fixed.onnx', Path(folder) / 'folded.onnx'
        onnx.save(model, str(fixed))
        # Basic optimisation folds the shape arithmetic into constants without
        # introducing Microsoft-only fused operators the converter cannot read.
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(folded)
        ort.InferenceSession(str(fixed), options, providers=['CPUExecutionProvider'])
        network = onnx2torch.convert(str(folded)).eval()
    example = torch.zeros(shape)
    with torch.no_grad():
        traced = torch.jit.trace(network, example)
    return ct.convert(
        traced, inputs=[ct.TensorType(name='input', shape=shape)],
        outputs=[ct.TensorType(name='output')], convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS14, compute_precision=precision)


def keras_to_coreml(h5_path, precision):
    import coremltools as ct
    import tensorflow as tf
    import tf_keras
    model = tf_keras.models.load_model(str(h5_path), compile=False)
    shape = (1,) + tuple(model.input_shape[1:])
    function = tf.function(lambda image: model(image, training=False)).get_concrete_function(
        tf.TensorSpec(shape, tf.float32, name='input'))
    converted = ct.convert(
        [function], source='tensorflow', inputs=[ct.TensorType(name='input', shape=shape)],
        convert_to='mlprogram', minimum_deployment_target=ct.target.macOS14,
        compute_precision=precision)
    # The TensorFlow frontend names the output Identity; renamed to match the others.
    ct.utils.rename_feature(converted._spec, converted._spec.description.output[0].name, 'output')
    return ct.models.MLModel(converted._spec, weights_dir=converted.weights_dir), shape


def compare(original, package, shape, kind):
    """Random-input agreement; the real-frame comparison is benchmark_mac.py."""
    import coremltools as ct
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    if kind == 'keras':
        import tf_keras
        x = np.clip(x, -1, 1)
        expected = np.asarray(tf_keras.models.load_model(str(original), compile=False)(x, training=False))
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(str(original), providers=['CPUExecutionProvider'])
        expected = session.run(None, {session.get_inputs()[0].name: x})[0]
    actual = ct.models.MLModel(str(package)).predict({'input': x})['output']
    if expected.shape[1] == 1:
        agreement = float(((actual > 0) == (expected > 0)).mean())
    else:
        agreement = float((actual.argmax(1) == expected.argmax(1)).mean())
    print(f'    max difference {np.abs(actual - expected).max():.5f}, decision agreement {agreement:.5%}')


def main():
    import coremltools as ct
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # fp32 reproduces the ONNX masks and measurements exactly. fp16 runs on the
    # Neural Engine and is faster again, but moved enough edge pixels on the
    # test videos to change the number of lines found, so it is opt-in only.
    parser.add_argument('--segmentation-precision', choices=('fp16', 'fp32'), default='fp32')
    parser.add_argument('--fabric-precision', choices=('fp16', 'fp32'), default='fp32')
    parser.add_argument('--check', action='store_true', help='compare each package against its original')
    args = parser.parse_args()
    precision = {'fp16': ct.precision.FLOAT16, 'fp32': ct.precision.FLOAT32}

    jobs = [
        ('onnx', ROOT / LOCAL['source_model'], source_shape(), args.segmentation_precision),
        ('onnx', ROOT / LOCAL['glue_model'], glue_shape(), args.segmentation_precision),
        ('keras', ROOT / 'auto_trigger' / 'models' / 'keras_model_2c.h5', None, args.fabric_precision),
    ]
    for kind, original, shape, chosen in jobs:
        print(f'{original.relative_to(ROOT)} -> Core ML {chosen}', flush=True)
        if kind == 'keras':
            converted, shape = keras_to_coreml(original, precision[chosen])
        else:
            converted = onnx_to_coreml(original, shape, precision[chosen])
        converted.short_description = f'{original.name} converted for Apple silicon ({chosen})'
        converted.user_defined_metadata.update(
            {'source_file': original.name, 'input_shape': json.dumps(list(shape)), 'precision': chosen})
        package = original.with_suffix('.mlpackage')
        if package.exists():
            shutil.rmtree(package)
        converted.save(str(package))
        print(f'    saved {package.relative_to(ROOT)} for input {shape}', flush=True)
        if args.check:
            compare(original, package, shape, kind)


if __name__ == '__main__':
    main()
