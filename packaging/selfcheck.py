"""Start-up check shipped inside every bundle as flod-mas-selfcheck.

It needs no camera and opens no window. It confirms that the bundle has its version
record, that every model file matches its checksum, that the heavy libraries load,
that each ONNX model opens in onnxruntime, and that the application's own modules
import. Exit code 0 means the bundle is complete. CI runs it on every build; a service
technician can run it on the inspection PC.
"""
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

BASE = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
FROZEN = hasattr(sys, '_MEIPASS')

LIBRARIES = ('numpy', 'cv2', 'PIL.ImageTk', 'scipy.ndimage', 'skimage', 'onnxruntime',
             'mediapipe', 'tensorflow', 'tf_keras', 'tkinter')
APP_MODULES = ('app_config', 'camera_handler', 'inspection.dashboard',
               'inspection.local.source_model', 'auto_trigger.controller',
               'CalibrateAPP.calibration_ui')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def models():
    listed = BASE / ('model-checksums.sha256' if FROZEN else 'packaging/model-checksums.sha256')
    for line in listed.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            digest, name = line.split(None, 1)
            yield name.strip(), digest


def main():
    failures = []

    def step(label, action):
        try:
            detail = action()
            print(f'ok    {label}' + (f'  ({detail})' if detail else ''))
        except Exception as exc:
            failures.append(label)
            print(f'FAIL  {label}: {exc}')
            if '--verbose' in sys.argv:
                traceback.print_exc()

    info = BASE / 'BUILD_INFO.json'
    if FROZEN:
        step('build info', lambda: ' '.join(
            f'{key}={value}' for key, value in json.loads(info.read_text()).items()))
    for name, digest in models():
        def check(name=name, digest=digest):
            if sha256(BASE / name) != digest:
                raise RuntimeError('checksum mismatch')
        step(f'model {name}', check)
    for module in LIBRARIES:
        step(f'import {module}', lambda module=module: getattr(
            importlib.import_module(module), '__version__', ''))
    import onnxruntime
    for name, _ in models():
        if name.endswith('.onnx'):
            step(f'open {name}', lambda name=name: len(onnxruntime.InferenceSession(
                str(BASE / name), providers=['CPUExecutionProvider']).get_inputs()))
    for module in APP_MODULES:
        step(f'import {module}', lambda module=module: importlib.import_module(module) and '')

    if failures:
        print(f'{len(failures)} check(s) failed.')
        return 1
    print('Bundle is complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
