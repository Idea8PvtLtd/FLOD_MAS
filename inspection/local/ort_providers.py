"""ONNX Runtime execution providers for the local models.

``local_inspection.execution_provider`` in config.json picks where both models
run. ``cpu`` (the default) is what every PC has always used. ``cuda`` and
``tensorrt`` need the GPU build of ONNX Runtime (onnxruntime-gpu, e.g. on the
Jetson); when that build or the GPU is missing, ONNX Runtime falls back to the
CPU, which is always last in the list.

TensorRT runs in full float32 precision: half precision changes the masks and so
the millimetre readings. Built engines are cached in the data folder, so only the
first start after a model or input size change spends minutes building them.
"""

KINDS = ('cpu', 'cuda', 'tensorrt')


def provider_list(kind='cpu', cache_dir=None):
    """Providers for ``ort.InferenceSession``, most preferred first."""
    if kind not in KINDS:
        raise ValueError(f'Execution provider must be one of {", ".join(KINDS)}')
    providers = []
    if kind == 'tensorrt':
        options = {'trt_fp16_enable': False}
        if cache_dir is not None:
            options.update(trt_engine_cache_enable=True, trt_engine_cache_path=str(cache_dir),
                           trt_timing_cache_enable=True, trt_timing_cache_path=str(cache_dir))
        providers.append(('TensorrtExecutionProvider', options))
    if kind in ('cuda', 'tensorrt'):
        # Grow the GPU memory pool only by what is asked for, not in powers of
        # two: on the Jetson the GPU shares the 8 GB with everything else, and
        # this saved about 360 MB at the same speed.
        providers.append(('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}))
    providers.append('CPUExecutionProvider')
    return providers


def default_cache_dir():
    """Folder for TensorRT engines, beside the other per-PC files."""
    from pathlib import Path
    from app_storage import data_path
    path = Path(data_path('Files/trt_cache'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_session(path, kind='cpu'):
    """An InferenceSession on the requested provider, falling back to the CPU."""
    from pathlib import Path
    import onnxruntime as ort
    available = ort.get_available_providers()
    cache_dir = default_cache_dir() if kind == 'tensorrt' else None
    requested = provider_list(kind, cache_dir)
    providers = [p for p in requested if (p[0] if isinstance(p, tuple) else p) in available]
    if kind != 'cpu':
        try:
            ort.preload_dlls()
        except Exception:
            pass
    session = ort.InferenceSession(str(path), providers=providers)
    wanted = requested[0][0] if isinstance(requested[0], tuple) else requested[0]
    if active_provider(session) != wanted:
        # ONNX Runtime falls back silently (e.g. GPU out of memory); say so in the log.
        print(f'[models] {Path(path).name}: asked for {wanted}, running on {active_provider(session)}',
              flush=True)
    return session


def active_provider(session):
    """Name of the provider that runs the model, e.g. 'TensorrtExecutionProvider'."""
    return session.get_providers()[0]
