"""Version 5 from tooling/pipeline-lab (methods C, D, E) on every still moment of a run.

    python -m tools.method_compare.methods_v5 --run DIR --method C|D|E

Runs the lab checkout (~/flod_lab) unchanged: SAM 2.1 in half precision on CUDA (C,
D) or the rough line only (E), then the trained-tree edges and the smooth ribbon.
Its glue-region ONNX model is run on CUDA here: the lab only has a GPU path for a
PyTorch checkpoint that is not in its repository.

Version 5 reads every line of a photo. The app reads only the lines new on the
board, so "photo_s" here is an estimate for a version 5 that did the same: the
photo's fixed stages plus the per-line stages' share for each new line. The
measured time for all lines is kept as "all_lines_s".
"""
import argparse
import collections
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import BY_ID, LAB
from .results import from_ring, load, match, save

FIXED = ('roi', 'cardboard')                                   # once per photo
PER_LINE = ('rough', 'sam', 'repair', 'states', 'gather', 'ribbon', 'stats')


def use_lab():
    """The lab's own packages first: it has its own tools/ and segmentation/."""
    sys.path[:0] = [str(LAB), str(LAB / 'tools')]
    os.chdir(LAB)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import onnxruntime
    plain = onnxruntime.InferenceSession

    def cuda_session(path, *args, providers=None, **kwargs):
        return plain(path, *args, providers=[('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}),
                                             'CPUExecutionProvider'], **kwargs)
    onnxruntime.InferenceSession = cuda_session
    import pipeline_steps as steps
    model_roi = steps.model_roi
    steps.model_roi = lambda image, *a, on_gpu=False, **k: model_roi(image, *a, on_gpu=False, **k)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--method', required=True, choices=[m['id'] for m in BY_ID.values() if m['family'] == 'v5'])
    args = parser.parse_args(argv)
    run, method = Path(args.run).resolve(), BY_ID[args.method]
    moments = load(run / 'moments.json')['moments']
    use_lab()
    import cv2
    import benchmark_flow as flow
    import pipeline_lab as lab
    sam = method['mode'] != 'nosam'
    recipe = {'roi': 'model', 'pipeline': [], 'point_source': 'lines', 'segmenter': 'sam3' if sam else 'local',
              'ribbon': True, 'local_model': method['mode'] if sam else lab.DEFAULT_LOCAL_MODEL,
              'sam_runs': 'local', 'ask_once': True, 'edge_method': 'trees'}
    spent, counts = collections.defaultdict(float), collections.defaultdict(int)
    flow.instrument(spent, counts)
    try:
        from plane_scale import load_frame_scale
    except Exception:
        load_frame_scale = None
    photos = [cv2.imread(str(run / 'photos' / m['photo'])) for m in moments]
    scale = None
    if photos and load_frame_scale is not None:
        try:
            scale = load_frame_scale(1, (photos[0].shape[1], photos[0].shape[0]))
        except Exception as error:
            print(f'No millimetre scale: {error}', flush=True)
    first_s = None
    if photos:
        started = time.perf_counter()
        lab.run_job(photos[0], recipe, None, [])
        first_s = time.perf_counter() - started

    rows, last_on_board = [], {}
    for moment, photo in zip(moments, photos):
        spent.clear()
        counts.clear()
        started = time.perf_counter()
        result = lab.run_job(photo, recipe, None, [])
        total = time.perf_counter() - started
        lines = []
        for number, found in enumerate(sorted(result['lines'], key=lambda l: float(np.mean(np.asarray(l['ring'])[:, 0]))), 1):
            try:
                lines.append(from_ring(found['ring'], number, scale))
            except ValueError:
                continue
        before = last_on_board.get(moment['board'], [])
        seen = {i for i, _ in match(lines, before, 120.0)}
        for i, line in enumerate(lines):
            line['new'] = i not in seen
        last_on_board[moment['board']] = lines
        new = sum(line['new'] for line in lines)
        stages = {key: round(spent[key], 3) for key, *_ in flow.STAGES}
        fixed = sum(stages[k] for k in FIXED) + max(0.0, total - sum(stages.values()))
        per_line = sum(stages[k] for k in PER_LINE) / max(1, len(lines))
        rows.append({'photo': moment['photo'], 'lines': lines, 'found': len(lines), 'new': new,
                     'stages': stages, 'sam_calls': counts['sam_calls'], 'all_lines_s': round(total, 3),
                     'fixed_s': round(fixed, 3), 'per_line_s': round(per_line, 3),
                     'photo_s': round(fixed + per_line * new, 3), 'errors': []})
        print(f"{moment['photo']}: {len(lines)} lines ({new} new), all lines {total:.2f} s, "
              f"estimated new-line photo {fixed + per_line * new:.2f} s", flush=True)
    save(run / f"{method['id']}.json", {
        'method': method['id'], 'label': method['label'], 'family': 'v5', 'mode': method['mode'],
        'first_photo_s': round(first_s, 2) if first_s is not None else None,
        'gpu_mb': round(flow.gpu_megabytes()),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        'time_note': 'photo = estimated: fixed stages + per-line share x lines new on this board; '
                     'all_lines_s = measured, every line',
        'photos': rows})
    return 0


if __name__ == '__main__':
    sys.exit(main())
