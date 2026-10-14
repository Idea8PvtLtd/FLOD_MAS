"""Traditional machine learning for the glue band's edges: gradient-boosted trees
that correct an edge found by slope, from the picture's profile around it.

The steepest-slope edge is noisy on pale fabric and is pulled by the shiny core
and by blur. Around each such edge the profile across the line is read in three
channels (lightness as averaged, lightness with the core removed, weave energy),
turned so that "outward" is the same for both edges and scaled by the band's own
depth; trees learn from painted glue lines, whose edges are known
(``synthetic_bead``), how far and which way the true edge lies.

The model is small and runs on the CPU. It was trained by ``tools/train_bead_ml.py``
on the ``tooling/pipeline-lab`` branch; ``MODEL_FILE`` is the copy shipped beside the
engine's other models.
"""
from pathlib import Path
import threading

import numpy as np

from . import edge_methods

MODEL_FILE = Path(__file__).resolve().parents[1] / 'models' / 'bead_edge_trees.txt'
REACH = 14               # pixels of profile read to each side of the first edge
LIMIT = 12.0             # the correction is never larger than this

_model = None
# Lines may be measured in parallel (local_inspection.line_workers); loading and
# predicting go through one lock so the shared Booster is never used by two at once.
_lock = threading.Lock()


def features(strip, offsets, read):
    """One row per sample and edge: ``(low rows, high rows)``."""
    smooth = read['smooth'].astype(np.float64)
    plain = read['plain'].astype(np.float64)
    weave = edge_methods.weave_energy(strip).astype(np.float64)
    count, width = smooth.shape
    columns = np.arange(width, dtype=np.float64)
    steps = np.arange(-REACH, REACH + 1, dtype=np.float64)
    depth = np.maximum(read['darkness'], 0.5)
    band = read['high_course'] - read['low_course']
    rows = {}
    for side, edge, outward in (('low', read['low_course'], -1.0), ('high', read['high_course'], 1.0)):
        table = np.zeros((count, 3 * len(steps) + 5))
        for index in range(count):
            at = (edge[index] - offsets[0]) + outward * steps          # inward ... outward
            parts = []
            for channel, scale in ((smooth, depth[index]), (plain, depth[index]), (weave, 1.0)):
                values = np.interp(at, columns, channel[index])
                parts.append((values - values.mean()) / scale)
            core_gap = outward * (edge[index] - read['core_at'][index])   # core's distance inside this edge
            table[index] = np.r_[np.concatenate(parts), depth[index], read['core'][index] / depth[index],
                                 core_gap, band[index], read['core'][index]]
        rows[side] = table
    return rows['low'], rows['high']


def load(path=None):
    global _model
    path = path or MODEL_FILE
    with _lock:
        return _load(path)


def _load(path):
    global _model
    if _model is None:
        import lightgbm
        if not Path(path).is_file():
            raise FileNotFoundError(f'No trained edge model at {path}')
        _model = lightgbm.Booster(model_file=str(path))
    return _model


def corrected(strip, offsets, read, model=None):
    """The edges after the trees' correction: ``(low, high)``."""
    model = model or load()
    low_rows, high_rows = features(strip, offsets, read)
    with _lock:
        low_shift, high_shift = model.predict(low_rows), model.predict(high_rows)
    low = read['low_course'] - np.clip(low_shift, -LIMIT, LIMIT)     # outward is -x here
    high = read['high_course'] + np.clip(high_shift, -LIMIT, LIMIT)
    return low, high


def trees(strip, offsets, read):
    """Edge method for ``edge_methods.METHODS``."""
    return corrected(strip, offsets, read)


edge_methods.METHODS['trees'] = trees
