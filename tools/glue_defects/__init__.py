"""Glue-width defect research: narrow glue against marked defects, several glue models.

Nothing here is used by the app. The tools read one data folder (default
``~/flod_mas_data/glue_defects``, or ``--data``):

    photos.json          {photo: {"file": "photos/x.png", "label": "...", "good": bool,
                                  "stickers": [[x, y], ...]}}
    photos/*.png         undistorted board photos (the app's undistorted.png)
    app/widths.json      every line: centre path, widths (measure_app.py)
    app/masks/*.png      the glue mask of each line, cropped (origin in widths.json)
    visibility.json      how sharp the glue edges are along each line (visibility.py)
    <source>/widths.json other glue models, same layout (sam_widths.py, labseg_widths.py)
    segments/widths.json main's graded segments (segments_check.py)

``good`` photos are known to be defect-free; a sticker within 250 px of a line marks a
defect on it. See docs/GLUE_DEFECTS.md for what was found.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / 'flod_mas_data' / 'glue_defects'
MM_PER_PX = 4 / 29.6          # uncalibrated: the 4 mm band is about 29.6 px wide
STICKER_REACH_PX = 250        # a sticker this close to a line marks it


def load_photos(data):
    return json.loads((Path(data) / 'photos.json').read_text())


def photo_of(line_name):
    """'13oct_1419_L2' -> '13oct_1419'."""
    return line_name.rsplit('_L', 1)[0]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def path_normals(xy, half_window=15):
    """Unit normals of a centre path, from its tangent averaged over 2*half_window+1 points."""
    xy = np.asarray(xy, float)
    k = half_window
    box = np.ones(2 * k + 1) / (2 * k + 1)
    smooth = lambda v: np.convolve(np.pad(v, k, mode='edge'), box, 'same')[k:-k]
    t = np.c_[np.gradient(smooth(xy[:, 0])), np.gradient(smooth(xy[:, 1]))]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.c_[-t[:, 1], t[:, 0]]
