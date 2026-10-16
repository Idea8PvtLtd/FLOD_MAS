"""One common form for a measured line, whichever method read it, and the run's files.

A line is a dict with lists of [x, y] in pixels of the undistorted photo:

    left, right      the two edges, sample by sample (same length, paired)
    centre           the middle between them
    outline          the glue's outline as drawn (a closed polygon)
    widths_px        the width at each pair, and width_px its mean, median_px its median
    segments_px      mean width of ten equal parts along the line
    width_mm         the mean width in millimetres, when the camera is calibrated

App lines come from the engine's samples (left_x ... right_y); version 5 lines from
its ring, whose first half is one edge and second half the other, reversed.
"""
import json
from pathlib import Path

import numpy as np

SEGMENTS = 10


def _points(array):
    return [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(array, float).reshape(-1, 2)]


def _segments(widths, count=SEGMENTS):
    if len(widths) < count:
        return [float(np.mean(widths))] if len(widths) else []
    return [round(float(part.mean()), 3) for part in np.array_split(np.asarray(widths, float), count)]


def from_edges(left, right, outline=None, number=None, scale=None, extra=None):
    """A line from two paired edges (N x 2 each)."""
    left, right = np.asarray(left, float).reshape(-1, 2), np.asarray(right, float).reshape(-1, 2)
    if len(left) != len(right) or not len(left):
        raise ValueError('The two edges must be paired and not empty')
    widths = np.hypot(*(left - right).T)
    centre = (left + right) / 2
    line = {'number': number, 'left': _points(left), 'right': _points(right), 'centre': _points(centre),
            'outline': _points(outline) if outline is not None else _points(np.vstack((left, right[::-1]))),
            'widths_px': [round(float(w), 3) for w in widths], 'width_px': round(float(widths.mean()), 3),
            'median_px': round(float(np.median(widths)), 3), 'segments_px': _segments(widths),
            'centroid': [round(float(v), 1) for v in centre.mean(0)], 'width_mm': None}
    if scale is not None:
        try:
            mm = np.hypot(*(scale.to_mm(left) - scale.to_mm(right)).T)
            line['width_mm'] = round(float(mm.mean()), 4)
        except Exception:
            pass
    line.update(extra or {})
    return line


def from_app(result, scale=None):
    """A line from one record of LiveInspection.measure_found."""
    import cv2
    main = max(result['components'], key=lambda c: c['length_px'])
    samples = [s for s in main['samples'] if s['valid']]
    left = [(s['left_x'], s['left_y']) for s in samples]
    right = [(s['right_x'], s['right_y']) for s in samples]
    outline = None
    mask = result.get('mask')
    if mask is not None:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            outline = cv2.approxPolyDP(biggest, 1.0, True)[:, 0, :] + np.asarray(result.get('origin', (0, 0)))
    ribbon = result.get('ribbon') or {}
    return from_edges(left, right, outline, result['selected_line'], scale,
                      {'pieces': len(result['components']), 'ribbon': ribbon.get('state')})


def from_ring(ring, number=None, scale=None):
    """A line from a version 5 ring: one edge, then the other edge backwards."""
    ring = np.asarray(ring, float).reshape(-1, 2)
    half = len(ring) // 2
    return from_edges(ring[:half], ring[half:2 * half][::-1], ring, number, scale)


def match(lines, others, limit=80.0):
    """Pairs (i, j) of lines that are the same glue line: nearest centroids within `limit` px."""
    pairs, used = [], set()
    for i, line in enumerate(lines):
        best, best_distance = None, limit
        for j, other in enumerate(others):
            if j in used:
                continue
            distance = float(np.hypot(*(np.subtract(line['centroid'], other['centroid']))))
            if distance < best_distance:
                best, best_distance = j, distance
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def tracks(photos, limit=120.0):
    """The same physical line followed across the photos of one board.

    ``photos`` = [lines of photo 1, lines of photo 2, ...] in order; returns a list of
    tracks, each a list of (photo index, line index). A board may shift a little
    between photos, so a line is matched to the track's latest position.
    """
    found = []
    for p, lines in enumerate(photos):
        last = [photos[t[-1][0]][t[-1][1]] for t in found]
        pairs = match(lines, last, limit)
        joined = {i for i, _ in pairs}
        for i, t in pairs:
            found[t].append((p, i))
        found.extend([(p, i)] for i in range(len(lines)) if i not in joined)
    return found


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())
