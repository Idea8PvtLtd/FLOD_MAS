"""The narrow-glue check and the edge-visibility gate, as plain functions.

A line is judged only where the photo shows both glue edges sharply enough: where the
light does not reach, the edges are not seen and nothing there may be called a defect.
Where it is judged, a stretch fails when its width stays below a share of the band's
own width nearby (a running median), so slow changes across the board, such as the
uncalibrated camera scale, are not taken for defects.
"""
import cv2
import numpy as np

from . import path_normals

TIP_MM = 5.0           # the rounded tips are not judged
VIS_MIN = 3.0          # edge sharpness (lightness levels per pixel) needed to judge
LIMIT = 0.90           # fail below this share of the width nearby
MIN_MM = 5.0           # ... over at least this length
LOCAL_MM = 60.0        # the width nearby: running median over this length (0: whole line)


def edge_visibility(lightness, xy, arc_mm, smooth_mm=3.0):
    """How sharp the weaker of the band's two edges is at each centre sample.

    From the photo alone (no model): the lightness profile across the line, the
    steepest change 8-24 px to each side, the smaller of the two, as a median over
    ``smooth_mm`` along the line. Lightness levels (0-255) per pixel.
    """
    xy = np.asarray(xy, float)
    n = path_normals(xy)
    s = np.arange(-30, 30.01, .5)
    profile = cv2.remap(lightness.astype(np.float32),
                        (xy[:, 0, None] + s[None] * n[:, 0, None]).astype(np.float32),
                        (xy[:, 1, None] + s[None] * n[:, 1, None]).astype(np.float32), cv2.INTER_LINEAR)
    profile = cv2.GaussianBlur(profile, (0, 0), sigmaX=2, sigmaY=2)
    slope = np.abs(np.gradient(profile, axis=1)) / .5
    left = slope[:, (s >= -24) & (s <= -8)].max(1)
    right = slope[:, (s >= 8) & (s <= 24)].max(1)
    vis = np.minimum(left, right)
    step = max(float(np.median(np.diff(arc_mm))), 1e-3)
    m = max(1, int(round(smooth_mm / step)))
    return np.array([np.median(vis[max(0, i - m // 2): i + m // 2 + 1]) for i in range(len(vis))])


def runs_of(flags, arc):
    """(start mm, end mm, first index, last index) of each run of True."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((arc[start], arc[i - 1], start, i - 1))
            start = None
    return out


def judge(arc, width, vis, limit=LIMIT, min_mm=MIN_MM, local_mm=LOCAL_MM, vis_min=VIS_MIN,
          tip_mm=TIP_MM, cut_by_edge=False):
    """Narrow stretches of one line.

    Returns (reference, ratio, judged, runs): the width each sample is compared
    with, the width averaged over ``min_mm`` as a share of it, which samples were
    judged, and [(start mm, end mm, lowest ratio)] of the failing stretches.
    A width of 0 means not measured.
    """
    arc, width, vis = (np.asarray(a, float) for a in (arc, width, vis))
    length = arc[-1]
    seen = (arc >= tip_mm) & (arc <= length - tip_mm) & (vis >= vis_min) & (width > 0)
    middle = (arc > 20) & (arc < length - 20) & (width > 0)
    whole = float(np.median(width[middle])) if middle.any() else float(np.median(width[width > 0]))
    if local_mm > 0:
        reference = np.array([np.median(width[seen & (np.abs(arc - a) <= local_mm / 2)])
                              if (seen & (np.abs(arc - a) <= local_mm / 2)).sum() > 5 else whole for a in arc])
    else:
        reference = np.full(len(arc), whole)
    step = max(float(np.median(np.diff(arc))), 1e-3)
    k = max(1, int(round(min_mm / step)))
    total = np.convolve(np.where(seen, width / reference, 0.), np.ones(k), 'same')
    count = np.convolve(seen.astype(float), np.ones(k), 'same')
    ratio = total / np.maximum(count, 1)
    judged = seen & (not cut_by_edge)
    runs = [(a, b, float(ratio[i:j + 1].min())) for a, b, i, j in runs_of(judged & (ratio < limit), arc)]
    return reference, ratio, judged, runs


def segment_check(widths, vis, limit=LIMIT, vis_min=VIS_MIN, neighbours=3):
    """Graded segments (main's method) checked against their neighbours.

    ``widths`` and ``vis``: per segment, in order (None: no width). A seen segment
    fails when it is below ``limit`` of the median of the seen segments within
    ``neighbours`` on either side. Returns the failing segment indices.
    """
    seen = [w is not None and v is not None and v >= vis_min for w, v in zip(widths, vis)]
    failing = []
    for i, w in enumerate(widths):
        if not seen[i]:
            continue
        near = [widths[j] for j in range(max(0, i - neighbours), min(len(widths), i + neighbours + 1))
                if j != i and seen[j]]
        if len(near) >= 2 and w < limit * float(np.median(near)):
            failing.append(i)
    return failing
