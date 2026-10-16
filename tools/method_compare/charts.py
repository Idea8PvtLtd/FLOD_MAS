"""Numbers for the viewer's charts: axis scales, bar positions, summaries. No Tk here."""
import math

import numpy as np

from .results import match, tracks


def nice_max(value, steps=(1, 2, 2.5, 5, 10)):
    """The smallest 'round' number at or above ``value`` (for an axis end)."""
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for step in steps:
        if step * power >= value - 1e-12:
            return float(step * power)
    return float(10 * power)


def ticks(top, count=5):
    """Evenly spaced round ticks from 0 to ``top``."""
    step = nice_max(top / count)
    return [round(i * step, 6) for i in range(int(top / step + 1e-9) + 1)]


def scale(value, low, high, start, end):
    """``value`` in [low, high] mapped onto [start, end] (pixels), clamped."""
    if high == low:
        return start
    share = min(1.0, max(0.0, (value - low) / (high - low)))
    return start + share * (end - start)


def time_colour(seconds, target):
    """'good' under the target, 'bad' over it, 'none' when unknown."""
    if seconds is None:
        return 'none'
    return 'good' if seconds <= target else 'bad'


def photo_parts(method_data, row):
    """The two parts of a photo's time for the speed bars: (first, second, label, label).

    App methods: finding the lines, then the new lines (both measured).
    Version 5: the fixed stages, then the new lines' estimated share.
    """
    if method_data['family'] == 'app':
        return row['find_s'], row['new_lines_s'], 'find the lines', 'new lines'
    return row['fixed_s'], row['photo_s'] - row['fixed_s'], 'fixed stages', 'new lines (estimated)'


def mean_parts(method_data):
    rows = method_data['photos']
    if not rows:
        return 0.0, 0.0
    parts = [photo_parts(method_data, row)[:2] for row in rows]
    return float(np.mean([p[0] for p in parts])), float(np.mean([p[1] for p in parts]))


def widths_against(reference_lines, lines):
    """Width differences (method minus reference) for the lines both read, in px."""
    return [lines[j]['width_px'] - reference_lines[i]['width_px'] for i, j in match(reference_lines, lines)]


def repeatability(method_data, moments):
    """Per board line: the spread (SD, px) of one method's width across the photos it appears in.

    Returns the list of SDs of lines seen at least twice.
    """
    by_board = {}
    for moment, row in zip(moments, method_data['photos']):
        by_board.setdefault(moment['board'], []).append(row['lines'])
    spreads = []
    for photos in by_board.values():
        for track in tracks(photos):
            if len(track) > 1:
                spreads.append(float(np.std([photos[p][i]['width_px'] for p, i in track])))
    return spreads


def summary(method_data, moments, reference=None):
    """One row of the speed and accuracy tables for a method."""
    rows = method_data['photos']
    times = [row['photo_s'] for row in rows]
    spreads = repeatability(method_data, moments)
    out = {'photo_mean_s': float(np.mean(times)) if times else None,
           'photo_max_s': float(np.max(times)) if times else None,
           'lines': sum(len(row['lines']) for row in rows),
           'spread_median_px': float(np.median(spreads)) if spreads else None,
           'spread_max_px': float(np.max(spreads)) if spreads else None,
           'difference_mean_px': None, 'difference_low_px': None, 'difference_high_px': None}
    if reference is not None and reference is not method_data:
        differences = [d for ref, mine in zip(reference['photos'], rows)
                       for d in widths_against(ref['lines'], mine['lines'])]
        if differences:
            out.update(difference_mean_px=float(np.mean(differences)),
                       difference_low_px=float(np.min(differences)),
                       difference_high_px=float(np.max(differences)))
    return out
