"""Statistics of a glue line's width: millimetres from pixels, and where the width
is larger or smaller than a good line's natural variation allows.

Three facts decide the method (measured on defect-free lines, see
``docs/glue-line/GLUE_LINE_FINDINGS.md``, *Width statistics*):

- The camera is not calibrated: a 4 mm strip reads 26-35 pixels depending on where
  it lies in the frame. So the scale is fitted, as a smooth surface over the frame
  with a factor per fabric colour, on the assumption that good lines average the
  nominal width.
- The width along a good line is not independent noise: neighbouring readings are
  correlated over hundreds of pixels, and the spread has heavier tails than a
  normal distribution. Limits taken from a normal table (3 sigma) would flag far
  too much.
- So the limits are empirical. The width's departure from the line's own gradual
  course is averaged over windows of several lengths; for each length, the limit is a
  percentile of the largest such average that good lines show. A line then has a
  known chance of being flagged by chance, whatever the distribution.

Everything here is plain arithmetic on arrays; nothing reads a picture.
"""
import numpy as np

NOMINAL_MM = 4.0
WINDOWS_PX = (40, 80, 160, 320)        # lengths over which the width is averaged
LEVELS = {'notable': 80.0, 'significant': 95.0}   # percentiles of good lines' largest average


def block_means(values, size):
    """Means of consecutive blocks of ``size`` samples (the remainder is dropped)."""
    values = np.asarray(values, np.float64)
    count = len(values) // size
    return values[:count * size].reshape(count, size).mean(axis=1)


def _design(xy, frame_size):
    """Quadratic surface over the frame, coordinates scaled to about -1..1."""
    half_width, half_height = frame_size[0] / 2.0, frame_size[1] / 2.0
    x = (np.asarray(xy)[:, 0] - half_width) / half_width
    y = (np.asarray(xy)[:, 1] - half_height) / half_height
    return np.column_stack([x, y, x * x, y * y, x * y])


def fit_scale(xy, widths_px, groups, frame_size, nominal_mm=NOMINAL_MM):
    """Pixels per millimetre over the frame, from widths taken to average ``nominal_mm``.

    ``xy`` are the places (pixels), ``widths_px`` the widths read there and
    ``groups`` a label per reading (the fabric colour) that gets its own factor.
    The logarithm of the width is fitted by a quadratic surface plus a constant per
    group, with readings far from the fit given less weight (Huber), so that odd
    places do not bend the scale. Returns a dict for ``pixels_per_mm``.
    """
    widths_px = np.asarray(widths_px, np.float64)
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    design = np.column_stack([np.column_stack([(groups == name).astype(float) for name in names]),
                              _design(xy, frame_size)])
    target = np.log(widths_px)
    weight = np.ones(len(target))
    for _ in range(8):
        root = np.sqrt(weight)[:, None]
        solution, *_ = np.linalg.lstsq(design * root, target * root[:, 0], rcond=None)
        residual = target - design @ solution
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        weight = np.minimum(1.0, 1.345 * spread / np.maximum(np.abs(residual), 1e-12))
    explained = 1.0 - residual.var() / max(target.var(), 1e-12)
    return {'groups': names, 'constants': solution[:len(names)].tolist(),
            'surface': solution[len(names):].tolist(), 'frame_size': list(frame_size),
            'nominal_mm': float(nominal_mm), 'explained': float(explained)}


def pixels_per_mm(scale, xy, group):
    """The fitted scale at places ``xy`` for one group."""
    constant = scale['constants'][scale['groups'].index(group)]
    log_width = constant + _design(xy, scale['frame_size']) @ np.asarray(scale['surface'])
    return np.exp(log_width) / scale['nominal_mm']


def slow_trend(values, degree=2):
    """A line's gradual change of width from end to end: a low-order curve fitted
    with little weight on readings far from it, so that a local change does not
    bend it. A defect is a local change; a gradual one along the whole line is
    reported separately and is mostly the uncalibrated camera."""
    values = np.asarray(values, np.float64)
    along = np.linspace(-1.0, 1.0, len(values))
    curve = np.polyfit(along, values, degree)
    for _ in range(3):
        residual = values - np.polyval(curve, along)
        spread = 1.4826 * np.median(np.abs(residual)) + 1e-9
        weight = np.minimum(1.0, 1.345 * spread / np.maximum(np.abs(residual), 1e-12))
        curve = np.polyfit(along, values, degree, w=np.sqrt(weight))
    return np.polyval(curve, along)


def window_means(values, size):
    """Mean of ``values`` in every window of ``size`` samples (same length; the
    ends, where the window does not fit, are NaN)."""
    values = np.asarray(values, np.float64)
    if size > len(values):
        return np.full(len(values), np.nan)
    total = np.cumsum(np.r_[0.0, values])
    means = (total[size:] - total[:-size]) / size
    result = np.full(len(values), np.nan)
    start = (size - 1) // 2
    result[start:start + len(means)] = means
    return result


def largest_departures(departure, sizes):
    """For each window length (samples): the largest and the smallest window mean
    of one line's departure from its median."""
    found = {}
    for size in sizes:
        means = window_means(departure, size)
        found[size] = (np.nanmax(means), np.nanmin(means)) if np.isfinite(means).any() else (np.nan, np.nan)
    return found


def limits_from_good_lines(departures, sizes, levels=LEVELS):
    """Empirical limits from good lines, holding for all window lengths together.

    ``departures`` is a list of per-line arrays (width minus the line's gradual
    course). For each window length the largest absolute window mean of every
    line is taken and put on a common scale (divided by that length's median over
    the lines). A line's score is its largest scaled value over all lengths; the
    limit is a percentile of that score. So with the 95th percentile, 5 good lines
    in 100 are flagged by chance, at any length, not 5 per length.
    Returns ``{level: {size: limit}}`` in the units of ``departures``.
    """
    extremes = []
    for departure in departures:
        found = largest_departures(departure, sizes)
        extremes.append([max(abs(found[size][0]), abs(found[size][1])) for size in sizes])
    extremes = np.array(extremes, np.float64)
    if extremes.size == 0:
        return {level: {size: np.nan for size in sizes} for level in levels}
    usual = np.nanmedian(extremes, axis=0)                      # per window length
    score = np.nanmax(extremes / np.maximum(usual, 1e-12), axis=1)
    score = score[np.isfinite(score)]
    return {level: {size: float(usual[number] * np.percentile(score, percentile))
                    for number, size in enumerate(sizes)}
            for level, percentile in levels.items()}


def scan(departure, limits, sizes):
    """Stretches of a line that are wider or narrower than good lines allow.

    Each window length is tested against its own limit; overlapping findings of
    the same sign are merged. Returns a list of dicts with ``first`` / ``last``
    (samples), ``change`` (the largest window mean in the stretch, signed),
    ``window`` (the length that showed it most clearly, samples) and ``level``.
    """
    count = len(departure)
    best = np.zeros(count)             # how far over the limit, signed by direction
    change = np.zeros(count)
    window = np.zeros(count, int)
    rank = np.zeros(count, int)
    order = list(limits)               # weaker level first
    for size in sizes:
        means = window_means(departure, size)
        for number, level in enumerate(order, 1):
            limit = limits[level].get(size, np.nan)
            if not np.isfinite(limit) or limit <= 0:
                continue
            centres = np.flatnonzero(np.abs(np.nan_to_num(means)) > limit)
            for centre in centres:
                first = max(0, centre - (size - 1) // 2)
                last = min(count, first + size)
                over = abs(means[centre]) / limit
                stronger = (number > rank[first:last]) | ((number == rank[first:last]) & (over > np.abs(best[first:last])))
                span = np.arange(first, last)[stronger]
                rank[span] = number
                best[span] = over * np.sign(means[centre])
                change[span] = means[centre]
                window[span] = size
    findings = []
    start = None
    for index in range(count + 1):
        sign = np.sign(best[index]) if index < count else 0
        if start is not None and (index == count or sign != np.sign(best[start])):
            stretch = slice(start, index)
            strongest = start + int(np.abs(best[stretch]).argmax())
            findings.append({'first': start, 'last': index - 1, 'change': float(change[strongest]),
                             'window': int(window[strongest]), 'level': order[int(rank[stretch].max()) - 1]})
            start = None
        if start is None and index < count and sign != 0:
            start = index
    return findings


def stretches(flags, shortest=1):
    """``(first, last)`` of every run of True at least ``shortest`` long."""
    flags = np.asarray(flags, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return [(int(first), int(last) - 1) for first, last in zip(edges[::2], edges[1::2])
            if last - first >= shortest]


def judge_change(finding, doubtful, other_departure, agree_share=0.5, overlap=0.3):
    """Is a width change found in the mask a change of the glue, or of the mask?

    ``doubtful`` flags the samples where the mask itself is in question (missing,
    fading out, or shifted sideways); ``other_departure`` is the same line's
    departure read another way, independent of the mask (the band's edges).

    - ``glitch``     the mask is in question over a good part of the stretch;
    - ``confirmed``  the other reading shows a change of the same sign, at least
                     ``agree_share`` of its size;
    - ``mask only``  neither: the mask changed, nothing else says the glue did.
    """
    part = slice(finding['first'], finding['last'] + 1)
    if np.mean(np.asarray(doubtful)[part]) >= overlap:
        return 'glitch'
    other = float(np.mean(np.asarray(other_departure)[part]))
    if np.sign(other) == np.sign(finding['change']) and abs(other) >= agree_share * abs(finding['change']):
        return 'confirmed'
    return 'mask only'

