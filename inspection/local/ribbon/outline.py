"""A segmentation mask read as edge evidence along a line, and the region glue can lie in.

Taken from ``segmentation/sam_refine.py`` and ``segmentation/glue_line.py`` on the
``tooling/pipeline-lab`` branch, where the mask came from SAM. Here it is the
adhesive model's mask; the reading is the same.
"""
import cv2
import numpy as np

TRIM_REACH = 70          # how far to each side of the line the mask's outline is followed
TRIM_STEP = 4.0          # spacing of the samples along the line
TRIM_TOLERANCE = 5.0     # an edge further than this from its smooth course has leaked
TRIM_NEAR = 20           # the mask must pass this close to the line to count
CARDBOARD_MIN_VALUE = 100  # cardboard and tape are at least this bright (HSV value)


def cardboard_mask(image, max_saturation=60.0, grow_px=9):
    """Cardboard, tape and skin: warm, saturated surfaces that are never fabric."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Bright as well as warm: sheer dark mesh lying over the board, or over a
    # panel, takes on its colour but stays dark, and that is fabric.
    warm = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > max_saturation)
            & (hsv[:, :, 2] > CARDBOARD_MIN_VALUE)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # not mesh speckle
    return cv2.dilate(warm, np.ones((grow_px, grow_px), np.uint8))


def centre_of(ring):
    """The middle line of a ring whose two halves are the two edges."""
    half = len(ring) // 2
    return (ring[:half] + ring[half:][::-1]) / 2.0


def _edges(mask, centre):
    """How far the mask reaches to each side of the line, sample by sample.

    Returns ``(points, normals, low, high)``; ``low``/``high`` are signed distances
    along the normal, NaN where the mask does not pass near the line.
    """
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    along = np.arange(0.0, arc[-1], TRIM_STEP)
    points = np.column_stack([np.interp(along, arc, centre[:, 0]), np.interp(along, arc, centre[:, 1])])
    reach = 7
    window = np.ones(2 * reach + 1) / (2 * reach + 1)
    points = np.column_stack([np.convolve(np.pad(points[:, axis], reach, mode='edge'), window, 'valid')
                              for axis in (0, 1)])
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(-TRIM_REACH, TRIM_REACH + 1)
    columns = np.clip(np.round(points[:, None, 0] + offsets[None] * normals[:, None, 0]).astype(int),
                      0, mask.shape[1] - 1)
    rows = np.clip(np.round(points[:, None, 1] + offsets[None] * normals[:, None, 1]).astype(int),
                   0, mask.shape[0] - 1)
    low = np.full(len(points), np.nan)
    high = np.full(len(points), np.nan)
    for index, row in enumerate(mask[rows, columns] > 0):
        inside = np.flatnonzero(row)
        if inside.size == 0:
            continue
        nearest = inside[np.abs(offsets[inside]).argmin()]
        if abs(offsets[nearest]) > TRIM_NEAR:
            continue
        first = last = nearest
        while first > 0 and row[first - 1]:
            first -= 1
        while last < len(row) - 1 and row[last + 1]:
            last += 1
        low[index], high[index] = offsets[first], offsets[last]
    return points, normals, low, high


def trim_ring(mask, centre, details=None):
    """An even ribbon from the mask, with the stretches where it strays repaired.

    The mask's two edges are read along the line; a stretch whose width or edge
    position departs from the line's own steady course is replaced by that course.
    Each edge is judged alone: a spill moves one edge and leaves the other.

    Returns ``(ring, seen, widths)`` -- ``seen`` is the share of the line where the
    mask's own edges were kept -- or None when there is nothing steady to go on.
    ``details``, a dict, receives the samples: ``points``, ``normals``, the two
    edges ``low`` / ``high`` (signed distance along the normal) and which of them
    are the mask's own (``low_seen`` / ``high_seen``).
    """
    from scipy.ndimage import median_filter
    points, normals, low, high = _edges(mask, centre)
    width = high - low
    reached = np.isfinite(width) & (low > -TRIM_REACH) & (high < TRIM_REACH)
    if reached.mean() < 0.3:
        return None
    usual = float(np.median(width[reached]))
    low_good = high_good = reached & (np.abs(width - usual) <= max(TRIM_TOLERANCE, 0.25 * usual))
    samples = np.arange(len(width))

    def course(values, steady):
        kept = np.flatnonzero(steady)
        if len(kept) < 10:
            return None
        return np.interp(samples, kept, median_filter(values[kept], 31, mode='nearest'))

    # The steady course comes from stretches where both edges are right. After
    # that each edge is judged alone.
    for _ in range(2):
        low_course, high_course = course(low, low_good), course(high, high_good)
        if low_course is None or high_course is None:
            return None
        low_good = reached & (np.abs(low - low_course) <= TRIM_TOLERANCE)
        high_good = reached & (np.abs(high - high_course) <= TRIM_TOLERANCE)
    low = np.where(low_good, low, low_course)
    high = np.where(high_good, high, high_course)
    if details is not None:
        details.update(points=points, normals=normals, low=low, high=high,
                       low_seen=low_good, high_seen=high_good)
    ring = np.vstack([points + normals * low[:, None], (points + normals * high[:, None])[::-1]])
    return ring, float((low_good.mean() + high_good.mean()) / 2.0), high - low
