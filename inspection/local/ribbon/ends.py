"""The two ends of a ribbon: where the glue really stops, and a round tip there.

``ribbon.fit_ribbon`` closes its outline with a straight line across each end, and
its ends lie where the band's darkness fades in the photo, which on pale fabric is
some pixels off. The adhesive mask has the better end: it stops on the glue's tip.
Here each end of the ribbon is moved to the mask's tip and closed with a half
circle of the ribbon's own width, which is how a glue line ends.
"""
import numpy as np

STEP = 2.0            # spacing of the samples along the centre line, as in bead_edges
TANGENT_REACH = 15    # samples (30 px) the end's direction is taken over
SEARCH_WIDTHS = 1.5   # the mask's tip is looked for this many strip widths either way
TRUST_WIDTHS = 1.0    # a tip further than this from the ribbon's end is not believed
MAX_HOLE = 2          # pixels: a hole in the mask this short does not end it
CAP_STEP = 2.0        # pixels between the points of the round tip


def _outward(centre, at_start):
    """The end point of the centre line and the unit direction leading out of it."""
    reach = min(TANGENT_REACH, len(centre) - 1)
    end, inner = (centre[0], centre[reach]) if at_start else (centre[-1], centre[-1 - reach])
    direction = end - inner
    return end, direction / max(float(np.linalg.norm(direction)), 1e-9)


def mask_tip(mask, centre, widths, at_start):
    """How far beyond the ribbon's end the mask's tip lies, in pixels along the
    line (negative: the mask stops before the ribbon does). None when the mask is
    not there, or runs on past the stretch that is searched."""
    end, direction = _outward(centre, at_start)
    width = float(widths[0] if at_start else widths[-1])
    normal = np.array([-direction[1], direction[0]])
    reach = int(np.ceil(SEARCH_WIDTHS * width))
    along = np.arange(-reach, reach + 1)
    across = np.arange(-int(np.ceil(width)), int(np.ceil(width)) + 1)
    columns = np.rint(end[0] + along[None] * direction[0] + across[:, None] * normal[0]).astype(int)
    rows = np.rint(end[1] + along[None] * direction[1] + across[:, None] * normal[1]).astype(int)
    inside = (columns >= 0) & (columns < mask.shape[1]) & (rows >= 0) & (rows < mask.shape[0])
    found = np.zeros(columns.shape, bool)
    found[inside] = mask[rows[inside], columns[inside]] > 0
    present = np.flatnonzero(found.any(axis=0))
    if present.size == 0 or present[0] > MAX_HOLE:
        return None                     # the mask does not reach in from the ribbon's body
    breaks = np.flatnonzero(np.diff(present) > MAX_HOLE + 1)
    last = present[breaks[0]] if breaks.size else present[-1]
    if last >= len(along) - 1:
        return None                     # still mask at the far side: its tip was not reached
    return float(along[last])


def capped_ring(centre, widths, start_moved=0.0, end_moved=0.0):
    """The ribbon's outline with round ends.

    ``start_moved`` / ``end_moved`` are how far each tip lies beyond the centre
    line's own end (``mask_tip``). The centre line is cut back or lengthened so
    that it stops half a width before each tip, and a half circle closes it.
    Returns None when the ends leave too little of the line.
    """
    centre = np.asarray(centre, np.float64)
    widths = np.asarray(widths, np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    first = widths[0] / 2.0 - start_moved
    last = arc[-1] + end_moved - widths[-1] / 2.0
    if last - first < 4 * STEP:
        return None
    along = np.linspace(first, last, max(3, int(round((last - first) / STEP)) + 1))
    points = np.column_stack([np.interp(along, arc, centre[:, axis]) for axis in (0, 1)])
    # Beyond the line's own ends it runs straight on.
    for at_start, beyond in ((True, along < 0.0), (False, along > arc[-1])):
        end, direction = _outward(centre, at_start)
        distance = -along[beyond] if at_start else along[beyond] - arc[-1]
        points[beyond] = end + distance[:, None] * direction
    half = np.interp(along, arc, widths) / 2.0
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    def cap(index, outward, begin):
        """Half circle from one edge round the tip to the other."""
        radius = half[index]
        angles = np.linspace(0.0, np.pi, max(5, int(np.ceil(np.pi * radius / CAP_STEP)) + 1))[1:-1]
        return (points[index] + radius * (np.cos(angles)[:, None] * begin
                                          + np.sin(angles)[:, None] * outward))

    return np.vstack([points - normals * half[:, None],
                      cap(-1, tangent[-1], -normals[-1]),
                      (points + normals * half[:, None])[::-1],
                      cap(0, -tangent[0], normals[0])])


def round_ends(mask, centre, widths):
    """The ribbon's outline ending round on the mask's tips.

    Returns ``(ring, moved, found)``: how far each end (start, end) was moved from
    the ribbon's own end, and whether the mask's tip was found near it. An end
    whose tip is not found stays where the ribbon put it and is still rounded.
    """
    moved, found = [], []
    for at_start in (True, False):
        tip = mask_tip(mask, centre, widths, at_start)
        width = float(widths[0] if at_start else widths[-1])
        known = tip is not None and abs(tip) <= TRUST_WIDTHS * width
        moved.append(float(tip) if known else 0.0)
        found.append(bool(known))
    ring = capped_ring(centre, widths, moved[0], moved[1])
    return ring, moved, found
