"""Finish a glue line's mask as one smooth ribbon of even width (opt-in).

The adhesive model's mask finds the line but its boundary wanders from sample to
sample, and on soft pictures it reads narrow or breaks. Here the mask's edges and
the band's own edges, read from the photo and placed by trained trees, are both
taken as evidence, weighted by what the picture is like at each place, and one
smooth ribbon is fitted to them.

The method and its measurements come from the ``tooling/pipeline-lab`` branch
(``docs/glue-line/GLUE_LINE_FINDINGS.md`` there). ``bead_edges``, ``ribbon``,
``edge_methods`` and ``bead_ml`` are copies of its ``segmentation`` modules.
"""
import cv2
import numpy as np

from . import bead_edges, edge_methods, ends, outline, ribbon

MAX_MISFIT = 6.0     # pixels: a ribbon further than this from its evidence is not used
MIN_COVER = 0.7      # nor one covering less than this share of the line it came from
CHECK_MISFIT = 2.0   # above this, or below CHECK_BACKED, the ribbon is used but marked 'check'
CHECK_BACKED = 0.8
NEAR_OUTLINE = 22    # the band's middle is looked for this close to the mask's middle


def glue_region(photo, source):
    """Where glue can lie: the panel strips of the source mask, cardboard removed."""
    return (source > 0).astype(np.uint8) & (1 - outline.cardboard_mask(photo))


def finish(photo, region, mask, centre):
    """The ribbon for one line, or why there is none.

    ``mask`` is the line's adhesive mask in photo pixels and ``centre`` its measured
    centreline. Returns a dict with ``state``: ``ok`` or ``check`` (``ring`` holds
    the ribbon's outline, round at both ends; ``check`` means the evidence fits it
    poorly or an end was not found in the mask), or
    ``refused`` (``reason`` says why; the mask is to be used as it is).
    """
    from . import bead_ml                      # registers the trees; needs lightgbm
    centre = np.asarray(centre, dtype=np.float64)
    if len(centre) < 10:
        return dict(state='refused', reason='line too short')
    evidence = {}
    trimmed = outline.trim_ring((mask > 0).astype(np.uint8), centre, details=evidence)
    if trimmed is None:
        return dict(state='refused', reason='mask has no steady edges')
    weights = dict(ribbon.STATE_WEIGHT)
    # The trees already allow for the core lying at an edge, which is why the
    # steepest slope is given little weight there and the trees are not.
    weights['edges'] = dict(weights['edges'], **{'one-sided': 1.0})
    middle = outline.centre_of(trimmed[0])
    measured = bead_edges.measure(photo, middle, region, edge_method=edge_methods.METHODS['trees'],
                                  search=NEAR_OUTLINE)
    if measured is None:
        return dict(state='refused', reason='no band found in the photo')
    low, high = ribbon.gather(measured, evidence, weights)
    fitted = ribbon.fit_ribbon(measured['points'][measured['span']], low, high)
    if fitted is None:
        return dict(state='refused', reason='too little evidence')
    covered = len(fitted['centre']) * bead_edges.STEP / max(
        1.0, float(np.linalg.norm(np.diff(middle, axis=0), axis=1).sum()))
    if fitted['misfit'] > MAX_MISFIT or covered < MIN_COVER:
        return dict(state='refused', reason=f"ribbon does not fit (misfit {fitted['misfit']:.0f} px, "
                                            f"{covered:.0%} of the line)")
    # The fitted outline is cut straight across at each end, a few pixels from the
    # glue's tip. The mask stops on the tip, so the ends are taken from it and rounded.
    ring, moved, found = ends.round_ends(mask, fitted['centre'], fitted['widths'])
    if ring is None:
        ring, moved, found = fitted['ring'], [0.0, 0.0], [False, False]
    doubtful = fitted['misfit'] > CHECK_MISFIT or fitted['backed'] < CHECK_BACKED or not all(found)
    return dict(state='check' if doubtful else 'ok', ring=ring,
                backed=float(fitted['backed']), misfit=float(fitted['misfit']),
                mask_kept=float(trimmed[1]), deviations=len(fitted['deviations']),
                end_moved_px=moved, end_found=found)


def ring_mask(shape, ring):
    """The ribbon's outline filled, as a mask of the photo's size."""
    filled = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(filled, [np.round(np.asarray(ring)).astype(np.int32)], 255)
    return filled
