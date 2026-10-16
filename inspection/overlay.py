"""Simple segment grades and calibrated width labels in image coordinates."""
import cv2
import numpy as np

# The adhesive mask is tinted before any grade is drawn. The alpha is low so the
# fabric texture stays readable through the fill, and the colour is deliberately
# none of the red/green/amber used for grades, so a tint can never be mistaken
# for a verdict.
MASK_TINT = (255, 150, 40)
MASK_OUTLINE = (255, 120, 0)
MASK_ALPHA = .30

# The ten graded segments are drawn as one continuous centreline, so the join
# between two of them is invisible until their grades differ. A bar across the
# adhesive at each join shows where one segment ends and the next begins while
# leaving the line unbroken. It is drawn white over a dark outline, like the
# width labels, so the separation reads against either fabric tone without
# introducing a colour that could be mistaken for a grade.
SEPARATOR_COLOR = (255, 255, 255)
SEPARATOR_OUTLINE = (0, 0, 0)

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = .55
LABEL_GAP = 8          # px between the glue edge and the nearest edge of a width label
LABEL_MAX_STEPS = 6    # at most this many label heights further out to clear earlier labels


def tint_mask(image, mask):
    """Translucent fill and outline of the detected adhesive, over a copy.

    The mask is drawn as the model produced it rather than as a fitted polygon,
    so the operator sees exactly the region that was measured. Called only from
    the automatic path: the single-line engine draws its own contours.
    """
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[..., 0]
    binary = (binary > 0).astype(np.uint8)
    # Checked before the empty case: a wrong-sized mask is a coordinate bug even
    # when it happens to be empty, and drawing nothing would hide it.
    if binary.shape != image.shape[:2]:
        raise ValueError('Adhesive mask and image must share the same frame size')
    if not binary.any():
        return image
    tinted = image.copy()
    tinted[binary > 0] = MASK_TINT
    overlay = cv2.addWeighted(tinted, MASK_ALPHA, image, 1 - MASK_ALPHA, 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, MASK_OUTLINE, 2)
    return overlay


def segment_joins(component):
    """Cross-adhesive bars separating each segment from the one before it.

    A segment's first sample lies exactly where the previous segment ended, and
    that sample's own left/right boundary points already span the adhesive, so
    the join can be drawn straight from the measurement rather than estimated.
    The first segment has no join ahead of it and is left out. A segment whose
    samples carry no boundary geometry is skipped, never guessed at.
    """
    samples = component.get('samples', ())
    joins = []
    for segment in list(component.get('segments', ()))[1:]:
        boundary = None
        for sample in samples:
            if sample.get('segment') != segment.get('segment'):
                continue
            points = [sample.get(key) for key in ('left_x', 'left_y', 'right_x', 'right_y')]
            if None in points:
                continue
            # A sample the measurement itself accepted is the boundary; the
            # first sample with geometry is kept only as a fallback, so a lone
            # rejected sample cannot hide the join between graded segments.
            if sample.get('valid'):
                boundary = points
                break
            if boundary is None:
                boundary = points
        if boundary is not None:
            joins.append(((int(boundary[0]), int(boundary[1])),
                          (int(boundary[2]), int(boundary[3]))))
    return joins


def segment_ends(component):
    """Bars where the first segment starts and the last segment ends.

    The same boundary rule as the joins, walked inward from each end: the
    outermost accepted sample of the end segment, falling back to the outermost
    sample with geometry. An end segment without boundary geometry adds no bar.
    """
    segments = list(component.get('segments', ()))
    if not segments:
        return []
    samples = component.get('samples', ())
    ends = []
    for number, ordered in ((segments[0].get('segment'), samples),
                            (segments[-1].get('segment'), list(reversed(samples)))):
        boundary = None
        for sample in ordered:
            if sample.get('segment') != number:
                continue
            points = [sample.get(key) for key in ('left_x', 'left_y', 'right_x', 'right_y')]
            if None in points:
                continue
            if sample.get('valid'):
                boundary = points
                break
            if boundary is None:
                boundary = points
        if boundary is not None:
            ends.append(((int(boundary[0]), int(boundary[1])),
                         (int(boundary[2]), int(boundary[3]))))
    return ends


def _unit(v, fallback=(0., 1.)):
    n = float(np.linalg.norm(v))
    return v / n if n else np.array(fallback)


def label_side(samples):
    """One side for all labels of a line: the normal of the line as a whole that points right,
    or down for a horizontal line. Per-segment normals would swap sides on every bend."""
    ends = np.array([[samples[0]['center_x'], samples[0]['center_y']],
                     [samples[-1]['center_x'], samples[-1]['center_y']]], float)
    tangent = _unit(ends[1] - ends[0])
    normal = np.array([tangent[1], -tangent[0]])
    return -normal if normal[0] < -1e-6 or (abs(normal[0]) <= 1e-6 and normal[1] < 0) else normal


def label_patch(label, color, rotated):
    """The label drawn on its own (outlined text and its mask), turned 90 degrees when rotated."""
    (tw, th), base = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, 4)
    w, h = tw + 4, th + base + 4
    patch = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for target, value_outline, value_text in ((patch, (0, 0, 0), color), (mask, 255, 255)):
        cv2.putText(target, label, (2, th + 2), LABEL_FONT, LABEL_SCALE, value_outline, 4, cv2.LINE_AA)
        cv2.putText(target, label, (2, th + 2), LABEL_FONT, LABEL_SCALE, value_text, 1, cv2.LINE_AA)
    if rotated:   # reads bottom to top, like a label on a drawing's side
        patch, mask = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return patch, mask


def draw_label(overlay, label, color, top_left, rotated):
    patch, mask = label_patch(label, color, rotated)
    x, y = top_left
    h, w = mask.shape
    region = overlay[y:y + h, x:x + w]
    sel = mask[:region.shape[0], :region.shape[1]] > 0
    region[sel] = patch[:region.shape[0], :region.shape[1]][sel]


def label_origin(shape, samples, label, side, placed, rotated=False):
    """Top-left corner of a label box beside a segment, clear of the glue, its bars and earlier labels.

    The label is pushed out from the segment's middle along the segment's normal (turned to
    `side`), past the glue edge, by enough that the box itself does not reach back over the
    glue. On a line running across the image the text is turned 90 degrees (`rotated`), so
    every label is as narrow along the line as on a vertical line and they line up on one side
    the same way. A box that would still overlap one in `placed` is pushed further out.
    """
    middle = samples[len(samples) // 2]
    centre = np.array([middle['center_x'], middle['center_y']], float)
    tangent = _unit(np.array([samples[-1]['center_x'] - samples[0]['center_x'],
                              samples[-1]['center_y'] - samples[0]['center_y']], float))
    normal = np.array([tangent[1], -tangent[0]])
    if normal @ side < 0:
        normal = -normal
    half_glue = max((float(np.hypot(middle[f'{s}_x'] - centre[0], middle[f'{s}_y'] - centre[1]))
                     for s in ('left', 'right') if middle.get(f'{s}_x') is not None), default=12.)
    (tw, th), base = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, 4)
    bw, bh = tw + 4, th + base + 4
    if rotated:
        bw, bh = bh, bw
    reach = bw / 2 * abs(normal[0]) + bh / 2 * abs(normal[1])
    distance = half_glue + LABEL_GAP + reach
    for _ in range(LABEL_MAX_STEPS):
        box_centre = centre + normal * distance
        x = int(round(box_centre[0] - bw / 2))
        y = int(round(box_centre[1] - bh / 2))
        x = max(0, min(x, shape[1] - bw))
        y = max(0, min(y, shape[0] - bh))
        box = (x, y, x + bw, y + bh)
        if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
            break
        distance += min(bw, bh) + 4
    placed.append(box)
    return x, y


def annotated_overlay(image, measurement, mask=None):
    overlay = tint_mask(image, mask) if mask is not None else image.copy()
    labels, placed = [], []
    for component in measurement['components']:
        graded = [s for s in component['samples'] if s['segment'] != -1 and s.get('center_x') is not None]
        side = label_side(graded or component['samples'])
        # All labels of a line sit on one side, written across the line: upright beside a line
        # running down the image, turned 90 degrees beside a line running across it.
        rotated = abs(side[1]) > abs(side[0])
        for segment in component['segments']:
            samples = [s for s in component['samples'] if s['segment'] == segment['segment']]
            if not samples:
                continue
            # None means this segment had no measurement to grade, so it is
            # marked WARNING at the segment itself rather than pass or fail.
            grade = segment['within_tolerance']
            color = (40, 40, 240) if grade is False else (60, 210, 60) if grade else (0, 190, 255)
            points = np.rint([[s['center_x'], s['center_y']] for s in samples]).astype(np.int32)
            cv2.polylines(overlay, [points], False, color, 5 if grade is False else 2)
            if grade is False:
                valid = [s for s in samples if s['valid']]
                if valid:
                    polygon = np.rint([[s['left_x'], s['left_y']] for s in valid] +
                                     [[s['right_x'], s['right_y']] for s in reversed(valid)]).astype(np.int32)
                    tint = overlay.copy()
                    cv2.fillPoly(tint, [polygon], color)
                    overlay = cv2.addWeighted(tint, .35, overlay, .65, 0)
            width = segment['average_width_mm']
            pixels = segment['average_width_px']
            label = (f'{width:.1f} mm' if width is not None else   # graded at 0.1 mm
                     f'{pixels:.1f} px' if pixels is not None else 'WARNING')
            labels.append((label, color, rotated,
                           label_origin(overlay.shape, samples, label, side, placed, rotated)))
        for start, end in segment_joins(component) + segment_ends(component):
            cv2.line(overlay, start, end, SEPARATOR_OUTLINE, 4, cv2.LINE_AA)
            cv2.line(overlay, start, end, SEPARATOR_COLOR, 2, cv2.LINE_AA)
    # Labels last, so no bar or line is drawn over a number.
    for label, color, rotated, top_left in labels:
        draw_label(overlay, label, color, top_left, rotated)
    return overlay
