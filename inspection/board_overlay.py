"""Pictures of a whole board: every glue line on one photo, and one small picture per line.

Used only by the scene capture mode, where one photo holds every line of a board.
Nothing here needs a window (cv2 and numpy only), so it can be tested and saved
to disk without Tk.

An ``entry`` is one line of the board as the pipeline reports it: ``line_id``,
``status``, ``measurement`` (None when the try failed), ``centre`` and ``outline``
in photo pixels, ``crop`` / ``crop_mask`` / ``origin`` and ``new``.
"""
import cv2
import numpy as np

from .overlay import annotated_overlay

# The same colours inspection/overlay.py gives a segment that passed, failed or
# could not be graded (BGR), so a line means the same thing in both pictures.
PASS_COLOR = (60, 210, 60)
FAIL_COLOR = (40, 40, 240)
WARNING_COLOR = (0, 190, 255)
STATUS_COLORS = {'PASS': PASS_COLOR, 'FAIL': FAIL_COLOR, 'WARNING': WARNING_COLOR}
# The tag of a line read from the latest photo. Deliberately none of the three
# verdict colours, so "new" can never be read as a grade.
NEW_COLOR = (250, 165, 96)
BACKGROUND = (17, 8, 3)     # ui_theme COLORS['camera'], for the empty part of a card picture

# Drawing sizes are given for a 1920 pixel overview and follow the picture's size.
# The overview is shown at roughly a third of that on a 1366 x 768 screen, so a
# badge must be this large for its number to be readable there.
REFERENCE_EDGE = 1920.0
BADGE_RADIUS = 36
OUTLINE_WIDTH = 3
NEW_OUTLINE_WIDTH = 7
FILL_ALPHA = .35


def entry_status(entry):
    """PASS, FAIL or WARNING for one line; a line that could not be measured is a FAIL."""
    if entry.get('measurement') is None:
        return 'FAIL'
    status = entry.get('status')
    return status if status in STATUS_COLORS else 'WARNING'


def aggregate_verdict(entries):
    """One verdict for the board: EMPTY, FAIL, WARNING or PASS.

    EMPTY with no line at all. One failed or unmeasured line fails the board;
    otherwise one line that could not be graded makes it a WARNING.
    """
    statuses = [entry_status(entry) for entry in entries]
    if not statuses:
        return 'EMPTY'
    if 'FAIL' in statuses:
        return 'FAIL'
    if 'WARNING' in statuses:
        return 'WARNING'
    return 'PASS'


def _points(values):
    """An N x 2 float array, empty when there is nothing usable."""
    if values is None:
        return np.zeros((0, 2))
    points = np.asarray(values, dtype=float)
    if points.size < 2:
        return np.zeros((0, 2))
    return points.reshape(-1, 2)


def line_is_upright(entry):
    """True when the line runs more up-down than left-right in the photo."""
    centre = _points(entry.get('centre'))
    if len(centre) < 2:
        return True
    extent = np.ptp(centre, axis=0)
    return bool(extent[1] >= extent[0])


def badge_position(centre, radius, size):
    """Where the number badge of a line goes: just beyond its top (or left) end.

    The badge sits on the line's own course continued past its end, so it names
    the line without covering it, whichever way the line runs. It is kept inside
    the picture, so a line that reaches the edge still gets a whole badge.
    """
    centre = _points(centre)
    width, height = size
    if not len(centre):
        return None
    first, last = centre[0], centre[-1]
    upright = np.ptp(centre[:, 1]) >= np.ptp(centre[:, 0])
    axis = 1 if upright else 0
    end, other = (first, last) if first[axis] <= last[axis] else (last, first)
    direction = end - other
    length = float(np.linalg.norm(direction))
    direction = direction / length if length > 1e-6 else np.array([0., -1.])
    x, y = end + direction * radius * 1.35
    return (int(min(max(x, radius + 2), max(radius + 2, width - radius - 3))),
            int(min(max(y, radius + 2), max(radius + 2, height - radius - 3))))


def _draw_badge(picture, entry, position, radius, unit):
    color = STATUS_COLORS[entry_status(entry)]
    cv2.circle(picture, position, radius + max(2, round(4 * unit)), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(picture, position, radius + max(1, round(2 * unit)), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(picture, position, radius, color, -1, cv2.LINE_AA)
    text = str(entry.get('line_id', '?'))
    thickness = max(2, round(4 * unit))
    scale = 1.6 * unit
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    # A two-digit number is shrunk to stay inside its badge.
    if text_width > radius * 1.5:
        scale *= radius * 1.5 / text_width
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(picture, text, (position[0] - text_width // 2, position[1] + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if not entry.get('new'):
        return
    label = 'NEW'
    scale = .8 * unit
    thickness = max(1, round(2 * unit))
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(3, round(7 * unit))
    left = position[0] + radius + pad
    if left + text_width + 2 * pad > picture.shape[1]:
        left = position[0] - radius - 3 * pad - text_width
    top = position[1] - text_height // 2 - pad
    cv2.rectangle(picture, (left, top), (left + text_width + 2 * pad, top + text_height + 2 * pad),
                  (0, 0, 0), -1)
    cv2.rectangle(picture, (left + 1, top + 1),
                  (left + text_width + 2 * pad - 1, top + text_height + 2 * pad - 1), NEW_COLOR, -1)
    cv2.putText(picture, label, (left + pad, top + pad + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def board_overview(image, entries, max_edge=1920):
    """The photo scaled down with every line outlined, coloured and numbered.

    The photo is scaled FIRST and everything is drawn at the scaled coordinates,
    so the 16 megapixel photo is never copied. Each line's adhesive outline is
    filled lightly and edged in the colour of its verdict; a line that has no
    outline (its measurement failed) is drawn along its course instead. A line
    read from the latest photo has a thicker edge and a NEW tag at its number.
    """
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        picture = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
    else:
        picture = image.copy()
    if picture.ndim == 2:
        picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    unit = max(picture.shape[:2]) / REFERENCE_EDGE
    radius = max(9, round(BADGE_RADIUS * unit))
    shapes = []
    for entry in entries:
        color = STATUS_COLORS[entry_status(entry)]
        thick = max(1, round((NEW_OUTLINE_WIDTH if entry.get('new') else OUTLINE_WIDTH) * unit))
        outline = _points(entry.get('outline')) * scale
        centre = _points(entry.get('centre')) * scale
        shapes.append((entry, color, thick, np.rint(outline).astype(np.int32),
                       np.rint(centre).astype(np.int32), centre))
    # One blend for all the fills, instead of one whole-picture blend per line.
    filled = [shape for shape in shapes if len(shape[3]) >= 3]
    if filled:
        tint = picture.copy()
        for _, color, _, outline, _, _ in filled:
            cv2.fillPoly(tint, [outline], color)
        picture = cv2.addWeighted(tint, FILL_ALPHA, picture, 1 - FILL_ALPHA, 0)
    for _, color, thick, outline, course, _ in shapes:
        if len(outline) >= 3:
            cv2.polylines(picture, [outline], True, color, thick, cv2.LINE_AA)
        elif len(course) >= 2:
            # No adhesive outline: show where the line was looked for.
            cv2.polylines(picture, [course], False, (0, 0, 0), thick + max(2, round(6 * unit)), cv2.LINE_AA)
            cv2.polylines(picture, [course], False, color, thick + max(1, round(2 * unit)), cv2.LINE_AA)
    # Badges last, so no line's outline is ever drawn across a number.
    size = (picture.shape[1], picture.shape[0])
    for entry, _, _, outline, _, centre in shapes:
        position = badge_position(centre if len(centre) else outline, radius, size)
        if position is not None:
            _draw_badge(picture, entry, position, radius, unit)
    return picture


def annotated_line(entry, margin=.15):
    """The line's crop with its grades drawn, cut close around the adhesive; None when failed.

    The measurement's samples are in photo pixels, so they are moved by the
    crop's ``origin`` before ``annotated_overlay`` draws on the crop (as
    ``LineResultView._apply_ratio_preview`` does). The entry's own measurement is
    not changed.
    """
    crop, measurement = entry.get('crop'), entry.get('measurement')
    if crop is None or measurement is None:
        return None
    origin_x, origin_y = entry.get('origin') or (0, 0)
    components = []
    for component in measurement['components']:
        samples = []
        for sample in component['samples']:
            moved = dict(sample)
            for key in ('center_x', 'left_x', 'right_x'):
                if moved.get(key) is not None:
                    moved[key] -= origin_x
            for key in ('center_y', 'left_y', 'right_y'):
                if moved.get(key) is not None:
                    moved[key] -= origin_y
            samples.append(moved)
        components.append(dict(component, samples=samples))
    mask = entry.get('crop_mask')
    picture = annotated_overlay(crop, dict(measurement, components=components), mask)
    # Cut to the adhesive plus a small margin, so the line fills
    # the card whatever margins the crop was made with.
    box = None
    if mask is not None and np.any(mask):
        box = cv2.boundingRect((np.asarray(mask) > 0).astype(np.uint8))
    else:
        centre = _points(entry.get('centre')) - (origin_x, origin_y)
        if len(centre):
            low, high = centre.min(axis=0), centre.max(axis=0)
            box = (int(low[0]), int(low[1]), int(high[0] - low[0]) + 1, int(high[1] - low[1]) + 1)
    if box is None:
        return picture
    x, y, w, h = box
    pad = int(max(12, margin * min(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(picture.shape[1], x + w + pad), min(picture.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return picture
    return picture[y0:y1, x0:x1]


def fit_card(picture, size, upright=True, stretch=3.0):
    """``picture`` placed in a ``size`` (width, height) box on a dark background.

    A whole glue line is about fifty times longer than it is wide, so shown to
    scale in a card it would be two pixels thick. The picture is therefore fitted
    along the line and may be widened ACROSS the line by up to ``stretch`` times
    to use the box. It is a quick look at where the line is good or bad, not a
    picture to measure from; the zoomed board is to scale.
    """
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    card = np.full((height, width, 3), BACKGROUND, np.uint8)
    if picture is None or picture.size == 0:
        # No picture exists for a line that could not be measured.
        inset = max(4, min(width, height) // 4)
        cv2.line(card, (inset, inset), (width - inset, height - inset), FAIL_COLOR, 2, cv2.LINE_AA)
        cv2.line(card, (width - inset, inset), (inset, height - inset), FAIL_COLOR, 2, cv2.LINE_AA)
        return card
    source_height, source_width = picture.shape[:2]
    scale = min(width / source_width, height / source_height)
    if upright:
        scale_x = min(width / source_width, scale * stretch)
        scale_y = scale
    else:
        scale_x = scale
        scale_y = min(height / source_height, scale * stretch)
    new_width = max(1, min(width, round(source_width * scale_x)))
    new_height = max(1, min(height, round(source_height * scale_y)))
    small = cv2.resize(picture, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    card[top:top + new_height, left:left + new_width] = small
    return card


def line_card_image(entry, size):
    """The small picture of one line for its card: ``size`` is (width, height).

    Which way the picture is fitted comes from the line's ``centre``, so a line
    lying sideways fills a wide box just as an upright one fills a tall box.
    """
    return fit_card(annotated_line(entry), size, line_is_upright(entry))
