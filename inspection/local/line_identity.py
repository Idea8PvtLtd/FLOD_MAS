"""Which glue lines on the bed are already measured: by place and look, not by number.

A capture finds every glue line on the bed. Counting them from the left does not
identify a line: take a finished fabric off the left and add a new one on the
right, and the new one takes the old one's number. So each measured line keeps a
fingerprint, where its glue edge runs and what the fabric around it looks like,
and a line in a later capture is the same line only when both match. A fabric
removed is simply gone; one added anywhere, or put in the place of one taken
away, is new; one nudged by a hand is still the same.

Fingerprints are plain JSON (a path of points and a greyscale image in base64),
so they travel in the capture metadata and are saved with it.

Calibrated on real captures of white lace fabric (2026-10-16): the same fabric in
a later photo moved 1-3 px and its looks correlated 0.97-1.00; a different fabric
in the same place correlated 0.31-0.52, and different fabrics lay 287 px or more
apart. The look compares only fine texture, the lace pattern: at the coarse scale
every one of these fabrics looks alike (0.98).
"""
import base64

import cv2
import numpy as np

PATH_POINTS = 32       # points kept along the glue edge
LOOK_SIZE = (64, 256)  # across x along the line, in look pixels
LOOK_MARGIN = 250      # camera pixels of fabric kept around the glue edge
TEXTURE_SIGMA = 3      # blur removed from the look, leaving the lace texture
MAX_SHIFT_PX = 80      # mean distance between two paths of the same line
MIN_LOOK_MATCH = .80   # texture correlation of the same line
# Below this texture (standard deviation, grey levels) a look cannot be compared
# and place alone decides. Real fabric measured 5.2 (white) to 14.2 (blue);
# camera noise on a flat surface 0.5.
MIN_TEXTURE = 2.0


def fingerprint(photo, line):
    """Place and look of a line given in camera-frame pixels."""
    points = np.asarray(line['original_xy'], float)
    keep = np.linspace(0, len(points) - 1, min(PATH_POINTS, len(points))).round().astype(int)
    height, width = photo.shape[:2]
    # The look also takes in whatever lies within LOOK_MARGIN of the edge. A
    # narrower look of the fabric alone was tried and matched the same fabric
    # far worse (0.46-0.81), being too sensitive to the 1-3 px the detected edge
    # moves between photos. So a neighbouring fabric moved within this margin
    # can make a measured line be measured again; it can never hide a line.
    x0, y0 = np.maximum(points.min(0) - LOOK_MARGIN, 0).astype(int)
    x1, y1 = np.minimum(points.max(0) + LOOK_MARGIN, [width, height]).astype(int)
    patch = photo[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # The look runs along the line whichever way it lies, so a line across the
    # photo (sideways fabric) is sampled as finely along its length.
    across, along = LOOK_SIZE
    size = (across, along) if patch.shape[0] >= patch.shape[1] else (along, across)
    look = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
    return {'path': np.rint(points[keep]).astype(int).tolist(), 'size': list(size),
            'look': base64.b64encode(np.ascontiguousarray(look, np.uint8).tobytes()).decode('ascii')}


def path_distance(a, b):
    """Mean distance from each path's points to the other path (pixels)."""
    p, q = np.asarray(a['path'], float), np.asarray(b['path'], float)
    gaps = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    return float((gaps.min(1).mean() + gaps.min(0).mean()) / 2)


def _texture(item):
    width, height = item['size']
    image = np.frombuffer(base64.b64decode(item['look']), np.uint8).reshape(height, width)
    image = image.astype(np.float32)
    detail = (image - cv2.GaussianBlur(image, (0, 0), TEXTURE_SIGMA)).ravel()
    return detail - detail.mean()


def look_match(a, b):
    """Correlation of the two looks' fine texture, from -1 to 1.

    0 for looks of different shapes; None when either has too little texture to
    compare (a plain fabric), which leaves the decision to place alone.
    """
    if a.get('size') != b.get('size'):
        return 0.
    p, q = _texture(a), _texture(b)
    if min(p.std(), q.std()) < MIN_TEXTURE:
        return None
    return float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q)))


def same_line(a, b):
    if path_distance(a, b) > MAX_SHIFT_PX:
        return False
    match = look_match(a, b)
    return match is None or match >= MIN_LOOK_MATCH


def already_measured(item, measured):
    return any(same_line(item, other) for other in measured)
