"""Which glue line is which, from one photo of the board to the next (scene mode).

The hand path names a line by its position counted from the left, which is only
right while lines are added on the right. Here a line is named by WHERE it runs in
the photo: its course is kept, and a line found in a later photo is the same line
when it runs along a known one. A line gets its number the first time it is seen
and keeps it, wherever later lines are added, and the first good reading of a line
is the one that stays: a later cloth may cover part of an earlier line, and the
piece still showing must neither be read again nor become a line of its own.

Numpy only; nothing here measures anything.
"""
import threading
import numpy as np
from .local.geometry import smoothed_path

STEP = 10.0          # pixels between the points of a kept course
SMOOTH_ROWS = 101    # the edge is raw mask pixels; its steps are averaged away first


def course(points, step=STEP):
    """A line's course: its points smoothed, then spaced evenly along its length."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return points.copy()
    points = smoothed_path(points, SMOOTH_ROWS)
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] <= 0:
        return points[:1].copy()
    keep = np.concatenate(([True], np.diff(arc) > 1e-9))
    at = np.linspace(0., arc[-1], max(2, int(round(arc[-1] / step)) + 1))
    return np.column_stack([np.interp(at, arc[keep], points[keep, i]) for i in range(2)])


def length_of(points):
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.


def along(found, known):
    """How ``found`` runs along ``known``.

    Returns (mean distance, overlap length, mean offset): over the stretch of
    ``found`` that lies beside ``known`` (its points whose nearest place on
    ``known`` is not one of the two ends of ``known``), the mean distance to
    ``known``, the length of that stretch and the mean step from ``found`` to
    ``known``. (inf, 0, None) when no part of it lies beside ``known``.
    """
    if len(found) < 2 or len(known) < 2:
        return float('inf'), 0., None
    start, step = known[:-1], np.diff(known, axis=0)
    squared = np.maximum(np.sum(step * step, axis=1), 1e-12)
    # Every point of the found line against every piece of the known one.
    place = np.einsum('nmk,mk->nm', found[:, None, :] - start[None, :, :], step) / squared
    inside = np.clip(place, 0., 1.)
    nearest = start[None, :, :] + inside[:, :, None] * step[None, :, :]
    distance = np.linalg.norm(found[:, None, :] - nearest, axis=2)
    piece = np.argmin(distance, axis=1)
    rows = np.arange(len(found))
    place = place[rows, piece]
    beside = ~(((piece == 0) & (place < 0)) | ((piece == len(start) - 1) & (place > 1)))
    if beside.sum() < 2:
        return float('inf'), 0., None
    spacing = length_of(found) / (len(found) - 1)
    offset = (nearest[rows, piece] - found)[beside]
    return float(distance[rows, piece][beside].mean()), float(beside.sum() * spacing), offset.mean(axis=0)


class BoardLines:
    """The lines known on the board in front of the camera, and their readings.

    ``match_distance_px``: a found line is a known one when it runs this close to
    it on average. ``min_overlap``: over at least this share of the SHORTER of the
    two, so a piece of a line still matches the whole line, and two lines that only
    meet end to end do not.

    ``min_new_length_ratio`` is the engine's ``min_line_length_ratio`` idea, applied
    here and only to lines that match nothing: such a line becomes a new line only
    if it is at least this share of the longest line known or found. The engine's
    own filter compares with the longest line of the one photo, which is no help
    when that longest line is itself a piece of a covered line; a known line is
    never refused for being short, since it is not measured again anyway.
    0 keeps every unmatched line.

    A line whose try failed is owed again, but only when it is seen at nearly its
    full known length (``RETRY_LENGTH``): a piece of it, read as if it were the
    whole line, would give a reading of the wrong thing.
    """
    RETRY_LENGTH = 0.9

    def __init__(self, match_distance_px=60.0, min_overlap=0.5, min_new_length_ratio=0.5):
        self.match_distance_px = float(match_distance_px)
        self.min_overlap = float(min_overlap)
        self.min_new_length_ratio = float(min_new_length_ratio)
        if not np.isfinite(self.match_distance_px) or self.match_distance_px <= 0:
            raise ValueError('Match distance must be positive and finite')
        if not 0 < self.min_overlap <= 1:
            raise ValueError('Minimum overlap must be above 0 and at most 1')
        if not 0 <= self.min_new_length_ratio < 1:
            raise ValueError('Minimum new line length ratio must be at least 0 and less than 1')
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        """A new board: nothing is known."""
        with self._lock:
            self._courses = {}      # line_id -> its course, the longest sighting of it
            self._entries = {}      # line_id -> entry (good or failed)
            self._good = set()      # line_ids with a good reading
            self.last_match = []    # what the latest owed() decided, for logs and checks
            self.last_offset = (0., 0.)

    @property
    def entries(self):
        with self._lock:
            return [self._entries[line_id] for line_id in sorted(self._entries)]

    @property
    def count(self):
        """How many lines are known on this board, read or not."""
        with self._lock:
            return len(self._courses)

    def _matches(self, found, shift):
        """For each found course the known line it runs along: (line_id, distance, offset) or None."""
        result = []
        for points in found:
            best = None
            moved = points + shift
            for line_id, known in self._courses.items():
                distance, overlap, offset = along(moved, known)
                shorter = min(length_of(points), length_of(known))
                if (distance <= self.match_distance_px and shorter > 0
                        and overlap >= self.min_overlap * shorter - STEP):
                    if best is None or distance < best[1]:
                        best = (line_id, distance, offset)
            result.append(best)
        return result

    def match(self, lines):
        """Which known line each of ``lines`` is: a list of line_id or None. Changes nothing."""
        with self._lock:
            return [None if m is None else m[0] for m in self._match([course(l['original_xy']) for l in lines])[0]]

    def _match(self, found):
        matches = self._matches(found, np.zeros(2))
        shift = np.zeros(2)
        offsets = [m[2] for m in matches if m is not None]
        if offsets:
            # The board may have been nudged between two photos: every line is then
            # off by the same step. Take it out and look again, so that a line a
            # little further off than the limit is still recognised.
            shift = np.median(np.array(offsets), axis=0)
            again = self._matches(found, shift)
            # A line recognised before the step was removed stays recognised.
            matches = [b if b is not None else a for a, b in zip(matches, again)]
        return matches, shift

    def owed(self, lines):
        """`lines`: the dicts from engine.find_lines (each has 'original_xy'). Returns
        [(index_in_lines, line_id), ...] for the lines to measure now: lines never seen
        (they get the next line_id, 1-based, in order first seen) and known lines whose
        earlier try failed. Known, already measured lines are not returned.

        Each call is a new photo: entries read from an earlier photo stop being 'new'.
        """
        with self._lock:
            for line_id, entry in list(self._entries.items()):
                if entry.get('new'):
                    # A copy: the entry may be on screen from the photo before.
                    self._entries[line_id] = dict(entry, new=False)
            found = [course(line['original_xy']) for line in lines]
            lengths = [length_of(points) for points in found]
            matches, shift = self._match(found)
            longest = max(lengths + [length_of(known) for known in self._courses.values()], default=0.)
            picks, taken, report = [], set(), []
            for index, (points, length, match) in enumerate(zip(found, lengths, matches)):
                if match is None:
                    if len(points) < 2 or length < self.min_new_length_ratio * longest:
                        report.append((index, None, 'short'))
                        continue
                    line_id = max(self._courses, default=0) + 1
                    self._courses[line_id] = points
                    picks.append((index, line_id))
                    taken.add(line_id)
                    report.append((index, line_id, 'new'))
                    continue
                line_id = match[0]
                known = length_of(self._courses[line_id])
                whole = length >= self.RETRY_LENGTH * known
                if line_id not in self._good and line_id not in taken and whole:
                    picks.append((index, line_id))
                    taken.add(line_id)
                    report.append((index, line_id, 'again'))
                else:
                    report.append((index, line_id, 'known'))
                if length > known and line_id not in self._good:
                    # Seen more fully than before and not read yet: this is its course.
                    self._courses[line_id] = points - shift
            self.last_match, self.last_offset = report, (float(shift[0]), float(shift[1]))
            return picks

    def record(self, line_id, entry):
        """A good reading of a line. The first one is kept."""
        with self._lock:
            if line_id not in self._courses:
                raise KeyError(f'Line {line_id} is not known on this board')
            if line_id not in self._good:
                self._entries[line_id] = entry
                self._good.add(line_id)

    def fail(self, line_id, entry):
        """A try that failed. Shown until a later photo gives a good reading."""
        with self._lock:
            if line_id not in self._courses:
                raise KeyError(f'Line {line_id} is not known on this board')
            if line_id not in self._good:
                self._entries[line_id] = entry
