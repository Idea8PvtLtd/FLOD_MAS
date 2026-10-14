"""Proposes the label files: python -m tools.scene_db.autolabel [--segments] [--stills] [--no-lines]

segments.csv: runs of frames of each recording, from its trace
  at rest, 2 or more ID-0 markers   bare_bed
  moving, hand model says hand       hand
  moving, no hand                    moving
  at rest, board in place            board_cloth or board_empty
stills.csv: every still photo in the data folders, by the same picture tests.

Cloth on a board is decided by the glue line finder (LiveInspection.find_lines) on the
middle frame of the segment: a line found means a cloth piece is there, and the number
of lines is proposed as the number of pieces. That is the test the inspection itself
will act on, and it does not depend on the cloth's colour: a dark cloth on the dark bed
cloth and a pale cloth on pale cardboard both defeat a plain colour threshold. A colour
measure (the part of the board that is not cardboard-coloured) is kept as a cross-check
and gives the cloth's colour name; where the two disagree the note says so.
Orientation comes from the lines: a line that runs across the photo more than down it
belongs to a piece laid sideways; without lines, from the shape of the largest piece of
the source mask.

Everything written here has reviewed=0. Rows already reviewed (reviewed=1) in an
existing file are kept as they are unless --overwrite is given.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from . import (BARE_MARKERS, ROOT, SEGMENT_COLUMNS, SEGMENTS, STILL_COLUMNS, STILLS,
               all_videos, count_markers, load_config, marker_detector, read_rows, read_trace,
               relative, trace_path, video_frames, write_rows)

STILL_MOTION = 0.01      # at rest at or below this share of changed pixels
MIN_CALM_FRAMES = 5      # a shorter rest inside movement is part of the movement
BLIP_FRAMES = 2          # a movement this short ...
BLIP_MOTION = 0.03       # ... and this weak, inside a rest, is a shake, not a movement
HAND_GAP = 3             # the hand model may miss this many frames inside a hand run
CLOTH_SHARE = 0.08       # colour cross-check: this share of the board is not cardboard
STILL_FOLDERS = ('data/non fabric', 'data/fabric', 'data/m1/captures', 'data/m4/captures',
                 'data/2-20261013T122546Z-1-001/2')


def runs(values):
    """[(start, end, value), ...] for the runs of equal neighbours."""
    found, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            found.append((start, index - 1, values[start]))
            start = index
    return found


def rest_flags(trace):
    """True for each frame that belongs to a period of rest, short runs smoothed away."""
    calm = [row['motion'] <= STILL_MOTION for row in trace]
    # Weak, short movement between two rests: a shake of the camera or the table.
    for start, end, value in runs(calm):
        if (not value and end - start + 1 <= BLIP_FRAMES and start > 0 and end < len(calm) - 1
                and max(row['motion'] for row in trace[start:end + 1]) < BLIP_MOTION):
            calm[start:end + 1] = [True] * (end - start + 1)
    # A rest too short to be one: the hand paused for a moment.
    for start, end, value in runs(calm):
        # (At either end of the recording a rest is cut short by the recording itself.)
        if value and end - start + 1 < MIN_CALM_FRAMES and start > 0 and end < len(calm) - 1:
            calm[start:end + 1] = [False] * (end - start + 1)
    return calm


def hand_flags(trace):
    """The hand column with short gaps inside a hand run filled."""
    hand = [bool(row['hand']) for row in trace]
    for start, end, value in runs(hand):
        if not value and end - start + 1 <= HAND_GAP and start > 0 and end < len(hand) - 1:
            hand[start:end + 1] = [True] * (end - start + 1)
    return hand


def frame_stages(trace):
    """A provisional stage per frame: bare_bed, board (to be decided), hand, moving."""
    calm, hand = rest_flags(trace), hand_flags(trace)
    stages = []
    for index, row in enumerate(trace):
        if calm[index]:
            stages.append('bare_bed' if row['markers_id0'] >= BARE_MARKERS else 'board')
        else:
            stages.append('hand' if hand[index] else 'moving')
    return stages


def cloth_colour(picture):
    """(share of the board that is not cardboard, colour name of that part).

    The board is taken as the hull of everything cardboard-coloured in the picture; what
    lies inside it and is neither cardboard nor yellow tape is counted as cloth.
    """
    small = cv2.resize(picture, (640, round(picture.shape[0] * 640 / picture.shape[1])),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cardboard = ((hue >= 8) & (hue <= 30) & (saturation >= 20) & (saturation <= 80)
                 & (value >= 90)).astype(np.uint8)
    cardboard = cv2.morphologyEx(cardboard, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tape = (hue >= 25) & (hue <= 40) & (saturation > 80)
    if cardboard.mean() < 0.03:
        # Hardly any cardboard in sight: a bare bed, or a board covered edge to edge. A
        # covered board shows as a large pale area on the dark bed (a board covered with
        # dark cloth cannot be told from the bed this way and counts as no cloth).
        pale = (value >= 110) & ~tape
        pale[:, :20] = pale[:, -20:] = False
        share = float(pale.mean())
        if share < 0.3:
            return 0.0, ''
        return share, colour_name(*np.median(hsv[pale], axis=0))
    hull = cv2.convexHull(cv2.findNonZero(cardboard))
    board = np.zeros_like(cardboard)
    cv2.fillConvexPoly(board, hull, 1)
    cloth = (board == 1) & (cardboard == 0) & ~tape
    share = float(cloth.sum()) / max(1, int(board.sum()))
    if cloth.sum() < 200:
        return share, ''
    return share, colour_name(*np.median(hsv[cloth], axis=0))


def colour_name(hue, saturation, value):
    """A coarse name for a median HSV colour (OpenCV scales: hue 0..180)."""
    if value < 70:
        return 'navy' if 95 <= hue <= 135 and saturation >= 70 else 'black'
    if saturation >= 18 and (hue >= 120 or hue <= 8) and value >= 110:
        return 'pink'
    if value >= 165:
        return 'white'
    return 'grey'


class LineFinder:
    """The glue line finder of the application, built as the dashboard builds it."""

    def __init__(self):
        from inspection.local.engine import LiveInspection
        settings = load_config()['local_inspection']
        self.inspection = LiveInspection(
            ROOT / settings['source_model'], ROOT / settings['glue_model'],
            settings['offset_pixels'], settings['source_width'],
            smooth_normal_rows=settings.get('smooth_normal_rows', 0),
            smooth_logits=settings.get('smooth_logits', False),
            min_line_length_ratio=settings.get('min_line_length_ratio', 0.))

    def lines(self, photo):
        """(number of lines, orientation). Orientation '' when there is nothing to tell."""
        try:
            try:
                source, lines = self.inspection.find_lines(photo, sideways=True)
            except TypeError:       # an engine without the sideways search
                source, lines = self.inspection.find_lines(photo)
        except ValueError:
            return 0, ''
        across = 0
        for line in lines:
            extent = np.ptp(np.asarray(line['original_xy'], dtype=float), axis=0)
            across += extent[0] > extent[1]
        if across == 0:
            return len(lines), 'upright'
        return len(lines), 'sideways' if across == len(lines) else 'mixed'


def describe_board(picture, finder):
    """(stage, cloth_count, colour, orientation, note) for a picture of a board at rest."""
    share, colour = cloth_colour(picture)
    count, orientation = finder.lines(picture) if finder else (0, '')
    note = f'auto: lines {count if finder else "-"}, not cardboard {share * 100:.0f}%'
    if finder:
        cloth = count > 0
        if cloth != (share >= CLOTH_SHARE):
            note += '; line finder and colour measure disagree'
    else:
        cloth = share >= CLOTH_SHARE
    if not cloth:
        return 'board_empty', 0, '', '', note
    return 'board_cloth', count, colour, orientation, note


def propose_segments(video, finder):
    """Label rows for one recording from its trace and the middle frames of its rests."""
    trace = read_trace(trace_path(video))
    found = runs(frame_stages(trace))
    middles = {(start + end) // 2 for start, end, stage in found if stage == 'board'}
    pictures = {}
    if middles:
        for index, _, frame in video_frames(video):
            if index in middles:
                pictures[index] = describe_board(frame, finder)
            if index >= max(middles):
                break
    rows, count = [], 0
    for start, end, stage in found:
        row = {'source': relative(video), 'start_frame': start, 'end_frame': end, 'stage': stage,
               'cloth_count': 0, 'colour': '', 'orientation': '', 'reviewed': 0, 'note': ''}
        if stage == 'board':
            (row['stage'], count, row['colour'], row['orientation'],
             row['note']) = pictures[(start + end) // 2]
        elif stage == 'bare_bed':
            count = 0
        # During movement the count is what lay on the board before it began.
        row['cloth_count'] = count
        rows.append(row)
    return rows


def still_paths():
    paths = []
    for folder in STILL_FOLDERS:
        paths += sorted(path for path in (ROOT / folder).iterdir()
                        if path.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    paths += sorted((ROOT / 'data' / 'real_caps').glob('*/original.png'))
    return paths


class StillHands:
    """The hand model on single photos: the same model file and thresholds as
    HandPresence, but in image mode, because unrelated stills must not be tracked from
    one to the next as the frames of a video are."""

    def __init__(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from auto_trigger.models import MODEL_DIR
        settings = load_config()['auto_trigger']
        self.width = settings['detect_width']
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')),
            running_mode=vision.RunningMode.IMAGE, num_hands=settings['max_hands'],
            min_hand_detection_confidence=settings['hand_confidence'],
            min_hand_presence_confidence=settings['hand_confidence']))

    def present(self, frame):
        import mediapipe as mp
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, round(height * self.width / width))))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        return bool(self.landmarker.detect(image).hand_landmarks)


def propose_stills(finder, with_hand=True):
    detector = marker_detector()
    hands = StillHands() if with_hand else None
    rows, seen = [], {}
    for number, path in enumerate(still_paths()):
        # m1 and m4 hold photos that are also in data/fabric: same name and size, same
        # picture, so the slow tests run once.
        identity = (path.name, path.stat().st_size) if 'real_caps' not in path.parts else path
        if identity not in seen:
            picture = cv2.imread(str(path))
            _, id0 = count_markers(detector, picture)
            if id0 >= BARE_MARKERS:
                described = ('bare_bed', 0, '', '', f'auto: {id0} ID-0 markers')
            else:
                described = describe_board(picture, finder)
                described = described[:4] + (described[4] + f', {id0} ID-0 markers',)
            seen[identity] = described + (int(hands.present(picture)) if hands else '',)
        stage, count, colour, orientation, note, has_hand = seen[identity]
        rows.append({'path': relative(path), 'stage': stage, 'cloth_count': count, 'colour': colour,
                     'orientation': orientation, 'has_hand': has_hand, 'reviewed': 0, 'note': note})
        if number % 50 == 49:
            print(f'  {number + 1} stills')
    return rows


def keep_reviewed(path, key, rows, overwrite):
    """Rows already checked by eye win over a new proposal for the same picture(s)."""
    if overwrite or not Path(path).is_file():
        return rows
    reviewed = [row for row in read_rows(path) if row.get('reviewed') == '1']
    if not reviewed:
        return rows
    if key == 'path':
        by_path = {row['path']: row for row in reviewed}
        return [by_path.get(row['path'], row) for row in rows]
    # Segments: a reviewed source is kept whole (its boundaries were set by hand).
    sources = {row['source'] for row in reviewed}
    kept = [row for row in read_rows(path) if row['source'] in sources]
    return kept + [row for row in rows if row['source'] not in sources]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--segments', action='store_true')
    parser.add_argument('--stills', action='store_true')
    parser.add_argument('--no-lines', action='store_true',
                        help='skip the line finder; cloth by the colour measure alone')
    parser.add_argument('--no-hand', action='store_true', help='stills: skip the hand model')
    parser.add_argument('--overwrite', action='store_true', help='replace reviewed rows too')
    arguments = parser.parse_args()
    both = not arguments.segments and not arguments.stills
    finder = None if arguments.no_lines else LineFinder()
    if arguments.segments or both:
        rows = []
        for video in all_videos():
            rows += propose_segments(video, finder)
            print(f'{video.name}: {sum(row["source"] == relative(video) for row in rows)} segments')
        write_rows(SEGMENTS, SEGMENT_COLUMNS,
                   keep_reviewed(SEGMENTS, 'source', rows, arguments.overwrite))
        print('->', SEGMENTS)
    if arguments.stills or both:
        rows = propose_stills(finder, not arguments.no_hand)
        write_rows(STILLS, STILL_COLUMNS, keep_reviewed(STILLS, 'path', rows, arguments.overwrite))
        print(f'{len(rows)} stills ->', STILLS)


if __name__ == '__main__':
    main()
