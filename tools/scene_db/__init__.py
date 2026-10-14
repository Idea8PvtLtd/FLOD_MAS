"""Image database and replay harness for the scene capture mode.

Run the parts from the repository root, e.g. ``python -m tools.scene_db.replay <video>``.

  extract        frames of the recordings as small JPEGs, key frames as full PNGs
  signals_trace  one row of signals per video frame (motion, markers, hand)
  autolabel      proposes segments.csv and stills.csv from the signals
  review         contact sheets for checking the labels by eye
  replay         plays a recording (or its trace) through SceneMonitor and scores it
  sweep          replays the traces over a grid of settings

An image is always named by (source, frame): the source is the path of the video
or still relative to the repository root, so the whole database can be rebuilt
from the recordings and the tracked label files in tests/fixtures/scene.
"""
import base64
import csv
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'tests' / 'fixtures' / 'scene'
DATABASE = ROOT / 'data' / 'scene_db'
VIDEO_FOLDER = ROOT / 'data' / 'procedure_videos'
SEGMENTS = FIXTURES / 'segments.csv'
STILLS = FIXTURES / 'stills.csv'
STAGES = ('bare_bed', 'board_empty', 'board_cloth', 'hand', 'moving')
SEGMENT_COLUMNS = ('source', 'start_frame', 'end_frame', 'stage', 'cloth_count', 'colour',
                   'orientation', 'reviewed', 'note')
STILL_COLUMNS = ('path', 'stage', 'cloth_count', 'colour', 'orientation', 'has_hand',
                 'reviewed', 'note')
TRACE_COLUMNS = ('frame', 'time_s', 'motion', 'markers_all', 'markers_id0', 'hand', 'thumb')
MARKER_WIDTH = 1280      # markers are counted on a picture this wide
BARE_MARKERS = 2         # this many ID-0 markers in view mean the bed is bare
THUMB_SHAPE = (18, 24)   # rows, columns of the tiny picture kept per frame in a trace


def relative(path):
    """A path as stored in the label files: relative to the repository, forward slashes."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def short_name(video):
    """'capture_20261007_223831.mp4' -> '223831': the time stamp names the trace."""
    return Path(video).stem.split('_')[-1]


def trace_path(video):
    return FIXTURES / f'trace_{short_name(video)}.csv'


def find_video(name):
    """A video by path, by file name or by its short time stamp."""
    path = Path(name)
    if path.is_file():
        return path
    for candidate in sorted(VIDEO_FOLDER.glob('*.mp4')):
        if name in (candidate.name, candidate.stem, short_name(candidate)):
            return candidate
    raise SystemExit(f'Video not found: {name}')


def all_videos():
    return sorted(VIDEO_FOLDER.glob('*.mp4'))


def video_frames(video):
    """(index, fps, frame) for every frame, decoded in order.

    Seeking in a 16 megapixel MJPEG file is slow, so everything here reads from
    the start to the end once.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f'Cannot open video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, fps, frame
            index += 1
    finally:
        capture.release()


def marker_detector():
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())


def count_markers(detector, frame, width=MARKER_WIDTH):
    """(all markers, ID-0 markers) seen on a `width` wide copy of the frame."""
    height, full = frame.shape[:2]
    if full > width:
        frame = cv2.resize(frame, (width, max(1, round(height * width / full))),
                           interpolation=cv2.INTER_AREA)
    _, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return 0, 0
    return int(len(ids)), int((ids.flatten() == 0).sum())


def own_signals(frame, previous=None, width=320, level=12):
    """Stand-in for auto_trigger.scene_monitor.scene_signals, same contract.

    Used only while that module does not exist yet.
    """
    step = max(1, frame.shape[1] // width)
    small = frame[::step, ::step]
    if small.ndim == 3:
        small = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_BGR2GRAY)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    if previous is None or previous.shape != small.shape:
        return small, 0.0
    return small, float((cv2.absdiff(small, previous) > level).mean())


def signal_function():
    """scene_signals of the scene monitor, or the stand-in while it is missing."""
    try:
        from auto_trigger.scene_monitor import scene_signals
        return scene_signals, 'auto_trigger.scene_monitor.scene_signals'
    except ImportError:
        return own_signals, 'tools.scene_db.own_signals (stand-in)'


def thumbnail(small):
    """The tiny copy of a `small` picture that a trace keeps for every frame.

    Replaying from a trace has no pictures, but the monitor's question "has the
    scene changed since the last key photo" needs two pictures. Averaging `small`
    down to 24 x 18 cells keeps enough of it: a cloth piece covers many cells,
    while noise and a pixel of camera shake average away.
    """
    rows, columns = THUMB_SHAPE
    return cv2.resize(small, (columns, rows), interpolation=cv2.INTER_AREA)


def encode_thumb(thumb):
    return base64.b64encode(np.ascontiguousarray(thumb, np.uint8).tobytes()).decode('ascii')


def decode_thumb(text):
    if not text:
        return None
    return np.frombuffer(base64.b64decode(text), np.uint8).reshape(THUMB_SHAPE)


def read_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, columns, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, columns, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path):
    """Trace rows with numbers as numbers; `hand` is None where it was not computed."""
    rows = []
    for row in read_rows(path):
        rows.append({'frame': int(row['frame']), 'time_s': float(row['time_s']),
                     'motion': float(row['motion']), 'markers_all': int(row['markers_all']),
                     'markers_id0': int(row['markers_id0']),
                     'hand': int(row['hand']) if row.get('hand') not in (None, '') else None,
                     'thumb': row.get('thumb') or ''})
    return rows


def read_segments(source=None, path=SEGMENTS):
    """Label runs, numbers as numbers, optionally for one source only."""
    rows = []
    for row in read_rows(path):
        if source is not None and row['source'] != source:
            continue
        row['start_frame'], row['end_frame'] = int(row['start_frame']), int(row['end_frame'])
        row['cloth_count'] = int(row['cloth_count'] or 0)
        row['reviewed'] = int(row['reviewed'] or 0)
        rows.append(row)
    return sorted(rows, key=lambda row: (row['source'], row['start_frame']))


def segment_of(segments, frame):
    for segment in segments:
        if segment['start_frame'] <= frame <= segment['end_frame']:
            return segment
    return None


def source_for_trace(path):
    """The video a trace file belongs to, as written in segments.csv."""
    stamp = Path(path).stem.split('_')[-1]
    for video in all_videos():
        if short_name(video) == stamp:
            return relative(video)
    # Without the recordings (a fresh clone) the labels still name the source.
    if SEGMENTS.is_file():
        for row in read_rows(SEGMENTS):
            if Path(row['source']).stem.endswith(stamp):
                return row['source']
    return f'data/procedure_videos/capture_20261007_{stamp}.mp4'


def load_config():
    return json.loads((ROOT / 'config.json').read_text())
