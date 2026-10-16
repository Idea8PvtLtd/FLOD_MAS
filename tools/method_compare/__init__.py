"""Every glue-line method on the same still moments of the recordings, to compare by eye.

    tools/jetson/run_method_compare.sh            prepare once, then open the viewer
    python -m tools.method_compare.prepare        still moments, frames, every method
    python -m tools.method_compare.viewer [RUN]   the window: watch, accuracy, speed

A run is a folder (default ~/flod_mas_data/method_compare/<date-time>) holding the
still moments as the app would have photographed them, small copies of every video
frame for playback, and one JSON file per method with its lines and its times.
Nothing here is imported by the app.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / 'flod_mas_data' / 'method_compare'
LAB = Path.home() / 'flod_lab'               # checkout of tooling/pipeline-lab (version 5)
TARGET_SECONDS = 4.0                          # a board photo should be read within this
FRAME_WIDTH = 1024                            # width of the playback copies of the frames

# id, label, family, colour (BGR, for drawing edges), and what the method runner needs.
METHODS = (
    {'id': 'A', 'label': 'A  App, no ribbon', 'family': 'app', 'colour': (230, 160, 40),
     'ribbon': False, 'cv_threads': 4, 'gpu_distance': False},
    {'id': 'B', 'label': 'B  App + ribbon (today)', 'family': 'app', 'colour': (60, 200, 60),
     'ribbon': True, 'cv_threads': 1, 'gpu_distance': False},
    {'id': 'B-fast', 'label': 'B-fast  B, faster settings', 'family': 'app',
     'colour': (40, 220, 220), 'ribbon': True, 'cv_threads': 4, 'gpu_distance': True,
     'dash': (6, 4)},
    {'id': 'C', 'label': 'C  v5, SAM 2.1 large', 'family': 'v5', 'colour': (220, 60, 220),
     'mode': 'sam2.1-large'},
    {'id': 'D', 'label': 'D  v5, SAM 2.1 tiny', 'family': 'v5', 'colour': (60, 60, 230),
     'mode': 'sam2.1-tiny'},
    {'id': 'E', 'label': 'E  v5, no SAM', 'family': 'v5', 'colour': (200, 200, 200),
     'mode': 'nosam'},
)
BY_ID = {method['id']: method for method in METHODS}
REFERENCE = 'B'          # widths are compared with the app as it runs today


def latest_run(folder=RUNS):
    """The newest run folder that has its moments, or None."""
    runs = sorted(path for path in Path(folder).glob('*') if (path / 'moments.json').is_file())
    return runs[-1] if runs else None
