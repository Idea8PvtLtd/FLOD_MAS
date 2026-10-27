"""MAC ONLY: run the automatic inspection headless and report where time goes.

    python debug_mac.py                          # 120 s of the test videos, Core ML
    FLOD_COREML=0 python debug_mac.py            # same, original CPU models
    python debug_mac.py --seconds 60 --video video_testcases/capture_20261013_171634.mp4

The exact objects main_1366.py uses (video_feed's stream, AutoController and
the dashboard's pipeline factory) run without Tk. Frames are submitted on the
preview interval, as update_video_feed does. Every controller event (status,
error, result) is printed with its time, and each model call and loop stage is
timed, so a stall, a failing step or a slow model shows up by name.

Without a calibration file the app refuses to measure. --identity-undistort
(on by default when the file is missing) passes frames through unchanged so
the models and geometry still run; widths are then in pixels only.
"""
import argparse
import collections
import functools
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
T0 = time.monotonic()
STATS = collections.defaultdict(list)
STATS_LOCK = threading.Lock()


def log(text):
    print(f'[{time.monotonic() - T0:7.2f}s] {text}', flush=True)


def timed(name, function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            with STATS_LOCK:
                STATS[name].append((time.perf_counter() - started) * 1000)
    return wrapper


class IdentityUndistorter:
    """TEST ONLY stand-in when no calibration exists: frames pass unchanged."""
    def undistort(self, frame):
        return frame


def instrument():
    """Time every model call and loop stage the controller makes."""
    from auto_trigger import models, bed_markers
    from inspection.local import engine, source_model, glue_model
    models.HandPresence.detect = timed('hand detect (MediaPipe)', models.HandPresence.detect)
    models.FabricGate.classify = timed('fabric classify (incl. preprocessing)', models.FabricGate.classify)
    source_model.SourceMaskOnnx.predict = timed('source mask (UNet + pre/post)', source_model.SourceMaskOnnx.predict)
    glue_model.SegformerOnnx.predict = timed('SegFormer per line', glue_model.SegformerOnnx.predict)
    engine.LiveInspection._inspect_line = timed('one line, end to end', engine.LiveInspection._inspect_line)
    engine.LiveInspection._accepted_lines = timed('line geometry (_accepted_lines)', engine.LiveInspection._accepted_lines)
    bed_markers.BedMarkers.detect_for_display = timed('bed markers for overlay', bed_markers.BedMarkers.detect_for_display)
    bed_markers.BedMarkers.update = timed('bed markers reset check', bed_markers.BedMarkers.update)
    # Which backend each model really loaded, read off the live objects.
    original_init = engine.LiveInspection.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name, model in (('source', self.source), ('glue', self.glue)):
            log(f'{name} model backend: {type(model.session).__name__} {model.session.get_providers()}')
    engine.LiveInspection.__init__ = init


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--video', type=Path, action='append', help='replay only these files')
    parser.add_argument('--identity-undistort', choices=('auto', 'on', 'off'), default='auto')
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # app_config must not see our flags

    import video_feed
    if args.video:
        video_feed.VIDEO_FILES = [str(path.resolve()) for path in args.video]
    from app_config import CONFIG, ACTIVE_CAMERAS, project_path
    import camera_handler
    from inspection.dashboard import AutomaticDashboard
    from auto_trigger.controller import AutoController
    instrument()
    log(f"Core ML {'OFF (FLOD_COREML=0)' if os.environ.get('FLOD_COREML') == '0' else 'on when packages exist'}")

    spec = ACTIVE_CAMERAS[0]
    stream = video_feed.VideoFileStream(spec['index'])
    stream._capture = timed('video decode thread (per loop)', stream._capture)
    undistorter = None
    try:
        undistorter = camera_handler.ImageUndistorter(project_path(spec['calibration_file']))
        log('calibration loaded')
    except Exception as error:
        log(f'NO CALIBRATION: {error}')
    if args.identity_undistort == 'on' or (args.identity_undistort == 'auto' and undistorter is None):
        undistorter = IdentityUndistorter()
        log('using identity undistortion (TEST ONLY): pixel widths, no millimetres')

    # The dashboard's own factory, reached without building any Tk widgets.
    holder = type('Holder', (), {})()
    holder.root = type('Root', (), {'after': lambda *a: None})()
    holder._show_startup_loading = lambda: None
    holder._poll_automatic = None  # this loop polls the controller instead
    AutomaticDashboard._init_automatic.__get__(holder)()
    controller = holder.auto_controller
    profile = {'size': CONFIG['inspection']['default_size'],
               'width': float(CONFIG['inspection']['strip_width_mm']),
               'tolerance': float(CONFIG['inspection']['strip_width_tolerance_mm']),
               'end_exclusion_percent': float(CONFIG['inspection'].get('end_exclusion_percent', 5.0))}

    interval = CONFIG['preview']['interval_ms'] / 1000
    submitted, last_sequence, trigger_at, reset_due = 0, None, None, None
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and not controller.stopped.is_set():
        raw, sequence = stream.read_snapshot()
        if raw is not None and sequence != last_sequence:
            controller.submit(raw, sequence, undistorter, profile)
            submitted += 1
            last_sequence = sequence
        for kind, _, payload in controller.poll():
            if kind == 'status':
                if payload == 'Checking fabric…':
                    trigger_at = time.monotonic()
                log(f'status: {payload}')
            elif kind == 'result':
                m = payload['measurement']
                late = f' ({time.monotonic() - trigger_at:.2f}s after trigger)' if trigger_at else ''
                log(f"RESULT line {m.get('selected_line')} of {m.get('line_count')}: {m.get('status')}"
                    f"{late}; warnings: {payload['warnings']}")
            else:
                if kind == 'reset_ready':
                    reset_due = time.monotonic() + 2.0  # the dashboard shows results 2 s more
                log(f'{kind.upper()}: {payload}')
        if reset_due is not None and time.monotonic() >= reset_due:
            reset_due = None
            controller.finish_reset()
            log('results dismissed (finish_reset), back to waiting for a hand')
        if stream.error:
            log(f'VIDEO ERROR: {stream.error}')
            break
        time.sleep(interval)
    controller.stop()
    stream.release()

    elapsed = time.monotonic() - T0
    print(f'\n=== {elapsed:.0f}s, {submitted} frames submitted ({submitted / elapsed:.1f}/s); '
          f'video produced {stream.sequence} frames ===')
    for name, values in sorted(STATS.items()):
        if name.startswith('video decode'):
            continue
        values = np.asarray(values)
        print(f'{name:40s} n={len(values):5d}  median {np.median(values):8.1f} ms  '
              f'p95 {np.percentile(values, 95):8.1f} ms  max {values.max():8.1f} ms')
    hand = STATS.get('hand detect (MediaPipe)', [])
    print(f'hand checks per second: {len(hand) / elapsed:.1f} '
          '(each needs a fresh frame; the controller drops frames while busy)')


if __name__ == '__main__':
    main()
