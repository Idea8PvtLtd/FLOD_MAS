"""Prepare a comparison run: still moments of the recordings, playback frames, every method.

    python -m tools.method_compare.prepare [--run DIR] [--methods A,B,...] [--videos ...]
                                           [--only-methods] [--skip-methods]

Run with the research venv (tools/jetson/run_method_compare.sh does): it needs the
app's packages plus PyTorch and CuPy for some methods. Each method runs in its own
process, one after another, so their GPU memory never adds up.

The still moments are the photos the app would take in scene mode: the same scene
signals, the same SceneMonitor settings and the same bare-bed rule (two or more
ID-0 markers in view). The hand check is not run here (MediaPipe is not in the
research venv); the moments of the two recordings were checked hand-free by eye.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2

from . import BY_ID, FRAME_WIDTH, METHODS, ROOT, RUNS
from .results import save


class MomentPicker:
    """SceneMonitor fed the way the app's controller feeds it, one frame at a time."""

    def __init__(self, monitor, bare_markers=2):
        self.monitor = monitor
        self.bare_markers = bare_markers
        self.board = 1

    def step(self, now, small, motion, id0_markers):
        """'capture', 'cycle_end' or None for one frame (``now`` in seconds)."""
        event = self.monitor.update(motion, id0_markers >= self.bare_markers, small, now)
        if event == 'capture':
            self.monitor.accept()
        elif event == 'cycle_end':
            self.board += 1
        return event


def default_videos(config):
    """The recordings the app plays on this PC (video_source.files), else data/procedure_videos."""
    try:
        import video_feed
        found = video_feed.resolve(video_feed.settings(config)['files'])
        if found:
            return [Path(path) for path in found]
    except Exception:
        pass
    return sorted((ROOT / 'data' / 'procedure_videos').glob('*.mp4'))


def undistorter(config):
    from app_config import project_path
    from measure.makeUndistored import ImageUndistorter
    path = Path(project_path(config['cameras'][0]['calibration_file']))
    return ImageUndistorter(str(path)) if path.is_file() else None


def find_moments(run, videos, config, frames=True):
    """Play each recording through the scene monitor; save its still moments and playback frames."""
    from auto_trigger.scene_monitor import SceneMonitor, scene_signals
    from tools.scene_db import count_markers, marker_detector, video_frames
    settings = config['auto_trigger']
    fix = undistorter(config)
    detector = marker_detector()
    (run / 'photos').mkdir(parents=True, exist_ok=True)
    moments, listing = [], []
    for video in videos:
        picker = MomentPicker(SceneMonitor.from_settings(settings),
                              settings.get('scene', {}).get('bare_bed_markers', 2))
        stem = Path(video).stem
        folder = run / 'frames' / stem
        folder.mkdir(parents=True, exist_ok=True)
        previous, count, fps = None, 0, 10.0
        started = time.perf_counter()
        for index, fps, frame in video_frames(video):
            small, motion = scene_signals(frame, previous)
            previous = small
            _, id0 = count_markers(detector, frame)
            event = picker.step(index / fps, small, motion, id0)
            if frames:
                height, width = frame.shape[:2]
                copy = cv2.resize(frame, (FRAME_WIDTH, round(height * FRAME_WIDTH / width)),
                                  interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(folder / f'{index:05d}.jpg'), copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if event == 'capture':
                photo = fix.undistort(frame) if fix is not None else frame
                name = f'{stem}_{index:05d}.png'
                cv2.imwrite(str(run / 'photos' / name), photo)
                moments.append({'video': stem, 'frame': index, 'time_s': round(index / fps, 2),
                                'board': f'{stem}#{picker.board}', 'photo': name})
                print(f'{stem} frame {index}: still moment ({len(moments)})', flush=True)
            elif event == 'cycle_end':
                print(f'{stem} frame {index}: board left the bed', flush=True)
            count = index + 1
        listing.append({'name': stem, 'file': str(video), 'fps': fps, 'frames': count,
                        'size': [int(frame.shape[1]), int(frame.shape[0])] if count else None})
        print(f'{stem}: {count} frames in {time.perf_counter() - started:.0f} s', flush=True)
    save(run / 'moments.json', {'videos': listing, 'moments': moments, 'undistorted': fix is not None,
                                'hand_check': 'not run (MediaPipe is not in the research venv)',
                                'monitor': settings.get('scene', {})})
    return moments


def describe_system():
    info = {'when': datetime.datetime.now().isoformat(timespec='seconds'), 'host': platform.node(),
            'python': platform.python_version()}
    for name, command in (('app_commit', ['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD']),
                          ('lab_commit', ['git', '-C', str(Path.home() / 'flod_lab'), 'rev-parse', '--short', 'HEAD']),
                          ('power_mode', ['nvpmodel', '-q'])):
        try:
            info[name] = subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            info[name] = None
    return info


def run_methods(run, chosen):
    for method in chosen:
        module = 'methods_app' if method['family'] == 'app' else 'methods_v5'
        print(f'=== method {method["id"]}: {method["label"]}', flush=True)
        started = time.perf_counter()
        done = subprocess.run([sys.executable, '-m', f'tools.method_compare.{module}',
                               '--run', str(run), '--method', method['id']], cwd=str(ROOT))
        state = 'done' if done.returncode == 0 else f'failed ({done.returncode})'
        print(f'=== method {method["id"]} {state} in {time.perf_counter() - started:.0f} s', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', help='run folder (default: a new one under ~/flod_mas_data/method_compare)')
    parser.add_argument('--methods', default=','.join(m['id'] for m in METHODS))
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--only-methods', action='store_true', help='keep the moments of an existing run')
    parser.add_argument('--skip-methods', action='store_true')
    parser.add_argument('--no-frames', action='store_true', help='do not save the playback frames')
    args = parser.parse_args(argv)
    from app_config import CONFIG
    run = Path(args.run) if args.run else RUNS / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True, exist_ok=True)
    chosen = [BY_ID[name.strip()] for name in args.methods.split(',') if name.strip()]
    if not args.only_methods:
        videos = [Path(v) for v in args.videos] if args.videos else default_videos(CONFIG)
        if not videos:
            raise SystemExit('No recordings found')
        find_moments(run, videos, CONFIG, frames=not args.no_frames)
    elif not (run / 'moments.json').is_file():
        raise SystemExit(f'{run} has no moments.json')
    info = describe_system()
    info['methods'] = [m['id'] for m in chosen]
    save(run / 'run.json', info)
    if not args.skip_methods:
        run_methods(run, chosen)
    print(f'Run ready: {run}')
    return 0


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    raise SystemExit(main())
