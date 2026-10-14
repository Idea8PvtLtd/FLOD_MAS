"""Recorded videos in place of the live camera (``video_source`` in config.json).

With ``video_source.files`` empty, the default, this module is never loaded and the
camera is used as before. With files listed, every camera the application opens
plays them instead: the dashboard, hand trigger, fabric check, lens correction and
inspection receive these frames exactly as they would receive camera frames, so
the whole application can be run without a camera.

Names follow ``video_feed.py`` on the ``dev-lm`` branch (``VideoFileStream``,
``install``); there the files were constants in the module.
"""
import logging
from pathlib import Path
import threading
import time
import cv2
import camera_handler
from camera_handler import CameraStream

log = logging.getLogger('flod.video')
CODE_ROOT = Path(__file__).resolve().parent
ROTATIONS = (0, 90, 180, 270)
DEFAULTS = {'files': [], 'loop': True, 'rotation': 0}


def settings(config):
    """The ``video_source`` section with defaults filled in. Raises ValueError if invalid."""
    section = config.get('video_source', {})
    if not isinstance(section, dict):
        raise ValueError('video_source must be a group of settings')
    merged = dict(DEFAULTS, **section)
    files = merged['files']
    if not isinstance(files, list) or not all(isinstance(name, str) and name.strip() for name in files):
        raise ValueError('video_source.files must be a list of file or folder names')
    if not isinstance(merged['loop'], bool):
        raise ValueError('video_source.loop must be true or false')
    if isinstance(merged['rotation'], bool) or merged['rotation'] not in ROTATIONS:
        raise ValueError('video_source.rotation must be 0, 90, 180 or 270')
    return merged


def resolve(names, root=CODE_ROOT):
    """The video files named, in order. A folder stands for its ``*.mp4`` by name."""
    found = []
    for name in names:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        if path.is_dir():
            inside = sorted(path.glob('*.mp4'))
            if not inside:
                raise RuntimeError(f'No .mp4 file in video folder: {path}')
            found.extend(inside)
        elif path.is_file():
            found.append(path)
        else:
            raise RuntimeError(f'Video not found: {path}')
    if not found:
        raise RuntimeError('video_source.files names no video')
    return found


class VideoFileStream(CameraStream):
    """CameraStream's interface, fed from recorded files in order."""

    files, loop, rotation_setting = (), True, 0

    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.device_lock = threading.Lock()
        self.frame = None
        self.error = None
        self.resolution_warning = None
        self.sequence = 0
        self.finished = False
        self.stopped = threading.Event()
        # The recording's own orientation, not the camera's: a file recorded from
        # the application's picture is already turned.
        self.rotation = self.rotation_setting
        self.cap = None
        self.current = 0
        self._open(self.current)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f'Video could not be read: {self.files[0]}')
        self.frame = frame
        self.size = self._rotated_size((frame.shape[1], frame.shape[0]))
        self.auto_control = self.force_automatic()
        self.thread = threading.Thread(target=self._capture, name='video-feed', daemon=True)
        self.thread.start()

    def _open(self, position):
        if self.cap is not None:
            self.cap.release()
        path = self.files[position]
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f'Video could not be opened: {path}')
        # Play at the recording's own speed; 10 fps matches the camera if unknown.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = 1.0 / fps if fps and fps > 0 else 0.1
        self._opened_at, self._frames_played = time.monotonic(), 0
        log.info('Playing %s (%d of %d): %d x %d, %.1f frames per second, %d frames',
                 path.name, position + 1, len(self.files),
                 int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 1.0 / self.interval, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def _ended(self):
        """Report the file just played; open the next. False when playing is over."""
        elapsed = max(1e-6, time.monotonic() - self._opened_at)
        log.info('End of %s: %d frames in %.1f s (%.1f per second; recorded at %.1f)',
                 self.files[self.current].name, self._frames_played, elapsed,
                 self._frames_played / elapsed, 1.0 / self.interval)
        last = self.current == len(self.files) - 1
        if last and not self.loop:
            return False
        if last:
            log.info('All videos played; starting again from the first')
        self.current = (self.current + 1) % len(self.files)
        self._open(self.current)
        return True

    def _capture(self):
        try:
            while not self.stopped.is_set():
                started = time.monotonic()
                with self.device_lock:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        if not self._ended():
                            # The last picture stays on screen; no new frame is offered.
                            self.finished = True
                            log.info('Video pass finished: the last frame is held')
                            self.stopped.wait()
                            return
                        ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f'Video could not be read: {self.files[self.current]}')
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                self._frames_played += 1
                time.sleep(max(0.0, self.interval - (time.monotonic() - started)))
        except Exception as error:
            log.exception('Video playing stopped')
            self.error = str(error)
            with self.lock:
                self.frame = None
        finally:
            if self.cap is not None:
                self.cap.release()

    def isOpened(self):
        return not self.stopped.is_set() and (self.thread.is_alive() or self.finished)

    # A file has no exposure or white balance to set.
    def set_auto_white_balance(self, enabled):
        return {'ok': False, 'requested': None, 'actual': None}

    def set_auto_exposure(self, enabled):
        return {'ok': False, 'requested': None, 'actual': None}


def install(config):
    """Make every camera opened from now on play the configured videos."""
    chosen = settings(config)
    VideoFileStream.files = tuple(resolve(chosen['files']))
    VideoFileStream.loop = chosen['loop']
    VideoFileStream.rotation_setting = chosen['rotation']
    camera_handler.CameraStream = VideoFileStream
    log.warning('VIDEO MODE: the camera is replaced by %d recorded file(s), %s, turned %d degrees: %s',
                len(VideoFileStream.files), 'looping' if chosen['loop'] else 'played once',
                chosen['rotation'], ', '.join(path.name for path in VideoFileStream.files))
    return VideoFileStream.files
