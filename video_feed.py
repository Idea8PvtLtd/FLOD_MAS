"""TEST ONLY: replay recorded videos in a loop instead of the live camera.

Switched on by the FLOD_VIDEO_FILES environment variable, a list of video paths
separated by os.pathsep (":" on Linux, ";" on Windows). Unset, the camera is used
as always. Ported from the dev-lm branch. Nothing else changes: the dashboard,
hand trigger, fabric check, undistortion and inspection receive these frames
exactly as they would receive camera frames, so the whole system can be
exercised without a camera.
"""
from pathlib import Path
import threading
import os
import time
import cv2
import camera_handler
from camera_handler import CameraStream

ENV_VAR = 'FLOD_VIDEO_FILES'
VIDEO_FILES = [path for path in os.environ.get(ENV_VAR, '').split(os.pathsep) if path]

# None applies the camera's configured rotation (config.json), as a live feed
# would. Set 0, 90, 180 or 270 if the recording is already rotated.
ROTATION = None


class VideoFileStream(CameraStream):
    """CameraStream's interface, fed from VIDEO_FILES in order, looping forever."""

    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.device_lock = threading.Lock()
        self.frame = None
        self.error = None
        self.resolution_warning = None
        self.sequence = 0
        self.stopped = threading.Event()
        spec = camera_handler.camera_config(index)
        self.rotation = spec.get('rotation', 0) if ROTATION is None else ROTATION
        missing = [path for path in VIDEO_FILES if not Path(path).is_file()]
        if missing:
            raise RuntimeError('Test video not found: ' + ', '.join(missing))
        self.cap = None
        self.current = 0
        self._open(self.current)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f'Test video could not be read: {VIDEO_FILES[0]}')
        self.frame = frame
        self.size = self._rotated_size((frame.shape[1], frame.shape[0]))
        self.auto_control = self.force_automatic()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _open(self, position):
        if self.cap is not None:
            self.cap.release()
        path = VIDEO_FILES[position]
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Test video could not be opened: {path}')
        # Play at the recording's own speed; 10 fps matches the camera if unknown.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = 1.0 / fps if fps and fps > 0 else 0.1
        print(f'[video_feed] playing {Path(path).name}')

    def _capture(self):
        try:
            while not self.stopped.is_set():
                started = time.monotonic()
                with self.device_lock:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        # End of this file: continue with the next, wrapping to the first.
                        self.current = (self.current + 1) % len(VIDEO_FILES)
                        self._open(self.current)
                        ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f'Test video could not be read: {VIDEO_FILES[self.current]}')
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                time.sleep(max(0.0, self.interval - (time.monotonic() - started)))
        except Exception as error:
            self.error = str(error)
            with self.lock:
                self.frame = None
        finally:
            if self.cap is not None:
                self.cap.release()


def install():
    """Make every CameraHandler created from now on play the test videos."""
    camera_handler.CameraStream = VideoFileStream
    print('[video_feed] TEST MODE: camera replaced by', len(VIDEO_FILES), 'looping video file(s)')
