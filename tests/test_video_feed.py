"""FLOD_VIDEO_FILES switches the camera to looping recordings; unset, nothing changes."""
import importlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import camera_handler


def load(value):
    env = {} if value is None else {'FLOD_VIDEO_FILES': value}
    with patch.dict(os.environ, env, clear=False):
        if value is None:
            os.environ.pop('FLOD_VIDEO_FILES', None)
        import video_feed
        return importlib.reload(video_feed)


class VideoFeedTests(unittest.TestCase):
    def setUp(self):
        self.original = camera_handler.CameraStream
        self.addCleanup(setattr, camera_handler, 'CameraStream', self.original)

    def test_unset_lists_no_videos(self):
        self.assertEqual(load(None).VIDEO_FILES, [])

    def test_paths_are_split_on_the_platform_separator(self):
        module = load(os.pathsep.join(['a.mp4', '', 'b.mp4']))
        self.assertEqual(module.VIDEO_FILES, ['a.mp4', 'b.mp4'])

    def test_installed_stream_plays_the_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            for value in range(5):
                writer.write(np.full((48, 64, 3), value * 40, np.uint8))
            writer.release()
            module = load(path)
            module.install()
            self.assertIs(camera_handler.CameraStream, module.VideoFileStream)
            with patch.object(camera_handler, 'camera_config', return_value={'rotation': 0}), \
                    patch.object(module.VideoFileStream, 'force_automatic', return_value=None):
                stream = module.VideoFileStream(0)
            try:
                time.sleep(0.3)
                self.assertIsNone(stream.error)
                self.assertEqual(stream.frame.shape, (48, 64, 3))
            finally:
                stream.stopped.set()
                stream.thread.join(2)


if __name__ == '__main__':
    unittest.main()
