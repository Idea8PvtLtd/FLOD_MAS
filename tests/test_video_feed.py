"""Recorded videos in place of the camera (video_source) and the optional log (logging)."""
import copy
import json
import logging
from pathlib import Path
import tempfile
import time
import unittest
import cv2
import numpy as np
import app_logging
import camera_handler
from config_validation import validate_config

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())


def write_video(path, values, size=(64, 48)):
    """A short recording whose frames are plain pictures of the given brightness."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 50, size)
    for value in values:
        frame = np.full((size[1], size[0], 3), value, np.uint8)
        frame[:8, :8] = 255          # a corner mark, to see a rotation
        writer.write(frame)
    writer.release()


def stop(stream):
    """Stop playing and wait until the file is closed.

    release() only asks the playing thread to stop; it closes the file a moment later.
    Windows cannot delete a file that is still open, so the temporary folder can only
    be removed once the thread has finished.
    """
    stream.release()
    stream.thread.join(timeout=5)


def close_log_files():
    logger = logging.getLogger(app_logging.ROOT)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()


def brightness(frame):
    return int(round(float(frame[20:40, 20:60].mean()) / 10) * 10)


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        import video_feed
        self.video_feed = video_feed
        self.original = camera_handler.CameraStream
        self.folder = tempfile.TemporaryDirectory()
        self.first = Path(self.folder.name) / 'a_first.avi'
        self.second = Path(self.folder.name) / 'b_second.avi'
        write_video(self.first, [40, 60, 80])
        write_video(self.second, [120, 140])

    def tearDown(self):
        camera_handler.CameraStream = self.original
        self.folder.cleanup()

    def play(self, loop, rotation=0, seconds=0.6):
        config = {'video_source': {'files': [str(self.first), str(self.second)], 'loop': loop, 'rotation': rotation}}
        self.video_feed.install(config)
        stream = camera_handler.CameraStream(0)
        seen, last = [], None
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            frame, sequence = stream.read_snapshot()
            if frame is not None and sequence != last:
                last = sequence
                seen.append(brightness(frame))
            time.sleep(0.002)
        return stream, seen

    def test_by_default_the_camera_is_untouched(self):
        self.assertEqual(CONFIG['video_source']['files'], [])
        self.assertIs(camera_handler.CameraStream, self.original)
        self.assertNotEqual(camera_handler.CameraStream.__name__, 'VideoFileStream')

    def test_files_play_in_order_and_the_last_frame_is_held_when_not_looping(self):
        stream, seen = self.play(loop=False)
        try:
            self.assertEqual(sorted(set(seen)), sorted(set(seen), key=seen.index))    # never goes back
            self.assertEqual(seen[-1], 140)
            self.assertTrue(set(seen) <= {40, 60, 80, 120, 140})
            self.assertTrue(stream.finished)
            self.assertTrue(stream.isOpened())
            frame, sequence = stream.read_snapshot()
            time.sleep(0.05)
            self.assertEqual(stream.read_snapshot()[1], sequence)                    # no new frame is offered
            self.assertEqual(brightness(frame), 140)
        finally:
            stop(stream)

    def test_looping_starts_again_from_the_first_file(self):
        stream, seen = self.play(loop=True)
        try:
            self.assertFalse(stream.finished)
            self.assertIn(140, seen)
            self.assertLess(seen[seen.index(140):].count(140), len(seen[seen.index(140):]))   # something follows the last file
            self.assertIn(seen[-1], {40, 60, 80, 120, 140})
        finally:
            stop(stream)

    def test_the_recordings_own_rotation_is_applied_not_the_cameras(self):
        stream, _ = self.play(loop=True, rotation=180, seconds=0.1)
        try:
            ok, frame = stream.read()
            self.assertTrue(ok)
            self.assertEqual(stream.rotation, 180)
            self.assertGreater(frame[-8:, -8:].mean(), 200)      # the corner mark is now bottom right
            self.assertLess(frame[:8, :8].mean(), 200)
        finally:
            stop(stream)

    def test_a_folder_stands_for_its_videos_by_name(self):
        for name in ('b.mp4', 'a.mp4'):
            (Path(self.folder.name) / name).write_bytes(b'')
        found = self.video_feed.resolve([self.folder.name])
        self.assertEqual([path.name for path in found], ['a.mp4', 'b.mp4'])
        with self.assertRaises(RuntimeError):
            self.video_feed.resolve([str(Path(self.folder.name) / 'missing.mp4')])

    def test_the_settings_are_validated(self):
        for key, value in (('files', 'video.mp4'), ('files', ['']), ('loop', 'yes'), ('rotation', 45), ('rotation', True)):
            config = copy.deepcopy(CONFIG)
            config['video_source'][key] = value
            with self.assertRaises(ValueError, msg=f'{key}={value!r}'):
                validate_config(config)
        config = copy.deepcopy(CONFIG)
        config['video_source'].update(files=['data/procedure_videos'], loop=False, rotation=180)
        validate_config(config)
        del config['video_source']
        validate_config(config)            # an older config.json without the section still loads


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        close_log_files()

    def test_by_default_nothing_is_written(self):
        self.assertIs(CONFIG['logging']['enabled'], False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'app.log'
            self.assertIsNone(app_logging.setup(CONFIG, path))
            logging.getLogger('flod.test').error('not written')
            self.assertFalse(path.exists())

    def test_when_enabled_messages_reach_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'app.log'
            config = {'logging': {'enabled': True, 'level': 'INFO'}}
            import sys, threading
            hooks = sys.excepthook, threading.excepthook
            try:
                self.assertEqual(app_logging.setup(config, path), path)
                logging.getLogger('flod.test').info('line %d measured', 3)
                logging.getLogger('flod.test').debug('too detailed')
            finally:
                sys.excepthook, threading.excepthook = hooks
            for handler in logging.getLogger(app_logging.ROOT).handlers:
                handler.flush()
            # Closed before the folder is removed: Windows cannot delete an open file.
            close_log_files()
            text = path.read_text(encoding='utf-8')
            self.assertIn('line 3 measured', text)
            self.assertNotIn('too detailed', text)

    def test_the_settings_are_validated(self):
        for key, value in (('enabled', 'yes'), ('level', 'LOUD'), ('max_bytes', 10), ('backups', -1), ('backups', 1.5)):
            config = copy.deepcopy(CONFIG)
            config['logging'][key] = value
            with self.assertRaises(ValueError, msg=f'{key}={value!r}'):
                validate_config(config)
        config = copy.deepcopy(CONFIG)
        del config['logging']
        validate_config(config)


if __name__ == '__main__':
    unittest.main()
