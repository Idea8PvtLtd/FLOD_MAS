"""capture_storage.background_save: same files, written in order on another thread."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from inspection import storage
from inspection.storage import CaptureStore

SETTINGS = {'max_sets': 5, 'jpeg_quality': 82, 'preview_max_edge': 64}


def write_set(store):
    photo = np.full((40, 60, 3), 90, np.uint8)
    metadata = {'n': 1}
    path = store.begin(photo, photo, metadata)
    metadata['n'] = 2                      # changed after handing over
    line = path / 'line_1'
    line.mkdir()
    mask = np.zeros((40, 60), np.uint8)
    store.save_mask(line, mask)
    mask[:] = 255                          # caller reuses its array
    store.save_overlay(line, photo)
    store.save_preview(line, 'overview', photo)
    store.finish(line, {'status': 'complete'})
    store.finish(path, {'status': 'complete'})
    return path


def snapshot(root):
    return {str(p.relative_to(root)).split('/', 1)[1]: p.read_bytes()
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


class BackgroundSaveTests(unittest.TestCase):
    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(CaptureStore(folder, SETTINGS).writer)

    def test_background_files_match_synchronous_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write_set(CaptureStore(a, SETTINGS))
            store = CaptureStore(b, dict(SETTINGS, background_save=True))
            path = write_set(store)
            store.flush()
            self.assertEqual(snapshot(a), snapshot(b))
            self.assertEqual(json.loads((path / 'capture.json').read_text()), {'n': 1})

    def test_result_json_is_written_last(self):
        order = []
        real_image, real_text = storage._write_image, storage._write_text
        def image(path, *args):
            order.append(path.name); real_image(path, *args)
        def text(path, *args):
            order.append(path.name); real_text(path, *args)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(storage, '_write_image', image), patch.object(storage, '_write_text', text):
            store = CaptureStore(folder, dict(SETTINGS, background_save=True))
            write_set(store)
            store.flush()
        self.assertEqual(order[-1], 'result.json')
        self.assertLess(order.index('undistorted.png'), order.index('mask.png'))

    def test_inspection_does_not_wait_for_the_disk(self):
        gate = threading.Event()
        real_image = storage._write_image
        def slow(*args):
            gate.wait(5); real_image(*args)
        with tempfile.TemporaryDirectory() as folder, patch.object(storage, '_write_image', slow):
            store = CaptureStore(folder, dict(SETTINGS, background_save=True))
            path = store.begin(np.zeros((8, 8, 3), np.uint8), None, {})
            self.assertFalse((path / 'original.png').exists())
            gate.set()
            store.flush()
            self.assertTrue((path / 'original.png').exists())


if __name__ == '__main__':
    unittest.main()
