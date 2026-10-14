"""local_inspection.line_workers: lines measured in parallel, results in order."""
import random
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from inspection.local.engine import LiveInspection


def inspector(workers, fail=()):
    engine = LiveInspection.__new__(LiveInspection)
    engine.offset, engine.smooth_normal_rows, engine.min_line_length_ratio = 100., 0, 0.
    engine.line_workers = workers
    engine.source = type('S', (), {'predict': staticmethod(lambda photo: np.zeros((10, 10), np.uint8))})()
    engine.active = 0
    engine.peak = 0
    lock = threading.Lock()

    def inspect(photo, source, line, index, count, started, progress):
        with lock:
            engine.active += 1
            engine.peak = max(engine.peak, engine.active)
        time.sleep(random.uniform(.01, .05))
        with lock:
            engine.active -= 1
        if index in fail:
            raise ValueError(f'line {index} failed')
        return {'selected_line': index, 'thread': threading.current_thread().name}
    engine._inspect_line = inspect
    engine._enhanced_photo = lambda photo: photo
    return engine


def lines(count):
    return [{'original_xy': [[x * 10, 0], [x * 10, 5]]} for x in range(count)]


@patch('inspection.local.engine.extract_lines', lambda *a, **k: [])
@patch('inspection.local.engine.filter_wave_lines', lambda *a, **k: (None, None, None))
class ParallelLinesTests(unittest.TestCase):
    def run_lines(self, engine, count=5, **kwargs):
        with patch('inspection.local.engine.offset_lines', lambda *a, **k: (None, None, lines(count))):
            return list(engine.run_lines(np.zeros((10, 10, 3), np.uint8), **kwargs))

    def test_results_keep_left_to_right_order(self):
        results = self.run_lines(inspector(3))
        self.assertEqual([r['selected_line'] for r in results], [1, 2, 3, 4, 5])
        self.assertTrue(all(r['thread'].startswith('inspect-line') for r in results))

    def test_never_more_lines_at_once_than_workers(self):
        engine = inspector(2)
        self.run_lines(engine, count=6)
        self.assertLessEqual(engine.peak, 2)

    def test_one_worker_runs_in_the_calling_thread(self):
        results = self.run_lines(inspector(1))
        self.assertEqual({r['thread'] for r in results}, {threading.current_thread().name})

    def test_errors_reported_in_order_and_others_still_measured(self):
        errors = []
        results = self.run_lines(inspector(3, fail={2, 4}),
                                 on_error=lambda index, error: errors.append(index))
        self.assertEqual([r['selected_line'] for r in results], [1, 3, 5])
        self.assertEqual(errors, [2, 4])

    def test_error_without_handler_is_raised(self):
        with self.assertRaises(ValueError):
            self.run_lines(inspector(3, fail={3}))

    def test_invalid_worker_count_is_refused(self):
        with patch('inspection.local.source_model.SourceMaskOnnx'), \
                patch('inspection.local.glue_model.SegformerOnnx'), self.assertRaises(ValueError):
            LiveInspection('a', 'b', line_workers=0)


if __name__ == '__main__':
    unittest.main()
