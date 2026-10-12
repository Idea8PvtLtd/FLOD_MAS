"""Extra grading checks: off by default, and what each one catches when it is set."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from config_validation import validate_config
from inspection.line_measurement import (CHECKS, grading_checks, measurement_record,
                                         with_distance_ratio)
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from plane_scale import PlaneScale
import tempfile
from unittest.mock import patch
from test_local_inspection import synthetic_inspection

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
SCALE = PlaneScale(np.diag([.1, .1, 1]), .1)        # 10 px per mm: the 30 px strip is 3 mm


def line_with(change_px=0, first=60, count=6, invalid=()):
    """The synthetic line with ``count`` neighbouring samples made wider or narrower."""
    _, _, result = synthetic_inspection()
    result = copy.deepcopy(result)
    samples = result['components'][0]['samples']
    for index in range(first, first + count):
        sample = samples[index]
        # Move the right edge point outwards along the measured cross-section.
        across = np.array([sample['right_x'] - sample['left_x'], sample['right_y'] - sample['left_y']])
        across /= np.linalg.norm(across)
        sample['right_x'] += change_px * across[0]
        sample['right_y'] += change_px * across[1]
    for index in invalid:
        samples[index]['valid'] = False
    return result


def grade(result, **checks):
    return measurement_record(result, SCALE, 3, 1, 5, checks=checks or None)


class DefaultTests(unittest.TestCase):
    def test_no_checks_are_set_in_the_shipped_configuration(self):
        self.assertEqual(grading_checks(CONFIG['inspection']), CHECKS)
        self.assertTrue(all(value == 0 for value in CHECKS.values()))

    def test_without_checks_a_short_wide_place_still_passes(self):
        # 3 mm of the line is 1.5 mm too wide, beyond the 1 mm tolerance, but the
        # 7 mm segment it lies in still averages inside it. This is the old result.
        record = grade(line_with(15))
        self.assertEqual(record['status'], 'PASS')
        self.assertEqual(record['local_defects'], [])
        self.assertEqual(record['fail_reasons'], [])
        widest = max(s['width_mm'] for s in record['components'][0]['samples'] if s['width_mm'])
        self.assertGreater(widest, 4.0)

    def test_checks_given_as_none_or_zero_grade_alike(self):
        result = line_with(15)
        plain = measurement_record(result, SCALE, 3, 1, 5)
        zero = measurement_record(result, SCALE, 3, 1, 5, checks=dict(CHECKS))
        self.assertEqual([s['within_tolerance'] for s in plain['segments']],
                         [s['within_tolerance'] for s in zero['segments']])
        self.assertEqual(plain['status'], zero['status'])


class LocalDefectTests(unittest.TestCase):
    def test_a_wide_place_longer_than_the_limit_fails_the_line(self):
        record = grade(line_with(15), local_defect_length_mm=2)
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(len(record['local_defects']), 1)
        defect = record['local_defects'][0]
        self.assertEqual(defect['kind'], 'wide')
        self.assertAlmostEqual(defect['length_mm'], 3.0, delta=.3)
        self.assertAlmostEqual(defect['extreme_width_mm'], 4.5, delta=.2)
        flagged = [s['segment'] for s in record['segments'] if s['local_defect']]
        self.assertEqual(flagged, defect['segments'])
        self.assertTrue(all(s['within_tolerance'] is False for s in record['segments'] if s['local_defect']))
        self.assertTrue(all(s['within_tolerance'] for s in record['segments'] if not s['local_defect']))
        self.assertIn('outside tolerance', record['fail_reasons'][0])

    def test_a_narrow_place_is_found_too(self):
        record = grade(line_with(-15), local_defect_length_mm=2)
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(record['local_defects'][0]['kind'], 'narrow')
        self.assertAlmostEqual(record['local_defects'][0]['extreme_width_mm'], 1.5, delta=.2)

    def test_a_place_shorter_than_the_limit_is_left_alone(self):
        record = grade(line_with(15, count=2), local_defect_length_mm=2)     # 1 mm long
        self.assertEqual(record['status'], 'PASS')
        self.assertEqual(record['local_defects'], [])

    def test_a_change_inside_the_tolerance_is_never_a_defect(self):
        record = grade(line_with(8, count=20), local_defect_length_mm=2)     # +0.8 mm
        self.assertEqual(record['status'], 'PASS')

    def test_an_unmeasured_sample_ends_a_run(self):
        # Six wide samples with the middle two unmeasured are two runs of 1 mm.
        record = grade(line_with(15, invalid=(62, 63)), local_defect_length_mm=2)
        self.assertEqual(record['local_defects'], [])

    def test_the_ends_left_out_of_grading_are_not_searched(self):
        record = grade(line_with(15, first=0, count=6), local_defect_length_mm=2)
        self.assertEqual(record['local_defects'], [])

    def test_the_developer_ratio_regrades_the_defect(self):
        record = grade(line_with(15), local_defect_length_mm=2)
        same = with_distance_ratio(record, record['distance_ratio'])
        self.assertEqual(same['status'], 'FAIL')
        self.assertEqual(len(same['local_defects']), 1)
        # Scaled down by 20 %, the 4.5 mm place reads 3.6 mm: inside 3 +- 1.
        smaller = with_distance_ratio(record, record['distance_ratio'] * .8)
        self.assertEqual(smaller['local_defects'], [])
        self.assertEqual(smaller['status'], 'PASS')
        self.assertEqual(record['status'], 'FAIL')          # the original is not modified

    def test_without_calibration_nothing_is_searched(self):
        record = measurement_record(line_with(15), None, 3, 1, 5, checks={'local_defect_length_mm': 2})
        self.assertEqual(record['status'], 'WARNING')
        self.assertEqual(record['local_defects'], [])


class CompleteLineTests(unittest.TestCase):
    """A line in pieces, or with adhesive on part of it only, is not a pass."""

    def broken(self):
        _, _, result = synthetic_inspection(broken=True)
        return result

    def test_coverage_and_gap_are_always_recorded(self):
        _, _, whole = synthetic_inspection()
        record = grade(whole)
        self.assertGreater(record['coverage_percent'], 85)
        self.assertEqual(record['largest_gap_mm'], 0)
        broken = grade(self.broken())
        self.assertLess(broken['coverage_percent'], record['coverage_percent'])
        self.assertGreater(broken['largest_gap_mm'], 3)

    def test_without_checks_two_pieces_inside_tolerance_still_pass(self):
        record = grade(self.broken())
        self.assertEqual(record['component_count'], 2)
        self.assertEqual(record['status'], 'PASS')

    def test_a_gap_above_the_limit_fails_the_line(self):
        gap = grade(self.broken())['largest_gap_mm']
        record = grade(self.broken(), max_gap_mm=gap - 1)
        self.assertEqual(record['status'], 'FAIL')
        self.assertIn('gap of', record['fail_reasons'][0])
        self.assertEqual(grade(self.broken(), max_gap_mm=gap + 1)['status'], 'PASS')

    def test_coverage_below_the_limit_fails_the_line(self):
        coverage = grade(self.broken())['coverage_percent']
        record = grade(self.broken(), min_coverage_percent=coverage + 5)
        self.assertEqual(record['status'], 'FAIL')
        self.assertIn('% of the line', record['fail_reasons'][0])
        self.assertEqual(grade(self.broken(), min_coverage_percent=coverage - 5)['status'], 'PASS')

    def test_an_incomplete_line_fails_even_without_calibration(self):
        record = measurement_record(self.broken(), None, 3, 1, 5, checks={'min_coverage_percent': 99})
        self.assertEqual(record['status'], 'FAIL')

    def stray(self):
        """The whole line plus a 2 mm scrap of adhesive beside its end."""
        _, _, result = synthetic_inspection()
        result = copy.deepcopy(result)
        scrap = copy.deepcopy(result['components'][0])
        scrap['samples'] = scrap['samples'][:5]
        scrap['length_px'] = 20.0
        for sample in scrap['samples']:
            for key in ('center_x', 'left_x', 'right_x'):
                sample[key] += 60
        result['components'].append(scrap)
        result['component_count'] = 2
        return result

    def test_a_scrap_shorter_than_one_segment_turns_the_line_to_warning_by_default(self):
        self.assertEqual(grade(self.stray())['status'], 'WARNING')

    def test_a_scrap_below_the_ignore_length_is_not_graded(self):
        record = grade(self.stray(), ignore_pieces_shorter_than_mm=5)
        self.assertEqual(record['status'], 'PASS')
        self.assertEqual([c['ignored'] for c in record['components']], [False, True])

    def test_the_longest_piece_is_never_ignored(self):
        record = grade(self.broken(), ignore_pieces_shorter_than_mm=1000)
        self.assertEqual(sum(not c['ignored'] for c in record['components']), 1)

    def test_the_ratio_preview_keeps_the_verdict(self):
        gap = grade(self.broken())['largest_gap_mm']
        record = grade(self.broken(), max_gap_mm=gap - 1)
        self.assertEqual(with_distance_ratio(record, record['distance_ratio'])['status'], 'FAIL')
        self.assertEqual(with_distance_ratio(record, record['distance_ratio'] * .5)['status'], 'FAIL')


class MissingLineTests(unittest.TestCase):
    """A line with no adhesive at all: an error by default, FAIL when coverage is checked."""

    def run_batch(self, coverage, with_missing=True):
        frame, inspector, _ = synthetic_inspection()
        real = inspector.glue.predict
        calls = []
        def first_line_empty(flat):
            calls.append(1)
            return np.zeros(flat.shape[:2], np.uint8) if len(calls) == 1 else real(flat)
        inspector.glue.predict = first_line_empty
        config = copy.deepcopy(CONFIG)
        config['inspection']['min_coverage_percent'] = coverage
        results, errors, missing = [], [], []
        with tempfile.TemporaryDirectory() as folder:
            store = CaptureStore(folder, config['capture_storage'])
            pipeline = InspectionPipeline(inspector, store, config)
            extra = {'on_missing': missing.append} if with_missing else {}
            with patch('inspection.pipeline.plane_scale.load_frame_scale', return_value=SCALE):
                pipeline.run_batch(frame, frame, {}, 3, 1, on_result=results.append,
                                   on_error=errors.append, **extra)
            record = json.loads((store._sets()[0] / 'result.json').read_text())
        return results, errors, missing, record

    def test_by_default_an_empty_line_is_an_error_as_before(self):
        results, errors, missing, record = self.run_batch(0)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(missing, [])
        self.assertNotIn('missing_lines', record)

    def test_with_a_coverage_check_an_empty_line_is_reported_as_failed(self):
        results, errors, missing, record = self.run_batch(80)
        self.assertEqual(len(results), 1)
        self.assertEqual(errors, [])
        self.assertEqual(len(missing), 1)
        self.assertIn('no adhesive found', missing[0])
        self.assertEqual(record['missing_lines'], [1])
        self.assertEqual(record['status'], 'complete_with_errors')

    def test_a_caller_without_the_callback_still_gets_the_error(self):
        _, errors, _, _ = self.run_batch(80, with_missing=False)
        self.assertEqual(len(errors), 1)

    def test_the_controller_turns_it_into_a_line_failed_event(self):
        from auto_trigger.controller import AutoController
        controller = AutoController.__new__(AutoController)
        import queue, threading
        controller.events = queue.SimpleQueue()
        controller.lock = threading.Lock()
        controller.wake = threading.Event()
        controller.stopped = threading.Event()
        controller.generation = 0
        controller.inspected_lines = set()
        controller.preparing_count = 0
        controller.pending_jobs = []
        controller.has_results = False
        controller.last_status = None
        class Pipeline:
            store = None
            def run_batch(self, original, corrected, metadata, width, tolerance, path,
                          on_result=None, on_error=None, on_missing=None):
                on_missing('Line 1: no adhesive found (FAIL)')
        controller.pipeline = Pipeline()
        controller._start_job({'original': None, 'corrected': None, 'metadata': {}, 'path': None,
                               'profile': {'width': 3, 'tolerance': 1}, 'generation': 0, 'epoch': 0})
        controller.future.result(timeout=5)
        events = []
        while not controller.events.empty():
            events.append(controller.events.get())
        self.assertIn(('line_failed', 0, 'Line 1: no adhesive found (FAIL)'), events)


class SettingsTests(unittest.TestCase):
    def test_values_are_read_from_the_inspection_section(self):
        self.assertEqual(grading_checks({'local_defect_length_mm': 3})['local_defect_length_mm'], 3.0)
        self.assertEqual(grading_checks(None), CHECKS)

    def test_bad_values_are_refused(self):
        for name in CHECKS:
            for bad in (-1, 'long', True, float('nan')):
                with self.subTest(name=name, bad=bad):
                    config = copy.deepcopy(CONFIG)
                    config['inspection'][name] = bad
                    with self.assertRaises(ValueError):
                        validate_config(config)

    def test_a_configuration_without_the_keys_is_valid(self):
        config = copy.deepcopy(CONFIG)
        for name in CHECKS:
            config['inspection'].pop(name, None)
        validate_config(config)


if __name__ == '__main__':
    unittest.main()
