"""Extra grading checks: off by default, and what each one catches when it is set."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from config_validation import validate_config
from inspection.line_measurement import (CHECKS, grading_checks, measurement_record,
                                         with_distance_ratio)
from plane_scale import PlaneScale
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
