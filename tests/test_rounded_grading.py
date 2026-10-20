"""Segment widths graded at 0.1 mm, a failing segment rounding toward its sides.

The examples are the ones the rule was agreed on, for a 3.7 ± 0.4 mm profile
(pass band 3.3 - 4.1 mm).
"""
import unittest

from inspection.line_measurement import grade_segments, rounded_record, _to_step

TARGET, TOLERANCE = 3.7, 0.4


def graded(widths, end_zones=()):
    segments = [{'segment': i + 1, 'average_width_mm': w} for i, w in enumerate(widths)]
    for index, zone in end_zones:
        segments[index]['end_zone'] = zone
    grade_segments(segments, TARGET, TOLERANCE)
    return ([s['graded_width_mm'] for s in segments],
            [s['within_tolerance'] for s in segments],
            [s['rounding'] for s in segments])


class RoundingTests(unittest.TestCase):
    def test_widths_round_half_up_to_one_decimal(self):
        self.assertEqual([_to_step(v, 'nearest') for v in (4.14, 4.15, 4.149, 3.25, 3.249)],
                         [4.1, 4.2, 4.1, 3.3, 3.2])

    def test_every_segment_is_graded_on_its_rounded_width(self):
        values, passes, _ = graded([3.92, 4.03, 3.86, 4.06, 4.13, 4.02, 3.88])
        self.assertEqual(values, [3.9, 4.0, 3.9, 4.1, 4.1, 4.0, 3.9])
        self.assertTrue(all(passes))


class SideRuleTests(unittest.TestCase):
    def test_one_defect_rounds_toward_lower_sides(self):
        values, passes, rounding = graded([3.92, 4.03, 3.86, 4.06, 4.16, 4.02, 3.88])
        self.assertEqual((values[4], passes[4], rounding[4]), (4.1, True, 'down'))

    def test_one_defect_on_the_low_side_rounds_up(self):
        values, passes, rounding = graded([3.61, 3.42, 3.24, 3.51, 3.66, 3.72, 3.80])
        self.assertEqual((values[2], passes[2], rounding[2]), (3.3, True, 'up'))

    def test_a_run_shares_the_direction_of_its_two_sides(self):
        values, passes, rounding = graded([3.90, 4.04, 3.95, 4.16, 4.18, 4.27, 4.01])
        self.assertEqual(values[3:6], [4.1, 4.1, 4.2])
        self.assertEqual(passes[3:6], [True, True, False])   # 4.27 is too far out either way
        self.assertEqual(rounding[3:6], ['down', 'down', 'down'])

    def test_a_run_at_the_end_of_the_line_uses_its_one_side(self):
        values, passes, rounding = graded([3.95, 4.02, 4.17, 4.19])
        self.assertEqual(values[2:], [4.1, 4.1])
        self.assertEqual(rounding[2:], ['down', 'down'])

    def test_an_end_zone_and_a_missing_width_are_skipped_as_sides(self):
        # Run in the middle; its left side is a segment with no width, so the
        # passing segment beyond it decides. An end zone side is skipped too.
        values, passes, _ = graded([4.0, None, 4.17, 3.95], end_zones=[(0, 'start')])
        self.assertEqual((values[2], passes[2]), (4.1, True))
        self.assertIsNone(passes[1])

    def test_a_defect_far_outside_stays_a_defect(self):
        values, passes, rounding = graded([3.9, 4.36, 3.9])
        self.assertEqual((values[1], passes[1], rounding[1]), (4.3, False, 'down'))

    def test_a_line_where_every_segment_fails_keeps_normal_rounding(self):
        values, passes, rounding = graded([4.16, 4.18])
        self.assertEqual((values, passes, rounding), ([4.2, 4.2], [False, False], ['nearest', 'nearest']))


class RecordTests(unittest.TestCase):
    def test_the_saved_record_holds_only_values_at_one_decimal(self):
        measurement = {'length_mm': 332.49, 'longest_length_mm': 332.49, 'mean_width_mm': 4.104,
                       'components': [{'length_mm': 332.49,
                                       'samples': [{'width_mm': 4.137}, {'width_mm': None}]}]}
        record = rounded_record(measurement)
        self.assertEqual((record['length_mm'], record['mean_width_mm']), (332.5, 4.1))
        self.assertEqual([s['width_mm'] for s in record['components'][0]['samples']], [4.1, None])
        self.assertEqual(measurement['mean_width_mm'], 4.104)   # the running result is untouched


if __name__ == '__main__':
    unittest.main()


def test_end_zones_pass_from_their_own_minimum():
    from inspection.line_measurement import grade_segments
    segments = [{'segment': 0, 'end_zone': 'start', 'average_width_mm': 2.9},
                {'segment': 1, 'average_width_mm': 4.0},
                {'segment': 2, 'average_width_mm': 2.94},
                {'segment': 3, 'end_zone': 'end', 'average_width_mm': 2.8}]
    grade_segments(segments, 4.0, 1.0, end_min=2.9)
    assert [s['within_tolerance'] for s in segments] == [True, True, True, False]   # middle 2.94 rounds up to 3.0 (its side is 4.0)
    grade_segments(segments, 4.0, 1.0)                     # without it, end zones use 3.0 like the rest
    assert segments[0]['within_tolerance'] is False
