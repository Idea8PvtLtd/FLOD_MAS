"""End zones graded as their own segments (inspection.end_exclusion_px; off by default)."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from inspection.line_measurement import measurement_record
from inspection.overlay import annotated_overlay, segment_ends
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from plane_scale import PlaneScale
from config_validation import validate_config
from test_local_inspection import synthetic_inspection, CONFIG

SCALE = PlaneScale(np.diag([.1, .1, 1]), .1)


def record(result, **ends):
    with patch('inspection.line_measurement.current_ratio', return_value=1.0):
        return measurement_record(result, SCALE, 3, .2, 5, **ends)


class EndZoneTests(unittest.TestCase):
    def test_off_and_then_nothing_changes(self):
        # Off is 0 or absent; the shipped config.json switches the end zones on for the site.
        _, _, result = synthetic_inspection()
        plain = record(result)
        self.assertEqual(record(result, end_exclusion_px=None, end_trim_px=7.0), plain)
        self.assertNotIn('end_exclusion_px', plain)
        for component in plain['components']:
            self.assertNotIn('trimmed_each_end_px', component)
            self.assertEqual(segment_ends(component), [])

    def test_each_end_is_a_segment_of_its_own_and_the_tip_is_left_out(self):
        frame, _, result = synthetic_inspection()
        zoned = record(result, end_exclusion_px=40, end_trim_px=10)
        self.assertEqual((zoned['end_exclusion_px'], zoned['end_trim_px']), (40.0, 10.0))
        for component in zoned['components']:
            length = component['length_px']
            numbers = [segment['segment'] for segment in component['segments']]
            self.assertEqual(numbers, list(range(0, 12)))
            self.assertEqual([component['segments'][i].get('end_zone') for i in (0, -1)], ['start', 'end'])
            self.assertAlmostEqual(component['inspected_length_px'], length - 80)
            for sample in component['samples']:
                arc = sample['arc_px']
                if arc < 7 or arc > length - 7:              # the tip: ignored
                    self.assertEqual(sample['segment'], -1)
                    self.assertTrue(sample['excluded_end'])
                    self.assertIsNone(sample['width_mm'])
                elif 13 < arc < 40:
                    self.assertEqual(sample['segment'], 0)
                    self.assertFalse(sample['excluded_end'])
                elif length - 40 < arc < length - 13:
                    self.assertEqual(sample['segment'], 11)
                elif 40 <= arc <= length - 40:
                    self.assertIn(sample['segment'], range(1, 11))
            for zone in (component['segments'][0], component['segments'][-1]):
                self.assertGreater(zone['sample_count'], 0)
                self.assertIsNotNone(zone['average_width_mm'])
                self.assertIsNotNone(zone['within_tolerance'])
            self.assertEqual(len(segment_ends(component)), 2)
        self.assertEqual(annotated_overlay(frame, zoned).shape, frame.shape)

    def test_a_bad_end_fails_the_line_only_when_ends_are_graded(self):
        _, _, result = synthetic_inspection()
        for component in result['components']:
            length = component['length_px']
            for sample in component['samples']:
                if sample['valid'] and 12 < sample['arc_px'] < 40:      # a thin start
                    left = np.array([sample['left_x'], sample['left_y']])
                    right = np.array([sample['right_x'], sample['right_y']])
                    centre, direction = (left + right) / 2, (right - left) / np.linalg.norm(right - left)
                    sample['left_x'], sample['left_y'] = centre - direction * .5
                    sample['right_x'], sample['right_y'] = centre + direction * .5
                    sample['width_px'] = 1.0
        self.assertEqual(record(result)['status'], 'PASS')
        zoned = record(result, end_exclusion_px=40, end_trim_px=10)
        self.assertEqual(zoned['status'], 'FAIL')
        self.assertIs(zoned['segments'][0]['within_tolerance'], False)
        self.assertTrue(all(segment['within_tolerance'] for segment in zoned['segments'][1:]))

    def test_a_line_shorter_than_both_zones_is_a_warning_not_an_error(self):
        _, _, result = synthetic_inspection()
        longest = max(component['length_px'] for component in result['components'])
        self.assertEqual(record(result, end_exclusion_px=longest)['status'], 'WARNING')

    def test_the_settings_are_validated(self):
        for key, bad in (('end_exclusion_px', -1), ('end_trim_px', 'x'), ('segment_count', 0),
                         ('segment_count', 2.5), ('min_line_length_mm', -5)):
            config = copy.deepcopy(CONFIG)
            config['inspection'][key] = bad
            with self.assertRaises(ValueError, msg=key):
                validate_config(config)
        config = copy.deepcopy(CONFIG)
        config['inspection'].update(end_exclusion_px=40, end_trim_px=10, segment_count=40, min_line_length_mm=150)
        validate_config(config)
        with self.assertRaises(ValueError):
            measurement_record(synthetic_inspection()[2], None, 3, .2, 5, end_exclusion_px=-1)

    def test_the_segment_count_reaches_the_measurement(self):
        frame, inspector, _ = synthetic_inspection()
        inspector.segments = 4
        result = inspector.run(frame)
        self.assertEqual(len(result['segments']), 4)
        self.assertEqual(len(record(result, end_exclusion_px=40)['segments']), 6)

    def run_pipeline(self, config, millimetres_per_pixel):
        frame, inspector, _ = synthetic_inspection()
        results, errors = [], []
        with tempfile.TemporaryDirectory() as directory, \
                patch('inspection.line_measurement.current_ratio', return_value=1.0), \
                patch('inspection.pipeline.plane_scale.load_frame_scale',
                      return_value=PlaneScale(np.diag([millimetres_per_pixel] * 2 + [1]), millimetres_per_pixel)):
            pipeline = InspectionPipeline(inspector, CaptureStore(directory, config['capture_storage']), config)
            pipeline.run_batch(frame, frame, {}, 10, .1, on_result=results.append, on_error=errors.append)
        return results, errors

    def test_glue_lines_shorter_than_the_minimum_are_reported_not_graded(self):
        config = copy.deepcopy(CONFIG)
        # About 720 px of glue per line: 72 mm at 0.1 mm a pixel, 180 mm at 0.25.
        results, errors = self.run_pipeline(config, .1)
        self.assertEqual((len(results), errors), (2, []))                 # no minimum: as before
        config['inspection']['min_line_length_mm'] = 150
        results, errors = self.run_pipeline(config, .1)
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 2)
        self.assertTrue(all('shorter than the 150 mm minimum' in error for error in errors), errors)
        results, errors = self.run_pipeline(config, .25)
        self.assertEqual([r['measurement']['selected_line'] for r in results], [1, 2])

    def test_the_pipeline_passes_the_end_zone_on(self):
        config = copy.deepcopy(CONFIG)
        results, _ = self.run_pipeline(config, .1)
        self.assertEqual(len(results[0]['measurement']['segments']), 10)
        config['inspection'].update(end_exclusion_px=40, end_trim_px=10)
        results, _ = self.run_pipeline(config, .1)
        self.assertEqual(len(results[0]['measurement']['segments']), 12)


if __name__ == '__main__':
    unittest.main()
