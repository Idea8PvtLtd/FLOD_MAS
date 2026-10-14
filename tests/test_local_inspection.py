"""Actual reference geometry with deterministic AI masks; no network or camera."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from concurrent.futures import Future
import cv2
import numpy as np
from inspection.local.engine import LiveInspection
from inspection.line_measurement import measurement_record
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from plane_scale import PlaneScale, PlaneScaleError
from config_validation import validate_config
from auto_trigger.controller import AutoController

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
# The shipped config.json carries the site's end zones, segment count and minimum
# line length. These tests build on the plain measurement, so they start with those
# stages off and switch on only what a test checks.
CONFIG['inspection'].update(end_exclusion_px=0, end_trim_px=0, segment_count=10, min_line_length_mm=0)


def synthetic_inspection(broken=False):
    frame = np.full((800, 1100, 3), 100, np.uint8)
    source = np.zeros(frame.shape[:2], np.uint8)
    y = np.arange(30, 770)
    for x in (350, 850):
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        polygon = np.vstack(([x-160, 30], boundary, [x-160, 769]))
        cv2.fillPoly(source, [np.rint(polygon).astype(np.int32)], 255)
    def adhesive(flat):
        h, w = flat.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[35:65, 15:w-15] = 255
        if broken:
            mask[:, w//2-60:w//2+60] = 0
        return mask
    inspector = LiveInspection.__new__(LiveInspection)
    inspector.offset = 100.
    inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
    inspector.glue = SimpleNamespace(width=1024, predict=adhesive)
    return frame, inspector, inspector.run(frame)


def stepped_edge():
    """A gently curved edge as an enlarged mask gives it: x in steps of four pixels."""
    y = np.arange(0, 1600, dtype=float)
    x = np.floor((400 + 60 * np.sin(y / 1600 * 2 * np.pi)) / 4) * 4
    return np.column_stack((x, y))


class BandPathTests(unittest.TestCase):
    """The inward band's inner edge must follow the line, not run back and forth."""

    def backwards(self, path, edge):
        from inspection.local.geometry import smoothed_path
        direction = np.gradient(smoothed_path(edge, 101), axis=0)
        return float(np.mean(np.sum(np.diff(path, axis=0) * direction[1:], axis=1) < 0))

    def length(self, path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def test_the_default_is_unchanged(self):
        from inspection.local.geometry import offset_path
        edge = stepped_edge()
        np.testing.assert_array_equal(offset_path(edge, 100.), offset_path(edge, 100., smooth_rows=0))
        self.assertGreater(self.backwards(offset_path(edge, 100.), edge), .02)

    def test_a_smoothed_direction_gives_a_path_that_does_not_run_backwards(self):
        from inspection.local.geometry import offset_path
        edge = stepped_edge()
        raw, smooth = offset_path(edge, 100.), offset_path(edge, 100., smooth_rows=101)
        self.assertLess(self.backwards(smooth, edge), .01)
        self.assertLess(self.length(smooth), 1.15 * self.length(edge))
        self.assertGreater(self.length(raw), 2 * self.length(smooth))
        # The offset distance itself is untouched: every point is still 100 px away.
        np.testing.assert_allclose(np.linalg.norm(smooth - edge, axis=1), 100., atol=1e-6)

    def test_the_engine_passes_the_setting_and_still_measures(self):
        frame, inspector, plain = synthetic_inspection()
        inspector.smooth_normal_rows = 101
        smooth = inspector.run(frame)
        self.assertEqual(smooth['line_count'], plain['line_count'])
        self.assertAlmostEqual(smooth['mean_width_px'], plain['mean_width_px'], delta=1.5)

    def test_the_setting_is_validated(self):
        config = copy.deepcopy(CONFIG)
        config['local_inspection']['smooth_normal_rows'] = 101
        validate_config(config)
        for bad in (-1, 2.5, 'many', True):
            config['local_inspection']['smooth_normal_rows'] = bad
            with self.assertRaises(ValueError):
                validate_config(config)
        del config['local_inspection']['smooth_normal_rows']
        validate_config(config)


class ScoreEnlargementTests(unittest.TestCase):
    """Where the edge lands when the model's 64 rows are enlarged to the strip's 100."""

    def scores(self, edge_row):
        """Two-class scores whose true edge lies at a fractional model row."""
        rows = np.arange(64, dtype=np.float32)[:, None] * np.ones((1, 32), np.float32)
        glue = np.clip(edge_row - rows, -4, 4)          # positive above the edge
        return np.stack([-glue, glue])[None]

    def edge(self, mask):
        return float(np.count_nonzero(mask[:, 5]))

    def test_the_default_is_the_argmax_then_nearest_recipe(self):
        from inspection.local.glue_model import segformer_mask
        scores = self.scores(20.4)
        labels = np.argmax(scores, axis=1)[0].astype(np.uint8) * 255
        expected = cv2.resize(labels, (50, 100), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(segformer_mask(scores, (100, 50)), expected)

    def test_smooth_enlargement_follows_a_moving_edge_in_finer_steps(self):
        from inspection.local.glue_model import segformer_mask
        positions = np.arange(20.0, 22.01, .1)
        plain = [self.edge(segformer_mask(self.scores(p), (100, 50))) for p in positions]
        smooth = [self.edge(segformer_mask(self.scores(p), (100, 50), smooth=True)) for p in positions]
        # Two model rows are 3.1 strip pixels. The plain mask can only move by a
        # whole model row, so it jumps two pixels at once; the smooth one moves
        # a pixel at a time and stays closer to where the edge really is.
        self.assertGreaterEqual(max(np.abs(np.diff(plain))), 2)
        self.assertLessEqual(max(np.abs(np.diff(smooth))), 1)
        truth = positions * 100 / 64
        self.assertLess(np.ptp(np.array(smooth) - truth), np.ptp(np.array(plain) - truth))

    def test_the_setting_is_validated(self):
        config = copy.deepcopy(CONFIG)
        config['local_inspection']['smooth_logits'] = True
        validate_config(config)
        config['local_inspection']['smooth_logits'] = 'yes'
        with self.assertRaises(ValueError):
            validate_config(config)


class FalseLineTests(unittest.TestCase):
    """A short false edge must not be numbered as a line when the filter is set."""

    def inspector(self, ratio):
        frame, inspector, _ = synthetic_inspection()
        whole = inspector.source.predict(frame)
        # A third, much shorter panel on the left: its edge is a line of its own.
        y = np.arange(300, 520)
        boundary = np.column_stack((120 + 12*np.sin((y-300)*2*np.pi/220), y))
        cv2.fillPoly(whole, [np.rint(np.vstack(([20, 300], boundary, [20, 519]))).astype(np.int32)], 255)
        inspector.source = SimpleNamespace(predict=lambda photo: whole.copy())
        inspector.min_line_length_ratio = ratio
        return frame, inspector

    def lines(self, ratio):
        frame, inspector = self.inspector(ratio)
        return [r['line_count'] for r in inspector.run_lines(frame, on_error=lambda index, error: None)]

    def test_by_default_every_line_is_kept(self):
        self.assertEqual(set(self.lines(0)), {3})

    def test_a_line_much_shorter_than_the_longest_is_dropped(self):
        self.assertEqual(self.lines(.5), [2, 2])

    def test_a_low_ratio_keeps_it(self):
        self.assertEqual(set(self.lines(.2)), {3})

    def test_the_setting_is_validated(self):
        config = copy.deepcopy(CONFIG)
        config['local_inspection']['min_line_length_ratio'] = .5
        validate_config(config)
        for bad in (-.1, 1, 'half', True):
            config['local_inspection']['min_line_length_ratio'] = bad
            with self.assertRaises(ValueError):
                validate_config(config)


class CaptureCostTests(unittest.TestCase):
    def test_the_photo_is_enhanced_once_for_all_lines_of_a_capture(self):
        frame, inspector, _ = synthetic_inspection()
        with patch('inspection.local.engine.enhance_image', wraps=__import__(
                'inspection.local.engine', fromlist=['enhance_image']).enhance_image) as enhance:
            results = list(inspector.run_lines(frame))
            self.assertEqual(len(results), 2)
            self.assertEqual(enhance.call_count, 1)
            # The next capture is a different photo and is enhanced afresh.
            list(inspector.run_lines(frame.copy()))
            self.assertEqual(enhance.call_count, 2)
        self.assertIsNone(inspector._enhanced)

    def test_the_result_is_the_same_as_enhancing_for_every_line(self):
        frame, inspector, _ = synthetic_inspection()
        rng = np.random.default_rng(1)
        frame = rng.integers(60, 200, frame.shape, dtype=np.uint8)
        seen = []
        real = inspector.glue.predict
        inspector.glue.predict = lambda flat: (seen.append(flat.copy()), real(flat))[1]
        list(inspector.run_lines(frame))
        once = list(seen)
        seen.clear()
        inspector._enhanced_photo = lambda photo: __import__(
            'inspection.local.engine', fromlist=['enhance_image']).enhance_image(photo, 2., 16, 16)
        list(inspector.run_lines(frame))
        self.assertEqual(len(once), len(seen))
        for a, b in zip(once, seen):
            np.testing.assert_array_equal(a, b)

    def test_each_line_reports_its_own_time(self):
        frame, inspector, _ = synthetic_inspection()
        first, second = list(inspector.run_lines(frame))
        self.assertLessEqual(first['line_seconds'], first['seconds'])
        self.assertLess(second['line_seconds'], second['seconds'])
        self.assertGreater(second['seconds'], first['seconds'])


class LocalGeometryTests(unittest.TestCase):
    def test_only_rightmost_line_is_refolded_to_full_frame(self):
        frame, _, result = synthetic_inspection()
        self.assertEqual(result['line_count'], 2)
        self.assertEqual(result['selected_line'], 2)
        self.assertEqual(result['mask'].shape, frame.shape[:2])
        self.assertEqual(result['overlay'].shape, frame.shape)
        self.assertFalse(result['mask'][:, :600].any())
        self.assertTrue(result['mask'][:, 700:].any())
        self.assertEqual(len(result['segments']), 10)
        self.assertAlmostEqual(result['mean_width_px'], 30, delta=2)
        # Returned intersections must be in the full undistorted frame, not crop pixels.
        valid = [s for s in result['components'][0]['samples'] if s['valid']]
        self.assertTrue(all(s['center_x'] > 600 and s['left_x'] > 600 for s in valid))

    def test_sideways_fabric_is_found_and_measured_in_the_camera_frame(self):
        # The same fabric laid sideways: turn the photo and its source mask a
        # quarter clockwise so both glue edges run left-to-right.
        frame, inspector, _ = synthetic_inspection()
        upright = list(inspector.run_lines(frame))
        source = inspector.source.predict(frame)
        sideways = np.ascontiguousarray(np.rot90(frame, -1))
        inspector.source = SimpleNamespace(
            predict=lambda photo: np.ascontiguousarray(np.rot90(source, -1)))
        turned = list(inspector.run_lines(sideways))
        self.assertEqual(len(turned), len(upright))
        # Upright, the left line comes first; sideways, the top one does.
        self.assertEqual([r['selected_line'] for r in turned], [1, 2])
        height, width = sideways.shape[:2]
        for before, after in zip(upright, turned):
            # The mask is the upright mask turned with the photo. The sideways line is
            # read in the camera frame, so its band edge is drawn there and a few edge
            # pixels can differ (12 of about 22,500 on this fabric).
            expected, got = np.rot90(before['mask'], -1) > 0, after['mask'] > 0
            self.assertGreaterEqual((expected & got).sum() / (expected | got).sum(), .995)
            self.assertEqual(after['overlay'].shape, sideways.shape)
            self.assertEqual(len(after['segments']), len(before['segments']))
            self.assertAlmostEqual(after['mean_width_px'], before['mean_width_px'], delta=1)
            self.assertAlmostEqual(after['length_px'], before['length_px'], delta=3)
            # Every measured point is in camera pixels, on the glue mask.
            for sample in after['components'][0]['samples']:
                if sample['valid']:
                    x, y = int(round(sample['center_x'])), int(round(sample['center_y']))
                    self.assertTrue(0 <= x < width and 0 <= y < height)
                    self.assertTrue(after['mask'][y, x], (x, y))
            boundary = np.asarray(after['selected_boundary']['original_xy'])
            self.assertTrue(after['selected_boundary'].get('turned'))
            self.assertTrue((boundary[:, 0] >= 0).all() and (boundary[:, 0] < width).all())
            self.assertTrue((boundary[:, 1] >= 0).all() and (boundary[:, 1] < height).all())
        # The top line is above the bottom one in the camera frame.
        self.assertLess(np.mean(np.nonzero(turned[0]['mask'])[0]),
                        np.mean(np.nonzero(turned[1]['mask'])[0]))

    def test_upright_fabric_is_not_turned(self):
        _, _, result = synthetic_inspection()
        self.assertNotIn('turned', result['selected_boundary'])

    def test_disconnected_glue_keeps_long_gap_and_sums_real_lengths(self):
        _, _, result = synthetic_inspection(broken=True)
        self.assertEqual(result['component_count'], 2)
        self.assertAlmostEqual(result['length_px'], sum(c['length_px'] for c in result['components']))
        _, _, continuous = synthetic_inspection()
        self.assertLess(result['length_px'], continuous['length_px'])
        main = max(result['components'], key=lambda c: c['length_px'])
        self.assertEqual(result['segments'], main['segments'])

    def test_empty_source_or_empty_adhesive_is_an_error(self):
        frame, inspector, _ = synthetic_inspection()
        inspector.source.predict = lambda f: np.zeros(f.shape[:2], np.uint8)
        with self.assertRaisesRegex(ValueError, 'No accepted source lines'):
            inspector.run(frame)
        _, inspector, _ = synthetic_inspection()
        inspector.glue.predict = lambda f: np.zeros(f.shape[:2], np.uint8)
        with self.assertRaisesRegex(ValueError, 'empty mask'):
            inspector.run(frame)


class CalibrationTests(unittest.TestCase):
    def test_metric_widths_transform_global_intersections_and_grade_all_segments(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        record = measurement_record(result, scale, 3, 2)
        self.assertEqual(record['status'], 'PASS')
        # Metric length uses transformed sampled centres; the reference stores
        # its pre-resampling arc length, so small chord discretization is expected.
        self.assertAlmostEqual(record['length_mm'], result['length_px']*.1, delta=.1)
        self.assertAlmostEqual(record['mean_width_mm'], result['mean_width_px']*.1, places=5)
        self.assertEqual(len(record['segments']), 10)
        self.assertTrue(all(s['within_tolerance'] for s in record['segments']))
        self.assertEqual(measurement_record(result, scale, 10, 1)['status'], 'FAIL')

    def test_projective_calibration_uses_endpoints_not_summary_scale(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.array([[.12, .01, 0], [0, .1, 0], [0, .0005, 1]]), 999)
        record = measurement_record(result, scale, 3, 2)
        sample = next(s for s in record['components'][0]['samples'] if s['valid'])
        endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                 [sample['right_x'], sample['right_y']]])
        self.assertAlmostEqual(sample['width_mm'], np.linalg.norm(endpoints[1]-endpoints[0]))
        self.assertLess(sample['width_mm'], 10)

    def test_missing_calibration_or_missing_segment_warns_instead_of_grading(self):
        _, _, result = synthetic_inspection()
        record = measurement_record(result, None, 4, 1)
        # No plane means nothing could be graded, so the fabric warns rather than
        # being called a failure it was never measured against.
        self.assertEqual(record['status'], 'WARNING')
        self.assertFalse(record['metric'])
        self.assertIsNone(record['length_mm'])
        for sample in result['components'][0]['samples']:
            if sample['segment'] == 1:
                sample['valid'] = False
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        self.assertEqual(measurement_record(result, scale, 3, 2)['status'], 'WARNING')

    def test_one_ungradeable_segment_warns_even_when_the_rest_are_out_of_tolerance(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        # Every measured segment is out of tolerance against a 10 mm target, so
        # this would be FAIL -- except segment 1 of the main component loses its
        # samples, and the missing measurement decides the label instead.
        for sample in result['components'][0]['samples']:
            if sample['segment'] == 1:
                sample['valid'] = False
        record = measurement_record(result, scale, 10, .2)
        self.assertEqual(record['status'], 'WARNING')
        # The segments that were measurable still carry their own verdicts.
        graded = {s['segment']: s['within_tolerance']
                  for s in record['components'][0]['segments']}
        self.assertIsNone(graded[1])
        self.assertIn(False, [value for key, value in graded.items() if key != 1])


class LocalPipelineTests(unittest.TestCase):
    def test_corrected_input_lossless_mask_and_serializable_metric_record(self):
        corrected, _, detected = synthetic_inspection()
        original = np.zeros_like(corrected)
        seen = []
        inspector = SimpleNamespace(run=lambda f, progress: (seen.append(f), detected)[1])
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, CONFIG['capture_storage'])
            pipeline = InspectionPipeline(inspector, store, CONFIG)
            with patch('inspection.pipeline.plane_scale.load_frame_scale',
                       return_value=PlaneScale(np.diag([.1,.1,1]), .1)):
                result = pipeline.run(original, corrected, {'size':'M'}, 3, 2)
            self.assertIs(seen[0], corrected)
            path = Path(result['path'])
            for name, expected in [('original.png',original), ('undistorted.png',corrected),
                                    ('mask.png',detected['mask']), ('overlay.png',detected['overlay'])]:
                np.testing.assert_array_equal(cv2.imread(str(path/name), cv2.IMREAD_UNCHANGED), expected)
            record = json.loads((path/'result.json').read_text())
            self.assertEqual(record['measurement']['status'], 'PASS')
            self.assertEqual(record['method'], 'local_onnx_rightmost_line')

    def test_missing_plane_keeps_pixel_result_and_warning(self):
        frame, _, detected = synthetic_inspection()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(SimpleNamespace(run=lambda f, progress: detected),
                                          CaptureStore(directory, CONFIG['capture_storage']), CONFIG)
            with patch('inspection.pipeline.plane_scale.load_frame_scale', side_effect=PlaneScaleError('missing')):
                result = pipeline.run(frame, frame, {}, 4, 1)
            self.assertTrue(result['warnings'])
            self.assertEqual(result['measurement']['status'], 'WARNING')

    def test_invalid_local_settings_are_rejected(self):
        for key, value in [('offset_pixels',0), ('offset_pixels',float('nan')),
                            ('source_width',31), ('source_width',True), ('glue_model','model.h5')]:
            config = copy.deepcopy(CONFIG)
            config['local_inspection'][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


class ResultLifecycleTests(unittest.TestCase):
    def test_obsolete_inspection_error_does_not_clear_new_results(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.generation = 2
        controller.job_generation = 1
        controller.future = Future()
        controller.future.set_exception(ValueError('old empty adhesive mask'))
        controller._complete()
        self.assertEqual(list(controller.poll()), [])

    def test_current_error_is_reported_and_stopped_result_is_suppressed(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.job_generation = controller.generation
        controller.future = Future()
        controller.future.set_exception(ValueError('empty adhesive mask'))
        controller._complete()
        self.assertEqual(list(controller.poll()), [('error', 0, 'empty adhesive mask')])
        controller.future = Future()
        controller.future.set_result({'old': True})
        controller.stop()
        controller._complete()
        self.assertNotIn('result', [event[0] for event in controller.poll()])


if __name__ == '__main__':
    unittest.main()
