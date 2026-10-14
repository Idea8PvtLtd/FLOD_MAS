"""The optional smooth ribbon after the adhesive mask (local_inspection.ribbon_finish)."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from inspection.local.engine import LiveInspection
from inspection.line_measurement import measurement_record
from config_validation import validate_config

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
HAS_TREES = importlib.util.find_spec('lightgbm') is not None


def painted_inspection(ribbon_finish, broken=False, wobble=4):
    """A photo with a dark band of even width where the glue is, and an adhesive
    mask whose edges wobble around that band, as the model's mask does."""
    source = np.zeros((800, 1100), np.uint8)
    y = np.arange(30, 770)
    for x in (350, 850):
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        polygon = np.vstack(([x-160, 30], boundary, [x-160, 769]))
        cv2.fillPoly(source, [np.rint(polygon).astype(np.int32)], 255)

    def adhesive(shift):
        def predict(flat):
            h, w = flat.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            for start in range(15, w-15, 8):
                low, high = (35 + shift(start), 65 + shift(start + 1)) if shift else (35, 65)
                mask[low:high, start:min(start+8, w-15)] = 255
            if broken:
                mask[:, w//2-60:w//2+60] = 0
            return mask
        return predict

    def inspector(predict, finish):
        made = LiveInspection.__new__(LiveInspection)
        made.offset = 100.
        made.source = SimpleNamespace(predict=lambda photo: source.copy())
        made.glue = SimpleNamespace(width=1024, predict=predict)
        made.ribbon_finish = finish
        return made

    # Where the even band lies in the photo: the engine's own refolding of it.
    plain = np.full((800, 1100, 3), 100, np.uint8)
    true_band = inspector(adhesive(None), False).run(plain)
    random = np.random.default_rng(3)
    photo = 150 + random.normal(0, 2, source.shape) - 40 * cv2.GaussianBlur(
        (true_band['mask'] > 0).astype(np.float32), (0, 0), 2)
    photo = cv2.cvtColor(np.clip(photo, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    steps = np.random.default_rng(5).integers(-wobble, wobble + 1, 4096)
    result = inspector(adhesive(lambda at: int(steps[at])), ribbon_finish).run(photo)
    return true_band, result


def widths(result):
    samples = [s['width_px'] for s in result['components'][0]['samples'] if s['valid']]
    return np.array(samples[len(samples)//10: len(samples) - len(samples)//10])


class RibbonSettingTests(unittest.TestCase):
    def test_by_default_no_ribbon_is_fitted_or_recorded(self):
        _, result = painted_inspection(False)
        self.assertNotIn('ribbon', result)
        self.assertNotIn('ribbon', measurement_record(result, None, 4.0, 1.0))

    def test_the_setting_is_validated(self):
        config = copy.deepcopy(CONFIG)
        self.assertIs(config['local_inspection']['ribbon_finish'], False)
        config['local_inspection']['ribbon_finish'] = True
        validate_config(config)
        config['local_inspection']['ribbon_finish'] = 'yes'
        with self.assertRaises(ValueError):
            validate_config(config)


@unittest.skipUnless(HAS_TREES, 'lightgbm is not installed')
class RibbonFinishTests(unittest.TestCase):
    def test_a_wobbling_mask_becomes_an_even_ribbon_on_the_band(self):
        true_band, mask_only = painted_inspection(False)
        _, finished = painted_inspection(True)
        self.assertIn(finished['ribbon']['state'], ('ok', 'check'))
        self.assertEqual(finished['component_count'], 1)
        # Steadier along the line than the mask it came from, and as wide as the band.
        self.assertLess(np.std(widths(finished)), 0.5 * np.std(widths(mask_only)))
        self.assertAlmostEqual(float(np.mean(widths(finished))), float(np.mean(widths(true_band))), delta=2.0)
        self.assertAlmostEqual(finished['ribbon']['mask_width_px'], mask_only['mean_width_px'], places=6)
        record = measurement_record(finished, None, 4.0, 1.0)
        self.assertEqual(record['ribbon'], finished['ribbon'])
        json.dumps(record['ribbon'])

    def test_the_ribbon_ends_round_on_the_tips_of_the_mask(self):
        # An even mask, so that its two tips are plain to read.
        _, mask_only = painted_inspection(False, wobble=0)
        _, finished = painted_inspection(True, wobble=0)
        self.assertEqual(finished['ribbon']['end_found'], [True, True])
        self.assertEqual(len(finished['ribbon']['end_moved_px']), 2)
        ribbon, mask = finished['mask'] > 0, mask_only['mask'] > 0
        # The line runs down the photo: its two tips are its first and last rows.
        for rows, tip, inward in ((np.flatnonzero(ribbon.any(axis=1)), 0, 1),
                                  (np.flatnonzero(ribbon.any(axis=1))[::-1], 0, -1)):
            mask_rows = np.flatnonzero(mask.any(axis=1))
            mask_tip = mask_rows[0] if inward == 1 else mask_rows[-1]
            self.assertLessEqual(abs(int(rows[tip]) - int(mask_tip)), 2)
            # Round, not cut straight across: narrow at the tip, full width a strip's width in.
            at_tip = np.count_nonzero(ribbon[rows[tip]])
            further_in = np.count_nonzero(ribbon[rows[tip] + inward * 30])
            self.assertLess(at_tip, 0.6 * further_in)
        from inspection.local.ribbon import ends
        ring = ends.capped_ring(np.column_stack((np.full(60, 50.0), np.arange(60) * 2.0 + 40)),
                                np.full(60, 30.0), start_moved=4.0, end_moved=-6.0)
        self.assertAlmostEqual(ring[:, 1].min(), 40 - 4.0, delta=0.5)        # the tips are where asked
        self.assertAlmostEqual(ring[:, 1].max(), 40 + 118 - 6.0, delta=0.5)
        self.assertAlmostEqual(np.ptp(ring[:, 0]), 30.0, delta=0.5)
        self.assertIsNone(ends.capped_ring(np.column_stack((np.zeros(5), np.arange(5) * 2.0)), np.full(5, 30.0)))

    def test_a_line_in_pieces_is_left_as_it_is_so_the_gap_stays(self):
        _, mask_only = painted_inspection(False, broken=True)
        _, finished = painted_inspection(True, broken=True)
        self.assertEqual(finished['ribbon']['state'], 'skipped')
        self.assertEqual(finished['component_count'], 2)
        self.assertTrue(np.array_equal(finished['mask'], mask_only['mask']))
        self.assertAlmostEqual(finished['length_px'], mask_only['length_px'])


if __name__ == '__main__':
    unittest.main()
