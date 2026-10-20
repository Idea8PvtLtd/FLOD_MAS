"""Width labels on failing segments only, or on every segment."""
import unittest
from unittest import mock

import numpy as np

from inspection import overlay


def measurement():
    samples, segments = [], []
    for index, ok in enumerate((True, False, True)):
        for y in range(index * 40, index * 40 + 40, 5):
            samples.append(dict(segment=index, center_x=50.0, center_y=float(y), left_x=45.0, left_y=float(y),
                                right_x=55.0, right_y=float(y), valid=True, arc_px=float(y)))
        segments.append(dict(segment=index, within_tolerance=ok, average_width_mm=4.0 if ok else 2.5,
                             average_width_px=30.0))
    return {'components': [dict(samples=samples, segments=segments)]}


class DefectLabelTests(unittest.TestCase):
    def labels(self, **kwargs):
        with mock.patch.object(overlay, 'draw_label') as draw:
            overlay.annotated_overlay(np.zeros((140, 120, 3), np.uint8), measurement(), **kwargs)
        return [call.args[1] for call in draw.call_args_list]

    def test_all_segments_are_labelled_by_default(self):
        self.assertEqual(self.labels(), ['4.0 mm', '2.5 mm', '4.0 mm'])

    def test_only_failing_segments_are_labelled_when_asked(self):
        self.assertEqual(self.labels(defect_labels_only=True), ['2.5 mm'])


if __name__ == '__main__':
    unittest.main()
