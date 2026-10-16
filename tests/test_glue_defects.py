"""tools/glue_defects: the narrow-glue check, the edge-visibility gate, the data layout."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from tools.glue_defects import path_normals, photo_of
from tools.glue_defects.check import edge_visibility, judge, segment_check
from tools.glue_defects.viewer import load_lines, sources, verdicts

ARC = np.arange(0, 300, .5)


def widths(narrow=None, value=3.0):
    w = np.full(len(ARC), 4.2)
    if narrow:
        w[(ARC >= narrow[0]) & (ARC <= narrow[1])] = value
    return w


class JudgeTest(unittest.TestCase):
    def test_a_lit_narrow_stretch_fails(self):
        runs = judge(ARC, widths((100, 115)), np.full(len(ARC), 10.))[3]
        self.assertEqual(len(runs), 1)
        self.assertLess(runs[0][0], 105)
        self.assertGreater(runs[0][1], 110)

    def test_a_dark_stretch_is_never_a_defect(self):
        vis = np.full(len(ARC), 10.)
        vis[(ARC >= 90) & (ARC <= 125)] = 1.0          # the light does not reach here
        reference, ratio, judged, runs = judge(ARC, widths((100, 115)), vis)
        self.assertEqual(runs, [])
        self.assertFalse(judged[(ARC >= 100) & (ARC <= 115)].any())

    def test_a_slow_change_across_the_board_passes_against_the_width_nearby(self):
        w = np.linspace(4.6, 3.6, len(ARC))            # camera scale, not glue
        vis = np.full(len(ARC), 10.)
        self.assertEqual(judge(ARC, w, vis, local_mm=60)[3], [])
        self.assertNotEqual(judge(ARC, w, vis, local_mm=0)[3], [])

    def test_the_tips_and_a_line_cut_by_the_photo_edge_are_not_judged(self):
        vis = np.full(len(ARC), 10.)
        self.assertEqual(judge(ARC, widths((0, 4), 1.0), vis)[3], [])
        self.assertEqual(judge(ARC, widths((100, 115)), vis, cut_by_edge=True)[3], [])

    def test_a_short_dip_under_the_minimum_length_passes(self):
        self.assertEqual(judge(ARC, widths((100, 101), 3.9), np.full(len(ARC), 10.), min_mm=10)[3], [])


class SegmentCheckTest(unittest.TestCase):
    def test_narrower_than_its_neighbours_inside_the_tolerance(self):
        w = [4.4, 4.5, 4.4, 3.8, 4.5, 4.4, 4.5]        # 3.8 is inside 4.0 +/- 1.0
        self.assertEqual(segment_check(w, [10] * 7), [3])

    def test_dark_and_unmeasured_segments(self):
        w = [4.4, 4.5, None, 3.8, 4.5, 4.4, 4.5]
        self.assertEqual(segment_check(w, [10, 10, 10, 1, 10, 10, 10]), [])
        self.assertEqual(segment_check(w, [10] * 7), [3])


class VisibilityTest(unittest.TestCase):
    def test_sharp_edges_score_higher_than_faint_ones(self):
        xy = np.c_[np.arange(100, 500, 2.), np.full(200, 200.)]
        arc = np.arange(200) * 2 * 4 / 29.6
        for contrast, low in ((80, True), (3, False)):
            image = np.full((400, 600), 150, np.uint8)
            image[185:216] = 150 - contrast
            vis = edge_visibility(image, xy, arc)
            self.assertEqual(bool(np.median(vis) > 3.0), low)

    def test_normals_are_unit_and_across(self):
        xy = np.c_[np.arange(50.), np.zeros(50)]
        n = path_normals(xy)
        np.testing.assert_allclose(np.abs(n[:, 1]), 1, atol=1e-6)


class DataLayoutTest(unittest.TestCase):
    def test_sources_lines_and_verdicts(self):
        with TemporaryDirectory() as folder:
            data = Path(folder)
            (data / 'photos').mkdir()
            cv2.imwrite(str(data / 'photos' / 'a.png'), np.zeros((50, 50, 3), np.uint8))
            photos = {'a': dict(file='photos/a.png', label='A', good=False, stickers=[[150, 30]]),
                      'b': dict(file='photos/a.png', label='B', good=True, stickers=[])}
            (data / 'photos.json').write_text(json.dumps(photos))
            xy = np.c_[np.arange(30, 30 + len(ARC) / 2 * 7.4, 3.7), np.full(len(ARC), 30.)][:len(ARC)]
            line = dict(arc=ARC.tolist(), xy=xy.tolist(), origin=[0, 0])
            (data / 'app').mkdir()
            (data / 'app' / 'widths.json').write_text(json.dumps({
                'a_L1': dict(line, w=widths((10, 30)).tolist()), 'b_L1': dict(line, w=widths().tolist())}))
            (data / 'visibility.json').write_text(json.dumps({'a_L1': [10.] * len(ARC), 'b_L1': [10.] * len(ARC)}))
            found = sources(data)
            self.assertEqual(list(found.values()), [data / 'app'])
            lines = load_lines(data, data / 'app')
            self.assertEqual([l['truth'] for l in lines], ['sticker', 'defect-free'])
            _, counts = verdicts(lines, dict(limit=.9, min_mm=5, local_mm=60, vis_min=3))
            self.assertEqual(counts, dict(caught=1, sticker=1, failed=0, passed=1))
        self.assertEqual(photo_of('13oct_1419_L2'), '13oct_1419')


if __name__ == '__main__':
    unittest.main()
