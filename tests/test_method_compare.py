"""The method comparison tool (tools/method_compare): its numbers, not its window."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from auto_trigger.scene_monitor import SceneMonitor
from tools.method_compare import BY_ID, METHODS, REFERENCE, latest_run
from tools.method_compare.charts import nice_max, photo_parts, repeatability, scale, summary, ticks, time_colour
from tools.method_compare.prepare import MomentPicker
from tools.method_compare.results import from_edges, from_ring, load, match, save, tracks
from tools.scene_db import FIXTURES, decode_thumb, read_trace


def strip(x, width, top=100, bottom=500, number=None):
    """An upright line: left edge at x, right edge width px further, sampled every 10 px."""
    ys = np.arange(top, bottom, 10.)
    return from_edges(np.column_stack((np.full_like(ys, x), ys)),
                      np.column_stack((np.full_like(ys, x + width), ys)), number=number)


class ResultsTests(unittest.TestCase):
    def test_edges_give_width_centre_and_segments(self):
        line = strip(200, 30, number=3)
        self.assertEqual(line['number'], 3)
        self.assertAlmostEqual(line['width_px'], 30.)
        self.assertEqual(line['centroid'], [215., 295.])
        self.assertEqual(len(line['segments_px']), 10)
        self.assertEqual(line['centre'][0], [215., 100.])
        self.assertIsNone(line['width_mm'])

    def test_unpaired_edges_are_refused(self):
        with self.assertRaises(ValueError):
            from_edges([[0, 0], [0, 1]], [[1, 0]])

    def test_a_ring_is_one_edge_then_the_other_backwards(self):
        left = [[100, y] for y in range(0, 50, 10)]
        right = [[128, y] for y in range(0, 50, 10)]
        line = from_ring(left + right[::-1], number=1)
        self.assertAlmostEqual(line['width_px'], 28.)
        self.assertEqual(line['left'][0], [100., 0.])
        self.assertEqual(line['right'][0], [128., 0.])

    def test_a_scale_gives_millimetres(self):
        class Half:
            @staticmethod
            def to_mm(points):
                return np.asarray(points, float) / 2
        ys = np.arange(0., 100., 10.)
        line = from_edges(np.column_stack((np.zeros_like(ys), ys)), np.column_stack((np.full_like(ys, 8.), ys)),
                          scale=Half())
        self.assertAlmostEqual(line['width_mm'], 4.)

    def test_lines_match_by_nearest_position(self):
        a = [strip(100, 30), strip(600, 30)]
        b = [strip(612, 31), strip(95, 29), strip(2000, 30)]
        self.assertEqual(match(a, b), [(0, 1), (1, 0)])
        self.assertEqual(match(a, [strip(400, 30)]), [])

    def test_tracks_follow_a_line_across_photos_of_a_board(self):
        photos = [[strip(1000, 30)], [strip(1010, 30), strip(600, 30)], [strip(605, 31), strip(1020, 30), strip(200, 30)]]
        found = sorted(tracks(photos))
        self.assertEqual(found, [[(0, 0), (1, 0), (2, 1)], [(1, 1), (2, 0)], [(2, 2)]])

    def test_files_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run' / 'B.json'
            save(path, {'lines': [strip(10, 5)]})
            self.assertEqual(load(path)['lines'][0]['width_px'], 5.)
            self.assertEqual(list(path.parent.glob('*.part')), [])

    def test_latest_run_needs_its_moments(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(latest_run(folder))
            for name in ('20261016_100000', '20261016_120000', '20261016_130000'):
                (Path(folder) / name).mkdir()
            for name in ('20261016_100000', '20261016_120000'):
                (Path(folder) / name / 'moments.json').write_text('{}')
            self.assertEqual(latest_run(folder).name, '20261016_120000')


class ChartTests(unittest.TestCase):
    def test_axis_ends_and_ticks_are_round(self):
        self.assertEqual(nice_max(3.7), 5.)
        self.assertEqual(nice_max(4.4), 5.)
        self.assertEqual(nice_max(11.), 20.)
        self.assertEqual(nice_max(0), 1.)
        self.assertEqual(ticks(5.), [0., 1., 2., 3., 4., 5.])

    def test_scale_maps_and_clamps(self):
        self.assertEqual(scale(2, 0, 4, 100, 300), 200)
        self.assertEqual(scale(9, 0, 4, 100, 300), 300)
        self.assertEqual(scale(1, 1, 1, 50, 60), 50)

    def test_time_colour_against_the_target(self):
        self.assertEqual(time_colour(3.9, 4.), 'good')
        self.assertEqual(time_colour(4.1, 4.), 'bad')
        self.assertEqual(time_colour(None, 4.), 'none')

    def test_photo_parts_per_family(self):
        app = {'family': 'app'}
        self.assertEqual(photo_parts(app, {'find_s': 1., 'new_lines_s': .5})[:2], (1., .5))
        v5 = {'family': 'v5'}
        self.assertEqual(photo_parts(v5, {'fixed_s': .6, 'photo_s': 2.})[:2], (.6, 1.4))

    def test_summary_against_the_reference_and_repeatability(self):
        moments = [{'board': 'v#1'}, {'board': 'v#1'}]
        reference = {'family': 'app', 'photos': [
            {'photo_s': 2., 'lines': [strip(500, 30)]},
            {'photo_s': 2.2, 'lines': [strip(505, 30), strip(900, 31)]}]}
        other = {'family': 'v5', 'photos': [
            {'photo_s': 3., 'lines': [strip(498, 27)]},
            {'photo_s': 3.4, 'lines': [strip(503, 28), strip(902, 28)]}]}
        self.assertEqual(repeatability(reference, moments), [0.])
        s = summary(other, moments, reference)
        self.assertAlmostEqual(s['photo_mean_s'], 3.2)
        self.assertEqual(s['lines'], 3)
        self.assertAlmostEqual(s['difference_mean_px'], -8 / 3)
        self.assertAlmostEqual(s['spread_median_px'], .5)
        self.assertIsNone(summary(reference, moments, reference)['difference_mean_px'])


class NewLineCountTests(unittest.TestCase):
    def test_version_5_estimates_use_the_apps_count_of_new_lines(self):
        from tools.method_compare.viewer import apply_new_line_counts
        reference = {'method': 'B', 'family': 'app', 'photos': [{'new': 1}, {'new': 1}]}
        v5 = {'family': 'v5', 'photos': [{'fixed_s': .5, 'per_line_s': 2., 'new': 1, 'photo_s': 2.5},
                                         {'fixed_s': .5, 'per_line_s': 2., 'new': 2, 'photo_s': 4.5}]}
        app = {'family': 'app', 'photos': [{'new': 1, 'photo_s': 1.}]}
        apply_new_line_counts({'B': reference, 'C': v5, 'A': app}, reference)
        self.assertEqual([row['photo_s'] for row in v5['photos']], [2.5, 2.5])
        self.assertEqual(app['photos'][0]['photo_s'], 1.)
        self.assertIn('count from B', v5['time_note'])


class MethodListTests(unittest.TestCase):
    def test_every_method_is_described(self):
        self.assertEqual([m['id'] for m in METHODS], ['A', 'B', 'B-fast', 'C', 'D', 'E'])
        self.assertIn(REFERENCE, BY_ID)
        for method in METHODS:
            self.assertIn(method['family'], ('app', 'v5'))
            self.assertEqual(len(method['colour']), 3)
            if method['family'] == 'app':
                self.assertIn('ribbon', method)
            else:
                self.assertIn(method['mode'], ('sam2.1-large', 'sam2.1-tiny', 'nosam'))


class MomentTests(unittest.TestCase):
    """The still moments of the two recordings, replayed from their signal traces."""

    def replay(self, name):
        picker = MomentPicker(SceneMonitor())
        events = []
        for row in read_trace(FIXTURES / f'trace_{name}.csv'):
            thumb = decode_thumb(row['thumb'])
            event = picker.step(row['time_s'], thumb, row['motion'], row['markers_id0'])
            if event:
                events.append((event, row['frame']))
        return events, picker.board

    def test_each_recording_gives_four_photos_and_one_board_end(self):
        for name in ('223831', '223948'):
            with self.subTest(recording=name):
                events, board = self.replay(name)
                kinds = [kind for kind, _ in events]
                self.assertEqual(kinds.count('capture'), 4, events)
                self.assertEqual(kinds.count('cycle_end'), 1, events)
                self.assertEqual(kinds[-1], 'cycle_end')
                self.assertEqual(board, 2)


if __name__ == '__main__':
    unittest.main()
