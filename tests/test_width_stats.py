"""Width statistics: scale from nominal width, empirical limits, scanning for changes."""
import unittest

import numpy as np

from segmentation import width_stats


def good_line(rng, count=900, spread=0.05, memory=30):
    """A good line's departure: correlated noise, as real widths are."""
    noise = np.convolve(rng.normal(0, 1, count + memory), np.ones(memory) / memory, 'valid')[:count]
    return spread * noise / noise.std()


class ScaleTests(unittest.TestCase):
    def test_the_scale_follows_position_and_colour_and_gives_the_nominal_width(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform((0, 0), (4608, 3456), (4000, 2))
        colours = rng.choice(['dark', 'pink'], 4000)
        true = 30.0 * (1 + 0.1 * (xy[:, 0] - 2304) / 2304) * np.where(colours == 'pink', 0.93, 1.0)
        scale = width_stats.fit_scale(xy, true * (1 + rng.normal(0, 0.01, 4000)), colours, (4608, 3456))
        for colour in ('dark', 'pink'):
            chosen = colours == colour
            millimetres = true[chosen] / width_stats.pixels_per_mm(scale, xy[chosen], colour)
            self.assertAlmostEqual(float(millimetres.mean()), 4.0, delta=0.02)
            self.assertLess(float(millimetres.std()), 0.02)
        self.assertGreater(scale['explained'], 0.9)

    def test_a_few_odd_readings_do_not_bend_the_scale(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform((0, 0), (4608, 3456), (2000, 2))
        widths = np.full(2000, 30.0)
        widths[:60] = 55.0
        scale = width_stats.fit_scale(xy, widths, np.array(['dark'] * 2000), (4608, 3456))
        self.assertAlmostEqual(float(np.median(width_stats.pixels_per_mm(scale, xy, 'dark'))), 7.5, delta=0.05)


class TrendTests(unittest.TestCase):
    def test_the_gradual_course_is_followed_and_a_local_change_is_left_standing(self):
        along = np.linspace(-1, 1, 800)
        width = 4.0 + 0.2 * along + 0.1 * along ** 2
        width[300:340] += 0.4
        departure = width - width_stats.slow_trend(width)
        self.assertGreater(float(departure[300:340].mean()), 0.35)
        self.assertLess(float(np.abs(departure[:250]).max()), 0.03)


class LimitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.sizes = (20, 40, 80, 160)
        self.lines = [good_line(rng) for _ in range(400)]
        self.limits = width_stats.limits_from_good_lines(self.lines[:200], self.sizes)

    def test_about_one_good_line_in_twenty_is_flagged_over_all_lengths_together(self):
        flagged = sum(any(f['level'] == 'significant' for f in width_stats.scan(line, self.limits, self.sizes))
                      for line in self.lines[200:])
        self.assertLess(flagged, 30)              # 5 % of 200 is 10; three times that is not chance
        self.assertGreater(flagged, 1)

    def test_longer_windows_have_tighter_limits(self):
        values = [self.limits['significant'][size] for size in self.sizes]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_a_change_is_found_with_its_place_sign_and_size(self):
        line = self.lines[250].copy()
        line[400:480] -= 0.3
        found = [f for f in width_stats.scan(line, self.limits, self.sizes) if f['level'] == 'significant']
        self.assertTrue(found)
        widest = max(found, key=lambda f: abs(f['change']))
        self.assertLess(widest['change'], -0.2)
        self.assertTrue(widest['first'] < 480 and widest['last'] > 400)


class GlitchTests(unittest.TestCase):
    """Telling a change of the glue from a change of the mask."""

    def test_runs_of_flags_are_found_with_their_ends(self):
        flags = np.zeros(50, bool)
        flags[5:9] = True
        flags[20:40] = True
        self.assertEqual(width_stats.stretches(flags), [(5, 8), (20, 39)])
        self.assertEqual(width_stats.stretches(flags, 10), [(20, 39)])

    def test_a_change_where_the_mask_is_in_question_is_a_glitch(self):
        finding = {'first': 100, 'last': 139, 'change': 0.4}
        doubtful = np.zeros(300, bool)
        doubtful[110:130] = True
        self.assertEqual(width_stats.judge_change(finding, doubtful, np.full(300, 0.4)), 'glitch')

    def test_a_change_the_picture_shows_too_is_confirmed_and_one_it_does_not_is_mask_only(self):
        finding = {'first': 100, 'last': 139, 'change': -0.4}
        calm = np.zeros(300, bool)
        other = np.zeros(300)
        self.assertEqual(width_stats.judge_change(finding, calm, other), 'mask only')
        other[100:140] = -0.3
        self.assertEqual(width_stats.judge_change(finding, calm, other), 'confirmed')
        other[100:140] = 0.3                                   # the other way: not the same change
        self.assertEqual(width_stats.judge_change(finding, calm, other), 'mask only')


if __name__ == '__main__':
    unittest.main()
