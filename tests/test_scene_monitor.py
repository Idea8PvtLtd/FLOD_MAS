"""Scene mode: the still-and-changed rule (auto_trigger/scene_monitor.py)."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_trigger.scene_monitor import SceneMonitor, changed_share, scene_signals

CALM = 0.001      # share of the picture changing while everything is at rest
BUSY = 0.30       # ... while a hand or the board moves


def picture(cloths=0):
    """A small grey 'board' with a number of bright cloth pieces on it."""
    small = np.full((48, 64), 90, np.uint8)
    for index in range(cloths):
        small[8:40, 4 + index * 12:14 + index * 12] = 200
    return small


class Trace:
    """Feeds the monitor frame by frame at ten pictures a second and notes the events."""

    def __init__(self, monitor=None):
        self.monitor = monitor or SceneMonitor()
        self.frame = 0
        self.events = []       # (seconds, event)

    @property
    def now(self):
        return self.frame / 10

    def play(self, seconds, motion=CALM, cloths=1, bare=False, answer='accept'):
        """`answer`: what the caller does with an offered photo ('accept', 'reject', None)."""
        got = []
        for _ in range(int(round(seconds * 10))):
            event = self.monitor.update(motion, bare, picture(cloths), self.now)
            if event:
                self.events.append((self.now, event))
                got.append(event)
                if event == 'capture' and answer == 'accept':
                    self.monitor.accept()
                elif event == 'capture' and answer == 'reject':
                    self.monitor.reject(self.now)
            self.frame += 1
        return got


class SceneMonitorTests(unittest.TestCase):
    def test_still_picture_is_offered_once(self):
        trace = Trace()
        self.assertEqual(trace.play(0.9), [])
        self.assertEqual(trace.monitor.state, SceneMonitor.SETTLING)
        self.assertEqual(trace.play(5.0), ['capture'])
        self.assertEqual(trace.events, [(1.0, 'capture')])
        self.assertEqual(trace.monitor.state, SceneMonitor.STILL)
        self.assertEqual(trace.monitor.photos, 1)

    def test_no_second_photo_when_nothing_changed(self):
        trace = Trace()
        trace.play(3.0)
        trace.play(1.0, motion=BUSY)                 # a hand passes, lays nothing
        self.assertEqual(trace.monitor.state, SceneMonitor.MOVING)
        self.assertEqual(trace.play(5.0), [])
        self.assertEqual(trace.monitor.state, SceneMonitor.STILL)
        self.assertLess(trace.monitor.last_change, 0.05)
        self.assertEqual(len(trace.events), 1)

    def test_burst_then_calm_offers_the_changed_picture(self):
        trace = Trace()
        trace.play(3.0, cloths=1)
        trace.play(1.5, motion=BUSY, cloths=2)
        self.assertEqual(trace.play(0.9, cloths=2), [])
        self.assertEqual(trace.play(3.0, cloths=2), ['capture'])
        self.assertEqual(trace.events[-1], (5.5, 'capture'))
        self.assertGreater(trace.monitor.photos, 1)

    def test_change_too_small_is_not_a_new_photo(self):
        trace = Trace(SceneMonitor(change=0.5))      # a cloth is about a tenth of the picture
        trace.play(3.0, cloths=1)
        trace.play(1.0, motion=BUSY, cloths=2)
        self.assertEqual(trace.play(3.0, cloths=2), [])
        self.assertAlmostEqual(trace.monitor.last_change, 320 / (48 * 64), places=3)

    def test_motion_while_settling_restarts_the_wait(self):
        trace = Trace()
        trace.play(0.7)
        trace.play(0.1, motion=BUSY)                 # frame at 0.7 s
        self.assertEqual(trace.monitor.state, SceneMonitor.MOVING)
        self.assertEqual(trace.play(1.0), [])        # calm 0.8 .. 1.7: only 0.9 s of it
        self.assertEqual(trace.play(1.0), ['capture'])
        self.assertEqual(trace.events, [(1.8, 'capture')])

    def test_motion_exactly_at_the_limit_counts_as_calm(self):
        trace = Trace(SceneMonitor(still_motion=0.01))
        self.assertEqual(trace.play(2.0, motion=0.01), ['capture'])
        trace = Trace(SceneMonitor(still_motion=0.01))
        self.assertEqual(trace.play(2.0, motion=0.0101), [])

    def test_veto_is_offered_again_after_the_wait(self):
        trace = Trace()
        self.assertEqual(trace.play(1.1, answer='reject'), ['capture'])
        self.assertEqual(trace.monitor.state, SceneMonitor.VETOED)
        self.assertEqual(trace.monitor.photos, 0)
        self.assertEqual(trace.play(0.9, answer='accept'), [])     # 1.1 .. 1.9
        self.assertEqual(trace.play(0.5, answer='accept'), ['capture'])
        self.assertEqual(trace.events, [(1.0, 'capture'), (2.0, 'capture')])
        self.assertEqual(trace.monitor.state, SceneMonitor.STILL)
        self.assertEqual(trace.monitor.photos, 1)
        self.assertEqual(trace.play(5.0), [])

    def test_veto_keeps_the_key_photo(self):
        trace = Trace()
        trace.play(2.0, cloths=1)
        trace.play(1.0, motion=BUSY, cloths=2)
        self.assertEqual(trace.play(1.5, cloths=2, answer='reject'), ['capture'])
        trace.play(1.0, motion=BUSY, cloths=1)       # the cloth is taken off again
        self.assertEqual(trace.play(5.0, cloths=1), [])   # same as the key photo: nothing new
        self.assertEqual(trace.monitor.photos, 1)

    def test_veto_then_motion_follows_the_normal_rule_again(self):
        trace = Trace()
        trace.play(1.1, answer='reject')             # offered at 1.0, refused
        trace.play(0.5, motion=BUSY)                 # 1.1 .. 1.5
        self.assertEqual(trace.monitor.state, SceneMonitor.MOVING)
        self.assertEqual(trace.monitor.retries, 0)
        self.assertEqual(trace.play(3.0), ['capture'])     # calm from 1.6
        self.assertEqual(trace.events[-1], (2.6, 'capture'))

    def test_retries_are_limited(self):
        trace = Trace(SceneMonitor(veto_retries=3, veto_retry_seconds=1.0))
        self.assertEqual(trace.play(20.0, answer='reject'), ['capture'] * 4)
        self.assertEqual([time for time, _ in trace.events], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(trace.monitor.state, SceneMonitor.VETOED)
        # movement gives a fresh start: one offer and three more tries
        trace.play(0.5, motion=BUSY)
        self.assertEqual(trace.play(20.0, answer='reject'), ['capture'] * 4)

    def test_no_retries_when_set_to_zero(self):
        trace = Trace(SceneMonitor(veto_retries=0))
        self.assertEqual(trace.play(10.0, answer='reject'), ['capture'])

    def test_bare_bed_at_the_start_gives_nothing(self):
        trace = Trace()
        self.assertEqual(trace.play(0.4, cloths=0, bare=True), [])
        self.assertNotEqual(trace.monitor.state, SceneMonitor.BARE)
        self.assertEqual(trace.play(10.0, cloths=0, bare=True), [])
        self.assertEqual(trace.monitor.state, SceneMonitor.BARE)

    def test_cycle_ends_once_after_an_accepted_photo(self):
        trace = Trace()
        trace.play(1.0, cloths=0, bare=True)
        trace.play(1.0, motion=BUSY, cloths=1)       # the board slides in
        self.assertEqual(trace.play(3.0, cloths=1), ['capture'])
        trace.play(1.0, motion=BUSY, cloths=1)       # the board is pulled out
        self.assertEqual(trace.play(0.5, motion=BUSY, cloths=0, bare=True), [])
        self.assertEqual(trace.play(5.0, cloths=0, bare=True), ['cycle_end'])
        self.assertEqual(trace.monitor.state, SceneMonitor.BARE)
        self.assertEqual(trace.monitor.photos, 0)
        # the next board starts a new cycle: its first rest is a photo whatever it shows
        trace.play(1.0, motion=BUSY, cloths=1)
        self.assertEqual(trace.play(3.0, cloths=1), ['capture'])
        trace.play(1.0, cloths=0, bare=True)
        self.assertEqual([event for _, event in trace.events],
                         ['capture', 'cycle_end', 'capture', 'cycle_end'])

    def test_no_cycle_end_without_an_accepted_photo(self):
        trace = Trace()
        trace.play(1.0, motion=BUSY)
        self.assertEqual(trace.play(1.5, answer='reject'), ['capture'])
        self.assertEqual(trace.play(3.0, cloths=0, bare=True), [])
        self.assertEqual(trace.monitor.state, SceneMonitor.BARE)

    def test_flickering_bare_reading_never_gives_a_photo_of_the_bed(self):
        trace = Trace()
        trace.play(1.0, cloths=0, bare=True)
        for _ in range(20):                          # markers lost for 0.3 s now and then
            self.assertEqual(trace.play(0.3, cloths=0, bare=False), [])
            self.assertEqual(trace.play(0.3, cloths=0, bare=True), [])

    def test_short_bare_reading_does_not_end_the_cycle(self):
        trace = Trace()
        trace.play(2.0)
        self.assertEqual(trace.play(0.4, bare=True), [])     # shorter than bare_seconds
        self.assertEqual(trace.play(5.0), [])                # still the same key photo
        self.assertEqual(trace.monitor.photos, 1)

    def test_reset(self):
        trace = Trace()
        trace.play(2.0)
        self.assertEqual(trace.monitor.photos, 1)
        trace.monitor.reset()
        self.assertEqual(trace.monitor.state, SceneMonitor.MOVING)
        self.assertEqual(trace.monitor.photos, 0)
        self.assertFalse(trace.monitor.waiting)
        # no key photo and no cycle any more: the same picture is a photo again, and a
        # bare bed straight after a reset is no cycle end
        self.assertEqual(trace.play(2.0), ['capture'])
        trace.monitor.reset()
        self.assertEqual(trace.play(3.0, cloths=0, bare=True), [])

    def test_only_one_candidate_at_a_time(self):
        trace = Trace()
        self.assertEqual(trace.play(10.0, answer=None), ['capture'])
        self.assertTrue(trace.monitor.waiting)
        self.assertEqual(trace.monitor.photos, 0)
        # movement drops the unanswered candidate; a late answer is then refused
        trace.play(0.5, motion=BUSY)
        self.assertFalse(trace.monitor.waiting)
        self.assertFalse(trace.monitor.accept())
        self.assertFalse(trace.monitor.reject(trace.now))
        self.assertEqual(trace.monitor.photos, 0)
        self.assertEqual(trace.monitor.state, SceneMonitor.MOVING)
        self.assertEqual(trace.play(3.0), ['capture'])

    def test_answers_without_a_candidate_do_nothing(self):
        monitor = SceneMonitor()
        self.assertFalse(monitor.accept())
        self.assertFalse(monitor.reject(0.0))
        self.assertEqual(monitor.state, SceneMonitor.MOVING)
        trace = Trace(monitor)
        trace.play(2.0)
        self.assertFalse(monitor.accept())           # already answered
        self.assertEqual(monitor.photos, 1)

    def test_states_and_describe(self):
        self.assertEqual(SceneMonitor.STATES,
                         ('BARE', 'MOVING', 'SETTLING', 'STILL', 'VETOED'))
        trace = Trace()
        seen = {trace.monitor.state}
        for seconds, kind in ((0.5, {}), (2.0, {}), (0.5, {'motion': BUSY}),
                              (2.0, {'cloths': 0, 'bare': True})):
            trace.play(seconds, **kind)
            seen.add(trace.monitor.state)
            self.assertIn(trace.monitor.state, trace.monitor.describe())
        self.assertEqual(seen, {'MOVING', 'SETTLING', 'STILL', 'BARE'})
        trace = Trace()
        trace.play(1.5, answer='reject')
        text = trace.monitor.describe()
        self.assertIn('VETOED', text)
        self.assertIn('calm 1.4s', text)
        self.assertNotIn('\n', text)

    def test_from_settings_without_the_scene_section(self):
        monitor = SceneMonitor.from_settings({})
        self.assertAlmostEqual(monitor.still_motion, 0.01)
        self.assertAlmostEqual(monitor.still_seconds, 1.0)
        self.assertAlmostEqual(monitor.change, 0.05)
        self.assertAlmostEqual(monitor.veto_retry_seconds, 1.0)
        self.assertEqual(monitor.veto_retries, 3)
        self.assertAlmostEqual(monitor.bare_seconds, 0.5)
        plain = SceneMonitor()
        for name in ('still_motion', 'still_seconds', 'change', 'veto_retry_seconds',
                     'veto_retries', 'bare_seconds', 'level'):
            self.assertEqual(getattr(monitor, name), getattr(plain, name), name)

    def test_from_settings_with_the_scene_section(self):
        monitor = SceneMonitor.from_settings({
            'marker_confirm_seconds': 0.3,
            'scene': {'still_motion_percent': 2.5, 'still_seconds': 1.4,
                      'change_percent': 8, 'veto_retry_seconds': 0.6, 'veto_retries': 5}})
        self.assertAlmostEqual(monitor.still_motion, 0.025)
        self.assertAlmostEqual(monitor.still_seconds, 1.4)
        self.assertAlmostEqual(monitor.change, 0.08)
        self.assertAlmostEqual(monitor.veto_retry_seconds, 0.6)
        self.assertEqual(monitor.veto_retries, 5)
        self.assertAlmostEqual(monitor.bare_seconds, 0.3)
        partial = SceneMonitor.from_settings({'scene': {'still_seconds': 2}})
        self.assertAlmostEqual(partial.still_seconds, 2.0)
        self.assertAlmostEqual(partial.change, 0.05)


class SceneSignalTests(unittest.TestCase):
    def frame(self, value=100):
        frame = np.full((480, 640, 3), value, np.uint8)
        frame[100:300, 200:400] = (40, 180, 220)
        return frame

    def test_small_picture_shape_and_type(self):
        small, motion = scene_signals(self.frame(), width=320)
        self.assertEqual(small.shape, (240, 320))
        self.assertEqual(small.dtype, np.uint8)
        self.assertEqual(motion, 0.0)
        small, _ = scene_signals(self.frame(), width=100)        # every 6th pixel
        self.assertEqual(small.shape, (80, 107))
        small, _ = scene_signals(np.zeros((30, 40, 3), np.uint8), width=320)
        self.assertEqual(small.shape, (30, 40))                  # never enlarged
        small, _ = scene_signals(np.zeros((480, 640), np.uint8))  # a grey frame works too
        self.assertEqual(small.shape, (240, 320))

    def test_equal_frames_show_no_motion(self):
        first, _ = scene_signals(self.frame())
        second, motion = scene_signals(self.frame(), first)
        self.assertEqual(motion, 0.0)
        self.assertTrue(np.array_equal(first, second))

    def test_motion_is_the_share_that_changed(self):
        first, _ = scene_signals(self.frame())
        moved = self.frame()
        moved[:, :320] = 250                                     # the left half changes
        _, motion = scene_signals(moved, first)
        self.assertAlmostEqual(motion, 0.5, delta=0.03)
        dimmer = self.frame(105)                                 # under the level: no motion
        dimmer[100:300, 200:400] = (45, 185, 225)
        self.assertEqual(scene_signals(dimmer, first)[1], 0.0)
        self.assertGreater(scene_signals(dimmer, first, level=2)[1], 0.9)

    def test_previous_of_another_shape_is_no_motion(self):
        other = np.zeros((10, 10), np.uint8)
        small, motion = scene_signals(self.frame(), other)
        self.assertEqual(motion, 0.0)
        self.assertEqual(small.shape, (240, 320))

    def test_changed_share_ignores_a_brightness_offset(self):
        key = picture(2)
        self.assertEqual(changed_share(key, key), 0.0)
        brighter = (key.astype(np.int16) + 30).astype(np.uint8)
        self.assertEqual(changed_share(brighter, key), 0.0)
        self.assertEqual(changed_share(key, brighter), 0.0)

    def test_changed_share_measures_a_new_cloth(self):
        share = changed_share(picture(3), picture(2))
        self.assertAlmostEqual(share, 320 / (48 * 64), places=3)
        brighter = (picture(3).astype(np.int16) + 20).astype(np.uint8)
        self.assertAlmostEqual(changed_share(brighter, picture(2)), share, places=3)
        self.assertEqual(changed_share(picture(3), picture(2), level=200), 0.0)

    def test_changed_share_of_unlike_pictures_is_everything(self):
        self.assertEqual(changed_share(picture(1), np.zeros((5, 5), np.uint8)), 1.0)
        self.assertEqual(changed_share(picture(1), None), 1.0)


if __name__ == '__main__':
    unittest.main()
