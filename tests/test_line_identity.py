"""Measured lines are known by place and look, so removing a fabric hides nothing."""
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from inspection.local import line_identity
from inspection.local.engine import LiveInspection

HEIGHT, WIDTH = 800, 1800


def texture(seed):
    """A lace-like pattern; the same seed is the same physical fabric."""
    noise = np.random.RandomState(seed).normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32)
    blobs = cv2.GaussianBlur(noise, (0, 0), 4)
    return blobs / blobs.std() * 25


def bed(fabrics):
    """A photo and source mask with one fabric (glue edge at x) per (x, seed)."""
    photo = np.full((HEIGHT, WIDTH), 100., np.float32)
    source = np.zeros((HEIGHT, WIDTH), np.uint8)
    y = np.arange(30, 770)
    for x, seed in fabrics:
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        piece = np.zeros_like(source)
        cv2.fillPoly(piece, [np.rint(np.vstack(([x-150, 30], boundary, [x-150, 769]))).astype(np.int32)], 255)
        # The pattern moves with its fabric, as real lace does.
        photo[piece > 0] = 180 + np.roll(texture(seed), x, axis=1)[piece > 0]
        source |= piece
    photo = cv2.cvtColor(np.clip(photo, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return photo, source


def inspector_for(source):
    def adhesive(flat):
        mask = np.zeros(flat.shape[:2], np.uint8)
        mask[35:65, 15:flat.shape[1]-15] = 255
        return mask
    inspector = LiveInspection.__new__(LiveInspection)
    inspector.offset = 100.
    inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
    inspector.glue = SimpleNamespace(width=1024, predict=adhesive)
    return inspector


def measure(fabrics, measured=()):
    photo, source = bed(fabrics)
    return list(inspector_for(source).run_lines(photo, measured=measured))


def edge_x(result):
    return round(float(np.mean(np.asarray(result['fingerprint']['path'])[:, 0])), -1)


class BedChangeTests(unittest.TestCase):
    # Edges 450 px apart: no neighbour lies within the look's margin, as on the
    # real bed. (A neighbour that close can make a line be measured again.)
    BED = [(250, 1), (700, 2), (1150, 3)]

    def setUp(self):
        self.first = measure(self.BED)
        self.memory = [r['fingerprint'] for r in self.first]

    def test_every_line_of_a_new_bed_is_measured(self):
        self.assertEqual([edge_x(r) for r in self.first], [250, 700, 1150])

    def test_the_same_bed_again_measures_nothing(self):
        self.assertEqual(measure(self.BED, self.memory), [])

    def test_a_fabric_taken_off_the_left_does_not_hide_one_added_on_the_right(self):
        # The case seen on 2026-10-16: still three lines, so counting from the
        # left called the new fabric "line 3" and skipped it as measured.
        later = measure([(700, 2), (1150, 3), (1600, 4)], self.memory)
        self.assertEqual([edge_x(r) for r in later], [1600])
        self.assertEqual([r['selected_line'] for r in later], [3])

    def test_a_new_fabric_in_the_place_of_a_measured_one_is_measured(self):
        later = measure([(250, 1), (700, 9), (1150, 3)], self.memory)
        self.assertEqual([edge_x(r) for r in later], [700])

    def test_a_fabric_nudged_a_little_is_still_the_same(self):
        self.assertEqual(measure([(250, 1), (715, 2), (1150, 3)], self.memory), [])


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        photo, _ = bed([(650, 2)])
        self.line = {'original_xy': [[650, y] for y in range(30, 770)]}
        self.print = line_identity.fingerprint(photo, self.line)

    def test_a_line_matches_itself(self):
        self.assertTrue(line_identity.same_line(self.print, self.print))

    def test_a_line_far_away_is_another_line(self):
        far = dict(self.print, path=[[x + 300, y] for x, y in self.print['path']])
        self.assertFalse(line_identity.same_line(self.print, far))

    def test_plain_fabric_is_decided_by_place_alone(self):
        plain = np.full((HEIGHT, WIDTH, 3), 180, np.uint8)
        a = line_identity.fingerprint(plain, self.line)
        self.assertIsNone(line_identity.look_match(a, a))
        self.assertTrue(line_identity.same_line(a, a))
        far = dict(a, path=[[x + 300, y] for x, y in a['path']])
        self.assertFalse(line_identity.same_line(a, far))

    def test_the_same_fabric_matches_when_its_edge_is_detected_longer(self):
        # Mini PC, 2026-10-16: the same fabric's edge ended 23 px further down in
        # the next photo. Its own look then no longer lined up (0.63) and it was
        # measured twice; cut from the stored box it matches.
        photo, _ = bed([(650, 2)])
        longer = line_identity.fingerprint(photo, {'original_xy': [[650, y] for y in range(30, 770)]})
        shorter = line_identity.fingerprint(photo, {'original_xy': [[650, y] for y in range(30, 700)]})
        self.assertTrue(line_identity.already_measured(longer, [shorter], photo))
        self.assertTrue(line_identity.already_measured(shorter, [longer], photo))

    def test_another_fabric_in_the_stored_box_is_not_the_same(self):
        before, _ = bed([(650, 2)])
        after, _ = bed([(650, 9)])
        stored = line_identity.fingerprint(before, self.line)
        self.assertFalse(line_identity.already_measured(line_identity.fingerprint(after, self.line), [stored], after))

    def test_fingerprints_are_plain_json(self):
        import json
        self.assertEqual(json.loads(json.dumps(self.print)), self.print)


if __name__ == '__main__':
    unittest.main()
