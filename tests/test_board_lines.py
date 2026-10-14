"""Scene mode: a glue line is told apart by where it runs, not by its place from the left."""
import unittest
import numpy as np
from inspection.board_lines import BoardLines, course, length_of


def wave(x, top=100, bottom=2900, shift=(0, 0)):
    """A source line as the engine gives it: one point per row, a gentle wave."""
    y = np.arange(top, bottom, dtype=float)
    points = np.column_stack((x + 30 * np.sin((y - 100) * 2 * np.pi / 2800), y)) + shift
    return {'original_xy': points.tolist()}


def read(board, lines):
    """Measure what a photo owes, as the pipeline does; returns the ids measured."""
    picks = board.owed(lines)
    for _, line_id in picks:
        board.record(line_id, {'line_id': line_id, 'status': 'PASS', 'new': True})
    return [line_id for _, line_id in picks]


class CourseTests(unittest.TestCase):
    def test_a_course_is_evenly_spaced_along_the_line(self):
        points = course(wave(500)['original_xy'])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertLess(np.ptp(steps), 0.5)
        self.assertAlmostEqual(float(steps.mean()), 10.0, delta=0.1)
        self.assertAlmostEqual(length_of(points), 2800, delta=40)


class BoardLinesTests(unittest.TestCase):
    def test_the_first_photo_numbers_its_lines_in_the_order_given(self):
        board = BoardLines()
        self.assertEqual(board.owed([wave(500), wave(1500)]), [(0, 1), (1, 2)])

    def test_the_same_line_shifted_twenty_pixels_is_the_same_line(self):
        board = BoardLines()
        self.assertEqual(read(board, [wave(500)]), [1])
        self.assertEqual(board.owed([wave(520)]), [])
        self.assertEqual(board.match([wave(520)]), [1])

    def test_a_line_three_hundred_pixels_away_is_a_new_line(self):
        board = BoardLines()
        read(board, [wave(500)])
        self.assertEqual(board.owed([wave(500), wave(800)]), [(1, 2)])

    def test_a_new_line_on_the_left_leaves_the_old_numbers_alone(self):
        board = BoardLines()
        self.assertEqual(read(board, [wave(2500), wave(4000)]), [1, 2])
        # The next cloth goes to the LEFT, so the photo's left-to-right order shifts.
        lines = [wave(1900), wave(2510), wave(4005)]
        self.assertEqual(board.match(lines), [None, 1, 2])
        self.assertEqual(read(board, lines), [3])
        lines = [wave(480), wave(1900), wave(2510), wave(4005)]
        self.assertEqual(read(board, lines), [4])
        self.assertEqual(board.match(lines), [4, 3, 1, 2])
        self.assertEqual([entry['line_id'] for entry in board.entries], [1, 2, 3, 4])
        # Found in another order, every line is still itself and nothing is owed.
        self.assertEqual(board.owed(lines[::-1]), [])
        self.assertEqual(board.match(lines[::-1]), [2, 1, 3, 4])

    def test_a_piece_of_a_known_line_is_that_line_and_never_a_new_one(self):
        board = BoardLines()
        read(board, [wave(500), wave(1500)])
        # 40 % of line 1 still shows, the rest is under a later cloth.
        piece = wave(500, top=1000, bottom=2120)
        self.assertEqual(board.match([piece, wave(1500)]), [1, 2])
        self.assertEqual(board.owed([piece, wave(1500)]), [])
        # In two pieces, 41 % in all.
        pieces = [wave(500, top=100, bottom=700), wave(500, top=2350, bottom=2900)]
        self.assertEqual(board.match(pieces), [1, 1])
        self.assertEqual(board.owed(pieces + [wave(1500)]), [])
        self.assertEqual([entry['line_id'] for entry in board.entries], [1, 2])

    def test_a_whole_line_matches_a_line_first_seen_as_a_piece(self):
        board = BoardLines(min_new_length_ratio=0.)
        read(board, [wave(500, top=1000, bottom=2500)])
        self.assertEqual(board.owed([wave(500)]), [])

    def test_lines_that_only_meet_end_to_end_are_two_lines(self):
        board = BoardLines(min_new_length_ratio=0.)
        read(board, [wave(500, top=100, bottom=1500)])
        self.assertEqual(board.owed([wave(500, top=1400, bottom=2900)]), [(0, 2)])

    def test_a_nudged_board_is_still_the_same_board(self):
        board = BoardLines()
        read(board, [wave(500), wave(1500), wave(2500)])
        nudged = [wave(x, shift=(30, 0)) for x in (500, 1500, 2500)]
        self.assertEqual(board.owed(nudged), [])
        self.assertEqual(board.match(nudged), [1, 2, 3])
        self.assertAlmostEqual(board.last_offset[0], -30, delta=2)

    def test_the_common_nudge_is_taken_out_before_a_line_further_off_is_judged(self):
        board = BoardLines()
        read(board, [wave(500), wave(1500), wave(2500)])
        # Two lines 50 px off, the third 75: beyond the limit alone, within it
        # once the 50 px every line shares is removed.
        nudged = [wave(550), wave(1550), wave(2575)]
        self.assertEqual(board.owed(nudged), [])
        self.assertEqual(board.match(nudged), [1, 2, 3])

    def test_a_short_stray_line_that_matches_nothing_is_not_a_new_line(self):
        board = BoardLines()
        read(board, [wave(500)])
        stray = wave(1500, top=100, bottom=900)
        self.assertEqual(board.owed([wave(500), stray]), [])
        self.assertEqual(board.last_match[1], (1, None, 'short'))
        # Compared with the longest line KNOWN, also when only a piece of it shows.
        self.assertEqual(board.owed([wave(500, top=100, bottom=700), stray]), [])
        self.assertEqual(BoardLines(min_new_length_ratio=0.).owed([wave(500), stray]), [(0, 1), (1, 2)])

    def test_a_failed_line_is_owed_again_and_a_good_reading_replaces_the_failure(self):
        board = BoardLines()
        self.assertEqual(board.owed([wave(500), wave(1500)]), [(0, 1), (1, 2)])
        board.record(1, {'line_id': 1, 'status': 'PASS', 'new': True})
        board.fail(2, {'line_id': 2, 'status': 'FAIL', 'error': 'no adhesive', 'new': True})
        self.assertEqual([e['status'] for e in board.entries], ['PASS', 'FAIL'])
        self.assertEqual(board.owed([wave(500), wave(1500)]), [(1, 2)])
        board.record(2, {'line_id': 2, 'status': 'PASS', 'new': True})
        self.assertEqual([e['status'] for e in board.entries], ['PASS', 'PASS'])
        self.assertEqual(board.owed([wave(500), wave(1500)]), [])

    def test_a_failed_line_of_which_only_a_piece_shows_is_not_tried_on_the_piece(self):
        board = BoardLines()
        board.owed([wave(500), wave(1500)])
        board.record(1, {'line_id': 1, 'status': 'PASS', 'new': True})
        board.fail(2, {'line_id': 2, 'status': 'FAIL', 'new': True})
        self.assertEqual(board.owed([wave(500), wave(1500, top=100, bottom=1800)]), [])

    def test_a_line_whose_try_was_never_reported_is_owed_again(self):
        board = BoardLines()
        self.assertEqual(board.owed([wave(500)]), [(0, 1)])
        self.assertEqual(board.entries, [])
        self.assertEqual(board.owed([wave(500)]), [(0, 1)])

    def test_the_first_good_reading_is_kept(self):
        board = BoardLines()
        board.owed([wave(500)])
        board.record(1, {'line_id': 1, 'status': 'PASS', 'new': True})
        board.record(1, {'line_id': 1, 'status': 'FAIL', 'new': True})
        board.fail(1, {'line_id': 1, 'status': 'FAIL', 'new': True})
        self.assertEqual(board.entries[0]['status'], 'PASS')
        with self.assertRaises(KeyError):
            board.record(7, {})

    def test_entries_stop_being_new_at_the_next_photo_without_changing_the_old_dict(self):
        board = BoardLines()
        board.owed([wave(500)])
        first = {'line_id': 1, 'status': 'PASS', 'new': True}
        board.record(1, first)
        self.assertTrue(board.entries[0]['new'])
        board.owed([wave(500)])
        self.assertFalse(board.entries[0]['new'])
        self.assertTrue(first['new'])       # what an earlier payload holds is left alone

    def test_reset_forgets_the_board(self):
        board = BoardLines()
        read(board, [wave(500), wave(1500)])
        board.reset()
        self.assertEqual(board.entries, [])
        self.assertEqual(board.owed([wave(1500)]), [(0, 1)])

    def test_sideways_lines_are_told_apart_the_same_way(self):
        def turned(line):
            return {'original_xy': np.asarray(line['original_xy'])[:, ::-1].tolist()}
        board = BoardLines()
        self.assertEqual(read(board, [turned(wave(500)), wave(2000)]), [1, 2])
        self.assertEqual(board.match([wave(2010), turned(wave(515)), turned(wave(900))]), [2, 1, None])

    def test_settings_are_checked(self):
        for arguments in (dict(match_distance_px=0), dict(min_overlap=0), dict(min_overlap=1.5),
                          dict(min_new_length_ratio=1.0)):
            with self.assertRaises(ValueError):
                BoardLines(**arguments)


if __name__ == '__main__':
    unittest.main()
