"""Scene mode: one photo of the whole board, only the lines not read before are measured."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from inspection.board_lines import BoardLines
from inspection.local.engine import LiveInspection, NoSourceLinesError
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from plane_scale import PlaneScale
from test_local_inspection import synthetic_inspection, CONFIG


def panels(xs, shape=(800, 1500)):
    """A source mask with one panel per x, as ``four_line_inspection`` draws them."""
    source = np.zeros(shape, np.uint8)
    y = np.arange(30, 770)
    for x in xs:
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        cv2.fillPoly(source, [np.rint(np.vstack(([x-150, 30], boundary, [x-150, 769]))).astype(np.int32)], 255)
    return source


def board_inspection(xs=(750, 1000)):
    """A frame and an inspector whose board can be given more panels: ``inspector.show(xs)``."""
    frame = np.full((800, 1500, 3), 100, np.uint8)
    def adhesive(flat):
        h, w = flat.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[35:65, 15:w-15] = 255
        return mask
    inspector = LiveInspection.__new__(LiveInspection)
    inspector.offset = 100.
    state = {'source': panels(xs)}
    inspector.source = SimpleNamespace(predict=lambda photo: state['source'].copy())
    inspector.glue = SimpleNamespace(width=1024, predict=adhesive)
    inspector.show = lambda xs: state.update(source=panels(xs))
    inspector.calls = []
    inspect = inspector._inspect_line
    def counted(photo, source, line, index, *args, **kwargs):
        inspector.calls.append(index)
        return inspect(photo, source, line, index, *args, **kwargs)
    inspector._inspect_line = counted
    return frame, inspector


def fake_overlay():
    """Stands in for inspection/board_overlay.py, which another part of the work owns."""
    module = ModuleType('inspection.board_overlay')
    module.board_overview = lambda image, entries, max_edge=1920: cv2.resize(image, (150, 80))
    def aggregate_verdict(entries):
        statuses = ['FAIL' if e['measurement'] is None else e['status'] for e in entries]
        return ('EMPTY' if not statuses else 'FAIL' if 'FAIL' in statuses
                else 'WARNING' if 'WARNING' in statuses else 'PASS')
    module.aggregate_verdict = aggregate_verdict
    return module


def mean_width(result):
    main = max(result['components'], key=lambda c: c['length_px'])
    return float(np.mean([s['width_px'] for s in main['samples'] if s['valid']]))


class BoardCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (patch.dict(sys.modules, {'inspection.board_overlay': fake_overlay()}),
                        patch('inspection.pipeline.plane_scale.load_frame_scale',
                              return_value=PlaneScale(np.diag([.1, .1, 1]), .1))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = copy.deepcopy(CONFIG)
        self.store = CaptureStore(self.directory.name, self.config['capture_storage'])

    def run_board(self, frame, inspector, board=None, metadata=None, **arguments):
        payloads, errors = [], []
        pipeline = InspectionPipeline(inspector, self.store, self.config)
        result = pipeline.run_board(frame, frame, {'frame_sequence': 7} if metadata is None else metadata,
                                    3, .5, board=board, on_board=payloads.append,
                                    on_error=errors.append, **arguments)
        self.assertIsNone(result)
        self.assertEqual(len(payloads), 1)      # exactly once
        return payloads[0], errors


class RunBoardTests(BoardCase):
    def test_the_first_photo_reads_every_line_as_new(self):
        frame, inspector = board_inspection()
        payload, errors = self.run_board(frame, inspector, BoardLines())
        self.assertEqual(errors, [])
        self.assertEqual(payload['new_lines'], [1, 2])
        self.assertEqual([e['line_id'] for e in payload['lines']], [1, 2])
        self.assertEqual(payload['verdict'], 'PASS')
        self.assertEqual(payload['frame_sequence'], 7)
        self.assertIs(payload['source_image'], frame)
        self.assertEqual(payload['overview'].shape, (80, 150, 3))
        self.assertGreater(payload['seconds'], 0)
        self.assertTrue(Path(payload['path']).is_dir())
        for entry in payload['lines']:
            measurement = entry['measurement']
            self.assertEqual(measurement['selected_line'], entry['line_id'])
            self.assertEqual(measurement['line_count'], 2)
            self.assertEqual(entry['status'], measurement['status'])
            self.assertIsNone(entry['error'])
            self.assertTrue(entry['new'])
            self.assertEqual(entry['photo_sequence'], 7)
            x, y = entry['origin']
            h, w = entry['crop_mask'].shape
            self.assertEqual(entry['crop'].shape, (h, w, 3))
            np.testing.assert_array_equal(entry['crop'], frame[y:y+h, x:x+w])
            self.assertFalse(np.shares_memory(entry['crop'], frame))
            # Room is left round the line as in the single-line view, cut at the photo's edge.
            ys, xs = np.nonzero(entry['crop_mask'])
            self.assertEqual((int(xs.min()), int(ys.min())), (160, 40))
            self.assertEqual(w - 1 - int(xs.max()), 179)
            self.assertEqual(y + h, 800)        # 40 px beyond the lower end would leave the photo
            for name in ('centre', 'outline'):
                points = entry[name]
                self.assertEqual(points.shape[1], 2)
                self.assertTrue(np.all((points >= (x, y)) & (points < (x + w, y + h))))
            # The outline is the mask's, in photo pixels.
            at = np.rint(entry['outline']).astype(int) - (x, y)
            self.assertTrue(np.all(entry['crop_mask'][at[:, 1], at[:, 0]] > 0))
        self.assertLess(payload['lines'][0]['centre'][:, 0].mean(), payload['lines'][1]['centre'][:, 0].mean())

    def test_the_same_photo_again_measures_nothing_and_keeps_the_readings(self):
        frame, inspector = board_inspection()
        board = BoardLines()
        first, _ = self.run_board(frame, inspector, board)
        self.assertEqual(inspector.calls, [1, 2])
        second, errors = self.run_board(frame, inspector, board)
        self.assertEqual(inspector.calls, [1, 2])       # nothing measured the second time
        self.assertEqual((second['new_lines'], errors), ([], []))
        self.assertEqual([e['line_id'] for e in second['lines']], [1, 2])
        for before, after in zip(first['lines'], second['lines']):
            self.assertIs(after['measurement'], before['measurement'])
            self.assertIs(after['crop'], before['crop'])
            self.assertFalse(after['new'])
            self.assertTrue(before['new'])      # the payload already shown is not changed
        self.assertNotEqual(first['path'], second['path'])
        self.assertEqual(json.loads((Path(second['path'])/'result.json').read_text())['lines'], [])

    def test_a_photo_with_one_more_line_measures_only_that_line(self):
        frame, inspector = board_inspection((750, 1000))
        board = BoardLines()
        self.run_board(frame, inspector, board)
        inspector.show((500, 750, 1000))        # the next cloth goes on the LEFT
        payload, _ = self.run_board(frame, inspector, board)
        self.assertEqual(inspector.calls, [1, 2, 3])
        self.assertEqual(payload['new_lines'], [3])
        self.assertEqual([(e['line_id'], e['new']) for e in payload['lines']], [(1, False), (2, False), (3, True)])
        centres = [e['centre'][:, 0].mean() for e in payload['lines']]
        self.assertLess(centres[2], centres[0])
        self.assertLess(centres[0], centres[1])
        self.assertEqual(payload['lines'][2]['measurement']['line_count'], 3)
        self.assertEqual(payload['lines'][0]['measurement']['line_count'], 2)
        inspector.show((250, 500, 750, 1000))
        payload, _ = self.run_board(frame, inspector, board)
        self.assertEqual(payload['new_lines'], [4])
        self.assertEqual(inspector.calls, [1, 2, 3, 4])

    def test_a_line_that_failed_is_owed_again(self):
        frame, inspector = board_inspection()
        counted = inspector._inspect_line
        def fail_second(photo, source, line, index, *args, **kwargs):
            if index == 2:
                raise ValueError('empty adhesive')
            return counted(photo, source, line, index, *args, **kwargs)
        inspector._inspect_line = fail_second
        board = BoardLines()
        payload, errors = self.run_board(frame, inspector, board)
        self.assertEqual(errors, ['Line 2: empty adhesive'])
        self.assertEqual(payload['new_lines'], [1])
        failed = payload['lines'][1]
        self.assertEqual((failed['line_id'], failed['status'], failed['measurement']), (2, 'FAIL', None))
        self.assertEqual(failed['error'], 'empty adhesive')
        self.assertEqual((failed['crop'], failed['crop_mask'], failed['outline'], failed['origin']),
                         (None, None, None, (0, 0)))
        self.assertEqual(failed['centre'].shape[1], 2)
        self.assertGreater(failed['centre'][:, 0].mean(), 900)
        self.assertEqual(payload['verdict'], 'FAIL')
        record = json.loads((Path(payload['path'])/'result.json').read_text())
        self.assertEqual((record['status'], record['errors']), ('complete_with_errors', ['Line 2: empty adhesive']))
        self.assertFalse((Path(payload['path'])/'line_2').exists())
        # The next photo: only the failed line is tried, and its good reading replaces the failure.
        inspector._inspect_line = counted
        inspector.calls.clear()
        payload, errors = self.run_board(frame, inspector, board)
        self.assertEqual((inspector.calls, payload['new_lines'], errors), ([2], [2], []))
        self.assertEqual([(e['line_id'], e['status'], e['new']) for e in payload['lines']],
                         [(1, 'PASS', False), (2, 'PASS', True)])
        self.assertEqual(payload['verdict'], 'PASS')

    def test_a_bare_photo_is_an_empty_board_and_not_an_error(self):
        frame, inspector = board_inspection(())
        with self.assertRaises(NoSourceLinesError):
            inspector.find_lines(frame)
        payload, errors = self.run_board(frame, inspector, BoardLines())
        self.assertEqual((payload['lines'], payload['new_lines'], payload['verdict'], errors),
                         ([], [], 'EMPTY', []))
        record = json.loads((Path(payload['path'])/'result.json').read_text())
        self.assertEqual((record['status'], record['lines'], record['known_lines']), ('complete', [], []))
        self.assertTrue((Path(payload['path'])/'board.jpg').is_file())

    def test_a_bare_photo_after_lines_were_read_still_lists_them(self):
        frame, inspector = board_inspection()
        board = BoardLines()
        self.run_board(frame, inspector, board)
        inspector.show(())
        payload, _ = self.run_board(frame, inspector, board)
        self.assertEqual(([e['line_id'] for e in payload['lines']], payload['new_lines'], payload['verdict']),
                         ([1, 2], [], 'PASS'))

    def test_without_a_board_every_line_of_the_photo_is_measured(self):
        frame, inspector = board_inspection()
        for _ in range(2):
            payload, _ = self.run_board(frame, inspector)
            self.assertEqual(payload['new_lines'], [1, 2])
        self.assertEqual(inspector.calls, [1, 2, 1, 2])

    def test_another_error_is_raised_and_recorded(self):
        frame, inspector = board_inspection()
        inspector.source = SimpleNamespace(predict=lambda photo: (_ for _ in ()).throw(RuntimeError('model broke')))
        pipeline = InspectionPipeline(inspector, self.store, self.config)
        shown = []
        with self.assertRaises(RuntimeError), self.assertLogs('flod.inspection', 'ERROR'):
            pipeline.run_board(frame, frame, {}, 3, .5, on_board=shown.append)
        self.assertEqual(shown, [])
        record = json.loads((self.store._sets()[0]/'result.json').read_text())
        self.assertEqual((record['status'], record['error']), ('error', 'model broke'))

    def test_a_line_below_the_minimum_length_is_not_kept(self):
        frame, inspector = board_inspection()
        self.config['inspection']['min_line_length_mm'] = 500
        payload, errors = self.run_board(frame, inspector, BoardLines())
        self.assertEqual(payload['new_lines'], [])
        self.assertEqual(len(errors), 2)
        self.assertIn('shorter than the 500 mm minimum', errors[0])
        self.assertTrue(all(e['measurement'] is None for e in payload['lines']))

    def test_each_measured_line_is_logged(self):
        frame, inspector = board_inspection()
        with self.assertLogs('flod.inspection', 'INFO') as logs:
            self.run_board(frame, inspector, BoardLines())
        text = '\n'.join(logs.output)
        self.assertIn('Frame 7, line 1 of 2: PASS', text)
        self.assertIn('Frame 7, line 2 of 2: PASS', text)
        self.assertIn('Frame 7 board: 2 line(s) found, 2 read now (1, 2), 0 failed, 2 known', text)


class BoardStorageTests(BoardCase):
    def test_what_a_board_photo_leaves_on_disk(self):
        frame, inspector = board_inspection()
        original = frame.copy()
        original[:10] = 7
        seen = {}
        pipeline = InspectionPipeline(inspector, self.store, self.config)
        def shown(payload):
            path = Path(payload['path'])
            # The result is shown before the disk is written to.
            seen.update(payload=payload, files=sorted(p.name for p in path.iterdir()))
        pipeline.run_board(original, frame, {'frame_sequence': 3}, 3, .5, board=BoardLines(), on_board=shown)
        self.assertEqual(seen['files'], ['.inspection-set', 'capture.json'])
        payload, path = seen['payload'], Path(seen['payload']['path'])
        self.assertEqual(sorted(p.name for p in path.iterdir()),
                         ['.inspection-set', 'board.jpg', 'capture.json', 'line_1', 'line_2',
                          'original.png', 'result.json', 'undistorted.png'])
        np.testing.assert_array_equal(cv2.imread(str(path/'undistorted.png')), frame)
        np.testing.assert_array_equal(cv2.imread(str(path/'original.png')), original)
        self.assertEqual(cv2.imread(str(path/'board.jpg')).shape, (80, 150, 3))
        self.assertEqual(json.loads((path/'capture.json').read_text()), {'frame_sequence': 3})
        record = json.loads((path/'result.json').read_text())
        self.assertEqual(record['method'], 'local_onnx_board')
        self.assertEqual((record['status'], record['known_lines'], record['new_lines']), ('complete', [1, 2], [1, 2]))
        self.assertEqual((record['errors'], record['warnings'], record['frame_size']), ([], [], [1500, 800]))
        self.assertEqual([m['selected_line'] for m in record['lines']], [1, 2])
        for entry in payload['lines']:
            line_path = path/f"line_{entry['line_id']}"
            self.assertEqual(sorted(p.name for p in line_path.iterdir()), ['mask.png', 'result.json'])
            np.testing.assert_array_equal(cv2.imread(str(line_path/'mask.png'), cv2.IMREAD_GRAYSCALE),
                                          entry['crop_mask'])
            saved = json.loads((line_path/'result.json').read_text())
            self.assertEqual((saved['status'], saved['line_id'], saved['origin']),
                             ('complete', entry['line_id'], list(entry['origin'])))
            self.assertEqual(saved['measurement']['mean_width_px'], entry['measurement']['mean_width_px'])

    def test_a_board_set_is_pruned_like_any_other(self):
        frame, inspector = board_inspection()
        self.config['capture_storage']['max_sets'] = 1
        self.store = CaptureStore(self.directory.name, self.config['capture_storage'])
        first, _ = self.run_board(frame, inspector)
        second, _ = self.run_board(frame, inspector)
        self.assertFalse(Path(first['path']).exists())
        self.assertEqual([p.name for p in self.store._sets()], [Path(second['path']).name])

    def test_a_set_begun_by_the_caller_keeps_its_own_sources(self):
        frame, inspector = board_inspection()
        path = self.store.begin(frame, frame, {'frame_sequence': 1})
        with patch.object(self.store, 'save_sources', side_effect=AssertionError('written twice')):
            payload, _ = self.run_board(frame, inspector, capture_path=path, metadata={'frame_sequence': 2})
        self.assertEqual(payload['path'], str(path))
        self.assertEqual(json.loads((path/'capture.json').read_text()), {'frame_sequence': 2})
        self.assertTrue((path/'line_1'/'mask.png').is_file())

    def test_an_unfinished_deferred_set_is_never_pruned(self):
        self.config['capture_storage']['max_sets'] = 1
        store = CaptureStore(self.directory.name, self.config['capture_storage'])
        path = store.begin_deferred({'a': 1})
        self.assertEqual(sorted(p.name for p in path.iterdir()), ['.inspection-set', 'capture.json'])
        with self.assertRaises(OSError):
            store.begin_deferred({})
        store.save_sources(path, None, np.zeros((4, 4, 3), np.uint8))
        self.assertFalse((path/'original.png').exists())
        self.assertTrue((path/'undistorted.png').is_file())

    def test_the_hand_path_is_unchanged_and_agrees_with_the_board_path(self):
        frame, inspector, _ = synthetic_inspection()
        pipeline = InspectionPipeline(inspector, self.store, self.config)
        results = []
        pipeline.run_batch(frame, frame, {}, 3, .5, on_result=results.append)
        self.assertEqual([r['measurement']['selected_line'] for r in results], [1, 2])
        for result in results:
            path = Path(result['path'])
            self.assertEqual(sorted(p.name for p in path.iterdir()),
                             ['mask.png', 'overlay.png', 'overview.jpg', 'result.json'])
            self.assertEqual(result['mask'].shape, frame.shape[:2])
            self.assertEqual(result['overlay'].shape, frame.shape)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()),
                         ['.inspection-set', 'capture.json', 'line_1', 'line_2', 'original.png',
                          'result.json', 'undistorted.png'])
        self.assertEqual(json.loads((path.parent/'result.json').read_text())['method'], 'local_onnx_lines')
        payload, _ = self.run_board(frame, inspector)
        for result, entry in zip(results, payload['lines']):
            hand, board = copy.deepcopy(result['measurement']), copy.deepcopy(entry['measurement'])
            for measurement in (hand, board):
                for name in ('seconds', 'line_seconds'):
                    measurement.pop(name)
            self.assertEqual(hand, board)
            x, y = entry['origin']
            h, w = entry['crop_mask'].shape
            np.testing.assert_array_equal(entry['crop_mask'], result['mask'][y:y+h, x:x+w])
            self.assertEqual(int(np.count_nonzero(entry['crop_mask'])), int(np.count_nonzero(result['mask'])))


class EngineBoardTests(unittest.TestCase):
    def test_find_lines_is_what_run_lines_finds(self):
        frame, inspector, _ = synthetic_inspection()
        source, lines = inspector.find_lines(frame)
        self.assertEqual(source.shape, frame.shape[:2])
        self.assertEqual(len(lines), 2)
        self.assertLess(np.mean(np.asarray(lines[0]['original_xy'])[:, 0]),
                        np.mean(np.asarray(lines[1]['original_xy'])[:, 0]))
        for line, result in zip(lines, inspector.run_lines(frame)):
            self.assertEqual(line, result['selected_boundary'])

    def test_measure_found_gives_the_record_of_run_lines_under_the_number_given(self):
        frame, inspector, _ = synthetic_inspection()
        source, lines = inspector.find_lines(frame)
        hand = list(inspector.run_lines(frame))
        steps = []
        found = list(inspector.measure_found(frame, source, [(lines[1], 9), (lines[0], 4)], 12, steps.append))
        self.assertEqual([(r['selected_line'], r['line_count']) for r in found], [(9, 12), (4, 12)])
        self.assertIn('Inspecting line 9 of 12', steps)
        for mine, theirs in zip(found, hand[::-1]):
            self.assertEqual(set(mine), set(theirs))
            np.testing.assert_array_equal(mine['mask'], theirs['mask'])
            np.testing.assert_array_equal(mine['overlay'], theirs['overlay'])
            self.assertEqual(mine['components'], theirs['components'])
            self.assertEqual(mine['band_length_px'], theirs['band_length_px'])
        self.assertIsNone(inspector._enhanced)

    def test_measure_found_reports_a_failed_line_and_goes_on(self):
        frame, inspector, _ = synthetic_inspection()
        source, lines = inspector.find_lines(frame)
        inspect = inspector._inspect_line
        def fail_first(photo, source, line, index, *args, **kwargs):
            if index == 1:
                raise ValueError('empty adhesive')
            return inspect(photo, source, line, index, *args, **kwargs)
        inspector._inspect_line = fail_first
        picks = [(lines[0], 1), (lines[1], 2)]
        failures = []
        found = list(inspector.measure_found(frame, source, picks, 2,
                                             on_error=lambda number, error: failures.append((number, str(error)))))
        self.assertEqual(failures, [(1, 'empty adhesive')])
        self.assertEqual([r['selected_line'] for r in found], [2])
        with self.assertRaises(ValueError):
            list(inspector.measure_found(frame, source, picks, 2))


class SidewaysTests(unittest.TestCase):
    """A piece laid a quarter turn round is found from the same source mask."""

    def turned(self, code=cv2.ROTATE_90_CLOCKWISE):
        frame, inspector, _ = synthetic_inspection()
        upright = inspector.source.predict(frame)
        source = cv2.rotate(upright, code)
        inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
        return cv2.rotate(frame, code), inspector, source

    def test_an_upright_board_is_read_as_before(self):
        frame, inspector, _ = synthetic_inspection()
        plain, also = inspector.find_lines(frame), inspector.find_lines(frame, sideways=True)
        np.testing.assert_array_equal(plain[0], also[0])
        self.assertEqual(plain[1], also[1])
        self.assertTrue(all('turned' not in line for line in also[1]))

    def test_an_upright_board_does_not_pay_for_a_second_pass(self):
        frame, inspector, _ = synthetic_inspection()
        with patch.object(inspector, '_lines_in', wraps=inspector._lines_in) as geometry:
            inspector.find_lines(frame, sideways=True)
        self.assertEqual(geometry.call_count, 1)

    def test_the_hand_path_does_not_look_for_sideways_pieces(self):
        frame, inspector, _ = self.turned()
        with self.assertRaises(ValueError):
            list(inspector.run_lines(frame))
        with self.assertRaises(NoSourceLinesError):
            inspector.find_lines(frame)

    def test_a_board_turned_a_quarter_reads_the_same_widths(self):
        frame, inspector, _ = synthetic_inspection()
        upright = list(inspector.run_lines(frame))
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            photo, inspector, source = self.turned(code)
            found, lines = inspector.find_lines(photo, sideways=True)
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(line['turned'] for line in lines))
            results = list(inspector.measure_found(photo, found, [(line, i+1) for i, line in enumerate(lines)], 2))
            self.assertEqual(len(results), 2)
            # Which upright line each is: the turn puts the upright lines in this order.
            pairs = zip(results, upright if code == cv2.ROTATE_90_COUNTERCLOCKWISE else upright[::-1])
            for result, reference in pairs:
                self.assertAlmostEqual(mean_width(result), mean_width(reference), delta=.3)
                self.assertAlmostEqual(result['length_px'], reference['length_px'], delta=3)
                self.assertAlmostEqual(result['band_length_px'], reference['band_length_px'], delta=3)
                self.assertEqual(result['component_count'], 1)
                # The mask lies in the photo as it is: the upright mask, turned.
                expected = cv2.rotate(reference['mask'], code)
                both = np.count_nonzero(result['mask'] & expected)
                either = np.count_nonzero(result['mask'] | expected)
                self.assertGreater(both / either, .97)
                # And so do the sample positions that millimetres are taken from.
                centres = np.array([[s['center_x'], s['center_y']] for s in result['components'][0]['samples']])
                at = np.rint(centres).astype(int)
                self.assertTrue(np.all(expected[at[:, 1], at[:, 0]] > 0))
                self.assertGreater(np.ptp(centres[:, 0]), 5 * np.ptp(centres[:, 1]))
                line = np.asarray(result['selected_boundary']['original_xy'])
                self.assertGreater(np.ptp(line[:, 0]), 5 * np.ptp(line[:, 1]))

    def test_upright_and_sideways_pieces_on_one_board(self):
        frame = np.full((1500, 1900, 3), 100, np.uint8)
        source = np.zeros(frame.shape[:2], np.uint8)
        source[:800, :1500] = panels((500, 1000))                      # two upright panels
        piece = cv2.rotate(panels((500,))[:, 300:560], cv2.ROTATE_90_CLOCKWISE)   # one panel, laid sideways
        source[1000:1000+piece.shape[0], 200:200+piece.shape[1]] = piece
        _, inspector = board_inspection()
        inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
        found, lines = inspector.find_lines(frame, sideways=True)
        self.assertEqual(sorted(bool(line.get('turned')) for line in lines), [False, False, True])
        self.assertEqual(len(inspector.find_lines(frame)[1]), 2)
        lines.sort(key=lambda line: not line.get('turned'))     # the sideways one first
        results = list(inspector.measure_found(frame, found, [(line, i+1) for i, line in enumerate(lines)], 3))
        self.assertEqual(len(results), 3)
        widths = [mean_width(result) for result in results]
        self.assertLess(max(widths) - min(widths), .3)
        ys, xs = np.nonzero(results[0]['mask'])
        self.assertGreater(ys.min(), 1000)
        self.assertGreater(np.ptp(xs), 5 * np.ptp(ys))
        board = BoardLines()
        self.assertEqual(board.owed(lines), [(0, 1), (1, 2), (2, 3)])


if __name__ == '__main__':
    unittest.main()


class CutByEdgeTests(unittest.TestCase):
    def mask_at(self, x0, x1, y0, y1, shape=(100, 200)):
        mask = np.zeros(shape, np.uint8)
        mask[y0:y1, x0:x1] = 1
        return {'mask': mask}

    def test_a_line_well_inside_the_photo_is_not_cut(self):
        from inspection.pipeline import cut_by_edge
        self.assertFalse(cut_by_edge(self.mask_at(50, 60, 30, 70), (100, 200), 20))

    def test_glue_near_any_border_is_cut(self):
        from inspection.pipeline import cut_by_edge
        for box in ((5, 15, 30, 70), (185, 195, 30, 70), (50, 60, 0, 40), (50, 60, 60, 100)):
            self.assertTrue(cut_by_edge(self.mask_at(*box), (100, 200), 20), box)

    def test_the_mask_origin_places_it_in_the_photo(self):
        from inspection.pipeline import cut_by_edge
        result = dict(self.mask_at(50, 60, 30, 70, shape=(100, 100)), origin=(930, 0))   # glue to x 989
        self.assertTrue(cut_by_edge(result, (100, 1000), 20))
        self.assertFalse(cut_by_edge(dict(result, origin=(400, 0)), (100, 1000), 20))

    def test_a_margin_of_zero_turns_the_check_off(self):
        from inspection.pipeline import cut_by_edge
        self.assertFalse(cut_by_edge(self.mask_at(0, 10, 0, 100), (100, 200), 0))


class EdgeOfPhotoTests(BoardCase):
    """A line that runs out of the photo is WARNING, not FAIL (local_inspection.board_edge_margin_px)."""

    def failing_board(self):
        frame, inspector = board_inspection(xs=(750,))
        def thin(flat):                 # 1 mm of glue against 3 +- 0.5 mm: every segment fails
            mask = np.zeros(flat.shape[:2], np.uint8)
            mask[45:55, 15:flat.shape[1]-15] = 255
            return mask
        inspector.glue = SimpleNamespace(width=1024, predict=thin)
        return frame, inspector

    def test_a_failing_line_within_the_margin_is_a_warning_with_its_reasons_kept(self):
        frame, inspector = self.failing_board()
        self.config['local_inspection']['board_edge_margin_px'] = 400    # reaches every line here
        payload, _ = self.run_board(frame, inspector, BoardLines())
        entry, = payload['lines']
        self.assertEqual(entry['status'], 'WARNING')
        self.assertTrue(entry['measurement']['cut_by_edge'])
        reasons = entry['measurement']['fail_reasons']
        self.assertEqual(reasons[0], 'line runs out of the photo — not fully in view')
        self.assertEqual(payload['verdict'], 'WARNING')

    def test_with_the_margin_off_the_same_line_fails(self):
        frame, inspector = self.failing_board()
        self.config['local_inspection']['board_edge_margin_px'] = 0
        payload, _ = self.run_board(frame, inspector, BoardLines())
        entry, = payload['lines']
        self.assertEqual(entry['status'], 'FAIL')
        self.assertNotIn('cut_by_edge', entry['measurement'])

    def test_a_passing_line_at_the_edge_stays_pass(self):
        frame, inspector = board_inspection(xs=(750,))
        self.config['local_inspection']['board_edge_margin_px'] = 400
        payload, _ = self.run_board(frame, inspector, BoardLines())
        self.assertEqual(payload['lines'][0]['status'], 'PASS')
