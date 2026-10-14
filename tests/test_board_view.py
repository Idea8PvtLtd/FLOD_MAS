"""Scene mode's board screen: overall verdict, numbered board picture, one card per line."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from auto_trigger.controller import AutoController
from inspection import dashboard
from inspection.board_overlay import (
    FAIL_COLOR, PASS_COLOR, WARNING_COLOR, aggregate_verdict, badge_position, board_overview,
    line_card_image, line_is_upright,
)
from inspection.dashboard import AutomaticDashboard
from inspection.line_measurement import measurement_record
from inspection.local.width_measurement import measure_component
from plane_scale import PlaneScale
from test_local_inspection import CONFIG

PHOTO_SIZE = (1200, 1600)      # rows, columns


def line_entry(line_id, position, status='PASS', sideways=False, new=False, scale=.1, zones=False):
    """One measured line of a board: a straight 40 pixel wide strip, 800 long.

    Measured for real, so the card texts come from a true measurement record;
    ``status`` is then set by hand, as the pipeline's grade would be.
    """
    mask = np.zeros(PHOTO_SIZE, np.uint8)
    if sideways:
        mask[position-20:position+20, 200:1000] = 255
    else:
        mask[200:1000, position-20:position+20] = 255
    component = measure_component(mask)
    source = dict(components=[component], line_count=4, selected_line=line_id, component_count=1,
                  length_px=component['length_px'], longest_length_px=component['length_px'],
                  selected_boundary={}, seconds=0)
    plane = None if scale is None else PlaneScale(np.diag([scale, scale, 1.]), scale)
    with patch('inspection.line_measurement.current_ratio', return_value=1.0):
        measurement = measurement_record(source, plane, 4.0, 1.0, 5.0,
                                         end_exclusion_px=60 if zones else None)
    measurement['status'] = status
    centre = np.array([[s['center_x'], s['center_y']] for s in component['samples']], dtype=float)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys, xs = np.nonzero(mask)
    x0, y0 = max(0, int(xs.min())-60), max(0, int(ys.min())-40)
    x1, y1 = int(xs.max())+60, int(ys.max())+40
    photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
    return {'line_id': line_id, 'status': status, 'measurement': measurement, 'error': None,
            'centre': centre, 'outline': max(contours, key=cv2.contourArea).reshape(-1, 2),
            'crop': photo[y0:y1, x0:x1].copy(), 'crop_mask': mask[y0:y1, x0:x1].copy(),
            'origin': (x0, y0), 'new': new, 'photo_sequence': 7}


def failed_entry(line_id, position, new=True):
    """A line that was found but could not be measured."""
    centre = np.column_stack((np.full(50, float(position)), np.linspace(200, 1000, 50)))
    return {'line_id': line_id, 'status': 'FAIL', 'measurement': None,
            'error': f'Line {line_id}: adhesive model returned an empty mask.',
            'centre': centre, 'outline': None, 'crop': None, 'crop_mask': None,
            'origin': (0, 0), 'new': new, 'photo_sequence': 7}


def board_payload(entries, seconds=1.5):
    photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
    return {'lines': entries, 'overview': board_overview(photo, entries, 800),
            'verdict': aggregate_verdict(entries),
            'new_lines': [e['line_id'] for e in entries if e['new']], 'seconds': seconds,
            'path': 'synthetic', 'frame_sequence': 7, 'source_image': photo}


def close_to(pixel, color, tolerance=12):
    return bool(np.all(np.abs(pixel.astype(int) - np.array(color)) <= tolerance))


class VerdictTests(unittest.TestCase):
    def test_board_verdict_table(self):
        def entry(status, measured=True):
            return {'status': status, 'measurement': {} if measured else None}
        table = [([], 'EMPTY'),
                 (['PASS'], 'PASS'),
                 (['PASS', 'PASS', 'PASS'], 'PASS'),
                 (['PASS', 'WARNING'], 'WARNING'),
                 (['WARNING'], 'WARNING'),
                 (['PASS', 'FAIL'], 'FAIL'),
                 (['WARNING', 'FAIL', 'PASS'], 'FAIL'),
                 (['FAIL'], 'FAIL')]
        for statuses, expected in table:
            self.assertEqual(aggregate_verdict([entry(s) for s in statuses]), expected, statuses)

    def test_a_line_without_a_measurement_fails_the_board_whatever_its_status(self):
        for status in ('PASS', 'WARNING', 'FAIL'):
            entries = [{'status': 'PASS', 'measurement': {}},
                       {'status': status, 'measurement': None}]
            self.assertEqual(aggregate_verdict(entries), 'FAIL', status)


class OverviewTests(unittest.TestCase):
    def test_photo_is_scaled_first_and_never_enlarged(self):
        photo = np.full((3456, 4608, 3), 120, np.uint8)
        with patch('inspection.board_overlay.cv2.fillPoly', wraps=cv2.fillPoly) as fill:
            picture = board_overview(photo, [line_entry(1, 400)], 1920)
        self.assertEqual(picture.shape, (1440, 1920, 3))
        # Drawn on the small picture: no copy of the 16 megapixel photo is made to draw on.
        self.assertEqual(fill.call_args.args[0].shape, (1440, 1920, 3))
        self.assertTrue((photo == 120).all())
        small = np.full((300, 400, 3), 120, np.uint8)
        self.assertEqual(board_overview(small, [], 1920).shape, (300, 400, 3))
        self.assertEqual(board_overview(np.full((400, 300), 120, np.uint8), [], 200).shape, (200, 150, 3))

    def test_each_line_carries_the_colour_of_its_verdict(self):
        entries = [line_entry(1, 300, 'PASS'), line_entry(2, 700, 'FAIL'),
                   line_entry(3, 1100, 'WARNING', new=True)]
        photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
        picture = board_overview(photo, entries, 800)
        self.assertEqual(picture.shape, (600, 800, 3))
        for entry, color in zip(entries, (PASS_COLOR, FAIL_COLOR, WARNING_COLOR)):
            x = round(entry['centre'][:, 0].mean() * .5)
            # The left edge of the outline, half way down the line.
            edge = picture[300, x-10]
            self.assertTrue(close_to(edge, color, 40), (entry['line_id'], edge))
            # The badge: a disc of the verdict's colour just above the top end.
            badge = badge_position(entry['centre'] * .5, 15, (800, 600))
            self.assertLess(badge[1], 100)
            self.assertAlmostEqual(badge[0], x, delta=2)
            ring = picture[badge[1], badge[0]-12]
            self.assertTrue(close_to(ring, color), (entry['line_id'], ring))
        # Away from every line the photo is untouched.
        self.assertTrue((picture[300, 250] == 120).all())
        self.assertTrue((picture[580, 400] == 120).all())

    def test_a_new_line_is_drawn_heavier_than_a_known_one(self):
        photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
        known = board_overview(photo, [line_entry(1, 700, 'PASS')], 800)
        new = board_overview(photo, [line_entry(1, 700, 'PASS', new=True)], 800)
        def painted(picture):
            return int(np.all(np.abs(picture[300].astype(int) - PASS_COLOR) <= 40, axis=1).sum())
        self.assertGreater(painted(new), painted(known))
        # The NEW tag beside the badge is only on the new line.
        self.assertTrue((known[:90, 380:470] != new[:90, 380:470]).any())

    def test_sideways_lines_are_numbered_at_their_left_end(self):
        entries = [line_entry(1, 300, 'PASS', sideways=True), line_entry(2, 800, 'FAIL', sideways=True)]
        self.assertFalse(line_is_upright(entries[0]))
        self.assertTrue(line_is_upright(line_entry(1, 300)))
        photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
        picture = board_overview(photo, entries, 800)
        for entry, color in zip(entries, (PASS_COLOR, FAIL_COLOR)):
            y = round(entry['centre'][:, 1].mean() * .5)
            self.assertTrue(close_to(picture[y-10, 300], color, 40), entry['line_id'])
            badge = badge_position(entry['centre'] * .5, 15, (800, 600))
            self.assertLess(badge[0], 100)
            self.assertAlmostEqual(badge[1], y, delta=2)
            self.assertTrue(close_to(picture[badge[1]-12, badge[0]], color), entry['line_id'])

    def test_a_badge_stays_inside_the_picture_when_the_line_reaches_the_edge(self):
        centre = np.column_stack((np.full(20, 5.), np.linspace(0, 500, 20)))
        self.assertEqual(badge_position(centre, 15, (800, 600)), (17, 17))
        self.assertIsNone(badge_position(None, 15, (800, 600)))

    def test_a_line_that_could_not_be_measured_is_still_drawn_and_numbered(self):
        entries = [line_entry(1, 300, 'PASS'), failed_entry(2, 900)]
        photo = np.full(PHOTO_SIZE + (3,), 120, np.uint8)
        picture = board_overview(photo, entries, 800)
        self.assertTrue(close_to(picture[300, 450], FAIL_COLOR, 40), picture[300, 450])
        badge = badge_position(entries[1]['centre'] * .5, 15, (800, 600))
        self.assertTrue(close_to(picture[badge[1], badge[0]-12], FAIL_COLOR))
        # An entry with nothing to draw is skipped without failing the rest.
        entries.append(dict(failed_entry(3, 0), centre=None))
        self.assertEqual(board_overview(photo, entries, 800).shape, (600, 800, 3))

    def test_card_picture_fills_its_box_for_either_direction_and_for_a_failed_line(self):
        upright = line_entry(1, 300, 'FAIL')
        before = [dict(s) for s in upright['measurement']['components'][0]['samples']]
        card = line_card_image(upright, (48, 200))
        self.assertEqual(card.shape, (200, 48, 3))
        # The line runs the whole height of a tall box...
        self.assertGreater(np.ptp(np.nonzero((card != (17, 8, 3)).any(axis=2))[0]), 190)
        # ...and the measurement was not moved to crop coordinates by drawing it.
        self.assertEqual(before, upright['measurement']['components'][0]['samples'])
        sideways = line_card_image(line_entry(1, 300, sideways=True), (300, 40))
        self.assertEqual(sideways.shape, (40, 300, 3))
        self.assertGreater(np.ptp(np.nonzero((sideways != (17, 8, 3)).any(axis=2))[1]), 290)
        failed = line_card_image(failed_entry(2, 900), (48, 200))
        self.assertEqual(failed.shape, (200, 48, 3))
        self.assertTrue(np.all(np.abs(failed.reshape(-1, 3).astype(int) - FAIL_COLOR) <= 12, axis=1).any())


class DashboardTests(unittest.TestCase):
    def app(self, controller, calls):
        return SimpleNamespace(
            auto_controller=controller, video_paused=False, video_streaming=True, _reset_due=None,
            root=SimpleNamespace(after=lambda *args: None),
            _show_automatic_results=lambda p: calls.append(('result', p)),
            _show_board_results=lambda p: calls.append(('board', p)),
            _dismiss_result_view=lambda: calls.append(('clear', None)),
            update_progress=lambda *args: None, set_pass_fail=lambda *args: None,
            _poll_automatic=lambda: None)

    def test_board_event_reaches_the_board_screen_and_stale_ones_do_not(self):
        controller = AutoController(CONFIG, lambda: None)
        calls = []
        app = self.app(controller, calls)
        payload = {'lines': [], 'verdict': 'EMPTY'}
        controller.events.put(('board', controller.generation, payload))
        controller.events.put(('board', controller.generation + 1, {'stale': True}))
        controller.events.put(('result', controller.generation, {'line': 1}))
        AutomaticDashboard._poll_automatic(app)
        self.assertEqual(calls, [('board', payload), ('result', {'line': 1})])
        # Not shown while the feed is paused, as for a line result.
        app.video_paused = True
        controller.events.put(('board', controller.generation, payload))
        AutomaticDashboard._poll_automatic(app)
        self.assertEqual(len(calls), 2)

    def board_app(self, statuses, progress, result_view=None):
        opened = []
        app = SimpleNamespace(result_view=result_view, set_pass_fail=statuses.append,
                              update_progress=lambda value, text: progress.append(text),
                              _dismiss_result_view=lambda: opened.append('dismissed'))
        return app, opened

    def test_board_screen_is_made_once_and_the_status_follows_the_verdict(self):
        class FakeBoardView:
            def __init__(self):
                self.shown = []
            def show_board(self, payload):
                self.shown.append(payload)

        def open_view(app, view_class):
            opened.append(view_class)
            app.result_view = view_class()

        statuses, progress = [], []
        app, opened = self.board_app(statuses, progress)
        entry = {'line_id': 2, 'status': 'FAIL', 'measurement': {}}
        payloads = [{'lines': [], 'verdict': 'EMPTY', 'new_lines': []},
                    {'lines': [{'line_id': 1, 'status': 'PASS', 'measurement': {}}],
                     'verdict': 'PASS', 'new_lines': [1]},
                    {'lines': [{'line_id': 1, 'status': 'PASS', 'measurement': {}}, entry],
                     'verdict': 'FAIL', 'new_lines': [2]}]
        with patch.object(dashboard, 'BoardResultView', FakeBoardView), \
                patch.object(dashboard, '_open_result_view', open_view), \
                self.assertLogs('flod.dashboard', 'INFO') as logged:
            for payload in payloads:
                AutomaticDashboard._show_board_results(app, payload)
        self.assertEqual(opened, [FakeBoardView])
        self.assertEqual(app.result_view.shown, payloads)
        # EMPTY has no verdict to show, so the status label stays neutral.
        self.assertEqual(statuses, ['READY', 'PASS', 'FAIL'])
        self.assertIn('no glue line yet', progress[0])
        self.assertIn('line 2 failed', progress[2])
        self.assertEqual(len(logged.output), 3)
        self.assertIn('Shown to the operator: board with 2 line(s) as FAIL (new: 2)', logged.output[2])

    def test_a_line_view_left_on_screen_is_replaced_by_the_board_screen(self):
        class FakeBoardView:
            def show_board(self, payload):
                self.payload = payload

        def open_view(app, view_class):
            opened.append('opened')
            app.result_view = view_class()

        app, opened = self.board_app([], [], result_view=object())
        with patch.object(dashboard, 'BoardResultView', FakeBoardView), \
                patch.object(dashboard, '_open_result_view', open_view):
            AutomaticDashboard._show_board_results(app, {'lines': [], 'verdict': 'EMPTY'})
        self.assertEqual(opened, ['dismissed', 'opened'])
        self.assertIsInstance(app.result_view, FakeBoardView)


class BoardViewTests(unittest.TestCase):
    def view(self, show_live_preview=False):
        import tkinter as tk
        from inspection.board_view import BoardResultView
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        view = BoardResultView(root, show_live_preview=show_live_preview)
        view.pack(fill='both', expand=True)
        self.addCleanup(view.destroy)
        return view

    def test_board_shown_twice_redraws_every_card_and_the_verdict(self):
        view = self.view()
        first = [line_entry(1, 300, 'PASS', new=True), line_entry(2, 600, 'PASS', new=True)]
        view.show_board(board_payload(first, seconds=1.5))
        self.assertEqual(len(view.cards), 2)
        self.assertEqual(view.verdict_label.cget('text'), 'PASS')
        self.assertEqual(view.info_label.cget('text'), '2 lines  ·  2 new  ·  last photo 1.5 s')
        self.assertEqual([card['number'].cget('text') for card in view.cards], ['1', '2'])
        self.assertEqual([card['status'].cget('text') for card in view.cards], ['PASS', 'PASS'])
        self.assertEqual(view.cards[0]['width'].cget('text'), '4.00 mm')
        self.assertTrue(all(card['new'] is not None for card in view.cards))
        self.assertEqual(len(view._tabs), 1)
        self.assertEqual(view._tabs[0]['image'].size, (800, 600))

        failing = line_entry(3, 900, 'FAIL', scale=.14, zones=True)
        failing['measurement']['fail_reasons'] = ['gap of 12.0 mm in the line, above 5 mm']
        second = [dict(first[0], new=False), dict(first[1], new=False), failing, failed_entry(4, 1200)]
        payload = board_payload(second, seconds=2.3)
        view.show_board(payload)
        self.assertEqual(len(view.cards), 4)
        self.assertEqual(len(view.cards_frame.winfo_children()), 5)     # four cards and the hidden note
        self.assertEqual(view.verdict_label.cget('text'), 'FAIL')
        self.assertEqual(view.info_label.cget('text'), '4 lines  ·  1 new  ·  last photo 2.3 s')
        self.assertEqual([card['status'].cget('text') for card in view.cards],
                         ['PASS', 'PASS', 'FAIL', 'FAIL'])
        self.assertEqual([card['new'] is not None for card in view.cards], [False, False, False, True])
        self.assertEqual(view.cards[2]['width'].cget('text'), '5.60 mm')
        detail = view.cards[2]['detail'].cget('text')
        self.assertIn('gap of 12.0 mm in the line, above 5 mm', detail)
        self.assertIn('Ends: 5.60 mm / 5.60 mm', detail)
        self.assertIn('Target 4 ± 1 mm', detail)
        # The line that could not be measured says why, and is a FAIL without a width.
        self.assertEqual(view.cards[3]['width'].cget('text'), '—')
        self.assertIn('adhesive model returned an empty mask', view.cards[3]['detail'].cget('text'))

        # Showing the same payload again changes nothing: the screen is made from it alone.
        texts = [(c['status'].cget('text'), c['width'].cget('text'), c['detail'].cget('text'))
                 for c in view.cards]
        view.show_board(payload)
        self.assertEqual(len(view.cards), 4)
        self.assertEqual(texts, [(c['status'].cget('text'), c['width'].cget('text'),
                                  c['detail'].cget('text')) for c in view.cards])

    def test_empty_board_and_pixel_widths(self):
        view = self.view()
        view.show_board(board_payload([]))
        self.assertEqual(view.cards, [])
        self.assertEqual(view.verdict_label.cget('text'), 'NO GLUE LINE YET')
        self.assertEqual(view.info_label.cget('text'), '0 lines  ·  last photo 1.5 s')
        view.show_board(board_payload([line_entry(1, 300, 'WARNING', scale=None)]))
        self.assertEqual(view.verdict_label.cget('text'), 'WARNING')
        self.assertEqual(view.cards[0]['width'].cget('text'), '40.0 px')
        self.assertIn('No millimetre scale', view.cards[0]['detail'].cget('text'))
        # Back to an empty board: the cards go, nothing is left from before.
        view.show_board(board_payload([]))
        self.assertEqual(view.cards, [])
        self.assertEqual(len(view.cards_frame.winfo_children()), 1)

    def test_card_click_zooms_to_its_line_and_whole_board_returns(self):
        view = self.view()
        # A withdrawn window has no size; the picture area is given one.
        view._canvas_size = lambda: (400, 300)
        entries = [line_entry(1, 300, 'PASS'), line_entry(2, 1300, 'FAIL')]
        view.show_board(board_payload(entries))
        tab = view._tabs[0]
        # Every part of a card answers a click (a real click is made in verify_board_ui.py;
        # a withdrawn window is sent no button events).
        for name in ('frame', 'status', 'width', 'detail', 'number', 'image'):
            self.assertTrue(view.cards[1][name].bind('<Button-1>'), name)
        view.zoom_to_line(2)
        self.assertEqual(view.selected_line, 2)
        self.assertGreater(tab['zoom'], 1)
        # The middle of line 2 (photo 1300, 600 -> overview 650, 300) is what is zoomed on:
        # the same zoom about any other point would leave it elsewhere.
        scale = tab['fit'] * tab['zoom']
        canvas = view._canvas_size()
        expected = [canvas[0] / 2 - 650 * scale, canvas[1] / 2 - 300 * scale]
        from crop_result_view import clamp_offset
        self.assertEqual(scale, 1.25)
        expected = clamp_offset((800, 600), canvas, scale, expected)
        self.assertAlmostEqual(tab['offset'][0], expected[0], delta=1)
        self.assertAlmostEqual(tab['offset'][1], expected[1], delta=1)
        # Line 2 is on screen and line 1 is not.
        self.assertTrue(0 < tab['offset'][0] + 650 * scale < 400)
        self.assertFalse(0 < tab['offset'][0] + 150 * scale < 400)
        view.zoom_to_line(99)       # an unknown line changes nothing
        self.assertEqual(view.selected_line, 2)
        view.show_whole_board()
        self.assertIsNone(view.selected_line)
        self.assertEqual(tab['zoom'], 1)
        # A new payload starts from the whole board again.
        view.zoom_to_line(1)
        view.show_board(board_payload(entries))
        self.assertIsNone(view.selected_line)
        self.assertEqual(view._tabs[0]['zoom'], 1)

    def test_live_preview_panel_behaves_as_in_the_line_view(self):
        view = self.view(show_live_preview=True)
        self.assertTrue(hasattr(view, 'live_panel'))
        frame = np.zeros((48, 64, 3), np.uint8)
        view.update_live_preview(frame, None)   # not drawn yet: nothing to do, no error
        view.set_paused(True)
        view.set_paused(False)
        view.clear_live_overlay()
        plain = self.view(show_live_preview=False)
        self.assertFalse(hasattr(plain, 'live_panel'))
        with patch('inspection.result_view.cv2.resize') as resize:
            plain.update_live_preview(frame, None)
        resize.assert_not_called()

    def test_six_lines_fit_two_columns_and_more_still_get_a_card_each(self):
        from inspection.board_view import card_grid
        self.assertEqual([card_grid(n) for n in (1, 2, 3, 4, 5, 6, 7, 9, 10)],
                         [(1, 2), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3), (3, 3), (3, 3), (3, 4)])
        view = self.view()
        entry = line_entry(1, 300, 'PASS')
        view.show_board(board_payload([dict(entry, line_id=n) for n in range(1, 9)]))
        self.assertEqual(len(view.cards), 8)
        self.assertEqual([card['number'].cget('text') for card in view.cards],
                         [str(n) for n in range(1, 9)])


if __name__ == '__main__':
    unittest.main()
