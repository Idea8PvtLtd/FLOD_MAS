"""Board result screen judged by eye, from a real photo of a full board (no camera).

Measures every line of one saved photo with the real models, builds board
payloads as the scene mode's pipeline will, shows them in the real dashboard at
1366 x 768 and saves a screenshot of each state:

    DISPLAY=:0 .venv/bin/python tests/verify_board_ui.py [output folder] [photo]

The photo has no millimetre scale here, so the first screens are WARNING with
widths in pixels. The later ones use a made-up scale (the mean width is called
4 mm) so that PASS and FAIL cards, end zones and a line that could not be
measured can be looked at too. Those numbers are for the look only.
"""
import copy
import glob
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageGrab

from app_config import CONFIG
from inspection.board_overlay import aggregate_verdict, board_overview
from inspection.line_measurement import measurement_record
from plane_scale import PlaneScale

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = '~/Documents/data files/Dataset_capture/Automatic/20261013_175023_*/undistorted.png'


def measure(photo):
    """Every line of the photo, as LiveInspection.run_lines reports them."""
    from inspection.local.engine import LiveInspection
    settings = CONFIG['local_inspection']
    inspector = LiveInspection(ROOT / settings['source_model'], ROOT / settings['glue_model'],
                               settings['offset_pixels'], settings['source_width'],
                               smooth_normal_rows=101, smooth_logits=True,
                               min_line_length_ratio=0.5, ribbon_finish=True)
    results = []
    for result in inspector.run_lines(photo):
        # The overlay is a whole-photo picture per line; the board screen does not use it.
        results.append({key: value for key, value in result.items() if key != 'overlay'})
    return results


def entry_from(result, measurement, photo, new, sequence=1):
    """One line of the board payload, as the frozen interface describes it."""
    mask = result['mask']
    main = max(measurement['components'], key=lambda c: c['length_px'])
    centre = np.array([[s['center_x'], s['center_y']] for s in main['samples']], dtype=float)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else None
    # Cut as LineResultView.show_inspection cuts its crop.
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, int(xs.min())-160), min(photo.shape[1], int(xs.max())+180)
    y0, y1 = max(0, int(ys.min())-40), min(photo.shape[0], int(ys.max())+40)
    return {'line_id': measurement['selected_line'], 'status': measurement['status'],
            'measurement': measurement, 'error': None, 'centre': centre, 'outline': outline,
            'crop': photo[y0:y1, x0:x1].copy(), 'crop_mask': mask[y0:y1, x0:x1].copy(),
            'origin': (x0, y0), 'new': new, 'photo_sequence': sequence}


def payload_from(photo, entries, seconds):
    return {'lines': entries, 'overview': board_overview(photo, entries, 1920),
            'verdict': aggregate_verdict(entries),
            'new_lines': [e['line_id'] for e in entries if e['new']], 'seconds': seconds,
            'path': 'verify_board_ui', 'frame_sequence': 1, 'source_image': photo}


def scaled_entry(result, photo, mm_per_px, new, tolerance=1.0):
    """The same line graded with a made-up scale and with end zones, for PASS/FAIL cards."""
    scale = PlaneScale(np.diag([mm_per_px, mm_per_px, 1.]), mm_per_px)
    measurement = measurement_record(result, scale, 4.0, tolerance, 5.0,
                                     end_exclusion_px=150, end_trim_px=20)
    return entry_from(result, measurement, photo, new)


def failed_entry(result, line_id):
    """A line whose measurement failed: where it was looked for, and why it failed."""
    course = np.asarray(result['selected_boundary']['original_xy'], dtype=float).reshape(-1, 2)
    return {'line_id': line_id, 'status': 'FAIL', 'measurement': None,
            'error': f'Line {line_id}: adhesive model returned an empty mask.',
            'centre': course, 'outline': None, 'crop': None, 'crop_mask': None,
            'origin': (0, 0), 'new': True, 'photo_sequence': 1}


def moved(entry, line_id, shift):
    """A copy of a line moved sideways under another number, to fill a six-line board."""
    other = copy.deepcopy(entry)
    other['line_id'] = line_id
    other['measurement']['selected_line'] = line_id
    other['centre'] = other['centre'] + (shift, 0)
    other['outline'] = other['outline'] + (int(shift), 0)
    return other


def main():
    # Outside the repository by default: the kept measurements are tens of megabytes.
    output = Path(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / 'board_ui_preview')
    output.mkdir(parents=True, exist_ok=True)
    found = sorted(glob.glob(os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else PHOTOS)))
    if not found:
        raise SystemExit('No photo found: ' + PHOTOS)
    photo = cv2.imread(found[0])
    # Measuring takes a while and the screens are looked at many times while the
    # layout is adjusted, so the measured lines are kept beside the screenshots.
    cache = output / 'measured_lines.pickle'
    started = time.perf_counter()
    if cache.exists():
        results, seconds = pickle.loads(cache.read_bytes())
    else:
        results = measure(photo)
        seconds = time.perf_counter() - started
        cache.write_bytes(pickle.dumps((results, seconds)))
    print(f'{len(results)} lines measured from {found[0]} in {seconds:.1f} s')

    real = [entry_from(r, measurement_record(r, None, 4.0, 1.0, 5.0), photo, new=False) for r in results]
    for entry in real[-1:]:
        entry['new'] = True
    first_two = [dict(entry, new=True) for entry in real[:2]]
    widths = [e['measurement']['mean_width_px'] for e in real]
    mm_per_px = 4.0 / float(np.mean(widths))
    graded = [scaled_entry(r, photo, mm_per_px * (1.4 if index == 1 else 1.0), new=index >= 2)
              for index, r in enumerate(results)]
    with_failure = graded[:-1] + [failed_entry(results[-1], graded[-1]['line_id'])]
    six = graded + [moved(graded[0], len(graded) + 1, 170), moved(graded[1], len(graded) + 2, 170)]
    sideways = sideways_payload(photo, results, mm_per_px)

    from main_1366 import IndustrialDashboard
    from auto_trigger.controller import AutoController
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch.object(IndustrialDashboard, '_init_automatic'):
        app = IndustrialDashboard(root, lambda: None)
    app.auto_controller = AutoController(CONFIG, lambda: None)  # never start models/workers
    app._reset_due = None
    small = cv2.resize(photo, (640, 480))

    def shot(name):
        # The dashboard fills the physical screen; the look is judged at the target size.
        root.geometry('1366x768+0+0')
        for _ in range(3):
            root.update()
            time.sleep(.15)
        app.result_view.update_live_preview(small, None)
        root.update()
        assert root.winfo_width() == 1366 and root.winfo_height() == 768, root.winfo_geometry()
        path = output / f'{name}.png'
        try:
            ImageGrab.grab(window=int(root.frame(), 16)).save(path)
        except OSError:
            # Under Wayland the X screen cannot be read; the window itself still can.
            subprocess.run(['import', '-window', root.frame(), str(path)], check=True)
        print('saved', path)

    def show(name, entries, took, picture=photo):
        payload = payload_from(picture, entries, took)
        controller = app.auto_controller
        controller.events.put(('board', controller.generation, payload))
        app.video_paused, app.video_streaming = False, True
        app._poll_automatic()
        root.after_cancel(app._auto_poll_job)
        view = app.result_view
        root.update()
        assert len(view.cards) == len(entries)
        assert app.status_result_label.cget('text') == ('READY' if not entries else payload['verdict'])
        shot(name)
        # Every card must lie wholly inside the cards' frame: nothing cut off.
        frame = view.cards_frame
        for card in view.cards:
            widget = card['frame']
            assert widget.winfo_ismapped()
            assert widget.winfo_rootx() >= frame.winfo_rootx()
            assert widget.winfo_rooty() >= frame.winfo_rooty()
            assert widget.winfo_rootx() + widget.winfo_width() <= frame.winfo_rootx() + frame.winfo_width()
            assert widget.winfo_rooty() + widget.winfo_height() <= frame.winfo_rooty() + frame.winfo_height()
            detail = card['detail']
            assert detail.winfo_reqheight() <= detail.winfo_height() + 1, \
                (name, card['line_id'], detail.winfo_reqheight(), detail.winfo_height())
        return view

    show('1_first_photo_two_lines', first_two, seconds / 2)
    view = show('2_real_four_lines_no_scale', real, seconds)
    view = show('3_pass_and_fail', graded, 1.8)
    # A real click on the card of the failed line.
    view.cards[1]['width'].event_generate('<Button-1>', x=5, y=5)
    root.update()
    assert view._tabs[0]['zoom'] > 1 and view.selected_line == graded[1]['line_id']
    shot('4_zoomed_to_failed_line')
    view.show_whole_board()
    assert view._tabs[0]['zoom'] == 1 and view.selected_line is None
    show('5_line_not_measured', with_failure, 2.1)
    show('6_six_lines', six, 2.4)
    show('7_sideways_lines', sideways[1], 2.0, sideways[0])
    show('8_empty_board', [], 0.4)
    app._dismiss_result_view()
    root.update()
    assert app.result_view is None and app.images_container.winfo_ismapped()
    assert app.bottom_panel.winfo_ismapped()
    assert not errors, errors
    root.destroy()
    print('Board screens saved to', output)


def sideways_payload(photo, results, mm_per_px):
    """The same board turned a quarter turn, so the lines lie sideways in the photo.

    The measurement is turned with the photo rather than made again: only the
    look of sideways lines is judged here.
    """
    height = photo.shape[0]
    turned = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
    entries = []
    for index, result in enumerate(results):
        other = {key: value for key, value in result.items() if key != 'mask'}
        other['mask'] = cv2.rotate(result['mask'], cv2.ROTATE_90_CLOCKWISE)
        other['components'] = copy.deepcopy(result['components'])
        for component in other['components']:
            for sample in component['samples']:
                for name in ('center', 'left', 'right'):
                    x, y = sample.get(f'{name}_x'), sample.get(f'{name}_y')
                    if x is not None and y is not None:
                        sample[f'{name}_x'], sample[f'{name}_y'] = height - 1 - y, x
        entries.append(scaled_entry(other, turned, mm_per_px * (1.4 if index == 2 else 1.0),
                                    new=index == 3))
    return turned, entries


if __name__ == '__main__':
    main()
