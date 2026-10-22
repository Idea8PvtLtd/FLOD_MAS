"""The app's own pipeline (methods A, B, B-fast) on every still moment of a run.

    python -m tools.method_compare.methods_app --run DIR --method A|B|B-fast

Both models run on CUDA. Times are what scene mode spends on a photo: finding the
lines, then measuring only the lines that are new on this board (inspection/
board_lines.BoardLines decides, as in the app). Every other line of the photo is
then measured too, for the viewer only, and its time is kept apart.
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import cv2

from . import BY_ID, ROOT
from .results import from_app, load, save


def gpu_distance_transform():
    """Full-size distance transforms on the GPU (CuPy): the same readings, measured bit-identical."""
    import cupy as cp
    from cupyx.scipy import ndimage

    plain = cv2.distanceTransform

    def distance_transform(src, *args, **kwargs):
        if src.size < 4_000_000:
            return plain(src, *args, **kwargs)
        return ndimage.distance_transform_edt(cp.asarray(src)).astype(cp.float32).get()
    cv2.distanceTransform = distance_transform


def inspector_for(method, config):
    """LiveInspection built as inspection/dashboard.py builds it, on CUDA."""
    from inspection.local.engine import LiveInspection
    settings = config['local_inspection']
    return LiveInspection(ROOT / settings['source_model'], ROOT / settings['glue_model'],
                          settings['offset_pixels'], settings['source_width'],
                          smooth_normal_rows=settings.get('smooth_normal_rows', 0),
                          smooth_logits=settings.get('smooth_logits', False),
                          min_line_length_ratio=settings.get('min_line_length_ratio', 0.),
                          ribbon_finish=method['ribbon'],
                          segments=config['inspection'].get('segment_count', 10),
                          execution_provider='cuda', line_workers=1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--method', required=True, choices=[m['id'] for m in BY_ID.values() if m['family'] == 'app'])
    args = parser.parse_args(argv)
    run, method = Path(args.run), BY_ID[args.method]
    from app_config import CONFIG
    from inspection.board_lines import BoardLines
    from inspection.local.engine import NoSourceLinesError
    from inspection.local.ort_providers import active_provider
    cv2.setNumThreads(method['cv_threads'])
    if method['gpu_distance']:
        gpu_distance_transform()
    moments = load(run / 'moments.json')['moments']
    started = time.perf_counter()
    inspector = inspector_for(method, CONFIG)
    load_s = time.perf_counter() - started

    scale_for = {}

    def scale(photo):
        size = (photo.shape[1], photo.shape[0])
        if size not in scale_for:
            try:
                from plane_scale import load_frame_scale
                scale_for[size] = load_frame_scale(1, size)
            except Exception as error:
                print(f'No millimetre scale: {error}', flush=True)
                scale_for[size] = None
        return scale_for[size]

    photos = [cv2.imread(str(run / 'photos' / m['photo'])) for m in moments]
    first_s = None
    if photos:
        # The first photo once, not counted: model warm-up.
        started = time.perf_counter()
        try:
            source, lines = inspector.find_lines(photos[0], sideways=True)
            list(inspector.measure_found(photos[0], source, [(lines[0], 1)], len(lines)))
        except NoSourceLinesError:
            pass
        first_s = time.perf_counter() - started

    boards, rows = {}, []
    distance = CONFIG.get('local_inspection', {}).get('board_match_distance_px', 60.0)
    for moment, photo in zip(moments, photos):
        board = boards.setdefault(moment['board'], BoardLines(distance))
        t0 = time.perf_counter()
        try:
            source, lines = inspector.find_lines(photo, sideways=True)
        except NoSourceLinesError:
            source, lines = None, []
        t1 = time.perf_counter()
        picks = board.owed(lines) if lines else []
        new_ids = {index: line_id for index, line_id in picks}
        readings, errors = {}, []

        def failed(number, error):
            errors.append(f'line {number}: {error}')
        if picks:
            for result in inspector.measure_found(photo, source, [(lines[i], i + 1) for i, _ in picks],
                                                  len(lines), on_error=failed):
                readings[result['selected_line'] - 1] = result
        t2 = time.perf_counter()
        rest = [i for i in range(len(lines)) if i not in new_ids]
        if rest:
            for result in inspector.measure_found(photo, source, [(lines[i], i + 1) for i in rest],
                                                  len(lines), on_error=failed):
                readings[result['selected_line'] - 1] = result
        t3 = time.perf_counter()
        for index, line_id in picks:
            if index in readings:
                board.record(line_id, {'line_id': line_id, 'new': True})
        measured = []
        for index in sorted(readings):
            line = from_app(readings[index], scale(photo))
            line['new'] = index in new_ids
            line['board_line'] = new_ids.get(index)
            line['seconds'] = round(float(readings[index].get('line_seconds') or 0.0), 3)
            measured.append(line)
        rows.append({'photo': moment['photo'], 'lines': measured, 'found': len(lines), 'new': len(picks),
                     'find_s': round(t1 - t0, 3), 'new_lines_s': round(t2 - t1, 3),
                     'other_lines_s': round(t3 - t2, 3), 'photo_s': round(t2 - t0, 3), 'errors': errors})
        print(f"{moment['photo']}: {len(lines)} lines ({len(picks)} new), photo {t2 - t0:.2f} s "
              f"(find {t1 - t0:.2f}, new lines {t2 - t1:.2f})", flush=True)
    save(run / f"{method['id']}.json", {
        'method': method['id'], 'label': method['label'], 'family': 'app', 'load_s': round(load_s, 2),
        'first_photo_s': round(first_s, 2) if first_s is not None else None,
        'providers': [active_provider(inspector.source.session), active_provider(inspector.glue.session)],
        'cv_threads': cv2.getNumThreads(), 'gpu_distance': method['gpu_distance'], 'ribbon': method['ribbon'],
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        'time_note': 'photo = find the lines + measure the lines new on this board (scene mode)',
        'photos': rows})
    return 0


if __name__ == '__main__':
    sys.exit(main())
