"""main's graded segments (40 per line, 4.0 +/- 1.0 mm) with the visibility gate and a neighbour rule.

    python -m tools.glue_defects.segments_check --repo ../FLOD_MAS_main [--data DIR]

``--repo`` is a checkout whose engine and measurement_record are used as they are
(main's, at the time of writing: end zones of 80 px, 50 px trimmed, widths graded at
0.1 mm). Each line gets main's own verdict, then the same segments again with two
changes: a segment where the glue edges are too dark to see is not judged
(check.VIS_MIN), and a seen segment fails below check.LIMIT of its seen neighbours
(check.segment_check). Writes segments/widths.json for the viewer.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import DATA, MM_PER_PX, STICKER_REACH_PX, load_photos, photo_of, save_json
from .check import LIMIT, VIS_MIN, segment_check


def nearest(xy, path, values):
    """Each point of ``xy`` takes the value of the nearest point of ``path``."""
    path = np.asarray(path, float)[::3]
    index = np.argmin(np.hypot(xy[:, None, 0] - path[None, :, 0], xy[:, None, 1] - path[None, :, 1]), axis=1) * 3
    return np.asarray(values)[np.clip(index, 0, len(values) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', type=Path, required=True, help='checkout of the branch whose method is checked')
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    data = args.data.resolve()
    photos = load_photos(data)
    app = json.loads((data / 'app' / 'widths.json').read_text())
    visibility = json.loads((data / 'visibility.json').read_text())
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import cv2
    import plane_scale
    from app_config import CONFIG
    from inspection.line_measurement import measurement_record
    from inspection.local.engine import LiveInspection
    s, ins = CONFIG['local_inspection'], CONFIG['inspection']
    import inspect
    extra = {key: s[key] for key in ('smooth_normal_rows', 'smooth_logits', 'min_line_length_ratio', 'ribbon_finish')
             if key in s}
    if 'segments' in inspect.signature(LiveInspection.__init__).parameters:
        extra['segments'] = ins.get('segment_count', 10)      # as the app builds it
    engine = LiveInspection(str(repo / s['source_model']), str(repo / s['glue_model']), s['offset_pixels'],
                            s['source_width'], **extra)
    out = {}
    for photo, info in photos.items():
        image = cv2.imread(str(data / info['file']))
        h, w = image.shape[:2]
        try:
            scale = plane_scale.load_frame_scale(1, (w, h))
        except plane_scale.PlaneScaleError:
            scale = None
        for result in engine.run_lines(image):
            m = measurement_record(result, scale, ins['strip_width_mm'], ins['strip_width_tolerance_mm'],
                                   ins.get('end_exclusion_percent', 5.0), ins.get('end_exclusion_px'),
                                   ins.get('end_trim_px', 0.0))
            main = max(m['components'], key=lambda c: c['length_px'])
            xy = np.array([[x['center_x'], x['center_y']] for x in main['samples']])
            seg_of = np.array([x['segment'] for x in main['samples']])
            name = f"{photo}_L{result['selected_line']}"
            # the same line in app/widths.json (for its visibility): the one whose path runs closest
            ref = min((k for k in app if photo_of(k) == photo),
                      key=lambda k: np.median(np.min(np.hypot(*(xy[::20, None] - np.array(app[k]['xy'])[None, ::20]).transpose(2, 0, 1)), axis=1)))
            vis = nearest(xy, app[ref]['xy'], visibility[ref])
            segments = m['segments']
            widths = [sg.get('average_width_mm') for sg in segments]
            seg_vis = [float(np.median(vis[seg_of == sg['segment']])) if (seg_of == sg['segment']).any() else None for sg in segments]
            edge = xy[:, 0].min() < 20 or xy[:, 1].min() < 20
            ours = [] if edge else [segments[i]['segment'] for i in segment_check(widths, seg_vis)]
            as_is = [(sg['segment'], sg.get('average_width_mm')) for sg in segments if sg.get('within_tolerance') is False]
            marked = set()
            for sx, sy in info.get('stickers', []):
                d = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)
                if d.min() < STICKER_REACH_PX:
                    marked |= set(range(seg_of[d.argmin()] - 2, seg_of[d.argmin()] + 3))
            note = ('  <- sticker caught' if marked & set(ours) else '  <- sticker missed') if marked else ''
            print(f"{name:36s} main as is: {m['status']} {as_is or ''}\n"
                  f"{'':36s} + visibility + neighbours: {'FAIL ' + str(ours) if ours else 'PASS'}{note}", flush=True)
            by_segment = {sg['segment']: sg.get('average_width_mm') for sg in segments}
            keep = [i for i in range(len(seg_of)) if by_segment.get(seg_of[i]) is not None]
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy[keep], axis=0).T))] * MM_PER_PX
            out[name] = dict(arc=arc.tolist(), w=[by_segment[seg_of[i]] for i in keep], xy=xy[keep].tolist(),
                             vis=vis[keep].tolist(), origin=[0, 0], status=m['status'], failing_segments=ours)
    save_json(data / 'segments' / 'widths.json', out)
    print(f'limit {LIMIT}, visibility >= {VIS_MIN}')


if __name__ == '__main__':
    main()
