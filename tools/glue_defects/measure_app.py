"""Every line of every photo through the app's own models, width without straightening.

    python -m tools.glue_defects.measure_app [--data DIR] [--source-model PATH]

The app's glue mask (ribbon finish off: the raw mask) is measured in the photo as it
is: the width at each centre sample is twice the largest circle that fits inside the
mask there (distance transform), so a curve cannot squeeze it. The centre paths
written here are the lines every other tool of this folder reads.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from app_config import CONFIG
from inspection.local.engine import LiveInspection

from . import DATA, MM_PER_PX, load_photos, save_json

ROOT = Path(__file__).resolve().parents[2]
DISC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))   # the circle may sit 7 px off the sample


def mask_widths(mask, xy, margin=60):
    """Widths (px) at the centre samples, the crop of the line's own mask and its origin."""
    xy = np.asarray(xy, float)
    x0, y0 = np.maximum(xy.min(0).astype(int) - margin, 0)
    x1, y1 = xy.max(0).astype(int) + margin
    crop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    at = np.clip(np.rint(xy - [x0, y0]).astype(int), 0, [crop.shape[1] - 1, crop.shape[0] - 1])
    _, labels = cv2.connectedComponents(crop)
    own = labels[at[:, 1], at[:, 0]]
    own = np.unique(own[own > 0])
    if len(own):
        crop = np.isin(labels, own).astype(np.uint8)
    radius = cv2.dilate(cv2.distanceTransform(crop, cv2.DIST_L2, cv2.DIST_MASK_PRECISE), DISC)[at[:, 1], at[:, 0]]
    return (2 * radius - 1).clip(0), crop, (int(x0), int(y0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--source-model', help='strip U-Net to use instead of the configured one')
    args = parser.parse_args()
    s = CONFIG['local_inspection']
    engine = LiveInspection(args.source_model or ROOT / s['source_model'], ROOT / s['glue_model'],
                            s['offset_pixels'], s['source_width'], smooth_normal_rows=101, smooth_logits=True,
                            min_line_length_ratio=.5, ribbon_finish=False, segments=40,
                            execution_provider=s.get('execution_provider', 'cpu'))
    out = {}
    (args.data / 'app' / 'masks').mkdir(parents=True, exist_ok=True)
    for photo, info in load_photos(args.data).items():
        image = cv2.imread(str(args.data / info['file']))
        source, lines = engine.find_lines(image, sideways=True)
        picks = [(line, k + 1) for k, line in enumerate(lines)]
        for k, result in enumerate(engine.measure_found(image, source, picks, len(lines))):
            if not isinstance(result, dict) or 'mask' not in result:
                continue
            main = max(result['components'], key=lambda c: c['length_px'])
            xy = np.array([[x['center_x'], x['center_y']] for x in main['samples']])
            arc = np.array([x['arc_px'] for x in main['samples']]) * MM_PER_PX
            width, crop, origin = mask_widths(result['mask'], xy)
            name = f'{photo}_L{k + 1}'
            cv2.imwrite(str(args.data / 'app' / 'masks' / f'{name}.png'), crop * 255)
            out[name] = dict(arc=(arc - arc[0]).tolist(), w=(width * MM_PER_PX).tolist(), xy=xy.tolist(),
                             origin=list(origin))
            print(f'{name:36s} {arc[-1] - arc[0]:5.0f} mm  median width {np.median(width) * MM_PER_PX:.2f} mm', flush=True)
    save_json(args.data / 'app' / 'widths.json', out)


if __name__ == '__main__':
    main()
