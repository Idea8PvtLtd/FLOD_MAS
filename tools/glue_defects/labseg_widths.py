"""Glue width from the pipeline-lab SegFormers, on the photo as it is.

    tools/jetson/run_glue_defects.sh labseg segformer_b2_plain [segformer_b0_plain segformer_b0_clahe]

Research venv only. Each model (``~/flod_lab/trained_models/<name>``) gives a glue
probability for every pixel from native-resolution 512 px tiles over a strip around
the line (segmentation/segformer_local.GlueMask). The width at each centre sample is
read across the line: the run over one half nearest the line, its ends between the
pixels where the probability crosses one half (mask_edges). A ``clahe`` model gets
the frame with CLAHE on lightness first, as it was trained (clip 3, 16x16).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import DATA, MM_PER_PX, load_photos, path_normals, photo_of, save_json

STRIP = 70          # the model is run within this distance of the line


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('models', nargs='+')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--lab', type=Path, default=Path.home() / 'flod_lab', help='pipeline-lab checkout')
    args = parser.parse_args()
    sys.path.insert(0, str(args.lab))
    from segmentation.segformer_local import GlueMask, mask_edges
    photos = load_photos(args.data)
    lines = json.loads((args.data / 'app' / 'widths.json').read_text())
    for model in args.models:
        glue = GlueMask(args.lab / 'trained_models' / model)
        out_dir = args.data / f'lab_{model}'
        (out_dir / 'masks').mkdir(parents=True, exist_ok=True)
        out, cache = {}, {}
        for name, line in lines.items():
            photo = photo_of(name)
            if photo not in cache:
                cache.clear()
                image = cv2.imread(str(args.data / photos[photo]['file']))
                if 'clahe' in model:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    lab[..., 0] = cv2.createCLAHE(3.0, (16, 16)).apply(lab[..., 0])
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                cache[photo] = image
            image = cache[photo]
            started = time.time()
            xy = np.array(line['xy'], float)
            roi = np.zeros(image.shape[:2], np.uint8)
            cv2.polylines(roi, [np.rint(xy).astype(np.int32)], False, 255, 2 * STRIP)
            probability = glue.probability(image, roi)
            low, high = mask_edges(probability, xy, path_normals(xy))
            width = np.where(np.isfinite(low) & np.isfinite(high), (high - low) * MM_PER_PX, 0.0)
            x0, y0 = np.maximum(xy.min(0).astype(int) - 60, 0)
            x1, y1 = xy.max(0).astype(int) + 60
            crop = (probability[y0:y1, x0:x1] > .5) & (roi[y0:y1, x0:x1] > 0)
            cv2.imwrite(str(out_dir / 'masks' / f'{name}.png'), crop.astype(np.uint8) * 255)
            out[name] = dict(arc=line['arc'], w=width.tolist(), xy=line['xy'], origin=[int(x0), int(y0)],
                             seconds=time.time() - started)
            print(f'{model:20s} {name:36s} median {np.median(width):.2f} mm  {time.time() - started:4.1f} s', flush=True)
        save_json(out_dir / 'widths.json', out)


if __name__ == '__main__':
    main()
