"""How sharp the glue edges are along every line, from the photos alone.

    python -m tools.glue_defects.visibility [--data DIR]

Writes visibility.json: per line, the sharpness of the weaker glue edge at each centre
sample (check.edge_visibility). Stretches below check.VIS_MIN are too dark to judge.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import DATA, load_photos, photo_of, save_json
from .check import edge_visibility


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    photos = load_photos(args.data)
    lines = json.loads((args.data / 'app' / 'widths.json').read_text())
    out, cache = {}, {}
    for name, line in lines.items():
        photo = photo_of(name)
        if photo not in cache:
            cache.clear()
            image = cv2.imread(str(args.data / photos[photo]['file']))
            cache[photo] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
        vis = edge_visibility(cache[photo], line['xy'], line['arc'])
        out[name] = vis.tolist()
        print(f'{name:36s} edge sharpness median {np.median(vis):5.1f}  10th percentile {np.percentile(vis, 10):5.1f}')
    save_json(args.data / 'visibility.json', out)


if __name__ == '__main__':
    main()
