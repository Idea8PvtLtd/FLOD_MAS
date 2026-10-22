"""Glue mask from SAM 2.1 on the photo as it is, then the width along each line.

    tools/jetson/run_glue_defects.sh sam [--model sam2.1-large] [--clahe]

Research venv only (PyTorch, transformers, the pipeline-lab checkout at ~/flod_lab for
segmentation.sam_local). Each line (centre path from app/widths.json) is covered by
overlapping 512 px tiles of the photo, enlarged 2x to SAM's own size. SAM is given
points on both halves of the band ("glue") and beside it ("not glue"); of its three
candidate masks the one that best covers the band's strip is kept. The tiles' masks
are voted together at twice the photo's resolution, and the width at each centre
sample is twice the largest circle inside the mask there. ``--clahe`` adds local
contrast to each tile first (geometry unchanged).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import DATA, MM_PER_PX, load_photos, path_normals, photo_of, save_json

TILE = 512          # photo pixels per tile edge
ZOOM = 2            # the tile is enlarged to 1024, SAM's own size; the vote canvas uses the same
STRIDE = 300        # line length between tile centres, photo pixels
POS_STEP = 40       # spacing of the "glue" points along the line
SIDE = 7.0          # "glue" points this far each side of the centre too
NEG = 45.0          # "not glue" points this far each side of the line
BAND_HALF = 20.0    # the band lies within this distance of the centre line
NEAR = 40.0         # the mask is kept within this distance of the line
EDGE = 24           # tile border (photo px) whose votes are not counted


def resample(xy, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    u = np.arange(0, d[-1], step)
    c = np.c_[np.interp(u, d, xy[:, 0]), np.interp(u, d, xy[:, 1])]
    return c, path_normals(c)


def line_mask(sam, image, xy, clahe):
    """SAM's glue mask around one line at ZOOM x the photo's resolution, and its origin."""
    height, width = image.shape[:2]
    c, n = resample(xy)
    x0, y0 = np.maximum(c.min(0).astype(int) - TILE // 2, 0)
    x1, y1 = np.minimum(c.max(0).astype(int) + TILE // 2, [width, height])
    cw, ch = (x1 - x0) * ZOOM, (y1 - y0) * ZOOM
    votes = np.zeros((ch, cw), np.float32)
    cover = np.zeros((ch, cw), np.float32)
    weight = np.zeros((TILE * ZOOM, TILE * ZOOM), np.float32)
    weight[EDGE * ZOOM:-EDGE * ZOOM, EDGE * ZOOM:-EDGE * ZOOM] = 1
    scores = []
    for i in list(range(0, len(c), STRIDE)) + [len(c) - 1]:
        tx0 = int(np.clip(c[i, 0] - TILE / 2, 0, width - TILE))
        ty0 = int(np.clip(c[i, 1] - TILE / 2, 0, height - TILE))
        tile = cv2.resize(image[ty0:ty0 + TILE, tx0:tx0 + TILE], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        if clahe:
            lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(lab[..., 0])
            tile = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        inside = np.where((c[:, 0] > tx0 + 20) & (c[:, 0] < tx0 + TILE - 20) &
                          (c[:, 1] > ty0 + 20) & (c[:, 1] < ty0 + TILE - 20))[0]
        if len(inside) < 10:
            continue
        pick = inside[::POS_STEP]
        to_tile = lambda p: {'x': (p[0] - tx0) * ZOOM, 'y': (p[1] - ty0) * ZOOM}
        points = [dict(to_tile(c[j] + s * n[j]), positive=True) for j in pick for s in (-SIDE, 0, SIDE)]
        for j in pick[::2]:
            for s in (-1, 1):
                p = c[j] + s * NEG * n[j]
                if tx0 < p[0] < tx0 + TILE and ty0 < p[1] < ty0 + TILE:
                    points.append(dict(to_tile(p), positive=False))
        strip = np.zeros((TILE * ZOOM, TILE * ZOOM), np.uint8)
        cv2.polylines(strip, [np.rint((c[inside] - [tx0, ty0]) * ZOOM).astype(np.int32)], False, 1, int(2 * BAND_HALF * ZOOM))

        def fit(candidate):
            m = candidate[0] > 0
            return (m & (strip > 0)).sum() - 2.0 * (m & (strip == 0)).sum()
        mask, score = max(sam.ask(sam.open(tile), points, multimask=True), key=fit)
        scores.append(score)
        ox, oy = (tx0 - x0) * ZOOM, (ty0 - y0) * ZOOM
        a0, b0 = max(ox, 0), max(oy, 0)
        a1, b1 = min(ox + TILE * ZOOM, cw), min(oy + TILE * ZOOM, ch)
        part = mask.astype(np.float32) * weight
        votes[b0:b1, a0:a1] += part[b0 - oy:b1 - oy, a0 - ox:a1 - ox]
        cover[b0:b1, a0:a1] += weight[b0 - oy:b1 - oy, a0 - ox:a1 - ox]
    near = np.zeros((ch, cw), np.uint8)
    pts = np.rint((c - [x0, y0]) * ZOOM).astype(np.int32)
    cv2.polylines(near, [pts], False, 1, int(2 * NEAR * ZOOM))
    mask = ((votes > .5 * cover) & (cover > 0) & (near > 0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    on = labels[np.clip(pts[:, 1], 0, ch - 1), np.clip(pts[:, 0], 0, cw - 1)]
    on = np.unique(on[on > 0])
    if len(on):
        mask = np.isin(labels, on).astype(np.uint8)
    return mask, (int(x0), int(y0)), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', default='sam2.1-large')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--lab', type=Path, default=Path.home() / 'flod_lab', help='pipeline-lab checkout')
    args = parser.parse_args()
    sys.path.insert(0, str(args.lab))
    from segmentation import sam_local
    sam = sam_local.get(args.model)
    name_out = f"sam_{args.model}{'_clahe' if args.clahe else ''}"
    (args.data / name_out / 'masks').mkdir(parents=True, exist_ok=True)
    photos = load_photos(args.data)
    lines = json.loads((args.data / 'app' / 'widths.json').read_text())
    out, cache = {}, {}
    for name, line in lines.items():
        photo = photo_of(name)
        if photo not in cache:
            cache.clear()
            cache[photo] = cv2.imread(str(args.data / photos[photo]['file']))
        started = time.time()
        xy = np.array(line['xy'], float)
        mask, (x0, y0), scores = line_mask(sam, cache[photo], xy, args.clahe)
        radius = cv2.dilate(cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15 * ZOOM, 15 * ZOOM)))
        at = np.rint((xy - [x0, y0]) * ZOOM).astype(int)
        at[:, 0] = np.clip(at[:, 0], 0, mask.shape[1] - 1)
        at[:, 1] = np.clip(at[:, 1], 0, mask.shape[0] - 1)
        width = (2 * radius[at[:, 1], at[:, 0]] / ZOOM - 1 / ZOOM).clip(0) * MM_PER_PX
        small = cv2.resize(mask * 255, (mask.shape[1] // ZOOM, mask.shape[0] // ZOOM), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.data / name_out / 'masks' / f'{name}.png'), np.uint8(small > 127) * 255)
        out[name] = dict(arc=line['arc'], w=width.tolist(), xy=line['xy'], origin=[x0, y0],
                         sam_score=float(np.mean(scores)) if scores else None, seconds=time.time() - started)
        print(f'{name:36s} tiles {len(scores):2d}  score {np.mean(scores):.2f}  median {np.median(width):.2f} mm'
              f'  {time.time() - started:5.1f} s', flush=True)
    save_json(args.data / name_out / 'widths.json', out)


if __name__ == '__main__':
    main()
