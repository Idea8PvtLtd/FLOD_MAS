"""Score the existing rule-based areas (segment_beads.py --template) on the same frames and metrics.

The annotated frames are the renamed captures (rotated 180 degrees from the raw camera image); the
rule-based pipeline was built on the raw orientation, so each frame is turned back by 180 degrees
(an exact pixel rotation), processed, and the area masks are turned again before scoring.

    python compare_rule_based.py --run m1m4_split        # that run's held-out frames
Writes baseline_rule_based/<split ids hash>/metrics.csv, summary.json and overlays.
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common import HERE, read_manifest
from evaluate import hand_mask, metrics, overlay, summarize

PKG = HERE.parent / 'glue_lines_package'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='Use the held-out frames of this training run')
    args = ap.parse_args()
    sys.path.insert(0, str(PKG))
    import segment_beads as sb
    ids = json.loads((HERE / 'runs' / args.run / 'config.json').read_text(encoding='utf-8'))['val_ids']
    items = [it for it in read_manifest() if it['id'] in ids]
    out = HERE / 'baseline_rule_based' / args.run
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for it in items:
            bgr = cv2.imread(it['image'])
            raw = cv2.rotate(bgr, cv2.ROTATE_180)
            src = tmp / f"{it['id']}.jpg"
            cv2.imwrite(str(src), raw, [cv2.IMWRITE_JPEG_QUALITY, 97])
            a = sb.parser().parse_args(['--input', str(src), '--output', str(tmp / 'o'), '--template',
                                        str(PKG / 'templates' / 'cup_panel_bead.json'), '--no-measure-width'])
            (tmp / 'o').mkdir(exist_ok=True)
            sb.process(src, a, tmp / 'o')
            union = np.zeros(raw.shape[:2], np.uint8)
            for f in (tmp / 'o' / it['id']).glob('area_*.png'):
                union |= cv2.imread(str(f), 0)
            pred = cv2.rotate(union, cv2.ROTATE_180)
            gt = hand_mask(it['mask'])
            m = metrics(pred, gt, it['machine'])
            rows.append(dict(id=it['id'], machine=it['machine'], kind=it['kind'], group=it['group'], **m))
            overlay(bgr, pred, gt, out / f"{it['id']}.jpg")
            shutil.rmtree(tmp / 'o' / it['id'], ignore_errors=True)
            print(f"{it['id']}: rule-based IoU {m['iou']:.3f}, strips {m['strips_found']}/{m['strips']}, "
                  f"outline error mean {m['boundary_mean_px']:.1f} px", flush=True)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
