"""Run a trained model on the fabric / non-fabric photos (dataset/check_fabric, dataset/negative).

Non-fabric frames (empty ArUco board, bare cardboard) should give no strip region at all; fabric
frames (panels on the board, no hand masks) should still give strips. Per frame: number of strip
regions and predicted strip share of the image (after clean_mask). Writes
runs/<run>/check_classification/ (metrics.csv, summary.json, overlays of every non-fabric frame
with a region and of fabric frames with none).

    python check_classification.py --run m1m4_all
    python check_classification.py --run neg_split --only-val     # held-out negatives only
    python check_classification.py --run neg_all --baseline m1m4_all   # fabric frames against an earlier run

--baseline reads that run's check_classification/metrics.csv and flags fabric frames whose strip
share dropped by more than DROP (as a share of the baseline).
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import HERE, clean_mask, load_checkpoint, predict_probability

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
DROP = 0.2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', default='best.pt')
    ap.add_argument('--root', default=str(HERE / 'dataset'),
                    help='Folder with negative/images (empty boards) and check_fabric/images (fabric photos)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--only-val', action='store_true', help='Non-fabric frames: only those held out by this run')
    ap.add_argument('--out', help='Output folder (default runs/<run>/check_classification)')
    ap.add_argument('--baseline', help='Earlier run whose check_classification/metrics.csv to compare fabric frames with')
    args = ap.parse_args()
    base = {}
    if args.baseline:
        with open(HERE / 'runs' / args.baseline / 'check_classification' / 'metrics.csv', newline='', encoding='utf-8') as fh:
            base = {r['file']: float(r['strip_share']) for r in csv.DictReader(fh) if r['label'] == 'fabric'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = HERE / 'runs' / args.run
    model, ck = load_checkpoint(run / args.checkpoint, device)
    trained_on = set(ck['config'].get('train_ids', []))
    held_out = set(ck['config'].get('val_ids', []))
    out = Path(args.out) if args.out else run / 'check_classification'
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    rows = []
    for label, folder in (('non_fabric', Path('negative', 'images')), ('fabric', Path('check_fabric', 'images'))):
        for f in sorted(p for p in (Path(args.root) / folder).iterdir() if p.suffix.lower() in EXTS):
            nid = f.stem if f.stem.startswith('neg_') else f'neg_{f.stem}'   # files are stored as neg_<name>
            role = 'train' if nid in trained_on else 'val' if nid in held_out else '-'
            if args.only_val and label == 'non_fabric' and role != 'val':
                continue
            bgr = cv2.imread(str(f))
            prob = predict_probability(model, bgr, device)
            mask = clean_mask(prob, args.threshold)
            n, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
            areas = sorted((st[1:, cv2.CC_STAT_AREA] / mask.size).tolist(), reverse=True)
            rows.append(dict(label=label, file=f.name, role=role, regions=n - 1, strip_share=round(float((mask > 0).mean()), 5),
                             largest_region=round(areas[0], 5) if areas else 0.0, max_prob=round(float(prob.max()), 3)))
            r = rows[-1]
            dropped = False
            if label == 'fabric' and f.name in base:
                r['baseline_share'] = base[f.name]
                dropped = r['strip_share'] < (1 - DROP) * base[f.name]
            extra = f" (baseline {r['baseline_share']:.2%}{', DROPPED' if dropped else ''})" if 'baseline_share' in r else ''
            print(f"{label:10s} {role:5s} {f.name}: {r['regions']} regions, {r['strip_share']:.2%}, max p {r['max_prob']:.2f}{extra}", flush=True)
            if (label == 'non_fabric') == (n > 1) or dropped:    # a mask where none belongs, none or much less where strips belong
                vis = bgr.copy()
                cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cnts, -1, (0, 0, 255), 12)
                cv2.imwrite(str(out / 'overlays' / f'{label}_{f.stem}.jpg'),
                            cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) + (['baseline_share'] if base else []), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    def agg(sel):
        out = dict(frames=len(sel), frames_with_mask=sum(r['regions'] > 0 for r in sel),
                   strip_share_mean=round(float(np.mean([r['strip_share'] for r in sel])), 5) if sel else None,
                   strip_share_max=round(float(max((r['strip_share'] for r in sel), default=0)), 5),
                   max_prob_max=round(float(max((r['max_prob'] for r in sel), default=0)), 3))
        compared = [r for r in sel if 'baseline_share' in r]
        if compared:
            out.update(baseline_share_mean=round(float(np.mean([r['baseline_share'] for r in compared])), 5),
                       frames_dropped=sum(r['strip_share'] < (1 - DROP) * r['baseline_share'] for r in compared))
        return out
    summary = {f'{label}/{role}': agg([r for r in rows if r['label'] == label and r['role'] == role])
               for label, role in sorted({(r['label'], r['role']) for r in rows})}
    summary['checkpoint'] = dict(path=str(run / args.checkpoint), epoch=ck.get('epoch'), threshold=args.threshold)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
