"""Segment the crop part (paper strips) in any photo(s) with a trained model.

    python predict.py --run m1m4_all --input ../real_captures --output predictions
    python predict.py --run m1m4_all --input photo.jpg --output out --compare painted_mask.png

Per image: <stem>_strip_mask.png (full resolution, 255 = strip), <stem>_strip_prob.png (probability
x255), <stem>_strip_overlay.jpg (quarter size, outline red) and <stem>_strips.json (one entry per
separate strip region: area, bounding box). --compare scores a hand mask of one or more strips:
for every hand region, the IoU of the predicted region that overlaps it most.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import HERE, clean_mask, load_checkpoint, predict_probability

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def compare(pred, hand):
    g = hand > 127
    n_g, lab_g = cv2.connectedComponents(g.astype(np.uint8))
    n_p, lab_p = cv2.connectedComponents((pred > 0).astype(np.uint8))
    out = []
    for k in range(1, n_g):
        gk = lab_g == k
        hits = np.bincount(lab_p[gk], minlength=n_p)
        hits[0] = 0
        j = int(hits.argmax())
        pk = lab_p == j if hits[j] else np.zeros_like(gk)
        out.append(dict(hand_region=k, iou=round(float((gk & pk).sum() / max((gk | pk).sum(), 1)), 4),
                        hand_px=int(gk.sum()), predicted_px=int(pk.sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', default='best.pt')
    ap.add_argument('--input', required=True, nargs='+', help='Image files, folders or glob patterns')
    ap.add_argument('--output', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--no-flip', action='store_true', help='Skip the mirrored second pass (faster)')
    ap.add_argument('--compare', help='Hand mask (white = strip) for a single input image')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_checkpoint(HERE / 'runs' / args.run / args.checkpoint, device)
    files = []
    for spec in args.input:
        for src in map(Path, sorted(glob.glob(spec)) or [spec]):
            files += sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f'skip {f}: unreadable')
            continue
        t0 = time.perf_counter()
        prob = predict_probability(model, bgr, device, flip=not args.no_flip)
        seconds = time.perf_counter() - t0
        mask = clean_mask(prob, args.threshold)
        n, lab, st, cen = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        strips = [dict(id=k, area_px=int(st[k, 4]), bbox_xywh=[int(v) for v in st[k, :4]],
                       centre=[round(float(v), 1) for v in cen[k]]) for k in range(1, n)]
        cv2.imwrite(str(out / f'{f.stem}_strip_mask.png'), mask)
        cv2.imwrite(str(out / f'{f.stem}_strip_prob.png'), (prob * 255).astype(np.uint8))
        vis = bgr.copy()
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 6)
        report = dict(image=str(f), run=args.run, epoch=ck.get('epoch'), threshold=args.threshold, strips=strips)
        if args.compare:
            hand = cv2.imread(args.compare, cv2.IMREAD_GRAYSCALE)
            if hand.shape != mask.shape:
                raise SystemExit(f'--compare mask {hand.shape} does not match image {mask.shape}')
            report['comparison'] = compare(mask, hand)
            hc, _ = cv2.findContours((hand > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, hc, -1, (0, 255, 0), 6)
        cv2.imwrite(str(out / f'{f.stem}_strip_overlay.jpg'), cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / f'{f.stem}_strips.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
        extra = f", hand-mask IoU {[c['iou'] for c in report['comparison']]}" if args.compare else ''
        sizes = ', '.join(f"{s['area_px'] / mask.size:.1%}" for s in strips)
        print(f'{f.name}: {len(strips)} strip regions ({sizes}), {seconds:.2f} s{extra}', flush=True)


if __name__ == '__main__':
    main()
