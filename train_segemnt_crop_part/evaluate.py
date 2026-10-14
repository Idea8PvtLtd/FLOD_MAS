"""Accuracy of a trained crop-part segmenter against the hand masks, at full resolution.

    python evaluate.py --run m1m4_split            # held-out frames of that run
    python evaluate.py --run m1m4_split --split all
    python evaluate.py --run neg_all --frames-of colours_split   # another run's held-out frames, to compare models

Per frame: pixel IoU, Dice, precision, recall; per hand-painted strip: IoU of the best-matching
predicted region (found = IoU >= 0.5); boundary error: distance from each predicted outline pixel
to the hand outline and back (mean and 95th percentile, px and mm). Writes runs/<run>/eval_<split>/
(metrics.csv, summary.json, overlays: hand outline green, prediction red).
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import HERE, clean_mask, load_checkpoint, predict_probability, read_manifest

MM_PER_PX = {'Machine1': 0.137}          # measured at the board; Machine4 not confirmed
MIN_STRIP = 0.002                        # hand regions smaller than this share of the frame are ignored


def hand_mask(path):
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return ((rgba[..., :3].max(-1) if rgba.ndim == 3 else rgba) > 127).astype(np.uint8) * 255


def outline(mask):
    m = (mask > 0).astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_distances(pred, gt):
    """Symmetric outline distances in px: every predicted outline pixel to the hand outline and back."""
    po, go = outline(pred), outline(gt)
    if not po.any() or not go.any():
        return np.array([np.inf])
    dt_g = cv2.distanceTransform((~go).astype(np.uint8), cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform((~po).astype(np.uint8), cv2.DIST_L2, 5)
    return np.concatenate([dt_g[po], dt_p[go]])


def metrics(pred, gt, machine):
    p, g = pred > 0, gt > 0
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    out = dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1),
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    n_g, lab_g, st_g, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), 8)
    n_p, lab_p = cv2.connectedComponents(p.astype(np.uint8), 8)
    strips = []
    for k in range(1, n_g):
        if st_g[k, cv2.CC_STAT_AREA] < MIN_STRIP * g.size:
            continue
        gk = lab_g == k
        hits = np.bincount(lab_p[gk], minlength=n_p)
        hits[0] = 0
        j = int(hits.argmax())
        pk = lab_p == j if hits[j] > 0 else np.zeros_like(gk)
        strips.append(float((gk & pk).sum() / max((gk | pk).sum(), 1)))
    d = boundary_distances(pred, gt)
    if not np.isfinite(d).all():
        d = np.array([np.inf])
    scale = MM_PER_PX.get(machine)
    out.update(strips=len(strips), strips_found=sum(s >= 0.5 for s in strips), strip_iou_median=float(np.median(strips)) if strips else 0.0,
               strip_iou_min=float(min(strips)) if strips else 0.0,
               boundary_mean_px=float(d.mean()), boundary_p95_px=float(np.percentile(d, 95)) if np.isfinite(d).all() else float('inf'),
               boundary_mean_mm=None if scale is None else float(d.mean() * scale))
    return out


def overlay(bgr, pred, gt, path):
    vis = bgr.copy()
    for mask, colour in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, colour, 6)
    cv2.imwrite(str(path), cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])


def finite_mean(values):
    v = [x for x in values if np.isfinite(x)]
    return round(float(np.mean(v)), 2) if v else None


def summarize(rows):
    def agg(sel):
        if not sel:
            return {}
        return dict(frames=len(sel), iou_mean=round(float(np.mean([r['iou'] for r in sel])), 4),
                    iou_median=round(float(np.median([r['iou'] for r in sel])), 4),
                    iou_min=round(float(min(r['iou'] for r in sel)), 4),
                    dice_mean=round(float(np.mean([r['dice'] for r in sel])), 4),
                    precision_mean=round(float(np.mean([r['precision'] for r in sel])), 4),
                    recall_mean=round(float(np.mean([r['recall'] for r in sel])), 4),
                    strips=int(sum(r['strips'] for r in sel)), strips_found=int(sum(r['strips_found'] for r in sel)),
                    # outline error only where something was predicted; empty predictions are counted apart
                    frames_without_prediction=int(sum(not np.isfinite(r['boundary_mean_px']) for r in sel)),
                    boundary_mean_px=finite_mean([r['boundary_mean_px'] for r in sel]),
                    boundary_p95_px=finite_mean([r['boundary_p95_px'] for r in sel]))
    out = dict(all=agg(rows))
    for key in sorted({f"{r['machine']}/{r['kind']}" for r in rows}):
        out[key] = agg([r for r in rows if f"{r['machine']}/{r['kind']}" == key])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--checkpoint', default='best.pt')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--frames-of', help='Score on the frames (by --split) of this run instead of the run of the model')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = HERE / 'runs' / args.run
    model, ck = load_checkpoint(run / args.checkpoint, device)
    config = (json.loads((HERE / 'runs' / args.frames_of / 'config.json').read_text(encoding='utf-8'))
              if args.frames_of else ck['config'])
    ids = set(config['val_ids'] if args.split == 'val' else config['train_ids'] if args.split == 'train'
              else config['train_ids'] + config['val_ids'])
    # frames with hand-painted strips only; empty-board frames are checked by check_classification.py
    items = [it for it in read_manifest() if it['id'] in ids and it['machine'] != 'empty']
    out = run / (f'eval_{args.split}_of_{args.frames_of}' if args.frames_of else f'eval_{args.split}')
    out.mkdir(exist_ok=True)
    rows = []
    for it in items:
        bgr = cv2.imread(it['image'])
        gt = hand_mask(it['mask'])
        pred = clean_mask(predict_probability(model, bgr, device), args.threshold)
        m = metrics(pred, gt, it['machine'])
        rows.append(dict(id=it['id'], machine=it['machine'], kind=it['kind'], group=it['group'], **m))
        overlay(bgr, pred, gt, out / f"{it['id']}.jpg")
        print(f"{it['id']}: IoU {m['iou']:.3f}, strips {m['strips_found']}/{m['strips']} (median {m['strip_iou_median']:.3f}),"
              f" outline error mean {m['boundary_mean_px']:.1f} px, p95 {m['boundary_p95_px']:.1f} px", flush=True)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary['checkpoint'] = dict(path=str(run / args.checkpoint), epoch=ck.get('epoch'), threshold=args.threshold)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'checkpoint'}, indent=1))


if __name__ == '__main__':
    main()
