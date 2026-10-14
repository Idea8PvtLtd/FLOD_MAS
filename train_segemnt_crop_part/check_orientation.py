"""Does the photo's orientation change the strip mask? Every frame is predicted at the training scale
turned 0/90/180/270 degrees and mirrored, and the mask is turned back.

  (a) held-out frames with hand masks (the run's val_ids): IoU with the hand mask per orientation
  (b) data_calassification fabric frames: IoU with the 0-degree mask (should stay near 1)
  (c) data_calassification non-fabric frames: frames with any mask per orientation (should be 0)

    python check_orientation.py --run neg_split
    python check_orientation.py --run neg_all --step 1        # every classification frame
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import HERE, TRAIN_WIDTH, clean_mask, load_checkpoint, read_manifest, to_tensor

TURNS = {'0': (0, False), '90': (1, False), '180': (2, False), '270': (3, False), 'mirror': (0, True)}


@torch.no_grad()
def turned_probability(model, small, k, mirror, device):
    im = small[:, ::-1] if mirror else small
    im = np.ascontiguousarray(np.rot90(im, k))
    h, w = im.shape[:2]
    im = cv2.copyMakeBorder(im, 0, (32 - h % 32) % 32, 0, (32 - w % 32) % 32, cv2.BORDER_REFLECT)
    x = to_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))[None].to(device)
    with torch.autocast(device.type, enabled=device.type == 'cuda'):
        p = torch.sigmoid(model(x)).float()[0, 0, :h, :w].cpu().numpy()
    p = np.rot90(p, -k)
    return np.ascontiguousarray(p[:, ::-1] if mirror else p)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', default='best.pt')
    ap.add_argument('--root', default=str(HERE.parent / 'data_calassification'))
    ap.add_argument('--step', type=int, default=3, help='Use every n-th classification frame')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_checkpoint(HERE / 'runs' / args.run / args.checkpoint, device)
    masks = lambda small, k, m: clean_mask(turned_probability(model, small, k, m, device)) > 0
    summary = {}
    val = set(ck['config']['val_ids'])
    items = [it for it in read_manifest() if it['id'] in val and it['machine'] != 'empty']
    if items:
        res = {t: [] for t in TURNS}
        for it in items:
            small = cv2.imread(str(HERE / 'data' / 'images' / f"{it['id']}.jpg"))
            gt = cv2.imread(str(HERE / 'data' / 'masks' / f"{it['id']}.png"), 0) > 127
            for t, (k, m) in TURNS.items():
                res[t].append(iou(masks(small, k, m), gt))
        summary['held_out_vs_hand_mask'] = {t: dict(iou_mean=round(np.mean(v), 4), iou_min=round(min(v), 4)) for t, v in res.items()}
    for label, folder in (('fabric', 'fabric'), ('non_fabric', 'non fabric')):
        files = sorted(p for p in (Path(args.root) / folder).iterdir() if p.suffix.lower() in {'.jpg', '.png'})[::args.step]
        share, agree = {t: [] for t in TURNS}, {t: [] for t in TURNS}
        for f in files:
            bgr = cv2.imread(str(f))
            s = TRAIN_WIDTH / max(bgr.shape[:2])
            small = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            base = masks(small, 0, False)
            for t, (k, m) in TURNS.items():
                mk = base if t == '0' else masks(small, k, m)
                share[t].append(float(mk.mean()))
                agree[t].append(iou(mk, base) if base.any() or mk.any() else 1.0)
        summary[label] = {t: dict(frames=len(files), frames_with_mask=sum(v > 0 for v in share[t]),
                                  strip_share_mean=round(float(np.mean(share[t])), 5),
                                  iou_with_0_mean=round(float(np.mean(agree[t])), 4), iou_with_0_min=round(float(min(agree[t])), 4))
                          for t in TURNS}
    out = HERE / 'runs' / args.run / 'check_orientation.json'
    out.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
