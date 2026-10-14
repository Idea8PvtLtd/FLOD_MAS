"""Build the training set for the crop-part (paper strip) segmenter.

Reads the colour folders of dataset/ (see dataset/inventory.csv):
  dataset/<colour>/images/<id>.<ext>  +  dataset/<colour>/masks/<id>.png     colour: ash, pink, white, blue, red
Only images that have a mask are used; images still without one are counted and skipped.
Masks are RGBA (hand-painted) or plain grey; a pixel is strip where its colour is bright (> 127).
The source of an image is in its id: m1_ Machine1, m4_ Machine4, blue_cap_ the older 4000x3000 rig,
blue_WIN_ the webcam, blue_vid / *_vid the machine video, red_ the red stills.

Writes data/images/<id>.jpg and data/masks/<id>.png scaled so the long side is TRAIN_WIDTH (1152 px,
the model's input scale; a 4608 px capture is 1/4 size as before, a 4000 px one keeps the same scale),
and data/manifest.csv.
Consecutive shots of one arrangement are near-identical, so frames are grouped (a new group starts
when a shot differs clearly from the previous one) and the held-out split takes whole groups, about
VAL_SHARE of each machine and fabric, so no test frame has a near-twin in training.
The Machine1/Machine4 frames get the same groups, split and pixels as the earlier latest_captures
layout: they are read in the same order and their strata sort first, so the random split draws for
them are unchanged; new sources are split after them.

Negatives: every photo in dataset/negative/images (empty ArUco board, bare cardboard; no panels) is
added with an all-zero mask as machine 'empty', kind photo / video / webcam / app_video by file name.
Their groups are capture sessions (SESSION_GAP_S between shots), and the sessions in NEG_VAL_SESSIONS
are held out.

    python prepare_data.py [--dataset dataset] [--out data]
"""
import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN_WIDTH = 1152       # long side of every stored frame: the model's input scale (common.TRAIN_WIDTH)
COLOURS = ('ash', 'pink', 'white', 'blue', 'red')   # read in this order; Machine1 capture order is ash, pink, white
VAL_SHARE = 0.2
NEW_GROUP_DIFF = 6.0     # mean grey difference of 96x72 thumbnails that starts a new arrangement
SEED = 0
SESSION_GAP_S = 20       # negatives: a longer pause between shots starts a new capture session
# held-out negative sessions, one per source: (kind, first shot of the session or file stem).
# The video session began at 01:24:02 before its two first snapshots, exact copies of earlier frames,
# were removed from dataset/negative; the same session now starts at 01:24:07.
NEG_VAL_SESSIONS = {('photo', '2026-10-07 00:38:36'), ('video', '2026-10-08 01:24:07'), ('webcam', 'original')}
EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def shot_time(name):
    """Capture time from the camera (20261007_002557_570749_...), VLC (vlcsnap-2026-10-08-01h23m25s541)
    or Windows camera (WIN_20261008_03_29_12_Pro) file name; None if there is none."""
    m = re.match(r'(\d{8})_(\d{6})_(\d{6})_', name)
    if m:
        return dt.datetime.strptime(m[1] + m[2], '%Y%m%d%H%M%S') + dt.timedelta(microseconds=int(m[3]))
    m = re.match(r'vlcsnap-(\d{4}-\d{2}-\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})', name)
    if m:
        return dt.datetime.strptime(f'{m[1]} {m[2]}{m[3]}{m[4]}', '%Y-%m-%d %H%M%S') + dt.timedelta(milliseconds=int(m[5]))
    m = re.match(r'WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})', name)
    if m:
        return dt.datetime.strptime(m[1] + m[2] + m[3] + m[4], '%Y%m%d%H%M%S')
    return None


def negative_kind(name):
    if name.startswith('vid'):
        return 'app_video'
    return 'video' if name.startswith('vlcsnap') else 'photo' if re.match(r'\d{8}_\d{6}_', name) else 'webcam'


def negatives(folder):
    """Empty-board photos, in capture order per source, with their session key. Files are named
    neg_<original name>; the original name carries the source and capture time."""
    out = []
    for p in sorted(q for q in Path(folder).iterdir() if q.suffix.lower() in EXTS):
        name = p.name[4:] if p.name.startswith('neg_') else p.name
        out.append(dict(id=p.stem if p.stem.startswith('neg_') else f'neg_{p.stem}', machine='empty',
                        kind=negative_kind(name), time=shot_time(name), image=p, name_stem=Path(name).stem))
    out.sort(key=lambda it: (it['kind'], it['time'] or dt.datetime.max, it['image'].name))
    last = {}
    for it in out:
        prev = last.get(it['kind'])
        if it['time'] is None:
            it['session'] = it['name_stem']
        elif prev is None or prev['time'] is None or (it['time'] - prev['time']).total_seconds() > SESSION_GAP_S:
            it['session'] = it['time'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            it['session'] = prev['session']
        last[it['kind']] = it
    return out


def scaled_size(w, h):
    """(width, height) with the long side TRAIN_WIDTH; 4608x3456 gives 1152x864, exactly 1/4."""
    s = TRAIN_WIDTH / max(w, h)
    return round(w * s), round(h * s)


def machine_of(item_id):
    """The capture source named by an id. New sources sort after Machine1/Machine4 (see module doc)."""
    for prefix, machine in (('m1_', 'Machine1'), ('m4_', 'Machine4'), ('blue_cap_', 'OldRig'),
                            ('blue_WIN_', 'Webcam'), ('blue_vid', 'Video'), ('red_', 'Stills')):
        if item_id.startswith(prefix):
            return machine
    return 'Video' if '_vid' in item_id else 'Other'


def pairs(dataset):
    """Labelled images per colour, in capture order (ids sort in capture order within a source)."""
    out, unlabelled = [], {}
    for colour in COLOURS:
        folder = dataset / colour
        images = {p.stem: p for p in (folder / 'images').glob('*') if p.suffix.lower() in EXTS}
        masks = {p.stem: p for p in (folder / 'masks').glob('*.png')}
        for stem in sorted(set(masks) - set(images)):
            raise SystemExit(f'Mask without an image: {masks[stem]}')
        unlabelled[colour] = len(set(images) - set(masks))
        for k, stem in enumerate(sorted(masks)):
            machine = machine_of(stem)
            order = int(stem[3:]) if machine == 'Machine1' else k
            out.append(dict(id=stem, machine=machine, order=order, kind=colour, image=images[stem], mask=masks[stem]))
    return out, unlabelled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=str(HERE / 'dataset'), help='Folder with the colour folders')
    ap.add_argument('--out', default=str(HERE / 'data'))
    ap.add_argument('--negatives', default=None,
                    help="Folder of photos without panels (empty mask); default dataset/negative/images; '' to leave them out")
    args = ap.parse_args()
    if args.negatives is None:
        args.negatives = str(Path(args.dataset) / 'negative' / 'images')
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    items, unlabelled = pairs(Path(args.dataset))
    if not items:
        raise SystemExit(f'No image/mask pairs found under {args.dataset}')
    print(f'unlabelled images skipped (no mask yet): {unlabelled}')
    previous, previous_kind = {}, {}
    group = 0
    skipped = []
    for it in items:
        bgr = cv2.imread(str(it['image']), cv2.IMREAD_COLOR)
        rgba = cv2.imread(str(it['mask']), cv2.IMREAD_UNCHANGED)
        if bgr is None or rgba is None:
            raise SystemExit(f"Cannot read {it['image']} or {it['mask']}")
        if rgba.shape[:2] != bgr.shape[:2]:
            # a mask exported from a cropped or resized copy cannot be placed without guessing: skip it
            print(f"SKIPPED {it['id']}: mask {rgba.shape[1]}x{rgba.shape[0]} does not match image {bgr.shape[1]}x{bgr.shape[0]}")
            skipped.append(dict(id=it['id'], image=str(it['image']), mask=str(it['mask']),
                                reason=f'mask {rgba.shape[1]}x{rgba.shape[0]} vs image {bgr.shape[1]}x{bgr.shape[0]}'))
            continue
        strip = (rgba[..., :3].max(-1) if rgba.ndim == 3 else rgba) > 127
        h, w = bgr.shape[:2]
        size = scaled_size(w, h)
        small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        # area-averaged mask, then threshold: the strip outline stays where half the pixels are strip
        msmall = cv2.resize(strip.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5
        cv2.imwrite(str(out / 'images' / f"{it['id']}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / 'masks' / f"{it['id']}.png"), msmall.astype(np.uint8) * 255)
        thumb = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (96, 72), interpolation=cv2.INTER_AREA).astype(np.float32)
        last = previous.get(it['machine'])
        diff = None if last is None else float(np.abs(thumb - last).mean())
        if diff is None or diff > NEW_GROUP_DIFF or it['kind'] != previous_kind.get(it['machine']):
            group += 1
        previous_kind[it['machine']] = it['kind']
        previous[it['machine']] = thumb
        it.update(group=group, diff_to_previous=None if diff is None else round(diff, 2), strip_fraction=round(float(strip.mean()), 4),
                  width=w, height=h)
        print(f"{it['id']}: group {group} ({'-' if diff is None else f'{diff:.1f}'}), strip {strip.mean():.1%}", flush=True)
    items = [it for it in items if 'group' in it]
    with (out / 'skipped.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'reason', 'image', 'mask'])
        writer.writeheader()
        writer.writerows(skipped)
    # held-out split: whole groups, about VAL_SHARE of each machine/fabric stratum
    rng = np.random.default_rng(SEED)
    for stratum in sorted({(it['machine'], it['kind']) for it in items}):
        groups = sorted({it['group'] for it in items if (it['machine'], it['kind']) == stratum})
        n_val = max(1, round(VAL_SHARE * len(groups))) if len(groups) > 1 else 0
        val = set(rng.choice(groups, n_val, replace=False).tolist()) if n_val else set()
        for it in items:
            if (it['machine'], it['kind']) == stratum:
                it['split'] = 'val' if it['group'] in val else 'train'
    # negatives after the positive split, so adding them leaves the positive split unchanged
    sessions = {}
    for it in negatives(args.negatives) if args.negatives else []:
        bgr = cv2.imread(str(it['image']), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Cannot read {it['image']}")
        h, w = bgr.shape[:2]
        size = scaled_size(w, h)
        cv2.imwrite(str(out / 'images' / f"{it['id']}.jpg"), cv2.resize(bgr, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / 'masks' / f"{it['id']}.png"), np.zeros(size[::-1], np.uint8))
        key = (it['kind'], it['session'])
        if key not in sessions:
            group += 1
            sessions[key] = group
        it.update(group=sessions[key], split='val' if key in NEG_VAL_SESSIONS else 'train', order=it['session'],
                  strip_fraction=0.0, width=w, height=h, mask='')
        items.append(it)
    missing = NEG_VAL_SESSIONS - set(sessions) if args.negatives else set()
    if missing:
        print(f'WARNING: held-out negative sessions not found: {sorted(missing)}')
    fields = ['id', 'machine', 'kind', 'group', 'split', 'order', 'diff_to_previous', 'strip_fraction', 'width', 'height',
              'image', 'mask']
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for it in items:
            writer.writerow({**it, 'image': str(it['image']), 'mask': str(it['mask'])})
    print(f"negatives: {sum(it['machine'] == 'empty' for it in items)} frames in {len(sessions)} sessions "
          f"{ {f'{k}/{s}': sum(1 for it in items if it['machine'] == 'empty' and (it['kind'], it['session']) == (k, s)) for k, s in sessions} }")
    for split in ('train', 'val'):
        sel = [it for it in items if it['split'] == split]
        kinds = {}
        for it in sel:
            kinds[f"{it['machine']}/{it['kind']}"] = kinds.get(f"{it['machine']}/{it['kind']}", 0) + 1
        print(f'{split}: {len(sel)} frames in {len({it["group"] for it in sel})} groups {kinds}')


if __name__ == '__main__':
    main()
