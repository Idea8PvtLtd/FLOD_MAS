"""Train the crop-part (paper strip) segmenter: U-Net, ImageNet ResNet-34 encoder.

    python prepare_data.py            # once: builds data/ from latest_captures
    python train.py --name m1m4       # trains on split=train, keeps the best held-out IoU
    python train.py --name all --all  # final model on every annotated frame (no held-out score)
    python train.py --name cv1 --folds 5 --fold 1   # grouped cross-validation fold

Writes runs/<name>/best.pt, last.pt, log.csv, config.json.
"""
import argparse
import json
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import HERE, build_model, read_manifest, to_tensor


def augment(crop):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),                       # sideways panels
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.RandomCrop(crop, crop),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
        A.HueSaturationValue(20, 30, 20, p=0.6),        # fabric colours: pink, blue, white, dark
        A.RandomGamma((70, 150), p=0.3),
        A.OneOf([A.GaussianBlur((3, 5)), A.MotionBlur(5), A.GaussNoise((5, 30))], p=0.4),
        A.ImageCompression(50, 100, p=0.4),             # webcam JPEG blocks
    ])


def is_negative(it):
    return it['machine'] == 'empty'


class Frames(Dataset):
    """Training: per epoch, per_image random crops of every frame with strips, plus enough crops of
    empty-board frames (negatives) to make up neg_share of the epoch; negatives are drawn with equal
    chance per source (photo / video / webcam) so one long video does not dominate.
    Validation: every frame once, whole."""

    def __init__(self, items, crop, per_image, train, neg_share=0.0):
        self.items, self.crop, self.per_image, self.train = items, crop, per_image, train
        self.images = [cv2.cvtColor(cv2.imread(str(HERE / 'data' / 'images' / f"{it['id']}.jpg")), cv2.COLOR_BGR2RGB)
                       for it in items]
        self.masks = [(cv2.imread(str(HERE / 'data' / 'masks' / f"{it['id']}.png"), 0) > 127).astype(np.uint8)
                      for it in items]
        self.aug = augment(crop) if train else None
        self.pos = [k for k, it in enumerate(items) if not is_negative(it)]
        neg = [k for k, it in enumerate(items) if is_negative(it)]
        self.n_pos = len(self.pos) * per_image
        self.n_neg = round(self.n_pos * neg_share / (1 - neg_share)) if train and neg and neg_share > 0 else 0
        kinds = {}
        for k in neg:
            kinds.setdefault(items[k]['kind'], []).append(k)
        self.neg = neg
        self.neg_p = np.array([1 / (len(kinds) * len(kinds[items[k]['kind']])) for k in neg]) if neg else None

    def __len__(self):
        return self.n_pos + self.n_neg if self.train else len(self.items)

    def __getitem__(self, i):
        if not self.train:
            k = i
        elif i < self.n_pos:
            k = self.pos[i % len(self.pos)]
        else:
            k = self.neg[np.random.choice(len(self.neg), p=self.neg_p)]
        img, mask = self.images[k], self.masks[k]
        if self.train:
            # half the crops are centred near the strip so thin strips are seen often enough
            if np.random.rand() < 0.5 and mask.any():
                ys, xs = np.nonzero(mask)
                j = np.random.randint(len(ys))
                h, w = mask.shape
                big = int(self.crop * 1.6)
                y0 = int(np.clip(ys[j] - big // 2, 0, max(0, h - big)))
                x0 = int(np.clip(xs[j] - big // 2, 0, max(0, w - big)))
                img, mask = img[y0:y0 + big, x0:x0 + big], mask[y0:y0 + big, x0:x0 + big]
            out = self.aug(image=img, mask=mask)
            img, mask = out['image'], out['mask']
        else:   # whole frame, padded to a multiple of 32
            h, w = mask.shape
            ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)   # 255 = ignore
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))


def seed_worker(worker_id):
    """Different numpy random state per worker and epoch (albumentations draws from it)."""
    np.random.seed(torch.initial_seed() % 2 ** 32)


def loss_fn(logits, target):
    valid = (target != 255).float()
    t = (target == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction='none') * valid).sum() / valid.sum()
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice


def split(items, args):
    if args.all:
        return items, []
    if args.folds:
        rng = np.random.default_rng(args.seed)
        val_groups = set()
        for stratum in sorted({(it['machine'], it['kind']) for it in items}):
            groups = sorted({int(it['group']) for it in items if (it['machine'], it['kind']) == stratum})
            rng.shuffle(groups)
            val_groups |= {g for i, g in enumerate(groups) if i % args.folds == args.fold - 1}
        return ([it for it in items if int(it['group']) not in val_groups],
                [it for it in items if int(it['group']) in val_groups])
    return [it for it in items if it['split'] == 'train'], [it for it in items if it['split'] == 'val']


@torch.no_grad()
def validate(model, loader, device, negative):
    """IoU over the frames with strips (pooled, and mean per frame) and, over the empty-board frames,
    the mean share of the image predicted as strip (should be 0)."""
    inter = union = 0.0
    per, fp = [], []
    for j, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            p = torch.sigmoid(model(x)).float()
        valid = y != 255
        pred, gt = (p[:, 0] >= 0.5) & valid, (y == 1) & valid
        if negative[j]:
            fp.append(pred.sum().item() / valid.sum().item())
            continue
        i, u = (pred & gt).sum().item(), (pred | gt).sum().item()
        inter += i
        union += u
        per.append(i / max(u, 1))
    return (inter / max(union, 1), float(np.mean(per)) if per else 0.0,
            float(np.mean(fp)) if fp else float('nan'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='run')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--per-image', type=int, default=4, help='Random crops per frame with strips per epoch')
    ap.add_argument('--neg-share', type=float, default=0.33, help='Share of each epoch taken by empty-board crops')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all', action='store_true', help='Train on every frame (final model)')
    ap.add_argument('--folds', type=int, default=0, help='Grouped cross-validation: number of folds')
    ap.add_argument('--fold', type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    items = read_manifest()
    train_items, val_items = split(items, args)
    out = HERE / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=True)
    model, weights = build_model(args.encoder)
    model.to(device)
    config = dict(vars(args), encoder_weights=weights, train_ids=[it['id'] for it in train_items],
                  val_ids=[it['id'] for it in val_items], train_width=1152)
    (out / 'config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
    train_loader = DataLoader(Frames(train_items, args.crop, args.per_image, True, args.neg_share), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(Frames(val_items, args.crop, 1, False), batch_size=1) if val_items else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    best = -1.0
    log = open(out / 'log.csv', 'w', encoding='utf-8')
    val_negative = [is_negative(it) for it in val_items]
    log.write('epoch,loss,val_iou,val_mean_frame_iou,val_neg_fp,lr,seconds\n')
    print(f'train {len(train_items)} frames ({sum(map(is_negative, train_items))} empty), val {len(val_items)} frames '
          f'({sum(val_negative)} empty), {len(train_loader.dataset)} crops per epoch, encoder weights: {weights}', flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                loss = loss_fn(model(x)[:, 0].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        model.eval()
        iou, frame_iou, neg_fp = validate(model, val_loader, device, val_negative) if val_loader else (float('nan'),) * 3
        state = dict(model=model.state_dict(), config=config, epoch=epoch, val_iou=iou)
        torch.save(state, out / 'last.pt')
        if val_loader and iou > best:
            best = iou
            torch.save(state, out / 'best.pt')
        line = f'{epoch},{total / len(train_loader):.4f},{iou:.4f},{frame_iou:.4f},{neg_fp:.5f},{sched.get_last_lr()[0]:.2e},{time.time() - t0:.1f}'
        log.write(line + '\n')
        log.flush()
        print(line, flush=True)
    if not val_loader:
        torch.save(state, out / 'best.pt')
    print(f'best held-out IoU {best:.4f}' if val_loader else 'trained on all frames', flush=True)


if __name__ == '__main__':
    main()
