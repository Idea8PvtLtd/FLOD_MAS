"""U-Net that marks the glue-bearing panel (ROI) in a whole frame, from the M1 masks."""
import glob, re, os, random, time, cv2, numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
M = '/home/admin-123/Documents/GitHub/FLOD_MAS/enhancement_images/M1_Masks/'
OUT = '/home/admin-123/flod_experiments/roi/'
W, H = 768, 576
MEAN = np.array([0.485, 0.456, 0.406]) * 255; STD = np.array([0.229, 0.224, 0.225]) * 255
TEST = {5, 12, 20, 28, 35, 40, 45, 64, 70, 74}      # held out, spread over colours and layouts
def pairs():
    caps = sorted(glob.glob(M + 'captures/*.jpg'))
    masks = {int(re.search(r'\((\d+)\)', f).group(1)): f for f in glob.glob(M + 'full_mask/*')}
    out = []
    for i in sorted(masks):
        im = cv2.resize(cv2.imread(caps[i - 1]), (W, H), interpolation=cv2.INTER_AREA)
        mk = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED)[..., 0][::-1, ::-1]      # masks are rotated 180 degrees
        mk = (cv2.resize(np.ascontiguousarray(mk), (W, H), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
        out.append((i, im, mk))
    return out
def tens(im):
    return torch.from_numpy(((im[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()).float()
def aug(im, mk):
    if random.random() < .5: im, mk = im[:, ::-1], mk[:, ::-1]
    if random.random() < .5: im, mk = im[::-1], mk[::-1]
    a = random.uniform(-8, 8); s = random.uniform(.85, 1.15); tx, ty = random.uniform(-40, 40), random.uniform(-30, 30)
    T = cv2.getRotationMatrix2D((W / 2, H / 2), a, s); T[:, 2] += (tx, ty)
    im = cv2.warpAffine(np.ascontiguousarray(im), T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mk = cv2.warpAffine(np.ascontiguousarray(mk), T, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    im = im.astype(np.float32) * random.uniform(.75, 1.25) + random.uniform(-25, 25)
    if random.random() < .5:                                    # colour shifts: fabric colours vary
        im = im * np.random.uniform(.85, 1.15, 3)
    if random.random() < .3: im = cv2.GaussianBlur(im, (0, 0), random.uniform(.5, 1.5))
    return np.clip(im, 0, 255).astype(np.uint8), mk
if __name__ == '__main__':
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    data = pairs(); train = [d for d in data if d[0] not in TEST]; test = [d for d in data if d[0] in TEST]
    print('train', len(train), 'test', len(test), flush=True)
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1).cuda()
    ITERS = 1800; opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=ITERS); scaler = torch.amp.GradScaler(); t0 = time.time()
    for it in range(ITERS):
        b = [aug(*random.choice(train)[1:]) for _ in range(6)]
        x = torch.stack([tens(i) for i, _ in b]).cuda(); y = torch.from_numpy(np.stack([m for _, m in b])).float().cuda()[:, None]
        with torch.autocast('cuda', dtype=torch.float16):
            lg = model(x).float()
        p = torch.sigmoid(lg); loss = F.binary_cross_entropy_with_logits(lg, y) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 300 == 0: print('iter', it, 'loss %.4f' % loss.item(), '%.0fs' % (time.time() - t0), flush=True)
    model.eval(); torch.save(model.state_dict(), OUT + 'roi_unet.pt')
    ious, tiles = [], []
    with torch.no_grad():
        for i, im, mk in data:
            pr = torch.sigmoid(model(tens(im)[None].cuda()))[0, 0].cpu().numpy() > .5
            iou = (pr & (mk > 0)).sum() / max((pr | (mk > 0)).sum(), 1)
            if i in TEST:
                ious.append(iou); v = im.copy()
                cv2.drawContours(v, cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
                cv2.drawContours(v, cv2.findContours(pr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
                cv2.putText(v, '#%d IoU %.2f' % (i, iou), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2); tiles.append(v)
                print('held-out #%d IoU %.3f' % (i, iou), flush=True)
    print('held-out mean IoU %.3f min %.3f' % (np.mean(ious), np.min(ious)))
    while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(OUT + 'heldout.jpg', np.vstack([np.hstack(tiles[k:k + 4]) for k in (0, 4, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # ONNX for use from OpenCV on the inspection PC
    torch.onnx.export(model.cpu(), torch.zeros(1, 3, H, W), OUT + 'roi_unet.onnx', opset_version=17, input_names=['image'], output_names=['logits'])
    print('DONE', flush=True)
