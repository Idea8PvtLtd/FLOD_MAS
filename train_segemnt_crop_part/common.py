"""Shared pieces: manifest, model, encoder weights, inference."""
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)
TRAIN_WIDTH = 1152                                   # images are trained at 1/4 of 4608 px


def read_manifest(path=HERE / 'data' / 'manifest.csv'):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_model(encoder='resnet34', pretrained=True):
    """U-Net; ImageNet encoder weights from the local Hugging Face cache (no download)."""
    model = smp.Unet(encoder, encoder_weights=None, classes=1)
    source = 'none'
    if pretrained:
        pattern = os.path.expanduser(f'~/.cache/huggingface/hub/models--smp-hub--{encoder}.imagenet/snapshots/*/model.safetensors')
        files = glob.glob(pattern)
        if files:
            from safetensors.torch import load_file
            state = load_file(files[0])
            keys = set(model.encoder.state_dict())
            model.encoder.load_state_dict({k: v for k, v in state.items() if k in keys}, strict=False)
            source = files[0]
        else:
            print(f'WARNING: no cached ImageNet weights for {encoder}; training from scratch')
    return model, source


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model, _ = build_model(ck['config']['encoder'], pretrained=False)
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_probability(model, bgr, device, flip=True):
    """Strip probability at the input image's full resolution. The image is scaled so its long side is
    TRAIN_WIDTH (the training scale, also for portrait photos), padded to a multiple of 32, predicted
    (with a mirrored pass averaged in), and the probability is resized back."""
    h, w = bgr.shape[:2]
    s = TRAIN_WIDTH / max(h, w)
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    ph, pw = (32 - sh % 32) % 32, (32 - sw % 32) % 32
    rgb = cv2.cvtColor(cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT), cv2.COLOR_BGR2RGB)
    x = to_tensor(rgb)[None].to(device)
    with torch.autocast(device.type, enabled=device.type == 'cuda'):
        p = torch.sigmoid(model(x)).float()
        if flip:
            p = (p + torch.sigmoid(model(torch.flip(x, [3]))).float().flip([3])) / 2
    p = p[0, 0, :sh, :sw].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def clean_mask(prob, threshold=0.5, min_fraction=0.002):
    """Threshold, drop specks, fill holes. Returns uint8 0/255."""
    m = (prob >= threshold).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_fraction * m.size
    m = keep[lab].astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for k in range(1, n):                      # holes: background regions not touching the border
        x, y, w, h = st[k, :4]
        if x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]:
            m[lab == k] = 1
    return m * 255
