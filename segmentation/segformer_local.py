"""A trained SegFormer that marks the glue band, and the band's width read from its mask.

The network is run on native-resolution tiles over the ROI (never on a reduced
frame: its output is a quarter of its input's size) and gives, for every pixel, the
probability of glue. The width at a place on a line is then read across the line
from that mask: the stretch of pixels over one half, with its two ends placed
between pixels where the probability crosses one half.

Needs ``torch`` and ``transformers`` and a model directory saved by training (see
``docs/glue-line/GLUE_LINE_FINDINGS.md``, *SegFormer and CLAHE*). SegFormer's code
and pretrained weights are under NVIDIA's non-commercial licence.
"""
import cv2
import numpy as np

TILE, STRIDE = 512, 384
MEAN = (0.485, 0.456, 0.406)       # ImageNet, RGB
STD = (0.229, 0.224, 0.225)
REACH = 45.0                       # pixels to each side of the line the mask is read
FINE = 0.5                         # spacing of the readings across the line


class GlueMask:
    def __init__(self, path, device=None):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.half = self.device.type == 'cuda'
        model = SegformerForSemanticSegmentation.from_pretrained(str(path)).to(self.device).eval()
        self.model = model.half() if self.half else model
        kind = torch.float16 if self.half else torch.float32
        self.mean = torch.tensor(MEAN, device=self.device, dtype=kind).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device, dtype=kind).view(1, 3, 1, 1)

    def probability(self, image, roi, batch=8):
        """Glue probability for every pixel of the ROI's surroundings (zero elsewhere)."""
        torch = self.torch
        x, y, w, h = cv2.boundingRect((roi > 0).astype(np.uint8))
        x0, y0 = max(0, x - 64), max(0, y - 64)
        x1, y1 = min(image.shape[1], x + w + 64), min(image.shape[0], y + h + 64)
        columns = list(range(x0, max(x0 + 1, x1 - TILE), STRIDE)) + [max(x0, x1 - TILE)]
        rows = list(range(y0, max(y0 + 1, y1 - TILE), STRIDE)) + [max(y0, y1 - TILE)]
        tiles = [(a, b) for b in rows for a in columns if roi[b:b + TILE:16, a:a + TILE:16].any()]
        total = np.zeros(image.shape[:2], np.float32)
        count = np.zeros(image.shape[:2], np.float32)
        with torch.inference_mode():
            for first in range(0, len(tiles), batch):
                chosen = tiles[first:first + batch]
                stack = np.stack([image[b:b + TILE, a:a + TILE] for a, b in chosen])
                pixels = torch.from_numpy(stack).to(self.device).permute(0, 3, 1, 2).flip(1)
                pixels = (pixels.half() if self.half else pixels.float()) / 255
                logits = self.model(pixel_values=(pixels - self.mean) / self.std).logits.float()
                part = torch.sigmoid(torch.nn.functional.interpolate(
                    logits, size=(TILE, TILE), mode='bilinear', align_corners=False))[:, 0].cpu().numpy()
                for (a, b), values in zip(chosen, part):
                    total[b:b + TILE, a:a + TILE] += values
                    count[b:b + TILE, a:a + TILE] += 1
        return total / np.maximum(count, 1)


def mask_edges(probability, points, normals, with_sharpness=False):
    """The mask's two edges across a line, at each sample: signed distances along
    the normal (NaN where the line is not inside the mask or the mask runs out of
    reach). The stretch over one half that lies nearest the line is taken; each end
    is placed between two readings where the probability passes one half.

    ``with_sharpness`` adds a third array: how steeply the probability falls at the
    softer of the two ends (per pixel). A sure mask ends abruptly; where the model
    is unsure the probability fades out, and the end's place means little."""
    offsets = np.arange(-REACH, REACH + FINE / 2, FINE)
    columns = (points[:, None, 0] + offsets[None] * normals[:, None, 0]).astype(np.float32)
    rows = (points[:, None, 1] + offsets[None] * normals[:, None, 1]).astype(np.float32)
    profiles = cv2.remap(probability, columns, rows, cv2.INTER_LINEAR)
    middle = len(offsets) // 2
    low = np.full(len(points), np.nan)
    high = np.full(len(points), np.nan)
    sharp = np.full(len(points), np.nan)
    for index, profile in enumerate(profiles):
        inside = np.flatnonzero(profile > 0.5)
        if inside.size == 0:
            continue
        first = last = inside[np.abs(inside - middle).argmin()]
        while first > 0 and profile[first - 1] > 0.5:
            first -= 1
        while last < len(profile) - 1 and profile[last + 1] > 0.5:
            last += 1
        if first == 0 or last == len(profile) - 1:
            continue
        low[index] = offsets[first] - FINE * (profile[first] - 0.5) / max(profile[first] - profile[first - 1], 1e-6)
        high[index] = offsets[last] + FINE * (profile[last] - 0.5) / max(profile[last] - profile[last + 1], 1e-6)
        wide = int(round(2.0 / FINE))                        # the fall over 2 px around each end
        fall_low = profile[min(first + wide // 2, len(profile) - 1)] - profile[max(first - wide // 2, 0)]
        fall_high = profile[max(last - wide // 2, 0)] - profile[min(last + wide // 2, len(profile) - 1)]
        sharp[index] = min(fall_low, fall_high) / 2.0
    return (low, high, sharp) if with_sharpness else (low, high)
