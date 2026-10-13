# The trained segmenter (SegFormer and CLAHE)

## Training and use

![The segmenter is trained on the flow's own best outlines and then used as a second source of evidence.](figures/segformer.svg)

SegFormer is a transformer segmentation network. It was fine-tuned to mark the glue band,
using the version 5 ribbons of the M1 frames as labels, so nothing was labelled by hand.

| | SegFormer-B0 | SegFormer-B2 |
|---|---|---|
| Parameters | 3.7 M | 27.3 M |
| Training | 3000 iterations, batch 6, 7.5-16 min | 4000 iterations, batch 3, 19 min |
| GPU memory, training / running | 1.8 GB / 0.66 GB | 3.3 GB / 1.9 GB |
| Time for a frame's mask | 0.3-0.9 s | 0.8-1.3 s |

- 52 frames to train on, 22 held out, cut per colour by capture time so repeat shots stay on
  one side. 2 102 tiles of 512 px at native resolution; glue is 5.5 % of the pixels.
- Loss: cross-entropy plus Dice. Augmentation: flips, transposition, brightness jitter.
- The network outputs at a quarter of the input resolution.

## The real test: frames of other pieces, never trained on

![A soft white frame from another image set. Version 5 with SAM 2.1 leaves the bead at one end; the SegFormer mask follows it.](figures/photo_segformer.jpg)

| Soft white frames, 17 lines | Version 5 (SAM 2.1) | SegFormer-B0 | B0 + CLAHE | **SegFormer-B2** |
|---|---|---|---|---|
| Width range, px | 22-36 | 26.4-32.0 | 26.4-31.2 | **28.0-32.0** |
| Outline source covers the line | 0-84 % | 60-95 % | 0-87 % | **69-100 %** |
| Lines marked "check by eye" | 9 of 17 | 1 of 17 | 1 of 17 | 1 of 17 |

It is the first method that handles the soft white frames, although it was trained on sharp
frames only, four of them white.

## CLAHE

Trained and run with CLAHE on lightness (clip 3, 16 x 16 grid): no gain on the M1 frames,
every figure equal to within 0.03 px, and worse on the soft frames, where one line gets no
mask at all. This agrees with every filter result for SAM and for the local detector: on
these pictures a contrast filter does not help a segmenter, and on soft pictures it hurts.

## Against SAM

| | SAM 2.1 large (version 5) | SegFormer-B2 |
|---|---|---|
| Needs a rough line first | yes | no |
| Needs training on this product | no | yes, about 20 minutes on this GPU |
| A new fabric colour or product | works untrained (three colours measured) | unknown; expect retraining |
| Soft frames | weak | good |
| Time for the outline | 3-4.5 s | about 1 s |
| Licence of the weights | Apache 2.0 | **non-commercial** (NVIDIA) |

## State

- Not in the Pipeline Lab. `segmentation/segformer_local.py` loads a trained model folder
  and is used by the width report (chapter 16).
- Models and training scripts are in `~/flod_experiments/segformer/`, outside the
  repository.
- Agreement with version 5 on held-out M1 frames shows it learned the task, not that it is
  more accurate: it was trained on those ribbons.
- Use in the product needs NVIDIA's agreement, or the same training on an openly licensed
  network such as the U-Net already used for the region model. That was not tried.
