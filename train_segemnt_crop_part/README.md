# Crop-part (paper strip) segmentation: training

Learns the region painted in the hand masks (the bare paper strip beside each glue bead, like
`WIN_20261006_21_43_37_Pro - Edited.jpg`) from the annotated, undistorted captures:

| Machine | Images | Masks |
|---|---|---|
| Machine1 | `latest_captures/Machine1/M1_full_mask/rename_captures/undis (N).jpg` | `.../full_mask/undis (N).png` (59) |
| Machine4 | `latest_captures/Machine4/M4_full_mask/rename_captures/<name>.jpg` | `.../full_mask/<name>.png` (14) |

Masks are RGBA; strip = bright pixels. Images and masks are used as given (no calibration step).

Negatives (no panels, so the whole mask is empty): the 193 photos in `data_calassification/non fabric`.

| Source | Shows | Frames |
|---|---|---|
| camera `2026100?_..._current.jpg` | empty ArUco board on grey cloth | 7 |
| video `vlcsnap-....png` | empty ArUco board, a sheet or hand coming into view in some | 175 |
| webcam `WIN_...jpg`, `original.png` | bare cardboard board | 11 |

`data_calassification/fabric` (186 frames with panels, no hand masks) is used only for checking.

## Steps

    pip install torch segmentation-models-pytorch==0.3.3 albumentations==1.3.1 opencv-python safetensors
    python prepare_data.py                    # data/images, data/masks at 1/4 size (+ negatives), data/manifest.csv
    python train.py --name neg_split          # held-out split from the manifest, ~22 min on an RTX 4060
    python evaluate.py --run neg_split        # accuracy on the held-out frames with strips, full resolution
    python check_classification.py --run neg_split --baseline m1m4_all   # no mask on empty boards; fabric vs old model
    python check_orientation.py --run neg_split                          # same masks at 0/90/180/270 deg and mirrored
    python compare_rule_based.py --run neg_split    # the rule-based areas on the same frames
    python train.py --name neg_all --all      # final model on every frame
    python predict.py --run neg_all --input ../real_captures --output predictions

- `prepare_data.py` groups near-identical consecutive shots of one arrangement and holds out
  whole groups (about 20% of each machine/fabric), so no test frame has a near-twin in training.
  Masks whose size differs from their image are skipped and listed in `data/skipped.csv`.
  Negatives are grouped by capture session (a pause of more than 20 s between shots starts a new one).
  One session per source is held out: camera 00:38 (4), video 01:24 (5) and `original.png`.
  `--negatives ''` leaves the negatives out.
- `train.py`: each epoch takes 4 crops of every frame with strips, plus empty-board crops making up a third
  of the epoch (`--neg-share 0.33`). Empty frames are drawn with equal chance per source, so the 165-frame
  video does not dominate. `log.csv` has `val_neg_fp`, the share of held-out empty frames predicted as strip.
  U-Net with an ImageNet ResNet-34 encoder (loaded from the local Hugging Face cache,
  no download), 512 px random crops of the 1/4-size frames (half centred near a strip), flips, 90°
  turns, ±30° rotation, ±25% scale, brightness/contrast/colour/gamma, blur/noise and JPEG
  artefacts; BCE + Dice loss, AdamW with one-cycle schedule, mixed precision. Keeps the
  checkpoint with the best held-out IoU (`--all`: last epoch). `--folds K --fold i` runs grouped
  cross-validation.
- `evaluate.py`: pixel IoU/Dice/precision/recall per frame, per hand-painted strip the IoU of the
  best-matching predicted region (found = IoU ≥ 0.5), and the outline error (distance between
  predicted and hand outlines, both ways: mean and 95th percentile, px; mm for Machine1 at
  0.137 mm/px). Overlays: hand outline green, prediction red.
- `predict.py`: any image size; the image's long side is scaled to the training width (1152 px), so portrait
  photos keep the training scale. It is predicted
  with a mirrored second pass averaged in, and the probability is resized back to full size.
  Outputs a full-size strip mask, probability map, overlay and a JSON list of strip regions.
  `--compare mask.png` scores a hand mask.

## Results (2026-10-07)

Data: 72 annotated frames (Machine1 59, Machine4 13; one Machine4 mask skipped, see
`data/skipped.csv`). Held-out split: 58 training frames (18 arrangements), 14 held-out frames
(5 arrangements: Machine1 dark 5, pink 2, white 5, Machine4 2). Training ~7.5 s per epoch on an
RTX 4060 laptop GPU; held-out IoU levelled off at 0.991 from about epoch 60.

Held-out frames, full resolution (`runs/m1m4_split/eval_val`), against the rule-based areas of
`segment_beads.py --template` on the same frames (`baseline_rule_based/m1m4_split`):

| Frames | Rule-based IoU | Model IoU | Strips found (model / rule-based) | Model outline error |
|---|---|---|---|---|
| Machine1 dark (5) | 0.60 | 0.991 | 20/20 / 11/20 | 0.9 px mean, 2.3 px p95 |
| Machine1 pink (2) | 0 (no bead found) | 0.989 | 6/6 / 0/6 | 1.2 px, 3.2 px |
| Machine1 white (5) | 0.68 | 0.989 | 14/14 / 12/14 | 1.7 px, 3.1 px |
| Machine4 (2) | 0.93 | 0.989 | 8/8 / 8/8 | 1.2 px, 3.3 px |
| All (14) | 0.59 | 0.990 (worst frame 0.987) | 48/48 / 31/48 | 1.3 px (about 0.17 mm on Machine1), 2.9 px |

Caveats: the 14 held-out frames come from 5 arrangements of the same sessions, so the score says
how well the strips are traced on these machines and fabrics, not on new set-ups. Touching strips
(painted as one region in some masks too) come out as one region; splitting them needs the bead
traces (one bead per strip).

New camera, not in the training data: on `real_captures/WIN_20261006_21_43_37_Pro.jpg` (webcam)
the strip painted in `WIN_20261006_21_43_37_Pro - Edited.jpg` is covered completely (recall 1.00,
IoU 0.91 for the prediction within 15 px of it); the predicted outline lies about 10 px outside
the painted one (median). The touching neighbour strip is joined to it as one region, so the
plain per-region IoU there is 0.45. All six `real_captures` frames give clean outlines
(`checks/real_captures`).

Earlier final model: `runs/m1m4_all/best.pt` (replaced by `runs/neg_all/best.pt`, see below), trained on all 72 frames for 120 epochs (no held-out
frames, so it has no own score; the split run above is its accuracy estimate). On the painted
webcam strip it gives the same IoU 0.91 (recall 1.00).

## Empty boards (2026-10-13): models `neg_split`, `neg_all`

The models above (`m1m4_*`) were trained only on frames with panels. On the 193 `non fabric` photos
`m1m4_all` drew a mask on **all 193** (mean 2.9% of the image, worst 9.2%). The false masks fell on:
- blobs on the bare cardboard and on the grey cloth
- the white squares
- the bright edges at the top and bottom of the frame
- the cardboard offcut at the left

Retrained with the negatives (`neg_split`, same 14 held-out frames with strips):

| Check | `m1m4` | `neg_split` |
|---|---|---|
| Held-out frames with strips: IoU, strips found, outline error | 0.990, 48/48, 1.3 px | 0.989 (worst 0.987), 48/48, 1.3 px |
| Held-out empty frames with a mask (10) | 10/10 | 2/10 |
| Empty frames used in training with a mask (183) | 183/183 | 1/183 |
| Fabric frames with strips (186) | 186/186 | 186/186, mean strip share 12.9% (was 13.1%) |

- The remaining false masks (3 frames, 0.4% of the image) are all in the video where a pink panel is
  being slid in at the left edge, so a real strip edge is in view.
- 7 fabric frames lost more than 20% of their strip area. These are single pink panels on the ArUco
  board, video 01:24–01:25. In them the old model's false blob on the white edge at the bottom
  (about 1% of the image) is gone; the strip itself is 4.0% against 4.2% before.
- Orientation (`check_orientation.py`): held-out IoU is 0.990–0.991 at 0/90/180/270° and mirrored.
  Fabric masks agree with the 0° mask better than with `m1m4` (worst frame 0.76 against 0.57).
  Empty frames get no mask at 180°, 270° or mirrored. At 90° the video frames show a small speck in the
  top-right corner, on a bright object at the edge of the frame. Photos from the camera in its own
  landscape orientation are the supported input.

Final model `runs/neg_all/best.pt`: all 72 frames with strips + 193 empty frames, 120 epochs. Results:
- No mask on any of the 193 `non fabric` frames with the mirrored pass, and on 192/193 with a single pass.
  The one left is 0.2% of the image, where a sheet comes in at the corner.
- All 186 `fabric` frames have strips (mean share 12.9%, was 13.1%).
- No mask on empty frames at any rotation.

Deployed as `model_test_and_running_scripts/models/strip_unet_resnet34.{pt,onnx}` and
`strip_and_topline/models/strip_unet_resnet34.onnx`. The ONNX mask matches the .pt one (IoU 1.000).
The previous model is kept as `*_v1`.

Caveat: the held-out empty frames come from the same sessions and boards as the training ones. A new
empty background (another cloth or board) may need a few more `non fabric` photos.
