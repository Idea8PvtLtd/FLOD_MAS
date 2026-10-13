# Trained models

Weights trained on this project's photos during the glue-line experiments, with the scripts
that made them. They are stored with Git LFS: install `git-lfs` and run `git lfs pull` after
cloning, otherwise each weight file is a small text pointer.

None of these is loaded by the inspection app. What was measured with them is in
`docs/glue-line/GLUE_LINE_FINDINGS.md`.

| Folder | Model | Input | Made by |
|---|---|---|---|
| `segformer_b0_plain/` | SegFormer-B0, one class (glue) | tiles of the photo, no CLAHE | `train.py` |
| `segformer_b0_clahe/` | SegFormer-B0, one class | tiles after CLAHE | `train.py` |
| `segformer_b2_plain/` | SegFormer-B2, one class | tiles of the photo, no CLAHE | `train.py` |
| `main_retrained_U1/` | SegFormer-B0 in the format of the `main` engine | unfolded band, CLAHE, 1024 x 64 | `retrain.py` |
| `main_retrained_U2/` | the same | unfolded band, no CLAHE, 2048 x 128 | `retrain.py` |
| `roi_unet/roi_unet.pt` | U-Net, ResNet-34 encoder: the glue-bearing panel in a whole frame | 768 x 576 | `roi_unet/train_roi.py` |

- Each SegFormer folder loads with
  `SegformerForSemanticSegmentation.from_pretrained(<folder>)` (package `transformers`).
  `eval.json` holds the per-frame results of `evalseg.py` on the held-out frames.
- `roi_unet.pt` is the checkpoint of `segmentation/models/glue_roi.onnx`; the ONNX file is
  the one the local detector (`segmentation/glue_line.py`) and `tools/roi_sweep.py` read.
- The strip model of the `main` engine, with its checkpoint, is in
  `model_for_segment_bra_roi places/model_test_and_running_scripts/models/`.

## Scripts

`prep.py` (tiles and labels from the version 5 ribbons), `train.py`, `evalseg.py`,
`retrain.py`, `retrain_eval.py` and `roi_unet/train_roi.py` are copies of the experiment
scripts. They were run from `~/flod_experiments/` and keep the paths of that machine:
change the paths at the top before running them elsewhere. The training photos
(`enhancement_images/`) are not in the repository.

## Licence

The SegFormer weights are fine-tuned from NVIDIA's pretrained encoders, which NVIDIA
releases for non-commercial use (see the licence note in the findings). Settle that before
any of the SegFormer models is used in production.

## Not included

Downloaded third-party weights (FastSAM, MobileSAM, SAM 2.1) are not copied here; they come
from their own publishers.
