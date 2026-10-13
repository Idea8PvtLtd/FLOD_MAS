# Accuracy: the truth bench

Until this work every result was judged against another method or by eye. "Half a pixel
from version 4" says two methods agree, not that either is right. The bench measures error
against a known truth and keeps a method only if the error falls.

## How the bench works

![Painted truth and repeat shots, and the rule for keeping a method.](figures/bench.svg)

- **Painted glue lines on real fabric.** A strip of plain fabric is cut from the same frame
  beside a real line, and a bead is painted into it with the measured depth, edge blur,
  core, dark side lines and loss of texture, along a chosen course with a chosen width. Its
  edges are therefore known exactly.
- **Repeat shots.** Fourteen bursts of 3-5 shots of the same layout (51 lines). The same
  line must read the same in each.
- A painted line is a model of the glue line. Passing the bench is necessary, not
  sufficient.

## Edge methods compared

![Edge error against painted truth on frames held out from training. Lower is better.](figures/chart_edge_error.svg)

| Method | Result | Verdict |
|---|---|---|
| Steepest slope, whole pixels | about 1 px RMS; pulled in by 1.3 px on pink where the core lies along one edge | the baseline (version 4) |
| Same, sub-pixel by parabola | no gain; worse on white | dropped: noise, not pixel size, is the limit |
| Steepest slope at a coarser scale | small gain; worse when blurred | dropped |
| Phase congruency | equal to the baseline | dropped |
| Weave-texture channel alone | 9-94 px | dropped: the texture edge cannot be placed |
| Lightness and texture fused | helps dark only | dropped |
| Parametric profile fit | better (0.7-0.9 px); reads wide when blurred | kept as an option |
| **Trained trees** | **0.63 px**; no bias from a one-sided core or from blur | **kept: version 5** |

**Trees** are gradient-boosted decision trees (LightGBM, 600 trees, 92 features) that
correct the steepest-slope edge from the profile around it in three channels: lightness,
lightness with the core removed, and weave energy. They were trained on painted lines from
52 frames and tested on 22 frames never seen. The model file is
`~/.cache/flod_sam/bead_edge_trees.txt`, written by `tools/train_bead_ml.py`.

## Precision on real repeat shots

| Edge method | Width of the same line between shots, SD px (dark / pink) | Width noise along a line, px |
|---|---|---|
| Steepest slope | 0.10 / 0.13 | 0.73 / 1.26 |
| Profile fit | 0.08 / 0.20 | 0.83 / 1.51 |
| Trees | **0.07 / 0.14** | **0.64 / 1.11** |

The whole-line width repeats to about a tenth of a pixel. Averaging several shots gains
almost nothing: what limits a reading is the fixed pattern of the weave, not camera noise.

## Width defects and false alarms

On the bench the first defect rule found 2 of 192 painted defects. The rule now compares
the line's width with its own running median and scales the threshold by the line's own
noise; it finds 85 of 192, and reliably only a defect of about 8 px over 60 px (about 1 mm
over 8 mm).

Then the user stated that the supplied strips have no defects. Every deviation reported on
them is therefore false, which gives a real false-alarm count:

![Good lines falsely flagged, of 257, by threshold. Band edges alone are unusable at any threshold.](figures/chart_false_alarms.svg)

- **Only the rule that requires SAM's edges and the band's edges to agree is usable.** At
  threshold 7 it flags 1 good line of 257.
- Deviations that recurred across repeat shots were steady artefacts of the picture, not
  real changes. Recurrence does not confirm a defect.

## In millimetres

With about 7.6 px per mm (uncalibrated): edge error of 0.5-0.7 px RMS is about 0.07-0.09
mm; the width of a whole line repeats between shots to 0.01-0.02 mm.

> **What this does not establish.** How many *real* defects the two-source rule finds is
> unknown: there is no defective sample, and the painted bench understates real
> disturbance. This is the largest open question of the whole project.
