# Measurement and grading

## From a polygon to a verdict

![The measurement of one strip. The same code runs for every provider.](figures/measurement.svg)

A provider returns one ring that traces both edges of the strip. A ring alone says nothing
about where the strip runs or how wide it is, so the analysis rebuilds both.

| Step | What is done | Value |
|---|---|---|
| Mask | The ring is filled | a strip under 40 px in area is a speck and is dropped |
| Centreline | Skeleton of the mask; its longest path | under 30 px is too short |
| Resample | Evenly along the path, then smoothed | up to 500 samples; smoothing 2 % of the length |
| Ends | Half a strip width is trimmed from each end | so the rounded tip is not measured |
| Width | A ray marches along the normal both ways until it leaves the mask | 0.5 px steps |
| Skew guard | Measured at three small angles; the middle value is kept | +-0.10 rad |
| Segments | The line is cut into equal lengths | `sam_detection.strip_segments` = 10 |
| Per segment | Average, minimum and maximum width | |

## Optional edge refinement

With `sam_detection.refine_edges` on, each edge is moved from the model's outline onto the
thin dark line that a bead's edge shows in the full-resolution image, within
`refine_search_px` (10). A lone edge that jumps to a fold is rejected by a 9-sample
outlier window. The record then holds the share of samples that were refined. It is off as
shipped.

## Millimetres

Pixels are not multiplied by a single factor. Both end points of each width sample are
mapped through the calibrated measurement plane (chapter 8) and the distance between the
mapped points is the width in millimetres. So perspective across the frame is accounted
for, sample by sample.

If no calibration is saved, the fabric is `UNMEASURED`: pixel figures are shown, a warning
is raised, and nothing is graded.

## Grading

A segment passes when

```text
target - tolerance  <=  segment average width (mm)  <=  target + tolerance
```

with `inspection.strip_width_mm` (4.0) and `inspection.strip_width_tolerance_mm` (1.0); both
can be changed per size profile in the window. A fabric is `PASS` when every segment
passes.

> **What this grading can and cannot see.** A segment is a tenth of the line (about 190 px,
> 25 mm). Its average hides a short narrow or wide spot, and a tolerance of +-1 mm is far
> wider than the 0.1-0.2 mm the user wants resolved. The statistical width test of chapter
> 16 was built for that finer question; it is a research tool and is not wired into the
> application.

## What is saved for each fabric

`result.json` holds, per fabric: status, the polygon in frame pixels, confidence, the
display box, and the measurement record with every segment's figures in pixels and
millimetres. Preview images are JPEG; the numbers do not depend on them.
