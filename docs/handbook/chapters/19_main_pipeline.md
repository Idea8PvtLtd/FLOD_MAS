# The inspection pipeline on `main`

## From a photo to graded segments

![The pipeline of `LiveInspection` and `measurement_record`, read from the source.](figures/main_pipeline.svg)

| Stage | Code on `main` | Detail |
|---|---|---|
| Source mask | `local/source_model.py` | the same strip U-Net as the research's region model, as ONNX on the CPU |
| Lines | `local/geometry.py`: `extract_lines`, `filter_wave_lines` | the curved side of each strip; hooks at the ends trimmed; lines without a wave refused |
| Band | `offset_lines`, `region_mask` | a second line `offset_pixels` (100) inside; glue is looked for between the two |
| Enhance | `enhance_image` | CLAHE, clip 2, 16 x 16 tiles |
| Unfold | `unfold_maps` | the band straightened along its middle line: 4500-6000 x 100 px for a line 2200-2900 px long, because the inner edge of the band zigzags (review finding M3) |
| Glue model | `local/glue_model.py` | band squeezed to 1024 x 64; SegFormer-B0, two classes; class chosen per pixel, enlarged by nearest neighbour |
| Clean | `clean_mask` | profile smoothing, gaps up to 12, width tolerance 15 %, rounded tips |
| Refold | `mapped_mask` | the mask mapped back into the photo |
| Measure | `local/width_measurement.py` | per connected piece: centreline, then width on perpendicular rays every 5 px in 0.25 px steps, with a sub-pixel crossing |
| Grade | `inspection/line_measurement.py` | 5 % of each end left out; ten equal segments; both edge points of each sample mapped to mm; times the user ratio |

Pieces of glue are measured separately; a gap is never bridged. The status is `PASS` when
every segment's average is inside target +- tolerance, `FAIL` when a graded segment is
outside, and `WARNING` when any segment could not be graded.

## How long it takes

Measured on the development PC (i5-13420H, no GPU used), nine frames, 32 lines, with the
model code of `main` run unchanged.

| | Seconds |
|---|---|
| One capture measuring one line, as the app runs it | 1.75 (1.6-1.9) |
| One frame with all its lines (2-4) | 3.4 (2.4-4.1) |
| Loading the two models, once | 0.65 |

| Stage | Per frame, s | Per line, s | Note |
|---|---|---|---|
| Strip U-Net (source mask) | 0.91 | | once per frame |
| Source lines: extract, wave filter, offset | 0.32 | | once per frame |
| CLAHE on the whole photo | 1.08 | 0.31 | computed again for every line |
| Unfold, crop, refold | 0.10 | 0.03 | |
| SegFormer-B0 | 0.09 | 0.026 | |
| Clean mask | 0.31 | 0.09 | |
| Width on rays | 0.31 | 0.09 | |
| Not timed singly | 0.30 | | remaps, connected pieces, overlay |

> **Finding:** the glue model is 26 ms per line. The largest cost is CLAHE, because the
> whole 16-megapixel photo is enhanced again for each line. Once per frame would save about
> 0.3 s for every line after the first.

The trigger wait (hands absent for 1.0 s), the fabric check, the undistortion and the saving
were read, not timed. The Windows inspection PC was not timed.

## Review findings

The full review, with file and line for each point, is `docs/glue-line/MAIN_BRANCH_REVIEW.md`.

| # | Severity | Finding | Evidence |
|---|---|---|---|
| M1 | High | Only the average of each 25 mm segment is graded, at +- 1.0 mm. A 2 mm blob must be 4.2 mm wider than the strip to fail; 5 mm long, 2.9 mm wider | 12 real masks, widened by drawing |
| M2 | High | A line in two in-tolerance pieces passes; a line with no glue found shows WARNING, not FAIL | code; soft frames |
| M3 | Medium | The band's inner edge zigzags: 18-20 % of its steps go backwards, so the strip is 4500-6000 px long | 8 lines |
| M4 | Medium | Moving the photo one pixel changes a segment average by 0.35 px on average and 4.0 px (0.53 mm) at worst | 37 frames |
| M5 | Medium | One model row is 0.2 mm across the line; the class is chosen before enlarging | model output 64 x 1024 |
| M6 | Medium | Soft white frames: good lines fail on one segment, lines break, extra lines appear | 3 frames |
| M7 | Medium | Changing the camera rotation does not invalidate the calibration | read |

Taking the band's normal from smoothed edge points (model not retrained) leaves segment
evenness as it is (worst segment 5.35 against 5.21 px) but improves repeatability: the
worst change under a one-pixel shift falls from 4.0 to 1.1 px.

## Trigger, queue and reset

![How a capture is taken and how the bed is reset, read from `auto_trigger/controller.py`.](figures/main_trigger.svg)

- **Lines are counted from the left** and a position names the same physical line in every
  capture of a cycle, because panels are only ever added on the right.
- Each capture measures the lines not yet inspected, leftmost first, and leaves one for
  each capture already queued behind it. A line that failed is offered again.
- **Reset.** When results are showing or work is in hand, two ArUco markers with ID 0 on
  the bed, both visible for half a second, mean the bed has been cleared. The controller
  waits for running inspections, shows the last result for two seconds and forgets which
  lines were inspected.
- The fabric check and the undistortion run on a separate worker thread so the frame loop
  is not held up.

## What is saved

One folder per capture, as on this branch, with a `line_N` folder inside for each line
measured: its mask, its overlay and its measurement. The record's method is
`local_onnx_lines`.

## Compared with the application on this branch

| | This branch (Part B) | `main` |
|---|---|---|
| Finds glue with | a provider: cloud workflow by default | two local ONNX models |
| Needs a connection | yes, by default | no |
| Unit of result | one fabric | one line |
| Several panels | all in one capture, up to four | one more line per capture, tracked across captures |
| Cleared by | fabric classifier saying "no fabric" | bed markers becoming visible |
| Ends of the line | half a width trimmed | 5 % of the length left out |
| Statuses | PASS, FAIL and four ungraded kinds | PASS, FAIL, WARNING |
| Width read by | rays across the provider's polygon | rays across the cleaned mask |
