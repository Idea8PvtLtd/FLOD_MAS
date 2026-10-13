# The two detection pipelines compared

## Side by side

![The research flow (version 5) and the pipeline on `main`. Both begin with the same strip model and both straighten the band; they differ in what reads it.](figures/side_by_side.svg)

| | Version 5 (research tool) | Glue model on `main` |
|---|---|---|
| Finds the object with | SAM 2.1, point prompts, native tiles | SegFormer-B0 trained for this glue |
| Resolution at the glue | native: band about 30 px wide | unfolded band squeezed 4.4-5.8 times along the line and 1.6 times across (one model row is 0.2 mm) |
| Contrast filter | none (measured: filters do not help) | CLAHE |
| Edges | band edges corrected by trained trees | boundary of the cleaned mask |
| Shape knowledge | smooth even-width ribbon | cleaning with a width tolerance |
| Hardware | GPU | CPU |
| Time per frame, same nine frames | 4.3 s | 3.4 s |
| Time for one line | 1.2 s (every line of the frame is done together) | 1.75 s for a capture measuring one line |
| Needs training per product | no | yes |

## Time on the same frames

Nine M1 frames (4 dark, 5 pink), 32 lines; both found the same lines on every frame.

![Seconds per frame by stage, mean of the same nine frames. Version 5 on the GPU, the pipeline on `main` on the CPU.](figures/chart_time_two_pipelines.svg)

| | Version 5 (GPU) | Pipeline on `main` (CPU) |
|---|---|---|
| One frame, all lines: mean (range) | 4.3 s (2.4-5.4) | 3.4 s (2.4-4.1) |
| Largest stage | SAM, 3.1 s (73 %) | CLAHE, 1.1 s (32 %), then the strip U-Net, 0.9 s |
| The stage that reads the glue | SAM 2.1: 3.1 s per frame | SegFormer-B0: 0.09 s per frame |
| Everything after the glue is found | 0.2 s (repair, states, trees, ribbon, widths) | 0.6 s (clean, width on rays) |
| Loading the models | 16 s | 0.65 s |
| GPU memory | 983 MB | none |

- The two take about the same time, on unequal hardware: `main` needs no GPU.
- `main` reads the glue 30 times faster than SAM does. Its time is spent around the model,
  not in it.
- What the research adds after the outline costs 0.2 s per frame. Put behind the model on
  `main` (the tested combination below) the estimate is 3.6 s per frame without a GPU; that
  combination was not timed as one program.

## Measured on the same 74 frames

The model was extracted to `~/flod_experiments/main_model/` and run unchanged on all 74 M1
frames (257 lines).

![Median line width by colour. The same 4 mm strip should read the same on every colour.](figures/chart_widths_by_colour.svg)

| | Dark | Pink | White |
|---|---|---|---|
| Lines found, as one piece | 126 of 126 | 113 of 113 | 18 of 18 |
| Width, median px: main model | 31.0 | 30.9 | 32.6 |
| Width, median px: version 5 | 31.0 | 28.7 | 31.3 |
| Centre line from version 5's, px | 0.68 | 0.81 | 0.80 |
| Width noise from sample to sample, px RMS | 1.25 | 1.29 | 1.35 |
| Largest segment average off the line's mean, px: mean / worst | 1.3 / 4.7 | 3.2 / 5.5 | 1.6 / 3.1 |
| Line width between repeat shots, SD px (version 5) | 0.12 (0.07) | 0.13 (0.21) | - |

- **It works and can be used.** Every line found as one piece, no errors, 3.4 s per frame
  without a GPU.
- **Its widths are equal on the three colours**, which is what one nozzle should give.
  Version 5 reads pink 2.6 px (about 0.3 mm) narrower. Which is right on pink is not known.
- **Its weakness is evenness along a line.** On good pink lines one of the ten segment
  averages is up to 5.5 px (0.7 mm) off the line's mean. A tolerance tighter than that
  would reject good lines.
- On soft white frames of other sets it finds the lines, but with readings far off and
  lines in two or three pieces.

## The one combination that was tested

![A soft white line. The main model's mask as shipped; the same mask finished with the trees and the ribbon; version 5.](figures/photo_main_model.jpg)

| Update to the main model | Result | Verdict |
|---|---|---|
| **Its mask as evidence for the trees and the ribbon** (no retraining) | centre 0.15-0.43 px from version 5; a 17.5 px reading becomes 26.0 px; two false lines refused | **recommended** |
| Enlarge the scores smoothly before choosing the class | edges 1.01 to 0.85 px on dark | harmless, small |
| Retrain in the same format on version 5 ribbons | not more even; worse on soft frames | not recommended |
| Retrain without CLAHE at 2048 x 128 | soft frames much steadier; noisier on M1 | promising; needs more data |

## What merging would involve

Not done. Eleven files were changed on both branches since the common base and would
conflict or need care: `inspection/pipeline.py`, `inspection/result_view.py`,
`camera_handler.py`, `config.json`, `config_validation.py`, `requirements.txt`,
`strip_analysis.py`, `sam_detection.py`, `tests/test_automatic_inspection.py`,
`tests/test_camera_runtime.py`, and `AUTOMATIC_INSPECTION.md` (rewritten on `main`, moved
into `docs/` here; `CONFIGURATION.md` was also edited there and moved here). The research modules themselves
(`segmentation/bead_edges.py`, `ribbon.py`, `bead_ml.py`, `width_stats.py` and the tools)
touch nothing on `main` and would come across unchanged.
