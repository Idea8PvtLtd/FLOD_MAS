# The detection flow in use, stage by stage

## The whole flow

![The flow of versions 4 and 5. Each stage does one job, and no stage is trusted for what it is bad at.](figures/master_flow.svg)

In the Pipeline Lab this is: ROI *Panel model*, no pre-processing steps, *SAM - points
placed automatically*, *Smooth ribbon from the true shape* ticked; for version 5 also *SAM
runs on: this PC's GPU*, *Ask SAM once per line* and *Edges placed by: trees*.

## One frame through the main stages

![A pink frame of three panels. Stage 2 marks where glue can be; stage 3 is only a course; stage 4 is the final outline of each line.](figures/photo_stages_frame.jpg)

## Stage 1: panel region

| Aspect | Detail |
|---|---|
| Job | Say where glue can be: one region per panel strip |
| How | U-Net with a ResNet-34 encoder, input 1152 px wide, mirrored second pass; cardboard removed from the mask |
| Code | `glue_line.glue_region`, `pipeline_steps.model_roi`; scripts in `model_for_segment_bra_roi places/` |
| Time | 2.1 s on the CPU (ONNX); 0.25 s on the GPU (PyTorch checkpoint; masks agree, IoU 0.99998) |
| Fails when | a panel is missed: then that line is never looked for |
| Why | everything outside the region (mesh, cups, cardboard) is the source of false lines |

## Stage 2: rough line

| Aspect | Detail |
|---|---|
| Job | Say roughly where each line runs, on the original pixels |
| How | The local detector in region-guided mode: one line inside a corridor along the curved side of each region |
| Code | `segmentation/glue_line.py`, `pipeline_steps.segment_local` |
| Time | 8-11 s with whole-frame filters; 0.7-0.9 s with `guided_skip_frame_filters` (same lines, 44 of 44) |
| Needs to be | within about 20 px; it is a course, not an outline |

## Stage 3: SAM with point prompts

| Aspect | Detail |
|---|---|
| Job | Say which object the line is |
| How | Points every 150 px along the rough line, on 1024 x 1024 tiles of **original** pixels. A far too wide answer is asked again with "not this" points 45 px to each side |
| Code | cloud: `segmentation/sam_refine.py` (SAM 3, Roboflow); local: `segmentation/sam_local.py` (SAM 2.1, half precision, a tile encoded once and asked many times) |
| Time | cloud 9-33 s per frame; local 3.2-4.6 s |
| Fails when | pale fabric: it spills into the band or mesh beside the bead over part of the line |
| Rule | never a reduced frame, never a filtered one |

## Stage 4: leak repair

![Leak repair. Each edge is judged alone because a spill moves one edge and leaves the other.](figures/leak_repair.svg)

In the cloud flow a line outlined on less than 95 % of its edges is put to SAM again, up to
twice, with other tile strides and point spacings (560/110 and 1000/200 px); the steady
parts of the answers are combined. In the local flow each line is asked once, because the
band edges of stage 6 cover what SAM missed.

## Stages 5 to 8: states, band edges, ribbon

![From the line's course to the ribbon. The states decide how much each reading counts.](figures/edges_ribbon.svg)

| State | What the picture shows there | Weight of the band edges | Weight of SAM's edges |
|---|---|---|---|
| shiny | band with a bright core in the middle | 1.0 | 1.0 |
| matte | band without a core | 1.0 | 1.0 |
| weak | the band is faint | 0.4 | 0.6 |
| one-sided | the core lies along one edge, which pulls the band edge inwards | 0.15 | 1.0 |
| none | no band | 0 | 0.3 |

The weights are `STATE_WEIGHT` in `segmentation/ribbon.py`.

![A white frame from the version 4 run: three lines at start, middle and end. Green: the final ribbon. Magenta, thin: SAM's own outline. Red: border of the panel region. Dots: state at each place (cyan shiny, blue matte, red one-sided, orange weak).](figures/photo_states.jpg)

The ribbon fit takes the centre line as a smooth curve and the width as slowly varying,
because that is what the glue is. It reports, per line, the share of its length **backed by
evidence** and the **misfit** in pixels, and marks the line CHECK BY EYE when either is
poor. A stretch where *both* SAM's edges and the band's edges are steadily wider or narrower
than the ribbon, by more than 7 spreads of the line's own noise, is drawn red as a width
deviation.

## What the flow gives on the sample set

| Version 5, 74 frames | Dark | Pink | White |
|---|---|---|---|
| Lines marked "ok" | 126 of 126 | 113 of 113 | 18 of 18 |
| Width, median (range), px | 32.0 (29.6-33.2) | 29.2 (24.8-34.8) | 32.2 (26.4-34.4) |
| Width deviations reported | 0 | 9 | 1 |
| Seconds per frame | 5.7 | 6.4 | 4.7 |

Since the samples have no defects, the ten deviations are false alarms; with the two-source
rule at threshold 7 the count over all 257 lines is 1 (chapter 14).
