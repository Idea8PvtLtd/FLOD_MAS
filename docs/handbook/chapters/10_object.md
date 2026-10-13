<!-- part: Part C - Glue-line detection research (this branch) -->
# The object and why it is hard

Part C describes research done with tools that are separate from the application. The full
record, with every table and every method that failed, is
`docs/glue-line/GLUE_LINE_FINDINGS.md`; this part is its structure and its pictures.

## What the glue line looks like

![The middle of one glue line on each fabric colour, as photographed and with contrast raised. Each picture is about 260 px (34 mm) across.](figures/photo_object.jpg)

- The glue is a thin glossy bead running along the curved edge of each panel strip. It
  makes a **band** about 30 px (4 mm) wide that is darker than the fabric, with a bright
  **core** that shows only where the light catches it.
- Every line has the same wave: two humps and a dip. Mirror images occur.
- A frame holds two to four lines, one per panel.
- Beside the bead on pale fabric lies a paler smooth band, 30-40 px wide. It is **not** the
  bead. Methods that outline 55-70 px have taken both together.

## The signal, in numbers

![Left: how much darker the band is than the fabric. Right: weave texture inside the band as a share of the fabric's. Measured on 37 frames, 46 000 samples.](figures/chart_signal.svg)

| | Dark | Pink | White |
|---|---|---|---|
| Band darker than fabric, grey levels | 43-50 | 7 | 7 |
| Blur of the band's edge, px (sigma) | 2.2 | 2.1-2.2 | 2.0-2.1 |
| Shiny core over the band, grey levels | 122 | 16 | 29 |
| Weave texture inside the band, share of the fabric's | 0.21 | 0.24 | 0.39 |
| Width, px | 31-32 | 30-31 | 33-34 |

Two consequences run through everything that follows.

1. **On pale fabric the signal is a seventh of what it is on dark fabric**, and it is of the
   same size as the weave's own pattern. Nothing can be read from single pixels there; the
   picture has to be averaged *along the line*.
2. **Glue fills the weave.** Where the band is barely darker than the fabric, it is still
   two to five times smoother. Used alone the texture cannot place an edge, but as an input
   to a trained edge corrector it helps (chapter 14).

## What makes it hard

| Difficulty | Where | What it causes |
|---|---|---|
| Low contrast | pink, white | segmenters outline the wrong, wider band or spill sideways |
| Shine comes and goes along one line | all colours | an edge finder steps sideways where the core lies along one edge |
| Soft frames | white frames of other image sets (sharpness 1-6 against 90-600) | SAM marks half the lines doubtful |
| Mesh, cups, folds, seams, cardboard | everywhere outside the panel strip | false lines unless the search is confined to the panel region |
| No calibration | all sample frames | the same strip reads 26-35 px by position |
| No defective samples | all | no method can be shown to catch a real defect |

## Images used

| Set | Path (git-ignored) | Content |
|---|---|---|
| M1 captures | `enhancement_images/M1_Masks/captures` | 74 frames, 4608 x 3456: 34 dark, 34 pink, 6 white; 257 lines |
| 10_05 | `enhancement_images/10_05` | 4000 x 3000: navy, soft white, another part |
| Real captures | `enhancement_images/real_captures` | 6 webcam frames |

Almost every number in Part C is from the M1 captures. Fourteen bursts of 3-5 shots of the
same layout inside that set serve as repeat shots.
