# Narrow glue: detecting width defects (research, 16 October 2026)

The defect to catch: a glue line whose width is not continuous, narrow in places. It
should FAIL. `tools/glue_defects` measures the glue width along every line with several
glue models and checks it against defects marked by hand. Nothing in it is used by the
app yet.

## Ground truth

| Recording | Content | Defects |
|---|---|---|
| `defect_video1.mp4` (13 Oct, navy pieces) | photos at frames 1419 (one piece) and 1482 (two) | orange star above the lower piece's line, about 85-115 mm from its left end |
| `defect_video2.mp4` (13 Oct, white pieces) | photos at frames 1599 and 1659 | orange dot at the left end of the lower line, pink dot at the right end of the upper line |
| `capture_20261007_223831.mp4`, `capture_20261007_223948.mp4` | 8 photos, 1-4 pieces each | none (known good) |

The small coloured stickers mark the defects (the large yellow tapes are only labels).
The marked white ends are visibly thinner than the unmarked ends. The upper navy piece
has no sticker but its right end tapers the same way; whether it is a defect is not
known yet.

## What the light does

The glue band is only as visible as the light makes it. `check.edge_visibility`
measures, from the photo alone, how sharp the weaker of the band's two edges is
(lightness levels per pixel):

| Where | Edge sharpness |
|---|---|
| Navy pieces, 13 Oct | about 16 (15-25 at the star) |
| White pieces, 13 Oct | 2-4 (marked upper end: 1.3-1.6) |
| All 7 Oct lines | 0.6-1.5 |

Where the edges cannot be seen every model guesses, and those guesses read as narrow
glue. **A stretch whose edges are not visible is not judged and can never FAIL**
(`check.VIS_MIN` = 3). The bright ridge on top of the glue is no help either: it
comes and goes with the light's angle (it vanishes at the same places of the frame on
good and defect lines).

Two more causes of false narrowing were removed:

- **Straightening.** Unbending the band along the line squeezes it on curves. Widths
  here are measured in the photo as it is (largest circle inside the mask, or across
  the line on the probability), and the viewer shows the glue turned, never bent.
  The app's own glue model still segments a straightened band; that cannot be
  changed here.
- **Camera scale.** The widths drift slowly across the board (uncalibrated). Each
  stretch is compared with the band's own width nearby (60 mm running median), not
  with the whole line.

## Results

Same check for all (`check.judge`: narrower than 0.90 of the width nearby over at
least 5 mm, edges visible); main's method uses `check.segment_check` on its own
segments. 5 sticker spots (the navy star and the white lower end appear in two photos
each), 20 unmarked lines (19 known good, the unmarked upper navy piece) plus one line
cut by the photo edge, not judged.

| Glue mask from | Sticker spots caught | Unmarked lines failed | Note |
|---|---|---|---|
| **main's method** (40 graded segments, 4.0 +/- 1.0 mm) + visibility + neighbours | **2 / 5** | 1 / 20 | the white lower end in both photos; the failing line is the upper navy piece's tapered end |
| main's method as it is | 0 / 5 | 2 / 20 (+1 warning) | the defects are 3.8 mm, inside 3.0-5.0; both fails are in the dark (sharpness 0.4-0.8) |
| app glue model, raw mask | 1 / 5 | 1 / 20 | |
| SAM 2.1 large + contrast, Jetson | 1 / 5 | 2 / 20 | loses the faint 7 Oct band over 50-150 mm stretches |
| SAM 2.1 large, Jetson | 0 / 5 | 3 / 20 | takes only the bright ridge on white |
| pipeline-lab SegFormer B0 / B0+CLAHE / B2, photo as is | 0 / 5 | 2 / 20 | widths smooth through the marked ends |

Not caught by any: the navy star (well lit, no narrowing in any model: 4.2 mm against
4.3-4.5 mm beside it) and the white upper end (too dark to judge). Glue models on the
Jetson: SAM 2.1 large 0.65 s per 512 px tile (about 6 s a line); the lab SegFormers
0.3-1.7 s a line.

**Recommendation.** Keep main's segments and add the two checks behind config keys
(off by default): segments too dark to see are not judged, and a segment fails below
0.90 of its seen neighbours as well as outside the fixed tolerance. More of the glue
becomes judgeable only with better light (even, diffuse light over the whole board, or
a low side light that makes the glue edges cast a line); today about 16 % of the line
length is lit well enough. More sticker-marked pieces are needed to set the limits.

## Running it

```bash
# data: photos.json + photos/ (the app's undistorted.png of each still moment)
python -m tools.glue_defects.measure_app [--source-model main's U-Net]   # line paths, app widths
python -m tools.glue_defects.visibility
tools/jetson/run_glue_defects.sh sam --clahe                               # Jetson, research venv
tools/jetson/run_glue_defects.sh labseg segformer_b2_plain segformer_b0_plain segformer_b0_clahe
python -m tools.glue_defects.segments_check --repo <checkout of main>
python -m tools.glue_defects.viewer            # window; --check prints every source's verdicts
```

`tools/glue_defects/__init__.py` describes the data folder. The viewer's sliders change
the four settings of the check; a grey bar under the plot marks what is too dark to
judge.
