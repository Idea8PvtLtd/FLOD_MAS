# Width in millimetres and width changes

The user's request: taking the average strip as 4 mm, turn pixels into millimetres, find by
a proper statistical method where a line is wider or narrower, show it on a page, and
resolve changes of 0.1-0.2 mm. Code: `segmentation/width_stats.py`,
`segmentation/segformer_local.py`, `tools/width_report.py`.

## The method

![From a mask to a judged width change.](figures/width_stats.svg)

1. **Width** is read from the segmentation mask, pixel by pixel across the line: the two
   places where the probability of glue passes one half, found between pixels.
2. **Scale.** The camera is uncalibrated, so pixels per millimetre is *fitted*: a quadratic
   surface over the frame with a factor per fabric colour, fitted robustly to the logarithm
   of width, on the assumption that good lines average 4 mm. Mean 7.57 px per mm.
3. **Departure** is the width minus the line's own gradual course. A gradual change along a
   whole line is mostly the camera and is reported separately.
4. **Test.** The departure is averaged over windows of 40, 80, 160 and 320 px. A window is
   flagged when its mean exceeds a limit taken from good lines.
5. **Judge** each flagged stretch: mask glitch, confirmed by the band edges, or mask only.

## Why the mask width, and why not three sigma

![The largest change of width inside one good line, by where the width is read from. A steadier signal can find a smaller defect.](figures/chart_width_signal.svg)

- All supplied lines are good, so the best signal is the one that varies least on them.
  Band-edge widths, the trees included, have a bump of about 0.8 mm somewhere on nearly
  every good line. The mask width is three to five times steadier.
- Limits from a normal table fail here. Neighbouring widths are correlated (0.8 at 10 px,
  0.4 at 40 px) and the spread has heavy tails (2-9 % of readings beyond three robust
  standard deviations, against 0.27 % for a normal distribution). Three-sigma limits would
  flag most good lines.
- So the limits are **empirical**: for each line the largest window mean is found; the
  limit is a percentile of that over good lines, taken over all window lengths together.
  At the 95th percentile, 5 good lines in 100 are flagged by chance at any length.
- A line's limits come from other frames (five groups by capture time), and only from the
  22 frames the mask model was not trained on.

## What it reaches

![Smallest width change marked significant, by fabric colour and by how long the change lasts. The green band is the user's target.](figures/chart_width_limit.svg)

| | Dark | Pink | White |
|---|---|---|---|
| Line width after the scale, mm (mean +- SD between lines) | 4.00 +- 0.06 | 4.02 +- 0.16 | 4.01 +- 0.08 |
| Spread of 40 px readings along a good line, mm | 0.048 | 0.168 | 0.130 |
| Smallest change found, lasting 5 mm | 0.36 | 0.58 | 0.76 |
| ... lasting 42 mm | **0.14** | 0.35 | 0.37 |

- **The target is met on dark fabric for changes 20-40 mm long** (0.14-0.20 mm), and not
  met on pink or white, nor for short changes on any colour.
- The limit is set by the natural variation of good lines, not by the measurement: a single
  40 px reading on dark fabric is steady to 0.05 mm.
- Check of the method: of 78 lines new to the mask model, 7 were marked significant (9 %;
  5 % expected) and 10 more notable (13 %; 15 % expected).

## Segmentation glitches

| Check along a line | A glitch when | Found on 257 lines |
|---|---|---|
| No mask | the line is outside the mask for 3 readings or more | 0 stretches |
| Mask fades out | the probability falls slowly at an edge, for 20 px or more | 115 stretches, typically 4 mm |
| Mask steps sideways | the mask's middle leaves its own course beyond what good lines show | 24 stretches, typically 19 mm |

| Width changes found | Glitch | Also in the band edges | Mask only |
|---|---|---|---|
| Significant (39) | 0 | 33 | 6 |
| Notable (60) | 2 | 36 | 22 |

Most flagged changes are real variation of good glue, seen by a second reading that owes
nothing to the mask.

![A frame of the width report. Each line is tinted by its departure (blue narrower, red wider); yellow marks a glitch stretch; the label is the line's mean width in mm.](figures/photo_width_report.jpg)

## What it does not cover

| Case | Found? | Why |
|---|---|---|
| A change lasting 40 px (5 mm) or more | yes | every window length has its own limit |
| A bump shorter than 40 px | only if large | the shortest window averages it: a 20 px bump counts at half its size |
| A step in width | late, and not located | no test compares neighbouring readings |
| The mask changes width without fading or moving sideways | as "mask only", not as a glitch | no glitch check looks at width |
| A real sudden defect that the mask rounds off | no | the model was trained on even ribbons and outputs at a quarter resolution |

> **Unproven where it matters.** The limits control how often a *good* line is flagged.
> How often a *defective* line is caught has not been measured, because no defective
> sample exists; and a mask trained on even-width labels may smooth a real defect away.
