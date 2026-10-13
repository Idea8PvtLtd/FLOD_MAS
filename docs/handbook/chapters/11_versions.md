# How the flow evolved

## Versions and the verdict on each

![Versions of the detection flow. The thick box is the one the user judged best.](figures/timeline.svg)

| Version | Outline comes from | Verdict |
|---|---|---|
| 1 | SAM 3, each answer accepted or thrown away whole | dark good; pale mostly thrown away |
| 2 | SAM 3 + leak repair + asking again | good; black perfect; white jagged, ends too short or too long |
| 3 | version 2 + band edges drawn directly as the outline | **worse** than 2: steps, notches, wrong ends |
| 4 | version 2 + band edges as evidence + states + smooth ribbon | **best so far** (user, 2026-10-08) |
| 5 | version 4 on the local GPU + trained-trees edges + noise-scaled defect rule | not yet judged by the user |
| 6 (candidate) | trained SegFormer-B2 mask in place of SAM, same trees and ribbon | not built into the tool; not judged |

## The same line through versions 2 to 5

![One white line: start, middle and end. Version 2 is jagged and runs past the tips; version 3 has straight but notched edges; versions 4 and 5 are smooth ribbons that end where the band ends.](figures/photo_versions.jpg)

## What each step taught

| Step | Lesson |
|---|---|
| 1 to 2 | SAM spills in places on pale fabric, and in *different* places each time it is asked. Keep its steady edges and ask again, instead of filtering the picture. |
| 2 to 3 | A classical edge finder is not a better outline. It follows the core where the core lies along one edge. |
| 3 to 4 | Neither source is right everywhere. Treat both as evidence, weight them by what the picture is like at each place, and fit the shape the glue is known to have: a smooth curve of even width. |
| 4 to 5 | Measure error against a known truth before changing anything. The cloud is the slow part; a local SAM asked once gives the same ribbon. |
| 5 to 6 | A network trained on the flow's own best outlines reads soft frames that SAM cannot. |

## What was tried and does not work

Measured, and recorded so that it is not tried again.

- **Any filter in front of SAM**: CLAHE, flatten lighting, bead enhance, contrast gain,
  denoise, unsharp mask, negative-kernel sharpen, negative, quantise, shine compress,
  dimming outside the ROI, smoothing along the line, and combinations. Equal at best; a
  contrast boost makes SAM outline the wider band.
- **Enlarging tiles** two or three times: the outline fragments.
- **A whole-image filter before the local detector**: 34 of 40 panels with or without.
- **Drawing band edges as the outline** (version 3).
- **Reporting a width deviation from one source alone**: band edges alone flag 40-87 % of
  good lines.
- **Sub-pixel peak fitting, phase congruency, the texture channel alone, averaging several
  shots**: no gain in edge accuracy.
- **MobileSAM and FastSAM** on pale fabric.
- **CLAHE for the trained segmenter**: no gain on sharp frames, worse on soft ones.
