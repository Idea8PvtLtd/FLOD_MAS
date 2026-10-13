<!-- part: Part E - State of the work -->
# Decision guide

## Which method for which need

![A summary of the findings as a decision tree. It is a guide for a person; no code selects a method this way.](figures/decision.svg)

## The options in one table

| Option | Where | Time per frame | Uploads | Strong at | Weak at | Judged by the user |
|---|---|---|---|---|---|---|
| Inspection app, cloud workflow | this branch | set by the connection | yes | whole cycle, any fabric | needs a connection; grades ten coarse segments | in use before this work |
| Inspection app, local ONNX | `main` | 3 s, CPU | no | offline, no GPU, equal widths on all colours | evenness along a line; soft frames | by its author |
| Version 4 | Pipeline Lab | 22-45 s | yes | the reference result | slow | **best so far** |
| Version 5 | Pipeline Lab | about 5 s, GPU | no | same result as 4, faster, more accurate edges | pink width in dispute; soft white frames | not yet |
| No SAM | Pipeline Lab | about 1 s | no | speed | 19 pink lines need a look | not yet |
| Version 6 candidate (SegFormer-B2) | scripts only | about 1 s for the mask | no | soft white frames | licence; training per product | not yet |
| Main model + trees + ribbon | scripts only | 3 s + under 1 s | no | even width from a CPU model | not built into either app | not yet |
| Width report | script | on cached runs | no | where a line widens or narrows, in mm | short and sudden changes; pale fabric | not yet |

## Rules that held throughout

- Work at full resolution. A reduced frame leaves the bead 2-5 px wide.
- Confine everything to the panel region, one line at a time.
- Do not filter the picture before a segmenter.
- Average along the line, never across it.
- Use two independent readings and report a width change only where they agree.
- Fit the shape the glue is known to have; report what departs from it.
- Measure error against something the method cannot satisfy by construction.
- New behaviour is opt-in, and the default reproduces the old result exactly.
