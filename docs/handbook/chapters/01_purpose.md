<!-- part: Part A - Overview -->
# Purpose and scope

## What the system does

FLOD_MAS inspects the glue line laid on fabric panels of a garment. An operator places
panels on a table under a camera and takes their hands away; the software photographs the
table, finds the glue line on every panel, measures how wide it is along its length and
shows a verdict for each panel. The one defect it looks for is **width out of range**.

The application is a Tkinter desktop program for a Windows inspection PC (Python 3.12). A
Linux PC with a CUDA graphics card is used for development and for the experiments this
handbook describes.

## What is known about the glue line

These facts were stated by the user or measured on the sample frames; the rest of the book
leans on them.

| Fact | Source |
|---|---|
| The glue is a thin, glossy bead: a darker band with a bright core that shows in places | user; measured |
| The normal strip is **4 mm** wide most of the time | user, 2026-10-08 |
| The camera is **not calibrated**, so millimetre values may be off | user |
| The supplied sample strips have **no defects** | user |
| The only defect is width out of range | user |
| The glue always lies inside the panel region that the region model marks; cardboard is not glue | user |
| The same line is shiny in one place and matte in the next | user; measured |
| A 4 mm strip reads about 30 px in a 4608 x 3456 frame: about 7.6 px per mm | measured, 257 lines |
| On pale fabric the band is only about 7 grey levels darker than the fabric; on dark fabric 43-50 | measured |
| The target is to resolve width changes of 0.1-0.2 mm | user, 2026-10-09 |

## What this handbook covers

There are two lines of code that grew from one base and have **not** been merged. The book
describes both and keeps them apart.

![The two lines of code. Parts B and C describe the left one; Part D the right one.](figures/code_lines.svg)

| Part | Subject | Branch |
|---|---|---|
| A | Overview, reading guide | both |
| B | The inspection application: modules, runtime flow, providers, measurement, calibration, settings | `tooling/pipeline-lab` |
| C | Glue-line detection research: the detection flow in use, speed, accuracy, trained segmenter, width statistics, tools | `tooling/pipeline-lab` |
| D | The colleague's application: its architecture and pipeline, and how the two compare | `main`, read at commit `485c4b6` |
| E | Decision guide, what is verified and what is not, appendices | both |

## How far each statement can be trusted

The book uses three words on purpose.

- **Measured**: a number produced by running code on the sample frames; the table it comes
  from is in `docs/glue-line/GLUE_LINE_FINDINGS.md`.
- **Read**: taken from the source code without running it. All of Part D's description of
  the application on `main` is read; only its glue model was run (chapter 20).
- **Not verified**: stated as such, and collected in chapter 22.

> **No ground truth exists.** No line has been measured with a caliper and no defective
> strip has been supplied. Accuracy figures are against painted glue lines (a model of the
> real thing) and against agreement between methods; whether any method catches a real
> width defect is not known.
