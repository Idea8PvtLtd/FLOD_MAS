# Reading guide and glossary

## Where to start

| If you want to... | Read |
|---|---|
| understand how the application is built | chapters 4-6 |
| follow one inspection from frame to verdict | chapter 5, then 7 |
| change a setting | chapter 9, then `docs/inspection/CONFIGURATION.md` |
| know how the glue line is found in the research flow | chapters 10-12 |
| know how fast and how accurate it is | chapters 13 and 14 |
| know how width changes are judged statistically | chapter 16 |
| run a research tool | chapter 17 and appendix 23 |
| understand the application on `main` | chapters 18-19 |
| decide which method to use | chapter 21 |
| know what has not been proven | chapter 22 |

## Glossary

| Term | Meaning |
|---|---|
| Panel, fabric | One piece of fabric on the table. A frame holds two to four. |
| Strip, glue line, bead | The line of glue on a panel. "Bead" is used for the glue seen in cross-section. |
| Band | The darker stripe the bead makes on the fabric, about 30 px (4 mm) wide. Its two edges are what is measured. |
| Core | The bright, shiny middle of the bead, visible only in places. |
| ROI, panel region | The part of a panel where the glue can be, marked by the region model (a U-Net). |
| Rough line | A first estimate of where a line runs, good to about 20 px, from the local detector. |
| SAM | Segment Anything Model. SAM 3 runs in the Roboflow cloud; SAM 2.1 runs on the PC's GPU. It is given points on the line and returns a mask. |
| Leak, spill | A place where SAM's mask runs out of the bead into the fabric beside it. |
| Straightened strip | The picture resampled along a line so that the line becomes straight; edges are then read across it. |
| State | What the picture is like at one place along a line: shiny, matte, one-sided, weak, none. |
| Trees | Gradient-boosted decision trees (LightGBM) trained to correct an edge position from the picture around it. |
| Ribbon | The final outline: a smooth centre line with a slowly varying width, fitted to all the evidence. |
| Mask | A picture-sized map saying, per pixel, glue or not (or a probability of glue). |
| SegFormer | A transformer segmentation network. Used as a trained glue segmenter in the research (B2) and on `main` (B0). |
| CLAHE | Contrast-limited adaptive histogram equalisation: a local contrast boost. |
| Glitch | A stretch where the mask itself is in doubt, so a width change read there may not be real. |
| Provider | A class that turns a frame into strip polygons for the application; chosen in `config.json`. |
| Capture set | The folder of files saved for one inspection. |
| Opt-in | A setting whose shipped value keeps the old behaviour. All new behaviour on this branch is opt-in. |

## Conventions

- File paths are relative to the repository root unless they start with `~`.
- `~/flod_experiments/` holds experiment scripts, trained models and result pages. It is
  outside the repository on purpose.
- Pixels always mean pixels of the full camera frame unless said otherwise.
- "Dark", "pink" and "white" are the three fabric colours in the sample set.
