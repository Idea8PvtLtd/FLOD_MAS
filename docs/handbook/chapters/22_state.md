# What is verified, what is not, and next steps

## Verified by measurement on the sample frames

| Statement | Evidence |
|---|---|
| Every glue line of the 74 M1 frames is outlined | 257 of 257 lines, versions 4 and 5, and the model on `main` |
| The local GPU flow gives the cloud flow's result | centre lines 0.1-0.5 px apart; 257 of 257 "ok" |
| The flow runs with no network | one frame with the network removed |
| Trained trees place edges better than the steepest slope | 0.63 against about 1.0 px RMS on held-out painted lines |
| Whole-line width repeats between shots | 0.07-0.14 px SD |
| Band edges alone cannot be used for defects | they flag 40-87 % of good lines |
| The two-source deviation rule rarely flags a good line | 1 of 257 at threshold 7 |
| Filters before a segmenter do not help | SAM, the local detector and SegFormer, each measured |
| The trained segmenter reads soft white frames | 1 of 17 lines doubtful against 9 of 17 |
| The glue model on `main` is usable | all lines, equal widths by colour, 3 s on the CPU |
| The width limits behave as designed | 9 % and 13 % flagged where 5 % and 15 % were expected |
| New settings do not change existing behaviour | default detector output byte-identical; 366 tests pass |

## Judged by the user

- Version 2: good, black perfect. Version 3: worse than 2. Version 4: best so far.
- Versions 5 and 6, the width report and the main-model comparison have **not** been judged
  by the user.

## Not verified

| Open point | Why it is open |
|---|---|
| **Whether any method catches a real width defect** | no defective sample; painted defects are a model |
| Sudden width changes and steps | no test for them; the mask may round them off |
| Width in millimetres | camera uncalibrated; the scale is fitted to the nominal 4 mm; no caliper reading |
| Pink width | version 5 and the model on `main` differ by 2.6 px |
| White fabric | 18 lines in the set, 6 held out |
| Soft white frames from other sets | only the trained segmenter handled them, on few frames |
| Line ends | the outline stops short of the bead's tips on some panels |
| The trained segmenter on another product | not tried; its licence is non-commercial |
| Local SAM 3 | gated weights; not run |
| The Windows inspection PC | none of the research has run there |
| Part D of this book | read from source at one commit; only the glue model was run |
| The application's own cycle time | not re-timed in this work |

## Next steps, in order of what they unlock

1. **Defective samples.** A few strips made too wide and too narrow on purpose, sudden and
   gradual. Nothing about catching defects can be settled without them.
2. **Calibrate the camera**, then caliper widths on a few lines per colour. Settles the
   scale, the width definition and the pink dispute.
3. **Short windows and a jump test** in `segmentation/width_stats.py`, measured on painted
   sudden defects; flag a change that the band edges show and the mask does not.
4. **Trees and ribbon after the glue model on `main`.** The best update found for it; it
   gives the even width the raw reading lacks, on the CPU.
5. **Decide on the trained segmenter**: licence, and retraining per product.
6. **Merge `main` into the working branch** when the two lines are to become one; eleven
   files need care (chapter 20).
7. For pink: a second light at a low angle.

## Rules for changing this system

- Work on a separate branch, never on `main`. `main` is distributed by the updater.
- Do not commit until the change is validated and asked for.
- New behaviour is opt-in: a settings key whose default reproduces the old behaviour, read
  with `.get`.
- Do not put `null` in `config.json`.
- `.env` holds the Roboflow key and is git-ignored. A cloud workflow or cloud SAM uploads
  image tiles.
- Transcripts and session files in the repository folder may contain the key; they are
  git-ignored and must stay out of commits.
