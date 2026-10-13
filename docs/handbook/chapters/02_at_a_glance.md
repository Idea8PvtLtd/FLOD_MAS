# System at a glance

## Who and what the software talks to

![System context. Solid arrows are always present; dashed arrows exist only when a cloud provider is chosen.](figures/context.svg)

- **The inspection application** owns the camera, the trigger, the result screen and the
  saved capture sets. With the default settings on this branch it sends a reduced copy of
  each capture to a Roboflow workflow, which returns the strip outlines.
- **The research tools** never touch the camera. They read saved frames, run the detection
  flow on them and write overlays, numbers and reports. They can run entirely offline on
  the GPU.
- **Files on disk** join the two: `config.json`, the saved calibration, and the capture sets.

## The idea in five steps

Every pipeline in this book, on either branch, is a variation of the same five steps.

| Step | Question it answers | This branch, app | This branch, research (version 5) | `main` |
|---|---|---|---|---|
| 1 | Where can glue be? | whole frame, or `segmentation.roi` | panel-region model (U-Net) | the same U-Net |
| 2 | Where does each line run? | the cloud workflow | rough line by the local detector | curved side of each region + 100 px band |
| 3 | Which pixels are glue? | SAM 3 in the workflow | SAM 2.1 with point prompts, repaired | SegFormer-B0 on the unfolded band |
| 4 | Where exactly are the two edges? | the polygon as returned (optional edge refinement) | band edges by trained trees, fitted as a smooth ribbon | the cleaned mask's boundary |
| 5 | How wide, and is that acceptable? | ten segment averages in mm against target and tolerance | width along the line; deviation rule; width statistics | ten segment averages in mm against target and tolerance |

## Colour key of the diagrams

![Colours and shapes used in every diagram.](figures/legend.svg)

## Numbers worth remembering

| Quantity | Value | Chapter |
|---|---|---|
| Frame size of the sample set (M1 captures) | 4608 x 3456, 74 frames, 257 glue lines | 10 |
| Width of the strip in the picture | about 30 px = 4 mm (7.6 px per mm, uncalibrated) | 16 |
| Time per frame, cloud flow (version 4) | 22 s dark, 45 s pink, 23 s white | 13 |
| Time per frame, local GPU flow (version 5) | about 5 s, nothing uploaded | 13 |
| Time per frame, glue model on `main` | 3 s on the CPU | 20 |
| Edge error against painted truth, version 5 | 0.5-0.7 px RMS (about 0.07-0.09 mm) | 14 |
| Smallest width change found, dark fabric | 0.14 mm if it lasts 42 mm; 0.36 mm if it lasts 5 mm | 16 |
| Smallest width change found, pink and white | 0.35-0.76 mm | 16 |
