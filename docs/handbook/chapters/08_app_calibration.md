# Calibration and millimetres

This chapter is a summary. The full account, with its proofs and acceptance test, is
`docs/calibration/CALIBRATION_PIPELINE.md`; nothing here was researched anew.

## From a board to a millimetre

![Calibration is done once; the two saved files are used on every capture.](figures/calibration.svg)

| Stage | What it gives | Accepted when |
|---|---|---|
| Intrinsics | Camera matrix and lens distortion, from about twenty views of a ChArUco board (one board, or two numbered boards) | reprojection error under the warning level (`camera_warning_rms_px`, 1.0) |
| Surface reference | The pose of the board lying flat on the measurement surface (`solvePnP`, IPPE start, refined) | at least 20 corners and at most 1.5 px RMS |
| Runtime | `ImageUndistorter` removes distortion; `plane_scale` turns a pixel into a ray and intersects it with the plane | files present and consistent with the frame size |

Two other modes exist: an ArUco marker detected in every inspection frame, and an optional
homography per crop region. Both are described in the calibration document.

## Limits that matter for the glue line

- The scale is valid **on the calibrated plane**. Fabric that stands proud of the table, or
  a table that has been moved, changes the answer.
- Far from where the board lay, the result leans on the plane being flat there.
- Changing camera resolution, rotation or field of view needs a new calibration.

## State on the development PC

> **The sample frames were taken with an uncalibrated camera.** Every millimetre figure in
> Part C therefore comes from a *fitted* scale that assumes good lines average 4 mm
> (chapter 16), not from this calibration. On those frames the same 4 mm strip reads 26 to
> 35 px depending on where it lies; about 7 % of a width reading is position, not glue.
> Calibrating the camera is the first of the next steps in chapter 22.
