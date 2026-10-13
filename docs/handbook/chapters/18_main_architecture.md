<!-- part: Part D - The application on main (read-only) -->
# Architecture of the application on `main`

> **How this part was written.** `main` was read at commit `485c4b6` ("updater_added",
> 2026-10-09) with `git show`; it was not checked out, merged or changed. The description
> is **read from the source**. Only the glue model in `inspection/local/` was run, at commit
> `4a0680f`; its code did not change between the two commits. This part goes out of date
> when `main` moves. The author of this code is Obhash Mannapperuma.

## What changed against the common base

Sixteen commits; 56 files; about 5 100 lines added and 520 removed.

![The application on `main` by layer. Purple: added, changed or replaced there. Grey: still in the repository but no longer called by the automatic flow.](figures/main_modules.svg)

| Area | Change |
|---|---|
| Detection | The cloud workflow provider is no longer used. `inspection/local/` finds the glue with two ONNX models on the CPU, offline. |
| Pipeline | `inspection/pipeline.py` is rewritten: `run` (rightmost line) and `run_batch` (every line a capture still owes). It takes an *inspector*, not a segmenter. |
| Measurement | `inspection/local/width_measurement.py` and `inspection/line_measurement.py` replace `strip_analysis` for the automatic flow. |
| Trigger | The hand cycle stays. Added: a FIFO queue of captures, a record of lines already inspected, and a reset when two bed markers become visible. |
| Fabric gate | A two-class classifier (`full_fabric`, `no_fabric`); accepted when the first score is larger. |
| Screen | Startup loading screen, pause, live overlay of hands and markers, per-line results, mask tint and segment joins on the overlay. |
| Developer controls | A user distance ratio that multiplies millimetre results, kept apart from the calibration. |
| Camera | Rotation, settle time before locking controls, changed default index. |
| Updater | `updater/updater.py`: a starter program that fetches `main`, installs requirements and launches the app. |
| Models in the repository | `segformer_b0.onnx` (15 MB), `strip_unet_resnet34.onnx` (98 MB), `keras_model_2c.h5`. On this branch model weights are git-ignored. |

## Settings added or changed

| Setting | Value | Meaning |
|---|---|---|
| `local_inspection.source_model` | `inspection/local/models/strip_unet_resnet34.onnx` | the panel-strip model |
| `local_inspection.glue_model` | `inspection/local/models/segformer_b0.onnx` | the glue model |
| `local_inspection.source_width` | 1152 | input width of the strip model |
| `local_inspection.offset_pixels` | 100 | width of the band inside the curved side |
| `inspection.end_exclusion_percent` | 5.0 | share of each end of a line left out of grading |
| `auto_trigger.marker_confirm_seconds` | 0.5 | how long both bed markers must be seen |
| `preview.show_overlay`, `marker_detect_width` | true, 1280 | live overlay of hands and markers |
| `capture.auto_lock_settle_seconds` | 5.0 | wait before camera controls are locked |
| `capture_storage.max_sets` | 200 (was 1000) | capture folders kept |
| `auto_trigger.accepted_labels`, `fabric_min_confidence` | removed | replaced by the two-class gate |
| `cameras[0].rotation` | 180 (was 0) | the camera is mounted the other way up |

## The updater

![The updater's loop, read from `updater/updater.py`. It was not run.](figures/updater.svg)

It keeps a bare mirror of the repository. **Once, when it is started,** it fetches `main`;
if the commit is new it extracts it into its own release folder, reinstalls requirements
when that file changed, keeps the three newest releases, launches `main_1366.py` and closes
itself 1.5 s later. When the PC is offline it starts the release it already has. The
60-second interval in the file is not used.

> **Consequence for branches.** The updater installs whatever is on `main`. Anything merged
> into `main` reaches a PC the next time the starter is run there. The research on
> `tooling/pipeline-lab` is not on `main` and is not distributed by it.

Findings of the review (`docs/glue-line/MAIN_BRANCH_REVIEW.md`), all from reading the code:

| # | Severity | Finding |
|---|---|---|
| U1 | High | A failed package install is recorded as done: the new build starts on the old packages and the install is not retried until another commit arrives |
| U2 | High | `config.json` lives in the build folder, so camera index, rotation, tolerance and offset return to the committed values at every update. Calibration and captures are in Documents and are kept |
| U3 | High | Every push to `main` is delivered with no test run, tag or way back; 2 of the 365 tests on `main` fail at `485c4b6` |
| U4 | Medium | `git` and `pip` have no time limit: a stalled network or a credential prompt hangs the starter |
| U5 | Medium | An environment that is missing or rebuilt is left without packages unless `requirements.txt` changed |
| U6 | Medium | Errors in the fallback launch are not caught; the window stays on its last message |
