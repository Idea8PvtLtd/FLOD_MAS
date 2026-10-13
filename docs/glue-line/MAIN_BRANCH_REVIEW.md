# Review of the implementation on `main`: measurement, bugs, updater

Reviewed 2026-10-12 at the user's request. Subject: `main` at commit `485c4b6` (the remote
tip on that day), author Obhash Mannapperuma. Written from `tooling/pipeline-lab`; `main`
was only read, none of its code was changed, and nothing here has been sent to its author.

File and line references are to `main` at `485c4b6`.

## How it was reviewed

- **Read in full:** `updater/updater.py`, `inspection/local/` (`engine.py`, `geometry.py`,
  `glue_model.py`, `source_model.py`, `source_reference.py`, `width_measurement.py`),
  `inspection/line_measurement.py`, `pipeline.py`, `overlay.py`, `storage.py`,
  `dashboard.py`, `auto_trigger/controller.py`, `bed_markers.py`, `app_storage.py`,
  `app_config.py`, `measurement_adjustment.py`, `developer_controls.py`, `config.json`,
  `requirements.txt`.
- **Read in part:** `inspection/result_view.py`, `camera_handler.py`, `startup_settings.py`.
  Not read: `main_1366.py`, `CalibrateAPP/`, the hand and fabric models.
- **Run:** the model code (`inspection/local/`, identical to the copy in
  `~/flod_experiments/main_model/`) on the M1 captures and three soft white frames; the
  test suite of `main` from an extract, with a throwaway home folder. Script:
  `~/flod_experiments/main_model/review_checks.py`.
- **Not run:** the app, the camera, the bed markers, the updater, anything on Windows.
  Findings marked *read* come from the code alone.
- Millimetres use 7.57 px/mm, the scale fitted to the M1 captures, not a calibration.

## Summary

| # | Severity | Finding |
|---|---|---|
| M1 | High | Grading by segment average: a local width change cannot fail a line |
| M2 | High | A broken or missing line is not a FAIL |
| U1 | High | A failed package install is recorded as done and never retried |
| U2 | High | Operator settings are lost on every update |
| U3 | High | Every push to `main` reaches the inspection PC untested; two tests fail today |
| M3 | Medium | The search band is built on a zigzag path |
| M4 | Medium | Segment averages change when the photo moves by one pixel |
| M5 | Medium | Resolution at the glue is 0.2 mm per model row |
| M6 | Medium | Soft white frames: good lines fail, extra lines appear |
| M7 | Medium | Changing the camera rotation does not invalidate the calibration |
| U4 | Medium | `git` and `pip` run with no time limit |
| U5 | Medium | A rebuilt environment is left empty |
| U6 | Medium | Errors in the fallback launch are not caught |
| M8-M14, U7-U9 | Low | Listed below |

The glue model itself is sound: it finds every M1 line as one piece and reads the same
width on the three colours (see *The main branch's local model* in
`GLUE_LINE_FINDINGS.md`). The findings below are about what is done with its mask, and
about how the app is delivered.

## Measurement

### M1 (High). A local width change cannot fail a line

`inspection/line_measurement.py:85`, `inspection/local/width_measurement.py:44`.

A line is cut into ten equal segments (about 190 px, 25 mm, each on the M1 lines). Only the
**average** of each segment is compared with target +- tolerance; the default tolerance in
`config.json` is 1.0 mm. Minimum and maximum per segment are stored and never graded.

Measured on 12 real masks, by widening a stretch of the mask (rounded ends) until the
status became FAIL:

| Stretch length | Extra width needed at +- 1.0 mm | at +- 0.5 mm |
|---|---|---|
| 2 mm | +4.2 mm | +2.4 mm |
| 5 mm | +2.9 mm | +1.6 mm |
| 10 mm | +1.8 mm | +1.1 mm |
| 25 mm (a whole segment) | +1.1 mm | +0.5 mm |

So a 2 mm blob has to make the 4 mm strip more than twice as wide before the line fails.
The check was made on the mask; a short change also has to survive the cleaning (M10)
first, which was not tested. The user's target is to resolve 0.1-0.2 mm.

*Suggested:* grade the samples as well as the averages, for example the longest run of
consecutive samples outside tolerance, or the per-segment minimum and maximum that are
already computed. The window method in `segmentation/width_stats.py` on this branch does
this with limits taken from good lines.

### M2 (High). A broken or missing line is not a FAIL

`inspection/local/engine.py:92-118`, `inspection/line_measurement.py:95`,
`auto_trigger/controller.py:225`, `inspection/dashboard.py:112-116`.

- Each connected piece of glue is measured and graded on its own. Nothing compares the
  total length with the length of the band, and nothing looks at the gap. Two pieces that
  are each in tolerance give **PASS**.
- A line where the model finds no glue raises "adhesive model returned an empty mask". The
  dashboard shows that as **WARNING**, not FAIL.
- A piece of 100 px of area or more is cut into ten segments however short it is. On
  `0049_frame`, pieces 15 and 23 px long gave 7 and 6 empty segments each, which turns the
  line to WARNING.
- A line detected for only 345 px (`0049_frame`, line 2) is graded on those 345 px.

*Suggested:* report coverage (measured length over band length) and the largest gap, and
fail or warn on them explicitly; do not segment pieces shorter than a segment.

### M3 (Medium). The search band is built on a zigzag path

`inspection/local/geometry.py:149-162` (`offset_path`), `:196` (`offset_lines`),
`:248-256` (`original_dimensions`).

The inner edge of the band is the panel edge moved 100 px along its normal. The normal is
taken from two raw mask points five rows apart, with no smoothing, and the mask is a 4x
enlargement, so its edge is stepped. The normal therefore swings from row to row and the
inner edge jumps back and forth along the line.

| | Measured (8 lines) |
|---|---|
| Panel edge path | 2200-2970 px |
| Inner edge path | 7600-9900 px |
| Steps of the inner edge that go backwards | 18-20 % |
| "Real size" unfolded strip | 4500-6000 x 100 px (not about 2000 x 100) |
| Squeeze along the line to reach 1024 | 4.4-5.8 times |

Effect, measured by taking the normal from the same points smoothed over 101 rows (37 M1
frames, 131 lines, model not retrained):

| | As shipped | Smoothed normal |
|---|---|---|
| Unfolded length, median px | 4774 | 2422 |
| Width noise from sample to sample, px RMS | 0.70 | 0.59 |
| Worst segment off the line's mean, px: mean / max | 2.20 / 5.35 | 1.91 / 5.21 |
| One-pixel shift, segment average change: mean / 95th / max | 0.35 / 0.89 / 4.00 | 0.24 / 0.60 / 1.09 |
| One-pixel shift, line mean change: mean / max | 0.16 / 0.87 | 0.11 / 0.35 |
| Mean width, px | 30.83 | 30.61 |
| Lines in pieces | 0 | 0 |

- The zigzag is **not** the cause of the uneven segments found earlier; those stay.
- It does cost repeatability: the worst change under a one-pixel shift falls from 4.0 to
  1.1 px (0.53 to 0.14 mm) with a smooth normal.
- The model was trained on strips unfolded the shipped way, so the smooth version is
  outside its training and still does no worse. Retraining on smooth strips was not tried.

*Suggested:* smooth the edge points (or fit them) before taking the normal.

### M4 (Medium). Segment averages change when the photo moves by one pixel

The same photo moved by one pixel (37 frames, 580 segment pairs): segment averages change
by 0.35 px on average, 0.89 px at the 95th percentile and 4.0 px (0.53 mm) at worst; line
means by up to 0.87 px (0.11 mm). A tolerance near those figures will pass and fail the
same good line from one capture to the next. Part of it is M3; the rest is M5.

### M5 (Medium). Resolution at the glue

`inspection/local/glue_model.py:16-17`.

The model's output is 64 x 1024. Across the line one model row is 1.56 px, about 0.2 mm;
along the line one column covers 4.4-5.8 unfolded pixels. The class is chosen first and
the labels are then enlarged by nearest neighbour, so each edge sits on a 0.2 mm grid
before cleaning. The ray crossing in 0.25 px steps (`width_measurement.py:23`) gives
decimals, not accuracy. Enlarging the scores smoothly before choosing the class improved
edges from 1.01 to 0.85 px on dark fabric (tested earlier, see the findings).

### M6 (Medium). Soft white frames

On three soft white frames from other sets, with 7.57 px/mm (assumed for these frames) and
+- 1.0 mm: three good lines of `WIN_..._21_30_28` FAIL on one segment each, a fourth breaks
in two, and `0049_frame` gives six lines where there are four, with means of 14-32 px.
The extra lines come from the source-line stage. They also shift the left-to-right
positions by which the queue remembers which line was measured
(`engine.py:46-57`), so a later capture can skip or repeat a line. That consequence is
*read*, not tested.

### M7 (Medium). Changing the camera rotation does not invalidate the calibration

`camera_handler.py:151-158`, `config.json` (`cameras[0].rotation` is 180), *read*.

Frames are rotated right after capture, so calibration, undistortion and the millimetre
plane all work on rotated frames. Neither calibration file records the rotation it was made
with. If `rotation` is changed afterwards, the old lens centre, distortion and plane are
applied to a turned image, with no error. Rotation was changed by a commit to
`config.json` (`87cb9e1`), and the updater delivers `config.json` (see U2). Whether the
inspection PC was recalibrated after that commit is not known.

*Suggested:* store the rotation in the calibration files and refuse a mismatch.

### Lower severity

| # | Finding | Where |
|---|---|---|
| M8 | Works for roughly vertical seams only. The same photo turned 90 degrees gives "No accepted source lines"; turned 180 or mirrored it agrees within 0.9 px. | `geometry.py:43-98`, `:196`; measured |
| M9 | `seconds` in a line's record counts from the start of the capture, so later lines include earlier ones. `mean_width_px` and `segments` describe the longest piece only. | `engine.py:36`, `:118-125` |
| M10 | Cleaning redraws the mask: per column it keeps first-to-last glue pixel (holes filled, side blobs absorbed), takes medians over about 40-50 unfolded pixels, limits smoothing to 15 % and draws rounded tips. The tips are drawn, not measured; the 5 % end exclusion hides them. | `geometry.py:342-403`, `engine.py:78` |
| M11 | CLAHE is recomputed on the whole photo for every line (0.3 s each), and its result depends on everything else in the frame. | `engine.py:72` |
| M12 | Bed-reset markers are ArUco ID 0 of `DICT_4X4_50`; the measurement-surface marker is ID 0 of `DICT_4X4_100`, the same pattern. If it is in view it counts as one of the two. Not checked on the bed. | `bed_markers.py:9-18`, `config.json:131-132` |
| M13 | The developer ratio multiplies every millimetre value. Set to match one reference piece, it hides a calibration error everywhere else. It is recorded with each result. | `line_measurement.py:43` |
| M14 | A capture leaves lines for captures queued behind it and counts those still in the fabric check. If one of them is rejected, its line waits for another trigger. *Read.* | `controller.py:209`, `engine.py:57` |

Also: before a result is shown, the capture is saved as two lossless 16-megapixel PNGs
(0.7 s each on this PC) and each line's overlay as a third (0.5 s). The first result
therefore appears about 2 s after the 1.75 s the engine takes
(`pipeline.py:45`, `:72-79`). And `startup_settings.py:17` reads a true/false setting as
`bool(text)`, so the text "False" becomes true; that line is the same on this branch.

## Updater (`updater/updater.py`)

All *read*; the updater was not run.

### U1 (High). A failed package install is recorded as done

Lines 633-646. When `pip install` fails, the new build is stored as the deployed one
(`state["sha"] = sha`) and started on the old packages. At the next start the commit
matches (line 602), the starter says "up to date" and the install is never tried again,
until another commit arrives. The message "so it can be retried later" is not what happens.

*Suggested:* do not record the build as deployed until its packages are installed; start
the previous build instead.

### U2 (High). Operator settings are lost on every update

`app_config.py:8`, `startup_settings.py:12`, updater lines 618-622.

`config.json` is read from the build folder and the startup form saves it back there. Every
update unpacks a new folder with the repository's `config.json`. Camera index, rotation,
strip width, tolerance and the band offset return to what was committed. Calibration files,
captures and the developer ratio are safe: they live in Documents. The history shows the
effect: machine settings are changed by commit (`0e15263` "Set default camera index to 0",
`2dd167a` "Change camera device index from 3 to 2", `87cb9e1` "Camera_rotation").

*Suggested:* keep machine settings in Documents and let the repository hold defaults only.

### U3 (High). Every push to `main` reaches the inspection PC untested

Lines 594-622. The starter takes the tip of `main` at each start. There is no test run, no
tag or release, no check that the new build opens, and no way back, although three builds
are kept. Anyone who can push to `main` runs code on that PC.

The test suite of `main` does not pass today: 2 of 365 fail
(`test_region_calibration_ui`: `test_the_device_index_is_mapped_to_the_side_that_owns_it`,
`test_an_unmarked_camera_closes_the_gate_for_both_regions`). Both depend on the camera
indices in the shipped `config.json`, which the commits above changed.

*Suggested:* deploy a tag or a release branch; run the tests before moving it; keep the
last build that started and fall back to it.

### Medium and low

| # | Severity | Finding | Lines |
|---|---|---|---|
| U4 | Medium | `git clone`, `git fetch` and `pip install` run with no time limit. A stalled network or a credential prompt hangs the starter and the app never opens. The internet check only opens a socket. | 121-126, 195-221, 270-272 |
| U5 | Medium | If the environment is missing or is rebuilt for a Python version change, it is created empty. Packages are installed only when `requirements.txt` changed, so the app then fails at its first import. | 249-267, 629-649, 694-701 |
| U6 | Medium | The fallback launch runs inside the error handler and catches only `RuntimeError`. A failing `venv` or `pip` there ends the thread; the window stays on its last message. | 665-669, 694-701 |
| U7 | Low | The "update loop" runs once and the starter closes 1.5 s after launching. `CHECK_INTERVAL`, `_stop_flag` and stopping a running app are dead code. An app still running from an earlier start is not detected; the second one cannot open the camera. | 56, 360, 618, 714 |
| U8 | Low | Tk is called from the worker thread (`self.after`). One environment serves all kept builds, so an older build would run on newer packages. `customtkinter` is not in `requirements.txt`. | 28, 363, 528 |
| U9 | Low | A 97 MB model is stored in git (GitHub refuses files over 100 MB) and every build folder is about 130 MB. Captures go to Documents, which the code allows to be a OneDrive folder: up to 200 sets of 25-50 MB each. | `app_storage.py:13-50`, `config.json` (`max_sets`) |

## Not covered

- No defect samples exist. M1 and M2 are shown with drawn changes and with soft frames.
- The Windows inspection PC, its camera, its calibration and the updater were not run.
- The hand trigger, the fabric check and the calibration tools were not reviewed.
- Suggested fixes are not implemented or tested, except the smoothed normal of M3.
