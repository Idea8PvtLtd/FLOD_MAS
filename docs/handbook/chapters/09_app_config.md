# Configuration and saved data

## Which module reads which settings

![Sections of `config.json` and the modules that read them.](figures/config.svg)

- `config.json` sits beside the code. At start, press **S** during the countdown for the
  settings form, or edit the file and restart.
- Invalid values are rejected at start by `config_validation.py`.
- **Do not put `null` in the file.** The settings form saves it back as the text "None".
- New keys are read with `.get` and a default, so an older `config.json` still loads.
- `.env` holds `ROBOFLOW_API_KEY` and is git-ignored.

## The settings that shape an inspection

| Setting | Shipped | Effect |
|---|---|---|
| `auto_trigger.hand_absence_seconds` | 1.0 | how long hands must stay away |
| `auto_trigger.debounce_frames` | 3 | consistent frames before presence changes |
| `auto_trigger.detect_width` | 320 | width of the copy given to the hand detector |
| `auto_trigger.accepted_labels` | full, half fabric | which classifier labels go on to inspection |
| `auto_trigger.fabric_min_confidence` | 0.5 | classifier threshold |
| `segmentation.provider` | `workflow` | which provider (chapter 6) |
| `segmentation.roi` | whole frame | part of the frame sent to the workflow |
| `segmentation.fragment_policy` | `error` | stop on fragments, or keep the largest |
| `segmentation.min_confidence` | 0.0 | below this a strip is not graded |
| `segmentation.max_fabrics` | 4 | most fabrics in one capture |
| `sam_detection.strip_segments` | 10 | segments per line |
| `sam_detection.refine_edges` | false | edge refinement (chapter 7) |
| `inspection.strip_width_mm` | 4.0 | target width |
| `inspection.strip_width_tolerance_mm` | 1.0 | allowed +- |
| `inspection.result_display_seconds` | 5.0 | seconds per result tab |
| `capture_storage.max_sets` | 1000 | capture folders kept |
| `capture_storage.save_rejected_triggers` | false | also keep captures the fabric gate refused |

Every key is described in `docs/inspection/CONFIGURATION.md` and
`docs/inspection/AUTOMATIC_INSPECTION.md`.

## A capture set on disk

![Files of one capture. The root is the user's Documents folder, not the code folder.](figures/capture_set.svg)

- `original.png` and `undistorted.png` are lossless and full resolution; the measurement
  was made on the undistorted one.
- Preview images are JPEG (`jpeg_quality` 82, longest edge `preview_max_edge` 1920).
- Polygons and numbers are in `result.json` in full-frame pixels, independent of previews.
- Retention deletes the oldest **completed** sets, whole folders at a time, and only
  folders that carry the `.inspection-set` marker. A set in progress is never deleted.
- A capture that was cancelled, rejected, superseded or failed is recorded with that status
  when it was saved at all.
