# Appendices

## File index (branch `tooling/pipeline-lab`)

One line per Python file, taken from the file's own description.

### Application, repository root

| File | What it is |
|---|---|
| `app_config.py` | Shared configuration loaded after the startup settings window |
| `app_storage.py` | Portable storage for user data, separate from application code and config |
| `camera_handler.py` | Background full-resolution capture, shared by setup and dashboard |
| `config_validation.py` | Checks every section of `config.json` at start |
| `crop_processing.py` | Persistent rotated crop geometry shared by setup and inspection |
| `crop_result_view.py` | Tabbed, zoomable view of the crops an inspection produced |
| `main_1366.py` | Entry point and main window (`IndustrialDashboard`) |
| `plane_scale.py` | Millimetre scale for a deskewed crop, from the calibrated measurement plane |
| `region_calibration.py` | Per-crop-region local homographies, as an alternative to the global plane |
| `sam_detection.py` | SAM3 object detection through the Roboflow serverless workflow |
| `startup_settings.py` | Startup countdown and form editor; no camera or app imports before selection |
| `strip_analysis.py` | Centreline, segmentation and width measurement for a detected strip |
| `ui_theme.py` | Shared, low-overhead visual system for the desktop application |

### `auto_trigger/`

| File | What it is |
|---|---|
| `auto_trigger/checkpoint.py` | Hand-cycle checkpoint and debouncer |
| `auto_trigger/controller.py` | Latest-frame hand monitor with fabric gating and stale-result protection |
| `auto_trigger/fabric_check.py` | Reference H5 loading and fabric preprocessing, copied from hand_traker |
| `auto_trigger/models.py` | Synchronous inference adapters derived from hand_traker; called off Tk |

### `inspection/`

| File | What it is |
|---|---|
| `inspection/dashboard.py` | Tk integration; cameras and calibration remain owned by the existing app |
| `inspection/measurement.py` | Reuse the established strip algorithm for every automatically found fabric |
| `inspection/pipeline.py` | One immutable capture -> segmentation -> existing measurements -> saved set |
| `inspection/result_view.py` | Large, persistent fabric tabs with the established zoom/pan image viewer |
| `inspection/storage.py` | One capture directory per detection; lossless sources and bounded retention |

### `segmentation/`

| File | What it is |
|---|---|
| `segmentation/__init__.py` | Replaceable segmentation boundary |
| `segmentation/bead_edges.py` | The glue bead's own two edges along a roughly known line, from the camera's pixels |
| `segmentation/bead_ml.py` | Traditional machine learning for the glue band's edges |
| `segmentation/edge_methods.py` | Ways of finding the glue band's two edges in a straightened strip |
| `segmentation/glue_line.py` | Local glue-line detector for pale fabric |
| `segmentation/glue_shape.py` | The known shape of a glue line, used as a condition on what is reported |
| `segmentation/provider.py` | Reference workflow preprocessing and coordinate mapping, isolated from UI |
| `segmentation/ribbon.py` | One smooth ribbon of even width from the edge evidence along a glue line |
| `segmentation/sam_local.py` | SAM on this computer's own GPU: nothing is uploaded |
| `segmentation/sam_refine.py` | Exact glue-line outline from SAM, prompted with points on the roughly found line |
| `segmentation/segformer_local.py` | A trained SegFormer that marks the glue band, and the band's width read from its mask |
| `segmentation/synthetic_bead.py` | Glue lines of known edges, painted onto real fabric: a truth to measure error against |
| `segmentation/width_stats.py` | Statistics of a glue line's width: millimetres from pixels |

### `tools/`

| File | What it is |
|---|---|
| `tools/accuracy_bench.py` | How far each way of finding the glue line's edges is from the truth |
| `tools/benchmark_flow.py` | Where the time goes in a Pipeline Lab run: every stage timed inside `run_job` |
| `tools/learn_bead_reference.py` | Learn what a glue line looks like from frames where it is seen perfectly |
| `tools/learn_glue_shape.py` | Learn the wave shape of a product's glue line from sample frames |
| `tools/pipeline_lab.py` | Pipeline Lab: try pre-processing and segmentation on any image, by hand |
| `tools/pipeline_steps.py` | Pre-processing steps for the Pipeline Lab, and running them in a chosen order |
| `tools/pipeline_sweep.py` | Sweep pre-processing recipes against the local glue-line detector, and score them |
| `tools/replay_capture.py` | Replay saved images through the automatic inspection stages, offline |
| `tools/roi_sweep.py` | Per-ROI recipe sweep: extract every region the trained model marks, then crop |
| `tools/train_bead_ml.py` | Train the gradient-boosted edge corrector (`segmentation.bead_ml`) on painted glue lines |
| `tools/width_report.py` | Width report: where each glue line is wider or narrower than a good line's natural variation allows |

### `CalibrateAPP/` and `measure/`

| File | What it is |
|---|---|
| `CalibrateAPP/calibration_math.py` | Reusable, testable camera-calibration math helpers |
| `CalibrateAPP/calibration_store.py` | Replace a calibration's related JSON files, rolling back failed saves |
| `CalibrateAPP/calibration_ui.py` | Themed two-stage ChArUco calibration for the industrial vision app |
| `CalibrateAPP/generate_board.py` | Print any saved ChArUco board profile at an exact physical size |
| `CalibrateAPP/generate_two_boards.py` | Generate the two uniquely numbered ChArUco boards used by two-board mode |
| `CalibrateAPP/measurement_accuracy.py` | Real-world ChArUco distance verification for a saved measurement plane |
| `CalibrateAPP/region_calibration_ui.py` | Operator page for the optional per-region homography calibration mode |
| `CalibrateAPP/verify_calibration_pipeline.py` | Deterministic numerical verification of calibration and plane projection math |
| `measure/aruco_plane.py` | Per-frame metric measurement plane from one configured ArUco marker |
| `measure/makeUndistored.py` | - |

## Documents

| Document | What it covers |
|---|---|
| `docs/README.md` | index of all documents |
| `docs/inspection/AUTOMATIC_INSPECTION.md` | operator cycle, providers, the local detector and its settings, Pipeline Lab, saved data |
| `docs/inspection/CONFIGURATION.md` | cameras, preview, crops, calibration and measurement settings |
| `docs/inspection/PERFORMANCE.md` | load on a low-power PC |
| `docs/inspection/UI_DESIGN.md` | the visual system of the windows |
| `docs/calibration/CALIBRATION_PIPELINE.md` | lens calibration, measurement plane, millimetre scale |
| `docs/glue-line/GLUE_LINE_FINDINGS.md` | every measurement of Part C, with the methods that failed |
| `docs/handbook/` | this book: chapters, diagrams, charts, build script |
| `CLAUDE.md`, `.claude/skills/glue-line-vision/` | layout, rules and glue-line knowledge for coding assistants |

## Result pages (outside the repository)

Each is a folder under `~/flod_experiments/out/` with an `index.html`.

| Folder | Shows |
|---|---|
| `lab_validation`, `lab_validation2` | leak repair on 18 validation frames |
| `v4_m1_all` | version 4 on all 74 frames |
| `local_gpu_m1_all` | the local GPU flow on all 74 frames |
| `v5_m1_all`, `v5b_m1_all` | version 5 on all 74 frames |
| `compare_stages` | versions 2 to 5 on the same lines |
| `segformer_vs_sam` | the trained segmenter against version 5 |
| `main_model_check` | the glue model on `main`, with and without trees and ribbon |
| `width_report` | width in mm along every line, changes and glitches |

## Environments

| Environment | Python | Used for |
|---|---|---|
| Inspection PC, project `.venv` | 3.12 | the application: NumPy, OpenCV, MediaPipe, TensorFlow / tf-keras, inference-sdk |
| `~/flod_experiments/gpu` | 3.12 | research: PyTorch 2.11 with CUDA, transformers, onnxruntime, ultralytics, scikit-learn, lightgbm; also `markdown` and `matplotlib` for this book |
| System Python on the development PC | 3.14 | lacks several dependencies; not used |

## Rebuilding this book

```text
python docs/handbook/build.py
```

- Text: `docs/handbook/chapters/*.md`. Diagrams: `docs/handbook/diagrams/*.dot` (Graphviz).
  Charts: `docs/handbook/charts.py`. Layout: `docs/handbook/style.css`.
- Needs the `markdown` package, `matplotlib`, Graphviz and Google Chrome or Chromium.
- The example pictures in `figures/*.jpg` are committed. `pictures.py` remakes them and
  needs the sample frames and result pages, which are not in the repository.
- Chapter, section and figure numbers and the contents list are made by the build.
