<!-- part: Part B - Inspection application (this branch) -->
# Module architecture

This part describes the application as it is on `tooling/pipeline-lab`. It is the
application of the common base plus an opt-in local detector; the colleague's later rewrite
is in Part D.

## Layers

![The application by layer. An arrow from one layer to the next means "calls" or "passes to".](figures/modules.svg)

The design keeps three things apart:

- **Tk widgets are touched by one thread only.** Everything slow runs elsewhere and reports
  back through a queue of events.
- **Segmentation is behind one contract** (chapter 6). The pipeline does not know whether
  the outline came from the cloud or from a local detector.
- **Measurement is the same code for every provider.** It needs only a polygon in frame
  pixels.

## Modules and what each is for

| File | Main names | Job |
|---|---|---|
| `main_1366.py` | `IndustrialDashboard` | The window: live view, bottom panel, settings, size and tolerance profile. Inherits the automatic behaviour from `AutomaticDashboard`. |
| `startup_settings.py` | `StartupWindow` | Five-second countdown at start; **S** opens the settings form. Runs before any module that caches settings. |
| `app_config.py`, `config_validation.py` | `CONFIG`, `validate_config` | Load `config.json`; reject invalid settings at start. |
| `camera_handler.py` | `CameraStream`, `CameraHandler` | Background capture at full resolution; keeps the newest frame; applies rotation and optional fixed exposure. |
| `measure/makeUndistored.py` | `ImageUndistorter` | Removes lens distortion with the saved calibration. |
| `inspection/dashboard.py` | `AutomaticDashboard` | Starts the controller, polls its events every 100 ms, shows results. |
| `auto_trigger/controller.py` | `AutoController` | A thread: hand detection, debounce, checkpoint, fabric gate, starting a capture, protection against stale results. |
| `auto_trigger/checkpoint.py` | `Checkpoint`, `Debouncer` | The hand-cycle state machine. |
| `auto_trigger/models.py`, `fabric_check.py` | `HandPresence`, `FabricGate` | MediaPipe hand landmarker on a 320 px wide copy; Keras fabric classifier. |
| `inspection/pipeline.py` | `InspectionPipeline` | One capture: segment, validate, measure, save. |
| `segmentation/__init__.py` | `create_segmenter`, `Instance`, `SegmentationResult`, `validate_result` | The provider contract. |
| `segmentation/provider.py`, `segmentation/workflow/` | `WorkflowSegmenter` | The cloud workflow provider (default). |
| `segmentation/glue_line.py`, `glue_shape.py` | `GlueLineSegmenter` | The local detector (opt-in), with the optional shape condition and region-guided search. |
| `inspection/measurement.py` | `measure_instances`, `FabricResult` | Per fabric: crop, measure, grade, annotate. |
| `sam_detection.py` | `analyze_detection`, `draw_analysis`, `strip_record` | Glue between a polygon and the strip analysis; drawing; the saved record. |
| `strip_analysis.py` | `analyze_ring`, `refine_widths`, `to_metric`, `grade_widths` | Centreline, widths, segments, millimetres, grading. |
| `plane_scale.py` | `load_frame_scale`, `PlaneScale` | Pixels to millimetres on the calibrated plane. |
| `CalibrateAPP/` | `CalibrationApp`, `calibration_math` | Calibration windows and their mathematics. |
| `inspection/storage.py` | `CaptureStore` | One folder per capture; retention limit. |
| `inspection/result_view.py`, `crop_result_view.py` | `FabricResultView` | Result tabs with zoom and pan. |
| `app_storage.py` | `data_path` | Where user data lives (Documents / `data files`), apart from code. |
| `tools/replay_capture.py` | | Runs saved images through the same pipeline, offline. |

## What is not part of the application

Everything in chapter 17 (the Pipeline Lab and the research modules `sam_refine`,
`sam_local`, `bead_edges`, `ribbon`, `bead_ml`, `edge_methods`, `synthetic_bead`,
`width_stats`, `segformer_local`) is used only by the tools. The application does not
import them, with one exception: `glue_line.py` imports `sam_refine` when the opt-in key
`sam_refine` is turned on.
