# Documentation

All project documents live in this folder, grouped by subject. `CLAUDE.md` stays in the
repository root because coding assistants read it from there; two READMEs stay beside the
code they describe.

## Inspection application

| Document | What it covers |
|---|---|
| [Automatic inspection](inspection/AUTOMATIC_INSPECTION.md) | Operator cycle, trigger, segmentation providers, the local glue-line detector and its opt-in guided search, the Pipeline Lab tool, the width report, saved data |
| [Application settings](inspection/CONFIGURATION.md) | Cameras, preview, crops, calibration and measurement settings of `config.json`; the trigger and segmentation keys are tabled in *Automatic inspection* |
| [Performance](inspection/PERFORMANCE.md) | Load on a low-power PC, what was changed, what to verify on the target machine |
| [Interface design](inspection/UI_DESIGN.md) | The shared Tk/ttk visual system |

## Calibration

| Document | What it covers |
|---|---|
| [Camera calibration pipeline](calibration/CALIBRATION_PIPELINE.md) | Lens calibration, measurement plane, millimetre scale, what is mathematically valid |

## Glue-line detection research

| Document | What it covers |
|---|---|
| [Findings and methodology](glue-line/GLUE_LINE_FINDINGS.md) | Strategies in use and those that failed, accuracy against painted truth, time per stage, local GPU flow, SegFormer and CLAHE, the glue model on `main`, width in millimetres with statistical limits and mask glitches, ground facts, what is unverified |

Start with its *Strategies* section; *Not verified* and *Next steps* close it. Scripts,
trained models and result pages of the experiments are kept outside the repository in
`~/flod_experiments/`.

| Section of the findings | Code in the repository |
|---|---|
| Leak repair, filters before SAM | `segmentation/sam_refine.py`, `tools/pipeline_lab.py`, `tools/pipeline_steps.py` |
| The bead's own edges, smooth ribbon | `segmentation/bead_edges.py`, `segmentation/ribbon.py`, `tools/learn_bead_reference.py` |
| Time per frame, local GPU flow | `tools/benchmark_flow.py`, `segmentation/sam_local.py` |
| Accuracy | `tools/accuracy_bench.py`, `segmentation/synthetic_bead.py`, `segmentation/edge_methods.py`, `segmentation/bead_ml.py`, `tools/train_bead_ml.py` |
| Width statistics | `segmentation/width_stats.py`, `segmentation/segformer_local.py`, `tools/width_report.py` |

## Beside the code

| Document | What it covers |
|---|---|
| [`esp32_buttons/README.md`](../esp32_buttons/README.md) | The ESP32 button firmware |
| [`model_for_segment_bra_roi places/model_test_and_running_scripts/README.md`](../model_for_segment_bra_roi%20places/model_test_and_running_scripts/README.md) | The panel-region (strip) model: running, testing, ONNX conversion |
| [`CLAUDE.md`](../CLAUDE.md) | Layout, commands and rules for changes, for coding assistants |

## Adding a document

Put it in the folder for its subject (make a new folder for a new subject), give it a row in
the table above, and refer to other documents by relative path.
