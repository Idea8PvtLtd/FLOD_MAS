# Research tools

## The map

![Tools, the modules they drive and what they write. None of the green boxes are used by the inspection application.](figures/tools_map.svg)

## The Pipeline Lab

![The Pipeline Lab window in an early state: region of interest, pre-processing steps and segmenter on the left; the picture and the per-line numbers on the right. Later options (where SAM runs, asking once, the ribbon, how edges are placed) were added below these.](figures/photo_lab_window.jpg)

`tools/pipeline_lab.py` is a window for trying things on any image. It is separate from the
application and changes nothing in it. A set-up can be saved and loaded as a recipe (JSON),
and `run_job(image, recipe, ...)` runs the same flow from a script.

| Recipe key | Values | Meaning |
|---|---|---|
| `roi` | `whole`, `model`, `box` | region of interest |
| `pipeline` | list of steps | pre-processing, top to bottom (none is recommended) |
| `segmenter` | `local`, `sam3` | local detector only, or SAM with prompts |
| `point_source` | `lines`, `clicks` | SAM's points placed along the rough lines, or clicked by hand |
| `sam_runs` | cloud, `local` | Roboflow, or this PC's GPU |
| `local_model` | `sam2.1-large`, `-base-plus`, `-small`, `-tiny`, `sam3`, `mobile-sam`, `fastsam-s`, `fastsam-x` | which local model |
| `ask_once` | true / false | skip the repeat askings |
| `ribbon` | true / false | fit the smooth ribbon |
| `edge_method` | `slope`, `profile_fit`, `trees` | how the band's edges are placed |

## Commands

All from the repository root, with `~/flod_experiments/gpu/bin/python` on the development
PC.

| Purpose | Command |
|---|---|
| Open the window | `python tools/pipeline_lab.py [image]` |
| Time each stage | `python tools/benchmark_flow.py --sam local` |
| Score edge methods against painted truth | `python tools/accuracy_bench.py --runs DIR` |
| Train the edge-correcting trees | `python tools/train_bead_ml.py --runs DIR` |
| Learn the black-fabric reference | `python tools/learn_bead_reference.py NAME frames...` |
| Learn a wave template for the shape condition | `python tools/learn_glue_shape.py NAME frames/*.png` |
| Write the width report | `python tools/width_report.py --runs DIR --out DIR --mask-model DIR --held-out FILE` |
| Replay saved images through the application's pipeline | `python tools/replay_capture.py images/*.jpeg` |
| Run the tests | `python -m unittest discover -s tests -p "test_*.py"` |

`DIR` is a folder of cached runs (one file per frame, as in `~/flod_experiments/segformer/runs`). Each script prints its own options with `--help`.

## What lives outside the repository

| Place | Content |
|---|---|
| `~/flod_experiments/gpu/` | Python environment: PyTorch with CUDA, transformers, onnxruntime, ultralytics, scikit-learn, lightgbm |
| `~/flod_experiments/segformer/` | cached version 5 runs of the 74 frames, training scripts, three trained models |
| `~/flod_experiments/main_model/` | read-only copy of `inspection/local/` from `main`, and the scripts that scored it |
| `~/flod_experiments/out/` | result pages, each with an `index.html` |
| `~/.cache/flod_sam/` | the trees model, MobileSAM and FastSAM weights |
| `enhancement_images/` (in the folder, git-ignored) | the sample frames |

Model weights (`*.onnx`, `*.pt`) and the image sets are git-ignored. The region model file
must be supplied separately.

## Tests

`tests/` holds 366 tests on this branch; all pass. Those for the research code are
`test_pipeline_lab.py`, `test_accuracy_bench.py` and `test_width_stats.py`. They use
synthetic images, never a camera, the cloud or a GPU.
