# FLOD MAS on Jetson: CUDA method timing and accuracy

16 October 2026. Measured on the Jetson; the tools are in `tools/method_compare` (docs/METHOD_COMPARE.md).

The fastest method on the Jetson is the app's own pipeline: both models on CUDA, the ribbon finish, the distance transform on the GPU and OpenCV on 4 threads. It reads a board photo in 1.8 s, under the 4 s target, with readings bit-identical to today's 3.0 s setting.

- **The models are not the bottleneck.** On CUDA they take 0.16 s of a photo; TensorRT FP32 and FP16 save at most 0.1 s and cost 0.5-1.4 GB of memory.
- **The time is in CPU steps around the models**, which the app runs on one OpenCV thread. Four threads and the GPU distance transform take a photo from 3.0 s to 1.8 s.
- **Moving the other steps to the GPU needs own CUDA kernels.** As library calls (CuPy, Kornia, VPI) they are slower on the Orin Nano than OpenCV, or change the readings.
- **Version 5 with SAM does not fit:** SAM 2.1 large takes 3.4-11 s a photo, SAM 2.1 tiny reads unstably.

## Setup

All timings are on the Jetson itself, in its fastest power mode, with nothing else running.

| Item | Value |
|---|---|
| Board | Jetson Orin Nano 8 GB Super, power mode MAXN_SUPER |
| Software | JetPack 6.2.2 (L4T r36.5.2), CUDA 12.6, TensorRT 10.3, Python 3.10 |
| ONNX Runtime | onnxruntime-gpu 1.24.0 with TensorRT, CUDA and CPU providers |
| Code | branch dev-jetson at 698bf6f (dev merged in: scene mode, ribbon finish) |
| Test photos | 8 scene-mode key photos from the two recordings (1, 2, 3 and 4 cloths per board), 4608 x 3456 px |
| Lines per photo | 1 to 4; 20 line readings per pass |

Photos go in as recorded, without lens correction, so every method reads exactly the same pixels. The first photo of each run is not counted: TensorRT builds or loads its engines there.

### What the Jetson runs today (dev-jetson)

The live app already shows a board photo in 2.9-3.9 s. These are its own log lines from the scene-mode run of 14 October: finding the lines takes 1.2-1.7 s and the one new line 1.3-2.2 s.

| Setting on the Jetson | Value | Why (commit on dev-jetson) |
|---|---|---|
| `local_inspection.execution_provider` | `cuda` | both ONNX models on the GPU; TensorRT was about 6 % faster but used about 1 GB more memory (c4024ef) |
| `local_inspection.ribbon_finish` | `true` | trees and ribbon over the glue mask (method B) |
| `local_inspection.line_workers` | `2` | two lines measured at once; 1.85 s to 1.67 s per capture, bit-identical (c4024ef) |
| `capture_storage.background_save` | `true` | the two 16 MP PNGs no longer delay the result, about 3 s saved (b1260e1) |
| Line geometry and enhancement | array passes instead of Python loops | bit-identical; about 7 s to 3 s per 2-line capture with the above (cb4df62, ee8af6d) |
| CUDA memory pool | grows only by what is asked | about 360 MB less, same speed (c4024ef) |
| `auto_trigger.mode` | `scene` | whole-board photo at each still moment |

Also on the Jetson but not used yet: `nvdec_reader.py`, unfinished work that decodes the MJPEG video on the Jetson's hardware decoder (about a fifth of the 90 ms of CPU per frame). It is not wired into `video_feed.py` and is not committed.

## Methods compared

Two families are timed: the app's own pipeline (the glue model from `main`) and the research pipeline, version 5, from `tooling/pipeline-lab`.

| Method | Finds the strip with | Reads the glue with | Edges and finish | Runs on |
|---|---|---|---|---|
| A. App pipeline | strip U-Net (ONNX) | SegFormer-B0 glue model (ONNX) | boundary of the cleaned mask | ONNX Runtime: CPU, CUDA, TensorRT FP32, TensorRT FP16 |
| B. App pipeline + ribbon finish | strip U-Net (ONNX) | SegFormer-B0 glue model (ONNX) | band edges corrected by the trained trees, smooth ribbon (`ribbon_finish`) | as A |
| C. Version 5, SAM 2.1 large | glue-region model (ONNX) | SAM 2.1 large, half precision, asked once per line | band edges, trees, ribbon | PyTorch on CUDA |
| D. Version 5, SAM 2.1 tiny | as C | SAM 2.1 tiny, half precision | as C | PyTorch on CUDA |
| E. Version 5 without SAM | as C | rough line only | band edges, trees, ribbon | ONNX + CPU |

On the development PC (RTX 3050) the handbook measured C at 4.8-6.5 s per frame, D at 2.6-3.3 s and E at 1.0-1.2 s (speed chapter). The same nine frames took 3.4 s with A on the PC's CPU (comparison chapter).

## Timing results

With CUDA and four OpenCV threads a board photo takes 2.1 s to read with the ribbon finish, against 3.0 s for the app today, and the readings are bit-identical. Every GPU setting meets the 4 s target; the GPU providers differ by under 0.1 s, because the two models are no longer what takes the time.

- **TensorRT does not pay on the Jetson.** FP32 gains nothing over CUDA and needs about 0.5 GB more memory; FP16 saves 0.1 s, needs 1.4 GB more and 6.5 minutes to build its engine the first time.
- **The ribbon finish costs 0.65 s per line**: one new line takes 1.47 s with it, 0.80 s without.
- **Two line workers only help when several lines are new at once**: 4 lines take 3.4 s instead of 4.0 s; one line takes the same.
- In the live app add about 0.15 s for the board picture and the screen: its own log shows 2.9-3.9 s.

### Where the time goes on the CUDA path

The GPU does about 5 % of the work. Three full-frame OpenCV calls on the CPU take 1.4 s of each photo, and they are what to move onto the GPU next. Profile of the app's case (find the lines + one new line, CUDA, ribbon on), 8 photos; the profiler itself adds about 8 %.

| Step | Per photo | Runs on | Called from |
|---|---|---|---|
| Connected components on full 16 MP masks | 0.65 s | CPU | source mask hole filling, wave-line filter, line mask cleaning |
| Colour conversions (BGR to LAB and back, HSV, grey) | 0.40 s | CPU | photo enhancement, cardboard mask, band edges |
| Distance transform | 0.39 s | CPU | wave-line filter, line geometry |
| Both ONNX models | 0.16 s | GPU | strip U-Net and SegFormer glue model |
| CLAHE | 0.12 s | CPU | photo enhancement |
| Python around them: geometry, trees, ribbon, widths | about 1.5 s | CPU | many small steps |

### Moving those steps to the GPU does not pay on the Orin Nano

The app runs OpenCV on one thread (`performance.opencv_threads: 1`, chosen for the low-power Windows PC). On the Jetson's six cores, OpenCV with more threads beats the GPU at these steps. GPU versions, timed on one 16 MP key photo and its source mask:

| Step | CPU, OpenCV 6 threads | GPU, data already there | GPU with copies | Same result as the app |
|---|---|---|---|---|
| Connected components (CuPy) | 13.6 ms | 17.9 ms | 30.0 ms | yes |
| Morphological gradient 3x3 (CuPy) | 13.4 ms | 47.2 ms | 52.0 ms | yes |
| Distance transform (CuPy, exact) | 256 ms | 53 ms | 112 ms | no: up to 16 px from OpenCV's mask-5 approximation |
| LAB + CLAHE enhancement (Kornia on PyTorch) | 113 ms | 307 ms | 316 ms | no: 2 levels off on average |
| BGR to grey (NVIDIA VPI) | 3.2 ms | 6.4 ms | 10.2 ms | within 1 level |

OpenCV's own CUDA build from the Jetson AI Lab index cannot load on JetPack 6: it links the desktop video codec libraries (`libnvcuvid`, `libnvidia-encode`) that the Jetson does not ship. Raising `performance.opencv_threads` gives the same CPU steps the free cores instead, with bit-identical readings: 4 threads gain as much as 6.

**The distance transform is the one step that gains on the GPU as a library call.** Run inside the app's pipeline with CuPy's exact transform for every full-size image, finding the lines falls from 1.06 s to 0.81 s (4 threads) and from 1.48 s to 1.22 s (1 thread). All 20 lines and every segment read bit-identically: the exact distances differ from OpenCV's only where the wave-line filter decides nothing.

### Version 5 (pipeline-lab) on the Jetson's GPU

SAM 2.1 large misses the target on the Jetson: 3.5 s for a photo with one line, 11.1 s with four, because SAM costs about 1.9 s per line here (0.3 s on the PC's RTX 3050). Run with the lab's own code, PyTorch 2.11 on CUDA in half precision, region model on CUDA, trained-tree edges and ribbon; mean of the same 16 photos.

| Version 5 | Photo, 1 line | Photo, 4 lines | One new line, estimated | GPU memory | Peak RAM | First photo |
|---|---|---|---|---|---|---|
| C. SAM 2.1 large | 3.5 s | 11.1 s | 3.4 s | 743 MB | 4.2 GB | 16.5 s |
| D. SAM 2.1 tiny | 2.0 s | 5.9 s | 1.9 s | 288 MB | 3.4 GB | 11.6 s |
| E. Without SAM | 1.3 s | 3.4 s | 1.2 s | none | 1.9 GB | 3.0 s |

Version 5 reads every line of a photo again; the app reads only the new one. "One new line, estimated" is the fixed cost of a photo plus one line's share, from the per-stage times, for a version 5 that read only new lines as the app does. Software for C and D: a separate venv with the Jetson AI Lab build of PyTorch 2.11 for CUDA 12.6 (PyPI's aarch64 build is for CUDA 13 and does not run on the Orin) and NVIDIA cuDSS.

## Accuracy results

The GPU reads the same widths as the CPU: on all 20 lines the mean width moves by at most 0.09 px (about 0.01 mm), so any GPU provider can be used without changing readings.

| Compared | Lines | Mean width, largest change | One segment, largest change | Centre line, largest shift |
|---|---|---|---|---|
| CUDA against CPU (ribbon off) | 20 of 20 | 0.04 px | 0.20 px | 0.09 px |
| TensorRT FP32 against CPU | 20 of 20 | 0.07 px | 0.28 px | 0.15 px |
| TensorRT FP16 against CPU | 20 of 20 | 0.09 px | 0.27 px | 0.21 px |
| CUDA, 2 line workers against 1 (ribbon on) | 20 of 20 | 0.00 px | 0.00 px | 0.00 px |
| TensorRT FP16 against CUDA (ribbon on) | 20 of 20 | 0.12 px | 0.83 px | 0.22 px |
| Ribbon on against off (both CUDA) | 20 of 20 | 1.30 px (mean +0.82 px) | 6.6 px | 5.3 px |

Version 5 against the app's CUDA reading with the ribbon finish, the same 20 lines matched by position:

| Method | Lines matched | Width, version 5 minus app | Same line across photos, SD median / worst |
|---|---|---|---|
| C. SAM 2.1 large | 20 of 20 | -3.0 px (-6.0 to -1.4) | 0.67 / 1.32 px |
| D. SAM 2.1 tiny | 20 of 20 | -0.7 px (-5.1 to +17.4) | 1.81 / 7.97 px |
| E. Without SAM | 20 of 20 | +0.15 px (-0.3 to +0.9) | 0.29 / 0.77 px |
| App, CUDA + ribbon (reference) | 20 | - | 0.14 / 0.54 px |

SAM 2.1 large reads about 3 px (0.4 mm) narrower, as the handbook found on pink fabric (2.6 px). SAM 2.1 tiny is unstable from photo to photo. Version 5 without SAM agrees with the app's ribbon finish to 0.15 px, as expected: the app's ribbon finish is that method, ported. Each version 5 run found one line more than the app (21); the app's 20 were all among them. Version 5 reports a line's median width, the app its mean.

- **The ribbon finish is a different reading, not noise.** It reads lines 0.8 px (about 0.1 mm) wider on average and moves single segments by up to 6.6 px, because it replaces the glue mask's own edge with band edges corrected by the trained trees.
- **The same line reads the same from photo to photo.** Across the photos of one board, a line's width varies by 0.0-0.5 px (SD), with or without the ribbon and on every provider.
- **Which reading is closer to the truth was measured on the PC, not here.** On painted lines with known edges the trained trees err by 0.63 px RMS and the plain steepest slope by about 1 px (accuracy chapter). That result does not depend on the hardware.

## Recommendation

Ship the app pipeline with the ribbon finish, both models on CUDA, the distance transform on the GPU and OpenCV on 4 threads: 1.8 s a photo, readings bit-identical to today's. Then move the remaining steps onto the GPU one at a time, each kept only when its readings are unchanged.

1. **Now, settings only** (Jetson's `machine_settings.json`): keep `execution_provider: cuda`, `ribbon_finish: true`, `line_workers: 2`; add `performance.opencv_threads: 4`. 3.0 s to 2.1 s per photo in the bench; to be confirmed in the running app, where video decoding, markers and the hand model share the cores.
2. **Next, small opt-in change on dev-jetson**: distance transforms of full-size images on the GPU with CuPy (`cupy-cuda12x` 14.2.0 in the Jetson venv), behind a new setting that defaults to off. 2.1 s to 1.8 s, bit-identical.
3. **Then, every remaining step onto the GPU**, keeping the photo and its masks on the GPU from upload to result, so nothing is copied back between steps:

| Step | Time now (4 threads) | GPU route | What is known | Risk to readings |
|---|---|---|---|---|
| Both models | 0.16 s | CUDA (done) | in use | none |
| Distance transform | 0.25 s saved | CuPy exact transform | measured: -0.25 s | none: bit-identical |
| Connected components, 5 call sites | about 0.2 s | CuPy labels, kept on the GPU | one call alone is slower (18 ms against 14 ms); gains only inside a GPU-resident chain | labels must match; areas matched |
| Morphology (gradient, dilate) | about 0.05 s | own CuPy kernel | library version 3.5 times slower; own kernel not yet written | none if exact |
| Photo enhancement, LAB + CLAHE | about 0.11 s | own CUDA kernel copying OpenCV's fixed-point LAB tables and CLAHE | Kornia version 2.7 times slower and 2 levels off | high: must be bit-exact or every mask moves |
| Trees and ribbon features | about 0.4 s | CuPy features; LightGBM stays on the CPU (no GPU predict on the Jetson) | not tried | low |
| Line geometry, Python array steps | about 0.8 s | CuPy or fused kernels | many small steps: kernel launches may cost more than they save | none if exact |
| Video decoding (outside inspection) | about 90 ms of CPU per frame | Jetson hardware decoder (`nvdec_reader.py`, unfinished) | measured in the earlier session: a fifth of the CPU | 1-2 levels on some pixels |

**Not recommended on the Jetson:** TensorRT FP32 and FP16 (no speed gain, 0.5-1.4 GB more memory, a 6.5-minute first build for FP16); SAM 2.1 large (3.4-11 s, 4.2 GB); SAM 2.1 tiny (unstable widths).

## Limits and next steps

All figures come from 8 photos of two recordings of one product family on one bed, on a Jetson with nothing else running; none is from a live camera.

- **Bench, not the running app.** The bench runs inspection alone. In the app, decoding 16 MP video, markers, the hand model and the screen share the six cores, so the 4-thread gain must be confirmed there.
- **Accuracy against the truth is from the PC.** Here every method is compared with the app's reading and with itself across photos; the painted-truth bench was not run on the Jetson.
- **Version 5 "one new line" times are estimates** from its stage times; it reads every line of a photo today.
- **Not tried:** own CUDA kernels for morphology, enhancement and geometry; `torch.compile` or TensorRT for SAM; the hardware video decoder in the app; power modes other than MAXN_SUPER.

- Set `performance.opencv_threads: 4` on the Jetson and time one cycle of both recordings in the running app
- Add the GPU distance transform to dev-jetson as an opt-in setting, with a bit-identical check on the 8 key photos
- Write and verify an own CUDA kernel for the morphology steps, the first of the GPU-resident chain
- Finish the hardware video decoder (`nvdec_reader.py`) to free CPU for inspection

The bench scripts and raw results are on the Jetson in `~/flod_bench` (`bench_onnx.py`, `bench_v5.py`, `gpu_ops_bench.py`, `callsites.py`, `results/`). The research venv is `~/.venvs/flod_lab`; the app's venv was not changed.

## Glue-width defects on the Jetson

SAM 2.1 large and the pipeline-lab SegFormers were also run on the Jetson to find narrow
glue on the sticker-marked defect recordings; see docs/GLUE_DEFECTS.md
(`tools/glue_defects`, `tools/jetson/run_glue_defects.sh`).
