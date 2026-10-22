# Comparing the glue-line methods on the Jetson

`tools/method_compare` runs every glue-line method on the still moments of the
recordings and shows them side by side in a window on the Jetson's desktop (VNC
`:1`, 1366 x 768). Nothing in it is used by the app.

| Id | Method | Runs on |
|---|---|---|
| A | App glue model, ribbon finish off | CUDA, OpenCV 4 threads |
| B | App glue model + ribbon finish, as the app runs today | CUDA, OpenCV 1 thread |
| B-fast | B with OpenCV 4 threads and the distance transform on the GPU (same readings as B) | CUDA + CuPy |
| C | Version 5 (pipeline-lab), SAM 2.1 large | PyTorch CUDA FP16 |
| D | Version 5, SAM 2.1 tiny | PyTorch CUDA FP16 |
| E | Version 5 without SAM | CUDA (region model) + CPU |

## Use

```bash
tools/jetson/run_method_compare.sh             # prepare once (about 10 minutes), then open the window
tools/jetson/run_method_compare.sh --prepare   # prepare a new run first
```

- **Watch**: the recording plays; at each still moment (the photo the app's scene
  mode takes) every method's lines appear with their widths and the time the photo
  took. Space plays and pauses, the arrow keys jump between still moments.
- **Accuracy**: one line close up with every method's two edges over the photo;
  wheel to zoom, drag to move, tick boxes to hide a method. Beside it, each line's
  width per method and how much a line's width changes from photo to photo.
- **Speed**: time per photo of each method against the 4 s target, and per still moment.

A run is kept in `~/flod_mas_data/method_compare/<date_time>/` (still-moment photos,
playback frames, one JSON per method). Open an older run with
`python -m tools.method_compare.viewer <folder>`; `--show Speed --moment 3
--screenshot out.png` saves a picture of a tab and closes.

## What the times mean

- App methods: finding the lines plus measuring the lines that are new on the board,
  measured, exactly as scene mode does it.
- Version 5 reads every line of a photo. Its photo time is an estimate for reading
  only the new lines: its fixed stages plus each new line's share of the per-line
  stages, with the new lines counted by the app (B). Its measured time for all lines
  is shown beside it.
- The still moments are picked by the app's own scene monitor; the hand check is not
  run (MediaPipe is not in the research venv).

## The research venv

Methods C, D and E and B-fast need `~/.venvs/flod_lab`, separate from the app's venv:

```bash
python3 -m venv ~/.venvs/flod_lab
F=https://pypi.jetson-ai-lab.io/jp6/cu126/+f
~/.venvs/flod_lab/bin/pip install numpy==2.2.6 scipy==1.15.3 scikit-image==0.25.2 lightgbm==4.7.0 pillow \
  opencv-contrib-python==4.12.0.88 transformers==5.18.0 cupy-cuda12x==14.2.0 \
  "torch @ $F/46b/b8b13f844b211/torch-2.11.0-cp310-cp310-linux_aarch64.whl" \
  "torchvision @ $F/d11/6f08d3d62417d/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl" \
  "onnxruntime-gpu @ $F/d98/0b934b9a29c1a/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"
~/.venvs/flod_lab/bin/pip install --no-deps nvidia-cudss-cu12
```

- torch must be the Jetson AI Lab build named above: PyPI's aarch64 torch of the same
  version is built for CUDA 13 and does not run on the Orin.
- torch needs NVIDIA cuDSS. Install it without its dependencies: the pip cuBLAS it
  would bring shadows the Jetson's own and the GPU fails (`CUBLAS_STATUS_ALLOC_FAILED`).
  `run_method_compare.sh` puts cuDSS and `/usr/local/cuda/lib64` on the library path.
- OpenCV's CUDA build from the same index does not load on JetPack 6 (it needs the
  desktop video codec libraries), so the plain OpenCV wheel is used.
- Version 5 needs the pipeline-lab checkout at `~/flod_lab`
  (`git clone -b tooling/pipeline-lab https://github.com/Idea8PvtLtd/FLOD_MAS.git ~/flod_lab`,
  with Git LFS), the SAM 2.1 weights in `~/.cache/huggingface/hub` and the trees model
  at `~/.cache/flod_sam/bead_edge_trees.txt`.
