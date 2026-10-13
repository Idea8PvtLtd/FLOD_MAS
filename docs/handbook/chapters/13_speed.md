# Speed: the local GPU flow

## Where the time goes

![Seconds per frame by stage, cloud flow against local GPU flow, mean of three frames per colour.](figures/chart_time_per_stage.svg)

![The two flows side by side. The stages are the same; where they run and how often SAM is asked differ.](figures/gpu_flow.svg)

| Flow | Seconds per frame: dark / pink / white | Uploads | GPU memory | Against cloud version 4 |
|---|---|---|---|---|
| Cloud version 4 | 22.1 / 44.9 / 23.0 | every tile | - | reference |
| **Local SAM 2.1 large, asked once, ribbon** | **5.7 / 6.5 / 4.8** | none | 1.0 GB | centre line 0.1-0.5 px away; width 0.4-1.7 px narrower; 257 of 257 lines "ok" |
| Local SAM 2.1 tiny | 3.1 / 3.3 / 2.6 | none | 0.6 GB | 4 pink lines "check" |
| No SAM: rough line + band edges + ribbon | 1.2 / 1.2 / 1.0 | none | 0.5 GB | 19 pink lines "check"; 2 lines over 3 px off |

Measured on the development PC: Intel i5-13420H, RTX 3050 Laptop GPU with 6 GB, PyTorch
2.11 with CUDA. The first run of a session takes 11-16 s while models load.

## What made the difference

| Change | Effect |
|---|---|
| SAM on the PC's own GPU in half precision | a tile is encoded in 0.3 s and can then be asked again for about 10 ms; a cloud call takes 0.7-0.9 s |
| Each line asked once | the band edges cover what SAM missed, so the repeat askings add nothing once the ribbon is on |
| Whole-frame filters skipped in the rough line | 90 % of that stage's time went into filters the region-guided search never read |
| Region model on the GPU through its PyTorch checkpoint | 2.1 s to 0.25 s |

Half precision is the one setting that matters for SAM: full precision is 5.8 times slower
and needs 3 GB. Batching tiles gains nothing.

## Proof that it is offline

One frame was run with the network removed (`unshare -n`), `HF_HUB_OFFLINE=1`, no API key in
the environment and the cloud client replaced by one that raises an error. It completed:
three lines, widths 33.6, 32.4 and 30.0 px.

## Not done

- Nothing was timed on the Windows inspection PC.
- `torch.compile`, TensorRT and ONNX export of SAM were not tried.
- Local SAM 3 was not run: its weights are gated and need a Hugging Face login.
- The application's own cycle (`docs/inspection/PERFORMANCE.md`) was not re-timed.
