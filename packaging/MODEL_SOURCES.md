## Model files

| File | Purpose | Origin |
|---|---|---|
| `inspection/local/models/strip_unet_resnet34.onnx` | Source-region segmentation | Trained in-house (Idea8) |
| `inspection/local/models/segformer_b0.onnx` | Adhesive segmentation | Trained in-house (Idea8) on a SegFormer-B0 architecture |
| `auto_trigger/models/hand_landmarker.task` | Hand detection for the automatic trigger | Google MediaPipe Hand Landmarker, Apache-2.0 |
| `auto_trigger/models/keras_model*.h5`, `labels*.txt` | Fabric presence classifier | Trained in-house with Teachable Machine |

Origins of the in-house models are stated by the build team and must be confirmed
before the first stable release (see the licence issue).
