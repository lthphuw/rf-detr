---
description: Run RF-DETR object detection on images, video, and streams. Nano to 2XLarge models with 2.3-17.2 ms latency and up to 60.1 AP on COCO.
---

# Run an RF-DETR Object Detection Model

RF-DETR is a real-time transformer architecture for object detection, built on a DINOv2 vision transformer backbone (a PE-Core-T backbone for Atto, Femto and Pico). The base models are trained on the Microsoft COCO dataset and achieve state-of-the-art accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR offers model sizes from Atto to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1. Core models (Nano to Large) are licensed under Apache 2.0. Atto, Femto, Pico, XLarge and 2XLarge (marked with △) are provided by the [`rfdetr_plus`](https://github.com/roboflow/rf-detr-plus) extension (`pip install rfdetr[plus]`) under the Platform Model License 1.0 and require a Roboflow account.

| Size | RF-DETR package class | Inference package alias | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :--: | :-------------------: | :---------------------- | :------------------: | :---------------------: | :----------: | :--------: | :--------: | :--------: |
|  A   |    `RFDETRAtto` △     | `rfdetr-atto`           |         49.4         |          30.5           |     1.0      |    7.4     |  380x380   |  PML 1.0   |
|  F   |    `RFDETRFemto` △    | `rfdetr-femto`          |         55.9         |          37.8           |     1.4      |    7.9     |  384x384   |  PML 1.0   |
|  P   |    `RFDETRPico` △     | `rfdetr-pico`           |         60.2         |          41.6           |     1.7      |    8.4     |  560x560   |  PML 1.0   |
|  N   |     `RFDETRNano`      | `rfdetr-nano`           |         67.6         |          48.4           |     2.3      |    30.5    |  384x384   | Apache 2.0 |
|  S   |     `RFDETRSmall`     | `rfdetr-small`          |         72.1         |          53.0           |     3.5      |    32.1    |  512x512   | Apache 2.0 |
|  M   |    `RFDETRMedium`     | `rfdetr-medium`         |         73.6         |          54.7           |     4.4      |    33.7    |  576x576   | Apache 2.0 |
|  L   |     `RFDETRLarge`     | `rfdetr-large`          |         75.1         |          56.5           |     6.8      |    33.9    |  704x704   | Apache 2.0 |
|  XL  |   `RFDETRXLarge` △    | `rfdetr-xlarge`         |         77.4         |          58.6           |     11.5     |   126.4    |  700x700   |  PML 1.0   |
| 2XL  |   `RFDETR2XLarge` △   | `rfdetr-2xlarge`        |         78.5         |          60.1           |     17.2     |   126.9    |  880x880   |  PML 1.0   |

> △ Requires the `rfdetr_plus` extension: `pip install rfdetr[plus]`

## Run on an Image

Perform inference on an image using either the `rfdetr` package or the `inference` package. To use a different model size, select the corresponding class or alias from the table above.

=== "rfdetr"

    ```python
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    ```

=== "inference"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from inference import get_model

    model = get_model("rfdetr-medium")

    image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
    predictions = model.infer(image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)

    annotated_image = sv.BoxAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
    ```

!!! note "Using COCO classes vs. fine-tuned model classes"

    `COCO_CLASSES` works for COCO-pretrained models (80 COCO classes, indexed 0-79). For fine-tuned models, use `detections.data["class_name"]` instead — it resolves class names from the checkpoint and works for both COCO and custom datasets.

`predict()` resizes without antialiasing by default (`antialias=False`), which matches checkpoints trained with the default CPU augmentation backend when `rfdetr[augment]` is installed (Albumentations). Pass `antialias=True` for checkpoints trained with torchvision resizing: the Kornia/GPU augmentation backend, the CPU backend without `rfdetr[augment]`, or older RF-DETR releases. Antialiasing costs a little extra preprocessing time per image. Exported models and the `rfdetr.export` runtime helpers always resize without antialiasing, so `antialias=True` results will not match them unless you pre-resize the image with antialiasing yourself. On MPS with PyTorch older than 2.7, `antialias=True` resizes on the CPU because MPS has no antialiased resize kernel there.

For repeated inference with the `rfdetr` package on CUDA, a fixed batch size, and a fixed resolution, the direct CUDA Graph backend records the TorchScript forward once and replays it. Capture allocates a graph-private memory pool holding the captured forward's intermediates plus the static input/output buffers; it persists for the graph's lifetime and scales with batch size, resolution and model, so this backend uses more device memory than the default TorchScript backend. The backend was measured on detection RF-DETR Nano at batch size 1; segmentation and keypoint models use the same path but are unmeasured, and replay/clone memory cost grows with masks and batch size. Outputs are cloned before they leave the graph, so predictions returned by an earlier call are not overwritten by the next one. The default stays `"torchscript"`:

```python
model.inference(compile_backend="cudagraph", batch_size=1, dtype="float16")
```

An operator that CUDA Graphs cannot capture makes `inference()` raise a `RuntimeError`. A failed capture can leave CUDA random-number state unusable, so restart the process before choosing another backend.

PyTorch Inductor is another opt-in backend for long-running inference. Cold compilation can have a higher one-time setup cost, but Inductor can apply broader graph optimizations and later processes may reuse its disk cache. Both examples require a compatible CUDA device, operators, and installed PyTorch version; `dtype="float16"` also requires FP16 support. The external `inference` package API shown above does not expose `RFDETR.inference()`:

```python
model.inference(compile_backend="inductor", batch_size=1, dtype="float16")
```

For memory-constrained inference-only deployments with the `rfdetr` package, optimize the loaded model in place before calling `predict()`. Pass `dtype="float16"` to halve weight memory in addition to clearing the base model reference. This operation is irreversible — to restore the original model, create a new `RFDETR` instance:

```python
model.inference(compile=False, inplace=True, dtype="float16")
```

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video source: <SOURCE_VIDEO_PATH>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    WEBCAM_INDEX = 0
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open webcam: {WEBCAM_INDEX}")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "stream"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream: <RTSP_STREAM_URL>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

### Run the detector on every other frame

`rfdetr.utilities.video.predict_video` runs the model on every `detect_every`-th frame and moves the boxes of the last detector frame with Lucas-Kanade optical flow on the frames in between. The flow runs on the CPU with [OpenCV](https://pypi.org/project/opencv-python-headless/), which `rfdetr` does not require directly (`pip install opencv-python-headless`; `pip install "rfdetr[augment]"` also brings it in). It needs no change to the model. Objects that appear between two detector frames are reported from the next detector frame on, and boxes drift on fast or non-rigid motion; a box can also stay in place for a few frames after its object has left the frame. Raise `detect_every` only as far as your scene allows. A frame that differs strongly from the previous one (most hard cuts) runs the detector, whatever `detect_every` says; a cut between two similar-looking or equally dark scenes can go unnoticed until the next scheduled detector frame. A propagated frame carries the boxes, classes, confidences and `data` of the detector frame, not `tracker_id` or `metadata`. Segmentation and keypoint models are refused because masks and keypoints are not propagated, and `include_source_image` cannot be passed.

```python
import cv2
from rfdetr import RFDETRSmall
from rfdetr.utilities.video import predict_video

model = RFDETRSmall()


def read_rgb_frames(path: str):
    video_capture = cv2.VideoCapture(path)
    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break
        yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    video_capture.release()


for detections in predict_video(model, read_rgb_frames("<SOURCE_VIDEO_PATH>"), detect_every=2, threshold=0.5):
    print(len(detections), "objects")
```

Measured with `RFDETRSmall` on an RTX 5070 (default eager `predict` with `threshold=0.25`, 150 frames per video, decoding excluded, best of five passes, on a machine that was also running an unrelated CPU job), against `predict` on every frame:

| video                           | detect_every=2 | detect_every=4 |
| ------------------------------- | -------------- | -------------- |
| `vehicles-2` (1080p)            | 1.35x faster   | 2.01x faster   |
| `people-walking` (1080p)        | 1.41x faster   | 2.17x faster   |
| `basketball-1` (1080p)          | 1.63x faster   | 2.81x faster   |
| `milk-bottling-plant` (1080p)   | 1.59x faster   | 2.68x faster   |
| `skiing` (1080p, moving camera) | 1.53x faster   | 2.59x faster   |
| `market-square` (2160x3840)     | 1.35x faster   | 2.11x faster   |

These are the [supervision](https://pypi.org/project/supervision/) video assets. The class-agnostic mean average precision (mAP50:95) of the propagated detections, scored against the detections of `predict` on every frame with confidence 0.5 or higher (not against human labels), was 0.82 to 0.97 at `detect_every=2` and 0.67 to 0.89 at `detect_every=4`.

The gain is the time of the skipped detector calls minus the optical-flow cost, which grows with the number of boxes: with hundreds of boxes (a very low `threshold` or a crowded scene) the flow can cost more than the detector call it replaces, so measure on your own footage. Against a model that is already optimized with `model.inference(compile=True, dtype=torch.float16)` (every-frame `predict` at 4.2 ms on 1080p), the same measurement gave 1.01x at `detect_every=2` and 1.37x at `detect_every=4` on `vehicles-2` (about 40 boxes per frame at `threshold=0.25`), and 1.25x and 1.96x on `skiing` (about 8 boxes per frame). With `model.inference(compile=True, dtype=torch.float16, compile_backend="cudagraph")` (3.9 ms) the figures were 0.98x and 1.30x on `vehicles-2`, and 1.21x and 1.87x on `skiing`.
