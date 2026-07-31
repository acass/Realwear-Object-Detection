# RealWear Pose Estimation

Real-time human pose estimation on the camera preview of a RealWear head-mounted
device, using YOLO26 nano pose running on-device via TensorFlow Lite. A 17-keypoint
skeleton is drawn over the live preview for every person in frame, and tracking can be
paused and resumed by voice through RealWear's WearHF system.

Status: verified on RealWear T21G (Navigator 500) hardware, Android 13 / arm64-v8a, at
roughly 4 frames per second (237 ms per frame). Whether the skeleton tracks a moving
person well enough at that rate is a judgement call yet to be made on the headset.

## How it works

```
CameraX preview  ──>  PreviewView            (what you see)
       │
       └─ ImageAnalysis ──> center-crop square ──> Detector ──> OverlayView
          (KEEP_ONLY_LATEST)                       (TFLite)     (skeletons)
```

- **[MainActivity.kt](app/src/main/java/com/crossmedia/objectdetect/MainActivity.kt)** —
  camera permission, CameraX binding, frame preprocessing, overlay sizing.
- **[Detector.kt](app/src/main/java/com/crossmedia/objectdetect/Detector.kt)** —
  TFLite interpreter, input packing, delegate choice.
- **[YoloPostProcessor.kt](app/src/main/java/com/crossmedia/objectdetect/YoloPostProcessor.kt)** —
  pose output decoding and non-max suppression.
- **[OverlayView.kt](app/src/main/java/com/crossmedia/objectdetect/OverlayView.kt)** —
  draws the current skeletons.

Frames are analyzed with `STRATEGY_KEEP_ONLY_LATEST`: only the most recent frame is
ever processed, and frames are dropped rather than queued. The preview keeps rendering
at full rate regardless of how slow inference is.

The model input is square, so each frame is rotated, center-cropped and scaled to the
model input in a single matrix draw into a reusable bitmap, which is what keeps a
steady-state frame free of allocation. The overlay is laid out as a centered square
scaled by the preview's cover factor, which is what keeps the skeleton aligned with
what you see.

A joint below the keypoint threshold is not drawn, and a bone is drawn only when both
of its ends clear it — an occluded wrist reported at low confidence would otherwise
drag a limb across the display. There is no temporal smoothing: what you see is each
frame's raw estimate.

### Delegates

On the first frame the detector times NNAPI against 4-thread CPU and keeps whichever
is faster, logging the result:

```
Using NNAPI delegate (215ms vs 233ms on CPU)
```

Which one wins is a property of the device, the driver and the model rather than
something a fixed threshold can predict, and it does not carry over between models: on
the same T21G, the old detection graph ran about 30% faster on 4-thread XNNPACK, while
this pose graph is about 8% faster on NNAPI. The app measures once at startup — roughly
a second — instead of guessing. If NNAPI cannot be constructed, or fails at run time,
the detector stays on CPU rather than propagating the failure.

## Model

| | |
|---|---|
| Architecture | YOLO26 nano pose, float32 |
| File | `app/src/main/assets/yolo26n_pose_fp32.tflite` (12.0 MB) |
| Input | `[1, 3, 320, 320]` float32 — **NCHW**, channels first |
| Output | `[1, 56, 2100]` float32 |
| Channels | 4 box + 1 person score + 17 x (x, y, confidence) |
| Keypoints | 17 COCO joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles |
| Confidence threshold | 0.5 (person), 0.5 (keypoint, drawing only) |
| IoU threshold (NMS) | 0.45 |

Box and keypoint coordinates come out normalized 0..1, verified by running the exported
model on a test image. The decoder guards the two coordinate spaces independently, so an
export that emits pixel-space values still decodes.

The detector reads input size, layout, tensor types and quantization parameters from the
model at load time, and handles both float32 and int8 tensors and both NCHW and NHWC
input. Ultralytics' LiteRT exporter keeps PyTorch's channels-first layout; the older
TensorFlow converter emitted channels-last. Feeding one layout to a model expecting the
other produces scrambled pixels rather than an error, so the layout is detected from the
input shape and logged.

The bundled export is float32 throughout, so the int8 quantize/dequantize branches never
execute — they exist for exports that use int8 I/O, which is the next lever if frame rate
proves unusable.

### Re-exporting

[scripts/export_pose.py](scripts/export_pose.py) regenerates the asset from
`yolo26n-pose.pt`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ultralytics litert-torch
.venv/bin/python scripts/export_pose.py
```

`end2end=False` is deliberate: YOLO26 defaults to an NMS-free one-to-one head with a
different output layout that this decoder does not read.

Thresholds live in the `companion object` in
[Detector.kt](app/src/main/java/com/crossmedia/objectdetect/Detector.kt).

## Build

Requires JDK 17 and the Android SDK (compileSdk 35).

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`local.properties` is not committed. If Android Studio hasn't created it, point it at
your SDK:

```
sdk.dir=/Users/you/Library/Android/sdk
```

Inference timings are logged per frame:

```bash
adb logcat -s ObjectDetect Detector
```

## Tests

```bash
./gradlew test
```

JVM tests via Robolectric, which supplies real `RectF`/`Rect`/`Matrix` so the pose and
layout geometry runs unmodified off-device. The TensorFlow Lite interpreter cannot: its
natives are Android-only, which is why the pose decoding and NMS live in
[YoloPostProcessor.kt](app/src/main/java/com/crossmedia/objectdetect/YoloPostProcessor.kt)
rather than behind the interpreter, and why the rotate/crop/scale transform is a pure
function (`MainActivity.cropMatrix`) separate from the draw that uses it.

Covered: keypoint decode and the pixel-space coordinate guards, confidence thresholding,
NMS and IoU edge cases, keypoint-to-view mapping, the keypoint visibility gate, skeleton
index integrity, and the crop transform at every camera rotation.

## Running on RealWear

Targets Android 9+ (minSdk 28), which covers the Navigator series. The activity is
locked to landscape.

Voice control comes free from WearHF: it derives commands from visible button labels,
so the on-screen "Pause Tracking" button is spoken as **"Pause Tracking"**, and becomes
"Resume Tracking" once paused. Paused means no inference runs and the overlay clears;
the preview keeps rendering. The command strings are the button labels in
[strings.xml](app/src/main/res/values/strings.xml) — changing a label changes the
command.

Confirmed on a Navigator 500: the model loads as `input 320x320 FLOAT32 NCHW, output
56x2100 FLOAT32`, NNAPI wins the startup benchmark, and steady-state inference is 237 ms
a frame. Still to judge on the headset, with a person in view: whether the skeleton
stays aligned at the device's real preview aspect ratio, and whether 4 FPS jitters
enough to want smoothing.

## Vocabulary

[CONTEXT.md](CONTEXT.md) defines the terms used throughout the code — pose, keypoint,
skeleton, overlay, paused, and so on.
