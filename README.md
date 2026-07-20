# RealWear Object Detection

Real-time object detection on the camera preview of a RealWear head-mounted device,
using YOLOv8 nano running on-device via TensorFlow Lite. Detections are drawn as
labelled boxes over the live preview, and detection can be paused and resumed by
voice through RealWear's WearHF system.

Status: builds and runs, verified on an Android emulator. Not yet tested on RealWear
hardware — see [Running on RealWear](#running-on-realwear).

## How it works

```
CameraX preview  ──>  PreviewView            (what you see)
       │
       └─ ImageAnalysis ──> center-crop square ──> Detector ──> OverlayView
          (KEEP_ONLY_LATEST)                       (TFLite)     (boxes + labels)
```

- **[MainActivity.kt](app/src/main/java/com/crossmedia/objectdetect/MainActivity.kt)** —
  camera permission, CameraX binding, frame preprocessing, overlay sizing.
- **[Detector.kt](app/src/main/java/com/crossmedia/objectdetect/Detector.kt)** —
  TFLite interpreter, input quantization, YOLOv8 output decoding, non-max suppression.
- **[OverlayView.kt](app/src/main/java/com/crossmedia/objectdetect/OverlayView.kt)** —
  draws the current detections.

Frames are analyzed with `STRATEGY_KEEP_ONLY_LATEST`: only the most recent frame is
ever processed, and frames are dropped rather than queued. The preview keeps rendering
at full rate regardless of how slow inference is.

The model input is square, so each frame is center-cropped before inference. The
overlay is laid out as a centered square scaled by the preview's cover factor, which
is what keeps the boxes aligned with what you see.

### Delegates

The detector tries the NNAPI delegate first and falls back to 4-thread CPU. It also
times the first inference: if NNAPI takes over 1.5s — which happens when the NNAPI
driver is a software fallback, as on an emulator — it rebuilds the interpreter on CPU.

## Model

| | |
|---|---|
| Architecture | YOLOv8 nano, int8-quantized |
| File | `app/src/main/assets/yolov8n_int8.tflite` (3.1 MB) |
| Classes | 80 COCO classes, listed in `app/src/main/assets/labels.txt` |
| Confidence threshold | 0.5 |
| IoU threshold (NMS) | 0.45 |

The detector reads input size, tensor types, and quantization parameters from the
model at load time, and handles both float32 and int8 tensors. Swapping in a
custom-trained YOLOv8 export is a matter of replacing the two asset files — as long
as the label file order matches the model's class indices, no code changes are needed.

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

## Running on RealWear

Targets Android 9+ (minSdk 28), which covers the Navigator series. The activity is
locked to landscape.

Voice control comes free from WearHF: it derives commands from visible button labels,
so the on-screen "Pause Detection" button is spoken as **"Pause Detection"**, and
becomes "Resume Detection" once paused. Paused means no inference runs and the overlay
clears; the preview keeps rendering. The command strings are the button labels in
[strings.xml](app/src/main/res/values/strings.xml) — changing a label changes the
command.

Untested on hardware so far. Two things worth checking on a real device before trusting
it: whether NNAPI on the Navigator's Snapdragon actually beats the CPU path, and
whether the overlay stays aligned at the device's real preview aspect ratio.

## Vocabulary

[CONTEXT.md](CONTEXT.md) defines the terms used throughout the code — detection,
overlay, live detection, paused, and so on.
