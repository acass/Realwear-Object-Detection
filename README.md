# RealWear Object Detection

Real-time object detection on the camera preview of a RealWear head-mounted device,
using YOLOv8 nano running on-device via TensorFlow Lite. Detections are drawn as
labelled boxes over the live preview, and detection can be paused and resumed by
voice through RealWear's WearHF system.

Status: verified on RealWear T21G (Navigator 520) hardware, Android 13 / arm64-v8a,
at roughly 6 frames per second.

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

The model input is square, so each frame is rotated, center-cropped and scaled to the
model input in a single matrix draw into a reusable bitmap, which is what keeps a
steady-state frame free of allocation. The overlay is laid out as a centered square
scaled by the preview's cover factor, which is what keeps the boxes aligned with what
you see.

### Delegates

On the first frame the detector times NNAPI against 4-thread CPU and keeps whichever
is faster, logging the result:

```
Using 4-thread CPU (133ms vs 194ms on NNAPI)
```

Which one wins is a property of the device, the driver and the model rather than
something a fixed threshold can predict. On a T21G, NNAPI runs this graph correctly
but XNNPACK on the CPU is about 30% faster, so the app measures once at startup —
roughly a second — instead of guessing. If NNAPI cannot be constructed, or fails at
run time, the detector stays on CPU rather than propagating the failure.

## Model

| | |
|---|---|
| Architecture | YOLOv8 nano, int8-quantized weights with float32 input/output |
| File | `app/src/main/assets/yolov8n_int8.tflite` (3.1 MB) |
| Input | 320x320 float32 |
| Output | 84x2100 float32 |
| Classes | 80 COCO classes, listed in `app/src/main/assets/labels.txt` |
| Confidence threshold | 0.5 |
| IoU threshold (NMS) | 0.45 |

The detector reads input size, tensor types, and quantization parameters from the
model at load time, and handles both float32 and int8 tensors. Note that this export
has int8 *weights* but float32 *tensors*, so the int8 input-quantization and
output-dequantization branches never execute with the bundled asset — they exist for
exports that use int8 I/O. Swapping in a custom-trained YOLOv8 export is a matter of
replacing the two asset files; if the label count does not match the model's class
count the detector warns at load time.

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

JVM tests via Robolectric, which supplies real `RectF`/`Rect`/`Matrix` so the
detection and layout geometry runs unmodified off-device. The TensorFlow Lite
interpreter cannot: its natives are Android-only, which is why the box decoding and
NMS live in
[YoloPostProcessor.kt](app/src/main/java/com/crossmedia/objectdetect/YoloPostProcessor.kt)
rather than behind the interpreter, and why the rotate/crop/scale transform is a pure
function (`MainActivity.cropMatrix`) separate from the draw that uses it.

Covered: box decode and the pixel-space coordinate guard, confidence thresholding,
NMS and IoU edge cases, callout placement and quadrant fallback, top-N selection, and
the crop transform at every camera rotation.

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
