# RealWear Object Detection

Real-time object detection with per-object distance on the camera preview of a RealWear
head-mounted device. YOLOv8 nano detects, YOLO26 nano estimates monocular depth, and each
callout reads the range to what it labels: `PERSON 2.3M`. Detection can be paused and
resumed by voice through RealWear's WearHF system.

Status: verified on RealWear **Navigator 500** (model T21G, Snapdragon 662 / SM6115),
Android 13 / arm64-v8a. Detection alone runs at roughly 6 fps; detection paired with depth
runs at roughly 1.8 fps.

**Distances are not yet calibrated.** `DepthSampler` reports what the model says, and the
model's baked scale was fit at 768 px on Ultralytics' own data, not at 320 px through this
camera. See [Calibration](#calibration).

## How it works

```
CameraX preview  ──>  PreviewView                      (what you see)
       │
       └─ ImageAnalysis ──> center-crop 320x320 ──┬──> Detector    ──┐
          (KEEP_ONLY_LATEST)   (one bitmap)       │    ~120ms        │
                                                  └──> DepthEstimator┤
                                                       ~430ms        │
                                                                     v
                                                            DepthSampler
                                                       (median of box centre)
                                                                     │
                                                                     v
                                                              OverlayView
                                                           (PERSON 2.3M)
```

Both models run on the **same** 320x320 bitmap, in sequence, on the one analysis thread.
That is deliberate: box coordinates come out normalized against exactly that crop, so
indexing the depth map is a plain multiply with no rescaling and no coordinate bugs. It
also means position and distance always come from the same instant -- important on a
head-mounted display, where 700ms of head yaw is a different scene entirely.

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

### Delegates

`DelegateRace` times CPU, GPU and NNAPI on the first frame and keeps the fastest, per
model. Measured on the Navigator 500, best-of-3, inference only:

| model | CPU 4t | GPU | NNAPI | winner |
|---|---|---|---|---|
| yolov8n detect @320 | 141 ms | **90 ms** | 196 ms | GPU, 1.6x |
| yolo26n-depth @320 | 984 ms | **362 ms** | 964 ms | GPU, 2.7x |

The GPU delegate is what makes depth viable at all -- on CPU it costs ~950 ms per frame.
NNAPI loses on both because this device ships no NNAPI vendor driver: `lshal` lists no
`neuralnetworks` HAL and `/vendor/lib64/` holds no NNAPI vendor libraries, so it falls
back to a CPU reference path. `/vendor/lib64/libOpenCL.so` is present, which is why the
Adreno 610 path works.

Measuring rather than guessing matters here: the same race picks GPU for both models on
this device, but which delegate wins is a property of the device, the driver and the
model together, and it changes when any of them does.

## Calibration

The depth head outputs `exp(clamp(logit, -4, 5))` metres, then applies a baked log-affine
calibration -- in `yolo26n-depth.pt` that is `cal_a = 1.0`, `cal_b = -0.19385`, a x0.8238
global scale. Ultralytics fit those at imgsz 768 on their pretraining validation mix.

This app runs the graph at 320 through a camera whose field of view RealWear does not
publish, so that scale does not transfer. `DepthSampler.SCALE_CORRECTION` exists to absorb
the difference and is **currently 1.0, i.e. uncorrected**.

To calibrate: place a target at 1 m, 2 m and 5 m, read the logged distances, and fit one
multiplier. If a single multiplier fits all three the residual is pure scale, which is the
expected failure mode. If it does not, running at 320 has distorted the depth structure
itself and no scalar will fix it -- that finding would reopen the input-resolution choice.

## Licence note

The Ultralytics models here (`yolov8n_int8.tflite`, `yolo26n-depth.tflite`) are released
under AGPL-3.0. Ultralytics lists commercial products, proprietary software and embedded
edge deployments as requiring their Enterprise licence. This is unresolved.

## Vocabulary

[CONTEXT.md](CONTEXT.md) defines the terms used throughout the code — detection,
overlay, live detection, paused, and so on.
