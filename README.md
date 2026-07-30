# RealWear Object Detection

Real-time instance segmentation on the camera preview of a RealWear head-mounted
device, using YOLOv8 nano segmentation running on-device via TensorFlow Lite. Each
detected object is drawn as a translucent per-pixel mask tinted by its class, with a
labelled callout over it, and detection can be paused and resumed by voice through
RealWear's WearHF system.

It also ships a hands-free [5-step guided procedure](#guided-procedure) for research
demos: each step names one object to find, the overlay locks onto that object alone,
and the run ends with a green/red summary of what you found and what the model saw.

Status: verified on RealWear T21G (Navigator 520) hardware, Android 13 / arm64-v8a,
at roughly 4 frames per second (226-238 ms per frame, measured).

## How it works

```
CameraX preview  ──>  PreviewView            (what you see)
       │
       └─ ImageAnalysis ──> center-crop square ──> Detector ──> OverlayView
          (KEEP_ONLY_LATEST)                       (TFLite)     (masks + callouts)
```

- **[MainActivity.kt](app/src/main/java/com/crossmedia/objectdetect/MainActivity.kt)** —
  camera permission, CameraX binding, frame preprocessing, overlay sizing.
- **[Detector.kt](app/src/main/java/com/crossmedia/objectdetect/Detector.kt)** —
  TFLite interpreter, frame preprocessing, and mask synthesis for the detections that
  survive.
- **[YoloPostProcessor.kt](app/src/main/java/com/crossmedia/objectdetect/YoloPostProcessor.kt)** —
  YOLOv8 output decoding, non-max suppression, target filtering and the top-N cap.
- **[MaskDecoder.kt](app/src/main/java/com/crossmedia/objectdetect/MaskDecoder.kt)** —
  turns 32 mask coefficients plus the prototype tensor into one cropped instance mask.
- **[OverlayView.kt](app/src/main/java/com/crossmedia/objectdetect/OverlayView.kt)** —
  draws the masks, then the callouts on top.
- **[ProcedureState.kt](app/src/main/java/com/crossmedia/objectdetect/ProcedureState.kt)** —
  the guided procedure's step machine: which object is being looked for, what the
  operator said about it, what the detector saw.

Frames are analyzed with `STRATEGY_KEEP_ONLY_LATEST`: only the most recent frame is
ever processed, and frames are dropped rather than queued. The preview keeps rendering
at full rate regardless of how slow inference is.

The model input is square, so each frame is rotated, center-cropped and scaled to the
model input in a single matrix draw into a reusable bitmap, which is what keeps a
steady-state frame free of allocation. The overlay is laid out as a centered square
scaled by the preview's cover factor, which is what keeps the masks aligned with what
you see.

### Delegates

On the first frame the detector times NNAPI against 4-thread CPU and keeps whichever
is faster, logging the result:

```
Using 4-thread CPU (211ms vs 211ms on NNAPI)
```

Which one wins is a property of the device, the driver and the model rather than
something a fixed threshold can predict. On a T21G the two are level on the
segmentation graph — they were 133ms vs 194ms in favour of the CPU on the old
detection-only model — so the app measures once at startup, roughly a second, instead
of guessing. If NNAPI cannot be constructed, or fails at run time, the detector stays
on CPU rather than propagating the failure.

## Model

| | |
|---|---|
| Architecture | YOLOv8 nano segmentation, float32 |
| File | `app/src/main/assets/yolov8n_seg.tflite` (13.2 MB) |
| Input | `[1, 3, 256, 256]` float32, **channels-first** |
| Predictions | `[1, 116, 1344]` float32 — 4 box + 80 class + 32 mask coefficients |
| Prototypes | `[1, 32, 64, 64]` float32, **channels-first** |
| Classes | 80 COCO classes, listed in `app/src/main/assets/labels.txt` |
| Confidence threshold | 0.5 |
| IoU threshold (NMS) | 0.45 |
| Mask threshold | 0.5 |

Tensors are **channels-first**. Ultralytics builds TFLite through LiteRT from PyTorch
now, so exports come out NCHW rather than the NHWC that older TFLite exports used.
This caught us out once; the detector asserts the layout at load rather than trusting
it.

The detector reads input size and every tensor shape from the model at load time, and
identifies the two outputs by *rank* — predictions are rank 3, prototypes rank 4 —
rather than by index, since export ordering is not a contract. Swapping in a
custom-trained YOLOv8-seg export is a matter of replacing the two asset files, at any
input size: 320x320 and 256x256 both run unmodified. If the label count does not match
the model's class count, or the model has no prototype tensor, the detector **throws at
load**. A mismatch there does not fail visibly — it produces mislabelled detections and
masks built from the wrong channels — so it is not survivable.

### Why float32, and not a quantized export

Both quantized options were exported and measured against the float32 baseline on one
COCO image:

| Export | Result | Size |
|---|---|---|
| float32 | bowl 0.850, broccoli 0.847 | 13.2 MB |
| `w8a32` (int8 weights, float32 activations) | bowl 0.840, broccoli 0.838 | 3.5 MB |
| full int8 | bowl 0.505, bowl 0.505 (class lost) | 3.6 MB |

Full int8 is unusable: confidences collapse to just above the 0.5 threshold, so real
detections disappear rather than merely scoring lower. `w8a32` keeps the accuracy at a
quarter of the size and would be the obvious choice, but it does not load on TFLite
2.16.1:

```
transpose_conv.cc:312 weights->type != input->type (INT8 != FLOAT32)
Node number 285 (TRANSPOSE_CONV) failed to prepare.
```

The prototype branch upsamples through `TRANSPOSE_CONV`, whose kernel in this runtime
requires weights and input to share a type. Revisit `w8a32` — and the 10 MB it saves —
if the TensorFlow Lite dependency is ever bumped.

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
and the mask synthesis in
[MaskDecoder.kt](app/src/main/java/com/crossmedia/objectdetect/MaskDecoder.kt)
rather than behind the interpreter, and why the rotate/crop/scale transform is a pure
function (`MainActivity.cropMatrix`) separate from the draw that uses it. The mask
index arithmetic in particular is not something to debug through a camera preview at
4 fps.

`ProcedureState` needs no Robolectric at all — it has no Android imports, which is the
point of splitting it out of the activity.

Covered: box decode and the pixel-space coordinate guard, confidence thresholding,
NMS and IoU edge cases, mask synthesis (the coefficient/prototype dot product, the
box crop, threshold boundaries, and off-frame and inverted boxes), target filtering
and the top-N cap, per-class mask colouring, callout placement and quadrant fallback,
the crop transform at every camera rotation, and the procedure's step arithmetic —
advance, undo, the summary transition, and peak-confidence accumulation.

Two of these exist because the change that added masks could have broken them
silently: a segmentation model's trailing 32 mask coefficients must not be mistaken
for class scores by the argmax, and the guided procedure's target must survive the
top-N cap even when it is not among the strongest detections in frame.

## Guided procedure

A five-step demo script, driven entirely by voice. Say **"Start Procedure"** and the
app walks you through five COCO objects one at a time:

| Step | COCO class |
|---|---|
| 1 | `cup` |
| 2 | `keyboard` |
| 3 | `laptop` |
| 4 | `bottle` |
| 5 | `cell phone` |

COCO has no "coffee cup" — the class is `cup`. The five are picked for hit rate at
256x256: no `mouse` or `scissors` (too small to clear a 0.5 threshold), no `book` (weak
class), no `person` (always in frame). The list is `ProcedureState.DEFAULT_TARGETS`.

Each step shows a banner, speaks the prompt over TTS ("Step one. Look for the cup."),
and filters the overlay so **only** the target object is drawn. Three commands are live:

| Command | Effect |
|---|---|
| **"Good"** | marks the step found, advances |
| **"Not Found"** | marks the step missing, advances |
| **"Go Back"** | returns to the previous step and clears its verdict and measurement |

Answering step 5 ends the run at a summary: one row per object, the operator's verdict
in green or red, and beside it the strongest confidence the detector ever reached for
that object — or "model never" if it never cleared the threshold. Operator and model
are recorded separately on purpose; the disagreements are the interesting result.
Inference is frozen on the summary. **"Restart Procedure"** runs it again,
**"Exit Procedure"** returns to free-running detection.

Note that at ~4 fps a fast pan can cross a target between frames, so "model never" can
mean "the model never got a frame of it" rather than "the model failed". That is a
property of the system, not a bug.

## Running on RealWear

Targets Android 9+ (minSdk 28), which covers the Navigator series. The activity is
locked to landscape.

Voice control comes free from WearHF: it derives commands from visible button labels,
so the on-screen "Pause Detection" button is spoken as **"Pause Detection"**, and
becomes "Resume Detection" once paused. Paused means no inference runs and the overlay
clears; the preview keeps rendering. The command strings are the button labels in
[strings.xml](app/src/main/res/values/strings.xml) — changing a label changes the
command.

That mechanism is also why the button bar's visibility is the procedure's grammar:
only visible buttons become commands, so the Pause button is hidden during a run and
"Good" / "Not Found" / "Go Back" exist only while a step is open. Visibility is set in
one place, `MainActivity.applyMode()`.

Verified on hardware for free-running detection: NNAPI loses to 4-thread XNNPACK on the
T21G, and the overlay stays aligned at the device's real preview aspect ratio.

The guided procedure is verified on a T21G too. WearHF picks up all three step commands
from the button labels, logging the grammar it built:

```
... |Select Item 1|GOOD|Select Item 2|NOT FOUND|Select Item 3|GO BACK
```

"Good" being a single syllable turned out not to be a problem in practice, and WearHF
adds "Select Item 1/2/3" as positional alternates for every button anyway, so there is
always a fallback if a phrase is misheard.

### TTS needs a `<queries>` declaration

Android 11+ package visibility hides other packages by default. Without an explicit
`<queries>` entry the platform filters out the TTS engine and `TextToSpeech` init fails
with `ERROR`, which on a T21G looks like this:

```
AppsFilter: com.crossmedia.objectdetect -> com.realwear.ttsservice BLOCKED
W ObjectDetect: TextToSpeech unavailable (-1), prompts are on-screen only
```

The fix is in [AndroidManifest.xml](app/src/main/AndroidManifest.xml) — a `<queries>`
block for `android.intent.action.TTS_SERVICE`, which is what
`com.realwear.ttsservice/.androidtts.RealWearTextToSpeechService` registers. With it the
log reads `Connected successfully to TTS engine: com.realwear.ttsservice`.

Speech failure is non-fatal by design: the banner still shows every prompt, so a device
with no TTS engine degrades to a silent but fully usable procedure.

Still unverified: whether TTS output through the headset speaker can self-trigger
WearHF's ASR. If a prompt ever fires its own command, all speech goes through
`MainActivity.speak()` and can be no-op'd in that one place.

## Vocabulary

[CONTEXT.md](CONTEXT.md) defines the terms used throughout the code — detection,
overlay, live detection, paused, and so on.
