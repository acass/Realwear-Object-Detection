# Context: RealWear Object Detection

Ubiquitous language for this project. Glossary only — no implementation detail.

## Terms

- **Detection** — one identified object on one frame: a bounding box, a class label, and a confidence score.
- **Live detection** — continuous best-effort inference on the camera preview stream. Only the latest frame is ever processed; frames are dropped, never queued.
- **Overlay** — the boxes and labels drawn over the camera preview representing current detections.
- **Model** — the swappable detector asset. Currently stock COCO-pretrained YOLOv8 nano; a custom-trained model can replace it without changing the app's meaning.
- **Class** — one of the object categories the Model can recognize (currently the 80 COCO classes).
- **Voice command** — a spoken command RealWear's WearHF system derives from a visible control's label (e.g. "Pause Detection"). The app defines controls, not speech handling.
- **Paused** — detection state where no inference runs and the Overlay is empty; the camera preview keeps rendering.
