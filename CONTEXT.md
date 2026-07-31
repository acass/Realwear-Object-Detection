# Context: RealWear Pose Estimation

Ubiquitous language for this project. Glossary only — no implementation detail.

## Terms

- **Pose** — one person's body position on one frame: a set of Keypoints and a confidence score.
- **Keypoint** — one body joint of one Pose: a position and a confidence. A Keypoint below the confidence threshold is treated as not seen rather than as being at its reported position.
- **Skeleton** — the fixed set of bones connecting Keypoints (the 17 standard COCO joints and the 19 bones between them). A property of the Model, not of any one Pose.
- **Live tracking** — continuous best-effort inference on the camera preview stream. Only the latest frame is ever processed; frames are dropped, never queued.
- **Overlay** — the Skeletons drawn over the camera preview representing the current Poses.
- **Model** — the swappable pose asset. Currently stock COCO-pretrained YOLO26 nano pose; a custom-trained model can replace it without changing the app's meaning.
- **Voice command** — a spoken command RealWear's WearHF system derives from a visible control's label (e.g. "Pause Tracking"). The app defines controls, not speech handling.
- **Paused** — tracking state where no inference runs and the Overlay is empty; the camera preview keeps rendering.
