"""Export yolo26n-pose.pt to the TFLite asset the Android app loads.

end2end=False keeps YOLO26's classic one-to-many head, so the output is the
familiar [1, 56, N] layout: cx, cy, w, h, person score, then 17 x (x, y, conf).
The NMS-free head would emit a fixed 300 candidates in a different layout that
YoloPostProcessor does not decode.

imgsz=320 rather than the stock 640: on a RealWear T21G, 640 is a slideshow.
Float32 rather than quantized: int8 needs a calibration set, and keypoint
regression is more quantization-sensitive than box regression.

Run with the project venv:  .venv/bin/python scripts/export_pose.py
"""

from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "yolo26n-pose.pt"
DEST = ROOT / "app/src/main/assets/yolo26n_pose_fp32.tflite"

IMGSZ = 320


def main() -> None:
    exported = Path(YOLO(SOURCE).export(format="tflite", imgsz=IMGSZ, end2end=False))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, DEST)
    print(f"{DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
