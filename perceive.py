"""Perception layer: MediaPipe Object Detector + Gesture Recognizer.

Produces one structured event per frame:
    {"objects": [{"label", "score", "bbox": {"x","y","width","height"}}],
     "gesture": {"name", "score", "handedness"}}
"""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import config


def ensure_model(filename: str, url: str, models_dir: str) -> str:
    """Download a model file on first run; return its local path."""
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        os.makedirs(models_dir, exist_ok=True)
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    return path


class Perceiver:
    def __init__(self, cfg=config):
        det_path = ensure_model(
            cfg.object_detector_model, cfg.object_detector_url, cfg.models_dir
        )
        ges_path = ensure_model(
            cfg.gesture_recognizer_model,
            cfg.gesture_recognizer_url,
            cfg.models_dir,
        )
        self.detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=det_path),
                score_threshold=cfg.score_threshold,
                max_results=cfg.max_results,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ges_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=cfg.num_hands,
            )
        )

    def perceive(self, bgr_frame) -> dict:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        det = self.detector.detect(image)
        objects = []
        for d in det.detections:
            cat = d.categories[0]
            bb = d.bounding_box
            objects.append(
                {
                    "label": cat.category_name,
                    "score": round(float(cat.score), 3),
                    "bbox": {
                        "x": int(bb.origin_x),
                        "y": int(bb.origin_y),
                        "width": int(bb.width),
                        "height": int(bb.height),
                    },
                }
            )

        rec = self.recognizer.recognize(image)
        gesture = {"name": "None", "score": 0.0, "handedness": None}
        if rec.gestures and rec.gestures[0]:
            top = rec.gestures[0][0]
            handed = None
            if rec.handedness and rec.handedness[0]:
                handed = rec.handedness[0][0].category_name
            gesture = {
                "name": top.category_name,
                "score": round(float(top.score), 3),
                "handedness": handed,
            }

        return {"objects": objects, "gesture": gesture}
