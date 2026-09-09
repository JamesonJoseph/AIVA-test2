"""Central settings, loaded from environment (.env). No agent logic here."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
    camera_index: int = int(os.environ.get("CAMERA_INDEX", "0"))
    fps: float = float(os.environ.get("FPS", "1.5"))
    score_threshold: float = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
    max_results: int = int(os.environ.get("MAX_RESULTS", "5"))
    num_hands: int = int(os.environ.get("NUM_HANDS", "2"))
    models_dir: str = os.environ.get("MODELS_DIR", "models")

    object_detector_model: str = "efficientdet_lite0.tflite"
    gesture_recognizer_model: str = "gesture_recognizer.task"
    object_detector_url: str = (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
    )
    gesture_recognizer_url: str = (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    )


config = Config()
