"""Phase 3: webcam -> MediaPipe perception -> Groq reasoning (stub tools)."""
import json
import time

import cv2

from config import config
from perceive import Perceiver
from reason import reason


def event_signature(event: dict) -> str:
    labels = sorted(o["label"] for o in event["objects"])
    return f"{labels}|{event['gesture']['name']}"


def main() -> None:
    perceiver = Perceiver()
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {config.camera_index}")
    period = 1.0 / config.fps
    print(f"Streaming perception events at ~{config.fps} fps. Ctrl+C to stop.")
    last_sig = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            event = perceiver.perceive(frame)
            print(json.dumps(event), flush=True)
            sig = event_signature(event)
            if sig == last_sig:
                print("(unchanged scene — skipping Groq call)", flush=True)
            else:
                last_sig = sig
                try:
                    out = reason(event)
                except Exception as e:  # keep the loop alive on API errors
                    print(f"[reason error] {type(e).__name__}: {e}", flush=True)
                    time.sleep(period)
                    continue
                print(f"THOUGHT: {out['thought']}", flush=True)
                for c in out["tool_calls"]:
                    print(f"STUB CALL: {c['tool']}{c['args']} -> {c['result']}",
                          flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
