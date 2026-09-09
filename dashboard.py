"""Web window: live annotated video + agent reasoning feed.

Responsive design — three threads so slow stages never block video:
  capture   (fast):  reads camera continuously, keeps only the newest frame
  inference (medium): detector + gesture recognizer run IN PARALLEL on the
                     newest frame; annotated JPEG + event published
  reason    (slow):  Groq called on scene changes only, off the video path

Run: ./venv/bin/python dashboard.py  ->  open http://localhost:8000
"""
import asyncio
import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from config import config
from main import event_signature
from perceive import Perceiver, gesture_from, objects_from, to_mp_image
from reason import reason

app = FastAPI(title="AIVA")
state = {
    "frame": "",  # base64 JPEG
    "frame_id": 0,
    "infer_ms": 0,
    "event": {"objects": [], "gesture": {"name": "-", "score": 0.0,
                                         "handedness": None}},
    "sig": None,
    "thought": "starting…",
    "actions": [],
}
lock = threading.Lock()
latest = {"frame": None, "id": 0}


def annotate(bgr, event):
    img = bgr.copy()
    for o in event["objects"]:
        b = o["bbox"]
        p1, p2 = (b["x"], b["y"]), (b["x"] + b["width"], b["y"] + b["height"])
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, f"{o['label']} {o['score']:.2f}",
                    (b["x"], max(0, b["y"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    g = event["gesture"]
    banner = f"GESTURE: {g['name']} ({g['score']:.2f})"
    if g["handedness"]:
        banner += f" {g['handedness']}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, banner, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def capture_loop():
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop stale frames, keep live edge
    if not cap.isOpened():
        with lock:
            state["thought"] = f"Cannot open camera {config.camera_index}"
        return
    while True:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            with lock:
                latest["frame"] = frame
                latest["id"] += 1
        else:
            time.sleep(0.05)


def inference_loop(perceiver, pool):
    last_done = 0
    while True:
        with lock:
            frame, fid = latest["frame"], latest["id"]
        if frame is None or fid == last_done:
            time.sleep(0.01)
            continue
        last_done = fid
        t0 = time.time()
        image = to_mp_image(frame)
        fut_det = pool.submit(perceiver.detector.detect, image)
        fut_rec = pool.submit(perceiver.recognizer.recognize, image)
        event = {"objects": objects_from(fut_det.result()),
                 "gesture": gesture_from(fut_rec.result())}
        infer_ms = int((time.time() - t0) * 1000)
        _, jpg = cv2.imencode(".jpg", annotate(frame, event),
                              [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        with lock:
            state["frame"] = b64
            state["frame_id"] = fid
            state["infer_ms"] = infer_ms
            state["event"] = event
            state["sig"] = event_signature(event)


def reason_loop():
    last_sig = None
    while True:
        with lock:
            sig, event = state["sig"], dict(state["event"])
        if sig is None or sig == last_sig:
            time.sleep(0.2)
            continue
        last_sig = sig
        try:
            out = reason(event)
            with lock:
                state["thought"] = out["thought"]
                state["actions"] = out["tool_calls"][-3:]
        except Exception as e:  # API hiccup: show it, keep video alive
            with lock:
                state["thought"] = f"[reason error] {e}"


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with lock:
                snap = {k: v for k, v in state.items() if k != "sig"}
            await websocket.send_json(snap)
            await asyncio.sleep(0.2)
    except Exception:
        pass


if __name__ == "__main__":
    perceiver = Perceiver()
    pool = ThreadPoolExecutor(max_workers=2)
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, args=(perceiver, pool),
                     daemon=True).start()
    threading.Thread(target=reason_loop, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
