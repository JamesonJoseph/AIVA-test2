"""Web window: live annotated video + agent reasoning feed.

Run: ./venv/bin/python dashboard.py  ->  open http://localhost:8000
"""
import base64
import asyncio
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from config import config
from main import event_signature
from perceive import Perceiver
from reason import reason

app = FastAPI(title="AIVA")
state = {
    "frame": "",  # base64 JPEG
    "event": {"objects": [], "gesture": {"name": "-", "score": 0.0,
                                         "handedness": None}},
    "thought": "starting…",
    "actions": [],
}
lock = threading.Lock()


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


def agent_loop():
    perceiver = Perceiver()
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        with lock:
            state["thought"] = f"Cannot open camera {config.camera_index}"
        return
    last_sig = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            time.sleep(0.5)
            continue
        event = perceiver.perceive(frame)
        _, jpg = cv2.imencode(".jpg", annotate(frame, event),
                              [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        sig = event_signature(event)
        with lock:
            state["frame"] = b64
            state["event"] = event
            if sig != last_sig:
                last_sig = sig
                try:
                    out = reason(event)
                    state["thought"] = out["thought"]
                    state["actions"] = out["tool_calls"][-3:]
                except Exception as e:
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
                snap = dict(state)
            await websocket.send_json(snap)
            await asyncio.sleep(0.5)
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=agent_loop, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
