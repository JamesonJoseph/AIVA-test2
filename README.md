# AIVA — Local Vision Agent

A Python agent that **watches your webcam, reasons about what it sees with
Groq, and acts on your system** through a strict whitelist. Gestures trigger
actions: show a thumbs-up, get your files; open palm, get your browser.

```
webcam → MediaPipe perception → Groq reasoning → whitelisted action
  1.5 fps    objects + gestures     tool calls        apps (logged)
```

## Features

- **Perception (MediaPipe Tasks, on CPU)** — EfficientDet-Lite0 object
  detection plus hand-gesture recognition, fused into one structured event
  per frame: `{objects: [{label, score, bbox}], gesture: {name, score,
  handedness}}`.
- **Reasoning (Groq)** — `qwen/qwen3.6-27b` with tool calling decides what
  to do. Unchanged scenes skip the API call to save cost.
- **Actions (whitelisted + logged)** — the model can only invoke functions
  in `actions.py`. Anything else is refused. Every launch is logged.
- **No raw shell** — model output never reaches a shell; only fixed
  commands from the allow-list ever execute.

## Gesture map

| Gesture    | Action             |
| ---------- | ------------------ |
| Open_Palm  | Open browser       |
| Victory    | Open editor        |
| Thumb_Up   | Open file manager  |
| Pointing_Up| Open calculator    |
| None       | Observe, do nothing|

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste your key: GROQ_API_KEY=...  (get one at console.groq.com/keys)

./venv/bin/python main.py
```

Models (~10 MB) auto-download to `models/` on first run. Aim a gesture at
the camera and watch the `THOUGHT:` / `ACTION:` lines.

## Configuration (`.env`)

| Variable          | Default         | Meaning                              |
| ----------------- | --------------- | ------------------------------------ |
| `GROQ_API_KEY`    | — (required)    | Groq API key                         |
| `GROQ_MODEL`      | `qwen/qwen3.6-27b` | Reasoning/vision model            |
| `CAMERA_INDEX`    | `0`             | `cv2.VideoCapture` device            |
| `FPS`             | `1.5`           | Agent loop rate (not video streaming)|
| `SCORE_THRESHOLD` | `0.5`           | Min detection confidence             |
| `MAX_RESULTS`     | `5`             | Max objects per frame                |
| `NUM_HANDS`       | `2`             | Max hands for gesture recognizer     |
| `MODELS_DIR`      | `models`        | Where `.tflite`/`.task` files live   |

## Project structure

```
config.py       # settings from .env (no logic)
perceive.py     # MediaPipe detector + gesture wrapper → event per frame
reason.py       # Groq tool-calling loop (thought + tool calls)
actions.py      # whitelisted open_application + logging
main.py         # capture → perceive → reason loop
requirements.txt
.env.example
```

Example console output:

```json
{"objects": [{"label": "person", "score": 0.81, "bbox": {"x": 83, "y": 213, "width": 501, "height": 266}}], "gesture": {"name": "None", "score": 0.0, "handedness": null}}
THOUGHT: A person is visible in the frame.
```

## Safety

- `.env` (your key) is git-ignored and never committed.
- `actions.py:ALLOW_LIST` is the only place that maps names to commands —
  edit it by hand, never via the model.
- Auto-launch mode logs every call (`LAUNCHED …` / `REFUSED …`).

## Roadmap

- [x] Phase 0 — API research
- [x] Phase 1 — venv + requirements + config
- [x] Phase 2 — capture + MediaPipe perception loop
- [x] Phase 3 — Groq reasoning with stub tools
- [x] Phase 4 — real whitelisted app launching
- [ ] Phase 5 — Chrome control via Playwright CDP
- [ ] Phase 6 — FastAPI/WebSocket live dashboard

## Requirements

Linux, Python 3.9+ (tested on 3.14), webcam, Groq API key.
