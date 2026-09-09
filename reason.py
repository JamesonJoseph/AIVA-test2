"""Reasoning layer ("the brain"): Groq tool-calling loop.

Phase 4: open_application is REAL (whitelisted in actions.py, every call
logged). control_browser is still a STUB until Phase 5.
"""
import json
import re

from groq import Groq

from actions import open_application as real_open_application
from config import config

SYSTEM_PROMPT = """You watch a webcam via structured perception events.
Each event has "objects" (label, score, bounding box) and a "gesture"
(name, score, handedness or null).

Rules:
- Only call a tool when the event justifies it. A hand gesture (anything
  other than "None") is the trigger for opening an app or acting in the
  browser. Plain observations (e.g. just a person sitting) need NO tool.
- open_application(name): open an allow-listed app. Allowed names ONLY:
  browser, editor, calculator, files. Anything else will be refused.
  Gesture map: Open_Palm -> browser, Victory -> editor,
  Thumb_Up -> files, Pointing_Up -> calculator. Otherwise say what you see.
- control_browser(action, target): navigate/click/type/screenshot in Chrome.
- Otherwise reply with one short observation sentence and no tool call."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Request opening an allow-listed application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "App to open: browser, editor, calculator, files.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_browser",
            "description": "Request a browser action in Chrome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "screenshot"],
                    },
                    "target": {
                        "type": "string",
                        "description": "URL, selector, text, or empty for screenshot.",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


def stub_open_application(name: str) -> str:
    # Kept for reference; Phase 4 wires the real implementation below.
    return json.dumps({"status": "stubbed", "would_open": name})


def stub_control_browser(action: str, target: str = "") -> str:
    return json.dumps(
        {"status": "stubbed", "would_do": action, "target": target}
    )


STUBS = {
    "open_application": real_open_application,  # REAL since Phase 4
    "control_browser": stub_control_browser,  # still a stub until Phase 5
}


def clean(text: str | None) -> str:
    """Strip <think>...</think> reasoning traces from model output."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_client() -> Groq:
    if not config.groq_api_key:
        raise SystemExit("GROQ_API_KEY is empty. Paste your key into .env.")
    return Groq(api_key=config.groq_api_key)


def reason(event: dict) -> dict:
    """Send one perception event to Groq; return thought + stub tool calls."""
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(event)},
    ]
    resp = client.chat.completions.create(
        model=config.groq_model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    msg = resp.choices[0].message
    calls = []
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = STUBS[tc.function.name](**args)
            calls.append(
                {"tool": tc.function.name, "args": args,
                 "result": json.loads(result)}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                }
            )
        final = client.chat.completions.create(
            model=config.groq_model, messages=messages, temperature=0
        )
        thought = clean(final.choices[0].message.content)
    else:
        thought = clean(msg.content)
    return {"thought": thought, "tool_calls": calls}
