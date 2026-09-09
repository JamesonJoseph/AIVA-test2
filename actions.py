"""Action layer: whitelisted system actions.

Every call goes through ALLOW_LIST and is logged. Anything not on the
list is refused — the model can never launch arbitrary executables.
"""
import json
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("actions")

# name -> argv. Resolved against this machine (GNOME, 2026-09-09):
#   browser=google-chrome, editor=code, calculator=gnome-calculator,
#   files=nautilus. Edit here (not via the model) to change the list.
ALLOW_LIST = {
    "browser": ["google-chrome"],
    "editor": ["code"],
    "calculator": ["gnome-calculator"],
    "files": ["nautilus"],
}


def open_application(name: str) -> str:
    """Launch an allow-listed app (detached). Returns a JSON result string."""
    key = (name or "").strip().lower()
    if key not in ALLOW_LIST:
        log.warning("REFUSED open_application(%r) — not on allow-list", name)
        return json.dumps(
            {
                "status": "refused",
                "reason": f"{name!r} is not on the allow-list",
                "allowed": sorted(ALLOW_LIST),
            }
        )
    cmd = ALLOW_LIST[key]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as e:
        log.error("FAILED open_application(%r): %s", key, e)
        return json.dumps({"status": "error", "app": key, "error": str(e)})
    log.info("LAUNCHED %s -> %s (pid=%s)", key, cmd, proc.pid)
    return json.dumps(
        {"status": "launched", "app": key, "command": cmd, "pid": proc.pid}
    )
