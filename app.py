"""
Web UI for the frame extractor pipeline.
Usage: python app.py   →  open http://localhost:5000
"""

import ipaddress
import json
import queue
import subprocess
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, render_template, request, send_file, abort
from flask import stream_with_context

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
DOWNLOADS = (BASE_DIR / "downloads").resolve()

# task_id → {"queue": Queue, "result": dict | None, "done": bool}
_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# URL sanitisation
# ---------------------------------------------------------------------------

def _sanitise_url(raw: str) -> str:
    """Strip common shell-escape artefacts (backslash before ? = & #)."""
    return raw.strip().replace("\\", "")


def _validate_url(raw: str) -> tuple[str | None, str | None]:
    url = _sanitise_url(raw)
    try:
        p = urlparse(url)
    except Exception:
        return None, "Invalid URL format"
    if p.scheme not in ("http", "https"):
        return None, "Only http:// and https:// URLs are accepted"
    if not p.netloc:
        return None, "URL is missing a host"
    host = p.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return None, "Private or internal addresses are not allowed"
    except ValueError:
        pass  # it's a hostname — fine
    return url, None


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

def _run_pipeline(task_id: str, url: str, phrase: str, use_ocr: bool, force_retranscribe: bool) -> None:
    q: queue.Queue = _tasks[task_id]["queue"]
    cmd = [
        str(BASE_DIR / "venv" / "bin" / "python3"),
        str(BASE_DIR / "pipeline.py"),
        url, phrase,
    ]
    if use_ocr:
        cmd.append("--use-ocr")
    if force_retranscribe:
        cmd.append("--retranscribe")

    result: dict = {"status": "NOT_FOUND", "timestamp": None, "frame_number": None, "frame": None}

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
        )
        for raw_line in (proc.stdout or []):
            line = raw_line.rstrip()
            q.put(line)
            # Parse key output lines to populate result dict
            if line == "FOUND":
                result["status"] = "FOUND"
            elif line.startswith("SPOKEN_NOT_SHOWN"):
                result["status"] = "SPOKEN_NOT_SHOWN"
            elif line.startswith("NOT_FOUND"):
                result["status"] = "NOT_FOUND"
            elif "timestamp" in line and ":" in line:
                try:
                    result["timestamp"] = float(line.split(":", 1)[1].strip().rstrip("s"))
                except ValueError:
                    pass
            elif "frame       :" in line:
                try:
                    result["frame_number"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif "frame_image :" in line:
                result["frame"] = line.split(":", 1)[1].strip()
        proc.wait()
    except Exception as exc:
        q.put(f"[error] {exc}")
        result["status"] = "ERROR"

    _tasks[task_id]["result"] = result
    _tasks[task_id]["done"] = True
    q.put(None)  # sentinel — signals SSE stream to send 'done' event


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True, silent=True) or {}

    url, err = _validate_url(data.get("url", ""))
    if err:
        return jsonify({"error": err}), 400

    phrase = (data.get("phrase") or "").strip()
    if not phrase:
        return jsonify({"error": "Target phrase is required"}), 400
    if len(phrase) > 500:
        return jsonify({"error": "Phrase is too long (max 500 characters)"}), 400

    use_ocr = bool(data.get("use_ocr", False))
    force_retranscribe = bool(data.get("force_retranscribe", False))

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"queue": queue.Queue(), "result": None, "done": False}
    threading.Thread(
        target=_run_pipeline, args=(task_id, url, phrase, use_ocr, force_retranscribe), daemon=True
    ).start()
    return jsonify({"task_id": task_id})


@app.route("/stream/<task_id>")
def stream(task_id):
    if task_id not in _tasks:
        return Response("data: Task not found\n\n", mimetype="text/event-stream")

    def generate():
        q = _tasks[task_id]["queue"]
        while True:
            line = q.get()
            if line is None:
                result = _tasks[task_id].get("result") or {}
                yield f"event: done\ndata: {json.dumps(result)}\n\n"
                return
            # Escape any embedded newlines so SSE framing stays intact
            safe = line.replace("\n", " ")
            yield f"data: {safe}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/frame/<task_id>")
def frame(task_id):
    task = _tasks.get(task_id)
    task_result: dict = (task or {}).get("result") or {}
    if not task or not task_result:
        abort(404)
    res = task_result
    img_path_str: str = res.get("frame") or ""
    if not img_path_str:
        abort(404)
    img_path = Path(img_path_str).resolve()
    # Path-traversal guard — only serve files inside downloads/
    if not str(img_path).startswith(str(DOWNLOADS)):
        abort(403)
    if not img_path.exists():
        abort(404)
    return send_file(img_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
