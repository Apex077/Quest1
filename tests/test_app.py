"""Tests for app.py — /run validation, /stream SSE, /frame serving."""
import queue
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import app as flask_app_module


# ── /run — URL and phrase validation ─────────────────────────────────────────

class TestRunValidation:
    def test_valid_request_returns_task_id(self, client):
        with patch.object(flask_app_module, "_run_pipeline"):
            resp = client.post("/run", json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "phrase": "never gonna give you up",
            })
        assert resp.status_code == 200
        assert "task_id" in resp.get_json()

    def test_missing_url_rejected(self, client):
        resp = client.post("/run", json={"url": "", "phrase": "hello"})
        assert resp.status_code == 400

    def test_no_url_field_rejected(self, client):
        resp = client.post("/run", json={"phrase": "hello"})
        assert resp.status_code == 400

    def test_ftp_scheme_rejected(self, client):
        resp = client.post("/run", json={"url": "ftp://example.com/v.mp4", "phrase": "hello"})
        assert resp.status_code == 400
        assert "http" in resp.get_json()["error"].lower()

    def test_private_ip_rejected(self, client):
        resp = client.post("/run", json={"url": "http://192.168.1.1/v.mp4", "phrase": "hi"})
        assert resp.status_code == 400

    def test_loopback_rejected(self, client):
        resp = client.post("/run", json={"url": "http://127.0.0.1/v.mp4", "phrase": "hi"})
        assert resp.status_code == 400

    def test_link_local_rejected(self, client):
        resp = client.post("/run", json={"url": "http://169.254.1.1/v.mp4", "phrase": "hi"})
        assert resp.status_code == 400

    def test_empty_phrase_rejected(self, client):
        resp = client.post("/run", json={"url": "https://example.com/v.mp4", "phrase": "   "})
        assert resp.status_code == 400

    def test_phrase_too_long_rejected(self, client):
        resp = client.post("/run", json={"url": "https://example.com/v.mp4", "phrase": "x" * 501})
        assert resp.status_code == 400

    def test_backslash_url_sanitised_and_accepted(self, client):
        # Common shell copy-paste artefact: https://ok.ru/video/123\?v\=abc
        with patch.object(flask_app_module, "_run_pipeline"):
            resp = client.post("/run", json={
                "url": r"https://ok.ru/video/248244667877\?v\=test",
                "phrase": "hello",
            })
        assert resp.status_code == 200

    def test_use_ocr_flag_accepted(self, client):
        with patch.object(flask_app_module, "_run_pipeline"):
            resp = client.post("/run", json={
                "url": "https://example.com/v.mp4",
                "phrase": "hello",
                "use_ocr": True,
            })
        assert resp.status_code == 200

    def test_force_retranscribe_flag_accepted(self, client):
        with patch.object(flask_app_module, "_run_pipeline"):
            resp = client.post("/run", json={
                "url": "https://example.com/v.mp4",
                "phrase": "hello",
                "force_retranscribe": True,
            })
        assert resp.status_code == 200


# ── /stream — SSE output ──────────────────────────────────────────────────────

class TestStream:
    def _inject_task(self, task_id, lines, result):
        q = queue.Queue()
        for line in lines:
            q.put(line)
        q.put(None)  # sentinel → triggers 'done' event
        flask_app_module._tasks[task_id] = {
            "queue": q,
            "result": result,
            "done": True,
        }

    def test_data_lines_present(self, client):
        self._inject_task("t1", ["[ingest] cached", "FOUND"], {
            "status": "FOUND", "timestamp": 5.0, "frame_number": 125, "frame": None,
        })
        text = client.get("/stream/t1").get_data(as_text=True)
        data_lines = [l[6:] for l in text.splitlines() if l.startswith("data:")]
        assert "[ingest] cached" in data_lines
        assert "FOUND" in data_lines

    def test_done_event_present(self, client):
        self._inject_task("t2", [], {"status": "NOT_FOUND"})
        text = client.get("/stream/t2").get_data(as_text=True)
        event_lines = [l for l in text.splitlines() if l.startswith("event:")]
        assert any("done" in l for l in event_lines)

    def test_done_event_data_is_json(self, client):
        import json
        result = {"status": "FOUND", "timestamp": 3.5, "frame_number": 87, "frame": None}
        self._inject_task("t3", [], result)
        text = client.get("/stream/t3").get_data(as_text=True)
        lines = text.splitlines()
        # Find the data line after 'event: done'
        for i, l in enumerate(lines):
            if l.startswith("event:") and "done" in l:
                data_line = next((x for x in lines[i+1:] if x.startswith("data:")), None)
                assert data_line is not None
                payload = json.loads(data_line[6:])
                assert payload["status"] == "FOUND"
                break

    def test_unknown_task_returns_error(self, client):
        text = client.get("/stream/no-such-task").get_data(as_text=True)
        assert "not found" in text.lower()

    def test_newlines_in_log_line_dont_break_sse(self, client):
        # Embedded newlines in a log line must be stripped before yielding
        self._inject_task("t4", ["line with\nnewline"], {"status": "NOT_FOUND"})
        text = client.get("/stream/t4").get_data(as_text=True)
        # Each SSE data: line must be a single line (no bare newline mid-value)
        data_lines = [l for l in text.splitlines() if l.startswith("data:")]
        for dl in data_lines:
            assert "\n" not in dl


# ── /frame — file serving ─────────────────────────────────────────────────────

class TestFrame:
    def test_missing_task_404(self, client):
        assert client.get("/frame/ghost").status_code == 404

    def test_task_without_result_404(self, client):
        flask_app_module._tasks["empty"] = {
            "queue": queue.Queue(), "result": None, "done": False,
        }
        assert client.get("/frame/empty").status_code == 404

    def test_task_without_frame_path_404(self, client):
        flask_app_module._tasks["noframe"] = {
            "queue": queue.Queue(),
            "result": {"status": "NOT_FOUND", "frame": None},
            "done": True,
        }
        assert client.get("/frame/noframe").status_code == 404

    def test_path_traversal_403(self, client, tmp_path):
        # Frame path outside downloads/ must be rejected
        evil = tmp_path / "secret.txt"
        evil.write_text("not a video frame")
        flask_app_module._tasks["traversal"] = {
            "queue": queue.Queue(),
            "result": {"status": "FOUND", "frame": str(evil)},
            "done": True,
        }
        assert client.get("/frame/traversal").status_code == 403

    def test_valid_frame_served(self, client):
        flask_app_module.DOWNLOADS.mkdir(exist_ok=True)
        img = flask_app_module.DOWNLOADS / "_test_frame.jpg"
        # Minimal valid JPEG bytes
        img.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xd9"
        )
        try:
            flask_app_module._tasks["validframe"] = {
                "queue": queue.Queue(),
                "result": {"status": "FOUND", "frame": str(img)},
                "done": True,
            }
            resp = client.get("/frame/validframe")
            assert resp.status_code == 200
            assert resp.content_type == "image/jpeg"
        finally:
            img.unlink(missing_ok=True)
