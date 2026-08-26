"""Tests for scan.py — pre-filters, OCR path, extract_frame_at."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scan import (
    ScanResult,
    _has_text_hint,
    _ocr,
    _strip_changed,
    _subtitle_strip,
    extract_frame_at,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

def white(h=480, w=640):
    return np.full((h, w, 3), 255, dtype=np.uint8)

def mock_cap(fps=25.0, frame_count=1000, frame=None, read_ok=True):
    cap = MagicMock()
    # cv2.CAP_PROP_FPS = 5, cv2.CAP_PROP_FRAME_COUNT = 7
    cap.get.side_effect = lambda p: {5: fps, 7: float(frame_count)}.get(p, 0.0)
    cap.read.return_value = (read_ok, frame if frame is not None else blank())
    return cap


# ── _subtitle_strip ──────────────────────────────────────────────────────────

def test_subtitle_strip_no_roi_returns_bottom_30pct():
    f = blank(h=100)
    strip = _subtitle_strip(f, None)
    assert strip.shape[0] == 30  # 100 * 0.30

def test_subtitle_strip_with_roi_crops_correctly():
    f = white(h=100, w=200)
    roi = (10, 20, 190, 80)
    strip = _subtitle_strip(f, roi)
    assert strip.shape == (60, 180, 3)


# ── _strip_changed ────────────────────────────────────────────────────────────

def test_strip_changed_none_prev_always_true():
    assert _strip_changed(None, blank(), None) is True

def test_strip_changed_identical_frames_false():
    f = blank()
    assert _strip_changed(f, f.copy(), None) is False

def test_strip_changed_different_frames_true():
    assert _strip_changed(blank(), white(), None) is True

def test_strip_changed_only_diffs_bottom_strip():
    # Top half changes (scene motion) — bottom strip identical → should NOT be changed
    prev = blank()
    curr = blank()
    curr[:240, :] = 255  # only top half differs
    # Bottom 30% is rows 336–480, still black → diff should be 0
    assert _strip_changed(prev, curr, None) is False


# ── _has_text_hint ────────────────────────────────────────────────────────────

def test_has_text_hint_blank_frame_false():
    assert _has_text_hint(blank(), None) is False

def test_has_text_hint_uniform_frame_false():
    assert _has_text_hint(white(), None) is False

def test_has_text_hint_edge_rich_bottom_strip_true():
    f = blank(h=100)
    # Alternating 5px bands — coarse enough to survive Canny's Gaussian blur
    for start in range(70, 100, 10):
        f[start:start + 5, :] = 255
    assert _has_text_hint(f, None) is True

def test_has_text_hint_edges_in_roi():
    f = blank(h=100, w=200)
    roi = (0, 70, 200, 100)
    for start in range(70, 100, 10):
        f[start:start + 5, :] = 255
    assert _has_text_hint(f, roi) is True


# ── _ocr ─────────────────────────────────────────────────────────────────────

def test_ocr_filters_low_confidence():
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        (None, "hello", 0.9),
        (None, "noise", 0.1),   # below 0.2 threshold
    ]
    result = _ocr(mock_reader, blank(), None)
    assert "hello" in result
    assert "noise" not in result

def test_ocr_with_roi_crops_first():
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [(None, "text", 0.8)]
    roi = (0, 300, 640, 480)
    _ocr(mock_reader, blank(), roi)
    # readtext called with the cropped region (180px tall), not the full frame
    called_img = mock_reader.readtext.call_args[0][0]
    assert called_img.shape[0] == 180

def test_ocr_no_roi_stacks_top_and_bottom():
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []
    frame = blank(h=100)
    _ocr(mock_reader, frame, None)
    # Combined strip = top 30 rows + bottom 30 rows = 60 rows
    called_img = mock_reader.readtext.call_args[0][0]
    assert called_img.shape[0] == 60


# ── extract_frame_at ─────────────────────────────────────────────────────────

def test_extract_frame_at_found():
    fake_frame = blank()
    cap = mock_cap(fps=25.0, frame=fake_frame)
    with patch("scan.cv2.VideoCapture", return_value=cap):
        result = extract_frame_at(Path("fake.mp4"), 2.0)
    assert result.status == "FOUND"
    assert result.timestamp == pytest.approx(2.0)
    assert result.frame_number == 50       # 2.0 * 25 fps
    assert result.frame_image is not None

def test_extract_frame_at_read_failure():
    cap = mock_cap(read_ok=False, frame=None)
    cap.read.return_value = (False, None)
    with patch("scan.cv2.VideoCapture", return_value=cap):
        result = extract_frame_at(Path("fake.mp4"), 0.0)
    assert result.status == "NOT_FOUND"
    assert result.frame_image is None

def test_extract_frame_at_correct_seek():
    cap = mock_cap(fps=30.0)
    with patch("scan.cv2.VideoCapture", return_value=cap):
        extract_frame_at(Path("fake.mp4"), 10.0)
    # Should seek to frame 300 (10s * 30fps); prop arg is cv2.CAP_PROP_POS_FRAMES = 1
    cap.set.assert_called_with(1, float(300))
