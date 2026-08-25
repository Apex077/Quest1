"""
Visual scanner: coarse stride → frame-diff filter → OCR → fuzzy match.
Once a match is found at time T, does a fine scan backwards from T to find
the actual first frame.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

FUZZY_THRESHOLD = 70
DIFF_THRESHOLD = 4.0      # mean-abs pixel diff below this → frame unchanged, skip OCR
COARSE_STRIDE_S = 2.0     # background scan interval — subtitles show ≥2s, nothing missed
PRIORITY_STRIDE_S = 0.5   # priority-window scan interval (seconds)
FINE_STRIDE_S = 0.1       # fine scan interval once we have a rough match
TEXT_EDGE_THRESHOLD = 0.04 # min fraction of edge pixels to bother with OCR


@dataclass
class ScanResult:
    status: str                          # FOUND | NOT_FOUND | SPOKEN_NOT_SHOWN
    timestamp: float | None = None
    frame_number: int | None = None
    ocr_text: str | None = None
    match_score: float | None = None
    frame_image: np.ndarray | None = None
    coverage: float = 1.0               # fraction of video actually scanned


def _crop(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def _subtitle_strip(frame: np.ndarray, roi: tuple | None) -> np.ndarray:
    """Return the subtitle candidate region: ROI if known, else bottom 25% of frame."""
    if roi is not None:
        return _crop(frame, roi)
    h = frame.shape[0]
    return frame[int(h * 0.75):, :]


def _strip_changed(prev: np.ndarray | None, curr: np.ndarray, roi: tuple | None) -> bool:
    """Diff only the subtitle strip — ignores scene motion, sensitive to subtitle changes."""
    if prev is None:
        return True
    a = _subtitle_strip(prev, roi).astype(np.float32)
    b = _subtitle_strip(curr, roi).astype(np.float32)
    return float(np.mean(np.abs(b - a))) > DIFF_THRESHOLD


def _has_text_hint(frame: np.ndarray, roi: tuple | None) -> bool:
    """
    ~2ms OpenCV pre-check: does the subtitle region contain text-like edges?
    Skips EasyOCR (1-3s) on blank or uniform regions.
    Uses Canny edge density — text has dense, short horizontal edges.
    """
    strip = _subtitle_strip(frame, roi)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = edges.sum() / (255 * edges.size)
    return float(edge_frac) >= TEXT_EDGE_THRESHOLD


def _ocr(reader, frame: np.ndarray, roi: tuple | None) -> str:
    if roi is not None:
        return " ".join(reader.readtext(_crop(frame, roi), detail=0))
    # No learned ROI: stack top-25% and bottom-25% strips into one image.
    # Single OCR call on ~50% of pixels — 2-3× faster than full-frame,
    # no accuracy loss since subtitles don't appear in the middle of the screen.
    h = frame.shape[0]
    combined = np.vstack([frame[:int(h * 0.25)], frame[int(h * 0.75):]])
    return " ".join(reader.readtext(combined, detail=0))


def _check_frame(
    cap: cv2.VideoCapture,
    reader,
    target: str,
    roi: tuple | None,
    fps: float,
    t: float,
    prev_frame: np.ndarray | None,
) -> tuple[ScanResult | None, np.ndarray | None]:
    """Seek to t, OCR if strip changed and text hint passes. Returns (match_or_None, frame_read)."""
    frame_no = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
    ok, frame = cap.read()
    if not ok:
        return None, None
    if not _strip_changed(prev_frame, frame, roi):  # ~1ms
        return None, frame
    if not _has_text_hint(frame, roi):              # ~2ms, skips EasyOCR on blank regions
        return None, frame
    text = _ocr(reader, frame, roi)
    score = fuzz.partial_ratio(target.lower(), text.lower())
    if score >= FUZZY_THRESHOLD:
        return ScanResult(
            status="FOUND",
            timestamp=t,
            frame_number=frame_no,
            ocr_text=text,
            match_score=float(score),
            frame_image=frame.copy(),
        ), frame
    return None, frame


def _scan_range(
    cap: cv2.VideoCapture,
    reader,
    target: str,
    roi: tuple | None,
    fps: float,
    t_start: float,
    t_end: float,
    stride: float,
    stop_before: float | None = None,
) -> ScanResult | None:
    """Scan [t_start, t_end] at stride. Returns earliest match or None."""
    best: ScanResult | None = None
    prev_frame: np.ndarray | None = None
    t = t_start
    while t <= t_end:
        if stop_before is not None and t >= stop_before:
            break
        match, prev_frame = _check_frame(cap, reader, target, roi, fps, t, prev_frame)
        if match:
            best = match
            stop_before = match.timestamp  # only look earlier from here
        t += stride
    return best


def _fine_scan(
    cap: cv2.VideoCapture,
    reader,
    target: str,
    roi: tuple | None,
    fps: float,
    rough_t: float,
) -> ScanResult:
    """
    Scan backwards from rough_t in FINE_STRIDE_S steps to find the actual
    first frame. Returns the earliest match found.
    """
    lookback = max(0.0, rough_t - 3.0)
    result = _scan_range(cap, reader, target, roi, fps, lookback, rough_t, FINE_STRIDE_S, rough_t)
    if result is not None and result.timestamp is not None and result.timestamp < rough_t:
        return result
    # rough_t itself was confirmed; re-fetch that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(rough_t * fps)))
    _, frame = cap.read()
    text = _ocr(reader, frame, roi) if frame is not None else ""
    return ScanResult(
        status="FOUND",
        timestamp=rough_t,
        frame_number=int(rough_t * fps),
        ocr_text=text,
        match_score=fuzz.partial_ratio(target.lower(), text.lower()),
        frame_image=frame.copy() if frame is not None else None,
    )


def find_first(
    video_path: Path,
    target: str,
    roi: tuple[int, int, int, int] | None,
    priority_windows: list[tuple[float, float]],
    reader,
    audio_had_hit: bool = False,
) -> ScanResult:
    """
    Full visual scan per CLAUDE.md:
    1. Priority windows first (subtitle/audio hints), PRIORITY_STRIDE_S.
    2. Background: rest of video at COARSE_STRIDE_S.
    3. Fine scan around any rough match to find exact first frame.
    Visual scan always covers the whole video.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    rough_match: ScanResult | None = None
    stop_before: float | None = None

    # --- 1. Priority windows ---
    covered: list[tuple[float, float]] = []
    for ws, we in sorted(priority_windows):
        t_end = min(we + 2.0, stop_before or duration)
        m = _scan_range(cap, reader, target, roi, fps, ws, t_end,
                        PRIORITY_STRIDE_S, stop_before)
        covered.append((ws, t_end))
        if m is not None and m.timestamp is not None:
            if rough_match is None or rough_match.timestamp is None or m.timestamp < rough_match.timestamp:
                rough_match = m
                stop_before = m.timestamp

    # --- 2. Background scan (uncovered segments) ---
    covered.sort()
    cursor = 0.0
    uncovered: list[tuple[float, float]] = []
    for cs, ce in covered:
        if cursor < cs:
            uncovered.append((cursor, cs))
        cursor = max(cursor, ce)
    end = stop_before or duration
    if cursor < end:
        uncovered.append((cursor, end))

    for seg_s, seg_e in uncovered:
        m = _scan_range(cap, reader, target, roi, fps, seg_s, seg_e,
                        COARSE_STRIDE_S, stop_before)
        if m is not None and m.timestamp is not None:
            if rough_match is None or rough_match.timestamp is None or m.timestamp < rough_match.timestamp:
                rough_match = m
                stop_before = m.timestamp

    # --- 3. Fine scan to find exact first frame ---
    if rough_match is not None and rough_match.timestamp is not None:
        result = _fine_scan(cap, reader, target, roi, fps, rough_match.timestamp)
        cap.release()
        return result

    cap.release()

    if audio_had_hit:
        return ScanResult(status="SPOKEN_NOT_SHOWN", coverage=1.0)
    return ScanResult(status="NOT_FOUND", coverage=1.0)
