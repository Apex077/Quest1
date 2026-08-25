"""
Visual scanner: coarse stride → frame-diff filter → OCR → fuzzy match.
Once a match is found at time T, does a fine scan backwards from T to find
the actual first frame. The fine scan bypasses pre-filters — it already knows
text is nearby and must not miss the true onset frame.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

FUZZY_THRESHOLD = 70
DIFF_THRESHOLD = 4.0       # mean-abs pixel diff below this → frame unchanged, skip OCR
COARSE_STRIDE_S = 2.0      # background scan interval — subtitles show ≥2s, nothing missed
PRIORITY_STRIDE_S = 0.5    # priority-window scan interval (seconds)
FINE_STRIDE_S = 0.1        # fine scan interval once we have a rough match
TEXT_EDGE_THRESHOLD = 0.04  # min fraction of Canny edge pixels to allow OCR


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
    """ROI if learned, else bottom 30% of frame (matches learn_roi band boundaries)."""
    if roi is not None:
        return _crop(frame, roi)
    h = frame.shape[0]
    return frame[int(h * 0.70):, :]


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
    """
    strip = _subtitle_strip(frame, roi)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = edges.sum() / (255 * edges.size)
    return float(edge_frac) >= TEXT_EDGE_THRESHOLD


_OCR_CONF_THRESHOLD = 0.2  # discard low-confidence EasyOCR detections (noise/fragments)


def _ocr(reader, frame: np.ndarray, roi: tuple | None, debug_t: float | None = None) -> str:
    if roi is not None:
        results = reader.readtext(_crop(frame, roi), detail=1)
    else:
        h = frame.shape[0]
        combined = np.vstack([frame[:int(h * 0.30)], frame[int(h * 0.70):]])
        results = reader.readtext(combined, detail=1)
    if debug_t is not None and results:
        for (_, text, conf) in results:
            print(f"  [raw t={debug_t:.2f}] conf={conf:.2f} {text!r}")
    return " ".join(text for (_, text, conf) in results if conf >= _OCR_CONF_THRESHOLD)


def _check_frame(
    cap: cv2.VideoCapture,
    reader,
    target: str,
    roi: tuple | None,
    fps: float,
    t: float,
    prev_frame: np.ndarray | None,
    stats: dict,
    skip_prefilter: bool = False,
) -> tuple[ScanResult | None, np.ndarray | None]:
    """Seek to t, run pre-checks (unless skip_prefilter), then OCR."""
    frame_no = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
    ok, frame = cap.read()
    if not ok:
        return None, None

    if not skip_prefilter:
        if not _strip_changed(prev_frame, frame, roi):
            stats["diff_skipped"] += 1
            return None, frame
        if not _has_text_hint(frame, roi):
            stats["hint_skipped"] += 1
            return None, frame

    stats["ocr_calls"] += 1
    text = _ocr(reader, frame, roi, debug_t=t if skip_prefilter else None)
    if text.strip():
        score_preview = fuzz.WRatio(target.lower(), text.lower())
        print(f"[ocr t={t:.2f}] score={score_preview} {text!r}")
    # Length guard: a single char like 'Y' scores 100 via partial alignment — skip noise.
    if len(text.strip()) < max(3, len(target) // 3):
        return None, frame
    score = fuzz.WRatio(target.lower(), text.lower())
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
    stats: dict,
    stop_before: float | None = None,
    skip_prefilter: bool = False,
) -> ScanResult | None:
    """Scan [t_start, t_end] at stride. Returns earliest match or None."""
    best: ScanResult | None = None
    prev_frame: np.ndarray | None = None
    t = t_start
    while t <= t_end:
        if stop_before is not None and t >= stop_before:
            break
        match, prev_frame = _check_frame(
            cap, reader, target, roi, fps, t, prev_frame, stats, skip_prefilter
        )
        if match:
            best = match
            stop_before = match.timestamp
        t += stride
    return best


def _fine_scan(
    cap: cv2.VideoCapture,
    reader,
    target: str,
    roi: tuple | None,
    fps: float,
    rough_t: float,
    stats: dict,
) -> ScanResult:
    """
    Scan the window before rough_t at FINE_STRIDE_S to find the actual first frame.
    Pre-filters are DISABLED — we know text is nearby and cannot afford to miss onset.
    Lookback is stride-aware: covers at least one full coarse stride plus 1s margin.
    """
    lookback_s = max(COARSE_STRIDE_S, PRIORITY_STRIDE_S) + 1.0
    lookback = max(0.0, rough_t - lookback_s)
    result = _scan_range(
        cap, reader, target, roi, fps, lookback, rough_t, FINE_STRIDE_S,
        stats, stop_before=rough_t, skip_prefilter=True,
    )
    if result is not None and result.timestamp is not None and result.timestamp < rough_t:
        return result
    # rough_t confirmed as earliest; re-fetch it (no filter needed here either)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(rough_t * fps)))
    _, frame = cap.read()
    stats["ocr_calls"] += 1
    text = _ocr(reader, frame, roi) if frame is not None else ""
    return ScanResult(
        status="FOUND",
        timestamp=rough_t,
        frame_number=int(rough_t * fps),
        ocr_text=text,
        match_score=fuzz.WRatio(target.lower(), text.lower()),
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
    3. Fine scan (no pre-filters) around rough match to find exact first frame.
    Visual scan always covers the whole video.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    stats: dict = {"diff_skipped": 0, "hint_skipped": 0, "ocr_calls": 0}
    rough_match: ScanResult | None = None
    stop_before: float | None = None

    # --- 1. Priority windows (pre-filters off — hint already confirmed text is here) ---
    covered: list[tuple[float, float]] = []
    for ws, we in sorted(priority_windows):
        t_end = min(we + 2.0, stop_before or duration)
        m = _scan_range(cap, reader, target, roi, fps, ws, t_end,
                        PRIORITY_STRIDE_S, stats, stop_before, skip_prefilter=True)
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
                        COARSE_STRIDE_S, stats, stop_before)
        if m is not None and m.timestamp is not None:
            if rough_match is None or rough_match.timestamp is None or m.timestamp < rough_match.timestamp:
                rough_match = m
                stop_before = m.timestamp

    print(f"[scan] stats — ocr_calls: {stats['ocr_calls']}, "
          f"diff_skipped: {stats['diff_skipped']}, hint_skipped: {stats['hint_skipped']}")

    # --- 3. Fine scan (pre-filters off) to find exact first frame ---
    if rough_match is not None and rough_match.timestamp is not None:
        result = _fine_scan(cap, reader, target, roi, fps, rough_match.timestamp, stats)
        cap.release()
        return result

    cap.release()

    if audio_had_hit:
        return ScanResult(status="SPOKEN_NOT_SHOWN", coverage=1.0)
    return ScanResult(status="NOT_FOUND", coverage=1.0)
