"""
Visual scanner: coarse stride → frame-diff filter → OCR → fuzzy match.
OCR mode is optional (--use-ocr flag). Default mode extracts the frame at the
audio-prior timestamp without any OCR. OCR mode adds visual text verification
but requires clean on-screen text (beta; struggles with stylised/period fonts).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

FUZZY_THRESHOLD = 70
DIFF_THRESHOLD = 4.0
COARSE_STRIDE_S = 2.0
PRIORITY_STRIDE_S = 0.5
FINE_STRIDE_S = 0.1
TEXT_EDGE_THRESHOLD = 0.04
_OCR_CONF_THRESHOLD = 0.2


@dataclass
class ScanResult:
    status: str                          # FOUND | NOT_FOUND | SPOKEN_NOT_SHOWN
    timestamp: float | None = None
    frame_number: int | None = None
    ocr_text: str | None = None
    match_score: float | None = None
    frame_image: np.ndarray | None = None
    coverage: float = 1.0


def extract_frame_at(video_path: Path, t: float) -> ScanResult:
    """Return the frame at time t without any OCR. Used in audio-only mode."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = round(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ScanResult(status="NOT_FOUND")
    return ScanResult(
        status="FOUND",
        timestamp=t,
        frame_number=frame_no,
        frame_image=frame,
    )


# ---------------------------------------------------------------------------
# OCR helpers (only used when --use-ocr is active)
# ---------------------------------------------------------------------------

def _crop(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def _subtitle_strip(frame: np.ndarray, roi: tuple | None) -> np.ndarray:
    if roi is not None:
        return _crop(frame, roi)
    h = frame.shape[0]
    return frame[int(h * 0.70):, :]


def _strip_changed(prev: np.ndarray | None, curr: np.ndarray, roi: tuple | None) -> bool:
    if prev is None:
        return True
    a = _subtitle_strip(prev, roi).astype(np.float32)
    b = _subtitle_strip(curr, roi).astype(np.float32)
    return float(np.mean(np.abs(b - a))) > DIFF_THRESHOLD


def _has_text_hint(frame: np.ndarray, roi: tuple | None) -> bool:
    strip = _subtitle_strip(frame, roi)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / (255 * edges.size)) >= TEXT_EDGE_THRESHOLD


def _ocr(reader, frame: np.ndarray, roi: tuple | None) -> str:
    if roi is not None:
        results = reader.readtext(_crop(frame, roi), detail=1)
    else:
        h = frame.shape[0]
        combined = np.vstack([frame[:int(h * 0.30)], frame[int(h * 0.70):]])
        results = reader.readtext(combined, detail=1)
    return " ".join(text for (_, text, conf) in results if conf >= _OCR_CONF_THRESHOLD)


def _check_frame(
    cap, reader, target, roi, fps, t, prev_frame, stats, skip_prefilter=False,
):
    frame_no = round(t * fps)
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
    text = _ocr(reader, frame, roi)
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


def _scan_range(cap, reader, target, roi, fps, t_start, t_end, stride, stats,
                stop_before=None, skip_prefilter=False):
    best = None
    prev_frame = None
    t = t_start
    while t <= t_end:
        if stop_before is not None and t >= stop_before:
            break
        match, prev_frame = _check_frame(
            cap, reader, target, roi, fps, t, prev_frame, stats, skip_prefilter)
        if match:
            best = match
            stop_before = match.timestamp
        t += stride
    return best


def _fine_scan(cap, reader, target, roi, fps, rough_t, stats, audio_had_hit=False):
    lookback = max(0.0, rough_t - (max(COARSE_STRIDE_S, PRIORITY_STRIDE_S) + 1.0))
    result = _scan_range(cap, reader, target, roi, fps, lookback, rough_t,
                         FINE_STRIDE_S, stats, stop_before=rough_t, skip_prefilter=True)
    if result is not None and result.timestamp is not None and result.timestamp < rough_t:
        return result
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(round(rough_t * fps)))
    _, frame = cap.read()
    stats["ocr_calls"] += 1
    text = _ocr(reader, frame, roi) if frame is not None else ""
    score = fuzz.WRatio(target.lower(), text.lower())
    if score < FUZZY_THRESHOLD:
        return ScanResult(status="SPOKEN_NOT_SHOWN" if audio_had_hit else "NOT_FOUND")
    return ScanResult(
        status="FOUND",
        timestamp=rough_t,
        frame_number=round(rough_t * fps),
        ocr_text=text,
        match_score=float(score),
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
    """Full OCR-based visual scan. Only called when --use-ocr is active."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

    stats: dict = {"diff_skipped": 0, "hint_skipped": 0, "ocr_calls": 0}
    rough_match = None
    stop_before = None

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

    if rough_match is not None and rough_match.timestamp is not None:
        result = _fine_scan(cap, reader, target, roi, fps, rough_match.timestamp, stats, audio_had_hit)
        cap.release()
        return result

    cap.release()
    if audio_had_hit:
        return ScanResult(status="SPOKEN_NOT_SHOWN", coverage=1.0)
    return ScanResult(status="NOT_FOUND", coverage=1.0)
