"""
Main pipeline orchestrator.
Usage: python pipeline.py <url> <target_phrase> [out_dir]
"""

import sys
from pathlib import Path

import cv2
import easyocr

from ingest import VideoInfo, ingest
from prior import audio_prior, subtitle_prior
from roi import learn_roi
from scan import ScanResult, find_first


def _save_frame(frame, out_dir: Path, timestamp: float) -> Path:
    path = out_dir / f"match_{timestamp:.3f}s.jpg"
    cv2.imwrite(str(path), frame)
    return path


def run(info: VideoInfo, target: str, out_dir: Path) -> ScanResult:
    # Single reader instance shared by ROI learning and visual scan.
    # GPU is 10-50× faster than CPU; auto-detect so this works on any machine.
    import torch
    gpu = torch.cuda.is_available()
    print(f"[pipeline] Loading OCR model (gpu={gpu})...")
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    # --- Subtitle prior (cheapest) ---
    sub_windows: list[tuple[float, float]] = []
    if info.subtitle_langs:
        print(f"[pipeline] Subtitle tracks available: {info.subtitle_langs}")
        sub_windows = subtitle_prior(info.video_path, target)
        print(f"[pipeline] Subtitle windows matched: {sub_windows or 'none'}")

    # --- Audio prior (only if no subtitle hit) ---
    audio_windows: list[tuple[float, float]] = []
    audio_had_hit = False
    if not sub_windows and info.audio_path:
        print("[pipeline] No subtitle match — transcribing audio...")
        audio_windows = audio_prior(info.audio_path, target)
        audio_had_hit = bool(audio_windows)
        print(f"[pipeline] Audio windows matched: {audio_windows or 'none'}")

    priority = sub_windows or audio_windows

    # --- Learn text region ---
    print("[pipeline] Sampling frames to learn text region...")
    roi = learn_roi(info.video_path, reader)
    print(f"[pipeline] ROI: {roi or 'none (full-frame scan)'}")

    # --- Visual scan (always covers the whole video) ---
    print("[pipeline] Starting visual scan...")
    result = find_first(info.video_path, target, roi, priority, reader, audio_had_hit)

    # --- Report ---
    if result.status == "FOUND":
        assert result.frame_image is not None
        img_path = _save_frame(result.frame_image, out_dir, result.timestamp)
        print(f"FOUND")
        print(f"  timestamp   : {result.timestamp:.3f}s")
        print(f"  frame       : {result.frame_number}")
        print(f"  ocr_text    : {result.ocr_text!r}")
        print(f"  match_score : {result.match_score:.1f}")
        print(f"  frame_image : {img_path}")
    elif result.status == "SPOKEN_NOT_SHOWN":
        print("SPOKEN_NOT_SHOWN — audio contains the phrase but it never appeared on screen")
    else:
        print(f"NOT_FOUND — scanned {result.coverage * 100:.0f}% of video, no visual match")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python pipeline.py <url> <target_phrase> [out_dir]")
        sys.exit(1)

    url = sys.argv[1]
    target = sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("downloads")

    info = ingest(url, out_dir)
    run(info, target, out_dir)
