"""
Main pipeline orchestrator.
Usage: python pipeline.py <url> <target_phrase> [out_dir] [--use-ocr]

Default: audio/subtitle priors → extract frame at matched timestamp (no OCR).
--use-ocr: enable full OCR-based visual scan (beta; may struggle with stylised fonts).
"""

import sys
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from ingest import VideoInfo, ingest
from prior import audio_prior, subtitle_prior
from scan import ScanResult, extract_frame_at, find_first


def _save_frame(frame, out_dir: Path, timestamp: float) -> Path:
    path = out_dir / f"match_{timestamp:.3f}s.jpg"
    cv2.imwrite(str(path), frame)
    return path


def _report(result: ScanResult, out_dir: Path) -> None:
    if result.status == "FOUND":
        assert result.frame_image is not None
        img_path = _save_frame(result.frame_image, out_dir, result.timestamp)
        print("FOUND")
        print(f"  timestamp   : {result.timestamp:.3f}s")
        print(f"  frame       : {result.frame_number}")
        if result.ocr_text is not None:
            print(f"  ocr_text    : {result.ocr_text!r}")
            print(f"  match_score : {result.match_score:.1f}")
        print(f"  frame_image : {img_path}")
    elif result.status == "SPOKEN_NOT_SHOWN":
        print("SPOKEN_NOT_SHOWN — audio contains the phrase but it never appeared on screen")
    else:
        print(f"NOT_FOUND — scanned {result.coverage * 100:.0f}% of video, no visual match")


def run(info: VideoInfo, target: str, out_dir: Path, use_ocr: bool = False) -> ScanResult:
    # --- Subtitle prior (cheapest; ffprobe on local file, always available) ---
    print("[pipeline] Checking embedded subtitle tracks...")
    sub_windows = subtitle_prior(info.video_path, target)
    print(f"[pipeline] Subtitle windows matched: {sub_windows or 'none'}")

    # --- Audio prior (if no subtitle hit) ---
    audio_windows: list[tuple[float, float]] = []
    audio_had_hit = False
    if not sub_windows and info.audio_path:
        print("[pipeline] No subtitle match — transcribing audio...")
        audio_windows = audio_prior(info.audio_path, target)
        audio_had_hit = bool(audio_windows)
        print(f"[pipeline] Audio windows matched: {audio_windows or 'none'}")

    priority = sub_windows or audio_windows

    # --- Audio-only mode (default): trust the prior, extract the frame ---
    if not use_ocr:
        if priority:
            t = priority[0][0]  # earliest matched window start
            print(f"[pipeline] Audio-only mode: extracting frame at {t:.3f}s")
            result = extract_frame_at(info.video_path, t)
        else:
            result = ScanResult(status="NOT_FOUND")
        _report(result, out_dir)
        return result

    # --- OCR mode (beta): full visual scan with text recognition ---
    print("[pipeline] OCR mode enabled (beta)")
    import easyocr
    import torch
    from roi import learn_roi

    gpu = torch.cuda.is_available()
    print(f"[pipeline] Loading OCR model (gpu={gpu})...")
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    print("[pipeline] Sampling frames to learn text region...")
    roi = learn_roi(info.video_path, reader)
    print(f"[pipeline] ROI: {roi or 'none (full-frame scan)'}")

    print("[pipeline] Starting visual scan...")
    result = find_first(info.video_path, target, roi, priority, reader, audio_had_hit)
    _report(result, out_dir)
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: python pipeline.py <url> <target_phrase> [out_dir] [--use-ocr]")
        sys.exit(1)

    url = args[0]
    target = args[1]
    use_ocr = "--use-ocr" in args
    remaining = [a for a in args[2:] if a != "--use-ocr"]
    out_dir = Path(remaining[0]) if remaining else Path("downloads")

    info = ingest(url, out_dir)
    run(info, target, out_dir, use_ocr=use_ocr)
