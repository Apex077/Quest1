"""
Prior generation: cheap hints for where to scan first.
Subtitles (cheapest) → audio transcription → nothing (plain chronological scan).
Neither prior ever causes the visual scan to skip or stop early.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 65  # partial_ratio minimum to count as a candidate window


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _ts_to_sec(ts: str) -> float:
    """'00:01:23,456' or '00:01:23.456' → seconds."""
    ts = ts.replace(",", ".")
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(text: str) -> list[tuple[float, float, str]]:
    """Returns list of (start_sec, end_sec, caption_text)."""
    out = []
    for block in re.split(r"\n{2,}", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        arrow = next((i for i, l in enumerate(lines) if " --> " in l), None)
        if arrow is None:
            continue
        try:
            start_s, end_s = lines[arrow].split(" --> ")
            caption = " ".join(lines[arrow + 1:])
            out.append((_ts_to_sec(start_s.strip()), _ts_to_sec(end_s.strip()), caption))
        except (ValueError, IndexError):
            continue
    return out


# ---------------------------------------------------------------------------
# Subtitle prior
# ---------------------------------------------------------------------------

def subtitle_prior(video_path: Path, target: str) -> list[tuple[float, float]]:
    """
    Extract any embedded subtitle streams, fuzzy-match against target,
    return matched (start, end) windows sorted by start time.
    """
    probe_out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True,
    )
    streams = json.loads(probe_out.stdout).get("streams", [])
    sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    if not sub_streams:
        return []

    windows: list[tuple[float, float]] = []
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
        tmp = Path(f.name)

    try:
        for i in range(len(sub_streams)):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-map", f"0:s:{i}", str(tmp), "-loglevel", "error"],
                capture_output=True,
            )
            if not tmp.exists() or tmp.stat().st_size < 10:
                continue
            for start, end, caption in _parse_srt(tmp.read_text(errors="replace")):
                if fuzz.partial_ratio(target.lower(), caption.lower()) >= FUZZY_THRESHOLD:
                    windows.append((start, end))
    finally:
        tmp.unlink(missing_ok=True)

    return sorted(set(windows))


# ---------------------------------------------------------------------------
# Audio prior
# ---------------------------------------------------------------------------

def _transcript_cache(audio_path: Path) -> Path:
    return audio_path.with_suffix(".transcript.json")


def _load_transcript(audio_path: Path) -> list[dict] | None:
    cache = _transcript_cache(audio_path)
    if cache.exists():
        return json.loads(cache.read_text())
    return None


def _save_transcript(audio_path: Path, segments: list[dict]) -> None:
    _transcript_cache(audio_path).write_text(json.dumps(segments))


def audio_prior(audio_path: Path, target: str) -> list[tuple[float, float]]:
    """
    Transcribe audio with faster-whisper (base model, CPU, int8),
    return segment windows whose text fuzzy-matches the target.
    Transcript is cached as audio.transcript.json — subsequent calls skip transcription.
    """
    cached = _load_transcript(audio_path)
    if cached is not None:
        print("[prior] Using cached transcript")
        segs = cached
    else:
        from faster_whisper import WhisperModel  # lazy import — heavy load
        # ponytail: base model + int8 for speed; swap to "small"/"medium" if accuracy matters
        model = WhisperModel("base", device="cpu", compute_type="int8")
        raw, _ = model.transcribe(str(audio_path), word_timestamps=False)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in raw]
        _save_transcript(audio_path, segs)
        print(f"[prior] Transcript cached ({len(segs)} segments)")

    windows: list[tuple[float, float]] = []
    for seg in segs:
        if fuzz.partial_ratio(target.lower(), seg["text"].lower()) >= FUZZY_THRESHOLD:
            windows.append((seg["start"], seg["end"]))

    return sorted(windows)
