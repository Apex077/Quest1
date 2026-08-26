"""
Prior generation: cheap hints for where to scan first.
VTT captions (most precise) → embedded subtitles → audio transcription → nothing.
Neither prior ever causes the visual scan to skip or stop early.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 65  # partial_ratio minimum to count as a candidate window

# ── YouTube VTT word timestamp pattern ──────────────────────────────────────
# Matches: <00:00:05.120><c> word</c>  (YouTube auto-caption cue bodies)
_VTT_WORD_RE = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<\n]+?)\s*</c>")


# ---------------------------------------------------------------------------
# SRT / VTT timestamp helpers
# ---------------------------------------------------------------------------

def _ts_to_sec(ts: str) -> float:
    """'00:01:23,456' or '00:01:23.456' → seconds."""
    ts = ts.replace(",", ".")
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

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
# VTT prior (YouTube auto-captions with per-word timestamps)
# ---------------------------------------------------------------------------

def _parse_vtt_words(text: str) -> list[dict]:
    """Extract {start, word} pairs from YouTube VTT cue bodies, deduped."""
    seen: set[tuple[str, str]] = set()
    words = []
    for m in _VTT_WORD_RE.finditer(text):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            words.append({"start": _ts_to_sec(m.group(1)), "word": m.group(2)})
    return words


def _word_window_match(words: list[dict], target: str) -> list[tuple[float, float]]:
    """Sliding-window phrase match over a word list. Returns tight (start, end) pairs."""
    target_words = target.lower().split()
    w = len(target_words)
    if w == 0 or len(words) < w:
        return []
    windows = []
    for i in range(len(words) - w + 1):
        chunk = words[i:i + w]
        chunk_text = " ".join(c["word"].lower() for c in chunk)
        if fuzz.partial_ratio(target.lower(), chunk_text) >= FUZZY_THRESHOLD:
            start = max(0.0, chunk[0]["start"] - 0.3)
            end = chunk[-1]["start"] + 0.5
            windows.append((start, end))
    return sorted(set(windows))


def vtt_prior(vtt_path: Path, target: str) -> list[tuple[float, float]]:
    """
    Parse a YouTube VTT caption file, fuzzy-match target phrase at word level,
    return tight (start, end) windows sorted by start time.
    Falls back to empty list if no word timestamps are present.
    """
    text = vtt_path.read_text(errors="replace")
    words = _parse_vtt_words(text)
    if not words:
        return []
    return _word_window_match(words, target)


# ---------------------------------------------------------------------------
# Subtitle prior (embedded streams in the container)
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
# Audio prior (faster-whisper with word-level timestamps)
# ---------------------------------------------------------------------------

def _transcript_cache(audio_path: Path, model_name: str = "small") -> Path:
    return audio_path.with_suffix(f".transcript.{model_name}.json")


def _load_transcript(audio_path: Path, model_name: str) -> list[dict] | None:
    cache = _transcript_cache(audio_path, model_name)
    if cache.exists():
        return json.loads(cache.read_text())
    return None


def _save_transcript(audio_path: Path, segments: list[dict], model_name: str) -> None:
    _transcript_cache(audio_path, model_name).write_text(json.dumps(segments))


def _match_windows(segs: list[dict], target: str) -> list[tuple[float, float]]:
    """Match target against segments; use word-level windows when available."""
    all_words = [w for s in segs for w in s.get("words", [])]
    if all_words:
        return _word_window_match(all_words, target)
    # Fallback: segment-level (old cache without word data)
    windows = []
    for seg in segs:
        if fuzz.partial_ratio(target.lower(), seg["text"].lower()) >= FUZZY_THRESHOLD:
            windows.append((seg["start"], seg["end"]))
    return sorted(windows)


def audio_prior(
    audio_path: Path,
    target: str,
    model_name: str = "small",
    force_retranscribe: bool = False,
) -> list[tuple[float, float]]:
    """
    Transcribe audio with faster-whisper (word-level timestamps),
    return tight phrase windows that fuzzy-match the target.
    Transcript is cached as audio.transcript.<model_name>.json.
    """
    if force_retranscribe:
        _transcript_cache(audio_path, model_name).unlink(missing_ok=True)
        print(f"[prior] Cleared cached transcript ({model_name}) — re-transcribing")

    cached = _load_transcript(audio_path, model_name)
    if cached is not None:
        print(f"[prior] Using cached transcript ({model_name})")
        segs = cached
    else:
        from faster_whisper import WhisperModel  # lazy import — heavy load
        print(f"[prior] Transcribing with whisper '{model_name}'...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        raw, _ = model.transcribe(str(audio_path), word_timestamps=True)
        segs = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (s.words or [])
                ],
            }
            for s in raw
        ]
        _save_transcript(audio_path, segs, model_name)
        print(f"[prior] Transcript cached ({len(segs)} segments, model={model_name})")

    return _match_windows(segs, target)
