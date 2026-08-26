"""Tests for prior.py — SRT parsing, timestamp conversion, transcript cache."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prior import _parse_srt, _ts_to_sec, audio_prior, _transcript_cache


# ── _ts_to_sec ──────────────────────────────────────────────────────────────

def test_ts_to_sec_comma():
    assert _ts_to_sec("00:01:23,456") == pytest.approx(83.456)

def test_ts_to_sec_dot():
    assert _ts_to_sec("00:00:05.000") == pytest.approx(5.0)

def test_ts_to_sec_hours():
    assert _ts_to_sec("01:00:00,000") == pytest.approx(3600.0)


# ── _parse_srt ───────────────────────────────────────────────────────────────

SRT_SAMPLE = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05,500 --> 00:00:07,000
My mind rebels at stagnation

"""

def test_parse_srt_count():
    assert len(_parse_srt(SRT_SAMPLE)) == 2

def test_parse_srt_timestamps():
    result = _parse_srt(SRT_SAMPLE)
    assert result[0][:2] == (pytest.approx(1.0), pytest.approx(3.0))
    assert result[1][:2] == (pytest.approx(5.5), pytest.approx(7.0))

def test_parse_srt_text():
    result = _parse_srt(SRT_SAMPLE)
    assert result[0][2] == "Hello world"
    assert "rebels" in result[1][2]

def test_parse_srt_empty():
    assert _parse_srt("") == []

def test_parse_srt_malformed_block_skipped():
    srt = "not a valid block\n\n"
    assert _parse_srt(srt) == []


# ── audio_prior ──────────────────────────────────────────────────────────────

def test_audio_prior_uses_cache(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    segs = [{"start": 10.0, "end": 12.0, "text": "my mind rebels at stagnation"}]
    _transcript_cache(audio).write_text(json.dumps(segs))

    windows = audio_prior(audio, "my mind rebels")
    assert windows == [(pytest.approx(10.0), pytest.approx(12.0))]


def test_audio_prior_no_match_returns_empty(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    segs = [{"start": 1.0, "end": 2.0, "text": "completely unrelated speech"}]
    _transcript_cache(audio).write_text(json.dumps(segs))

    windows = audio_prior(audio, "xyz123notaword")
    assert windows == []


def test_audio_prior_force_retranscribe_clears_cache(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    old_segs = [{"start": 1.0, "end": 2.0, "text": "old cached text"}]
    cache = _transcript_cache(audio)
    cache.write_text(json.dumps(old_segs))

    new_seg = MagicMock()
    new_seg.start = 5.0
    new_seg.end = 7.0
    new_seg.text = "my mind rebels at stagnation"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([new_seg]), None)

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        windows = audio_prior(audio, "my mind rebels", force_retranscribe=True)

    # Cache should now contain the new transcript
    assert cache.exists()
    new_cache = json.loads(cache.read_text())
    assert new_cache[0]["start"] == pytest.approx(5.0)
    assert windows == [(pytest.approx(5.0), pytest.approx(7.0))]


def test_audio_prior_multiple_matching_windows(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    segs = [
        {"start": 10.0, "end": 12.0, "text": "my mind rebels"},
        {"start": 50.0, "end": 52.0, "text": "something else"},
        {"start": 90.0, "end": 92.0, "text": "my mind rebels again"},
    ]
    _transcript_cache(audio).write_text(json.dumps(segs))

    windows = audio_prior(audio, "my mind rebels")
    assert len(windows) == 2
    assert windows[0][0] < windows[1][0]  # sorted by start
