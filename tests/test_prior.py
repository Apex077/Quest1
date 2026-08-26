"""Tests for prior.py — SRT parsing, timestamp conversion, VTT parsing, transcript cache."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prior import (
    _parse_srt,
    _parse_vtt_words,
    _ts_to_sec,
    _transcript_cache,
    _word_window_match,
    audio_prior,
    vtt_prior,
)


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


# ── _parse_vtt_words ─────────────────────────────────────────────────────────

VTT_SAMPLE = """\
WEBVTT
Kind: captions
Language: en

00:00:05.000 --> 00:00:08.000 align:start position:0%

<00:00:05.120><c> My</c><00:00:05.520><c> mind</c><00:00:05.920><c> rebels</c>

00:00:05.200 --> 00:00:08.000 align:start position:0%

<00:00:05.120><c> My</c><00:00:05.520><c> mind</c><00:00:05.920><c> rebels</c>
"""

def test_parse_vtt_words_extracts_words():
    words = _parse_vtt_words(VTT_SAMPLE)
    texts = [w["word"] for w in words]
    assert "My" in texts
    assert "mind" in texts
    assert "rebels" in texts

def test_parse_vtt_words_deduplicates():
    # Duplicate cues in the VTT sample should yield only 3 unique (ts, word) pairs
    words = _parse_vtt_words(VTT_SAMPLE)
    assert len(words) == 3

def test_parse_vtt_words_timestamps():
    words = _parse_vtt_words(VTT_SAMPLE)
    assert words[0]["start"] == pytest.approx(5.12)
    assert words[1]["start"] == pytest.approx(5.52)

def test_parse_vtt_words_no_tags_returns_empty():
    plain = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n"
    assert _parse_vtt_words(plain) == []


# ── vtt_prior ────────────────────────────────────────────────────────────────

def test_vtt_prior_finds_phrase(tmp_path):
    vtt = tmp_path / "video.en.vtt"
    vtt.write_text(VTT_SAMPLE)
    windows = vtt_prior(vtt, "my mind rebels")
    assert len(windows) >= 1
    # Start should be just before "My" (5.12s - 0.3s = 4.82s)
    assert windows[0][0] == pytest.approx(4.82)

def test_vtt_prior_no_match_returns_empty(tmp_path):
    vtt = tmp_path / "video.en.vtt"
    vtt.write_text(VTT_SAMPLE)
    assert vtt_prior(vtt, "completely unrelated phrase") == []

def test_vtt_prior_plain_vtt_returns_empty(tmp_path):
    vtt = tmp_path / "video.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n")
    assert vtt_prior(vtt, "hello world") == []


# ── _word_window_match ────────────────────────────────────────────────────────

def test_word_window_match_tight_window():
    words = [
        {"start": 10.0, "word": "my"},
        {"start": 10.4, "word": "mind"},
        {"start": 10.8, "word": "rebels"},
        {"start": 11.2, "word": "at"},
        {"start": 11.6, "word": "stagnation"},
    ]
    windows = _word_window_match(words, "my mind rebels")
    assert len(windows) >= 1
    start, end = windows[0]
    assert start == pytest.approx(9.7)   # 10.0 - 0.3
    assert end == pytest.approx(11.3)    # 10.8 + 0.5

def test_word_window_match_no_match():
    words = [{"start": 1.0, "word": "hello"}, {"start": 1.5, "word": "world"}]
    assert _word_window_match(words, "foo bar baz") == []

def test_word_window_match_empty_words():
    assert _word_window_match([], "target") == []


# ── audio_prior ──────────────────────────────────────────────────────────────

def test_audio_prior_uses_cache(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    # Cache format includes words list (even if empty — triggers word-level matching)
    segs = [{"start": 10.0, "end": 12.0, "text": "my mind rebels at stagnation", "words": [
        {"word": "my", "start": 10.0, "end": 10.3},
        {"word": "mind", "start": 10.3, "end": 10.6},
        {"word": "rebels", "start": 10.6, "end": 11.0},
        {"word": "at", "start": 11.0, "end": 11.2},
        {"word": "stagnation", "start": 11.2, "end": 12.0},
    ]}]
    _transcript_cache(audio, "small").write_text(json.dumps(segs))

    windows = audio_prior(audio, "my mind rebels", model_name="small")
    assert len(windows) >= 1
    # Word-level: start of "my" is 10.0, minus 0.3 = 9.7
    assert windows[0][0] == pytest.approx(9.7)


def test_audio_prior_cache_filename_includes_model(tmp_path):
    audio = tmp_path / "audio.wav"
    assert _transcript_cache(audio, "small").name == "audio.transcript.small.json"
    assert _transcript_cache(audio, "base").name == "audio.transcript.base.json"
    assert _transcript_cache(audio, "medium").name == "audio.transcript.medium.json"


def test_audio_prior_no_match_returns_empty(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    segs = [{"start": 1.0, "end": 2.0, "text": "completely unrelated speech", "words": []}]
    _transcript_cache(audio, "small").write_text(json.dumps(segs))

    windows = audio_prior(audio, "xyz123notaword", model_name="small")
    assert windows == []


def test_audio_prior_force_retranscribe_clears_cache(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    old_segs = [{"start": 1.0, "end": 2.0, "text": "old cached text", "words": []}]
    cache = _transcript_cache(audio, "small")
    cache.write_text(json.dumps(old_segs))

    new_seg = MagicMock()
    new_seg.start = 5.0
    new_seg.end = 7.0
    new_seg.text = "my mind rebels at stagnation"
    new_seg.words = []

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([new_seg]), None)

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        windows = audio_prior(audio, "my mind rebels", model_name="small", force_retranscribe=True)

    # Cache should now contain the new transcript
    assert cache.exists()
    new_cache = json.loads(cache.read_text())
    assert new_cache[0]["start"] == pytest.approx(5.0)
    # Segment-level fallback (empty words): (5.0, 7.0)
    assert windows == [(pytest.approx(5.0), pytest.approx(7.0))]


def test_audio_prior_multiple_matching_windows(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    segs = [
        {"start": 10.0, "end": 12.0, "text": "my mind rebels", "words": []},
        {"start": 50.0, "end": 52.0, "text": "something else", "words": []},
        {"start": 90.0, "end": 92.0, "text": "my mind rebels again", "words": []},
    ]
    _transcript_cache(audio, "small").write_text(json.dumps(segs))

    windows = audio_prior(audio, "my mind rebels", model_name="small")
    assert len(windows) == 2
    assert windows[0][0] < windows[1][0]  # sorted by start
