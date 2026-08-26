# Video Frame Extractor

Given a video URL and a target phrase, finds the **first frame** where that text is visible on screen. Returns the timestamp, frame number, any extracted text, and a saved frame image.

Works with YouTube, ok.ru, and most other sites supported by yt-dlp.

---

## How it works

1. **Download** — yt-dlp fetches the video and any available auto-generated captions (VTT).
2. **Prior** — looks for the phrase in captions/subtitles to get a precise timestamp hint:
   - YouTube VTT captions (word-level timestamps, most accurate)
   - Embedded subtitle streams in the container
   - Whisper audio transcription (word-level, fallback)
3. **Extract** — seeks to that timestamp and saves the frame.
4. **OCR mode (optional)** — runs a full visual scan with EasyOCR to verify text on screen and find the exact first frame, rather than trusting the audio/caption timestamp alone.

---

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` (must be on your `PATH`)

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

---

## Setup

```bash
git clone <repo-url>
cd Quest1-Video-Frame-Extractor

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Web UI

```bash
source venv/bin/activate
python app.py
```

Open `http://localhost:5000` in your browser.

**Options in the sidebar:**

| Option | Default | Description |
|---|---|---|
| Whisper model | small | Accuracy/speed trade-off for audio transcription (used only when no captions are found) |
| Keep downloaded video | on | Uncheck to delete the video file after extraction, saving disk space. The extracted frame image is always kept. |
| Regenerate transcript | off | Clears the cached Whisper transcript and re-transcribes from scratch |
| OCR mode | off | Beta: runs a full visual scan to verify and pinpoint text on screen |

Results stream live in the right panel. The extracted frame image is shown when a match is found.

---

## CLI

```bash
source venv/bin/activate

python pipeline.py <url> <phrase> [out_dir] [options]
```

**Options:**

| Flag | Description |
|---|---|
| `--use-ocr` | Enable full visual OCR scan (beta) |
| `--retranscribe` | Clear cached transcript and re-run Whisper |
| `--no-keep` | Delete the downloaded video after extraction |
| `--whisper-model <name>` | Whisper model size: `tiny`, `base`, `small` (default), `medium` |

**Examples:**

```bash
# Basic usage — extracts the frame where the phrase is heard/shown
python pipeline.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' 'never gonna give you up'

# Specify output directory
python pipeline.py 'https://youtu.be/abc123' 'hello world' ./my_output

# Delete video after extraction (saves disk space)
python pipeline.py 'https://youtu.be/abc123' 'hello world' --no-keep

# Use the more accurate medium model for transcription
python pipeline.py 'https://youtu.be/abc123' 'target phrase' --whisper-model medium

# Full visual scan with OCR (slower but more precise)
python pipeline.py 'https://youtu.be/abc123' 'hello world' --use-ocr

# Force re-transcription (ignore cached transcript)
python pipeline.py 'https://youtu.be/abc123' 'target phrase' --retranscribe
```

**Output:**
```
FOUND
  timestamp   : 42.300s
  frame       : 1057
  frame_image : downloads/match_42.300s.jpg
```

Possible statuses:
- `FOUND` — phrase matched; frame image saved
- `NOT_FOUND` — phrase not detected in audio or on screen
- `SPOKEN_NOT_SHOWN` — audio/captions matched the phrase but no matching on-screen text was found (OCR mode only)

---

## Output files

All files are saved to `downloads/` by default (or the `out_dir` you specify):

| File | Description |
|---|---|
| `video.mp4` | Downloaded video (kept unless `--no-keep` / uncheck in UI) |
| `video.url` | URL sidecar — used to detect when the URL changes |
| `video.en.vtt` | Downloaded YouTube auto-captions (if available) |
| `audio.wav` | Extracted audio stream (16kHz mono WAV) |
| `audio.transcript.small.json` | Cached Whisper transcript (model name in filename) |
| `match_<timestamp>s.jpg` | Extracted frame image |

On the next run with the **same URL**, the video and transcript are reused automatically. Change the URL and the old video is replaced.

---

## Running tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```
