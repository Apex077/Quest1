# CLAUDE.md

## What this project does

Given a video URL and a target dialogue string, find the first frame where that text
is visible on screen. Output: timestamp, frame number, extracted text, frame image.

Don't tune anything to the specific reference video/string — it will be tested on a
different video and phrase.

## Core rule: visual scan is the source of truth, audio is just a hint

Do NOT build this as "audio finds a timestamp, OCR verifies it." That fails whenever text is on-screen but never spoken (title cards, signs, captions in a different language than the audio).

Instead:

- The visual scan always runs and always covers the whole video. It's the only thing
  allowed to say "found" or "not found."
- Audio transcription (if audio exists and language matches) just produces a ranked
  list of time windows to check **first**. It never causes the scan to skip or stop
  early.
- Soft subtitle tracks (if the container has any — check with `ffprobe`/`yt-dlp
  --list-subs`) are a better, cheaper prior than audio. Check for these before
  bothering with speech-to-text.
- If there's no audio, or audio is corrupted, or language doesn't match, just skip
  straight to the plain chronological visual scan. No special-case handling needed,
  the priority list is simply empty.
- Once a confirmed visual match is found at time T, only things earlier than T still
  need checking (we want the *first* occurrence). Nothing after T matters anymore.

## Core rule: don't hardcode where text appears on screen

Do NOT hardcode a crop region, since text position varies by video and can even shift within one video.

Instead, learn it:

1. Sample ~40-60 frames spread across the whole video (not just the start).
2. Run OCR **detection only** (find text boxes, don't read them yet) on those samples.
3. Cluster the box positions to find where text tends to appear in this video.
4. If there's a clear cluster, use that region as the fast-path crop for the real scan.
5. If there's no clear cluster (scattered positions), fall back to full-frame scanning
   instead of guessing.
6. Occasionally double-check with a full-frame pass even after learning a region, in
   case something appears outside it later.

## If subtitles aren't available

No subtitle track just means you lose that prior condition (which allows you to make optimizations for) and we skip it and fall back to audio (if usable), or plain chronological visual scan if not. The visual scan still always runs and still covers the whole video either way, just without a hint on where to look first.

If the scan finds nothing anywhere, report `NOT_FOUND` (not an error), include how much of the video was actually covered so the result is checkable and better observability. If audio clearly contains the line but the visual scan finds no matching on-screen text, report that as a distinct outcome (e.g. `SPOKEN_NOT_SHOWN`) rather than a flat `NOT_FOUND`.

## General approach

- `yt-dlp` to download, `ffmpeg`/`ffprobe` to inspect and split streams.
- Cheap frame-differencing pass first (e.g. SSIM) to find where content changes at all, so you only run real OCR on frames that actually changed.
- Use a fuzzy match (not exact string match) to compare OCR output against the target text, since OCR won't be perfect.
- Report uncertainty honestly if the match is borderline, rather than forcing a yes/no answer.
