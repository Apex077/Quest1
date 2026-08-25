"""Phase 1: download video, probe metadata, extract audio stream."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget


@dataclass
class VideoInfo:
    video_path: Path
    audio_path: Path | None  # None if no audio stream
    duration: float          # seconds
    fps: float
    width: int
    height: int
    subtitle_langs: list[str] = field(default_factory=list)


_YDL_OPTS = {
    "nocheckcertificate": True,
    "impersonate": ImpersonateTarget(client="chrome"),
}


def _download_and_meta(url: str, out_dir: Path) -> tuple[Path, list[str]]:
    """Download video and return (video_path, subtitle_langs) in one yt-dlp session.
    If a video file already exists in out_dir, skip the network call entirely."""
    out_dir.mkdir(parents=True, exist_ok=True)

    url_record = out_dir / "video.url"
    existing = [p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".url")]
    if existing and url_record.exists() and url_record.read_text().strip() == url:
        video_path = max(existing, key=lambda p: p.stat().st_size)
        print(f"[ingest] Using cached video: {video_path}")
        return video_path, []
    elif existing:
        print("[ingest] URL changed — removing old video and re-downloading")
        for p in existing:
            p.unlink()

    opts = {
        **_YDL_OPTS,
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True) or {}

    subs = {**info.get("subtitles", {}), **info.get("automatic_captions", {})}
    subtitle_langs = list(subs.keys())

    candidates = [p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".url")]
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no output in {out_dir}")
    video_path = max(candidates, key=lambda p: p.stat().st_size)
    url_record.write_text(url)
    return video_path, subtitle_langs


def _probe(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(video_path),
    ]
    return json.loads(subprocess.check_output(cmd))


def _extract_audio(video_path: Path, out_dir: Path) -> Path | None:
    audio_path = out_dir / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path), "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not audio_path.exists():
        return None  # no audio stream
    # empty wav = header only (~44 bytes)
    if audio_path.stat().st_size < 100:
        audio_path.unlink()
        return None
    return audio_path


def ingest(url: str, out_dir: Path) -> VideoInfo:
    video_path, subtitle_langs = _download_and_meta(url, out_dir)
    probe = _probe(video_path)

    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num, den = map(int, video_stream["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(probe["format"]["duration"])
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    audio_path = _extract_audio(video_path, out_dir)

    return VideoInfo(
        video_path=video_path,
        audio_path=audio_path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        subtitle_langs=subtitle_langs,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python ingest.py <url> [out_dir]")
        sys.exit(1)
    url = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("downloads")
    info = ingest(url, out)
    print(f"video   : {info.video_path}")
    print(f"audio   : {info.audio_path}")
    print(f"duration: {info.duration:.1f}s  fps: {info.fps:.2f}  {info.width}x{info.height}")
    print(f"subs    : {info.subtitle_langs or 'none'}")
