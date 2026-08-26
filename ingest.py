"""Phase 1: download video, probe metadata, extract audio stream."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
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
    vtt_paths: list[Path] = field(default_factory=list)


_YDL_OPTS = {
    "nocheckcertificate": True,
    "impersonate": ImpersonateTarget(client="chrome"),
}

# Prefer H.264 MP4 — AV1 is unsupported by most OpenCV builds (no hw accel).
# Falls back to any format if H.264 isn't available for the source.
_VIDEO_FORMAT = (
    "bestvideo[ext=mp4][vcodec!*=av01]+bestaudio[ext=m4a]"
    "/bestvideo[ext=mp4]+bestaudio"
    "/bestvideo[vcodec!*=av01]+bestaudio"
    "/bestvideo+bestaudio/best"
)


def _opencv_can_read(video_path: Path) -> bool:
    """Quick check: can OpenCV actually decode frames from this file?"""
    cap = cv2.VideoCapture(str(video_path))
    ok, _ = cap.read()
    cap.release()
    return ok


def _download_and_meta(url: str, out_dir: Path) -> tuple[Path, list[Path]]:
    """Download video (+ VTT captions if available) and return (video_path, vtt_paths).
    If a video file already exists in out_dir, skip the network call entirely."""
    out_dir.mkdir(parents=True, exist_ok=True)

    url_record = out_dir / "video.url"
    existing = [p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".url", ".vtt")]
    if existing and url_record.exists() and url_record.read_text().strip() == url:
        video_path = max(existing, key=lambda p: p.stat().st_size)
        if _opencv_can_read(video_path):
            print(f"[ingest] Using cached video: {video_path}")
            vtt_paths = sorted(out_dir.glob("video.*.vtt"))
            return video_path, vtt_paths
        else:
            print(f"[ingest] Cached video is not readable by OpenCV (likely AV1) — re-downloading")
            for p in existing:
                p.unlink()
            url_record.unlink(missing_ok=True)
    elif existing:
        print("[ingest] URL changed — removing old video and re-downloading")
        for p in existing:
            p.unlink()

    opts = {
        **_YDL_OPTS,
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "format": _VIDEO_FORMAT,
        "merge_output_format": "mp4",
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type]
        ydl.extract_info(url, download=True)

    candidates = [p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".url", ".vtt")]
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no output in {out_dir}")
    video_path = max(candidates, key=lambda p: p.stat().st_size)
    url_record.write_text(url)
    vtt_paths = sorted(out_dir.glob("video.*.vtt"))
    return video_path, vtt_paths


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
    video_path, vtt_paths = _download_and_meta(url, out_dir)
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
        vtt_paths=vtt_paths,
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
    print(f"vtt     : {[str(p) for p in info.vtt_paths] or 'none'}")
