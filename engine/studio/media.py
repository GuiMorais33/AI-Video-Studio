"""Operações de vídeo via FFmpeg: inspeção, recorte/redução, extração e codificação."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class MediaError(RuntimeError):
    pass


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} não encontrado no PATH. Instale o FFmpeg (veja o README).")
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"Falha ao executar {Path(cmd[0]).name}:\n{tail}")
    return proc


def _parse_rate(value: str | None) -> Fraction | None:
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def fps_arg(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    out = _run([
        ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("O arquivo não contém trilha de vídeo.")
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if fps is None:
        raise MediaError("Não foi possível determinar a taxa de quadros do vídeo.")
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def prepare_clip(
    src: Path,
    dst: Path,
    *,
    start: float = 0.0,
    max_seconds: float = 5.0,
    max_side: int = 854,
    max_fps: float = 30.0,
) -> VideoInfo:
    """Recorta `max_seconds` a partir de `start`, reduz a resolução e fixa o FPS.

    A rotação de vídeos de celular é aplicada pelo FFmpeg (autorotate), então o
    clipe de saída já está na orientação correta. O resultado é H.264 CFR com o
    áudio original em AAC, base para todas as etapas seguintes.
    """
    src_info = probe(src)
    if src_info.duration and start >= src_info.duration:
        raise MediaError(
            f"O início ({start:.2f}s) está além da duração do vídeo ({src_info.duration:.2f}s)."
        )
    fps = min(src_info.fps, Fraction(max_fps).limit_denominator(1001))
    vf = ",".join([
        f"scale=w='min({max_side},iw)':h='min({max_side},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2",
        f"fps={fps_arg(fps)}",
        "format=yuv420p",
    ])
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg_bin(), "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{max_seconds:.3f}", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst),
    ])
    return probe(dst)


def extract_frames(clip: Path, frames_dir: Path) -> int:
    """Extrai quadros JPEG nomeados 00000.jpg, 00001.jpg... (formato aceito pelo SAM 2)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg_bin(), "-y", "-v", "error", "-i", str(clip),
        "-q:v", "2", "-start_number", "0", str(frames_dir / "%05d.jpg"),
    ])
    count = len(list(frames_dir.glob("*.jpg")))
    if count == 0:
        raise MediaError("Nenhum quadro foi extraído do vídeo.")
    return count


def encode_frames(
    frames_dir: Path,
    fps: Fraction,
    out: Path,
    *,
    pattern: str = "%05d.jpg",
    audio_from: Path | None = None,
    crf: int = 20,
) -> Path:
    """Codifica uma sequência de imagens em MP4 (H.264), opcionalmente com o áudio de outro arquivo."""
    cmd = [
        ffmpeg_bin(), "-y", "-v", "error",
        "-framerate", fps_arg(fps), "-start_number", "0", "-i", str(frames_dir / pattern),
    ]
    if audio_from is not None:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-shortest"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd)
    return out
