from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from studio.config import Settings

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("FFmpeg não instalado", allow_module_level=True)


def make_video(
    path: Path,
    *,
    size: str = "640x360",
    rate: int = 30,
    seconds: float = 1.0,
    audio: bool = True,
    rotation: int | None = None,
) -> Path:
    """Gera um vídeo sintético (padrão de teste + tom de 440 Hz)."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    if rotation is not None:
        rotated = path.with_name(path.stem + "-rot" + path.suffix)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(path), "-c", "copy", str(rotated)],
            check=True,
        )
        return rotated
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", tracker="demo")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "input.mp4")
