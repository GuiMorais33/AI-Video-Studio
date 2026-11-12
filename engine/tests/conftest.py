from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
    # Sem FaceFusion: um instalado de verdade no computador não pode entrar nos testes. A memória
    # livre do computador de teste também não decide nada (o FaceFusion dos testes é falso).
    # Sem RIFE por padrão: os testes não baixam os pesos (os de tests/test_finish.py usam um minúsculo).
    return Settings(data_dir=tmp_path / "data", tracker="demo", facefusion_dir=tmp_path / "sem-facefusion",
                    facefusion_min_memory_gb=0, interpolate=False)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "input.mp4")


def read_frame(path: Path, idx: int = 0) -> np.ndarray:
    """Quadro `idx` de um vídeo (RGB float32)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    width, height = (int(v) for v in out.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3).astype(np.float32)


def clip_view(root: Path, clip_idx: int, size: tuple[int, int]) -> np.ndarray:
    """O que o vídeo final mostra do original no quadro `clip_idx` do clipe: o recorte do pacote
    (wan_geometry) e o corte central para cobrir `size`, nas coordenadas do original."""
    with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
        job = json.loads(zf.read("job.json"))
    geo, gw, gh = job["geometry"], job["width"], job["height"]
    src = read_frame(next(root.glob("source*")), clip_idx)
    sh, sw = src.shape[:2]
    s = max(size[0] / gw, size[1] / gh)
    x0, y0 = (gw - size[0] / s) / 2, (gh - size[1] / s) / 2
    fx, fy = sw / geo["scaled_width"], sh / geo["scaled_height"]
    box = ((x0 + geo["crop_x"]) * fx, (y0 + geo["crop_y"]) * fy,
           (gw - x0 + geo["crop_x"]) * fx, (gh - y0 + geo["crop_y"]) * fy)
    return np.asarray(Image.fromarray(src.astype(np.uint8)).resize(size, Image.Resampling.LANCZOS, box=box), np.float32)


def source_view(root: Path, package_idx: int, size: tuple[int, int]) -> np.ndarray:
    """clip_view do quadro do clipe que virou o quadro `package_idx` do pacote."""
    with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
        indices = json.loads(zf.read("job.json"))["source_frame_indices"]
    return clip_view(root, indices[package_idx], size)
