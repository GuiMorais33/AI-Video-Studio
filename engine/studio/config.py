"""Configuração do motor, lida de variáveis de ambiente com padrões seguros."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# FaceFusion 3.9.1 instalado pelo scripts/setup.sh (código + Python próprio em .venv), para o
# refino do rosto neste computador. Os modelos (~1,7 GB) baixam em .assets/models no 1º uso.
FACEFUSION_DIR = REPO_ROOT / "engine" / ".vendor" / "facefusion"

# Limites de upload/preparo. Clipes longos multiplicam memória e tempo de CPU do
# SAM 2 (cada quadro é carregado em 1024x1024 float32, ~12 MB por quadro).
MAX_CLIP_SECONDS = 15.0


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    # "sam2" usa o modelo real; "demo" usa um rastreador geométrico sem IA,
    # apenas para desenvolver a interface sem PyTorch instalado.
    tracker: str = "sam2"
    sam2_model: str = "tiny"
    device: str = "auto"
    max_seconds: float = 5.0
    max_side: int = 854
    max_fps: float = 30.0
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")
    # Refino do rosto (FaceFusion na CPU). O Python padrão é o <facefusion_dir>/.venv/bin/python.
    facefusion_dir: Path = FACEFUSION_DIR
    facefusion_python: Path | None = None
    facefusion_threads: int = 4
    # Sem nenhuma saída do FaceFusion por este tempo, ele é encerrado (download travado, processo
    # preso). Não há limite total: na CPU, o refino de um clipe leva de minutos a mais de uma hora.
    facefusion_idle_timeout_s: float = 20 * 60
    # Refina o rosto sozinho logo que o vídeo do Wan chega (do Kaggle ou importado à mão).
    face_refine_auto: bool = True

    @property
    def facefusion_python_path(self) -> Path:
        return self.facefusion_python or self.facefusion_dir / ".venv" / "bin" / "python"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or REPO_ROOT / "data").expanduser()
        origins = os.environ.get("STUDIO_CORS_ORIGINS")
        ff_python = os.environ.get("STUDIO_FACEFUSION_PYTHON")
        return cls(
            data_dir=data_dir.resolve(),
            tracker=os.environ.get("STUDIO_TRACKER", "sam2"),
            sam2_model=os.environ.get("STUDIO_SAM2_MODEL", "tiny"),
            device=os.environ.get("STUDIO_DEVICE", "auto"),
            max_seconds=_env_float("STUDIO_MAX_SECONDS", 5.0),
            max_side=_env_int("STUDIO_MAX_SIDE", 854),
            max_fps=_env_float("STUDIO_MAX_FPS", 30.0),
            cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else cls.cors_origins,
            facefusion_dir=Path(os.environ.get("STUDIO_FACEFUSION_DIR") or FACEFUSION_DIR).expanduser(),
            facefusion_python=Path(ff_python).expanduser() if ff_python else None,
            facefusion_threads=_env_int("STUDIO_FACEFUSION_THREADS", 4),
            facefusion_idle_timeout_s=_env_float("STUDIO_FACEFUSION_IDLE_TIMEOUT_S", 20 * 60),
            face_refine_auto=os.environ.get("STUDIO_FACE_REFINE_AUTO", "1") != "0",
        )
