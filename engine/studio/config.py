"""Configuração do motor, lida de variáveis de ambiente com padrões seguros."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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
        return cls(
            data_dir=data_dir.resolve(),
            tracker=os.environ.get("STUDIO_TRACKER", "sam2"),
            sam2_model=os.environ.get("STUDIO_SAM2_MODEL", "tiny"),
            device=os.environ.get("STUDIO_DEVICE", "auto"),
            max_seconds=_env_float("STUDIO_MAX_SECONDS", 5.0),
            max_side=_env_int("STUDIO_MAX_SIDE", 854),
            max_fps=_env_float("STUDIO_MAX_FPS", 30.0),
            cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else cls.cors_origins,
        )
