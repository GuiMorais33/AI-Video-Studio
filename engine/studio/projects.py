"""Layout de arquivos de um projeto, criação a partir de um vídeo e exportação."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from . import media
from .config import Settings
from .db import Database
from .masks import load_mask, load_rgb, object_color, overlay

ProgressFn = Callable[..., None]

EXPORT_FILES = ("preview.mp4", "mask.mp4", "masks.zip")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def clip(self) -> Path:
        return self.root / "clip.mp4"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def history(self) -> Path:
        return self.root / "history"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def frame(self, idx: int) -> Path:
        return self.frames / f"{idx:05d}.jpg"

    def mask(self, obj_id: int, idx: int) -> Path:
        return self.masks / f"obj{obj_id}" / f"{idx:05d}.png"

    def history_mask(self, click_id: int) -> Path:
        return self.history / f"{click_id}.png"

    def obj_ids(self) -> list[int]:
        if not self.masks.exists():
            return []
        ids = (int(p.name[3:]) for p in self.masks.iterdir() if re.fullmatch(r"obj\d+", p.name))
        return sorted(ids)


def project_paths(settings: Settings, project_id: str) -> ProjectPaths:
    return ProjectPaths(settings.projects_dir / project_id)


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ""


def create_project(
    db: Database,
    settings: Settings,
    video: Path,
    *,
    source_name: str,
    name: str | None = None,
    start: float = 0.0,
    seconds: float | None = None,
    move: bool = False,
) -> dict[str, Any]:
    """Guarda o vídeo original, gera o clipe reduzido e extrai os quadros."""
    project_id = uuid.uuid4().hex[:12]
    paths = project_paths(settings, project_id)
    paths.root.mkdir(parents=True)
    try:
        source = paths.root / f"source{_safe_suffix(source_name)}"
        (shutil.move if move else shutil.copyfile)(video, source)
        info = media.prepare_clip(
            source, paths.clip,
            start=start,
            max_seconds=seconds or settings.max_seconds,
            max_side=settings.max_side,
            max_fps=settings.max_fps,
        )
        num_frames = media.extract_frames(paths.clip, paths.frames)
        return db.create_project(
            id=project_id,
            name=name or Path(source_name).stem or project_id,
            source_name=source_name,
            clip_start=start,
            width=info.width,
            height=info.height,
            fps=media.fps_arg(info.fps),
            num_frames=num_frames,
            duration=info.duration,
            has_audio=int(info.has_audio),
        )
    except BaseException:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise


def export_project(project: dict[str, Any], paths: ProjectPaths, progress: ProgressFn) -> str:
    """Gera preview.mp4 (pessoa destacada, com áudio), mask.mp4 (preto/branco) e masks.zip (PNGs)."""
    obj_ids = paths.obj_ids()
    if not obj_ids:
        raise ValueError("Não há máscaras para exportar. Selecione a pessoa e rastreie primeiro.")
    fps = Fraction(project["fps"])
    total = project["num_frames"]
    paths.exports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.root, prefix="export-") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "overlay").mkdir()
        (tmp / "mask").mkdir()
        for idx in range(total):
            frame = load_rgb(paths.frame(idx))
            masks = [(load_mask(paths.mask(obj, idx)), object_color(obj)) for obj in obj_ids]
            Image.fromarray(overlay(frame, masks)).save(tmp / "overlay" / f"{idx:05d}.jpg", quality=92)
            union = np.zeros(frame.shape[:2], dtype=bool)
            for mask, _ in masks:
                if mask is not None:
                    union |= mask
            Image.fromarray(union.astype(np.uint8) * 255).save(tmp / "mask" / f"{idx:05d}.png")
            progress(0.8 * (idx + 1) / total)

        progress(0.85, "Codificando vídeos")
        audio = paths.clip if project["has_audio"] else None
        media.encode_frames(tmp / "overlay", fps, tmp / "preview.mp4", audio_from=audio)
        media.encode_frames(tmp / "mask", fps, tmp / "mask.mp4", pattern="%05d.png", crf=12)
        with zipfile.ZipFile(tmp / "masks.zip", "w", compression=zipfile.ZIP_STORED) as zf:
            for obj in obj_ids:
                for png in sorted((paths.masks / f"obj{obj}").glob("*.png")):
                    zf.write(png, f"obj{obj}/{png.name}")
        for name in EXPORT_FILES:
            os.replace(tmp / name, paths.exports / name)
    return f"Exportados {total} quadros."
