"""Leitura/gravação de máscaras binárias e renderização de sobreposições."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

# Cores por objeto (RGB). O objeto 1 é a pessoa principal.
OBJECT_COLORS = [(0, 190, 255), (255, 80, 160), (120, 220, 60), (255, 170, 0)]


def object_color(obj_id: int) -> tuple[int, int, int]:
    return OBJECT_COLORS[(obj_id - 1) % len(OBJECT_COLORS)]


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Grava a máscara como PNG 0/255 de forma atômica (leitores nunca veem arquivo parcial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(tmp, format="PNG")
    os.replace(tmp, path)


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with Image.open(path) as img:
        return np.asarray(img.convert("L")) > 127


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def erode(mask: np.ndarray) -> np.ndarray:
    """Erosão de 1 pixel (vizinhança 4) só com NumPy."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def contour(mask: np.ndarray, width: int = 2) -> np.ndarray:
    inner = mask
    for _ in range(width):
        inner = erode(inner)
    return mask & ~inner


def overlay(
    frame: np.ndarray,
    masks: list[tuple[np.ndarray, tuple[int, int, int]]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Pinta cada máscara sobre o quadro com transparência e contorno sólido."""
    out = frame.astype(np.float32)
    for mask, color in masks:
        if mask is None or not mask.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[mask] = out[mask] * (1 - alpha) + c * alpha
        out[contour(mask)] = c
    return out.clip(0, 255).astype(np.uint8)
