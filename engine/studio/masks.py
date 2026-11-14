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


def anchor_points(
    mask: np.ndarray,
    k: int = 3,
    avoid: list[list[float]] | None = None,
    avoid_ratio: float = 0.15,
) -> list[list[float]]:
    """Até `k` pontos (x, y) bem no miolo da máscara, espalhados e longe dos pontos em `avoid`.

    A "profundidade" de cada pixel é a distância até a borda, medida por erosões
    sucessivas. Pontos a menos de `avoid_ratio` da diagonal da máscara de algum
    ponto de `avoid` são descartados.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    depth = np.zeros(mask.shape, dtype=np.int32)
    current, level = mask.copy(), 1
    while current.any() and level < 256:
        depth[current] = level
        current = erode(current)
        level += 1
    ys, xs = np.nonzero(mask)
    radius = avoid_ratio * float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for ax, ay in avoid or []:
        depth[(xx - ax) ** 2 + (yy - ay) ** 2 <= radius ** 2] = 0
    points: list[list[float]] = []
    for _ in range(k):
        idx = int(np.argmax(depth))
        level = int(depth.flat[idx])
        if level <= 0:
            break
        y, x = divmod(idx, mask.shape[1])
        points.append([float(x), float(y)])
        spread = max(8, 2 * level)
        depth[(xx - x) ** 2 + (yy - y) ** 2 <= spread ** 2] = 0
    return points


# --------------------------------------------------------------------------- máscara da geração
# A mesma do notebook do Kaggle: dilatação 7x7 x3 do pré-processamento oficial do Wan e blocos de
# 16 px do template ComfyUI. O personagem gerado só vale dentro dela.
GEN_MASK_KERNEL = 7
GEN_MASK_ITERATIONS = 3
GEN_MASK_BLOCK = 16
# Borda suave da colagem do personagem sobre o fundo (o composite_feather do notebook).
COMPOSITE_FEATHER = 6


def dilate(mask: np.ndarray, kernel: int, iterations: int) -> np.ndarray:
    """Igual a cv2.dilate(mask, ones((k, k)), iterations): quadrado de lado (k-1)*it+1."""
    radius = (kernel - 1) // 2 * iterations
    out = mask.astype(bool)
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in (0, 1)])
        acc = np.zeros_like(out)
        size = out.shape[axis]
        for shift in range(2 * radius + 1):
            acc |= padded[shift:shift + size] if axis == 0 else padded[:, shift:shift + size]
        out = acc
    return out


def blockify(mask: np.ndarray, block: int) -> np.ndarray:
    """Cada bloco block x block com algum pixel da máscara vira bloco cheio (contorno em degraus)."""
    if block <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    ph, pw = -h % block, -w % block
    padded = np.pad(mask.astype(bool), ((0, ph), (0, pw)))
    grid = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block).any(axis=(1, 3))
    return np.kron(grid, np.ones((block, block), bool))[:h, :w]


def generation_mask(mask: np.ndarray) -> np.ndarray:
    """Área que o Wan gera, a partir da máscara do SAM 2 no tamanho da geração (bool)."""
    return blockify(dilate(mask, GEN_MASK_KERNEL, GEN_MASK_ITERATIONS), GEN_MASK_BLOCK)
