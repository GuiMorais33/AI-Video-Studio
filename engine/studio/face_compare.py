"""Comparar ajustes do refino do rosto num trecho curto, lado a lado, antes do refino completo.

Que trocador (hyperswap 1a, 1b ou 1c) e que força do restaurador (GFPGAN) deixam o rosto mais natural
depende do rosto e do vídeo: as fontes divergem e o restaurador forte tende a deixar a pele lisa
demais. Em vez de chutar, o estúdio:

1. acha um trecho de ~1,5 s em que o rosto da pessoa rastreada está de frente (o mesmo medidor do
   refino, facefusion_probe.py);
2. roda o FaceFusion nesse trecho só com a troca, para cada trocador, e depois só o restaurador (força
   100) sobre a troca. Como o FaceFusion mistura o restaurador linearmente
   (saída = troca + força x (restaurado - troca)), qualquer força sai desses dois vídeos, sem rodar
   de novo;
3. monta um vídeo em grade, ampliado no rosto e com rótulos: sem refino, o ajuste atual e as variações.

A escolha fica em rosto_ajustes.json (face_refine.save_face_settings) e vale para o próximo refino.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import face_refine, media
from .config import Settings
from .face_refine import FaceRefineError

RESULT_FILE = "comparar_rosto.json"
VIDEO_FILE = "comparar_rosto.mp4"
IMAGE_FILE = "comparar_rosto.png"
LOG_FILE = "facefusion_comparar.log"
STOPPED = 'Comparação interrompida: o motor foi encerrado. Use "Comparar ajustes do rosto" de novo.'
# ~1,5 s a 16 fps: dá para ver o rosto se mexer sem gastar o tempo de um refino inteiro.
EXCERPT_FRAMES = 24
# Forças do restaurador testadas em cada trocador (além do ajuste atual, sempre incluído).
BLENDS = (40, 0)
CELL = 360
COLUMNS = 4
LABEL_HEIGHT = 34
# Lado do quadrado ampliado em volta do rosto, em relação ao maior lado da caixa do rosto.
ZOOM = 1.9
SWAPPER_NAMES = {"hyperswap_1a_256": "1a", "hyperswap_1b_256": "1b", "hyperswap_1c_256": "1c"}

ProgressFn = Callable[..., None]


def read_result(root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((root / "exports" / RESULT_FILE).read_text())
    except (OSError, ValueError):
        return None


def label(swapper: str | None, blend: int | None) -> str:
    if swapper is None:
        return "Sem refino (rosto do Wan)"
    name = f"hyperswap {SWAPPER_NAMES.get(swapper, swapper)}"
    return f"{name}, sem restaurador" if blend == 0 else f"{name}, restaurador {blend}%"


def variants(current: dict[str, Any]) -> list[dict[str, Any]]:
    """Sem refino, o ajuste atual e, para cada trocador, as forças de BLENDS."""
    combos = [(current["trocador"], current["restaurador"])]
    for swapper in face_refine.SWAPPERS:
        combos += [(swapper, blend) for blend in BLENDS if (swapper, blend) not in combos]
    out = [{"id": "sem", "trocador": None, "restaurador": None, "rotulo": label(None, None), "atual": False}]
    for i, (swapper, blend) in enumerate(combos):
        out.append({"id": f"{swapper}@{blend}", "trocador": swapper, "restaurador": blend,
                    "rotulo": label(swapper, blend) + (" (atual)" if i == 0 else ""), "atual": i == 0})
    return out


def choose_excerpt(weights: list[float], length: int) -> int | None:
    """Início do trecho de `length` quadros com mais rosto de frente; None se não houver nenhum."""
    if not any(weights):
        return None
    length = min(length, len(weights))
    sums = np.convolve(np.asarray(weights, float), np.ones(length), "valid")
    return int(np.argmax(sums))


def zoom_box(boxes: list[list[float]], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Quadrado em volta de todas as caixas do rosto no trecho (ZOOM vezes o maior lado), dentro do quadro."""
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    side = min(max(x1 - x0, y1 - y0) * ZOOM, *size)
    cx = min(max((x0 + x1) / 2, side / 2), size[0] - side / 2)
    cy = min(max((y0 + y1) / 2, side / 2), size[1] - side / 2)
    return round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fonte fixa pequena
        return ImageFont.load_default()


def grid_frame(cells: list[np.ndarray], labels: list[str], columns: int) -> np.ndarray:
    """Células (CELL x CELL) em grade, cada uma com o rótulo numa faixa escura em cima."""
    rows = math.ceil(len(cells) / columns)
    canvas = Image.new("RGB", (columns * CELL, rows * (CELL + LABEL_HEIGHT)), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font = _font(19)
    for i, (cell, text) in enumerate(zip(cells, labels)):
        x, y = (i % columns) * CELL, (i // columns) * (CELL + LABEL_HEIGHT)
        draw.text((x + 8, y + 7), text, fill=(235, 235, 235), font=font)
        canvas.paste(Image.fromarray(cell), (x, y + LABEL_HEIGHT))
    width, height = canvas.size
    return np.asarray(canvas)[: height - height % 2, : width - width % 2]


def _trim(src: Path, dst: Path, start: int, count: int) -> None:
    media.run([media.ffmpeg_bin(), "-y", "-v", "error", "-i", str(src), "-vf",
               f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)])


def compare(project: dict[str, Any], root: Path, settings: Settings, progress: ProgressFn = lambda *a: None,
            stop: threading.Event | None = None) -> str:
    """Roda as variações num trecho curto e grava comparar_rosto.mp4/.png/.json em exports/."""
    try:
        return _compare(root, settings, progress, stop)
    except FaceRefineError as exc:
        if str(exc) == face_refine.STOPPED:  # as partes reaproveitadas do refino falam em "refino"
            raise FaceRefineError(STOPPED) from None
        raise


def _compare(root: Path, settings: Settings, progress: ProgressFn, stop: threading.Event | None) -> str:
    exports = root / "exports"
    result = exports / "wan_result.mp4"
    if not result.exists():
        raise FaceRefineError("Importe ou gere o resultado do Wan antes de comparar os ajustes do rosto.")
    progress(0.0, "Conferindo o FaceFusion")
    info = face_refine.status(settings, deep=True)
    if not info["installed"]:
        raise FaceRefineError(face_refine.unavailable_message(info))
    sources = face_refine.face_sources(root)
    if not sources:
        raise FaceRefineError("Envie a imagem do personagem ou fotos do rosto antes de comparar os ajustes.")
    need = settings.facefusion_min_memory_gb
    free = face_refine.free_memory_gb() if need > 0 else None
    if free is not None and free < need:
        raise FaceRefineError(face_refine.low_memory_message(free, need))
    started = time.monotonic()
    vinfo = media.probe(result)
    size = (vinfo.width - vinfo.width % 2, vinfo.height - vinfo.height % 2)
    frames = media.count_frames(result)
    try:
        masks = face_refine.PackageMasks(exports / "wan_package.zip", face_refine.work_size(size))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise FaceRefineError("A comparação precisa das máscaras do pacote do Wan: gere o pacote de novo.") from exc
    current = face_refine.face_settings(root)
    options = variants(current)
    swappers = list(dict.fromkeys(v["trocador"] for v in options if v["trocador"]))
    log_path = exports / LOG_FILE
    try:
        if masks.count != frames:
            raise FaceRefineError(f"O vídeo tem {frames} quadros e as máscaras do pacote, {masks.count}: gere o pacote de novo.")
        with tempfile.TemporaryDirectory(dir=root, prefix="comparar-") as tmp_name:
            tmp = Path(tmp_name)
            photos = tmp / "fotos"
            photos.mkdir()
            copies = []
            for i, src in enumerate(sources, 1):
                copies.append(photos / f"{i:02d}{src.suffix}")
                shutil.copyfile(src, copies[-1])

            progress(0.01, "Separando o personagem do fundo")
            full = tmp / "alvo.mp4"
            face_refine.write_target(result, masks, full, vinfo, size, lambda f: progress(0.01 + 0.03 * f), stop)
            progress(0.04, "Procurando um trecho com o rosto de frente")
            faces = face_refine.probe_faces(settings, full, tmp, stop)
            if faces is None:
                raise FaceRefineError("Não deu para medir o rosto no vídeo (veja se o FaceFusion está instalado inteiro).")
            person = lambda i: masks.person(i)[..., 0] > 0.5  # noqa: E731
            weights = face_refine.pose_weights(faces, person, frames)
            start = choose_excerpt(weights, EXCERPT_FRAMES)
            if start is None:
                raise FaceRefineError("O rosto aparece de perfil ou de costas no vídeo todo: não há o que comparar "
                                      "(o refino só entra com o rosto de frente até três quartos).")
            count = min(EXCERPT_FRAMES, frames)
            boxes = [face["box"] for t in range(start, start + count)
                     if (face := face_refine.tracked_face(faces[t] if t < len(faces) else [], person(t))) is not None]
            box = zoom_box(boxes, masks.size)
            excerpt = tmp / "trecho.mp4"
            _trim(full, excerpt, start, count)

            runs = 2 * len(swappers)
            outputs: dict[str, tuple[Path, Path]] = {}  # trocador -> (só a troca, troca + restaurador 100)
            for k, swapper in enumerate(swappers):
                name = SWAPPER_NAMES.get(swapper, swapper)
                swapped, enhanced = tmp / f"{name}-troca.mp4", tmp / f"{name}-restaurado.mp4"
                for step, (what, target, out, opts) in enumerate((
                        ("troca do rosto", excerpt, swapped, {"swapper": swapper, "enhancer_blend": 0}),
                        ("restaurador", swapped, enhanced, {"processors": ("face_enhancer",), "enhancer_blend": 100}))):
                    done = 2 * k + step
                    base_pct, share = 0.08 + 0.84 * done / runs, 0.84 / runs
                    title = f"Testando o hyperswap {name} ({what}, {done + 1} de {runs})"
                    progress(base_pct, title)

                    def on_stage(stage: str, pct: int, base_pct=base_pct, share=share, title=title, name=name) -> None:
                        if stage == "downloading":
                            progress(base_pct, f"Baixando o modelo do hyperswap {name} (só na 1ª vez): {pct}%")
                        elif stage == "processing":
                            progress(base_pct + share * pct / 100, f"{title}: {pct}%")

                    run = face_refine.run_facefusion(settings, copies, target, out, tmp, log_path, on_stage, stop, **opts)
                    if run.stopped:
                        raise FaceRefineError(STOPPED)
                    reason = face_refine.failure_reason(run, settings)
                    if reason is not None:
                        raise FaceRefineError(f"A comparação parou no hyperswap {name} ({what}): {reason}.")
                outputs[swapper] = (swapped, enhanced)

            progress(0.93, "Montando a comparação lado a lado")
            columns = COLUMNS
            labels = [v["rotulo"] for v in options]
            writer = media.FrameWriter(tmp / VIDEO_FILE, *grid_size(len(options), columns), vinfo.fps, crf=16)
            readers: list[media.FrameReader] = []
            try:
                base_reader = media.FrameReader(result, vinfo.width, vinfo.height)
                readers.append(base_reader)
                pairs = {}
                for swapper, (swapped, enhanced) in outputs.items():
                    pairs[swapper] = (media.FrameReader(swapped, *masks.size), media.FrameReader(enhanced, *masks.size))
                    readers += list(pairs[swapper])
                for _ in range(start):
                    base_reader.read()
                middle = None
                for t in range(count):
                    _check(stop)
                    raw = base_reader.read()
                    if raw is None:
                        raise FaceRefineError("O vídeo do Wan acabou antes do trecho.")
                    base = face_refine._resized(face_refine._even(raw, *size), masks.size)
                    inside = masks.binary(start + t)[..., None]
                    got = {}
                    for swapper, (r_swap, r_enh) in pairs.items():
                        swapped, enhanced = r_swap.read(), r_enh.read()
                        if swapped is None or enhanced is None:
                            raise FaceRefineError("O FaceFusion devolveu menos quadros que o trecho.")
                        got[swapper] = (swapped.astype(np.float32), enhanced.astype(np.float32))
                    cells = []
                    for v in options:
                        if v["trocador"] is None:
                            frame = base
                        else:
                            swapped, enhanced = got[v["trocador"]]
                            mixed = np.clip(swapped + v["restaurador"] / 100 * (enhanced - swapped) + 0.5, 0, 255)
                            frame = np.where(inside, mixed.astype(np.uint8), base)
                        crop = Image.fromarray(frame).crop(box).resize((CELL, CELL), Image.Resampling.LANCZOS)
                        cells.append(np.asarray(crop))
                    grid = grid_frame(cells, labels, columns)
                    writer.write(grid)
                    if t == count // 2:
                        middle = grid
                    progress(0.93 + 0.06 * (t + 1) / count)
            except BaseException:
                writer.abort()
                raise
            finally:
                for reader in readers:
                    reader.close()
            writer.close()
            Image.fromarray(middle).save(tmp / IMAGE_FILE)
            os.replace(tmp / VIDEO_FILE, exports / VIDEO_FILE)
            os.replace(tmp / IMAGE_FILE, exports / IMAGE_FILE)
    finally:
        masks.close()

    seconds = round(time.monotonic() - started, 1)
    data = {
        "variantes": options, "colunas": columns, "trecho": {"inicio": start, "quadros": count,
                                                             "segundos": round(count / float(vinfo.fps), 2)},
        "fotos": len(sources), "segundos": seconds,
        "concluido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp_json = exports / (RESULT_FILE + ".tmp")
    tmp_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp_json, exports / RESULT_FILE)
    minutes = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"
    return (f"Comparação pronta em {minutes}: {len(options)} versões do rosto num trecho de "
            f"{data['trecho']['segundos']:g} s. Escolha a mais natural e refaça o refino do rosto.")


def grid_size(cells: int, columns: int) -> tuple[int, int]:
    rows = math.ceil(cells / columns)
    width, height = columns * CELL, rows * (CELL + LABEL_HEIGHT)
    return width - width % 2, height - height % 2


def _check(stop: threading.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise FaceRefineError(STOPPED)
