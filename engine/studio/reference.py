"""Imagem do personagem que vai para o Kaggle: o seu rosto já nela e recortada justa em volta da pessoa.

O Wan tira o rosto do personagem da imagem de referência. Se ela é um personagem do ChatGPT, os
quadros de perfil (onde a troca de rosto não entra) mostram o rosto do ChatGPT: ao virar a cabeça, a
pessoa "vira outra". E o Wan reduz a referência inteira ao tamanho da geração (ex.: 464x832), então
numa foto de corpo inteiro com muito fundo o rosto fica com poucos pixels.

Aqui, no computador, antes do pacote:
1. o FaceFusion troca o rosto da imagem pelo das fotos do rosto (o mesmo trocador e restaurador do
   refino). O Kaggle recebe só a imagem pronta, como se fosse uma foto sua de corpo inteiro: nenhuma
   ferramenta de troca de rosto roda lá;
2. com fundo liso, a imagem é recortada em volta da pessoa na proporção da geração, para ela ocupar
   o quadro todo (nunca corta a pessoa: falta de espaço vira fundo da cor da borda).

Se não der (sem fotos, FaceFusion não instalado, rosto não achado, fundo com detalhes), vai a imagem
como está, com o motivo no aviso.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from . import face_refine, wan
from .config import Settings

OUTPUT = "referencia_wan.png"
LOG_FILE = "facefusion_referencia.log"
# Fundo liso: pelo menos esta fração da borda da imagem fica a menos de PLAIN_BORDER_DIFF (0-255) da
# cor mediana dela.
PLAIN_BORDER_SHARE = 0.6
PLAIN_BORDER_DIFF = 20.0
# Diferença da cor da borda a partir da qual o ponto é da pessoa.
PERSON_THRESHOLD = 30.0
# Folga em volta da pessoa, em fração do lado maior da caixa dela.
MARGIN = 0.06
# Só recorta se a pessoa ocupa menos que isto da imagem na direção que limita a escala.
MIN_GAIN = 0.9
# Memória livre (RAM + swap) para o FaceFusion numa imagem: ele chega a 3,5 GB (medido), menos que os
# 5,4 GB de um vídeo. Sem ela, o pacote sai sem o rosto na imagem, com aviso.
IMAGE_MEMORY_GB = 4.0


def person_box(img: Image.Image) -> tuple[int, int, int, int] | str:
    """Caixa (esq, topo, dir, baixo) da pessoa numa imagem de fundo liso; ou o motivo de não achar."""
    rgb = np.asarray(img.convert("RGB"), np.float32)
    h, w = rgb.shape[:2]
    band = max(2, round(min(h, w) * 0.03))
    border = np.concatenate([rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
                             rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    # Liso: a maior parte da borda tem a cor do fundo (a pessoa pode encostar num lado da imagem).
    if float((np.abs(border - background).mean(axis=1) < PLAIN_BORDER_DIFF).mean()) < PLAIN_BORDER_SHARE:
        return "o fundo da imagem não é liso"
    diff = np.abs(rgb - background).mean(axis=2)
    blurred = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    person = np.asarray(blurred) > PERSON_THRESHOLD
    rows = np.flatnonzero(person.sum(axis=1) > max(2, 0.005 * w))
    cols = np.flatnonzero(person.sum(axis=0) > max(2, 0.005 * h))
    if len(rows) == 0 or len(cols) == 0:
        return "a pessoa não se destaca do fundo"
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
    if not 0.03 <= area <= 0.97:
        return "a pessoa não se destaca do fundo"
    return box


def tight_crop(img: Image.Image, aspect: float) -> tuple[Image.Image, str | None]:
    """Recorte em volta da pessoa na proporção `aspect` (largura/altura). Devolve (imagem, aviso)."""
    found = person_box(img)
    if isinstance(found, str):
        return img, f"sem recorte: {found}"
    x0, y0, x1, y1 = found
    pad = MARGIN * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    bw, bh = x1 - x0, y1 - y0
    # Cresce o lado curto até a proporção da geração (nunca encolhe: não corta a pessoa).
    if bw / bh < aspect:
        grow = bh * aspect - bw
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    else:
        grow = bw / aspect - bh
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    # Ganho: quanto a pessoa cresce em relação a encaixar a imagem inteira na mesma proporção.
    whole = min(img.width / aspect, img.height)  # altura útil da imagem inteira nessa proporção
    if (y1 - y0) >= MIN_GAIN * whole:
        return img, None
    rgb = np.asarray(img.convert("RGB"))
    border = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    left, top = round(x0), round(y0)
    out = Image.new("RGB", (round(x1) - left, round(y1) - top), fill)
    out.paste(img.convert("RGB"), (-left, -top))
    return out, None


def prepare(root: Path, settings: Settings, *, aspect: float, put_face: bool,
            stop: threading.Event | None = None,
            before_facefusion: Callable[[], None] = lambda: None) -> tuple[Path, dict[str, Any]]:
    """Grava exports/referencia_wan.png (a imagem que vai no pacote) e devolve (caminho, informações).

    `before_facefusion` roda logo antes de abrir o FaceFusion (o motor solta o vídeo do SAM 2)."""
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    reference = root / "reference.png"
    info: dict[str, Any] = {"rosto": False, "recorte": False, "avisos": []}
    image = Image.open(reference).convert("RGB")
    if put_face:
        photos = [root / wan.FACES_DIR / n for n in wan.list_face_photos(root)]
        status = face_refine.status(settings)
        if not photos:
            info["avisos"].append("sem fotos do rosto: o rosto da imagem do personagem foi como está")
        elif not status["installed"]:
            info["avisos"].append(f"o rosto não foi aplicado na imagem ({face_refine.unavailable_message(status)})")
        elif (need := min(settings.facefusion_min_memory_gb, IMAGE_MEMORY_GB)) > 0 and \
                (free := face_refine.free_memory_gb()) is not None and free < need:
            info["avisos"].append(f"o rosto não foi aplicado na imagem: pouca memória ({free:.1f} GB livres, o "
                                  f"FaceFusion precisa de uns {need:g} GB); feche outros programas e gere o pacote de novo")
        else:
            before_facefusion()
            choice = face_refine.face_settings(root)
            with tempfile.TemporaryDirectory(dir=root, prefix="referencia-") as tmp_name:
                tmp = Path(tmp_name)
                target, output = tmp / "referencia.png", tmp / "saida.png"
                image.save(target)
                copies = []
                for i, src in enumerate(photos, 1):
                    copies.append(tmp / f"foto{i:02d}{src.suffix}")
                    shutil.copyfile(src, copies[-1])
                run = face_refine.run_facefusion(settings, copies, target, output, tmp, exports / LOG_FILE, stop=stop,
                                                 swapper=choice["trocador"], enhancer_blend=choice["restaurador"])
                reason = face_refine.failure_reason(run, settings)
                if run.stopped:
                    raise face_refine.FaceRefineError("Pacote interrompido: o motor foi encerrado.")
                if reason is None:
                    image = Image.open(output).convert("RGB")
                    info["rosto"] = True
                else:
                    info["avisos"].append(f"o rosto não foi aplicado na imagem: {reason}")
    cropped, note = tight_crop(image, aspect)
    if note:
        info["avisos"].append(note)
    info["recorte"] = cropped is not image
    info["tamanho"] = list(cropped.size)
    path = exports / OUTPUT
    tmp_out = path.with_name(path.name + ".tmp.png")
    cropped.save(tmp_out)
    tmp_out.replace(path)
    return path, info
