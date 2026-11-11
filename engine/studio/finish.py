"""Vídeo final: a pessoa gerada colada sobre o vídeo ORIGINAL, em resolução cheia.

O Wan trabalha num clipe reduzido (ex.: 464x832). Antes, o final era esse vídeo inteiro esticado até
1080x1920, inclusive o fundo, que nem tinha sido mexido. Agora só a área do personagem (a máscara da
geração, com borda suave) vem do vídeo do Wan ou do refino do rosto. O resto vem do vídeo enviado
(source.*), nos mesmos quadros e com o mesmo enquadramento do pacote:

- os quadros: o mesmo corte (-ss/-t) e o mesmo fps do clipe, e depois os índices do pacote;
- o enquadramento: o mesmo recorte do pacote (wan_geometry), levado para o tamanho do original;
- as cores: lidas como o clipe chega ao Wan (media.CLIP_COLORS).

Antes de gravar, cada quadro do original é conferido contra o vídeo do pacote (wan_input.mp4). Se não
bater (quadro trocado, enquadramento ou cor diferentes), o final volta ao jeito antigo e o motivo fica
em exports/final.json.
"""

from __future__ import annotations

import json
import os
import zipfile
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from . import media
from .masks import COMPOSITE_FEATHER, generation_mask

INFO_FILE = "final.json"
# Diferença média (0-255) aceita entre o original, levado ao enquadramento do pacote, e o vídeo do
# pacote. Os dois passam por caminhos diferentes (JPEG e redução do clipe x leitura direta), o que dá
# 1 a 4 em vídeo real; um quadro trocado num vídeo com movimento, ou outra cor, passa bem disso.
MATCH_TOLERANCE = 8.0
LANCZOS = Image.Resampling.LANCZOS


class Skipped(Exception):
    """O final não pode sair sobre o original; a mensagem diz por quê (vai para final.json)."""


def find_source(root: Path) -> Path | None:
    """O vídeo enviado, guardado na criação do projeto como source.<extensão>."""
    found = sorted(p for p in root.glob("source*") if p.is_file() and p.stem == "source")
    return found[0] if found else None


def read_info(root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((root / "exports" / INFO_FILE).read_text())
    except (OSError, ValueError):
        return None


def write_info(root: Path, data: dict[str, Any]) -> None:
    path = root / "exports" / INFO_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def soft_alpha(mask_png: bytes, size: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Máscara da geração com borda suave (como o composite_background do notebook), levada para `size`."""
    with Image.open(BytesIO(mask_png)) as img:
        mask = np.asarray(img.convert("L")) > 127
    soft = Image.fromarray(generation_mask(mask).astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(COMPOSITE_FEATHER))
    return np.asarray(soft.resize(size, Image.Resampling.BILINEAR, box=box), np.float32)[..., None] / 255.0


def compose(project: dict[str, Any], root: Path, video: Path, output: Path, size: tuple[int, int]) -> dict[str, Any]:
    """Grava em `output` (sem áudio, BT.709, `size`) a pessoa de `video` sobre o original.

    `video` é o resultado do Wan ou o do refino do rosto, no tamanho da geração ou ampliado (mesma
    proporção). Levanta Skipped quando não dá; aí `output` não fica para trás."""
    exports = root / "exports"
    source = find_source(root)
    if source is None:
        raise Skipped("o vídeo original não está mais no projeto")
    package, wan_input = exports / "wan_package.zip", exports / "wan_input.mp4"
    if not package.exists() or not wan_input.exists():
        raise Skipped("falta o pacote do Wan (gere o pacote de novo)")

    with zipfile.ZipFile(package) as zf:
        job = json.loads(zf.read("job.json"))
        gen_w, gen_h, count = int(job["width"]), int(job["height"]), int(job["num_frames"])
        geo, indices = job["geometry"], [int(i) for i in job["source_frame_indices"]]
        if job["source"]["width"] != project["width"] or job["source"]["height"] != project["height"]:
            raise Skipped("o pacote é de outro clipe")
        info = media.probe(video)
        frames = media.count_frames(video)
        if frames != count:
            raise Skipped(f"o vídeo tem {frames} quadros e o pacote, {count}")
        layer_w, layer_h = info.width - info.width % 2, info.height - info.height % 2
        if abs(layer_w / layer_h - gen_w / gen_h) > 0.02:
            raise Skipped(f"o vídeo ({layer_w}x{layer_h}) não tem a proporção do pacote ({gen_w}x{gen_h})")

        start = float(project["clip_start"])
        src_w, src_h = media.first_frame_size(source, start)
        width, height = size
        # Parte do quadro da geração que o final mostra (cobre o tamanho final, corte central).
        scale = max(width / gen_w, height / gen_h)
        view_w, view_h = width / scale, height / scale
        x0, y0 = (gen_w - view_w) / 2, (gen_h - view_h) / 2

        def in_source(x: float, y: float) -> tuple[float, float]:
            # geração -> clipe (escala + recorte do pacote) -> original (o clipe é o original reduzido)
            return ((x + geo["crop_x"]) * src_w / geo["scaled_width"], (y + geo["crop_y"]) * src_h / geo["scaled_height"])

        source_view = (*in_source(x0, y0), *in_source(x0 + view_w, y0 + view_h))
        source_gen = (*in_source(0, 0), *in_source(gen_w, gen_h))
        layer_view = (x0 * layer_w / gen_w, y0 * layer_h / gen_h,
                      (x0 + view_w) * layer_w / gen_w, (y0 + view_h) * layer_h / gen_h)
        mask_view = (x0, y0, x0 + view_w, y0 + view_h)

        clip_fps = Fraction(project["fps"])
        writer = media.FrameWriter(output, width, height, Fraction(job["fps"]), crf=16, preset="medium", bt709=True)
        diffs: list[float] = []
        try:
            with media.FrameReader(
                source, src_w, src_h,
                input_args=("-ss", f"{start:.3f}", "-t", f"{float(project['duration']) + 1:.3f}"),
                vf=f"fps={media.fps_arg(clip_fps)},format=yuv420p,{media.CLIP_COLORS}",
            ) as source_reader, media.FrameReader(video, info.width, info.height) as layer_reader, \
                    media.FrameReader(wan_input, gen_w, gen_h) as input_reader:
                position, original = -1, None
                for i, wanted in enumerate(indices):
                    while position < wanted:
                        original = source_reader.read()
                        position += 1
                        if original is None:
                            raise Skipped("o vídeo original acabou antes do clipe")
                    layer, packaged = layer_reader.read(), input_reader.read()
                    if layer is None or packaged is None:
                        raise Skipped("o vídeo acabou antes do pacote")
                    src_img = Image.fromarray(original)
                    check = np.asarray(src_img.resize((gen_w, gen_h), LANCZOS, box=source_gen), np.int16)
                    diffs.append(float(np.abs(check - packaged).mean()))
                    if i == 0 and diffs[0] > MATCH_TOLERANCE:
                        raise Skipped(f"o vídeo original não bate com o do pacote (diferença {diffs[0]:.1f})")
                    background = np.asarray(src_img.resize(size, LANCZOS, box=source_view), np.float32)
                    person = np.asarray(Image.fromarray(layer[:layer_h, :layer_w]).resize(size, LANCZOS, box=layer_view),
                                        np.float32)
                    alpha = soft_alpha(zf.read(f"masks/{i:05d}.png"), size, mask_view)
                    writer.write(np.clip(background + alpha * (person - background) + 0.5, 0, 255).astype(np.uint8))
            if median(diffs) > MATCH_TOLERANCE:
                raise Skipped(f"o vídeo original não bate com o do pacote (diferença {median(diffs):.1f})")
        except BaseException:
            writer.abort()
            output.unlink(missing_ok=True)
            raise
        writer.close()
    return {
        "fundo": "original", "original": [src_w, src_h], "tamanho": [width, height],
        "diferenca_pacote": round(median(diffs), 2), "pessoa": [layer_w, layer_h],
    }
