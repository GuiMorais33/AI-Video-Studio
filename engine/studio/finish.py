"""Vídeo final: a pessoa gerada colada sobre o vídeo ORIGINAL, em resolução cheia.

O Wan trabalha num clipe reduzido (ex.: 464x832). Antes, o final era esse vídeo inteiro esticado até
1080x1920, inclusive o fundo, que nem tinha sido mexido. Agora só a área do personagem (a máscara da
geração, com borda suave) vem do vídeo do Wan ou do refino do rosto. O resto vem do vídeo enviado
(source.*), nos mesmos quadros e com o mesmo enquadramento do pacote:

- os quadros: o mesmo corte (-ss/-t) e o mesmo fps do clipe, e depois os índices do pacote;
- o enquadramento: o mesmo recorte do pacote (wan_geometry), levado para o tamanho do original;
- as cores: lidas como o clipe chegou ao Wan (media.clip_colors: pela marcação do vídeo, ou BT.601
  nos pacotes de projetos antigos).

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
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from . import media, rife
from .masks import COMPOSITE_FEATHER, dilate, generation_mask

INFO_FILE = "final.json"
# Diferença média (0-255) aceita entre o original, levado ao enquadramento do pacote, e o vídeo do
# pacote. Os dois passam por caminhos diferentes (JPEG e redução do clipe x leitura direta), o que dá
# 1 a 4 em vídeo real; um quadro trocado num vídeo com movimento, ou outra cor, passa bem disso.
MATCH_TOLERANCE = 8.0
LANCZOS = Image.Resampling.LANCZOS
# Borda: a máscara da geração é a da pessoa alargada (~9 px) e em blocos de 16 px. Nessa faixa em volta
# da pessoa, o Wan redesenha o fundo (sem vê-lo) em 480p: colado inteiro sobre o original nítido, vira
# uma "aura" mais mole. Ali, onde o vídeo do Wan ficou parecido com o original (diferença média entre
# DIFF_LOW e DIFF_HIGH, 0-255, depois de um desfoque leve), entra o original; onde difere (a pessoa
# gerada, cabelo e roupa além da silhueta), fica o Wan. Na silhueta da pessoa original (+ CORE_MARGIN
# px) fica sempre o Wan: ela nunca pode aparecer.
DIFF_LOW = 10.0
DIFF_HIGH = 28.0
DIFF_BLUR = 2
CORE_MARGIN = 4


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


def soft_mask(mask: np.ndarray) -> np.ndarray:
    """Máscara da geração com borda suave (como o composite_background do notebook), no tamanho da
    geração, uint8 0..255."""
    img = Image.fromarray(generation_mask(mask).astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(COMPOSITE_FEATHER)))


def core_mask(mask: np.ndarray) -> np.ndarray:
    """Silhueta da pessoa original com folga de CORE_MARGIN px e borda suave (uint8, tamanho da geração)."""
    grown = dilate(mask, 2 * CORE_MARGIN + 1, 1)
    return np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(CORE_MARGIN / 2)))


def changed_weight(layer_gen: np.ndarray, original_gen: np.ndarray) -> np.ndarray:
    """0..1 (tamanho da geração): quanto o vídeo do Wan difere do original em cada ponto (0 = igual)."""
    diff = np.abs(layer_gen.astype(np.int16) - original_gen.astype(np.int16)).mean(axis=2)
    # x4 para não perder precisão no uint8 do desfoque.
    blurred = Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(DIFF_BLUR))
    level = np.asarray(blurred, np.float32) / 4
    return np.clip((level - DIFF_LOW) / (DIFF_HIGH - DIFF_LOW), 0, 1)


def blend_mask(generated: np.ndarray, core: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Onde vale o vídeo do Wan (uint8): a silhueta original sempre; o resto da área da geração na
    proporção do quanto ele difere do original."""
    out = np.maximum(core.astype(np.float32), generated.astype(np.float32) * weight)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def to_alpha(soft: np.ndarray, size: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Máscara suave (uint8, tamanho da geração) levada para `size`, 0..1 (altura x largura x 1)."""
    resized = Image.fromarray(soft).resize(size, Image.Resampling.BILINEAR, box=box)
    return np.asarray(resized, np.float32)[..., None] / 255.0


def soft_alpha(mask_png: bytes, size: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Máscara da geração (PNG do pacote) com borda suave, levada para `size`."""
    with Image.open(BytesIO(mask_png)) as img:
        mask = np.asarray(img.convert("L")) > 127
    return to_alpha(soft_mask(mask), size, box)


Interpolate = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def compose(project: dict[str, Any], root: Path, video: Path, output: Path, size: tuple[int, int], *,
            make_interpolator: Callable[[], Interpolate] | None = None,
            progress: Callable[[float], None] = lambda f: None) -> dict[str, Any]:
    """Grava em `output` (sem áudio, BT.709, `size`) a pessoa de `video` sobre o original.

    `video` é o resultado do Wan ou o do refino do rosto, no tamanho da geração ou ampliado (mesma
    proporção). Com `make_interpolator` (RIFE), o final sai no fps do clipe: nos quadros do clipe que o
    pacote pulou (ex.: 30 -> 16 fps), a pessoa é criada entre os dois quadros vizinhos. Levanta Skipped
    quando não dá; aí `output` não fica para trás. Erro do RIFE no meio vira rife.RifeError (quem chama
    refaz sem ele)."""
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
        gaps = any(b - a > 1 for a, b in zip(indices, indices[1:]))
        interpolate: Interpolate | None = None
        note: str | None = None
        if gaps and make_interpolator is not None:
            try:
                interpolate = make_interpolator()
            except Exception as exc:  # sem PyTorch, download falhou: o final sai no fps do pacote
                note = str(exc) or type(exc).__name__
        wanted = list(range(indices[0], indices[-1] + 1)) if interpolate else indices
        out_fps = clip_fps if interpolate else Fraction(job["fps"])

        def clip_raw(k: int) -> np.ndarray | None:
            """Máscara do SAM 2 do quadro `k` do clipe, no enquadramento da geração (bool)."""
            path = root / "masks" / "obj1" / f"{k:05d}.png"
            if not path.exists():
                return None
            with Image.open(path) as img:
                fitted = img.convert("L").resize((geo["scaled_width"], geo["scaled_height"]), Image.Resampling.NEAREST)
            box = (geo["crop_x"], geo["crop_y"], geo["crop_x"] + gen_w, geo["crop_y"] + gen_h)
            return np.asarray(fitted.crop(box)) > 127

        writer = media.FrameWriter(output, width, height, out_fps, crf=16, preset="medium", bt709=True)
        diffs: list[float] = []
        recovered: list[float] = []  # fração da área da geração que voltou a ser o original (a "aura")
        made = 0
        try:
            with media.FrameReader(
                source, src_w, src_h,
                input_args=("-ss", f"{start:.3f}", "-t", f"{float(project['duration']) + 1:.3f}"),
                vf=f"fps={media.fps_arg(clip_fps)},format=yuv420p,{media.clip_colors(job)}",
            ) as source_reader, media.FrameReader(video, info.width, info.height) as layer_reader, \
                    media.FrameReader(wan_input, gen_w, gen_h) as input_reader:

                def package_frame(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
                    """(pessoa, quadro do pacote, área da geração suave, silhueta suave) do quadro `i`."""
                    if i >= count:
                        return None
                    layer, packaged = layer_reader.read(), input_reader.read()
                    if layer is None or packaged is None:
                        raise Skipped("o vídeo acabou antes do pacote")
                    with Image.open(BytesIO(zf.read(f"masks/{i:05d}.png"))) as img:
                        mask = np.asarray(img.convert("L")) > 127
                    return np.ascontiguousarray(layer[:layer_h, :layer_w]), packaged, soft_mask(mask), core_mask(mask)

                i, current, following = 0, package_frame(0), package_frame(1)
                position, original = -1, None
                for n, k in enumerate(wanted):
                    while i + 1 < count and indices[i + 1] <= k:
                        i, current, following = i + 1, following, package_frame(i + 2)
                    while position < k:
                        original = source_reader.read()
                        position += 1
                        if original is None:
                            raise Skipped("o vídeo original acabou antes do clipe")
                    src_img = Image.fromarray(original)
                    original_gen = np.asarray(src_img.resize((gen_w, gen_h), LANCZOS, box=source_gen))
                    if k == indices[i]:
                        layer, _, soft, core = current
                        diffs.append(float(np.abs(original_gen.astype(np.int16) - current[1]).mean()))
                        if len(diffs) == 1 and diffs[0] > MATCH_TOLERANCE:
                            raise Skipped(f"o vídeo original não bate com o do pacote (diferença {diffs[0]:.1f})")
                    else:
                        t = (k - indices[i]) / (indices[i + 1] - indices[i])
                        try:
                            layer = interpolate(current[0], following[0], t)
                        except Exception as exc:
                            raise rife.RifeError(f"o RIFE falhou no meio ({type(exc).__name__}: {exc})") from exc
                        # A pessoa original nunca pode aparecer por baixo: a área vale onde ela estava
                        # nos dois vizinhos e onde o SAM 2 a viu neste quadro.
                        here = clip_raw(k)
                        masks = [current[2], following[2]] + ([soft_mask(here)] if here is not None else [])
                        soft = np.maximum.reduce(masks)
                        core = core_mask(here) if here is not None else np.maximum(current[3], following[3])
                        made += 1
                    background = np.asarray(src_img.resize(size, LANCZOS, box=source_view), np.float32)
                    person = np.asarray(Image.fromarray(layer).resize(size, LANCZOS, box=layer_view), np.float32)
                    layer_gen = layer if layer.shape[:2] == (gen_h, gen_w) else \
                        np.asarray(Image.fromarray(layer).resize((gen_w, gen_h), LANCZOS))
                    mixed = blend_mask(soft, core, changed_weight(layer_gen, original_gen))
                    area = soft > 127
                    if area.any():
                        recovered.append(float((mixed[area] < 128).mean()))
                    alpha = to_alpha(mixed, size, mask_view)
                    writer.write(np.clip(background + alpha * (person - background) + 0.5, 0, 255).astype(np.uint8))
                    progress((n + 1) / len(wanted))
            if median(diffs) > MATCH_TOLERANCE:
                raise Skipped(f"o vídeo original não bate com o do pacote (diferença {median(diffs):.1f})")
        except BaseException:
            writer.abort()
            output.unlink(missing_ok=True)
            raise
        writer.close()
    data = {
        "fundo": "original", "original": [src_w, src_h], "tamanho": [width, height],
        "diferenca_pacote": round(median(diffs), 2), "pessoa": [layer_w, layer_h],
        "fps": media.fps_arg(out_fps), "quadros": len(wanted), "interpolados": made,
        "borda_original": round(float(np.mean(recovered)), 3) if recovered else 0.0,
    }
    if gaps and interpolate is None:
        data["sem_interpolacao"] = note or "interpolação desligada"
    return data
