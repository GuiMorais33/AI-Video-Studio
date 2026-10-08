"""Pacote de entrada do Wan 2.2 Animate (modo replacement) e importação do resultado.

O pacote leva para a GPU remota (Kaggle) tudo que a geração precisa e que já
temos localmente: o vídeo no tamanho do modelo (múltiplos de 16), a máscara da
pessoa quadro a quadro (do SAM 2.1, já corrigida), a imagem de referência do
personagem e os parâmetros. Pose e rosto são extraídos lá, na GPU.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from . import __version__, media
from .masks import load_mask

PACKAGE_FORMAT = 1
# Área em pixels de cada resolução de geração (mesma regra "por área" do pré-processamento oficial).
RESOLUTIONS = {"480p": 832 * 480, "720p": 1280 * 720}
WAN_FILES = ("wan_package.zip", "wan_input.mp4", "wan_result.mp4", "wan_final.mp4", "comparativo.mp4")
MAX_REFERENCE_BYTES = 25 * 1024 * 1024


class WanError(ValueError):
    pass


@dataclass(frozen=True)
class Geometry:
    """Como o quadro do clipe vira o quadro do Wan: escala e depois recorte central."""

    width: int
    height: int
    scaled_width: int
    scaled_height: int
    crop_x: int
    crop_y: int


def wan_geometry(src_w: int, src_h: int, area: int, divisor: int = 16) -> Geometry:
    """Maior tamanho múltiplo de `divisor` com área <= `area` na proporção do clipe.

    Igual ao pré-processamento oficial (resize_by_area), mas em vez de faixas pretas
    usa escala + recorte central, que perde no máximo alguns pixels nas bordas.
    """
    ratio = src_w / src_h
    ideal_h = math.sqrt(area / ratio)
    width = max(divisor, int(ideal_h * ratio // divisor) * divisor)
    height = max(divisor, int(ideal_h // divisor) * divisor)
    scale = max(width / src_w, height / src_h)
    scaled_w = max(width, round(src_w * scale))
    scaled_h = max(height, round(src_h * scale))
    return Geometry(width, height, scaled_w, scaled_h, (scaled_w - width) // 2, (scaled_h - height) // 2)


def frame_indices(num_frames: int, src_fps: Fraction, dst_fps: Fraction) -> list[int]:
    """Quadros do clipe usados em `dst_fps` (mesma regra de get_frame_indices do Wan)."""
    if dst_fps >= src_fps:
        return list(range(num_frames))
    count = max(1, int(num_frames / src_fps * dst_fps))
    return [min(num_frames - 1, round(i / dst_fps * src_fps)) for i in range(count)]


def _fit(img: Image.Image, geo: Geometry, resample: Image.Resampling) -> Image.Image:
    img = img.resize((geo.scaled_width, geo.scaled_height), resample)
    return img.crop((geo.crop_x, geo.crop_y, geo.crop_x + geo.width, geo.crop_y + geo.height))


def save_reference(upload: Path, dest: Path) -> tuple[int, int]:
    """Valida e normaliza a imagem de referência (PNG RGB; transparência vira fundo branco)."""
    if upload.stat().st_size > MAX_REFERENCE_BYTES:
        raise WanError("Imagem de referência muito grande (máximo 25 MB).")
    try:
        with Image.open(upload) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                img = Image.new("RGB", rgba.size, (255, 255, 255))
                img.paste(rgba, mask=rgba.getchannel("A"))
            else:
                img = img.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise WanError("O arquivo enviado não é uma imagem válida.") from exc
    if min(img.size) < 256:
        raise WanError("Imagem de referência muito pequena (mínimo 256 px no menor lado).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, dest)
    return img.size


def build_package(
    project: dict[str, Any],
    root: Path,
    *,
    resolution: str = "480p",
    fps: float | None = None,
    prompt: str = "",
    progress: Callable[..., None] = lambda *a: None,
) -> str:
    """Gera exports/wan_package.zip (vídeo + máscaras + referência + job.json) e exports/wan_input.mp4."""
    if resolution not in RESOLUTIONS:
        raise WanError(f"Resolução inválida: {resolution}. Opções: {', '.join(RESOLUTIONS)}.")
    frames_dir, masks_dir, exports = root / "frames", root / "masks" / "obj1", root / "exports"
    reference = root / "reference.png"
    if not reference.exists():
        raise WanError("Envie a imagem de referência do personagem antes de gerar o pacote.")
    if project["tracked_revision"] is None:
        raise WanError("Rastreie a pessoa antes de gerar o pacote.")
    if project["tracked_revision"] != project["revision"]:
        raise WanError("A seleção mudou desde o último rastreamento. Rastreie de novo antes de gerar o pacote.")
    total = project["num_frames"]
    missing = [i for i in range(total) if not (masks_dir / f"{i:05d}.png").exists()]
    if missing:
        raise WanError(f"Faltam máscaras em {len(missing)} quadro(s). Rastreie a pessoa de novo.")

    src_fps = Fraction(project["fps"])
    dst_fps = src_fps if not fps or Fraction(fps).limit_denominator(1001) >= src_fps else Fraction(fps).limit_denominator(1001)
    indices = frame_indices(total, src_fps, dst_fps)
    geo = wan_geometry(project["width"], project["height"], RESOLUTIONS[resolution])

    exports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix="wan-") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "frames").mkdir()
        (tmp / "masks").mkdir()
        empty = 0
        for out_idx, src_idx in enumerate(indices):
            with Image.open(frames_dir / f"{src_idx:05d}.jpg") as frame:
                _fit(frame.convert("RGB"), geo, Image.Resampling.LANCZOS).save(
                    tmp / "frames" / f"{out_idx:05d}.png"
                )
            mask = load_mask(masks_dir / f"{src_idx:05d}.png")
            mask_img = _fit(Image.fromarray(mask.astype(np.uint8) * 255), geo, Image.Resampling.NEAREST)
            mask_img.save(tmp / "masks" / f"{out_idx:05d}.png")
            empty += not np.asarray(mask_img).any()
            progress(0.8 * (out_idx + 1) / len(indices))
        if empty == len(indices):
            raise WanError("A máscara está vazia em todos os quadros. Selecione a pessoa e rastreie de novo.")

        progress(0.85, "Codificando o vídeo do pacote")
        media.encode_frames(tmp / "frames", dst_fps, tmp / "video.mp4", pattern="%05d.png", crf=12)
        job = {
            "format": PACKAGE_FORMAT,
            "studio_version": __version__,
            "project_id": project["id"],
            "project_name": project["name"],
            "mode": "replace",
            "resolution": resolution,
            "width": geo.width,
            "height": geo.height,
            "fps": media.fps_arg(dst_fps),
            "num_frames": len(indices),
            "source_frame_indices": indices,
            "source": {"width": project["width"], "height": project["height"], "fps": project["fps"]},
            "geometry": asdict(geo),
            "prompt": prompt.strip(),
            "empty_mask_frames": empty,
            "files": {"video": "video.mp4", "masks": "masks/%05d.png", "reference": "reference.png"},
        }
        package = tmp / "wan_package.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp / "video.mp4", "video.mp4", compress_type=zipfile.ZIP_STORED)
            zf.write(reference, "reference.png", compress_type=zipfile.ZIP_STORED)
            for png in sorted((tmp / "masks").glob("*.png")):
                zf.write(png, f"masks/{png.name}")
            zf.writestr("job.json", json.dumps(job, indent=2, ensure_ascii=False))
        os.replace(tmp / "video.mp4", exports / "wan_input.mp4")
        os.replace(package, exports / "wan_package.zip")
    return f"Pacote pronto: {len(indices)} quadros {geo.width}x{geo.height} @ {float(dst_fps):.2f} fps."


def import_result(project: dict[str, Any], root: Path, upload: Path) -> str:
    """Recebe o vídeo gerado no Kaggle, junta o áudio original e monta o comparativo lado a lado."""
    exports = root / "exports"
    wan_input = exports / "wan_input.mp4"
    if not wan_input.exists():
        raise WanError("Gere o pacote do Wan antes de importar um resultado.")
    try:
        info = media.probe(upload)
    except media.MediaError as exc:
        raise WanError(f"O arquivo enviado não é um vídeo válido: {exc}") from exc
    exports.mkdir(parents=True, exist_ok=True)
    os.replace(upload, exports / "wan_result.mp4")
    result = exports / "wan_result.mp4"
    clip = root / "clip.mp4"
    has_audio = bool(project["has_audio"])

    ff = media.ffmpeg_bin()
    final_tmp = exports / "wan_final.tmp.mp4"
    cmd = [ff, "-y", "-v", "error", "-i", str(result)]
    if has_audio:
        cmd += ["-i", str(clip), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(final_tmp)]
    media.run(cmd)
    os.replace(final_tmp, exports / "wan_final.mp4")

    # Comparativo: original (no tamanho do pacote) | resultado, na mesma altura e no fps do resultado.
    height = info.height - info.height % 2
    comp_tmp = exports / "comparativo.tmp.mp4"
    graph = (
        f"[0:v]fps={media.fps_arg(info.fps)},scale=-2:{height},setsar=1[a];"
        f"[1:v]scale=-2:{height},setsar=1[b];[a][b]hstack=inputs=2[v]"
    )
    cmd = [ff, "-y", "-v", "error", "-i", str(wan_input), "-i", str(result)]
    if has_audio:
        cmd += ["-i", str(clip)]
    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "2:a:0?", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(comp_tmp)]
    media.run(cmd)
    os.replace(comp_tmp, exports / "comparativo.mp4")
    return f"Resultado importado: {info.width}x{info.height}, {info.duration:.1f}s."
