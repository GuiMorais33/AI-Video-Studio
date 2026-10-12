"""Refino do rosto neste computador: FaceFusion 3.9.1 na CPU, depois que o vídeo do Wan volta.

O Kaggle bloqueia o FaceFusion e ferramentas parecidas (aviso da equipe do Kaggle, 18/08/2026):
a troca do rosto saiu do notebook e roda aqui, num processo à parte com o Python do FaceFusion
(engine/.vendor/facefusion/.venv, instalado pelo scripts/setup.sh).

1. As máscaras do pacote passam pelo mesmo processamento do notebook (dilatação 7x7 x3 e blocos
   de 16) e são ajustadas ao tamanho do vídeo (escala para cobrir + recorte central).
2. Só o personagem vai para o FaceFusion (fora da máscara, preto): ninguém do fundo tem o rosto
   trocado e o maior rosto que sobra é sempre o do personagem.
3. O FaceFusion troca o rosto pelo das fotos (hyperswap) e restaura a nitidez (GFPGAN). O filtro
   de conteúdo dele fica sempre ligado.
4. Só a região que o FaceFusion mudou (o rosto) volta para o vídeo, com borda suave e dentro da
   máscara; o resto do quadro continua o do Wan, sem a recompressão do FaceFusion. O acabamento
   (tamanho dos Reels, áudio original, comparativo) é refeito.

Se o FaceFusion não conseguir (foto sem rosto humano, filtro de conteúdo, travou), o wan_final.mp4
fica sem o refino e a tarefa termina com um aviso.
"""

from __future__ import annotations

import functools
import io
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from . import media, wan
from .config import Settings

# Os mesmos parâmetros da etapa "rosto" que o notebook do Kaggle usava antes do bloqueio.
FACE_SWAPPER_MODEL = "hyperswap_1a_256"
FACE_SWAPPER_PIXEL_BOOST = "512x512"
# 0,5 = só a identidade das fotos; acima disso afasta ainda mais do rosto gerado.
FACE_SWAPPER_WEIGHT = 0.75
FACE_ENHANCER_MODEL = "gfpgan_1.4"
FACE_ENHANCER_BLEND = 80
# Máscara igual à da geração (pré-processamento oficial do Wan + blocos do template ComfyUI).
MASK_KERNEL = 7
MASK_ITERATIONS = 3
MASK_BLOCK = 16
COMPOSITE_FEATHER = 6
# Onde a diferença entre o vídeo do FaceFusion e o do Wan passa disto (média dos canais, 0-255), o
# FaceFusion mudou o quadro de verdade: é o rosto. Abaixo disso é só a recompressão do vídeo dele.
CHANGE_THRESHOLD = 10
# Quanto a região do rosto cresce antes da borda suave (pixels no tamanho da geração).
FACE_GROW = 8
# Modelos principais: sem eles, o 1º refino baixa ~1,7 GB.
MODEL_FILES = ("hyperswap_1a_256.onnx", "gfpgan_1.4.onnx")
RESULT_FILE = "face_refine.json"
LOG_FILE = "facefusion_rosto.log"

FACE_PROGRESS_RE = re.compile(r"processing:\s+(\d+)%")
# Barras do tqdm no log: baixando modelos, filtro de conteúdo analisando, processando os quadros.
STAGE_RE = re.compile(r"(downloading|analysing|processing):\s+(\d+)%")
# Uma barra inteira, com etapa e contador (ex.: "downloading:  20%|##  | 9.54M/47.7M [00:01<?, ?B/s]").
BAR_RE = re.compile(r"(\w+):\s+\d+%\|[^|]*\|\s*(\S+).*")
# O setup.sh grava este arquivo no .venv do FaceFusion só quando a instalação inteira deu certo.
INSTALL_MARKER = ".aivs-ok"

NOT_INSTALLED = "Refino do rosto não instalado: rode o instalador de novo (bash scripts/setup.sh)."
STOPPED = "Refino do rosto interrompido: o motor foi encerrado. Use \"Refazer o refino do rosto\"."

ProgressFn = Callable[..., None]


class FaceRefineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- instalação

def models_present(directory: Path) -> bool:
    models = directory / ".assets" / "models"
    return all((models / name).exists() for name in MODEL_FILES)


@functools.lru_cache(maxsize=1)  # vai em todo payload: o ffmpeg só é consultado uma vez
def ffmpeg_version() -> tuple[int, int] | None:
    try:
        out = subprocess.run([media.ffmpeg_bin(), "-version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired, media.MediaError):
        return None
    found = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
    return (int(found[1]), int(found[2])) if found else None


def status(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    """FaceFusion instalado? O modo simples olha os arquivos e a versão do ffmpeg (barato, vai em
    todo payload); o completo também abre o FaceFusion (`--version`), o que pega dependências faltando.

    `needs_upgrade`: o motivo é do sistema (ffmpeg antigo), e rodar o instalador de novo não resolve."""
    directory, python = settings.facefusion_dir, settings.facefusion_python_path
    info: dict[str, Any] = {
        "installed": False, "dir": str(directory), "python": str(python), "version": None,
        "models": models_present(directory), "reason": None, "needs_upgrade": False,
    }
    version = ffmpeg_version()
    if not (directory / "facefusion.py").exists():
        info["reason"] = f"FaceFusion não encontrado em {directory}"
    elif not os.access(python, os.X_OK):
        info["reason"] = f"Python do FaceFusion não encontrado em {python}"
    elif settings.facefusion_python is None and not (directory / ".venv" / INSTALL_MARKER).exists():
        info["reason"] = "a instalação do FaceFusion não terminou"
    elif version is not None and version < (5, 1):
        # Ubuntu 22.04 vem com o 4.4: o FaceFusion falharia ao ler e gravar o vídeo.
        info["reason"] = (f"o ffmpeg {version[0]}.{version[1]} deste computador é antigo (o FaceFusion pede 5.1 ou mais "
                          "novo; o Ubuntu 24.04 já vem com ele)")
        info["needs_upgrade"] = True
    elif deep:
        try:
            proc = subprocess.run([str(python), "facefusion.py", "--version"], cwd=directory,
                                  capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["reason"] = f"o FaceFusion não abriu ({type(exc).__name__})"
        else:
            if proc.returncode == 0 and "facefusion" in proc.stdout.lower():
                info["version"] = proc.stdout.strip().splitlines()[-1]
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
                info["reason"] = "o FaceFusion não abriu (dependências incompletas?)" + (f": {tail[0]}" if tail else "")
    info["installed"] = info["reason"] is None
    return info


def unavailable_message(info: dict[str, Any]) -> str:
    """O que dizer quando o refino não pode rodar, com o motivo e o que fazer."""
    if info.get("needs_upgrade"):
        return (f"Refino do rosto indisponível: {info['reason']}. Rodar o instalador de novo não resolve: atualize o "
                "Ubuntu para o 24.04 (sudo do-release-upgrade) e depois rode o instalador.")
    if info.get("reason"):
        return f"Refino do rosto não instalado ({info['reason']}): rode o instalador de novo (bash scripts/setup.sh)."
    return NOT_INSTALLED


def read_result(root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((root / "exports" / RESULT_FILE).read_text())
    except (OSError, ValueError):
        return None


def face_sources(root: Path) -> list[Path]:
    """Fotos que definem o rosto: as do rosto (vários ângulos, o FaceFusion combina todas) ou,
    sem elas, a imagem do personagem."""
    faces = [root / wan.FACES_DIR / name for name in wan.list_face_photos(root)]
    reference = root / "reference.png"
    return faces or ([reference] if reference.exists() else [])


# --------------------------------------------------------------------------- máscaras

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


def fit_cover(img: Image.Image, size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    """Escala para cobrir `size` (largura, altura) e corta o excesso no centro, como o
    `scale=...:force_original_aspect_ratio=increase,crop` do acabamento."""
    width, height = size
    if img.size == size:
        return img
    scale = max(width / img.width, height / img.height)
    scaled = (max(width, round(img.width * scale)), max(height, round(img.height * scale)))
    img = img.resize(scaled, resample)
    left, top = (scaled[0] - width) // 2, (scaled[1] - height) // 2
    return img.crop((left, top, left + width, top + height))


class PackageMasks:
    """Máscaras do wan_package.zip (tamanho da geração), processadas como no notebook e
    ajustadas ao tamanho do vídeo do resultado. Lidas uma a uma, sem guardar todas na memória."""

    def __init__(self, package: Path, size: tuple[int, int]):
        self.size = size
        self._zip = zipfile.ZipFile(package)
        try:
            job = json.loads(self._zip.read("job.json"))
            self.count = int(job["num_frames"])
            names = set(self._zip.namelist())
            if self.count < 1 or any(f"masks/{i:05d}.png" not in names for i in range(self.count)):
                raise ValueError("máscaras faltando no pacote")
            # Quanto o tamanho da geração foi ampliado para cobrir o vídeo (1 no fluxo normal).
            self.scale = max(size[0] / int(job["width"]), size[1] / int(job["height"]))
        except Exception:
            self._zip.close()
            raise

    def _processed(self, idx: int) -> Image.Image:
        with Image.open(io.BytesIO(self._zip.read(f"masks/{idx:05d}.png"))) as img:
            mask = np.asarray(img.convert("L")) > 127
        mask = blockify(dilate(mask, MASK_KERNEL, MASK_ITERATIONS), MASK_BLOCK)
        return Image.fromarray(mask.astype(np.uint8) * 255)

    def binary(self, idx: int) -> np.ndarray:
        """Área do personagem (bool, altura x largura)."""
        return np.asarray(fit_cover(self._processed(idx), self.size, Image.Resampling.NEAREST)) > 127

    def alpha(self, idx: int) -> np.ndarray:
        """Máscara 0..1 com a borda esfumada (altura x largura x 1), para colar sem emenda."""
        soft = self._processed(idx).filter(ImageFilter.GaussianBlur(COMPOSITE_FEATHER))
        return np.asarray(fit_cover(soft, self.size, Image.Resampling.BILINEAR), np.float32)[..., None] / 255.0

    def close(self) -> None:
        self._zip.close()


def face_alpha(face: np.ndarray, base: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Região que o FaceFusion realmente mudou (o rosto trocado), 0..1 com borda suave
    (altura x largura x 1). Zero quando ele não achou rosto no quadro.

    Fora do rosto, o vídeo dele só difere do Wan pela recompressão: colar o personagem inteiro
    deixaria uma emenda visível no contorno da máscara."""
    diff = np.abs(face.astype(np.int16) - base.astype(np.int16)).mean(axis=2)
    # x8 para não perder precisão no uint8 do desfoque (o limiar fica em 80 de 255).
    img = Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8))
    changed = np.asarray(img.filter(ImageFilter.GaussianBlur(2 * scale))) > CHANGE_THRESHOLD * 8
    if not changed.any():
        return np.zeros((*diff.shape, 1), np.float32)
    grown = Image.fromarray(changed.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(FACE_GROW * scale))
    region = Image.fromarray(((np.asarray(grown) > 25) * 255).astype(np.uint8))
    soft = region.filter(ImageFilter.GaussianBlur(COMPOSITE_FEATHER * scale))
    return np.asarray(soft, np.float32)[..., None] / 255.0


# --------------------------------------------------------------------------- FaceFusion

def facefusion_command(settings: Settings, sources: list[Path], target: Path, output: Path,
                       work: Path) -> list[str]:
    """headless-run do FaceFusion na CPU: troca o rosto principal de cada quadro pelo das fotos e
    restaura a nitidez. Mesmos processadores e opções da etapa "rosto" do notebook."""
    return [
        str(settings.facefusion_python_path), "facefusion.py", "headless-run",
        "-s", *(str(p) for p in sources), "-t", str(target), "-o", str(output),
        "--temp-path", str(work / "ff-temp"), "--jobs-path", str(work / "ff-jobs"),
        "--processors", "face_swapper", "face_enhancer",
        "--face-swapper-model", FACE_SWAPPER_MODEL,
        "--face-swapper-pixel-boost", FACE_SWAPPER_PIXEL_BOOST,
        "--face-swapper-weight", str(FACE_SWAPPER_WEIGHT),
        "--face-enhancer-model", FACE_ENHANCER_MODEL,
        "--face-enhancer-blend", str(FACE_ENHANCER_BLEND),
        "--face-detector-model", "yolo_face", "--face-detector-size", "640x640", "--face-detector-score", "0.5",
        "--face-landmarker-model", "2dfan4", "--face-landmarker-score", "0.5",
        # Só o rosto maior de cada quadro: o personagem (o fundo já chega apagado).
        "--face-selector-mode", "one", "--face-selector-order", "large-small",
        "--face-tracker-score", "0.3", "--target-frame-amount", "2",
        # Caixa + oclusão: mantém o rosto inteiro e respeita mãos e cabelo na frente.
        "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_1",
        "--face-mask-blur", "0.3", "--face-mask-padding", "0",
        "--workflow-strategy", "memory",
        # Vídeo intermediário: o acabamento codifica de novo no tamanho final.
        "--output-video-encoder", "libx264", "--output-video-quality", "90", "--output-video-preset", "veryfast",
        "--output-audio-volume", "0",
        "--execution-providers", "cpu", "--execution-thread-count", str(settings.facefusion_threads),
        "--download-providers", "github", "--log-level", "info",
    ]


@dataclass
class FaceFusionRun:
    code: int
    log: str  # texto do log, em minúsculas
    output_ok: bool
    timed_out: bool = False
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and self.output_ok and not (self.timed_out or self.stopped)


def run_facefusion(settings: Settings, sources: list[Path], target: Path, output: Path, work: Path,
                   log_path: Path, on_stage: Callable[[str, int], None] = lambda stage, pct: None,
                   stop: threading.Event | None = None) -> FaceFusionRun:
    """Roda o FaceFusion no seu próprio grupo de processos e grava o log inteiro em `log_path`.

    O grupo inteiro (FaceFusion + curl/ffmpeg que ele abre) é encerrado se ficar
    `facefusion_idle_timeout_s` sem escrever nada, ou quando `stop` liga (motor encerrando).
    Não há limite de tempo total: na CPU o refino é longo, mas escreve progresso a cada quadro.
    """
    command = facefusion_command(settings, sources, target, output, work)
    # CONDA_READY: o FaceFusion não tenta reiniciar dentro do conda. Sem buffer: o log chega na hora.
    env = {**os.environ, "CONDA_READY": "1", "PYTHONUNBUFFERED": "1"}
    output.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    idle_limit = settings.facefusion_idle_timeout_s
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(command, cwd=settings.facefusion_dir, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        last_output = [time.monotonic()]
        finished = threading.Event()
        why: list[str] = []

        def kill(reason: str) -> None:
            why.append(reason)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        def watchdog() -> None:
            while not finished.wait(min(0.5, idle_limit / 4)):
                if stop is not None and stop.is_set():
                    kill("stop")
                    return
                if time.monotonic() - last_output[0] > idle_limit:
                    kill("idle")
                    return

        guard = threading.Thread(target=watchdog, name="facefusion-watchdog", daemon=True)
        guard.start()
        try:
            last: tuple[str, int] | None = None
            last_bar: tuple[str, str] | None = None
            pending = ""  # pedaço de linha que ainda não terminou (\r ou \n)
            # read1: devolve o que já chegou, sem esperar encher o bloco (o progresso fica em dia).
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                log.write(chunk)
                *lines, pending = re.split(r"[\r\n]", pending + chunk.decode(errors="replace"))
                for line in lines:
                    line = line.strip()
                    # Sinal de vida = texto novo ou barra que andou. Ao retomar um download parado, o
                    # FaceFusion redesenha a mesma barra sem parar: isso não conta, e o vigia age.
                    bar = BAR_RE.fullmatch(line)
                    if bar is None:
                        if line:
                            last_output[0] = time.monotonic()
                    elif (bar[1], bar[2]) != last_bar:
                        last_bar = (bar[1], bar[2])
                        last_output[0] = time.monotonic()
                    found = STAGE_RE.search(line)
                    if found and (found[1], int(found[2])) != last:
                        last = (found[1], int(found[2]))
                        on_stage(*last)
            code = proc.wait()
        finally:
            finished.set()
            guard.join()
            if proc.poll() is None:  # erro no meio da leitura: não deixa o FaceFusion órfão
                kill("error")
                proc.wait()
            proc.stdout.close()
    return FaceFusionRun(
        code=code,
        log=log_path.read_text(errors="replace").lower(),
        output_ok=output.exists() and output.stat().st_size > 0,
        timed_out="idle" in why,
        stopped="stop" in why,
    )


def failure_reason(run: FaceFusionRun, settings: Settings) -> str | None:
    """Por que o refino não saiu (None se saiu), com as mesmas regras do notebook."""
    if run.ok:
        return None
    if run.timed_out:
        minutes = settings.facefusion_idle_timeout_s / 60
        return (f"o FaceFusion ficou {minutes:g} min sem dar sinal e foi encerrado "
                "(download de modelo travado? veja a internet e tente de novo)")
    if "no source face" in run.log:
        return "as fotos não têm um rosto humano detectável (personagem sem rosto?)"
    if "analysing: 100%" in run.log and not FACE_PROGRESS_RE.search(run.log):
        # A análise terminou e a barra "processing: NN%" nunca apareceu: o filtro recusou o vídeo.
        return "o filtro de conteúdo do FaceFusion barrou o vídeo"
    if run.code == -signal.SIGKILL:  # o nosso SIGKILL já virou timed_out/stopped: este veio de fora
        return ("o sistema encerrou o FaceFusion à força, quase sempre por falta de memória: feche outros "
                "programas (ou aumente a memória do WSL com memory= no .wslconfig) e use \"Refazer o refino do rosto\"")
    if run.code < 0:
        return f"o FaceFusion caiu com o sinal {-run.code} (veja exports/facefusion_rosto.log)"
    if run.code == 0:
        return "o FaceFusion terminou sem gerar o vídeo (veja exports/facefusion_rosto.log)"
    return f"o FaceFusion terminou com código {run.code} (veja exports/facefusion_rosto.log)"


# --------------------------------------------------------------------------- vídeo

def _check_stop(stop: threading.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise FaceRefineError(STOPPED)


def _even(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return frame[:height, :width] if frame.shape[:2] != (height, width) else frame


def write_target(result: Path, masks: PackageMasks, target: Path, info: media.VideoInfo,
                 size: tuple[int, int], progress: Callable[[float], None], stop: threading.Event | None) -> int:
    """Vídeo que vai para o FaceFusion: o personagem e, fora da máscara, preto. Devolve os quadros."""
    width, height = size
    writer = media.FrameWriter(target, width, height, info.fps, crf=12)
    count = 0
    try:
        with media.FrameReader(result, info.width, info.height) as reader:
            while (frame := reader.read()) is not None and count < masks.count:
                _check_stop(stop)
                writer.write(_even(frame, width, height) * masks.binary(count)[..., None])
                count += 1
                progress(count / masks.count)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return count


def paste_back(result: Path, refined: Path, masks: PackageMasks, output: Path, info: media.VideoInfo,
               size: tuple[int, int], frames: int, progress: Callable[[float], None],
               stop: threading.Event | None) -> bool:
    """Rosto refinado só onde o FaceFusion mudou o quadro e dentro da máscara (borda suave); no
    resto, o resultado do Wan intacto. False se o FaceFusion devolveu outro número de quadros."""
    width, height = size
    writer = media.FrameWriter(output, width, height, info.fps, crf=12)
    count = 0
    try:
        with media.FrameReader(result, info.width, info.height) as base_reader, \
                media.FrameReader(refined, width, height) as face_reader:
            while True:
                _check_stop(stop)
                base, face = base_reader.read(), face_reader.read()
                if base is None or face is None or count >= frames:
                    break
                base = _even(base, width, height)
                # Fora da máscara o FaceFusion só viu preto: ali vale o quadro do Wan, sem escurecer a borda.
                face = np.where(masks.binary(count)[..., None], face, base)
                alpha = masks.alpha(count) * face_alpha(face, base, masks.scale)
                mixed = alpha * face.astype(np.float32) + (1.0 - alpha) * base.astype(np.float32)
                writer.write(np.clip(mixed, 0, 255).round().astype(np.uint8))
                count += 1
                progress(count / frames)
    except BaseException:
        writer.abort()
        raise
    if count != frames:
        writer.abort()
        return False
    writer.close()
    return True


# --------------------------------------------------------------------------- tarefa

def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


def _save_result(root: Path, data: dict[str, Any]) -> None:
    path = root / "exports" / RESULT_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def refine(project: dict[str, Any], root: Path, settings: Settings, progress: ProgressFn = lambda *a: None,
           stop: threading.Event | None = None) -> str:
    """Refina o rosto do exports/wan_result.mp4 e refaz o wan_final.mp4 e o comparativo.mp4.

    Falha do FaceFusion não é erro da tarefa: o final fica sem o refino e a mensagem explica.
    Erro (exceção) só quando nem dá para começar: sem resultado, sem fotos, FaceFusion não instalado.
    """
    exports = root / "exports"
    result = exports / "wan_result.mp4"
    if not result.exists():
        raise FaceRefineError("Importe ou gere o resultado do Wan antes do refino do rosto.")
    progress(0.0, "Conferindo o FaceFusion")
    info = status(settings, deep=True)
    if not info["installed"]:
        raise FaceRefineError(unavailable_message(info))
    sources = face_sources(root)
    if not sources:
        raise FaceRefineError("Envie a imagem do personagem ou fotos do rosto antes do refino do rosto.")
    previous = read_result(root)
    started = time.monotonic()
    info = media.probe(result)
    size = (info.width - info.width % 2, info.height - info.height % 2)
    frames = media.count_frames(result)
    warnings: list[str] = []

    masks: PackageMasks | None = None
    package = exports / "wan_package.zip"
    try:
        masks = PackageMasks(package, size) if package.exists() else None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        masks = None
    whole_frame = "o refino usa o quadro inteiro (troca o maior rosto de cada quadro)"
    if masks is None:
        warnings.append(f"sem as máscaras do pacote do Wan, {whole_frame}")
    elif masks.count != frames:
        warnings.append(f"o vídeo tem {frames} quadros e as máscaras do pacote, {masks.count}: {whole_frame}")
        masks.close()
        masks = None

    log_path = exports / LOG_FILE
    reported: dict[str, int] = {}
    processing_started: list[float] = []

    def on_stage(stage: str, pct: int) -> None:
        # Download e análise andam rápido na barra: só de 5 em 5% (cada mensagem grava no banco).
        # Um valor menor é a barra do próximo arquivo de modelo.
        last = reported.get(stage)
        if stage != "processing" and last is not None and 0 <= pct - last < 5 and pct < 100:
            return
        reported[stage] = pct
        if stage == "downloading":
            progress(0.06, f"Baixando os modelos do refino do rosto (só na 1ª vez, ~1,7 GB): {pct}%")
        elif stage == "analysing":
            progress(0.07, f"Conferindo o vídeo (filtro de conteúdo do FaceFusion): {pct}%")
        else:
            if not processing_started:
                processing_started.append(time.monotonic())
            eta = ""
            if 5 <= pct < 100:  # antes disso a estimativa ainda inclui carregar os modelos
                left = (time.monotonic() - processing_started[0]) / pct * (100 - pct)
                eta = f" (faltam ~{_duration(left)})"
            progress(0.08 + 0.8 * pct / 100, f"Refinando o rosto na CPU: {pct}%{eta}")

    try:
        with tempfile.TemporaryDirectory(dir=root, prefix="rosto-") as tmp_name:
            tmp = Path(tmp_name)
            photos = tmp / "fotos"
            photos.mkdir()
            copies = []
            for i, src in enumerate(sources, 1):  # cópias: trocar as fotos durante o refino não muda nada
                copies.append(photos / f"{i:02d}{src.suffix}")
                shutil.copyfile(src, copies[-1])
            if masks is not None:
                progress(0.01, "Separando o personagem do fundo")
                target = tmp / "alvo.mp4"
                write_target(result, masks, target, info, size, lambda f: progress(0.01 + 0.04 * f), stop)
            else:
                target = result
            progress(0.05, "Abrindo o FaceFusion (pode levar alguns minutos)")
            face_video = tmp / "rosto.mp4"
            run = run_facefusion(settings, copies, target, face_video, tmp, log_path, on_stage, stop)
            if run.stopped:
                raise FaceRefineError(STOPPED)
            reason = failure_reason(run, settings)
            refined = face_video
            if reason is None and masks is not None:
                progress(0.89, "Colando o rosto refinado no vídeo")
                refined = tmp / "composto.mp4"
                out = media.probe(face_video)
                if (out.width, out.height) != size:
                    reason = f"o FaceFusion devolveu o vídeo em {out.width}x{out.height} (esperado {size[0]}x{size[1]})"
                elif not paste_back(result, face_video, masks, refined, info, size, frames,
                                    lambda f: progress(0.89 + 0.04 * f), stop):
                    reason = f"o FaceFusion devolveu {media.count_frames(face_video)} de {frames} quadros"
            if reason is None:
                progress(0.94, "Montando o wan_final.mp4 e o comparativo")
                wan.finish_result(project, root, refined)
            elif previous and previous.get("ok"):
                # O final atual tinha o refino anterior (outras fotos): volta para o vídeo sem refino.
                progress(0.94, "Refazendo o wan_final.mp4 sem o refino")
                wan.finish_result(project, root, result)
    finally:
        if masks is not None:
            masks.close()

    seconds = round(time.monotonic() - started, 1)
    data: dict[str, Any] = {
        "ok": reason is None, "fotos": len(sources), "segundos": seconds, "quadros": frames,
        "mascara": masks is not None, "avisos": warnings, "trocador": FACE_SWAPPER_MODEL,
        "restaurador": FACE_ENHANCER_MODEL, "provedor": "cpu", "threads": settings.facefusion_threads,
        "concluido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if reason is not None:
        data["motivo"] = reason
    _save_result(root, data)
    notes = "".join(f" Aviso: {w}." for w in warnings)
    if reason is not None:
        return f"Refino do rosto não aplicado: {reason}. O wan_final.mp4 ficou sem o refino.{notes}"
    return f"Rosto refinado com {len(sources)} foto(s) em {_duration(seconds)}.{notes}"
