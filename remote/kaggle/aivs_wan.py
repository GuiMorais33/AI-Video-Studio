"""AI Video Studio — Wan 2.2 Animate (modo replace) numa GPU gratuita do Kaggle.

Lê o pacote gerado pelo estúdio (wan_package.zip: vídeo, máscaras do SAM 2.1,
imagem de referência e job.json) e:

1. processa a máscara como o pré-processamento oficial (dilatação 7x7 x3 e blocos)
   e monta o fundo (área da pessoa em preto);
2. extrai a pose com o ViTPose oficial do Wan, usando a caixa da NOSSA máscara em vez
   do detector YOLO (acerta a pessoa escolhida mesmo com várias no vídeo), e recorta
   o rosto em 512x512;
3. codifica o texto (umT5) e libera a memória;
4. gera com o WanAnimatePipeline do diffusers: transformer GGUF + LoRAs de poucos
   passos + offload em blocos, em fp16 (a T4 não tem bf16 nativo);
5. grava resultado.mp4, prévias de depuração e relatorio.json em /kaggle/working.

Todo caminho de modelo aceita um arquivo/pasta local no lugar do Hugging Face: é assim
que os testes rodam este fluxo inteiro na CPU com modelos minúsculos.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

PACKAGE_FORMAT = 1
DEFAULT_PROMPT = "People in the video are doing actions."


@dataclass
class Config:
    # Pesos (Hugging Face "repo" + "arquivo", ou caminhos locais)
    repo: str = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
    gguf_repo: str = "QuantStack/Wan2.2-Animate-14B-GGUF"
    gguf_file: str = "Wan2.2-Animate-14B-Q4_K_M.gguf"
    lora_repo: str = "Kijai/WanVideo_comfy"
    # (arquivo, nome do adaptador, força, obrigatória)
    loras: tuple = (
        ("Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "lightx2v", 1.0, True),
        ("LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors", "relight", 1.0, False),
    )
    pose_repo: str = "Wan-AI/Wan2.2-Animate-14B"
    # No repositório oficial isto é uma PASTA (end2end.onnx + pesos em arquivos separados, ~2,5 GB).
    pose_path: str = "process_checkpoint/pose2d/vitpose_h_wholebody.onnx"
    # Código oficial do pré-processamento (ViTPose + desenho do esqueleto), versão fixa.
    wan_git: str = "https://github.com/Wan-Video/Wan2.2.git"
    wan_commit: str = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
    wan_dir: str | None = None  # clone local já existente (testes)
    # Geração
    steps: int = 6
    shift: float = 8.0
    guidance_scale: float = 1.0
    seed: int = 42
    segment_frames: int = 77
    prev_segment_frames: int = 1
    dtype: str = "float16"
    offload: bool = True
    # Decodifica o VAE em blocos sobrepostos: evita estourar a VRAM no fim da geração.
    vae_tiling: bool = True
    # Máscara (iguais ao pré-processamento oficial; bloco = template ComfyUI)
    mask_kernel: int = 7
    mask_iterations: int = 3
    mask_block: int = 16
    # Pastas
    input_root: str = "/kaggle/input"
    scratch: str = "/kaggle/tmp"
    output: str = "/kaggle/working"
    # Testes: troca o ViTPose por um objeto compatível (preprocess + __call__)
    pose_model: Any = field(default=None, repr=False)


# --------------------------------------------------------------------------- utilidades

def progress(fraction: float, message: str) -> None:
    """Linha lida pelo estúdio no log do Kaggle para mostrar o andamento."""
    print(f"AIVS_PROGRESS {max(0.0, min(1.0, fraction)):.3f} {message}", flush=True)


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"etapas": {}, "avisos": []}

    @contextmanager
    def etapa(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data["etapas"][f"{name}_s"] = round(time.perf_counter() - start, 1)
            self.save()

    def aviso(self, text: str) -> None:
        print("AVISO:", text, flush=True)
        self.data["avisos"].append(text)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str))


def _torch():
    import torch

    return torch


def environment() -> dict[str, Any]:
    torch = _torch()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    for mod in ("diffusers", "transformers", "peft", "accelerate", "gguf", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def resolve_file(repo: str, filename: str, cache: Path) -> str:
    """Caminho local do arquivo: usa o local se existir, senão baixa do Hugging Face."""
    if Path(filename).exists():
        return str(filename)
    from huggingface_hub import hf_hub_download

    for attempt in range(3):
        try:
            return hf_hub_download(repo, filename, local_dir=str(cache / repo.replace("/", "__")))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("inalcançável")


def resolve_folder(repo: str, folder: str, cache: Path) -> str:
    """Pasta local: usa a local se existir, senão baixa só essa pasta do repositório."""
    if Path(folder).exists():
        return str(folder)
    from huggingface_hub import snapshot_download

    local = cache / repo.replace("/", "__")
    for attempt in range(3):
        try:
            snapshot_download(repo, allow_patterns=[f"{folder}/**"], local_dir=str(local))
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    path = local / folder
    if not path.exists():
        raise FileNotFoundError(f"{folder} não encontrado em {repo}")
    return str(path)


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)


# --------------------------------------------------------------------------- pacote

def find_package(input_root: Path, work: Path) -> Path:
    """Pasta com job.json: o pacote pode vir já descompactado pelo Kaggle ou como .zip."""
    for job in sorted(input_root.rglob("job.json")):
        if (job.parent / "video.mp4").exists():
            return job.parent
    zips = sorted(input_root.rglob("wan_package.zip"))
    if not zips:
        raise FileNotFoundError(f"Nenhum wan_package.zip ou job.json em {input_root}")
    dest = work / "pacote"
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(dest)
    return dest


def read_video(path: Path, width: int, height: int) -> list[np.ndarray]:
    """Quadros RGB uint8 via ffmpeg (presente no Kaggle)."""
    out = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout
    frame_size = width * height * 3
    count = len(out) // frame_size
    return [np.frombuffer(out, np.uint8, frame_size, i * frame_size).reshape(height, width, 3) for i in range(count)]


def load_job(job_dir: Path) -> tuple[dict, list[np.ndarray], list[np.ndarray], Image.Image]:
    job = json.loads((job_dir / "job.json").read_text())
    if job.get("format") != PACKAGE_FORMAT or job.get("mode") != "replace":
        raise ValueError(f"Pacote incompatível: format={job.get('format')} mode={job.get('mode')}")
    width, height = int(job["width"]), int(job["height"])
    if width % 16 or height % 16:
        raise ValueError(f"Tamanho {width}x{height} não é múltiplo de 16")
    frames = read_video(job_dir / "video.mp4", width, height)
    masks = []
    for i in range(int(job["num_frames"])):
        with Image.open(job_dir / "masks" / f"{i:05d}.png") as img:
            masks.append(np.asarray(img.convert("L")) > 127)
    count = min(len(frames), len(masks))
    if count < 2:
        raise ValueError("O pacote precisa de pelo menos 2 quadros")
    reference = Image.open(job_dir / "reference.png").convert("RGB")
    return job, frames[:count], masks[:count], reference


# --------------------------------------------------------------------------- máscara e fundo

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


def process_masks(masks: list[np.ndarray], cfg: Config) -> list[np.ndarray]:
    return [blockify(dilate(m, cfg.mask_kernel, cfg.mask_iterations), cfg.mask_block) for m in masks]


def backgrounds(frames: list[np.ndarray], masks: list[np.ndarray]) -> list[Image.Image]:
    """Fundo do modo replace: o quadro com a área a gerar em preto (como o src_bg oficial)."""
    return [Image.fromarray(f * (~m)[..., None].astype(np.uint8)) for f, m in zip(frames, masks)]


# --------------------------------------------------------------------------- pose e rosto

def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 1.0], dtype=np.float32)


def wan_preprocess_dir(cfg: Config, work: Path) -> Path:
    """Pasta wan/modules/animate/preprocess do repositório oficial (clonado na versão fixa)."""
    if cfg.wan_dir:
        root = Path(cfg.wan_dir)
    else:
        root = work / "Wan2.2"
        if not (root / ".git").exists():
            root.mkdir(parents=True, exist_ok=True)
            run(["git", "init", "-q", str(root)])
            run(["git", "-C", str(root), "fetch", "-q", "--depth", "1", cfg.wan_git, cfg.wan_commit])
            run(["git", "-C", str(root), "checkout", "-q", "FETCH_HEAD"])
    path = root / "wan" / "modules" / "animate" / "preprocess"
    if not path.exists():
        raise FileNotFoundError(f"Pré-processamento oficial não encontrado em {path}")
    return path


def import_pose_helpers(preprocess_dir: Path):
    """Funções puras (numpy/cv2) do pré-processamento oficial."""
    sys.path.insert(0, str(preprocess_dir))
    import human_visualization
    import pose2d_utils
    import utils as wan_utils

    return pose2d_utils, human_visualization, wan_utils


def extract_pose_face(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    pose_model: Any,
    helpers: tuple,
    on_frame: Callable[[int], None] = lambda i: None,
) -> tuple[list[Image.Image], list[Image.Image], int]:
    """Esqueleto (ViTPose wholebody) e rosto 512x512 por quadro, guiados pela máscara.

    Igual ao pré-processamento oficial, exceto que a caixa da pessoa vem da máscara do
    SAM 2.1 (não do YOLO). Os quadros passam pela mesma troca de canais que o
    Pose2d.load_images aplica no fluxo oficial.
    """
    pose2d_utils, human_visualization, wan_utils = helpers
    height, width = frames[0].shape[:2]
    keypoints, guided = [], 0
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        bbox = mask_bbox(mask)
        guided += bbox is not None
        image = np.ascontiguousarray(frame[..., ::-1])
        inp, center, scale = pose_model.preprocess(image, bbox)
        keypoints.append(pose_model(inp[None], center[None], scale[None]))
        on_frame(i)
    metas = pose2d_utils.load_pose_metas_from_kp2ds_seq(np.concatenate(keypoints, 0), width=width, height=height)
    poses, faces = [], []
    for frame, meta in zip(frames, metas):
        canvas = np.zeros((height, width, 3), np.uint8)
        aa_meta = pose2d_utils.AAPoseMeta.from_humanapi_meta(meta)
        poses.append(Image.fromarray(human_visualization.draw_aapose_by_meta_new(canvas, aa_meta)))
        x1, x2, y1, y2 = wan_utils.get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(height, width))
        crop = frame[y1:y2, x1:x2] if (x2 - x1) > 4 and (y2 - y1) > 4 else frame
        faces.append(Image.fromarray(np.ascontiguousarray(crop)).resize((512, 512), Image.BILINEAR))
    return poses, faces, guided


def pose_device() -> str:
    """Com duas GPUs (T4 x2), a pose usa a segunda: a memória que o ONNX Runtime reserva
    não atrapalha o texto e a geração, que ficam na primeira."""
    torch = _torch()
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def load_pose_model(cfg: Config, preprocess_dir: Path, cache: Path):
    if cfg.pose_model is not None:
        return cfg.pose_model
    sys.path.insert(0, str(preprocess_dir))
    from pose2d import ViTPose

    if Path(cfg.pose_path).exists():
        onnx = cfg.pose_path
    else:
        folder = str(Path(cfg.pose_path).parent)
        onnx = str(Path(resolve_folder(cfg.pose_repo, folder, cache)) / Path(cfg.pose_path).name)
    # ViTPose aceita a pasta (procura end2end.onnx dentro) ou um arquivo .onnx.
    device = pose_device()
    if device != "cpu":
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):  # acha CUDA/cuDNN instalados junto com o PyTorch
            onnxruntime.preload_dlls()
    return ViTPose(onnx, device=device)


# --------------------------------------------------------------------------- geração

def segment_length(num_frames: int, maximum: int) -> int:
    """Segmento 4n+1 que cobre o vídeo inteiro (1 segmento) ou o máximo permitido."""
    needed = 4 * math.ceil((num_frames - 1) / 4) + 1
    return min(needed, 4 * ((maximum - 1) // 4) + 1)


def encode_prompt(cfg: Config, prompt: str, device: str):
    """Etapa só de texto: carrega o umT5 (~11 GB), gera os embeddings e libera tudo."""
    torch = _torch()
    from diffusers import WanAnimatePipeline

    dtype = getattr(torch, cfg.dtype)
    pipe = WanAnimatePipeline.from_pretrained(cfg.repo, transformer=None, vae=None, image_encoder=None, torch_dtype=dtype)
    pipe.text_encoder.to(device)
    # encode_prompt do diffusers não desliga o gradiente: sem no_grad, o embedding guardaria o
    # grafo e manteria o umT5 inteiro (~11 GB) na GPU depois de apagar o pipeline.
    with torch.no_grad():
        embeds, _ = pipe.encode_prompt(
            prompt=prompt, do_classifier_free_guidance=False, max_sequence_length=512, device=device
        )
    if not torch.isfinite(embeds).all():
        raise FloatingPointError("O codificador de texto gerou NaN/inf")
    embeds = embeds.detach().to("cpu", dtype)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeds


def build_pipeline(cfg: Config, device: str, cache: Path, report: Report):
    torch = _torch()
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
        WanAnimatePipeline,
        WanAnimateTransformer3DModel,
    )
    from safetensors.torch import load_file

    dtype = getattr(torch, cfg.dtype)
    gguf = resolve_file(cfg.gguf_repo, cfg.gguf_file, cache)
    qcfg = GGUFQuantizationConfig(compute_dtype=dtype)
    # Se os pesos do codificador de movimento vierem quantizados, o forward quebra:
    # mantê-los sem quantização é inofensivo quando já vêm em F16/F32.
    qcfg.modules_to_not_convert = ["motion_encoder"]
    transformer = WanAnimateTransformer3DModel.from_single_file(
        gguf, config=cfg.repo, subfolder="transformer", quantization_config=qcfg, torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(cfg.repo, subfolder="vae", torch_dtype=torch.float32)
    if cfg.vae_tiling:
        vae.enable_tiling()
    pipe = WanAnimatePipeline.from_pretrained(
        cfg.repo, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dtype
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=cfg.shift)

    names, weights = [], []
    for filename, name, strength, required in cfg.loras:
        try:
            path = resolve_file(cfg.lora_repo, filename, cache)
            state = {k: v.to(dtype) for k, v in load_file(path).items()}
            pipe.load_lora_weights(state, adapter_name=name)
            layers = sum(1 for module, _ in pipe.transformer.named_modules() if module.endswith(f"lora_A.{name}"))
            if layers == 0:
                raise ValueError("nenhuma camada recebeu a LoRA (formato de chaves desconhecido)")
        except Exception as exc:
            if required:
                raise RuntimeError(f"LoRA obrigatória '{name}' falhou: {exc}") from exc
            report.aviso(f"LoRA opcional '{name}' ignorada: {exc}")
            continue
        report.data.setdefault("loras", {})[name] = layers
        names.append(name)
        weights.append(strength)
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        for pname, param in pipe.transformer.named_parameters():
            if "lora_" in pname and param.dtype != dtype:
                param.data = param.data.to(dtype)

    if cfg.offload:
        on_cuda = device.startswith("cuda")
        pipe.transformer.enable_group_offload(
            onload_device=torch.device(device), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=1,
            use_stream=on_cuda, low_cpu_mem_usage=on_cuda,
        )
    else:
        pipe.transformer.to(device)
    pipe.vae.to(device)
    pipe.image_encoder.to(device)
    return pipe


def generate(pipe, cfg: Config, job: dict, reference, poses, faces, backgrounds_, masks, embeds, device: str):
    torch = _torch()
    num = len(poses)
    seg = segment_length(num, cfg.segment_frames)
    step_seg = seg - cfg.prev_segment_frames
    segments = 1 if num <= seg else 1 + math.ceil((num - seg) / step_seg)
    total = cfg.steps * segments
    done = {"n": 0}

    def on_step(_pipe, _i, _t, kwargs):
        done["n"] += 1
        progress(0.45 + 0.5 * done["n"] / total, f"Gerando: passo {done['n']}/{total}")
        return kwargs

    mask_images = [Image.fromarray(m.astype(np.uint8) * 255, mode="L") for m in masks]
    result = pipe(
        image=reference,
        pose_video=poses,
        face_video=faces,
        background_video=backgrounds_,
        mask_video=mask_images,
        prompt_embeds=embeds.to(device),
        height=int(job["height"]),
        width=int(job["width"]),
        mode="replace",
        segment_frame_length=seg,
        prev_segment_conditioning_frames=cfg.prev_segment_frames,
        num_inference_steps=cfg.steps,
        guidance_scale=cfg.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(cfg.seed),
        output_type="np",
        callback_on_step_end=on_step,
    ).frames[0]
    if not np.isfinite(result).all():
        raise FloatingPointError("A geração produziu NaN. Tente dtype='float32' (mais lento).")
    return result, {"segment_frames": seg, "segments": segments, "steps_total": total}


# --------------------------------------------------------------------------- saída

def write_video(frames: list[np.ndarray] | np.ndarray, fps: str, path: Path, crf: int = 16) -> None:
    frames = [np.asarray(f) for f in frames]
    first = frames[0]
    if first.dtype != np.uint8:
        frames = [(np.clip(f, 0, 1) * 255).round().astype(np.uint8) for f in frames]
    if frames[0].ndim == 2:
        frames = [np.repeat(f[..., None], 3, axis=2) for f in frames]
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-framerate", fps, "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg falhou: {err[-800:]}")


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    torch = _torch()
    out_dir, scratch = Path(cfg.output), Path(cfg.scratch)
    if not scratch.exists():
        try:
            scratch.mkdir(parents=True)
        except OSError:
            scratch = Path("/tmp/aivs")
            scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "modelos"
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    report = Report(out_dir / "relatorio.json")
    report.data["config"] = {k: v for k, v in asdict(cfg).items() if k != "pose_model"}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_file = next(Path(cfg.input_root).rglob("aivs_run.json"), None)
    if run_file is not None:  # enviado pelo estúdio: identifica esta execução
        report.data["run_id"] = json.loads(run_file.read_text()).get("run_id")
        # Antes de qualquer progresso: o estúdio só confia nas linhas que vêm depois desta.
        print("AIVS_RUN", report.data["run_id"], flush=True)
    try:
        with report.etapa("ambiente"):
            report.data["ambiente"] = environment()
            print("AIVS_AMBIENTE", json.dumps(report.data["ambiente"]), flush=True)
            progress(0.01, "Ambiente verificado")
        with report.etapa("pacote"):
            job_dir = find_package(Path(cfg.input_root), scratch)
            job, frames, masks, reference = load_job(job_dir)
            report.data["job"] = {k: job[k] for k in ("project_id", "width", "height", "fps", "num_frames")}
            progress(0.03, f"Pacote: {len(frames)} quadros {job['width']}x{job['height']}")
        with report.etapa("mascaras"):
            gen_masks = process_masks(masks, cfg)
            bg = backgrounds(frames, gen_masks)
        with report.etapa("pose_rosto"):
            preprocess_dir = wan_preprocess_dir(cfg, scratch)
            helpers = import_pose_helpers(preprocess_dir)
            pose_model = load_pose_model(cfg, preprocess_dir, cache)
            session = getattr(pose_model, "session", None)
            if session is not None:
                report.data["pose_providers"] = session.get_providers()
                report.data["pose_device"] = getattr(pose_model, "device", None)
                if device != "cpu" and "CUDAExecutionProvider" not in report.data["pose_providers"]:
                    report.aviso("Pose na CPU: o onnxruntime não achou o CUDA (mais lento, mesmo resultado)")
            session = None  # sem referências soltas: a sessão do ONNX Runtime segura ~3 GB de VRAM
            poses, faces, guided = extract_pose_face(
                frames, masks, pose_model, helpers,
                on_frame=lambda i: progress(0.05 + 0.1 * (i + 1) / len(frames), f"Pose {i + 1}/{len(frames)}"),
            )
            if guided < len(frames):
                report.aviso(f"{len(frames) - guided} quadro(s) sem máscara: pose estimada no quadro inteiro")
            del pose_model
            gc.collect()
        with report.etapa("texto"):
            progress(0.16, "Codificando o texto (umT5)")
            embeds = encode_prompt(cfg, job.get("prompt") or DEFAULT_PROMPT, device)
            if torch.cuda.is_available():
                report.data["vram_texto_pico_gb"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
                report.data["vram_apos_texto_gb"] = round(torch.cuda.memory_allocated(0) / 2**30, 2)
        with report.etapa("modelo"):
            progress(0.25, "Carregando o Wan 2.2 Animate (GGUF) e as LoRAs")
            pipe = build_pipeline(cfg, device, cache, report)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        with report.etapa("geracao"):
            progress(0.45, "Gerando o personagem")
            result, info = generate(pipe, cfg, job, reference, poses, faces, bg, gen_masks, embeds, device)
            report.data["geracao"] = info
            if torch.cuda.is_available():
                report.data["vram_pico_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        with report.etapa("exportacao"):
            write_video(result, job["fps"], out_dir / "resultado.mp4")
            debug = out_dir / "depuracao"
            write_video([np.asarray(p) for p in poses], job["fps"], debug / "pose.mp4", crf=23)
            write_video([np.asarray(f) for f in faces], job["fps"], debug / "rosto.mp4", crf=23)
            write_video([m.astype(np.uint8) * 255 for m in gen_masks], job["fps"], debug / "mascara.mp4", crf=23)
            progress(1.0, "Pronto: resultado.mp4")
        report.data["ok"] = True
    except Exception as exc:
        report.data["ok"] = False
        report.data["erro"] = f"{type(exc).__name__}: {exc}"
        report.data["traceback"] = traceback.format_exc()[-6000:]
        # Uma linha só: o estúdio lê o erro no log mesmo sem conseguir baixar a saída.
        print("AIVS_ERRO", report.data["erro"].replace("\n", " ")[:2000], flush=True)
        raise
    finally:
        report.save()
        resumo = {k: v for k, v in report.data.items() if k not in ("config", "traceback")}
        print("AIVS_RELATORIO", json.dumps(resumo, ensure_ascii=False, default=str), flush=True)
    return report.data
