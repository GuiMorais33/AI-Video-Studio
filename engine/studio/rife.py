"""Quadros intermediários com o RIFE (IFNet, Practical-RIFE de hzwer, licença MIT), na CPU.

O Wan gera em 16 fps; o vídeo filmado costuma ter 30. Repetir quadros deixa o movimento picotado, e
misturar dois quadros deixa fantasmas. O RIFE estima o movimento entre dois quadros e cria o do meio
(ou qualquer instante entre eles).

A rede é a do Practical-RIFE reescrita aqui (sem o repositório dele), montada a partir do arquivo de
pesos: a v4.6 (sem codificador) e a v4.7/4.9 (com o codificador de detalhes "encode"). Os pesos vêm de
cópias públicas no GitHub, conferidas por SHA-256, e ficam em data/models/rife/.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RifeModel:
    name: str
    url: str
    sha256: str


# Escolhida pela medida em vídeo real (11/2026): o quadro do meio recriado a partir dos vizinhos e
# comparado com o verdadeiro (PSNR). Crianças pulando na cama (movimento rápido, 960x540): repetir o
# quadro 23,4 dB, misturar os dois 26,0, v4.6 30,5, v4.7 30,3, v4.9 30,1. Rosto falando: 36,6 / 43,2 /
# 44,4 / 44,6 / 44,2. A v4.6 ganha no movimento rápido (dança) e leva ~0,3 s por quadro em 960x540
# numa CPU de 4 núcleos. Cópia do arquivo oficial publicada pelo vs-rife (HolyWu).
MODEL = RifeModel(
    "flownet_v4.6.pkl",
    "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.6.pkl",
    "008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db",
)


class RifeError(RuntimeError):
    pass


def available() -> str | None:
    """None se dá para interpolar; senão, o motivo (sem PyTorch)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return "o PyTorch não está instalado no motor (rode o instalador de novo)"
    return None


def fetch(models_dir: Path, model: RifeModel | None = None) -> Path:
    """Arquivo de pesos no cache (baixa do GitHub na primeira vez, conferindo o SHA-256)."""
    model = model or MODEL
    dest = models_dir / "rife" / model.name
    if dest.exists() and _sha256(dest) == model.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(model.url, tmp)
        if _sha256(tmp) != model.sha256:
            raise RifeError(f"o arquivo baixado ({model.name}) não confere com o esperado")
        os.replace(tmp, dest)
    except OSError as exc:
        raise RifeError(f"não foi possível baixar os pesos do RIFE ({type(exc).__name__}: {exc})") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(path: Path):
    """IFNet com os pesos de `path` (formato do Practical-RIFE; prefixo "module." aceito)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    encoded = "encode.0.weight" in state

    grids: dict[Any, Any] = {}

    def warp(image, flow):
        key = (str(flow.device), flow.shape)
        if key not in grids:
            h, w = flow.shape[2], flow.shape[3]
            horizontal = torch.linspace(-1.0, 1.0, w, device=flow.device).view(1, 1, 1, w).expand(flow.shape[0], -1, h, -1)
            vertical = torch.linspace(-1.0, 1.0, h, device=flow.device).view(1, 1, h, 1).expand(flow.shape[0], -1, -1, w)
            grids[key] = torch.cat([horizontal, vertical], 1)
        flow = torch.cat([flow[:, 0:1] / ((image.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((image.shape[2] - 1.0) / 2.0)], 1)
        grid = (grids[key] + flow).permute(0, 2, 3, 1)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def conv(cin: int, cout: int):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1), nn.LeakyReLU(0.2, True))

    class ResConv(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.conv = nn.Conv2d(c, c, 3, 1, 1)
            self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            return self.relu(self.conv(x) * self.beta + x)

    class IFBlock(nn.Module):
        def __init__(self, cin: int, c: int):
            super().__init__()
            self.conv0 = nn.Sequential(conv(cin, c // 2), conv(c // 2, c))
            self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
            self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

        def forward(self, x, flow, scale: float):
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
                x = torch.cat((x, flow), 1)
            tmp = self.lastconv(self.convblock(self.conv0(x)))
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
            return tmp[:, :4] * scale, tmp[:, 4:5]

    class IFNet(nn.Module):
        def __init__(self):
            super().__init__()
            extra = 8 if encoded else 0  # f0 e f1 (4 canais cada) do codificador
            first = state["block0.conv0.0.0.weight"].shape[1]
            widths = [state[f"block{i}.conv0.1.0.weight"].shape[0] for i in range(4)]
            self.block0 = IFBlock(first, widths[0])
            self.block1 = IFBlock(first + 5, widths[1])
            self.block2 = IFBlock(first + 5, widths[2])
            self.block3 = IFBlock(first + 5, widths[3])
            if first != 7 + extra:
                raise RifeError(f"pesos do RIFE num formato desconhecido ({first} canais de entrada)")
            if encoded:
                self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))

        def forward(self, img0, img1, timestep: float, scale: float = 1.0):
            t = img0[:, :1] * 0 + timestep
            scales = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
            blocks = [self.block0, self.block1, self.block2, self.block3]
            f0 = f1 = None
            if encoded:
                f0, f1 = self.encode(img0), self.encode(img1)
            flow = mask = None
            warped0, warped1 = img0, img1
            for block, s in zip(blocks, scales):
                if flow is None:
                    parts = (img0, img1, f0, f1, t) if encoded else (img0, img1, t)
                    flow, mask = block(torch.cat(parts, 1), None, s)
                else:
                    if encoded:
                        parts = (warped0, warped1, warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]), t, mask)
                    else:
                        parts = (warped0, warped1, t, mask)
                    delta, m = block(torch.cat(parts, 1), flow, s)
                    flow = flow + delta
                    # v4.7+: cada bloco refaz a máscara; v4.6: soma as correções.
                    mask = m if encoded else mask + m
                warped0, warped1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
            mask = torch.sigmoid(mask)
            return warped0 * mask + warped1 * (1 - mask)

    net = IFNet()
    net.load_state_dict(state, strict=True)
    return net.eval()


class Interpolator:
    """Cria quadros entre dois quadros RGB uint8 (altura x largura x 3) do mesmo tamanho."""

    def __init__(self, path: Path, threads: int | None = None, scale: float = 1.0):
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.net = build(path)
        # Escala do fluxo: 0.5 calcula o movimento na metade da resolução (bem mais rápido em vídeo
        # grande, quase igual); o Practical-RIFE sugere 1 até 1080p e 0.5 acima disso.
        self.scale = scale

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        torch = self.torch
        if t <= 0:
            return a
        if t >= 1:
            return b
        h, w = a.shape[:2]
        step = max(64, int(64 / self.scale))
        ph, pw = -h % step, -w % step
        with torch.inference_mode():
            x0 = torch.from_numpy(np.array(a, np.float32)).permute(2, 0, 1)[None] / 255.0
            x1 = torch.from_numpy(np.array(b, np.float32)).permute(2, 0, 1)[None] / 255.0
            if ph or pw:
                x0 = torch.nn.functional.pad(x0, (0, pw, 0, ph), mode="replicate")
                x1 = torch.nn.functional.pad(x1, (0, pw, 0, ph), mode="replicate")
            out = self.net(x0, x1, float(t), self.scale)[0, :, :h, :w]
        return (out.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


ProgressFn = Callable[[float], None]
