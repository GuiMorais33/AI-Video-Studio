"""RIFE (quadros intermediários) na CPU, com uma rede minúscula no formato dos pesos reais.

Com AIVS_RIFE apontando para o flownet_v4.6.pkl de verdade e AIVS_RIFE_VIDEO para um vídeo com movimento,
um teste extra confere que o quadro do meio recriado fica mais perto do verdadeiro que misturar os dois.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studio import rife  # noqa: E402


def tiny_rife(path: Path, width: int = 8, encoded: bool = False) -> Path:
    """Pesos aleatórios com os nomes e formatos do IFNet do Practical-RIFE (v4.6, ou v4.7+ com `encoded`)."""
    gen = torch.Generator().manual_seed(0)
    first = 15 if encoded else 7
    state = {}

    def put(name: str, *shape: int) -> None:
        state[name] = torch.randn(*shape, generator=gen) * 0.05

    for b in range(4):
        cin, c = (first if b == 0 else first + 5), width
        put(f"block{b}.conv0.0.0.weight", c // 2, cin, 3, 3)
        put(f"block{b}.conv0.0.0.bias", c // 2)
        put(f"block{b}.conv0.1.0.weight", c, c // 2, 3, 3)
        put(f"block{b}.conv0.1.0.bias", c)
        for r in range(8):
            state[f"block{b}.convblock.{r}.beta"] = torch.ones(1, c, 1, 1)
            put(f"block{b}.convblock.{r}.conv.weight", c, c, 3, 3)
            put(f"block{b}.convblock.{r}.conv.bias", c)
        put(f"block{b}.lastconv.0.weight", c, 24, 4, 4)
        put(f"block{b}.lastconv.0.bias", 24)
    if encoded:
        put("encode.0.weight", 16, 3, 3, 3)
        put("encode.0.bias", 16)
        put("encode.1.weight", 16, 4, 4, 4)
        put("encode.1.bias", 4)
    # Como o arquivo oficial da v4.6: chaves com o prefixo "module." do DataParallel.
    torch.save({f"module.{k}" if not encoded else k: v for k, v in state.items()}, path)
    return path


def _frames(h: int = 70, w: int = 90) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8), rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("encoded", [False, True], ids=["v4.6", "v4.7+"])
def test_interpolates_any_size(tmp_path, encoded):
    net = rife.Interpolator(tiny_rife(tmp_path / "rife.pkl", encoded=encoded))
    a, b = _frames()  # 70x90: precisa completar até múltiplos de 64 e cortar de volta
    out = net(a, b, 0.5)
    assert out.shape == a.shape and out.dtype == np.uint8
    assert net(a, b, 0.0) is a and net(a, b, 1.0) is b


def test_unknown_format_is_refused(tmp_path):
    path = tiny_rife(tmp_path / "rife.pkl")
    state = torch.load(path, weights_only=True)
    state["module.block0.conv0.0.0.weight"] = torch.zeros(4, 9, 3, 3)
    state["module.block1.conv0.0.0.weight"] = torch.zeros(4, 14, 3, 3)
    state["module.block2.conv0.0.0.weight"] = torch.zeros(4, 14, 3, 3)
    state["module.block3.conv0.0.0.weight"] = torch.zeros(4, 14, 3, 3)
    torch.save(state, path)
    with pytest.raises(rife.RifeError, match="formato desconhecido"):
        rife.build(path)


def test_fetch_checks_the_hash(tmp_path, monkeypatch):
    weights = tiny_rife(tmp_path / "pesos.pkl")
    good = rife._sha256(weights)
    model = rife.RifeModel("pesos.pkl", weights.resolve().as_uri(), good)  # file://: "baixa" sem internet
    got = rife.fetch(tmp_path / "models", model)
    assert got == tmp_path / "models" / "rife" / "pesos.pkl" and rife._sha256(got) == good
    weights.unlink()
    assert rife.fetch(tmp_path / "models", model) == got  # já no cache: não baixa de novo
    bad = rife.RifeModel("outro.pkl", got.resolve().as_uri(), "0" * 64)
    with pytest.raises(rife.RifeError, match="não confere"):
        rife.fetch(tmp_path / "models", bad)
    assert not list((tmp_path / "models" / "rife").glob("*.part"))
    missing = rife.RifeModel("x.pkl", (tmp_path / "nao-existe.pkl").resolve().as_uri(), good)
    with pytest.raises(rife.RifeError, match="não foi possível baixar"):
        rife.fetch(tmp_path / "models", missing)


@pytest.mark.skipif(not (os.environ.get("AIVS_RIFE") and os.environ.get("AIVS_RIFE_VIDEO")),
                    reason="AIVS_RIFE/AIVS_RIFE_VIDEO não apontam para os pesos e um vídeo reais")
def test_real_weights_beat_blending():
    """Pesos reais: o quadro do meio recriado dos vizinhos bate mais com o verdadeiro que a mistura."""
    path, video = Path(os.environ["AIVS_RIFE"]), os.environ["AIVS_RIFE_VIDEO"]
    assert rife._sha256(path) == rife.MODEL.sha256
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", video], check=True, capture_output=True, text=True).stdout
    w, h = (int(v) for v in size.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", "select=between(n\\,59\\,61)", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    a, mid, b = (np.frombuffer(raw, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in range(3))

    def psnr(x: np.ndarray) -> float:
        return float(10 * np.log10(255 ** 2 / np.mean((x.astype(np.float64) - mid) ** 2)))

    made = rife.Interpolator(path)(a, b, 0.5)
    blend = ((a.astype(np.float32) + b) / 2 + 0.5).astype(np.uint8)
    assert psnr(made) > psnr(blend) + 1.5, (psnr(made), psnr(blend))
