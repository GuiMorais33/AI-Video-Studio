"""Ampliação do resultado com o Real-ESRGAN no notebook (só precisa do PyTorch, na CPU).

Os pesos de verdade (RealESRGAN_x2plus.pth, 64 MB) não são baixados aqui: uma rede minúscula no mesmo
formato de chaves do basicsr ("params_ema") faz o papel. Com AIVS_REALESRGAN apontando para o arquivo
real, um teste extra confere que ele carrega e amplia uma imagem de verdade.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aivs_wan  # noqa: E402


def tiny_rrdbnet(path: Path, scale: int = 2, nf: int = 8, gc: int = 4, blocks: int = 2) -> Path:
    """Pesos aleatórios com os nomes e formatos do RRDBNet do basicsr (como os .pth do Real-ESRGAN)."""
    gen = torch.Generator().manual_seed(0)
    in_ch = {4: 3, 2: 12, 1: 48}[scale]

    def conv(out_ch: int, in_c: int) -> dict:
        return {"weight": torch.randn(out_ch, in_c, 3, 3, generator=gen) * 0.05, "bias": torch.zeros(out_ch)}

    state = {}
    layers = {"conv_first": conv(nf, in_ch)}
    for b in range(blocks):
        for r in range(1, 4):
            for k in range(1, 6):
                layers[f"body.{b}.rdb{r}.conv{k}"] = conv(nf if k == 5 else gc, nf + (k - 1) * gc)
    for name in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        layers[name] = conv(nf, nf)
    layers["conv_last"] = conv(3, nf)
    for name, params in layers.items():
        for kind, tensor in params.items():
            state[f"{name}.{kind}"] = tensor
    torch.save({"params_ema": state}, path)
    return path


def _frames(count: int = 2, size: tuple[int, int] = (48, 32)) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8) for _ in range(count)]


@pytest.mark.parametrize(("scale", "blocks"), [(2, 2), (4, 1), (1, 1)])
def test_rrdbnet_shape_comes_from_the_weights(tmp_path, scale, blocks):
    net = aivs_wan.load_rrdbnet(str(tiny_rrdbnet(tmp_path / "m.pth", scale=scale, blocks=blocks)))
    assert net.scale == scale and len(net.body) == blocks
    with torch.inference_mode():
        out = net(torch.rand(1, 3, 32, 48))
    assert out.shape == (1, 3, 32 * scale, 48 * scale)


def test_upscale_frames_doubles_each_frame(tmp_path):
    model = tiny_rrdbnet(tmp_path / "x2.pth")
    cfg = aivs_wan.Config(upscale=2, upscale_model=str(model), upscale_sha256=None, upscale_strength=1.0)
    seen = []
    out = aivs_wan.upscale_frames(_frames(3), cfg, "cpu", tmp_path / "cache", on_frame=seen.append)
    assert seen == [0, 1, 2]
    assert [f.shape for f in out] == [(64, 96, 3)] * 3 and all(f.dtype == np.uint8 for f in out)


def test_strength_mixes_with_the_plain_upscale(tmp_path):
    """Força 0 = ampliação simples (Lanczos); 0,5 = metade do caminho até o Real-ESRGAN puro."""
    model = str(tiny_rrdbnet(tmp_path / "x2.pth"))
    frames = _frames(1)
    run = lambda k: aivs_wan.upscale_frames(frames, aivs_wan.Config(upscale_model=model, upscale_sha256=None,
                                                                   upscale_strength=k), "cpu", tmp_path)[0]
    plain = np.asarray(Image.fromarray(frames[0]).resize((96, 64), Image.LANCZOS))
    pure, half = run(1.0).astype(np.float32), run(0.5).astype(np.float32)
    assert np.array_equal(run(0.0), plain)
    assert np.abs(half - (plain + pure) / 2).max() <= 1


def test_x4_model_is_resized_to_2x(tmp_path):
    model = tiny_rrdbnet(tmp_path / "x4.pth", scale=4)
    cfg = aivs_wan.Config(upscale=2, upscale_model=str(model), upscale_sha256=None)
    out = aivs_wan.upscale_frames(_frames(1), cfg, "cpu", tmp_path / "cache")
    assert out[0].shape == (64, 96, 3)


def test_fetch_url_checks_the_hash(tmp_path):
    data = tmp_path / "pesos.pth"
    data.write_bytes(b"pesos")
    url = data.resolve().as_uri()  # file://: "baixa" sem internet
    good = hashlib.sha256(b"pesos").hexdigest()
    cache = tmp_path / "cache"
    assert aivs_wan.fetch_url(str(data), cache, good) == str(data)  # caminho local volta como está
    got = aivs_wan.fetch_url(url, cache, good)
    assert Path(got).read_bytes() == b"pesos" and Path(got).parent == cache / "urls"
    # Já no cache e certo: não baixa de novo.
    data.unlink()
    assert aivs_wan.fetch_url(url, cache, good) == got
    # Hash diferente: erro, e nada pela metade no cache.
    other = tmp_path / "outro.pth"
    other.write_bytes(b"corrompido")
    with pytest.raises(ValueError, match="SHA-256"):
        aivs_wan.fetch_url(other.resolve().as_uri(), cache, good, retry_wait=0)
    assert not (cache / "urls" / "outro.pth").exists() and not list((cache / "urls").glob("*.part"))


@pytest.mark.skipif(not os.environ.get("AIVS_REALESRGAN"), reason="AIVS_REALESRGAN não aponta para os pesos reais")
def test_real_weights_upscale_a_real_image(tmp_path):
    """Com o RealESRGAN_x2plus.pth de verdade: carrega estrito e a ampliação parece a imagem (não ruído)."""
    path = os.environ["AIVS_REALESRGAN"]
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == aivs_wan.Config.upscale_sha256
    yy, xx = np.mgrid[0:64, 0:96]
    img = np.stack([xx * 2.5, yy * 3.5, (xx + yy) * 1.5], axis=2).clip(0, 255).astype(np.uint8)
    cfg = aivs_wan.Config(upscale=2, upscale_model=path)
    out = aivs_wan.upscale_frames([img], cfg, "cpu", tmp_path / "cache")[0]
    ref = np.asarray(Image.fromarray(img).resize((192, 128), Image.LANCZOS), np.float32)
    assert out.shape == (128, 192, 3)
    assert np.corrcoef(out.astype(np.float32).ravel(), ref.ravel())[0, 1] > 0.98
