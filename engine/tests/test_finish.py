"""Vídeo final sobre o original em resolução cheia (finish.py), com o rastreador demo e um "Kaggle" falso."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import clip_view, make_bt709_video, make_video
from studio import finish, media, rife, wan
from studio.api import create_app
from studio.projects import project_paths


def _frame(path: Path, idx: int) -> np.ndarray:
    info = media.probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3).astype(np.float32)


def _project(client, tmp_path: Path, size: str = "720x1280") -> str:
    """Projeto em pé com o original em resolução maior que o clipe (720x1280 -> clipe 480x854 ->
    pacote 464x832, 16 quadros a 16 fps) e a pessoa no centro."""
    video = make_video(tmp_path / "original.mp4", size=size)
    with open(video, "rb") as f:
        pid = client.post("/projects", files={"file": ("original.mp4", f)}).json()["id"]
    project = client.get(f"/projects/{pid}").json()
    client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": project["width"] // 2,
                                                 "y": project["height"] // 2, "label": 1})
    client.post(f"/projects/{pid}/track")
    client.app.state.jobs.wait_idle(timeout=60)
    ref = tmp_path / "personagem.png"
    Image.new("RGB", (512, 768), (40, 120, 200)).save(ref)
    with open(ref, "rb") as f:
        client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)})
    resp = client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "fps": 16})
    client.app.state.jobs.wait_idle(timeout=120)
    assert client.get(f"/jobs/{resp.json()['id']}").json()["status"] == "done"
    return pid


def _kaggle_result(root: Path, out: Path, scale: int = 1) -> Path:
    """Como o notebook entrega: o vídeo do pacote com a área do personagem (máscara da geração) pintada
    de vermelho, opcionalmente ampliado (Real-ESRGAN)."""
    with media.FrameReader(root / "exports" / "wan_input.mp4", 464, 832) as reader:
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            writer = media.FrameWriter(out, 464 * scale, 832 * scale, media.probe(root / "exports" / "wan_input.mp4").fps,
                                       crf=12)
            i = 0
            while (frame := reader.read()) is not None:
                alpha = finish.soft_alpha(zf.read(f"masks/{i:05d}.png"), (464, 832), (0, 0, 464, 832))
                painted = frame + alpha * (np.array([230, 20, 20], np.float32) - frame)
                img = Image.fromarray(np.clip(painted + 0.5, 0, 255).astype(np.uint8))
                writer.write(np.asarray(img.resize((464 * scale, 832 * scale), Image.Resampling.LANCZOS)))
                i += 1
            writer.close()
    return out


def _import(client, pid: str, video: Path) -> dict:
    with open(video, "rb") as f:
        resp = client.post(f"/projects/{pid}/wan-result", files={"file": ("resultado.mp4", f)})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _source_view(root: Path, idx: int, size: tuple[int, int] = (1080, 1920)) -> np.ndarray:
    """Quadro `idx` do CLIPE no original, no enquadramento do final (ver conftest.clip_view)."""
    return clip_view(root, idx, size)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@pytest.mark.parametrize("scale", [1, 2])
def test_final_is_composited_on_the_full_resolution_original(settings, tmp_path, scale):
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        body = _import(client, pid, _kaggle_result(root, tmp_path / "resultado.mp4", scale))
        assert "Fundo: o seu vídeo original em resolução cheia" in body["message"], body["message"]

        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["original"] == [720, 1280] and info["tamanho"] == [1080, 1920]
        assert info["pessoa"] == [464 * scale, 832 * scale] and info["diferenca_pacote"] < finish.MATCH_TOLERANCE
        assert client.get(f"/projects/{pid}").json()["wan_final_info"]["fundo"] == "original"

        final = root / "exports" / "wan_final.mp4"
        meta = media.probe(final)
        assert (meta.width, meta.height) == (1080, 1920) and meta.has_audio
        assert media.count_frames(final) == 16
        tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=color_space,color_primaries,color_transfer", "-of", "csv=p=0", str(final)],
                              check=True, capture_output=True, text=True).stdout.strip()
        assert tags == "bt709,bt709,bt709"

        # Quadro 5 do pacote = quadro 9 do clipe (30 -> 16 fps): o fundo é o do original nesse instante.
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            indices = json.loads(zf.read("job.json"))["source_frame_indices"]
        assert indices[5] == 9
        got = _frame(final, 5)
        original = _source_view(root, indices[5])
        # O final antigo: o quadro do pacote (clipe reduzido) ampliado, no mesmo corte central.
        gen = Image.fromarray(_frame(root / "exports" / "wan_input.mp4", 5).astype(np.uint8))
        view_h = 1920 / max(1080 / 464, 1920 / 832)
        old = np.asarray(gen.resize((1080, 1920), Image.Resampling.LANCZOS,
                                    box=(0, (832 - view_h) / 2, 464, (832 + view_h) / 2)), np.float32)
        # O quadriculado com ruído fino (canto de baixo, fora da pessoa): bate com o original em
        # resolução cheia, bem mais do que com o clipe reduzido ampliado (o final antigo).
        detail = (slice(1220, 1660), slice(700, 920))
        # (ruído puro: o H.264 do final perde parte dele, daí a folga; medido 12 x 19 do final antigo)
        assert _diff(got[detail], original[detail]) < 15, _diff(got[detail], original[detail])
        assert _diff(got[detail], original[detail]) * 1.5 < _diff(got[detail], old[detail])
        # O ruído do quadriculado muda a cada quadro: o quadro vizinho do original não bate (sincronia).
        other = _source_view(root, indices[5] + 1)
        assert _diff(got[detail], original[detail]) * 1.4 < _diff(got[detail], other[detail])
        # No miolo da máscara, a pessoa do resultado (vermelho).
        center = got[900:1020, 480:600].mean(axis=(0, 1))
        assert center[0] > 200 and center[1] < 50 and center[2] < 50, center


def test_mismatched_original_falls_back_to_the_old_final(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        result = _kaggle_result(root, tmp_path / "resultado.mp4")
        # Outro vídeo no lugar do original (mesmo tamanho): não bate com o pacote.
        source = next(root.glob("source*"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "mandelbrot=size=720x1280:rate=30",
                        "-t", "1", "-pix_fmt", "yuv420p", str(tmp_path / "outro.mp4")], check=True)
        shutil.copyfile(tmp_path / "outro.mp4", source)
        body = _import(client, pid, result)
        info = finish.read_info(root)
        assert info["fundo"] == "reduzido" and "não bate" in info["motivo"], info
        assert "Fundo ampliado do vídeo do Wan" in body["message"]
        final = root / "exports" / "wan_final.mp4"
        assert (media.probe(final).width, media.probe(final).height) == (1080, 1920)
        assert media.count_frames(final) == 16 and not list((root / "exports").glob("*.tmp.mp4"))


def test_missing_original_falls_back(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        result = _kaggle_result(root, tmp_path / "resultado.mp4")
        next(root.glob("source*")).unlink()
        _import(client, pid, result)
        info = finish.read_info(root)
        assert info["fundo"] == "reduzido" and "original não está" in info["motivo"]
        assert media.count_frames(root / "exports" / "wan_final.mp4") == 16


def test_wrong_frame_count_falls_back(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        short = tmp_path / "curto.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(_kaggle_result(root, tmp_path / "r.mp4")),
                        "-frames:v", "10", "-pix_fmt", "yuv420p", str(short)], check=True)
        _import(client, pid, short)
        info = finish.read_info(root)
        assert info["fundo"] == "reduzido" and "10 quadros" in info["motivo"]


def test_landscape_original_with_odd_framing(settings, tmp_path):
    """Deitado e com recorte do pacote nos dois eixos (1280x720 -> clipe 854x480 -> pacote 832x464)."""
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path, size="1280x720")
        root = project_paths(settings, pid).root
        with media.FrameReader(root / "exports" / "wan_input.mp4", 832, 464) as reader:
            writer = media.FrameWriter(tmp_path / "r.mp4", 832, 464, media.probe(root / "exports" / "wan_input.mp4").fps)
            while (frame := reader.read()) is not None:
                writer.write(frame)
            writer.close()
        _import(client, pid, tmp_path / "r.mp4")
        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["tamanho"] == [1920, 1080], info
        assert info["diferenca_pacote"] < finish.MATCH_TOLERANCE


def test_rotated_phone_video(settings, tmp_path):
    """Celular com rotação nos metadados (gravado 1280x720, mostrado 720x1280): o original é lido já girado."""
    with TestClient(create_app(settings)) as client:
        video = make_video(tmp_path / "celular.mp4", size="1280x720", rotation=90)
        with open(video, "rb") as f:
            pid = client.post("/projects", files={"file": ("celular.mp4", f)}).json()["id"]
        project = client.get(f"/projects/{pid}").json()
        assert project["height"] > project["width"]
        client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": project["width"] // 2,
                                                     "y": project["height"] // 2, "label": 1})
        client.post(f"/projects/{pid}/track")
        client.app.state.jobs.wait_idle(timeout=60)
        ref = tmp_path / "p.png"
        Image.new("RGB", (512, 768), (40, 120, 200)).save(ref)
        with open(ref, "rb") as f:
            client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)})
        client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "fps": 16})
        client.app.state.jobs.wait_idle(timeout=120)
        root = project_paths(settings, pid).root
        _import(client, pid, _kaggle_result(root, tmp_path / "r.mp4"))
        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["original"] == [720, 1280], info
        assert info["diferenca_pacote"] < finish.MATCH_TOLERANCE


# --------------------------------------------------------------------------- 30 fps (RIFE)

@pytest.fixture
def rife_settings(settings, tmp_path, monkeypatch):
    """Interpolação ligada com um RIFE minúsculo local (file://, sem internet)."""
    pytest.importorskip("torch")
    from test_rife import tiny_rife

    weights = tiny_rife(tmp_path / "rife-minusculo.pkl")
    monkeypatch.setattr(rife, "MODEL", rife.RifeModel("rife-minusculo.pkl", weights.resolve().as_uri(),
                                                      rife._sha256(weights)))
    return dataclasses.replace(settings, interpolate=True)


def test_final_at_clip_fps_with_rife(rife_settings, tmp_path):
    """Clipe a 30 fps, pacote a 16: o final volta a 30 fps, com o RIFE criando a pessoa nos quadros que
    o Wan não gerou, sobre o quadro certo do original."""
    with TestClient(create_app(rife_settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(rife_settings, pid).root
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            indices = json.loads(zf.read("job.json"))["source_frame_indices"]
        assert indices[:3] == [0, 2, 4] and indices[-1] == 28 and len(indices) == 16
        _import(client, pid, _kaggle_result(root, tmp_path / "resultado.mp4"))
        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["fps"] == "30/1", info
        assert info["quadros"] == 29 and info["interpolados"] == 13 and "sem_interpolacao" not in info
        assert (rife_settings.models_dir / "rife" / "rife-minusculo.pkl").exists()
        final = root / "exports" / "wan_final.mp4"
        meta = media.probe(final)
        assert meta.fps == 30 and media.count_frames(final) == 29 and meta.has_audio
        assert media.probe(root / "exports" / "comparativo.mp4").fps == 30

        # Quadro 1 do clipe: não está no pacote (criado pelo RIFE entre os quadros 0 e 2).
        got = _frame(final, 1)
        detail = (slice(1220, 1660), slice(700, 920))
        here, before, after = (_source_view(root, k) for k in (1, 0, 2))
        assert _diff(got[detail], here[detail]) * 1.4 < min(_diff(got[detail], before[detail]),
                                                            _diff(got[detail], after[detail]))
        center = got[900:1020, 480:600].mean(axis=(0, 1))
        assert center[0] > 180 and center[1] < 70 and center[2] < 70, center  # a pessoa (vermelho), não o original


def test_rife_failure_falls_back_to_package_fps(rife_settings, tmp_path, monkeypatch):
    def broken(self, a, b, t):
        raise RuntimeError("sem memória")

    monkeypatch.setattr(rife.Interpolator, "__call__", broken)
    with TestClient(create_app(rife_settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(rife_settings, pid).root
        _import(client, pid, _kaggle_result(root, tmp_path / "resultado.mp4"))
        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["fps"] == "16/1" and info["interpolados"] == 0
        assert "RIFE falhou" in info["sem_interpolacao"] and "sem memória" in info["sem_interpolacao"]
        assert media.count_frames(root / "exports" / "wan_final.mp4") == 16


def test_rife_unavailable_keeps_package_fps(rife_settings, tmp_path, monkeypatch):
    def offline(models_dir, model=None):
        raise rife.RifeError("não foi possível baixar os pesos do RIFE (sem internet)")

    monkeypatch.setattr(rife, "fetch", offline)
    with TestClient(create_app(rife_settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(rife_settings, pid).root
        _import(client, pid, _kaggle_result(root, tmp_path / "resultado.mp4"))
        info = finish.read_info(root)
        assert info["fps"] == "16/1" and "sem internet" in info["sem_interpolacao"]


def test_interpolation_off_by_setting(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        _import(client, pid, _kaggle_result(root, tmp_path / "resultado.mp4"))
        info = finish.read_info(root)
        assert info["fps"] == "16/1" and info["sem_interpolacao"] == "interpolação desligada"


# --------------------------------------------------------------------------- cores (BT.709)

def _phone_final(settings, tmp_path, legacy_frames: bool) -> tuple[Path, dict, list[int]]:
    with TestClient(create_app(settings)) as client:
        video = make_bt709_video(tmp_path / "celular.mp4")
        with open(video, "rb") as f:
            pid = client.post("/projects", files={"file": ("celular.mp4", f)}).json()["id"]
        root = project_paths(settings, pid).root
        if legacy_frames:
            (root / "frames" / media.FAITHFUL_COLORS_MARKER).unlink()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(root / "clip.mp4"), "-q:v", "2", "-start_number", "0",
                            str(root / "frames" / "%05d.jpg")], check=True)
        project = client.get(f"/projects/{pid}").json()
        client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": project["width"] // 2,
                                                     "y": project["height"] // 2, "label": 1})
        client.post(f"/projects/{pid}/track")
        client.app.state.jobs.wait_idle(timeout=60)
        ref = tmp_path / "p.png"
        Image.new("RGB", (512, 768), (40, 120, 200)).save(ref)
        with open(ref, "rb") as f:
            client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)})
        client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "fps": 16})
        client.app.state.jobs.wait_idle(timeout=120)
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            job = json.loads(zf.read("job.json"))
        _import(client, pid, _kaggle_result(root, tmp_path / "r.mp4"))
        return root, job, job["source_frame_indices"]


def test_final_keeps_the_colors_of_a_phone_video(settings, tmp_path):
    """Vídeo de celular (BT.709): o final tem as cores do original como um player mostra."""
    root, job, indices = _phone_final(settings, tmp_path, legacy_frames=False)
    assert job["cores"] == "fieis"
    info = finish.read_info(root)
    # Medido: ~3,1 com a leitura certa; 8 com a trocada (BT.601), o que já dispara a reserva.
    assert info["fundo"] == "original" and info["diferenca_pacote"] < 4.5, info
    got = _frame(root / "exports" / "wan_final.mp4", 5)
    truth = clip_view(root, indices[5], (1080, 1920))  # o ffmpeg lê o original pela marcação BT.709
    bars = (slice(40, 700), slice(0, 1080))  # faixas de cores fortes, longe da pessoa
    assert _diff(got[bars], truth[bars]) < 4, _diff(got[bars], truth[bars])
    assert np.abs((got[bars] - truth[bars]).mean(axis=(0, 1))).max() < 1.5


def test_old_projects_keep_matching_colors(settings, tmp_path):
    """Projeto antigo (JPEG sem converter): o original é lido como antes (BT.601), e o fundo continua
    batendo com a pessoa gerada."""
    root, job, _ = _phone_final(settings, tmp_path, legacy_frames=True)
    assert job["cores"] == "bt601"
    info = finish.read_info(root)
    # Medido: ~3,2 com a leitura antiga (certa para este projeto); 9 lendo pela marcação.
    assert info["fundo"] == "original" and info["diferenca_pacote"] < 4.5, info


# --------------------------------------------------------------------------- borda (aura)

def test_band_around_the_person_comes_from_the_sharp_original(settings, tmp_path):
    """Na faixa da máscara da geração em volta da pessoa, onde o Wan redesenhou o fundo quase igual
    (aqui, 6 níveis mais claro), vale o original nítido; o personagem, mesmo além da silhueta
    original, continua inteiro."""
    from studio.masks import generation_mask

    with TestClient(create_app(settings)) as client:
        pid = _project(client, tmp_path)
        root = project_paths(settings, pid).root
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            raw = [np.asarray(Image.open(BytesIO(zf.read(f"masks/{i:05d}.png")))) > 127 for i in range(16)]
        yy, xx = np.mgrid[0:832, 0:464]
        rings = {}
        result = tmp_path / "resultado.mp4"
        fps = media.probe(root / "exports" / "wan_input.mp4").fps
        writer = media.FrameWriter(result, 464, 832, fps, crf=10)
        with media.FrameReader(root / "exports" / "wan_input.mp4", 464, 832) as reader:
            for i in range(16):
                frame = reader.read().astype(np.float32)
                cy, cx = np.argwhere(raw[i]).mean(axis=0)
                radius = np.sqrt(raw[i].sum() / np.pi)
                dist = np.hypot(yy - cy, xx - cx)
                character = dist <= radius + 7  # passa 7 px da silhueta original (cabelo, roupa larga)
                area = generation_mask(raw[i])
                out = np.where(area[..., None], np.clip(frame + 6, 0, 255), frame)  # faixa: quase igual
                out[character] = (230, 20, 20)
                writer.write(np.clip(out + 0.5, 0, 255).astype(np.uint8))
                if i == 5:
                    rings["faixa"] = area & (dist > radius + 16)          # longe do personagem
                    rings["alem"] = (dist > radius + 2) & (dist < radius + 5)  # personagem fora da silhueta
        writer.close()
        _import(client, pid, result)

        info = finish.read_info(root)
        assert info["fundo"] == "original" and info["borda_original"] > 0.2, info
        got = _frame(root / "exports" / "wan_final.mp4", 5)
        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            indices = json.loads(zf.read("job.json"))["source_frame_indices"]
        original = _source_view(root, indices[5])
        view_h = 1920 / max(1080 / 464, 1920 / 832)
        box = (0, (832 - view_h) / 2, 464, (832 + view_h) / 2)

        def in_final(mask: np.ndarray) -> np.ndarray:
            img = Image.fromarray(mask.astype(np.uint8) * 255).resize((1080, 1920), Image.Resampling.NEAREST, box=box)
            return np.asarray(img) > 127

        band, beyond = in_final(rings["faixa"]), in_final(rings["alem"])
        assert band.sum() > 2000 and beyond.sum() > 500
        # Faixa: o original (sem os +6 do Wan); antes desta mudança, a média ficava em ~+6.
        bias = float((got[band] - original[band]).mean())
        assert abs(bias) < 2, bias
        # Além da silhueta original, o personagem continua (vermelho), não o fundo.
        red = got[beyond].mean(axis=0)
        assert red[0] > 170 and red[1] < 90 and red[2] < 90, red
