"""Vídeo final sobre o original em resolução cheia (finish.py), com o rastreador demo e um "Kaggle" falso."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_video, source_view
from studio import finish, media, wan
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
    """Quadro `idx` do CLIPE no original, no enquadramento do final (ver conftest.source_view)."""
    with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
        indices = json.loads(zf.read("job.json"))["source_frame_indices"]
    if idx in indices:
        return source_view(root, indices.index(idx), size)
    # Quadro do clipe que não está no pacote (sincronia): troca o índice do pacote 0 por ele.
    with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
        job = json.loads(zf.read("job.json"))
    job["source_frame_indices"][0] = idx
    tmp = root / "exports" / "pacote-teste.zip"
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("job.json", json.dumps(job))
    backup = root / "exports" / "wan_package.zip"
    backup.rename(root / "exports" / "pacote-original.zip")
    tmp.rename(backup)
    try:
        return source_view(root, 0, size)
    finally:
        backup.unlink()
        (root / "exports" / "pacote-original.zip").rename(backup)


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
