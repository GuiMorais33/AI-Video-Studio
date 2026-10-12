"""Refino do rosto neste computador com um FaceFusion falso (rastreador demo, sem IA)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_video
from studio import diagnose, face_refine, kaggle_remote, media
from studio.api import create_app
from studio.projects import project_paths

FAKE_FACEFUSION = r'''
"""FaceFusion falso: guarda os argumentos e uma cópia do alvo. Modo ok: inverte as cores do alvo."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

args = sys.argv[1:]
here = Path(__file__).parent
if args == ["--version"]:
    print("FaceFusion 3.9.1 (falso)")
    sys.exit(0)
value = lambda flag: args[args.index(flag) + 1]
(here / "args.json").write_text(json.dumps(args))
shutil.copyfile(value("-t"), here / "alvo.mp4")
mode = os.environ.get("FAKE_FACEFUSION_MODE", "ok")
if mode == "spin":  # download retomado que não anda: o FaceFusion redesenha a mesma barra para sempre
    (here / "pids").write_text(str(os.getpid()))
    while True:
        sys.stdout.write("\rdownloading:  20%|##        | 9.54M/47.7M [00:01<?, ?B/s]")
        sys.stdout.flush()
        time.sleep(0.05)
if mode == "hang":  # download travado: um filho (como o curl) segura a saída aberta
    child = subprocess.Popen(["sleep", "300"])
    (here / "pids").write_text(f"{os.getpid()} {child.pid}")
    time.sleep(300)
if mode == "nsfw":  # o filtro de conteúdo recusa: a análise termina e o processamento nunca começa
    print("[FACEFUSION.CORE] processing step 1 of 1\nanalysing: 100%|##| 16/16 [00:01, rate=0.6]", flush=True)
    sys.exit(3)
sources = args[args.index("-s") + 1:args.index("-t")]
assert sources and all(Path(s).exists() for s in sources), sources
if mode == "noface":
    print("[FACEFUSION.FACE_SWAPPER] no source face detected!", flush=True)
    sys.exit(1)
sys.stdout.write("analysing: 100%|##| 16/16\rprocessing:  50%|#  | 8/16\rprocessing: 100%|##| 16/16\n")
sys.stdout.flush()
# Modo face: como o real, só muda o rosto (um quadrado no centro) e recomprime o resto do alvo.
effect = "drawbox=x=iw/2-24:y=ih/2-24:w=48:h=48:color=red:t=fill" if mode == "face" else "negate"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", value("-t"), "-vf", effect, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", value("-o")], check=True)
'''


@pytest.fixture
def fake_facefusion(tmp_path) -> Path:
    root = tmp_path / "facefusion"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "facefusion.py").write_text(FAKE_FACEFUSION)
    python = root / ".venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    (root / ".venv" / face_refine.INSTALL_MARKER).touch()  # o setup.sh grava ao terminar
    return root


@pytest.fixture
def ff_settings(settings, fake_facefusion):
    return dataclasses.replace(settings, facefusion_dir=fake_facefusion)


def _video(path: Path, size: str, frames: int, rate: int = 16) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
                    "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _frame(path: Path, idx: int = 0) -> np.ndarray:
    info = media.probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3).astype(np.float32)


def _cover(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(frame.astype(np.uint8))
    return np.asarray(face_refine.fit_cover(img, size, Image.Resampling.LANCZOS), np.float32)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _portrait_project(client, tmp_path: Path, faces: int = 2) -> str:
    """Projeto em pé (360x640 -> pacote 464x832, 16 quadros a 16 fps) com pessoa no centro e o pacote pronto."""
    video = make_video(tmp_path / "em-pe.mp4", size="360x640")
    with open(video, "rb") as f:
        pid = client.post("/projects", files={"file": ("em-pe.mp4", f)}).json()["id"]
    client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 180, "y": 320, "label": 1})
    client.post(f"/projects/{pid}/track")
    client.app.state.jobs.wait_idle(timeout=60)
    ref = tmp_path / "personagem.png"
    Image.new("RGB", (512, 768), (40, 120, 200)).save(ref)
    with open(ref, "rb") as f:
        client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)})
    for i in range(faces):
        photo = tmp_path / f"rosto{i}.png"
        Image.new("RGB", (300, 300), (200 - 10 * i, 150, 120)).save(photo)
        with open(photo, "rb") as f:
            client.post(f"/projects/{pid}/faces", files=[("files", (photo.name, f))])
    resp = client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "fps": 16})
    client.app.state.jobs.wait_idle(timeout=120)
    assert client.get(f"/jobs/{resp.json()['id']}").json()["status"] == "done"
    return pid


def _import(client, pid: str, video: Path) -> dict:
    with open(video, "rb") as f:
        resp = client.post(f"/projects/{pid}/wan-result", files={"file": ("resultado.mp4", f)})
    assert resp.status_code == 200, resp.text
    client.app.state.jobs.wait_idle(timeout=120)
    return resp.json()


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in "ZX"


def _wait_dead(pids: list[int], timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


# --------------------------------------------------------------------------- fluxo completo

def test_import_refines_face_automatically(ff_settings, fake_facefusion, tmp_path):
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path)
        root = project_paths(ff_settings, pid).root
        result = _video(tmp_path / "resultado.mp4", "464x832", 16)
        body = _import(client, pid, result)
        assert "refino do rosto começou neste computador" in body["message"]

        detail = client.get(f"/projects/{pid}").json()
        job = detail["jobs"]["face_refine"]
        assert job["status"] == "done" and job["message"].startswith("Rosto refinado com 2 foto(s)"), job
        assert detail["active_job"] is None
        refine = detail["face_refine"]
        assert refine["installed"] is True
        assert refine["result"]["ok"] is True and refine["result"]["fotos"] == 2 and refine["result"]["mascara"] is True
        assert refine["result"]["quadros"] == 16 and refine["result"]["avisos"] == []

        # FaceFusion na CPU, com as opções do notebook e as duas fotos do rosto.
        args = json.loads((fake_facefusion / "args.json").read_text())
        assert len(args[args.index("-s") + 1:args.index("-t")]) == 2
        assert args[args.index("--processors") + 1:args.index("--processors") + 3] == ["face_swapper", "face_enhancer"]
        assert args[args.index("--execution-providers") + 1] == "cpu"
        assert args[args.index("--execution-thread-count") + 1] == "4"
        assert args[args.index("--face-swapper-model") + 1] == "hyperswap_1a_256"
        assert args[args.index("--face-swapper-weight") + 1] == "0.75"
        assert args[args.index("--face-enhancer-blend") + 1] == "80"
        assert args[args.index("--download-providers") + 1] == "github"
        log = (root / "exports" / "facefusion_rosto.log").read_text()
        assert "processing: 100%" in log
        assert client.get(f"/projects/{pid}/files/facefusion_rosto.log").headers["content-type"].startswith("text/plain")

        # Só o personagem foi para o FaceFusion: fora da máscara, preto.
        target = _frame(fake_facefusion / "alvo.mp4")
        assert target.shape == (832, 464, 3)
        assert target[:64, :64].mean() < 3 and target[380:450, 200:260].mean() > 40

        # Final no tamanho dos Reels, com o áudio do clipe: fundo igual ao resultado do Wan e rosto
        # (miolo da máscara) trocado pela saída do FaceFusion.
        final = media.probe(root / "exports" / "wan_final.mp4")
        assert (final.width, final.height) == (1080, 1920) and final.has_audio
        assert media.count_frames(root / "exports" / "wan_final.mp4") == 16
        got = _frame(root / "exports" / "wan_final.mp4", 3)
        plain = _cover(_frame(result, 3), (1080, 1920))
        corner, center = (slice(0, 160), slice(0, 160)), (slice(900, 1020), slice(480, 600))
        assert _diff(got[corner], plain[corner]) < 12 < _diff(got[corner], 255 - plain[corner])
        assert _diff(got[center], 255 - plain[center]) < 25 < _diff(got[center], plain[center])
        comp = media.probe(root / "exports" / "comparativo.mp4")
        assert (comp.width, comp.height) == (2 * 540, 960) and comp.has_audio
        assert not list(root.glob("rosto-*"))  # temporários apagados


def test_manual_endpoint_and_redo(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        assert client.post(f"/projects/{pid}/face-refine").status_code == 400  # sem resultado ainda
        result = _video(tmp_path / "resultado.mp4", "464x832", 16)
        body = _import(client, pid, result)
        assert "refin" not in body["message"]
        detail = client.get(f"/projects/{pid}").json()
        assert "face_refine" not in detail["jobs"] and detail["face_refine"]["result"] is None
        root = project_paths(settings, pid).root
        plain = _frame(root / "exports" / "wan_final.mp4")

        resp = client.post(f"/projects/{pid}/face-refine")
        assert resp.status_code == 202 and resp.json()["kind"] == "face_refine"
        again = client.post(f"/projects/{pid}/face-refine")
        assert again.status_code == 409 and "andamento" in again.json()["detail"]
        assert client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 10, "y": 10}).status_code == 409
        client.app.state.jobs.wait_idle(timeout=120)
        job = client.get(f"/jobs/{resp.json()['id']}").json()
        assert job["status"] == "done" and "1 foto(s)" in job["message"]
        args = json.loads((fake_facefusion / "args.json").read_text())
        assert args[args.index("-s") + 1:args.index("-t")] == [args[args.index("-s") + 1]]  # só a imagem do personagem
        refined = _frame(root / "exports" / "wan_final.mp4")
        center = (slice(900, 1020), slice(480, 600))
        assert _diff(refined[center], plain[center]) > 60

        # Refazer com uma foto sem rosto: volta para o final sem o refino (o anterior era de outras fotos).
        monkeypatch.setenv("FAKE_FACEFUSION_MODE", "noface")
        resp = client.post(f"/projects/{pid}/face-refine")
        client.app.state.jobs.wait_idle(timeout=120)
        assert client.get(f"/jobs/{resp.json()['id']}").json()["status"] == "done"
        assert _diff(_frame(root / "exports" / "wan_final.mp4")[center], plain[center]) < 8


def test_only_the_face_is_pasted_back(ff_settings, monkeypatch, tmp_path):
    """O FaceFusion devolve o alvo (preto fora da máscara) com só o rosto trocado: o final muda só ali.
    Colar o personagem inteiro escurecia o contorno da máscara (o preto entrava pela borda suave)."""
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "face")
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        root = project_paths(settings, pid).root
        plain = _frame(root / "exports" / "wan_final.mp4", 5)  # o mesmo acabamento, sem o refino
        client.post(f"/projects/{pid}/face-refine")
        client.app.state.jobs.wait_idle(timeout=120)
        assert client.get(f"/projects/{pid}").json()["face_refine"]["result"]["ok"] is True
        got = _frame(root / "exports" / "wan_final.mp4", 5)
    face = (slice(930, 990), slice(510, 570))  # o quadrado de 48 px no centro, ampliado ~2,3x
    assert got[face][..., 0].mean() > 200 and got[face][..., 1:].mean() < 80
    diff = np.abs(got - plain).mean(axis=2)
    around = np.ones(diff.shape, bool)
    around[850:1070, 430:650] = False  # fora do rosto e da borda suave dele (a máscara vai bem além)
    assert (diff[around] > 20).mean() < 0.001, "o final mudou fora do rosto (borda da máscara?)"


def test_face_alpha_follows_what_changed():
    rng = np.random.default_rng(1)
    base = rng.integers(60, 200, (200, 160, 3)).astype(np.uint8)
    noise = np.clip(base.astype(np.int16) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    assert face_refine.face_alpha(noise, base).max() == 0  # só recompressão: nada a colar
    face = noise.copy()
    face[80:120, 60:100] = 255 - face[80:120, 60:100]
    alpha = face_refine.face_alpha(face, base)[..., 0]
    assert alpha.shape == (200, 160)
    assert alpha[90:110, 70:90].min() > 0.95  # o rosto entra inteiro
    assert alpha[:40].max() < 0.01 and alpha[:, 140:].max() < 0.01  # longe dele, o quadro do Wan


def test_kaggle_import_starts_face_refine(ff_settings, monkeypatch, tmp_path):
    """O resultado que volta do Kaggle já começa o refino na fila local (a tarefa do Kaggle segue ativa)."""
    result = _video(tmp_path / "resultado.mp4", "464x832", 16)

    def fake_run(project, root, progress, import_result):
        upload = root / ".kaggle-resultado.mp4"
        upload.write_bytes(result.read_bytes())
        return import_result(upload) + " (Kaggle: 1 min)"

    monkeypatch.setattr(kaggle_remote, "token_status", lambda: {"configured": True, "source": "teste"})
    monkeypatch.setattr(kaggle_remote, "run_on_kaggle", fake_run)
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path)
        resp = client.post(f"/projects/{pid}/kaggle")
        assert resp.status_code == 202, resp.text
        client.app.state.remote_jobs.wait_idle(timeout=60)
        client.app.state.jobs.wait_idle(timeout=120)
        jobs = client.get(f"/projects/{pid}").json()["jobs"]
        assert jobs["kaggle"]["status"] == "done" and "refino do rosto começou neste computador" in jobs["kaggle"]["message"]
        assert jobs["face_refine"]["status"] == "done", jobs["face_refine"]


def test_kaggle_running_blocks_manual_refine(ff_settings, tmp_path):
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        client.app.state.db.create_job(pid, "kaggle")
        resp = client.post(f"/projects/{pid}/face-refine")
        assert resp.status_code == 409 and "Kaggle" in resp.json()["detail"]


# --------------------------------------------------------------------------- quando o FaceFusion não consegue

@pytest.mark.parametrize(("mode", "reason"), [
    ("noface", "as fotos não têm um rosto humano detectável"),
    ("nsfw", "o filtro de conteúdo do FaceFusion barrou o vídeo"),
])
def test_failure_keeps_unrefined_final(ff_settings, monkeypatch, tmp_path, mode, reason):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", mode)
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path)
        result = _video(tmp_path / "resultado.mp4", "464x832", 16)
        _import(client, pid, result)
        detail = client.get(f"/projects/{pid}").json()
        job = detail["jobs"]["face_refine"]
        assert job["status"] == "done" and "não aplicado" in job["message"] and reason in job["message"]
        assert detail["face_refine"]["result"]["ok"] is False
        assert detail["face_refine"]["result"]["motivo"].startswith(reason)
        root = project_paths(ff_settings, pid).root
        got = _frame(root / "exports" / "wan_final.mp4")
        plain = _cover(_frame(result), (1080, 1920))
        assert _diff(got, plain) < 12  # final sem o refino


def test_inactivity_timeout_kills_the_group(ff_settings, monkeypatch, fake_facefusion, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "hang")
    settings = dataclasses.replace(ff_settings, facefusion_idle_timeout_s=1.5)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        start = time.monotonic()
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert time.monotonic() - start < 60
        job = client.get(f"/projects/{pid}").json()["jobs"]["face_refine"]
        assert job["status"] == "done" and "sem dar sinal" in job["message"], job
    pids = [int(p) for p in (fake_facefusion / "pids").read_text().split()]
    assert _wait_dead(pids), "sobrou processo do FaceFusion"


def test_repeated_progress_bar_is_not_a_sign_of_life(ff_settings, monkeypatch, fake_facefusion, tmp_path):
    """Ao retomar um download parado, o FaceFusion redesenha a mesma barra sem parar: o vigia age igual."""
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "spin")
    settings = dataclasses.replace(ff_settings, facefusion_idle_timeout_s=1.5)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        start = time.monotonic()
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert time.monotonic() - start < 60
        job = client.get(f"/projects/{pid}").json()["jobs"]["face_refine"]
        assert job["status"] == "done" and "sem dar sinal" in job["message"], job
    assert _wait_dead([int((fake_facefusion / "pids").read_text())]), "sobrou processo do FaceFusion"


def test_signal_kill_explains_memory(ff_settings):
    oom = face_refine.FaceFusionRun(code=-9, log="processing: 30%", output_ok=False)
    assert "falta de memória" in face_refine.failure_reason(oom, ff_settings)
    crash = face_refine.FaceFusionRun(code=-11, log="processing: 30%", output_ok=False)
    assert "sinal 11" in face_refine.failure_reason(crash, ff_settings)


def test_engine_shutdown_stops_facefusion(ff_settings, monkeypatch, fake_facefusion, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "hang")
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        with open(_video(tmp_path / "resultado.mp4", "464x832", 16), "rb") as f:
            client.post(f"/projects/{pid}/wan-result", files={"file": ("r.mp4", f)})
        deadline = time.monotonic() + 30
        while not (fake_facefusion / "pids").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert (fake_facefusion / "pids").exists()
        db = client.app.state.db
    # Saiu do TestClient: o motor encerrou (lifespan) e o FaceFusion tem de morrer junto.
    pids = [int(p) for p in (fake_facefusion / "pids").read_text().split()]
    assert _wait_dead(pids), "o FaceFusion continuou rodando depois de o motor encerrar"
    deadline = time.monotonic() + 30
    while (job := db.latest_jobs(pid)["face_refine"])["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
    assert job["status"] == "error" and "interrompido" in job["message"]


# --------------------------------------------------------------------------- instalação e vídeos antigos

def test_not_installed(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        body = _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert "Refino do rosto não instalado" in body["message"] and "setup.sh" in body["message"]
        detail = client.get(f"/projects/{pid}").json()
        state = detail["face_refine"]
        assert state["installed"] is False and state["result"] is None
        assert state["message"].startswith("Refino do rosto não instalado (FaceFusion não encontrado")
        assert "face_refine" not in detail["jobs"]
        resp = client.post(f"/projects/{pid}/face-refine")
        assert resp.status_code == 400 and resp.json()["detail"] == state["message"]
    info = face_refine.status(settings, deep=True)
    assert info["installed"] is False and "não encontrado" in info["reason"]


def test_old_ffmpeg_is_reported(ff_settings, monkeypatch, tmp_path):
    """Ubuntu 22.04 (ffmpeg 4.4): não adianta rodar o instalador de novo, e o refino nem começa."""
    monkeypatch.setattr(face_refine, "ffmpeg_version", lambda: (4, 4))
    for deep in (False, True):
        info = face_refine.status(ff_settings, deep=deep)
        assert info["installed"] is False and info["needs_upgrade"] is True
        assert "ffmpeg 4.4" in info["reason"] and "5.1" in info["reason"]
    message = face_refine.unavailable_message(info)
    assert "não resolve" in message and "24.04" in message and "setup.sh" not in message
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        body = _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert message in body["message"]
        detail = client.get(f"/projects/{pid}").json()
        assert "face_refine" not in detail["jobs"] and detail["face_refine"]["message"] == message


def test_broken_install_fails_the_job(ff_settings, fake_facefusion, tmp_path):
    """Instalação pela metade (Python do FaceFusion sem as dependências): a tarefa explica o que fazer."""
    (fake_facefusion / "facefusion.py").write_text("import onnxruntime_que_nao_existe\n")
    assert face_refine.status(ff_settings)["installed"] is True  # o modo simples só olha os arquivos
    deep = face_refine.status(ff_settings, deep=True)
    assert deep["installed"] is False and "dependências" in deep["reason"]
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        job = client.get(f"/projects/{pid}").json()["jobs"]["face_refine"]
        assert job["status"] == "error" and "dependências" in job["message"] and "setup.sh" in job["message"]


def test_unfinished_install_is_not_installed(ff_settings, fake_facefusion):
    """O setup.sh caiu depois de criar o .venv: sem o marcador, não conta como instalado."""
    (fake_facefusion / ".venv" / face_refine.INSTALL_MARKER).unlink()
    info = face_refine.status(ff_settings)
    assert info["installed"] is False and "não terminou" in info["reason"]
    assert "rode o instalador de novo" in face_refine.unavailable_message(info)


def test_old_1080x1920_result_uses_package_masks(ff_settings, fake_facefusion, tmp_path):
    """Resultado do notebook antigo (já em 1080x1920): as máscaras do pacote cobrem e cortam igual."""
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        result = _video(tmp_path / "antigo.mp4", "1080x1920", 16)
        _import(client, pid, result)
        detail = client.get(f"/projects/{pid}").json()
        assert detail["face_refine"]["result"]["ok"] is True and detail["face_refine"]["result"]["mascara"] is True
        target = _frame(fake_facefusion / "alvo.mp4")
        assert target.shape == (1920, 1080, 3)
        assert target[:150, :150].mean() < 3 and target[900:1020, 480:600].mean() > 40
        root = project_paths(ff_settings, pid).root
        got, plain = _frame(root / "exports" / "wan_final.mp4"), _frame(result)
        assert _diff(got[:160, :160], plain[:160, :160]) < 12
        assert _diff(got[900:1020, 480:600], 255 - plain[900:1020, 480:600]) < 25


def test_frame_count_mismatch_refines_whole_frame(ff_settings, fake_facefusion, tmp_path):
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        result = _video(tmp_path / "resultado.mp4", "464x832", 20)
        _import(client, pid, result)
        detail = client.get(f"/projects/{pid}").json()
        refine = detail["face_refine"]["result"]
        assert refine["ok"] is True and refine["mascara"] is False and "20 quadros" in refine["avisos"][0]
        assert "Aviso:" in detail["jobs"]["face_refine"]["message"]
        args = json.loads((fake_facefusion / "args.json").read_text())
        assert args[args.index("-t") + 1].endswith("wan_result.mp4")
        root = project_paths(ff_settings, pid).root
        got = _frame(root / "exports" / "wan_final.mp4")
        plain = _cover(_frame(result), (1080, 1920))
        assert _diff(got[:160, :160], 255 - plain[:160, :160]) < 25  # o quadro inteiro passou pelo FaceFusion


# --------------------------------------------------------------------------- partes

def test_dilate_and_blockify_match_reference():
    rng = np.random.default_rng(0)
    mask = rng.random((60, 80)) > 0.97
    ys, xs = np.nonzero(mask)
    ref = np.zeros_like(mask)
    for y, x in zip(ys, xs):  # cv2.dilate com 7x7 x3 = quadrado de raio 9
        ref[max(0, y - 9):y + 10, max(0, x - 9):x + 10] = True
    assert np.array_equal(face_refine.dilate(mask, 7, 3), ref)
    one = np.zeros((48, 64), bool)
    one[17, 33] = True
    out = face_refine.blockify(one, 16)
    assert out[16:32, 32:48].all() and out.sum() == 256


@pytest.mark.parametrize(("src", "dst"), [((464, 832), (1080, 1920)), ((832, 464), (1920, 1080)), ((464, 832), (464, 832))])
def test_fit_cover_sizes(src, dst):
    img = Image.new("L", src)
    assert face_refine.fit_cover(img, dst, Image.Resampling.NEAREST).size == dst


def test_diagnose_reports_facefusion(settings, ff_settings):
    missing = diagnose.collect(settings)
    assert missing["ml"]["facefusion"]["installed"] is False
    assert any("Refino do rosto" in t and "setup.sh" in t for t in missing["recommendations"])
    assert "FaceFusion" in diagnose.format_report(missing)
    ok = diagnose.collect(ff_settings)
    assert ok["ml"]["facefusion"]["installed"] is True and "3.9.1" in ok["ml"]["facefusion"]["version"]
    assert not any("Refino do rosto" in t and "não instalado" in t for t in ok["recommendations"])
    assert any("1,7 GB" in t for t in ok["recommendations"])  # modelos ainda não baixados
