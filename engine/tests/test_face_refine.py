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

from conftest import make_video, source_view
from studio import diagnose, face_compare, face_refine, kaggle_remote, media
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
with open(here / "calls.jsonl", "a") as calls:  # todas as chamadas, em ordem (a comparação roda várias)
    calls.write(json.dumps(args) + "\n")
image = value("-o").endswith(".png")  # imagem do personagem (reference.py), não vídeo
shutil.copyfile(value("-t"), here / ("alvo.png" if image else "alvo.mp4"))
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
# FAKE_FACEFUSION_NEIGHBOR="x:y:l:a": troca também o rosto de quem está colado no personagem.
effect = "drawbox=x=iw/2-24:y=ih/2-24:w=48:h=48:color=red:t=fill" if mode == "face" else "negate"
if mode == "face" and os.environ.get("FAKE_FACEFUSION_NEIGHBOR"):  # o "rosto" vizinho: cores invertidas
    x, y, w, h = os.environ["FAKE_FACEFUSION_NEIGHBOR"].split(":")
    effect = f"{effect},split[a][b];[b]crop={w}:{h}:{x}:{y},negate[c];[a][c]overlay={x}:{y}"
codec = [] if image else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", value("-t"), "-filter_complex", effect, *codec, value("-o")],
               check=True)
'''


# Pacote falso para o facefusion_probe.py: um rosto no centro do quadro, de frente ou de perfil.
# FAKE_FACEFUSION_POSE: "frontal" (padrão), "profile", "profile:8-15" (perfil nesses quadros) ou "crash".
FAKE_FACE_CREATOR = r'''
import os
from types import SimpleNamespace
COUNT = [0]

def get_many_faces(frames):
    index, COUNT[0] = COUNT[0], COUNT[0] + 1
    pose = os.environ.get("FAKE_FACEFUSION_POSE", "frontal")
    if pose == "crash":
        raise RuntimeError("modelo ausente")
    if ":" in pose:
        start, end = (int(v) for v in pose.split(":")[1].split("-"))
        pose = "profile" if start <= index <= end else "frontal"
    h, w = frames[0].shape[:2]
    cx, cy = w / 2, h / 2
    nose = cx + (30 if pose == "profile" else 0)
    return [SimpleNamespace(bounding_box=[cx - 24, cy - 24, cx + 24, cy + 24],
                            score_set={"detector": 0.8, "landmarker": 0.9},
                            landmark_set={"5/68": [[cx - 10, cy - 6], [cx + 10, cy - 6], [nose, cy + 4]]})]
'''


@pytest.fixture
def fake_facefusion(tmp_path) -> Path:
    root = tmp_path / "facefusion"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "facefusion.py").write_text(FAKE_FACEFUSION)
    package = root / "facefusion"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "state_manager.py").write_text("def init_item(key, value):\n    pass\n")
    (package / "face_creator.py").write_text(FAKE_FACE_CREATOR)
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


def _portrait_project(client, tmp_path: Path, faces: int = 2, face_on_reference: bool = True) -> str:
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
    resp = client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "fps": 16,
                                                             "rosto_na_referencia": face_on_reference})
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

        # Só o personagem foi para o FaceFusion: fora da máscara, preto. Ampliado 2x (464x832 -> 928x1664):
        # o rosto fica maior e o FaceFusion acerta mais detalhe, no mesmo tempo.
        target = _frame(fake_facefusion / "alvo.mp4")
        assert target.shape == (1664, 928, 3)
        assert target[:128, :128].mean() < 3 and target[760:900, 400:520].mean() > 40

        # Final no tamanho dos Reels, com o áudio do clipe: fundo do vídeo ORIGINAL (não o do resultado)
        # e rosto (miolo da máscara) trocado pela saída do FaceFusion.
        final = media.probe(root / "exports" / "wan_final.mp4")
        assert (final.width, final.height) == (1080, 1920) and final.has_audio
        assert media.count_frames(root / "exports" / "wan_final.mp4") == 16
        assert detail["wan_final_info"]["fundo"] == "original"
        got = _frame(root / "exports" / "wan_final.mp4", 3)
        plain = _cover(_frame(result, 3), (1080, 1920))
        original = source_view(root, 3, (1080, 1920))
        corner, center = (slice(0, 160), slice(0, 160)), (slice(900, 1020), slice(480, 600))
        assert _diff(got[corner], original[corner]) < 12 < _diff(got[corner], plain[corner])
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


@pytest.mark.parametrize("touching", [False, True])
def test_neighbor_faces_are_never_pasted(ff_settings, monkeypatch, tmp_path, touching):
    """Quem está colado no personagem cai na área em blocos que o FaceFusion vê e tem o rosto trocado
    lá; no vídeo final só volta o que está na silhueta da pessoa rastreada. Encostado nela (ex.: a
    mão do personagem no rosto de outra pessoa), o rosto vizinho fica de fora inteiro."""
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        root = project_paths(settings, pid).root
        masks = face_refine.PackageMasks(root / "exports" / "wan_package.zip", (464, 832))
        area, person = masks.binary(5), masks.person(5)[..., 0]
        masks.close()
        # Um "rosto" de 12x12 na área que o FaceFusion vê: fora da silhueta, ou encostado nela
        # (um quarto dentro).
        def place(x, y):
            box = person[y:y + 12, x:x + 12]
            if not area[y:y + 12, x:x + 12].all():
                return False
            return 0.15 < (box > 0.5).mean() < 0.35 if touching else box.max() < 0.01
        spots = [(x, y) for y in range(0, 820, 2) for x in range(0, 452, 2) if place(x, y)]
        assert spots, "a área em blocos deveria ir além da silhueta"
        x, y = spots[len(spots) // 2]
        monkeypatch.setenv("FAKE_FACEFUSION_MODE", "face")
        monkeypatch.setenv("FAKE_FACEFUSION_NEIGHBOR", f"{x}:{y}:12:12")
        plain = _frame(root / "exports" / "wan_final.mp4", 5)
        client.post(f"/projects/{pid}/face-refine")
        client.app.state.jobs.wait_idle(timeout=120)
        assert client.get(f"/projects/{pid}").json()["face_refine"]["result"]["ok"] is True
        got = _frame(root / "exports" / "wan_final.mp4", 5)
    face = (slice(930, 990), slice(510, 570))
    assert got[face][..., 0].mean() > 200  # o rosto do personagem foi trocado
    scale = 1080 / 464  # o final é o pacote ampliado (e cortado ~8 px em cima)
    top, left = int(y * scale - 8), int(x * scale)
    neighbor = (slice(top + 2, top + 26), slice(left + 2, left + 26))
    assert np.abs(got[neighbor] - plain[neighbor]).mean(axis=2).max() < 40, "o rosto do vizinho voltou para o vídeo"


def test_profile_frames_keep_the_wan_face(ff_settings, monkeypatch, tmp_path):
    """De perfil o FaceFusion estraga a troca: nesses quadros fica o rosto do Wan, com transição suave."""
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "face")
    monkeypatch.setenv("FAKE_FACEFUSION_POSE", "profile:8-15")
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        root = project_paths(settings, pid).root
        plain = [_frame(root / "exports" / "wan_final.mp4", i) for i in (3, 7, 12)]
        client.post(f"/projects/{pid}/face-refine")
        client.app.state.jobs.wait_idle(timeout=120)
        detail = client.get(f"/projects/{pid}").json()
        got = [_frame(root / "exports" / "wan_final.mp4", i) for i in (3, 7, 12)]
    result = detail["face_refine"]["result"]
    assert result["ok"] is True and result["quadros_refinados"] == 8
    assert any("perfil ou de costas em 8 de 16 quadros" in a for a in result["avisos"])
    face = (slice(930, 990), slice(510, 570))
    assert got[0][face][..., 0].mean() > 200  # de frente: trocado
    assert _diff(got[2][face], plain[2][face]) < 6  # de perfil: o do Wan
    # Último quadro de frente antes do perfil: troca pela metade (saída suave), nem inteira nem nada.
    assert 12 < _diff(got[1][face], plain[1][face]) < _diff(got[0][face], plain[0][face]) - 12


def test_all_profile_is_not_applied(ff_settings, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "face")
    monkeypatch.setenv("FAKE_FACEFUSION_POSE", "profile")
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        result = _video(tmp_path / "resultado.mp4", "464x832", 16)
        _import(client, pid, result)
        detail = client.get(f"/projects/{pid}").json()
        root = project_paths(ff_settings, pid).root
        got = _frame(root / "exports" / "wan_final.mp4", 5)
    job = detail["jobs"]["face_refine"]
    assert job["status"] == "done" and "não aplicado" in job["message"] and "perfil ou de costas" in job["message"]
    assert detail["face_refine"]["result"]["ok"] is False
    assert _diff(got, _cover(_frame(result, 5), (1080, 1920))) < 12


def test_probe_failure_refines_every_frame(ff_settings, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "face")
    monkeypatch.setenv("FAKE_FACEFUSION_POSE", "crash")
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, faces=0)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        result = client.get(f"/projects/{pid}").json()["face_refine"]["result"]
    assert result["ok"] is True and result["quadros_refinados"] is None
    assert any("não deu para medir o ângulo do rosto" in a for a in result["avisos"])


def test_pose_weights_runs_and_ramps():
    def face(r5, lmk=0.9):
        return [{"box": [0, 0, 10, 10], "detector": 0.8, "landmarker": lmk, "r5": r5}]
    front, side = face(0.5), face(3.0)
    seq = [side, front, side, side] + [front] * 4 + [side] + [front] * 4 + [face(0.5, lmk=0.2)] + [side] * 2
    w = face_refine.pose_weights(seq, lambda i: None, len(seq))
    assert w[1] == 0  # quadro bom isolado: trocar e destrocar piscaria
    assert w[8] > 0  # quadro ruim isolado entre bons: preenchido
    assert w[4] == pytest.approx(0.6) and w[5] == pytest.approx(0.8) and w[6] == 1.0  # entrada suave
    assert w[12] == pytest.approx(0.6) and w[13] == 0 and w[15] == 0  # saída suave; 68 pontos ruins = perfil
    everywhere = face_refine.pose_weights([front] * 5, lambda i: None, 5)
    assert everywhere == [1.0] * 5  # vídeo todo de frente: sem rampa no começo e no fim
    # Rosto fora da pessoa rastreada não conta.
    assert face_refine.pose_weights([front] * 5, lambda i: np.zeros((20, 20), bool), 5) == [0.0] * 5


def test_pose_weights_fade_as_the_head_turns():
    """A troca sai aos poucos conforme a cabeça gira (não de uma vez no limite), e volta igual."""
    def face(turn):
        return [{"box": [0, 0, 10, 10], "detector": 0.8, "landmarker": 0.9, "r5": 0.5 + turn}]
    turns = [0, 0, 0.1, 0.2, 0.3, 0.38, 0.45, 0.5, 0.6, 0.7, 0.6, 0.5, 0.45, 0.38, 0.3, 0.2, 0, 0]
    w = face_refine.pose_weights([face(x) for x in turns], lambda i: None, len(turns))
    assert w[:4] == [1.0] * 4  # de frente até um quarto: inteira
    going = w[3:10]
    assert all(a >= b for a, b in zip(going, going[1:])) and w[9] == 0  # vai saindo, sem degrau
    assert 0.2 < w[6] < 0.8  # no meio do giro, parcial (antes era 1 até o limite e 0 depois)
    assert w[8] == 0 and w[-1] == 1.0
    assert face_refine.turn_factor(face(0.3)[0]) == pytest.approx(1.0) and face_refine.turn_factor(face(0.55)[0]) == 0.0
    assert face_refine.turn_factor(face(-0.425)[0]) == pytest.approx(0.5)  # para o outro lado, igual


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


# --------------------------------------------------------------------------- memória e motor de verdade

def test_refine_releases_the_tracker_video(ff_settings, fake_facefusion, tmp_path):
    """O SAM 2 guarda o clipe inteiro (~3 GB em 5 s a 30 fps): sai da memória antes do FaceFusion."""
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, face_on_reference=False)  # o pacote não abre o FaceFusion
        session = client.app.state.session
        assert session._loaded_project == pid  # a seleção deixou o vídeo carregado
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert client.get(f"/projects/{pid}").json()["jobs"]["face_refine"]["status"] == "done"
        assert session._loaded_project is None
        # A seleção continua funcionando: o próximo clique recarrega o vídeo e as máscaras do disco.
        assert client.post(f"/projects/{pid}/clicks", json={"frame_idx": 3, "x": 180, "y": 320, "label": 1}).status_code == 200


def test_low_memory_stops_before_facefusion(ff_settings, monkeypatch, fake_facefusion, tmp_path):
    monkeypatch.setattr(face_refine, "free_memory_gb", lambda: 3.2)
    settings = dataclasses.replace(ff_settings, facefusion_min_memory_gb=5.5)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        job = client.get(f"/projects/{pid}").json()["jobs"]["face_refine"]
    assert job["status"] == "error" and "3.2 GB livres" in job["message"] and "5.5 GB" in job["message"], job
    assert "README" in job["message"]
    assert not (fake_facefusion / "args.json").exists(), "abriu o FaceFusion sem memória"


def test_free_memory_counts_swap():
    free = face_refine.free_memory_gb()
    assert free is not None and free > 0


@pytest.mark.skipif(not face_refine._pdeathsig_prefix(), reason="sem setpriv --pdeathsig")
def test_facefusion_dies_with_its_parent_and_is_the_oom_victim(tmp_path):
    """Motor morto à força (falta de memória, janela fechada): o FaceFusion não fica órfão."""
    script = tmp_path / "pai.py"
    script.write_text(
        "import os, sys, threading, time\n"
        f"sys.path.insert(0, {str(Path(face_refine.__file__).parents[1])!r})\n"
        "from studio import face_refine\n"
        "def start():\n"  # como no motor: a thread da fila de tarefas abre o FaceFusion e espera por ele
        "    proc = face_refine._spawn_child(['sleep', '300'])\n"
        "    print(proc.pid, flush=True)\n"
        "    proc.wait()\n"
        "threading.Thread(target=start).start()\n"
        "time.sleep(300)\n"
    )
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        assert _alive(child)
        assert Path(f"/proc/{child}/oom_score_adj").read_text().strip() == face_refine.OOM_SCORE_ADJ
        parent.kill()
        parent.wait()
        assert _wait_dead([child]), "o filho continuou vivo depois de o motor morrer"
    finally:
        parent.kill()
        subprocess.run(["kill", "-9", str(child)], capture_output=True)


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(settings, fake_facefusion: Path, port: int) -> subprocess.Popen:
    import os

    env = {**os.environ, "STUDIO_DATA_DIR": str(settings.data_dir), "STUDIO_TRACKER": "demo",
           "STUDIO_FACEFUSION_DIR": str(fake_facefusion), "STUDIO_FACEFUSION_MIN_MEMORY_GB": "0",
           "FAKE_FACEFUSION_MODE": "hang"}
    return subprocess.Popen([sys.executable, "-m", "studio.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _job_row(settings, kind: str) -> tuple[str, str]:
    import sqlite3

    # Direto no SQLite: abrir um Database marcaria a tarefa como interrompida.
    with sqlite3.connect(settings.data_dir / "studio.db") as conn:
        return conn.execute("SELECT status, message FROM jobs WHERE kind = ? ORDER BY created_at DESC LIMIT 1",
                            (kind,)).fetchone()


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_real_server_stops_cleanly_and_runs_once(ff_settings, fake_facefusion, tmp_path, sig):
    """`studio serve` de verdade: um 2º motor na mesma pasta não sobe (cortaria a tarefa do 1º), e
    fechar a janela (SIGHUP) ou o start.sh encerrando (SIGTERM) para o FaceFusion e grava o motivo."""
    import signal

    import httpx

    settings = ff_settings
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
    port = _free_port()
    server = _serve(settings, fake_facefusion, port)
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                httpx.get(f"{url}/health", timeout=2)
                break
            except httpx.HTTPError:
                assert server.poll() is None and time.monotonic() < deadline, server.stdout.read()
                time.sleep(0.2)
        with open(_video(tmp_path / "resultado.mp4", "464x832", 16), "rb") as f:
            # O refino do rosto começa sozinho depois da importação.
            assert httpx.post(f"{url}/projects/{pid}/wan-result", files={"file": ("r.mp4", f)}, timeout=120).status_code == 200
        deadline = time.monotonic() + 30
        while not (fake_facefusion / "pids").exists():
            assert time.monotonic() < deadline, "o FaceFusion não começou"
            time.sleep(0.1)
        ff_pids = [int(p) for p in (fake_facefusion / "pids").read_text().split()]

        second = _serve(settings, fake_facefusion, _free_port())
        out, _ = second.communicate(timeout=60)
        assert second.returncode == 3 and "já está aberto" in out, out
        assert _job_row(settings, "face_refine")[0] == "running", "o 2º motor marcou a tarefa do 1º como interrompida"

        server.send_signal(getattr(signal, sig))
        server.wait(timeout=30)
        assert _wait_dead(ff_pids[:1]), "o FaceFusion continuou rodando depois de o motor encerrar"
        status, message = _job_row(settings, "face_refine")
        assert status == "error" and "interrompido" in message, (status, message)
        assert "GET /health" not in server.stdout.read(), "log de acesso ligado"
    finally:
        server.kill()
        for p in [int(p) for p in (fake_facefusion / "pids").read_text().split()] if (fake_facefusion / "pids").exists() else []:
            subprocess.run(["kill", "-9", str(p)], capture_output=True)


def test_restart_explains_the_interruption(settings):
    from studio.db import Database

    db = Database(settings.db_path)
    project = db.create_project(id="p1", name="p", source_name="v.mp4", clip_start=0.0, width=2, height=2,
                                fps="1/1", num_frames=1, duration=1.0, has_audio=0)
    job = db.create_job(project["id"], "face_refine")
    db.update_job(job["id"], status="running")
    kaggle = db.create_job(project["id"], "kaggle")
    db.update_job(kaggle["id"], status="running")
    db = Database(settings.db_path)  # o motor subindo de novo
    jobs = db.latest_jobs("p1")
    message = jobs["face_refine"]["message"]
    assert "janela fechada, falta de memória ou computador em suspensão" in message
    assert "motor.anterior.log" in message  # o start.sh já guardou o registro da queda com esse nome
    # A geração continua no Kaggle: mandar "rodar de novo" gastaria a cota de GPU à toa.
    assert jobs["kaggle"]["status"] == "error" and "continua no Kaggle" in jobs["kaggle"]["message"]
    assert "Não envie de novo" in jobs["kaggle"]["message"]


def test_no_selection_while_refining(ff_settings, monkeypatch, fake_facefusion, tmp_path):
    """Selecionar a pessoa noutro projeto durante o refino carregaria o vídeo no SAM 2 junto com o
    FaceFusion: a combinação que estourava a memória."""
    with TestClient(create_app(ff_settings)) as client:
        a = _portrait_project(client, tmp_path)
        monkeypatch.setenv("FAKE_FACEFUSION_MODE", "hang")  # só o refino trava (o pacote já saiu)
        video = make_video(tmp_path / "outro.mp4", size="360x640")
        with open(video, "rb") as f:
            b = client.post("/projects", files={"file": ("outro.mp4", f)}).json()["id"]
        with open(_video(tmp_path / "resultado.mp4", "464x832", 16), "rb") as f:
            client.post(f"/projects/{a}/wan-result", files={"file": ("r.mp4", f)})
        deadline = time.monotonic() + 30
        while not (fake_facefusion / "pids").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        resp = client.post(f"/projects/{b}/clicks", json={"frame_idx": 0, "x": 180, "y": 320, "label": 1})
        assert resp.status_code == 409 and "refino do rosto" in resp.json()["detail"], resp.text
        assert client.app.state.session._loaded_project is None


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
        assert _diff(got[:160, :160], source_view(root, 0, (1080, 1920))[:160, :160]) < 12  # fundo: o original
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


# --------------------------------------------------------------------------- comparação de ajustes

def _calls(fake_facefusion: Path) -> list[list[str]]:
    path = fake_facefusion / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _opt(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def test_compare_runs_each_swapper_and_builds_the_grid(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
        root = project_paths(settings, pid).root
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        assert client.get(f"/projects/{pid}").json()["face_refine"]["settings"] == {
            "trocador": "hyperswap_1a_256", "restaurador": 80}
        (fake_facefusion / "calls.jsonl").unlink(missing_ok=True)

        resp = client.post(f"/projects/{pid}/face-compare")
        assert resp.status_code == 202, resp.text
        client.app.state.jobs.wait_idle(timeout=300)
        detail = client.get(f"/projects/{pid}").json()
        job = detail["jobs"]["face_compare"]
        assert job["status"] == "done" and job["message"].startswith("Comparação pronta"), job

        # Cada trocador: só a troca e, sobre ela, só o restaurador a 100% (as forças saem da mistura).
        calls = _calls(fake_facefusion)
        assert [(_opt(a, "--face-swapper-model"), a[a.index("--processors") + 1]) for a in calls] == [
            ("hyperswap_1a_256", "face_swapper"), (None, "face_enhancer"),
            ("hyperswap_1b_256", "face_swapper"), (None, "face_enhancer"),
            ("hyperswap_1c_256", "face_swapper"), (None, "face_enhancer")]
        assert all(_opt(a, "--face-enhancer-blend") in (None, "100") for a in calls)
        assert _opt(calls[1], "-t").endswith("1a-troca.mp4")  # o restaurador vai sobre a troca

        compare = detail["face_refine"]["compare"]
        labels = [v["rotulo"] for v in compare["variantes"]]
        assert labels[:3] == ["Sem refino (rosto do Wan)", "hyperswap 1a, restaurador 80% (atual)",
                              "hyperswap 1a, restaurador 40%"] and len(labels) == 8
        assert compare["trecho"] == {"inicio": 0, "quadros": 16, "segundos": 1.0}
        video = root / "exports" / "comparar_rosto.mp4"
        info = media.probe(video)
        assert (info.width, info.height) == (4 * 360, 2 * (360 + 34)) and media.count_frames(video) == 16
        assert client.get(f"/projects/{pid}/files/comparar_rosto.png").headers["content-type"] == "image/png"
        assert not list(root.glob("comparar-*"))

        # Ampliado no rosto (centro do quadro): sem refino = o vídeo do Wan; sem restaurador = a troca
        # (o falso inverte as cores); restaurador 80% = quase o restaurado (o falso inverte de volta).
        grid = _frame(video, 8)
        cell = lambda i: grid[34 + (i // 4) * 394 + 150:34 + (i // 4) * 394 + 210, (i % 4) * 360 + 150:(i % 4) * 360 + 210]
        plain, blend80, blend0 = cell(0), cell(1), cell(3)
        assert _diff(blend0, 255 - plain) < 30 < _diff(blend0, plain)
        assert _diff(blend80, plain) < _diff(blend80, 255 - plain)

        # Escolher um ajuste vale para o próximo refino.
        assert client.put(f"/projects/{pid}/face-settings", json={"trocador": "hyperswap_9z", "restaurador": 0}).status_code == 422
        assert client.put(f"/projects/{pid}/face-settings", json={"trocador": "hyperswap_1b_256", "restaurador": 101}).status_code == 422
        state = client.put(f"/projects/{pid}/face-settings", json={"trocador": "hyperswap_1b_256", "restaurador": 0}).json()
        assert state["settings"] == {"trocador": "hyperswap_1b_256", "restaurador": 0}
        client.post(f"/projects/{pid}/face-refine")
        client.app.state.jobs.wait_idle(timeout=300)
        args = json.loads((fake_facefusion / "args.json").read_text())
        assert _opt(args, "--face-swapper-model") == "hyperswap_1b_256" and "face_enhancer" not in args
        refine = client.get(f"/projects/{pid}").json()["face_refine"]["result"]
        assert refine["trocador"] == "hyperswap_1b_256" and refine["restaurador_forca"] == 0 and refine["restaurador"] is None

        # Um vídeo novo do Wan apaga a comparação do anterior (os ajustes escolhidos continuam).
        _import(client, pid, _video(tmp_path / "outro.mp4", "464x832", 16))
        detail = client.get(f"/projects/{pid}").json()
        assert detail["face_refine"]["compare"] is None and not detail["wan"]["comparar_rosto.mp4"]
        assert detail["face_refine"]["settings"]["trocador"] == "hyperswap_1b_256"


def test_compare_with_profile_face_explains(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_POSE", "profile")
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
        _import(client, pid, _video(tmp_path / "resultado.mp4", "464x832", 16))
        client.post(f"/projects/{pid}/face-compare")
        client.app.state.jobs.wait_idle(timeout=300)
        job = client.get(f"/projects/{pid}").json()["jobs"]["face_compare"]
        assert job["status"] == "error" and "perfil" in job["message"]


def test_compare_needs_a_result_and_blocks_clicks(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    settings = dataclasses.replace(ff_settings, face_refine_auto=False)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
        assert client.post(f"/projects/{pid}/face-compare").status_code == 400
        # Enquanto a comparação roda (FaceFusion ~5 GB), nenhum clique carrega o SAM 2.
        db = client.app.state.db
        job = db.create_job(pid, "face_compare")
        db.update_job(job["id"], status="running")
        resp = client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 10, "y": 10, "label": 1})
        assert resp.status_code == 409


def test_choose_excerpt_and_zoom_box():
    assert face_compare.choose_excerpt([0, 0, 0], 2) is None
    assert face_compare.choose_excerpt([0, 0, 1, 1, 1, 0.6, 0, 0], 3) == 2
    assert face_compare.choose_excerpt([1, 1], 24) == 0
    # Quadrado de 1,9x o maior lado, empurrado para dentro do quadro.
    assert face_compare.zoom_box([[10, 10, 50, 60]], (928, 1664)) == (0, 0, 95, 95)
    assert face_compare.zoom_box([[400, 800, 440, 840], [410, 810, 450, 850]], (928, 1664)) == (378, 778, 472, 872)
