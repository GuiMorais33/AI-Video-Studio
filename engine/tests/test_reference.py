"""Imagem do personagem que vai para o Kaggle (reference.py): o rosto das fotos e o recorte justo."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import face_refine, reference
from studio.api import create_app
from studio.projects import project_paths
from test_face_refine import _portrait_project, fake_facefusion, ff_settings  # noqa: F401 (fixtures)

GEN_ASPECT = 464 / 832


def _figure(size=(600, 900), box=(240, 200, 360, 700), background=(245, 245, 240)) -> Image.Image:
    img = Image.new("RGB", size, background)
    img.paste((40, 60, 120), box)
    return img


def test_tight_crop_frames_the_person_in_the_generation_shape():
    img = _figure()
    out, note = reference.tight_crop(img, GEN_ASPECT)
    assert note is None and out.size != img.size
    assert out.width / out.height == pytest.approx(GEN_ASPECT, abs=0.01)
    # A pessoa (500 px de altura) ocupa quase o quadro todo, inteira.
    arr = np.asarray(out)
    rows = np.flatnonzero((np.abs(arr.astype(int) - (40, 60, 120)).sum(axis=2) < 30).any(axis=1))
    assert rows[0] > 0 and rows[-1] < out.height - 1 and (rows[-1] - rows[0] + 1) / out.height > 0.85


def test_tight_crop_never_cuts_and_pads_with_the_background():
    """Pessoa larga junto da borda: o recorte cresce para fora da imagem e completa com a cor do fundo."""
    img = _figure(size=(400, 900), box=(0, 300, 400, 700))
    out, note = reference.tight_crop(img, GEN_ASPECT)
    assert note is None
    arr = np.asarray(out).astype(int)
    person = np.abs(arr - (40, 60, 120)).sum(axis=2) < 30
    assert person.sum() == 400 * 400  # inteira
    assert np.abs(arr[0, 0] - (245, 245, 240)).sum() < 6  # sobra = cor do fundo


def test_tight_crop_skips_busy_backgrounds_and_full_frames():
    rng = np.random.default_rng(0)
    busy = Image.fromarray(rng.integers(0, 256, (900, 600, 3), dtype=np.uint8))
    out, note = reference.tight_crop(busy, GEN_ASPECT)
    assert out is busy and "não é liso" in note
    full = _figure(box=(30, 20, 570, 880))
    out, note = reference.tight_crop(full, GEN_ASPECT)
    assert out is full and note is None  # já ocupa a imagem: nada a ganhar


def _package_zip(root: Path) -> tuple[dict, Image.Image]:
    with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
        return json.loads(zf.read("job.json")), Image.open(io.BytesIO(zf.read("reference.png"))).convert("RGB")


def _add_face(client, pid: str, tmp_path: Path, name: str) -> None:
    photo = tmp_path / name
    Image.new("RGB", (300, 300), (190, 150, 120)).save(photo)
    with open(photo, "rb") as f:
        client.post(f"/projects/{pid}/faces", files=[("files", (photo.name, f))])


def test_package_carries_the_face_from_the_photos(ff_settings, fake_facefusion, tmp_path):
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path)
        root = project_paths(ff_settings, pid).root
        job, ref = _package_zip(root)
        assert job["referencia"]["rosto"] is True
        # O FaceFusion falso inverte as cores: a imagem que vai para o Kaggle é a do FaceFusion.
        original = np.asarray(Image.open(root / "reference.png").convert("RGB"), np.float32)
        assert np.abs(np.asarray(ref, np.float32) - (255 - original)).mean() < 3
        args = json.loads((fake_facefusion / "args.json").read_text())
        assert args[args.index("-t") + 1].endswith(".png") and args[args.index("-o") + 1].endswith(".png")
        assert len(args[args.index("-s") + 1:args.index("-t")]) == 2  # as duas fotos do rosto
        detail = client.get(f"/projects/{pid}").json()
        assert detail["wan"]["referencia_wan.png"] and detail["wan_package_stale"] is False
        assert "com o rosto das suas fotos" in detail["jobs"]["wan_package"]["message"]
        assert client.get(f"/projects/{pid}/files/referencia_wan.png").headers["content-type"] == "image/png"
        # Foto do rosto nova: o Kaggle receberia o rosto antigo, então o pacote fica desatualizado.
        time.sleep(0.05)
        _add_face(client, pid, tmp_path, "nova.png")
        assert client.get(f"/projects/{pid}").json()["wan_package_stale"] is True


def test_without_the_face_option_photos_do_not_touch_the_package(ff_settings, fake_facefusion, tmp_path):
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path, face_on_reference=False)
        root = project_paths(ff_settings, pid).root
        job, ref = _package_zip(root)
        assert job["referencia"]["rosto"] is False
        assert np.abs(np.asarray(ref, np.float32) - np.asarray(Image.open(root / "reference.png").convert("RGB"),
                                                                 np.float32)).mean() < 1
        assert not (fake_facefusion / "args.json").exists()
        time.sleep(0.05)
        _add_face(client, pid, tmp_path, "nova.png")
        assert client.get(f"/projects/{pid}").json()["wan_package_stale"] is False


def test_low_memory_skips_the_face_and_says_so(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    import dataclasses

    monkeypatch.setattr(face_refine, "free_memory_gb", lambda: 2.5)
    settings = dataclasses.replace(ff_settings, facefusion_min_memory_gb=5.5)
    with TestClient(create_app(settings)) as client:
        pid = _portrait_project(client, tmp_path)
        job, _ = _package_zip(project_paths(settings, pid).root)
        assert job["referencia"]["rosto"] is False
        assert any("pouca memória" in a and "4 GB" in a for a in job["referencia"]["avisos"]), job["referencia"]
        assert not (fake_facefusion / "args.json").exists()


def test_no_face_found_keeps_the_image(ff_settings, fake_facefusion, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_FACEFUSION_MODE", "noface")
    with TestClient(create_app(ff_settings)) as client:
        pid = _portrait_project(client, tmp_path)
        job, _ = _package_zip(project_paths(ff_settings, pid).root)
        assert job["referencia"]["rosto"] is False
        assert any("rosto humano" in a for a in job["referencia"]["avisos"]), job["referencia"]
