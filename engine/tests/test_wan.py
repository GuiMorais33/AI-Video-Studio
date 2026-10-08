"""Pacote do Wan Animate e importação do resultado (rastreador demo, sem IA)."""

from __future__ import annotations

import json
import subprocess
import zipfile
from fractions import Fraction

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import media
from studio.api import create_app
from studio.projects import project_paths
from studio.wan import WanError, fit_segments, frame_indices, save_reference, wan_geometry


@pytest.mark.parametrize(
    ("src", "area", "expected"),
    [
        ((854, 480), 832 * 480, (832, 464)),
        ((480, 854), 832 * 480, (464, 832)),
        ((640, 360), 832 * 480, (832, 464)),
        ((1280, 720), 1280 * 720, (1280, 720)),
        ((640, 640), 832 * 480, (624, 624)),
    ],
)
def test_wan_geometry_multiple_of_16_within_area(src, area, expected):
    geo = wan_geometry(*src, area)
    assert (geo.width, geo.height) == expected
    assert geo.width % 16 == 0 and geo.height % 16 == 0
    assert geo.width * geo.height <= area
    # escala cobre o quadro e o recorte fica centralizado dentro da imagem escalada
    assert geo.scaled_width >= geo.width and geo.scaled_height >= geo.height
    assert 0 <= geo.crop_x <= geo.scaled_width - geo.width
    assert 0 <= geo.crop_y <= geo.scaled_height - geo.height


def test_frame_indices_resample():
    assert frame_indices(5, Fraction(30), Fraction(30)) == [0, 1, 2, 3, 4]
    idx = frame_indices(150, Fraction(30), Fraction(16))
    assert len(idx) == 80 and idx[0] == 0 and idx[-1] <= 149
    assert all(b > a for a, b in zip(idx, idx[1:]))


@pytest.mark.parametrize(
    ("count", "expected"),
    [(16, 16), (77, 77), (80, 77), (85, 77), (86, 86), (150, 150), (155, 153), (153, 153)],
)
def test_fit_segments_trims_small_overflow(count, expected):
    assert fit_segments(count) == expected


def test_save_reference_flattens_transparency(tmp_path):
    src = tmp_path / "ref.png"
    img = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (100, 100, 200, 300))
    img.save(src)
    assert save_reference(src, tmp_path / "out" / "reference.png") == (300, 400)
    out = Image.open(tmp_path / "out" / "reference.png")
    assert out.mode == "RGB"
    assert out.getpixel((5, 5)) == (255, 255, 255)
    assert out.getpixel((150, 150)) == (255, 0, 0)

    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (100, 100)).save(tiny)
    with pytest.raises(WanError):
        save_reference(tiny, tmp_path / "x.png")
    bogus = tmp_path / "bogus.png"
    bogus.write_text("nope")
    with pytest.raises(WanError):
        save_reference(bogus, tmp_path / "x.png")


def _ready_project(client, video):
    with open(video, "rb") as f:
        project = client.post("/projects", files={"file": ("clip.mp4", f)}).json()
    pid = project["id"]
    client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 320, "y": 180, "label": 1})
    client.post(f"/projects/{pid}/track")
    client.app.state.jobs.wait_idle(timeout=60)
    return pid


def _run_job(client, path, body):
    resp = client.post(path, json=body)
    assert resp.status_code == 202, resp.text
    client.app.state.jobs.wait_idle(timeout=120)
    return client.get(f"/jobs/{resp.json()['id']}").json()


def test_package_and_result_round_trip(settings, video, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _ready_project(client, video)
        root = project_paths(settings, pid).root

        job = _run_job(client, f"/projects/{pid}/wan-package", {"resolution": "480p"})
        assert job["status"] == "error" and "referência" in job["message"]

        ref = tmp_path / "personagem.png"
        Image.new("RGB", (512, 768), (40, 120, 200)).save(ref)
        with open(ref, "rb") as f:
            assert client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)}).json() == {
                "width": 512, "height": 768
            }
        assert client.get(f"/projects/{pid}/reference").headers["content-type"] == "image/png"

        job = _run_job(client, f"/projects/{pid}/wan-package", {"resolution": "480p", "fps": 16, "prompt": "robô"})
        assert job["status"] == "done", job
        detail = client.get(f"/projects/{pid}").json()
        assert detail["reference"] is True and detail["wan"]["wan_package.zip"] is True

        with zipfile.ZipFile(root / "exports" / "wan_package.zip") as zf:
            names = set(zf.namelist())
            meta = json.loads(zf.read("job.json"))
            zf.extractall(tmp_path / "pkg")
        assert {"video.mp4", "reference.png", "job.json"} <= names
        assert (meta["width"], meta["height"], meta["fps"]) == (832, 464, "16/1")
        assert meta["num_frames"] == 16 and len(meta["source_frame_indices"]) == 16
        assert meta["prompt"] == "robô" and meta["mode"] == "replace" and meta["trimmed_frames"] == 0
        masks = sorted((tmp_path / "pkg" / "masks").glob("*.png"))
        assert len(masks) == meta["num_frames"]
        first = np.asarray(Image.open(masks[0]))
        assert first.shape == (464, 832) and set(np.unique(first)) <= {0, 255} and first.any()
        info = media.probe(tmp_path / "pkg" / "video.mp4")
        assert (info.width, info.height, info.fps) == (832, 464, Fraction(16))
        assert not info.has_audio

        # "Resultado" falso: um vídeo qualquer no tamanho do pacote
        fake = tmp_path / "resultado.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=832x464:rate=16:duration=1",
                        "-pix_fmt", "yuv420p", str(fake)], check=True)
        with open(fake, "rb") as f:
            resp = client.post(f"/projects/{pid}/wan-result", files={"file": ("resultado.mp4", f)})
        assert resp.status_code == 200, resp.text
        wan = resp.json()["project"]["wan"]
        assert wan["wan_result.mp4"] and wan["wan_final.mp4"] and wan["comparativo.mp4"]
        final = media.probe(root / "exports" / "wan_final.mp4")
        assert final.has_audio and (final.width, final.height) == (832, 464)
        comp = media.probe(root / "exports" / "comparativo.mp4")
        assert comp.width == 2 * 832 and comp.height == 464
        assert client.get(f"/projects/{pid}/files/comparativo.mp4").status_code == 200

        bogus = tmp_path / "bogus.mp4"
        bogus.write_text("nope")
        with open(bogus, "rb") as f:
            assert client.post(f"/projects/{pid}/wan-result", files={"file": ("x.mp4", f)}).status_code == 400


def test_package_requires_current_tracking(settings, video, tmp_path):
    with TestClient(create_app(settings)) as client:
        pid = _ready_project(client, video)
        ref = tmp_path / "p.png"
        Image.new("RGB", (512, 512)).save(ref)
        with open(ref, "rb") as f:
            client.post(f"/projects/{pid}/reference", files={"file": ("p.png", f)})
        client.post(f"/projects/{pid}/clicks", json={"frame_idx": 5, "x": 100, "y": 100, "label": 1})
        job = _run_job(client, f"/projects/{pid}/wan-package", {})
        assert job["status"] == "error" and "Rastreie de novo" in job["message"]
