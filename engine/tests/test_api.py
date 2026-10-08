"""Fluxo completo da API com o rastreador demo (sem IA)."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from studio import media
from studio.api import create_app
from studio.masks import load_mask
from studio.projects import project_paths


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def upload(client, video, **form):
    with open(video, "rb") as f:
        resp = client.post("/projects", files={"file": ("dança.mp4", f, "video/mp4")}, data=form)
    assert resp.status_code == 201, resp.text
    return resp.json()


def run_job(client, path):
    resp = client.post(path)
    assert resp.status_code == 202, resp.text
    client.app.state.jobs.wait_idle(timeout=60)
    job = client.get(f"/jobs/{resp.json()['id']}").json()
    assert job["status"] == "done", job
    return job


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["tracker"] == "demo"
    assert body["ffmpeg"] is True


def test_upload_creates_reduced_clip(client, video):
    project = upload(client, video, seconds="0.5")
    assert project["name"] == "dança"
    assert (project["width"], project["height"]) == (640, 360)
    assert project["num_frames"] == 15
    assert project["has_audio"] == 1
    assert project["clicks"] == []
    assert client.get("/projects").json()[0]["id"] == project["id"]
    assert client.get(f"/projects/{project['id']}/frames/0").headers["content-type"] == "image/jpeg"
    assert client.get(f"/projects/{project['id']}/frames/15").status_code == 404


def test_upload_rejects_invalid_files_and_limits(client, tmp_path, video):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a video")
    with open(bogus, "rb") as f:
        assert client.post("/projects", files={"file": ("bogus.mp4", f)}).status_code == 400
    with open(video, "rb") as f:
        assert client.post("/projects", files={"file": ("a.mp4", f)}, data={"seconds": "60"}).status_code == 400
    assert client.get("/projects").json() == []


def test_click_track_correct_export(client, settings, video):
    project = upload(client, video)
    pid = project["id"]
    paths = project_paths(settings, pid)

    resp = client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 320, "y": 180, "label": 1})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mask_area"] > 0
    assert load_mask(paths.mask(1, 0))[180, 320]

    # Fora do quadro
    assert client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 999, "y": 10}).status_code == 400

    run_job(client, f"/projects/{pid}/track")
    detail = client.get(f"/projects/{pid}").json()
    assert detail["tracked_revision"] == detail["revision"]
    assert detail["objects"] == [1]
    assert all(paths.mask(1, i).exists() for i in range(project["num_frames"]))

    # Correção num quadro rastreado parte da máscara existente
    resp = client.post(f"/projects/{pid}/clicks", json={"frame_idx": 20, "x": 360, "y": 180, "label": 0})
    assert resp.status_code == 200, resp.text
    corrected = load_mask(paths.mask(1, 20))
    assert not corrected[180, 340]
    assert corrected[180, 290]  # o lado não corrigido do disco rastreado continua lá
    detail = client.get(f"/projects/{pid}").json()
    assert detail["tracked_revision"] != detail["revision"]

    # Desfazer restaura exatamente a máscara rastreada
    before = load_mask(paths.mask(1, 21))
    removed = client.post(f"/projects/{pid}/clicks/undo").json()["removed"]
    assert removed["frame_idx"] == 20
    assert np.array_equal(load_mask(paths.mask(1, 20)), before)

    overlay = client.get(f"/projects/{pid}/overlay/0")
    assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/jpeg"

    run_job(client, f"/projects/{pid}/export")
    detail = client.get(f"/projects/{pid}").json()
    assert detail["exports"] == {"preview.mp4": True, "mask.mp4": True, "masks.zip": True}
    assert detail["exported_mask_version"] == detail["mask_version"]

    preview = media.probe(paths.exports / "preview.mp4")
    assert preview.has_audio
    assert (preview.width, preview.height) == (640, 360)
    with zipfile.ZipFile(paths.exports / "masks.zip") as zf:
        assert len(zf.namelist()) == project["num_frames"]
    resp = client.get(f"/projects/{pid}/files/preview.mp4")
    assert resp.status_code == 200 and resp.headers["content-type"] == "video/mp4"
    assert client.get(f"/projects/{pid}/files/secret.txt").status_code == 404


def test_track_requires_clicks_and_export_requires_masks(client, video):
    pid = upload(client, video)["id"]
    assert client.post(f"/projects/{pid}/track").status_code == 400
    resp = client.post(f"/projects/{pid}/export")
    client.app.state.jobs.wait_idle(timeout=30)
    job = client.get(f"/jobs/{resp.json()['id']}").json()
    assert job["status"] == "error"
    assert "máscaras" in job["message"]


def test_engine_restart_keeps_clicks_and_masks(settings, video):
    with TestClient(create_app(settings)) as c:
        pid = upload(c, video)["id"]
        c.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 320, "y": 180})
    with TestClient(create_app(settings)) as c:
        detail = c.get(f"/projects/{pid}").json()
        assert len(detail["clicks"]) == 1
        run_job(c, f"/projects/{pid}/track")
        mask = load_mask(project_paths(settings, pid).mask(1, 29))
        assert mask[180, 320]


def test_clear_and_delete(client, settings, video):
    pid = upload(client, video)["id"]
    client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 100, "y": 100})
    assert client.delete(f"/projects/{pid}/clicks").status_code == 200
    detail = client.get(f"/projects/{pid}").json()
    assert detail["clicks"] == [] and detail["objects"] == []
    assert client.delete(f"/projects/{pid}").status_code == 204
    assert client.get(f"/projects/{pid}").status_code == 404
    assert not project_paths(settings, pid).root.exists()
