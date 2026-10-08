"""Exercita o SAM 2.1 oficial da Meta de ponta a ponta na CPU.

Usa pesos aleatórios (sem checkpoint) para não depender de download: valida
que nossa sessão chama a API real do SAM 2 corretamente (cliques, correção após
rastreamento, desfazer, reinício do motor e rastreamento para trás), não a
qualidade das máscaras. Pulado se PyTorch/SAM 2 não estiverem instalados.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sam2")

import torch  # noqa: E402

from studio.db import Database  # noqa: E402
from studio.masks import load_mask  # noqa: E402
from studio.models import get_model  # noqa: E402
from studio.projects import create_project, export_project, project_paths  # noqa: E402
from studio.session import TrackingSession  # noqa: E402
from studio.tracking import Sam2Tracker  # noqa: E402
from conftest import make_video  # noqa: E402


@pytest.fixture(scope="module")
def tracker():
    torch.manual_seed(0)
    return Sam2Tracker(get_model("tiny").config, checkpoint=None, device="cpu")


@pytest.fixture
def sam_settings(settings):
    return dataclasses.replace(settings, tracker="sam2")


def new_project(db, settings, tmp_path, frames=6):
    video = make_video(tmp_path / "v.mp4", size="320x180", rate=frames, seconds=1.0)
    return create_project(db, settings, video, source_name="v.mp4")


def test_click_track_correct_undo_restart(tracker, sam_settings, tmp_path):
    db = Database(sam_settings.db_path)
    project = new_project(db, sam_settings, tmp_path)
    pid, n = project["id"], project["num_frames"]
    paths = project_paths(sam_settings, pid)
    session = TrackingSession(db, sam_settings, lambda: tracker)

    _, mask, _ = session.add_click(pid, frame_idx=0, x=160, y=90, label=1)
    assert mask.shape == (180, 320) and mask.dtype == bool
    _, mask, _ = session.add_click(pid, frame_idx=0, x=40, y=40, label=0)
    assert mask.shape == (180, 320)

    session.propagate(pid, lambda *a: None)
    assert all(paths.mask(1, i).exists() for i in range(n))

    # Correção num quadro já rastreado (caminho de semear com a máscara do disco).
    before = load_mask(paths.mask(1, 3))
    session.add_click(pid, frame_idx=3, x=100, y=60, label=0)
    session.undo(pid)
    assert np.array_equal(load_mask(paths.mask(1, 3)), before)

    # Motor "reiniciado": sessão nova reconstrói o estado a partir do disco.
    restarted = TrackingSession(db, sam_settings, lambda: tracker)
    restarted.add_click(pid, frame_idx=2, x=200, y=100, label=1)
    restarted.propagate(pid, lambda *a: None)
    assert all(paths.mask(1, i).exists() for i in range(n))

    message = export_project(db.get_project(pid), paths, lambda *a: None)
    assert "Exportados" in message
    assert (paths.exports / "preview.mp4").exists()


def test_propagates_backwards_from_later_frame(tracker, sam_settings, tmp_path):
    db = Database(sam_settings.db_path)
    project = new_project(db, sam_settings, tmp_path)
    pid, n = project["id"], project["num_frames"]
    session = TrackingSession(db, sam_settings, lambda: tracker)
    session.add_click(pid, frame_idx=4, x=160, y=90, label=1)
    seen = []
    session.propagate(pid, lambda fraction, *a: seen.append(fraction))
    paths = project_paths(sam_settings, pid)
    assert all(paths.mask(1, i).exists() for i in range(n))
    assert seen[-1] == pytest.approx(1.0)
