"""Sessão interativa de seleção e rastreamento sobre um rastreador (SAM 2 ou demo).

As máscaras gravadas em disco são a fonte da verdade. O estado do rastreador
vive só na memória do motor; quando ele precisa ser reconstruído (motor
reiniciado, troca de projeto, desfazer), cada quadro que recebeu cliques volta
a ser condição do rastreamento usando a máscara atual em disco.
"""

from __future__ import annotations

import os
import shutil
import threading
from typing import Any, Callable

import numpy as np

from .config import Settings
from .db import Database
from .masks import load_mask, save_mask
from .projects import ProjectPaths, project_paths
from .tracking import Tracker, TrackerError

Key = tuple[int, int]  # (frame_idx, obj_id)


class TrackingSession:
    def __init__(self, db: Database, settings: Settings, tracker_factory: Callable[[], Tracker]):
        self.db = db
        self.settings = settings
        self._factory = tracker_factory
        self._lock = threading.RLock()
        self._tracker: Tracker | None = None
        self._loaded_project: str | None = None
        # Cliques enviados ao rastreador nesta sessão, por quadro/objeto.
        self._points: dict[Key, tuple[list[list[float]], list[int]]] = {}
        # Quadros/objetos que receberam a máscara do disco como ponto de partida.
        self._seeded: set[Key] = set()

    @property
    def tracker(self) -> Tracker | None:
        return self._tracker

    def _get_tracker(self) -> Tracker:
        if self._tracker is None:
            self._tracker = self._factory()
        return self._tracker

    def _project(self, project_id: str) -> tuple[dict[str, Any], ProjectPaths]:
        project = self.db.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        return project, project_paths(self.settings, project_id)

    def _rebuild(self, project_id: str, reload_video: bool) -> None:
        tracker = self._get_tracker()
        _, paths = self._project(project_id)
        if reload_video:
            self._loaded_project = None
            tracker.load_video(paths.frames)
        else:
            tracker.reset()
        self._points = {}
        self._seeded = set()
        self._loaded_project = project_id
        cond = {(c["frame_idx"], c["obj_id"]) for c in self.db.list_clicks(project_id)}
        for frame_idx, obj_id in sorted(cond):
            mask = load_mask(paths.mask(obj_id, frame_idx))
            if mask is not None:  # máscara vazia também conta: "a pessoa não está aqui"
                tracker.add_mask(frame_idx, obj_id, mask)
                self._seeded.add((frame_idx, obj_id))

    def _ensure_loaded(self, project_id: str) -> Tracker:
        if self._loaded_project != project_id:
            self._rebuild(project_id, reload_video=True)
        return self._get_tracker()

    def add_click(
        self, project_id: str, *, frame_idx: int, x: float, y: float, label: int, obj_id: int = 1
    ) -> tuple[dict[str, Any], np.ndarray, int]:
        with self._lock:
            tracker = self._ensure_loaded(project_id)
            _, paths = self._project(project_id)
            key = (frame_idx, obj_id)
            mask_path = paths.mask(obj_id, frame_idx)
            previous = load_mask(mask_path)
            if key not in self._points and key not in self._seeded and previous is not None and previous.any():
                # Corrigir um quadro já rastreado: parte da máscara atual em vez do zero.
                tracker.add_mask(frame_idx, obj_id, previous)
                self._seeded.add(key)
            points, labels = self._points.get(key, ([], []))
            points, labels = points + [[x, y]], labels + [label]
            mask = tracker.add_points(frame_idx, obj_id, points, labels)
            self._points[key] = (points, labels)

            click = self.db.add_click(
                project_id, obj_id=obj_id, frame_idx=frame_idx, x=x, y=y, label=label,
                had_prev_mask=previous is not None,
            )
            if previous is not None:
                save_mask(paths.history_mask(click["id"]), previous)
            save_mask(mask_path, mask)
            return click, mask, self.db.bump_mask_version(project_id)

    def undo(self, project_id: str) -> dict[str, Any] | None:
        """Desfaz o último clique, restaurando a máscara que o quadro tinha antes dele."""
        with self._lock:
            _, paths = self._project(project_id)
            click = self.db.last_click(project_id)
            if click is None:
                return None
            mask_path = paths.mask(click["obj_id"], click["frame_idx"])
            snapshot = paths.history_mask(click["id"])
            if click["had_prev_mask"] and snapshot.exists():
                os.replace(snapshot, mask_path)
            else:
                mask_path.unlink(missing_ok=True)
            self.db.delete_click(click["id"])
            self.db.bump_mask_version(project_id)
            if self._loaded_project == project_id:
                self._rebuild(project_id, reload_video=False)
            return click

    def clear(self, project_id: str) -> None:
        with self._lock:
            _, paths = self._project(project_id)
            self.db.clear_clicks(project_id)
            shutil.rmtree(paths.masks, ignore_errors=True)
            shutil.rmtree(paths.history, ignore_errors=True)
            self.db.bump_mask_version(project_id)
            if self._loaded_project == project_id:
                self._rebuild(project_id, reload_video=False)

    def propagate(self, project_id: str, progress: Callable[..., None]) -> str:
        with self._lock:
            clicks = self.db.list_clicks(project_id)
            if not clicks:
                raise TrackerError("Clique na pessoa antes de rastrear.")
            project, paths = self._project(project_id)
            revision = project["revision"]
            progress(0.0, "Carregando o vídeo no rastreador")
            tracker = self._ensure_loaded(project_id)
            # As condições já estão no rastreador; o resultado reescreve todos os quadros.
            shutil.rmtree(paths.masks, ignore_errors=True)
            total = project["num_frames"]
            seen: set[int] = set()

            def on_frame(frame_idx: int, masks: dict[int, np.ndarray]) -> None:
                for obj_id, mask in masks.items():
                    save_mask(paths.mask(obj_id, frame_idx), mask)
                seen.add(frame_idx)
                progress(len(seen) / total, f"Rastreando quadro {len(seen)}/{total}")

            start = min(c["frame_idx"] for c in clicks)
            tracker.propagate(start, on_frame)
            self.db.set_tracked(project_id, revision)
            self.db.bump_mask_version(project_id)
            return f"{len(seen)} quadros rastreados."

    def forget(self, project_id: str) -> None:
        with self._lock:
            if self._loaded_project == project_id and self._tracker is not None:
                self._tracker.unload()
                self._loaded_project = None
                self._points = {}
                self._seeded = set()
