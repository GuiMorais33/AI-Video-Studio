"""Persistência em SQLite: projetos, cliques de seleção e tarefas em segundo plano."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    clip_start REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps TEXT NOT NULL,
    num_frames INTEGER NOT NULL,
    duration REAL NOT NULL,
    has_audio INTEGER NOT NULL,
    -- incrementa a cada clique adicionado/desfeito
    revision INTEGER NOT NULL DEFAULT 0,
    -- revision usada no último rastreamento concluído
    tracked_revision INTEGER,
    -- incrementa sempre que alguma máscara em disco muda
    mask_version INTEGER NOT NULL DEFAULT 0,
    -- mask_version usada na última exportação concluída
    exported_mask_version INTEGER
);
CREATE TABLE IF NOT EXISTS clicks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    obj_id INTEGER NOT NULL,
    frame_idx INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    label INTEGER NOT NULL,
    -- 1 se existia máscara do objeto neste quadro antes do clique (para desfazer)
    had_prev_mask INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS clicks_by_project ON clicks(project_id, id);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_by_project ON jobs(project_id, created_at);
"""

ACTIVE_JOB_STATUSES = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Tarefas que estavam rodando quando o motor parou não vão terminar.
            conn.execute(
                "UPDATE jobs SET status = 'error', message = 'Interrompida: o motor foi reiniciado.', "
                "updated_at = ? WHERE status IN ('queued', 'running')",
                (_now(),),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- projetos -------------------------------------------------------

    def create_project(self, **fields: Any) -> dict[str, Any]:
        fields = {"created_at": _now(), **fields}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            conn.execute(f"INSERT INTO projects ({cols}) VALUES ({marks})", tuple(fields.values()))
        return self.get_project(fields["id"])

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return _row(conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_project(self, project_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def bump_mask_version(self, project_id: str) -> int:
        with self._connect() as conn:
            conn.execute("UPDATE projects SET mask_version = mask_version + 1 WHERE id = ?", (project_id,))
            return conn.execute("SELECT mask_version FROM projects WHERE id = ?", (project_id,)).fetchone()[0]

    def set_tracked(self, project_id: str, revision: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE projects SET tracked_revision = ? WHERE id = ?", (revision, project_id))

    def set_exported(self, project_id: str, mask_version: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE projects SET exported_mask_version = ? WHERE id = ?", (mask_version, project_id))

    # --- cliques --------------------------------------------------------

    def add_click(
        self, project_id: str, *, obj_id: int, frame_idx: int, x: float, y: float, label: int, had_prev_mask: bool
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO clicks (project_id, obj_id, frame_idx, x, y, label, had_prev_mask, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, obj_id, frame_idx, x, y, label, int(had_prev_mask), _now()),
            )
            conn.execute("UPDATE projects SET revision = revision + 1 WHERE id = ?", (project_id,))
            return _row(conn.execute("SELECT * FROM clicks WHERE id = ?", (cur.lastrowid,)).fetchone())

    def list_clicks(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM clicks WHERE project_id = ? ORDER BY id", (project_id,)).fetchall()
        return [dict(r) for r in rows]

    def last_click(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return _row(conn.execute(
                "SELECT * FROM clicks WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,)
            ).fetchone())

    def delete_click(self, click_id: int) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT project_id FROM clicks WHERE id = ?", (click_id,)).fetchone()
            conn.execute("DELETE FROM clicks WHERE id = ?", (click_id,))
            if row is not None:
                conn.execute("UPDATE projects SET revision = revision + 1 WHERE id = ?", (row[0],))

    def clear_clicks(self, project_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM clicks WHERE project_id = ?", (project_id,))
            conn.execute(
                "UPDATE projects SET revision = revision + 1, tracked_revision = NULL WHERE id = ?",
                (project_id,),
            )

    # --- tarefas --------------------------------------------------------

    def create_job(self, project_id: str, kind: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, project_id, kind, status, progress, created_at, updated_at) "
                "VALUES (?, ?, ?, 'queued', 0, ?, ?)",
                (job_id, project_id, kind, now, now),
            )
        return self.get_job(job_id)

    def update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return _row(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def active_job(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return _row(conn.execute(
                "SELECT * FROM jobs WHERE project_id = ? AND status IN ('queued', 'running') "
                "ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            ).fetchone())

    def latest_jobs(self, project_id: str) -> dict[str, dict[str, Any]]:
        """Última tarefa de cada tipo para o projeto."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at, rowid", (project_id,)
            ).fetchall()
        return {r["kind"]: dict(r) for r in rows}
