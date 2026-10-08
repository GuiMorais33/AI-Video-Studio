"""Execução serial de tarefas longas (rastreamento, exportação) com progresso no SQLite."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .db import Database

log = logging.getLogger(__name__)


class JobConflict(RuntimeError):
    pass


class JobRunner:
    def __init__(self, db: Database):
        self.db = db
        # Um único trabalhador: o modelo e a CPU não comportam duas tarefas pesadas ao mesmo tempo.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-job")
        self._submit_lock = threading.Lock()

    def submit(self, project_id: str, kind: str, fn: Callable[[Callable[..., None]], str | None]) -> dict[str, Any]:
        with self._submit_lock:
            if self.db.active_job(project_id) is not None:
                raise JobConflict("Já existe uma tarefa em andamento neste projeto.")
            job = self.db.create_job(project_id, kind)
        self._pool.submit(self._run, job["id"], fn)
        return job

    def _run(self, job_id: str, fn: Callable[[Callable[..., None]], str | None]) -> None:
        self.db.update_job(job_id, status="running")
        last = {"progress": 0.0}

        def progress(fraction: float, message: str | None = None) -> None:
            fraction = max(0.0, min(1.0, fraction))
            # Limita gravações no banco: só a cada 1% ou quando há mensagem nova de etapa.
            if message is None and fraction - last["progress"] < 0.01:
                return
            last["progress"] = fraction
            fields: dict[str, Any] = {"progress": round(fraction, 4)}
            if message is not None:
                fields["message"] = message
            self.db.update_job(job_id, **fields)

        try:
            message = fn(progress)
        except Exception as exc:  # a mensagem vai para a interface
            log.exception("Tarefa %s falhou", job_id)
            self.db.update_job(job_id, status="error", message=str(exc) or exc.__class__.__name__)
        else:
            self.db.update_job(job_id, status="done", progress=1.0, message=message)

    def wait_idle(self, timeout: float | None = None) -> None:
        """Bloqueia até a fila esvaziar (usado em testes e na linha de comando)."""
        self._pool.submit(lambda: None).result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
