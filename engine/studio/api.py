"""API HTTP local do motor (consumida pelo painel Next.js)."""

from __future__ import annotations

import importlib.util
import io
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__, diagnose
from .config import MAX_CLIP_SECONDS, Settings
from .db import Database
from .jobs import REMOTE_KINDS, JobConflict, JobRunner
from . import kaggle_remote
from .masks import load_mask, load_rgb, object_color, overlay
from .media import MediaError
from .models import checkpoint_path, get_model
from .projects import EXPORT_FILES, create_project, export_project, project_paths
from .session import TrackingSession
from .tracking import DemoTracker, Sam2Tracker, Tracker, TrackerError
from .wan import WAN_FILES, WanError, build_package, import_result, save_reference

log = logging.getLogger(__name__)


def default_tracker_factory(settings: Settings) -> Callable[[], Tracker]:
    def factory() -> Tracker:
        if settings.tracker == "demo":
            return DemoTracker()
        model = get_model(settings.sam2_model)
        return Sam2Tracker(model.config, checkpoint_path(settings.models_dir, settings.sam2_model), settings.device)

    return factory


class KaggleTokenIn(BaseModel):
    token: str = Field(min_length=1, max_length=600)


class WanPackageIn(BaseModel):
    resolution: Literal["480p", "720p"] = "480p"
    fps: float | None = Field(default=None, gt=0, le=60)
    prompt: str = Field(default="", max_length=1000)


class ClickIn(BaseModel):
    frame_idx: int = Field(ge=0)
    x: float
    y: float
    label: Literal[0, 1] = 1
    obj_id: int = Field(default=1, ge=1, le=8)


def create_app(
    settings: Settings | None = None,
    tracker_factory: Callable[[], Tracker] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    session = TrackingSession(db, settings, tracker_factory or default_tracker_factory(settings))
    jobs = JobRunner(db)
    remote_jobs = JobRunner(db, name="studio-remote")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()
        remote_jobs.shutdown()

    app = FastAPI(title="AI Video Studio — motor", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.db = db
    app.state.session = session
    app.state.jobs = jobs
    app.state.remote_jobs = remote_jobs

    @app.exception_handler(TrackerError)
    async def tracker_error(_req: Request, exc: TrackerError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(WanError)
    async def wan_error(_req: Request, exc: WanError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(kaggle_remote.KaggleError)
    async def kaggle_error(_req: Request, exc: kaggle_remote.KaggleError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(JobConflict)
    async def job_conflict(_req: Request, exc: JobConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def get_project(project_id: str) -> dict[str, Any]:
        project = db.get_project(project_id)
        if project is None:
            raise HTTPException(404, "Projeto não encontrado.")
        return project

    def ensure_idle(project_id: str, include_remote: bool = False) -> None:
        if db.active_job(project_id, exclude=() if include_remote else REMOTE_KINDS) is not None:
            raise HTTPException(409, "Há uma tarefa em andamento neste projeto. Aguarde terminar.")

    def check_frame(project: dict[str, Any], idx: int) -> None:
        if not 0 <= idx < project["num_frames"]:
            raise HTTPException(404, "Quadro inexistente.")

    def payload(project: dict[str, Any]) -> dict[str, Any]:
        paths = project_paths(settings, project["id"])
        return {
            **project,
            "clicks": db.list_clicks(project["id"]),
            "objects": paths.obj_ids(),
            "active_job": db.active_job(project["id"], exclude=REMOTE_KINDS),
            "remote_job": db.active_job(project["id"], kinds=REMOTE_KINDS),
            "jobs": db.latest_jobs(project["id"]),
            "exports": {name: (paths.exports / name).exists() for name in EXPORT_FILES},
            "reference": (paths.root / "reference.png").exists(),
            "wan": {name: (paths.exports / name).exists() for name in WAN_FILES},
        }

    def save_upload(upload: UploadFile, directory) -> Any:
        tmp = directory / f".upload-{uuid.uuid4().hex}"
        with open(tmp, "wb") as out:
            shutil.copyfileobj(upload.file, out, 1 << 20)
        return tmp

    @app.get("/health")
    def health() -> dict[str, Any]:
        ckpt = checkpoint_path(settings.models_dir, settings.sam2_model)
        tracker = session.tracker
        return {
            "status": "ok",
            "version": __version__,
            "tracker": settings.tracker,
            "tracker_loaded": tracker is not None,
            "device": tracker.device if tracker is not None else None,
            "sam2": {
                "installed": importlib.util.find_spec("sam2") is not None,
                "model": settings.sam2_model,
                "checkpoint_present": ckpt.exists(),
            },
            "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
            "limits": {
                "default_seconds": settings.max_seconds,
                "max_seconds": MAX_CLIP_SECONDS,
                "max_side": settings.max_side,
                "max_fps": settings.max_fps,
            },
        }

    @app.get("/system")
    def system() -> dict[str, Any]:
        return diagnose.collect(settings)

    @app.get("/projects")
    def list_projects() -> list[dict[str, Any]]:
        return db.list_projects()

    @app.post("/projects", status_code=201)
    def upload_project(
        file: UploadFile = File(...),
        name: str | None = Form(None),
        start: float = Form(0.0),
        seconds: float | None = Form(None),
    ) -> dict[str, Any]:
        seconds = seconds or settings.max_seconds
        if not 0 < seconds <= MAX_CLIP_SECONDS:
            raise HTTPException(400, f"A duração deve ficar entre 0 e {MAX_CLIP_SECONDS:g} segundos.")
        if start < 0:
            raise HTTPException(400, "O início não pode ser negativo.")
        upload = settings.projects_dir / f".upload-{uuid.uuid4().hex}"
        try:
            with open(upload, "wb") as out:
                shutil.copyfileobj(file.file, out, 1 << 20)
            project = create_project(
                db, settings, upload,
                source_name=file.filename or "video", name=name or None,
                start=start, seconds=seconds, move=True,
            )
        except MediaError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            upload.unlink(missing_ok=True)
        return payload(project)

    @app.get("/projects/{project_id}")
    def get_project_detail(project_id: str) -> dict[str, Any]:
        return payload(get_project(project_id))

    @app.delete("/projects/{project_id}", status_code=204)
    def delete_project(project_id: str) -> Response:
        get_project(project_id)
        ensure_idle(project_id)
        session.forget(project_id)
        db.delete_project(project_id)
        shutil.rmtree(project_paths(settings, project_id).root, ignore_errors=True)
        return Response(status_code=204)

    @app.get("/projects/{project_id}/frames/{idx}")
    def get_frame(project_id: str, idx: int) -> FileResponse:
        project = get_project(project_id)
        check_frame(project, idx)
        return FileResponse(project_paths(settings, project_id).frame(idx), media_type="image/jpeg")

    @app.get("/projects/{project_id}/overlay/{idx}")
    def get_overlay(project_id: str, idx: int) -> Response:
        project = get_project(project_id)
        check_frame(project, idx)
        paths = project_paths(settings, project_id)
        frame = load_rgb(paths.frame(idx))
        masks = [(load_mask(paths.mask(obj, idx)), object_color(obj)) for obj in paths.obj_ids()]
        buf = io.BytesIO()
        Image.fromarray(overlay(frame, masks)).save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/projects/{project_id}/masks/{obj_id}/{idx}")
    def get_mask(project_id: str, obj_id: int, idx: int) -> FileResponse:
        project = get_project(project_id)
        check_frame(project, idx)
        path = project_paths(settings, project_id).mask(obj_id, idx)
        if not path.exists():
            raise HTTPException(404, "Sem máscara neste quadro.")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/projects/{project_id}/clicks")
    def add_click(project_id: str, body: ClickIn) -> dict[str, Any]:
        project = get_project(project_id)
        ensure_idle(project_id)
        check_frame(project, body.frame_idx)
        if not (0 <= body.x < project["width"] and 0 <= body.y < project["height"]):
            raise HTTPException(400, "Clique fora do quadro.")
        click, mask, version = session.add_click(
            project_id, frame_idx=body.frame_idx, x=body.x, y=body.y, label=body.label, obj_id=body.obj_id
        )
        return {"click": click, "mask_area": int(mask.sum()), "mask_version": version}

    @app.post("/projects/{project_id}/clicks/undo")
    def undo_click(project_id: str) -> dict[str, Any]:
        get_project(project_id)
        ensure_idle(project_id)
        return {"removed": session.undo(project_id)}

    @app.delete("/projects/{project_id}/clicks")
    def clear_clicks(project_id: str) -> dict[str, Any]:
        get_project(project_id)
        ensure_idle(project_id)
        session.clear(project_id)
        return {"ok": True}

    @app.post("/projects/{project_id}/track", status_code=202)
    def track(project_id: str) -> dict[str, Any]:
        get_project(project_id)
        if not db.list_clicks(project_id):
            raise HTTPException(400, "Clique na pessoa antes de rastrear.")
        return jobs.submit(project_id, "track", lambda progress: session.propagate(project_id, progress))

    @app.post("/projects/{project_id}/export", status_code=202)
    def export(project_id: str) -> dict[str, Any]:
        get_project(project_id)

        def run(progress: Callable[..., None]) -> str:
            project = get_project(project_id)
            message = export_project(project, project_paths(settings, project_id), progress)
            db.set_exported(project_id, project["mask_version"])
            return message

        return jobs.submit(project_id, "export", run)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Tarefa não encontrada.")
        return job

    @app.post("/projects/{project_id}/reference")
    def upload_reference(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        get_project(project_id)
        root = project_paths(settings, project_id).root
        tmp = save_upload(file, root)
        try:
            width, height = save_reference(tmp, root / "reference.png")
        finally:
            tmp.unlink(missing_ok=True)
        return {"width": width, "height": height}

    @app.get("/projects/{project_id}/reference")
    def get_reference(project_id: str) -> FileResponse:
        get_project(project_id)
        path = project_paths(settings, project_id).root / "reference.png"
        if not path.exists():
            raise HTTPException(404, "Nenhuma imagem de referência enviada.")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/projects/{project_id}/wan-package", status_code=202)
    def wan_package(project_id: str, body: WanPackageIn) -> dict[str, Any]:
        get_project(project_id)

        def run(progress: Callable[..., None]) -> str:
            return build_package(
                get_project(project_id), project_paths(settings, project_id).root,
                resolution=body.resolution, fps=body.fps, prompt=body.prompt, progress=progress,
            )

        def conflict() -> str | None:
            if db.active_job(project_id, exclude=REMOTE_KINDS):
                return "Já existe uma tarefa em andamento neste projeto."
            if db.active_job(project_id, kinds=REMOTE_KINDS):
                return "O Kaggle está usando o pacote atual. Aguarde a geração terminar."
            return None

        return jobs.submit(project_id, "wan_package", run, conflict=conflict)

    @app.get("/kaggle")
    def kaggle_status() -> dict[str, Any]:
        return {
            **kaggle_remote.token_status(),
            "installed": importlib.util.find_spec("kaggle") is not None,
            "running": db.active_job(None, kinds=REMOTE_KINDS),
        }

    @app.post("/kaggle/token")
    def kaggle_token(body: KaggleTokenIn) -> dict[str, Any]:
        return kaggle_remote.save_and_check_token(body.token)

    @app.post("/projects/{project_id}/kaggle", status_code=202)
    def kaggle_run(project_id: str) -> dict[str, Any]:
        get_project(project_id)
        root = project_paths(settings, project_id).root

        def conflict() -> str | None:
            if db.active_job(None, kinds=REMOTE_KINDS):
                return "Já há uma geração no Kaggle em andamento (um notebook por vez)."
            if db.active_job(project_id, exclude=REMOTE_KINDS):
                return "Aguarde a tarefa atual do projeto terminar."
            if not kaggle_remote.token_status()["configured"]:
                return "Configure o token do Kaggle primeiro."
            if not (root / "exports" / "wan_package.zip").exists():
                return "Gere o pacote do Wan antes de enviar ao Kaggle."
            return None

        def run(progress: Callable[..., None]) -> str:
            return kaggle_remote.run_on_kaggle(
                get_project(project_id), root, progress,
                import_result=lambda path: import_result(get_project(project_id), root, path),
            )

        return remote_jobs.submit(project_id, "kaggle", run, conflict=conflict)

    @app.post("/projects/{project_id}/wan-result")
    def wan_result(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        project = get_project(project_id)
        ensure_idle(project_id, include_remote=True)
        root = project_paths(settings, project_id).root
        tmp = save_upload(file, root)
        try:
            message = import_result(project, root, tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return {"message": message, "project": payload(get_project(project_id))}

    @app.get("/projects/{project_id}/files/{name}")
    def get_file(project_id: str, name: str) -> FileResponse:
        get_project(project_id)
        paths = project_paths(settings, project_id)
        if name == "clip.mp4":
            path = paths.clip
        elif name in EXPORT_FILES or name in WAN_FILES:
            path = paths.exports / name
        else:
            raise HTTPException(404, "Arquivo desconhecido.")
        if not path.exists():
            raise HTTPException(404, "Arquivo ainda não gerado.")
        media_type = "application/zip" if name.endswith(".zip") else "video/mp4"
        return FileResponse(path, media_type=media_type, filename=f"{project_id}-{name}",
                            headers={"Cache-Control": "no-store"})

    return app
