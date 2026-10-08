"""Linha de comando: `studio diagnose | models | serve | track`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from .config import MAX_CLIP_SECONDS, Settings


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides = {}
    if getattr(args, "model", None):
        overrides["sam2_model"] = args.model
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "tracker", None):
        overrides["tracker"] = args.tracker
    return dataclasses.replace(settings, **overrides)


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnose import collect, format_report

    report = collect(_settings(args))
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_report(report))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .models import SAM2_MODELS, checkpoint_path, download_checkpoint

    settings = _settings(args)
    if args.action == "list":
        for key, model in SAM2_MODELS.items():
            present = checkpoint_path(settings.models_dir, key).exists()
            print(f"{key:<10} {model.checkpoint:<28} ~{model.approx_mb:>4} MB  {'baixado' if present else '-'}")
        return 0
    path = download_checkpoint(settings.models_dir, settings.sam2_model, force=args.force)
    print(f"Checkpoint pronto: {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from .api import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def _parse_point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use o formato x,y (ex.: 0.5,0.4)") from None
    if not (0 <= x < 1 and 0 <= y < 1):
        raise argparse.ArgumentTypeError("coordenadas normalizadas entre 0 e 1 (ex.: 0.5,0.4)")
    return x, y


def _bar(label: str):
    def progress(fraction: float, message: str | None = None) -> None:
        filled = int(fraction * 30)
        sys.stderr.write(f"\r{label} [{'#' * filled}{'.' * (30 - filled)}] {fraction * 100:5.1f}%")
        sys.stderr.flush()
        if fraction >= 1:
            sys.stderr.write("\n")

    return progress


def cmd_track(args: argparse.Namespace) -> int:
    """Fluxo completo sem interface: preparar clipe, clicar, rastrear e exportar."""
    from .api import default_tracker_factory
    from .db import Database
    from .projects import create_project, export_project, project_paths
    from .session import TrackingSession

    settings = _settings(args)
    if args.seconds is not None and not 0 < args.seconds <= MAX_CLIP_SECONDS:
        raise ValueError(f"--seconds deve ficar entre 0 e {MAX_CLIP_SECONDS:g}")
    db = Database(settings.db_path)
    session = TrackingSession(db, settings, default_tracker_factory(settings))
    video = Path(args.video)
    t0 = time.perf_counter()
    project = create_project(
        db, settings, video, source_name=video.name, name=args.name, start=args.start, seconds=args.seconds
    )
    print(f"Projeto {project['id']}: {project['num_frames']} quadros {project['width']}x{project['height']} "
          f"@ {project['fps']} fps ({time.perf_counter() - t0:.1f}s para preparar)")
    if not 0 <= args.frame < project["num_frames"]:
        raise ValueError(f"--frame deve ficar entre 0 e {project['num_frames'] - 1}")

    t0 = time.perf_counter()
    clicks = [(p, 1) for p in args.point] + [(p, 0) for p in args.negative]
    for (x, y), label in clicks:
        _, mask, _ = session.add_click(
            project["id"], frame_idx=args.frame,
            x=x * project["width"], y=y * project["height"], label=label,
        )
    print(f"Seleção no quadro {args.frame}: {int(mask.sum())} pixels "
          f"({time.perf_counter() - t0:.1f}s, inclui carregar o modelo; dispositivo: {session.tracker.device})")

    t0 = time.perf_counter()
    session.propagate(project["id"], _bar("Rastreando "))
    elapsed = time.perf_counter() - t0
    print(f"Rastreamento: {elapsed:.1f}s ({elapsed / project['num_frames']:.2f}s por quadro)")

    project = db.get_project(project["id"])
    paths = project_paths(settings, project["id"])
    export_project(project, paths, _bar("Exportando "))
    db.set_exported(project["id"], project["mask_version"])
    print("Arquivos:")
    for name in ("preview.mp4", "mask.mp4", "masks.zip"):
        print(f"  {paths.exports / name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description="Motor local do AI Video Studio")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("diagnose", help="verificar sistema, hardware e dependências")
    p.add_argument("--json", action="store_true", help="saída em JSON")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("models", help="listar ou baixar checkpoints do SAM 2.1")
    p.add_argument("action", choices=["list", "download"])
    p.add_argument("--model", help="tiny (padrão), small, base_plus ou large")
    p.add_argument("--force", action="store_true", help="baixar de novo mesmo se já existir")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("serve", help="iniciar a API local usada pelo painel")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", help="modelo SAM 2.1 (padrão: tiny)")
    p.add_argument("--device", help="auto (padrão), cpu, cuda ou mps")
    p.add_argument("--tracker", choices=["sam2", "demo"], help="'demo' simula o SAM 2 (só para testar a interface)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("track", help="rastrear uma pessoa em um vídeo sem abrir o painel")
    p.add_argument("video", help="arquivo de vídeo")
    p.add_argument("--point", type=_parse_point, action="append", required=True,
                   help="clique positivo normalizado x,y (0–1); repita para vários")
    p.add_argument("--negative", type=_parse_point, action="append", default=[],
                   help="clique negativo normalizado x,y (exclui a região)")
    p.add_argument("--frame", type=int, default=0, help="quadro onde os cliques são feitos (padrão 0)")
    p.add_argument("--start", type=float, default=0.0, help="início do trecho em segundos")
    p.add_argument("--seconds", type=float, help=f"duração do trecho (padrão 5, máx. {MAX_CLIP_SECONDS:g})")
    p.add_argument("--name", help="nome do projeto")
    p.add_argument("--model", help="modelo SAM 2.1 (padrão: tiny)")
    p.add_argument("--device", help="auto (padrão), cpu, cuda ou mps")
    p.add_argument("--tracker", choices=["sam2", "demo"], help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_track)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # erros previsíveis viram mensagem curta
        from .media import MediaError
        from .tracking import TrackerError

        if isinstance(exc, (MediaError, TrackerError, OSError, ValueError)):
            print(f"Erro: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
