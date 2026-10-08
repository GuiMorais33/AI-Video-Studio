"""Execução do Wan Animate no Kaggle pela API oficial (sem interface, sem túnel).

Fluxo de uma geração:
1. O pacote (wan_package.zip) vira um dataset PRIVADO "<usuário>/aivs-<projeto>".
2. O notebook remote/kaggle/wan_animate_kaggle.ipynb é enviado como kernel privado com
   GPU T4 x2 e internet, tendo o dataset como entrada, e executado em lote.
3. O motor acompanha o status; ao terminar, baixa resultado.mp4 e importa no projeto.

Credenciais: token de API do Kaggle salvo em ~/.kaggle/access_token (ou KAGGLE_API_TOKEN).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .config import REPO_ROOT

KERNEL_SLUG = "ai-video-studio-wan-animate"
KERNEL_TITLE = "AI Video Studio Wan Animate"
NOTEBOOK = REPO_ROOT / "remote" / "kaggle" / "wan_animate_kaggle.ipynb"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 12 * 3600  # limite de uma sessão do Kaggle
RESULT_NAME = "resultado.mp4"
# O notebook imprime linhas "AIVS_PROGRESS <0-1> <mensagem>" que viram progresso aqui.
# O log do Kaggle pode vir como JSON numa linha só: a mensagem para em aspas, barra ou fim de linha.
PROGRESS_RE = re.compile(r"AIVS_PROGRESS\s+([0-9.]+)\s+([^\"\\\n]*)")


class KaggleError(RuntimeError):
    pass


def token_status() -> dict[str, Any]:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"configured": True, "source": "variável KAGGLE_API_TOKEN"}
    if TOKEN_FILE.exists():
        return {"configured": True, "source": str(TOKEN_FILE)}
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        return {"configured": True, "source": str(legacy)}
    return {"configured": False, "source": None}


def save_token(token: str) -> None:
    token = token.strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{20,512}", token):
        raise KaggleError("Token inválido. Copie o token gerado em kaggle.com > Settings > API.")
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.chmod(TOKEN_FILE, 0o600)


def _api():
    try:
        # Importar o módulo de baixo nível evita o login automático de "import kaggle".
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise KaggleError("Pacote 'kaggle' não instalado. Rode: bash scripts/setup.sh") from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except (SystemExit, Exception) as exc:  # o CLI chama exit() quando não acha credenciais
        raise KaggleError("Não foi possível entrar no Kaggle. Confira o token de API.") from exc
    user = api.get_config_value("username")
    if not user:
        raise KaggleError("Token do Kaggle sem usuário associado.")
    return api, user


def check_account() -> dict[str, Any]:
    api, user = _api()
    quota: Any = None
    try:
        quota = api.quota_view()
    except Exception:  # cota é informativa
        pass
    return {"username": user, "quota": str(quota) if quota is not None else None}


def _status_name(response: Any) -> str:
    status = getattr(response, "status", response)
    return getattr(status, "name", str(status)).upper()


def _wait_dataset(api, dataset_id: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = str(api.dataset_status(dataset_id)).lower()
        except Exception:
            status = "pending"
        if status == "ready":
            return
        if status == "error":
            raise KaggleError(f"O Kaggle rejeitou o dataset {dataset_id}.")
        time.sleep(5)
    raise KaggleError("O dataset demorou demais para ficar pronto no Kaggle.")


def _dataset_exists(api, dataset_id: str) -> bool:
    try:
        api.dataset_status(dataset_id)
        return True
    except Exception:
        return False


def run_on_kaggle(
    project: dict[str, Any],
    root: Path,
    progress: Callable[..., None],
    import_result: Callable[[Path], str],
    notebook: Path = NOTEBOOK,
) -> str:
    package = root / "exports" / "wan_package.zip"
    if not package.exists():
        raise KaggleError("Gere o pacote do Wan antes de enviar ao Kaggle.")
    if not notebook.exists():
        raise KaggleError(f"Notebook não encontrado: {notebook}")
    api, user = _api()
    dataset_slug = f"aivs-{project['id']}"
    dataset_id = f"{user}/{dataset_slug}"
    kernel_id = f"{user}/{KERNEL_SLUG}"

    with tempfile.TemporaryDirectory(prefix="aivs-kaggle-") as tmp_name:
        tmp = Path(tmp_name)
        data_dir = tmp / "dataset"
        data_dir.mkdir()
        shutil.copyfile(package, data_dir / "wan_package.zip")
        (data_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": dataset_slug,
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }))
        progress(0.02, "Enviando o pacote ao Kaggle (dataset privado)")
        if _dataset_exists(api, dataset_id):
            api.dataset_create_version(str(data_dir), version_notes="AI Video Studio", quiet=True, dir_mode="skip")
        else:
            api.dataset_create_new(str(data_dir), public=False, quiet=True, dir_mode="skip")
        _wait_dataset(api, dataset_id)

        kernel_dir = tmp / "kernel"
        kernel_dir.mkdir()
        shutil.copyfile(notebook, kernel_dir / notebook.name)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_id,
            "title": KERNEL_TITLE,
            "code_file": notebook.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [dataset_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }, indent=2))
        progress(0.05, "Iniciando o notebook na GPU T4 do Kaggle")
        api.kernels_push(str(kernel_dir), timeout=str(MAX_WAIT_SECONDS))

        started = time.monotonic()
        last_fraction = 0.05
        while True:
            time.sleep(POLL_SECONDS)
            status = _status_name(api.kernels_status(kernel_id))
            message = None
            try:
                matches = PROGRESS_RE.findall(api.kernels_logs(kernel_id) or "")
            except Exception:
                matches = []
            if matches:
                value, message = matches[-1]
                last_fraction = max(last_fraction, 0.05 + 0.85 * min(1.0, float(value)))
            minutes = int((time.monotonic() - started) / 60)
            progress(last_fraction, f"Kaggle: {status.lower()} ({minutes} min){' · ' + message if message else ''}")
            if status in ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
                break
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise KaggleError("O notebook passou de 12 h sem terminar.")

        out_dir = tmp / "output"
        out_dir.mkdir()
        try:
            api.kernels_output(kernel_id, str(out_dir), force=True, quiet=True)
        except Exception:
            pass
        report = next(out_dir.rglob("relatorio.json"), None)
        if status != "COMPLETE":
            detail = ""
            if report is not None:
                detail = json.loads(report.read_text()).get("erro", "")
            raise KaggleError(
                f"O notebook terminou com status {status.lower()}. {detail} "
                f"Veja o log em https://www.kaggle.com/code/{kernel_id}"
            )
        result = next(out_dir.rglob(RESULT_NAME), None)
        if result is None:
            raise KaggleError(f"O notebook terminou sem gerar {RESULT_NAME}. Veja https://www.kaggle.com/code/{kernel_id}")
        if report is not None:
            shutil.copyfile(report, root / "exports" / "kaggle_relatorio.json")
        progress(0.95, "Importando o resultado")
        upload = root / f".kaggle-{RESULT_NAME}"
        shutil.copyfile(result, upload)
        try:
            message = import_result(upload)
        finally:
            upload.unlink(missing_ok=True)
    return f"{message} (Kaggle: {int((time.monotonic() - started) / 60)} min)"
