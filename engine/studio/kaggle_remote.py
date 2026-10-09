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
import uuid
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


def _legacy_file() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def token_status() -> dict[str, Any]:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"configured": True, "source": "variável KAGGLE_API_TOKEN"}
    for path in (TOKEN_FILE, _legacy_file()):
        if path.exists():
            return {"configured": True, "source": str(path)}
    return {"configured": False, "source": None}


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(path, 0o600)


def save_token(token: str) -> Path:
    """Aceita o token copiado de kaggle.com > Settings > API ou o conteúdo do kaggle.json."""
    text = token.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            raise KaggleError("O conteúdo colado do kaggle.json não é um JSON válido.") from None
        username, key = str(data.get("username", "")), str(data.get("key", ""))
        if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,64}", username) or not re.fullmatch(r"[A-Za-z0-9]{20,128}", key):
            raise KaggleError("O kaggle.json precisa ter os campos username e key.")
        path = _legacy_file()
        _write_private(path, json.dumps({"username": username, "key": key}))
        return path
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{20,512}", text):
        raise KaggleError("Token inválido. Copie o token gerado em kaggle.com > Settings > API.")
    _write_private(TOKEN_FILE, text)
    return TOKEN_FILE


def _api():
    try:
        # O __init__ do pacote tenta um login automático e engole a falha (kaggle 2.2.4);
        # aqui autenticamos de novo, de forma explícita, para dar uma mensagem clara.
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


def save_and_check_token(token: str) -> dict[str, Any]:
    """Salva o token e confere a conta; um token recusado não fica salvo."""
    path = save_token(token)
    try:
        return check_account()
    except KaggleError:
        path.unlink(missing_ok=True)
        raise


def _hours(value: Any) -> float | None:
    seconds = value.total_seconds() if hasattr(value, "total_seconds") else value
    return round(float(seconds) / 3600, 1) if isinstance(seconds, (int, float)) else None


def quota_summary(quota: Any) -> dict[str, Any] | None:
    """Cota semanal de GPU. Lê os campos, não o texto: o texto do kagglesdk perde os dias
    da duração (30 h aparecem como "21600s")."""
    gpu = getattr(quota, "gpu_quota", None)
    if gpu is None:
        return None
    used, total = _hours(getattr(gpu, "time_used", None)), _hours(getattr(gpu, "total_time_allowed", None))
    if used is None or total is None:
        return None
    refresh = getattr(quota, "quota_refresh_time", None)
    return {
        "gpu_hours_used": used,
        "gpu_hours_total": total,
        "refresh": refresh.date().isoformat() if hasattr(refresh, "date") else None,
    }


def check_account() -> dict[str, Any]:
    api, user = _api()
    quota: Any = None
    try:
        quota = quota_summary(api.quota_view())
    except Exception:  # cota é informativa
        pass
    return {"username": user, "quota": quota}


def _status_name(response: Any) -> str:
    status = getattr(response, "status", response)
    return getattr(status, "name", str(status)).upper()


ACTIVE = ("QUEUED", "RUNNING", "NEW_SCRIPT")
TERMINAL = ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED")


def _check(result: Any, what: str) -> Any:
    """As chamadas da API devolvem o erro dentro da resposta em vez de lançar exceção."""
    if result is None:
        raise KaggleError(f"{what}: o Kaggle não respondeu.")
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    if error or (isinstance(status, str) and status and status.lower() != "ok"):
        raise KaggleError(f"{what}: {error or status}")
    return result


def _dataset_state(api, dataset_id: str) -> tuple[str, int]:
    """(status, versão atual) do dataset; ("missing", 0) se ainda não existe."""
    try:
        data = json.loads(api.dataset_status(dataset_id, format="json(status,current_version_number)"))
    except Exception:
        return "missing", 0
    return str(data.get("status", "")).lower(), int(data.get("current_version_number") or 0)


def _wait_dataset(api, dataset_id: str, after_version: int, timeout: float = 900) -> None:
    """Espera a versão NOVA ficar pronta (logo após o envio, o status ainda é o da anterior)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, version = _dataset_state(api, dataset_id)
        if status == "error":
            raise KaggleError(f"O Kaggle rejeitou o dataset {dataset_id}.")
        if status == "ready" and version > after_version:
            return
        time.sleep(5)
    raise KaggleError("O dataset demorou demais para ficar pronto no Kaggle.")


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
        # Identifica esta execução: o notebook copia para o relatorio.json e conferimos na volta.
        run_id = uuid.uuid4().hex
        (data_dir / "aivs_run.json").write_text(json.dumps({"run_id": run_id, "project_id": project["id"]}))
        (data_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": dataset_slug,
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }))
        progress(0.02, "Enviando o pacote ao Kaggle (dataset privado)")
        state, version = _dataset_state(api, dataset_id)
        if state != "missing":
            _check(api.dataset_create_version(str(data_dir), version_notes="AI Video Studio", quiet=True,
                                              dir_mode="skip"), "Envio do pacote")
        else:
            _check(api.dataset_create_new(str(data_dir), public=False, quiet=True, dir_mode="skip"),
                   "Envio do pacote")
        _wait_dataset(api, dataset_id, after_version=version)

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
        pushed = _check(api.kernels_push(str(kernel_dir), timeout=str(MAX_WAIT_SECONDS)), "Envio do notebook")
        if getattr(pushed, "invalidDatasetSources", None):
            raise KaggleError(f"O Kaggle não anexou o pacote ao notebook: {pushed.invalidDatasetSources}")

        started = time.monotonic()
        last_fraction = 0.05
        seen_active, polls = False, 0
        while True:
            time.sleep(POLL_SECONDS)
            polls += 1
            status = _status_name(api.kernels_status(kernel_id))
            seen_active = seen_active or status in ACTIVE
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
            # Logo após o envio o status ainda pode ser o da execução anterior: só aceita o fim
            # depois de ver a nova versão na fila/rodando (ou após alguns minutos).
            if status in TERMINAL and (seen_active or polls >= 6):
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
        report_data = json.loads(report.read_text()) if report is not None else {}
        if report is not None and report_data.get("run_id") != run_id:
            raise KaggleError(
                "O Kaggle devolveu a saída de outra execução (relatório de outro envio). "
                f"Tente de novo ou confira https://www.kaggle.com/code/{kernel_id}"
            )
        if status != "COMPLETE":
            detail = report_data.get("erro", "")
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
