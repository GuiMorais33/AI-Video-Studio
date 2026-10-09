"""Automação do Kaggle com uma API simulada (sem rede)."""

from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio import kaggle_remote


OK = SimpleNamespace(status="ok", error="")


class FakeApi:
    def __init__(self, final_status="COMPLETE", produce_result=True, dataset_exists=False,
                 statuses=None, stale=False, dataset_error=""):
        self.final_status = final_status
        self.produce_result = produce_result
        self.dataset_exists = dataset_exists
        self.statuses = statuses  # sequência de status a devolver (padrão: 2x RUNNING e o final)
        self.stale = stale  # devolve a saída de uma execução anterior
        self.dataset_error = dataset_error
        self.calls: list[tuple] = []
        self.status_calls = 0
        self.pushed_metadata: dict = {}
        self.uploaded: list[str] = []
        self.run_id = None
        self.version = 3 if dataset_exists else 0
        self.uploads = 0
        self.status_polls_after_upload = 0

    def dataset_status(self, dataset_id, format=None):
        self.calls.append(("dataset_status", dataset_id))
        if not self.dataset_exists:
            raise RuntimeError("404")
        assert format and format.startswith("json")
        # Logo após o envio, o Kaggle ainda mostra a versão anterior por uma consulta.
        self.status_polls_after_upload += 1
        version = self.version + (1 if self.uploads and self.status_polls_after_upload > 1 else 0)
        return json.dumps({"status": "ready", "current_version_number": version})

    def _upload(self, folder):
        self.uploaded = sorted(os.listdir(folder))
        meta = json.loads(Path(folder, "dataset-metadata.json").read_text())
        assert meta["id"].startswith("maria/aivs-")
        self.run_id = json.loads(Path(folder, "aivs_run.json").read_text())["run_id"]
        self.dataset_exists = True
        self.uploads += 1
        self.status_polls_after_upload = 0
        if self.dataset_error:
            return SimpleNamespace(status="error", error=self.dataset_error)
        return OK

    def dataset_create_new(self, folder, public=False, quiet=False, dir_mode="skip"):
        assert public is False
        self.calls.append(("create_new",))
        return self._upload(folder)

    def dataset_create_version(self, folder, version_notes, quiet=False, dir_mode="skip"):
        self.calls.append(("create_version",))
        return self._upload(folder)

    def kernels_push(self, folder, timeout=None):
        self.pushed_metadata = json.loads(Path(folder, "kernel-metadata.json").read_text())
        assert Path(folder, self.pushed_metadata["code_file"]).exists()
        self.calls.append(("push", timeout))
        return SimpleNamespace(error=None, invalidDatasetSources=[], versionNumber=7, url="u")

    def kernels_status(self, kernel_id):
        self.status_calls += 1
        sequence = self.statuses or ["RUNNING", "RUNNING", self.final_status]
        status = sequence[min(self.status_calls, len(sequence)) - 1]
        return SimpleNamespace(status=SimpleNamespace(name=status), failure_message=None)

    def kernels_logs(self, kernel_id):
        return json.dumps([{"data": f"AIVS_PROGRESS 0.{self.status_calls}0 Gerando segmento {self.status_calls}\n"}])

    def kernels_output(self, kernel_id, path, force=False, quiet=True):
        out = Path(path)
        report = {"etapas": {"geracao_s": 123}, "run_id": "outro" if self.stale else self.run_id}
        if self.final_status != "COMPLETE":
            report["erro"] = "CUDA out of memory"
        (out / "relatorio.json").write_text(json.dumps(report))
        if self.produce_result:
            (out / "resultado.mp4").write_bytes(b"fake mp4")
        return [], ""


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "proj"
    (root / "exports").mkdir(parents=True)
    with zipfile.ZipFile(root / "exports" / "wan_package.zip", "w") as zf:
        zf.writestr("job.json", "{}")
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text("{}")
    return root, notebook


def _patch(monkeypatch, api):
    monkeypatch.setattr(kaggle_remote, "_api", lambda: (api, "maria"))
    monkeypatch.setattr(kaggle_remote, "POLL_SECONDS", 0)
    monkeypatch.setattr(kaggle_remote.time, "sleep", lambda s: None)


def test_run_on_kaggle_success(monkeypatch, project_root):
    root, notebook = project_root
    api = FakeApi()
    _patch(monkeypatch, api)
    progress, imported = [], []

    def fake_import(path: Path) -> str:
        imported.append(path.read_bytes())
        return "Resultado importado"

    message = kaggle_remote.run_on_kaggle(
        {"id": "abc123"}, root, lambda f, m=None: progress.append((f, m)), fake_import, notebook=notebook
    )
    assert message.startswith("Resultado importado")
    assert imported == [b"fake mp4"]
    assert ("create_new",) in api.calls
    assert api.uploaded == ["aivs_run.json", "dataset-metadata.json", "wan_package.zip"]
    meta = api.pushed_metadata
    assert meta["id"] == "maria/ai-video-studio-wan-animate"
    assert meta["dataset_sources"] == ["maria/aivs-abc123"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_internet"] == "true"
    assert meta["is_private"] == "true"
    assert any(m and "Gerando segmento" in m for _, m in progress)
    fractions = [f for f, _ in progress]
    assert fractions == sorted(fractions)
    assert json.loads((root / "exports" / "kaggle_relatorio.json").read_text())["etapas"]["geracao_s"] == 123


def test_run_on_kaggle_reuses_dataset_and_reports_errors(monkeypatch, project_root):
    root, notebook = project_root
    api = FakeApi(final_status="ERROR", produce_result=False, dataset_exists=True)
    _patch(monkeypatch, api)
    with pytest.raises(kaggle_remote.KaggleError, match="CUDA out of memory"):
        kaggle_remote.run_on_kaggle({"id": "abc123"}, root, lambda *a: None, lambda p: "", notebook=notebook)
    assert ("create_version",) in api.calls


def test_requires_package(monkeypatch, tmp_path):
    _patch(monkeypatch, FakeApi())
    (tmp_path / "exports").mkdir()
    with pytest.raises(kaggle_remote.KaggleError, match="pacote"):
        kaggle_remote.run_on_kaggle({"id": "x"}, tmp_path, lambda *a: None, lambda p: "")


def test_save_token_is_private(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_remote, "TOKEN_FILE", tmp_path / ".kaggle" / "access_token")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(kaggle_remote.Path, "home", lambda: tmp_path)
    assert kaggle_remote.token_status()["configured"] is False
    kaggle_remote.save_token("  KGAT_abcdefghijklmnopqrstuvwxyz0123  ")
    path = tmp_path / ".kaggle" / "access_token"
    assert path.read_text() == "KGAT_abcdefghijklmnopqrstuvwxyz0123"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert kaggle_remote.token_status()["configured"] is True
    with pytest.raises(kaggle_remote.KaggleError):
        kaggle_remote.save_token("curto")


def test_calls_match_installed_kaggle_api():
    """As chamadas usadas existem com esses parâmetros na versão fixada do pacote kaggle."""
    import inspect

    extended = pytest.importorskip("kaggle.api.kaggle_api_extended")
    api = extended.KaggleApi
    expected = {
        "authenticate": [],
        "get_config_value": ["name"],
        "dataset_status": ["dataset", "format"],
        "dataset_create_new": ["folder", "public", "quiet", "dir_mode"],
        "dataset_create_version": ["folder", "version_notes", "quiet", "dir_mode"],
        "kernels_push": ["folder", "timeout"],
        "kernels_status": ["kernel"],
        "kernels_logs": ["kernel"],
        "kernels_output": ["kernel", "path", "force", "quiet"],
        "quota_view": [],
    }
    for name, params in expected.items():
        signature = inspect.signature(getattr(api, name))
        assert set(params) <= set(signature.parameters), (name, signature)


def test_waits_for_new_run_before_accepting_complete(monkeypatch, project_root):
    """Logo após o envio, o status ainda é o "COMPLETE" da execução anterior."""
    root, notebook = project_root
    api = FakeApi(statuses=["COMPLETE", "QUEUED", "RUNNING", "COMPLETE"])
    _patch(monkeypatch, api)
    kaggle_remote.run_on_kaggle({"id": "abc123"}, root, lambda *a: None, lambda p: "ok", notebook=notebook)
    assert api.status_calls == 4


def test_rejects_output_from_another_run(monkeypatch, project_root):
    root, notebook = project_root
    api = FakeApi(stale=True)
    _patch(monkeypatch, api)
    imported = []
    with pytest.raises(kaggle_remote.KaggleError, match="outra execução"):
        kaggle_remote.run_on_kaggle({"id": "abc123"}, root, lambda *a: None, imported.append, notebook=notebook)
    assert imported == []


def test_dataset_errors_inside_response_are_raised(monkeypatch, project_root):
    root, notebook = project_root
    api = FakeApi(dataset_error="Quota exceeded")
    _patch(monkeypatch, api)
    with pytest.raises(kaggle_remote.KaggleError, match="Quota exceeded"):
        kaggle_remote.run_on_kaggle({"id": "abc123"}, root, lambda *a: None, lambda p: "", notebook=notebook)
    assert not any(c[0] == "push" for c in api.calls)


def test_waits_for_new_dataset_version(monkeypatch, project_root):
    root, notebook = project_root
    api = FakeApi(dataset_exists=True)
    _patch(monkeypatch, api)
    kaggle_remote.run_on_kaggle({"id": "abc123"}, root, lambda *a: None, lambda p: "ok", notebook=notebook)
    status_checks = [c for c in api.calls if c[0] == "dataset_status"]
    push_index = next(i for i, c in enumerate(api.calls) if c[0] == "push")
    assert len([c for c in api.calls[:push_index] if c[0] == "dataset_status"]) >= 3  # antes, envio, 2 esperas
    assert status_checks


def test_rejected_token_is_not_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_remote, "TOKEN_FILE", tmp_path / ".kaggle" / "access_token")

    def refuse():
        raise kaggle_remote.KaggleError("Não foi possível entrar no Kaggle.")

    monkeypatch.setattr(kaggle_remote, "check_account", refuse)
    with pytest.raises(kaggle_remote.KaggleError):
        kaggle_remote.save_and_check_token("KGAT_tokeninvalido0123456789abcdef")
    assert not (tmp_path / ".kaggle" / "access_token").exists()


def test_accepts_kaggle_json_contents(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_remote, "TOKEN_FILE", tmp_path / ".kaggle" / "access_token")
    monkeypatch.setattr(kaggle_remote.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    pasted = '  {"username":"guimo","key":"0123456789abcdef0123456789abcdef"}\n'
    path = kaggle_remote.save_token(pasted)
    assert path == tmp_path / ".kaggle" / "kaggle.json"
    assert json.loads(path.read_text()) == {"username": "guimo", "key": "0123456789abcdef0123456789abcdef"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert kaggle_remote.token_status() == {"configured": True, "source": str(path)}
    with pytest.raises(kaggle_remote.KaggleError, match="username e key"):
        kaggle_remote.save_token('{"username": "guimo"}')
    with pytest.raises(kaggle_remote.KaggleError, match="JSON"):
        kaggle_remote.save_token("{nao é json")


def test_rejected_kaggle_json_is_not_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_remote.Path, "home", lambda: tmp_path)

    def refuse():
        raise kaggle_remote.KaggleError("Não foi possível entrar no Kaggle.")

    monkeypatch.setattr(kaggle_remote, "check_account", refuse)
    with pytest.raises(kaggle_remote.KaggleError):
        kaggle_remote.save_and_check_token('{"username":"x","key":"0123456789abcdef0123456789abcdef"}')
    assert not (tmp_path / ".kaggle" / "kaggle.json").exists()
