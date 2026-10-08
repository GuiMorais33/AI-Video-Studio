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


class FakeApi:
    def __init__(self, final_status="COMPLETE", produce_result=True, dataset_exists=False):
        self.final_status = final_status
        self.produce_result = produce_result
        self.dataset_exists = dataset_exists
        self.calls: list[tuple] = []
        self.status_calls = 0
        self.pushed_metadata: dict = {}
        self.uploaded: list[str] = []

    def dataset_status(self, dataset_id):
        self.calls.append(("dataset_status", dataset_id))
        if not self.dataset_exists:
            raise RuntimeError("404")
        return "ready"

    def _upload(self, folder):
        self.uploaded = sorted(os.listdir(folder))
        meta = json.loads(Path(folder, "dataset-metadata.json").read_text())
        assert meta["id"].startswith("maria/aivs-")
        self.dataset_exists = True

    def dataset_create_new(self, folder, public=False, quiet=False, dir_mode="skip"):
        assert public is False
        self.calls.append(("create_new",))
        self._upload(folder)

    def dataset_create_version(self, folder, version_notes, quiet=False, dir_mode="skip"):
        self.calls.append(("create_version",))
        self._upload(folder)

    def kernels_push(self, folder, timeout=None):
        self.pushed_metadata = json.loads(Path(folder, "kernel-metadata.json").read_text())
        assert Path(folder, self.pushed_metadata["code_file"]).exists()
        self.calls.append(("push", timeout))

    def kernels_status(self, kernel_id):
        self.status_calls += 1
        status = "RUNNING" if self.status_calls < 3 else self.final_status
        return SimpleNamespace(status=SimpleNamespace(name=status), failure_message=None)

    def kernels_logs(self, kernel_id):
        return json.dumps([{"data": f"AIVS_PROGRESS 0.{self.status_calls}0 Gerando segmento {self.status_calls}\n"}])

    def kernels_output(self, kernel_id, path, force=False, quiet=True):
        out = Path(path)
        report = {"etapas": {"geracao_s": 123}}
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
    assert api.uploaded == ["dataset-metadata.json", "wan_package.zip"]
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
        "dataset_status": ["dataset"],
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
