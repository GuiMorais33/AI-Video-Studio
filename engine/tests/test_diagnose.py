"""Dicas do diagnóstico sobre a placa NVIDIA: as mesmas regras do instalador (scripts/torch-choice.sh)."""

from __future__ import annotations

from typing import Any

from studio import diagnose


def _report(gpus: list[dict[str, Any]], *, cuda: bool = False, gpu_failed: bool = False) -> dict[str, Any]:
    found = {"found": True, "libx264": True}
    return {
        "platform": {"system": "Linux"},
        "tools": {name: found for name in ("ffmpeg", "ffprobe", "git", "node", "npm")},
        "hardware": {"ram_total_gb": 16, "nvidia_gpus": gpus},
        "ml": {"torch": {"installed": True, "cuda": cuda}, "sam2_installed": True,
               "sam2_checkpoint_present": True, "torch_gpu_failed": gpu_failed},
        "disk": {"free_gb": 40},
    }


def _gpu(name: str, cap: float | None, driver: str | None) -> dict[str, Any]:
    return {"name": name, "vram_gb": 6.0, "compute_cap": cap, "driver": driver}


def _torch_tip(report: dict[str, Any]) -> str:
    tips = [t for t in diagnose.recommendations(report) if "PyTorch" in t or "placa" in t.lower()]
    return " ".join(tips)


def test_old_gpu_is_not_told_to_install_cuda():
    tip = _torch_tip(_report([_gpu("NVIDIA GeForce GTX 1060 6GB", 6.1, "581.57")]))
    assert "anterior às RTX 20" in tip and "TORCH=cuda" not in tip


def test_old_driver_asks_for_driver_update():
    tip = _torch_tip(_report([_gpu("NVIDIA GeForce RTX 3060", 8.6, "566.36")]))
    assert "atualize o driver" in tip


def test_supported_gpu_points_to_installer():
    tip = _torch_tip(_report([_gpu("NVIDIA GeForce RTX 3060", 8.6, "581.57")]))
    assert "rode o instalador de novo" in tip and "12 GB" in tip


def test_previous_gpu_failure_explains_how_to_retry():
    tip = _torch_tip(_report([_gpu("NVIDIA GeForce RTX 3060", 8.6, "581.57")], gpu_failed=True))
    assert "já falhou" in tip and "TORCH=cuda bash scripts/setup.sh" in tip


def test_no_tip_when_torch_uses_the_gpu():
    tips = diagnose.recommendations(_report([_gpu("NVIDIA GeForce RTX 3060", 8.6, "581.57")], cuda=True))
    assert not any("só-CPU" in t for t in tips)


def test_nvidia_gpus_parses_capability_and_driver(monkeypatch):
    answers = {
        "--query-gpu=name,memory.total": "NVIDIA GeForce RTX 3060, 12288",
        "--query-gpu=compute_cap,driver_version": "8.6, 581.57",
    }
    monkeypatch.setattr(diagnose, "_cmd_output", lambda cmd, timeout=15: answers[cmd[1]])
    assert diagnose._nvidia_gpus() == [
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "compute_cap": 8.6, "driver": "581.57"}
    ]


def test_nvidia_gpus_survives_drivers_without_compute_cap(monkeypatch):
    answers = {
        "--query-gpu=name,memory.total": "NVIDIA GeForce GTX 960, 2048",
        "--query-gpu=compute_cap,driver_version": 'Field "compute_cap" is not a valid field to query.',
    }
    monkeypatch.setattr(diagnose, "_cmd_output", lambda cmd, timeout=15: answers[cmd[1]])
    gpus = diagnose._nvidia_gpus()
    assert gpus == [{"name": "NVIDIA GeForce GTX 960", "vram_gb": 2.0, "compute_cap": None, "driver": None}]
    assert "PyTorch" in _torch_tip(_report(gpus))
