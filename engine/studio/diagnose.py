"""Diagnóstico do computador: sistema, hardware, ferramentas e pilha de IA."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Settings
from .models import checkpoint_path

GB = 1024 ** 3


def _cmd_output(cmd: list[str], timeout: float = 15) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or proc.stderr).strip() or None


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def _is_wsl() -> bool:
    if "microsoft" in platform.release().lower():
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _memory() -> tuple[float | None, float | None]:
    """(total, disponível) em GB."""
    system = platform.system()
    if system == "Linux":
        info = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
        except (OSError, ValueError):
            return None, None
        return info.get("MemTotal", 0) / GB, info.get("MemAvailable", 0) / GB
    if system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return status.ullTotalPhys / GB, status.ullAvailPhys / GB
    if system == "Darwin":
        total = _cmd_output(["sysctl", "-n", "hw.memsize"])
        return (int(total) / GB if total and total.isdigit() else None), None
    return None, None


def _cpu_model() -> str | None:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Darwin":
        return _cmd_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or None


def _nvidia_gpus() -> list[dict[str, Any]]:
    out = _cmd_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            gpus.append({"name": parts[0], "vram_gb": round(int(parts[1]) / 1024, 1)})
    return gpus


def _tool(name: str, version_args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    return {"found": path is not None, "path": path, "version": _first_line(_cmd_output([name, *version_args]))}


def _torch_info() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch
    except Exception as exc:  # instalação quebrada
        return {"installed": False, "error": str(exc)}
    mps = getattr(torch.backends, "mps", None)
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": bool(mps is not None and mps.is_available()),
        "threads": torch.get_num_threads(),
    }
    if info["cuda"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def collect(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    total_ram, avail_ram = _memory()
    disk_root = settings.data_dir if settings.data_dir.exists() else Path.cwd()
    ffmpeg = _tool("ffmpeg", ["-version"])
    if ffmpeg["found"]:
        encoders = _cmd_output(["ffmpeg", "-hide_banner", "-encoders"]) or ""
        ffmpeg["libx264"] = bool(re.search(r"\blibx264\b", encoders))
    ckpt = checkpoint_path(settings.models_dir, settings.sam2_model)
    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "wsl": _is_wsl(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "hardware": {
            "cpu": _cpu_model(),
            "cpu_threads": os.cpu_count(),
            "ram_total_gb": round(total_ram, 1) if total_ram else None,
            "ram_available_gb": round(avail_ram, 1) if avail_ram else None,
            "nvidia_gpus": _nvidia_gpus(),
        },
        "disk": {"path": str(disk_root), "free_gb": round(shutil.disk_usage(disk_root).free / GB, 1)},
        "tools": {
            "ffmpeg": ffmpeg,
            "ffprobe": _tool("ffprobe", ["-version"]),
            "git": _tool("git", ["--version"]),
            "node": _tool("node", ["--version"]),
            "npm": _tool("npm", ["--version"]),
        },
        "ml": {
            "torch": _torch_info(),
            "sam2_installed": importlib.util.find_spec("sam2") is not None,
            "sam2_model": settings.sam2_model,
            "sam2_checkpoint": str(ckpt),
            "sam2_checkpoint_present": ckpt.exists(),
        },
        "data_dir": str(settings.data_dir),
    }
    report["recommendations"] = recommendations(report)
    return report


def recommendations(r: dict[str, Any]) -> list[str]:
    tips = []
    hw, tools, ml = r["hardware"], r["tools"], r["ml"]
    if r["platform"]["system"] == "Windows":
        tips.append("Windows nativo detectado: rode o motor dentro do WSL2 (Ubuntu), como recomenda a Meta para o SAM 2.")
    if not tools["ffmpeg"]["found"] or not tools["ffprobe"]["found"]:
        tips.append("FFmpeg ausente: instale (no Ubuntu/WSL: sudo apt install ffmpeg).")
    elif not tools["ffmpeg"].get("libx264"):
        tips.append("Este FFmpeg não tem o codificador libx264; instale uma versão completa.")
    for tool in ("git", "node", "npm"):
        if not tools[tool]["found"]:
            tips.append(f"{tool} ausente: necessário para o painel web e atualizações.")
    ram = hw["ram_total_gb"]
    if ram is not None and ram < 8:
        tips.append(f"Apenas {ram} GB de RAM: use clipes de até 3 s e resolução 480p.")
    elif ram is not None and ram < 16:
        tips.append(f"{ram} GB de RAM: clipes de até 5 s em 480p funcionam com o SAM 2.1 Tiny.")
    gpus = hw["nvidia_gpus"]
    best_vram = max((g["vram_gb"] for g in gpus), default=0)
    if not gpus:
        tips.append(
            "Sem GPU NVIDIA: SAM 2.1 Tiny em CPU serve para clipes curtos. "
            "O Wan 2.2 Animate precisará de GPU remota gratuita (ex.: Kaggle)."
        )
    elif best_vram < 16:
        tips.append(f"GPU com {best_vram} GB: ótima para o SAM 2; o Wan 2.2 Animate deve rodar remotamente.")
    else:
        tips.append(f"GPU com {best_vram} GB: dá para testar o Wan 2.2 Animate quantizado (GGUF) localmente.")
    if gpus and ml["torch"].get("installed") and not ml["torch"].get("cuda"):
        tips.append(
            "Há GPU NVIDIA, mas o PyTorch instalado é só-CPU (o SAM 2 funciona, só mais devagar). "
            "Para usar a placa (driver 580+, ~12 GB livres): TORCH=cuda bash scripts/setup.sh"
        )
    if not ml["torch"].get("installed"):
        tips.append("PyTorch não instalado: rode scripts/setup.sh.")
    if not ml["sam2_installed"]:
        tips.append("SAM 2 não instalado: rode scripts/setup.sh.")
    if not ml["sam2_checkpoint_present"]:
        tips.append("Checkpoint do SAM 2.1 ausente: rode `studio models download`.")
    if r["disk"]["free_gb"] < 30:
        tips.append(f"Só {r['disk']['free_gb']} GB livres: os modelos das próximas fases ocupam 20–40 GB.")
    return tips


def format_report(r: dict[str, Any]) -> str:
    def yn(value: bool) -> str:
        return "sim" if value else "NÃO"

    p, hw, tools, ml = r["platform"], r["hardware"], r["tools"], r["ml"]
    torch_info = ml["torch"]
    lines = [
        "AI Video Studio — diagnóstico",
        "",
        f"Sistema     {p['system']} {p['release']} ({p['machine']}){'  [WSL]' if p['wsl'] else ''}",
        f"Python      {r['python']['version']}  ({r['python']['executable']})",
        f"CPU         {hw['cpu'] or '?'}  — {hw['cpu_threads']} threads",
        f"RAM         {hw['ram_total_gb'] or '?'} GB total, {hw['ram_available_gb'] or '?'} GB livres",
        "GPU NVIDIA  " + (", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in hw["nvidia_gpus"]) or "nenhuma"),
        f"Disco       {r['disk']['free_gb']} GB livres em {r['disk']['path']}",
        "",
    ]
    for name, info in tools.items():
        extra = ""
        if name == "ffmpeg" and info["found"]:
            extra = "  libx264: " + yn(info.get("libx264", False))
        lines.append(f"{name:<11} {info['version'] or 'NÃO ENCONTRADO'}{extra}")
    lines += [
        "",
        "PyTorch     " + (
            f"{torch_info['version']}  CUDA: {yn(torch_info['cuda'])}  MPS: {yn(torch_info['mps'])}"
            if torch_info.get("installed") else "NÃO INSTALADO"
        ),
        f"SAM 2       instalado: {yn(ml['sam2_installed'])}  modelo: {ml['sam2_model']}  "
        f"checkpoint: {yn(ml['sam2_checkpoint_present'])}",
        "",
        "Recomendações:",
    ]
    lines += [f"  - {tip}" for tip in r["recommendations"]] or ["  - nenhuma"]
    return "\n".join(lines)
