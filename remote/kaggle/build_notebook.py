"""Gera wan_animate_kaggle.ipynb a partir de aivs_wan.py (fonte única, testada na CPU).

    python remote/kaggle/build_notebook.py          # regrava o notebook
    python remote/kaggle/build_notebook.py --check  # falha se o notebook estiver desatualizado
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "wan_animate_kaggle.ipynb"

# Versões validadas com os testes de remote/kaggle/tests (o PyTorch do Kaggle é mantido).
PACKAGES = [
    "diffusers==0.41.0",
    "transformers==5.19.0",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "gguf==0.19.0",
    "ftfy",
    "sentencepiece",
    # FaceFusion 3.9.1 (refino do rosto): o resto do que ele usa (OpenCV, SciPy, tqdm) já vem no Kaggle.
    "onnx==1.23.1",
]

INTRO = """# AI Video Studio — troca de personagem com Wan 2.2 Animate

Este notebook é gerado pelo AI Video Studio. Ele:

1. lê o `wan_package.zip` anexado como dataset;
2. extrai a pose e o rosto da pessoa;
3. gera o personagem da imagem de referência no lugar dela, com o Wan 2.2 Animate;
4. troca o rosto gerado pelo rosto das fotos enviadas (FaceFusion), quadro a quadro;
5. grava `resultado.mp4` e `relatorio.json` na aba **Output**.

**Antes de rodar:**

- **Acelerador:** *GPU T4 x2*.
- **Internet:** ligada (Settings → Internet on). Requer o telefone verificado na conta.
- **Dados:** o dataset com o `wan_package.zip` anexado (Add Input).

Depois use **Save Version → Save & Run All**. Um clipe de 5 s a 16 fps leva cerca de 35 minutos numa
T4: ~9 min de instalação e download dos modelos (~28 GB por sessão) e ~21 min de geração.

**Regras para não ter a conta bloqueada:**

- notebook e dataset **privados**;
- só conteúdo apropriado (o Kaggle bane conteúdo adulto automaticamente);
- nada de túneis ou interfaces web;
- use apenas vídeos seus ou com autorização.
"""

DIAGNOSTICS = """import subprocess, sys, shutil
print(sys.version)
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("ATENÇÃO: sem GPU. Ative o acelerador 'GPU T4 x2' nas configurações do notebook.")
for path in ("/kaggle/working", "/kaggle/tmp", "/tmp"):
    try:
        print(path, "livre:", round(shutil.disk_usage(path).free / 2**30, 1), "GB")
    except FileNotFoundError:
        pass"""

# onnxruntime-gpu (pose) precisa da mesma versão principal de CUDA que o PyTorch da imagem:
# 1.26.0 é a última feita para CUDA 12; a partir da 1.27 é CUDA 13.
ONNXRUNTIME = {"12": "onnxruntime-gpu==1.26.0", "13": "onnxruntime-gpu==1.30.0"}

INSTALL = (
    "import subprocess, sys\n"
    "import torch\n"
    "# Mantém o PyTorch da imagem do Kaggle; fixa o resto nas versões testadas.\n"
    f"onnxruntime = {ONNXRUNTIME!r}.get((torch.version.cuda or '12').split('.')[0], {ONNXRUNTIME['12']!r})\n"
    "print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, '->', onnxruntime)\n"
    "# O torchao que vem na imagem é antigo e quebra a importação do diffusers (não é usado aqui).\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)\n"
    f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {', '.join(repr(p) for p in PACKAGES)}, onnxruntime], check=True)"
)

CONFIG = """from aivs_wan import Config, main

cfg = Config()
# Ajustes opcionais (os padrões foram escolhidos para a T4 grátis):
# cfg.steps = 6          # passos de geração (LoRA lightx2v: 4 a 8)
# cfg.seed = 42          # mude para obter outra variação
# cfg.mask_block = 16    # 32 = contorno mais folgado (mais espaço para roupas largas)
# cfg.dtype = "float32"  # use se o relatório acusar NaN (bem mais lento)
# cfg.face_refine = False                      # desliga o refino do rosto (FaceFusion)
# cfg.face_swapper_model = "inswapper_128"     # outro trocador de rosto do FaceFusion
cfg"""

RUN = """import json
relatorio = main(cfg)
print(json.dumps({k: relatorio.get(k) for k in ("ok", "etapas", "geracao", "vram_pico_gb", "rosto", "avisos")}, indent=2, ensure_ascii=False))"""


def _cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build() -> dict:
    module = (HERE / "aivs_wan.py").read_text()
    cells = [
        _cell("markdown", INTRO),
        _cell("code", DIAGNOSTICS),
        _cell("code", INSTALL),
        _cell("markdown", "Código do AI Video Studio (gerado de `remote/kaggle/aivs_wan.py`; não edite aqui)."),
        _cell("code", "%%writefile aivs_wan.py\n" + module),
        _cell("code", CONFIG),
        _cell("code", RUN),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        if not NOTEBOOK.exists() or NOTEBOOK.read_text() != text:
            sys.exit("wan_animate_kaggle.ipynb desatualizado: rode python remote/kaggle/build_notebook.py")
        print("notebook em dia")
    else:
        NOTEBOOK.write_text(text)
        print(f"gravado {NOTEBOOK}")
