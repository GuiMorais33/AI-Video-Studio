"""O notebook do Kaggle é gerado de aivs_wan.py e roda num kernel Jupyter de verdade."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def test_notebook_is_up_to_date():
    result = subprocess.run([sys.executable, str(HERE / "build_notebook.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_notebook_embeds_module_and_pins_versions():
    nb = json.loads((HERE / "wan_animate_kaggle.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"]]
    embedded = next(s for s in sources if s.startswith("%%writefile aivs_wan.py\n"))
    assert embedded.removeprefix("%%writefile aivs_wan.py\n") == (HERE / "aivs_wan.py").read_text()
    install = next(s for s in sources if "'-m', 'pip', 'install'" in s)
    assert "diffusers==0.41.0" in install
    assert not re.search(r"'torch(vision|audio)?(==|')", install)  # mantém o PyTorch do Kaggle
    assert "'uninstall', '-y', '-q', 'torchao'" in install
    namespace: dict = {}
    exec(next(line for line in install.splitlines() if line.startswith("onnxruntime = ")).replace(
        "torch.version.cuda", "cuda"), {"cuda": "12.8"}, namespace)
    assert namespace["onnxruntime"] == "onnxruntime-gpu==1.26.0"
    assert any("main(cfg)" in s for s in sources)


def test_notebook_cells_run_in_a_kernel(tmp_path):
    """Executa diagnóstico, %%writefile e configuração num kernel real (sem instalar nada nem gerar)."""
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(HERE / "wan_animate_kaggle.ipynb", as_version=4)
    def skip(source: str) -> bool:  # não instala pacotes nem roda a geração
        return "'-m', 'pip', 'install'" in source or "main(cfg)" in source

    keep = [c for c in nb.cells if c.cell_type == "code" and not skip("".join(c.source))]
    keep.append(nbformat.v4.new_code_cell("print('CFG', cfg.repo, cfg.steps, cfg.dtype)"))
    nb.cells = keep
    nbclient.NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert (tmp_path / "aivs_wan.py").read_text() == (HERE / "aivs_wan.py").read_text()
    out = "".join(o.get("text", "") for o in nb.cells[-1].outputs)
    assert "CFG Wan-AI/Wan2.2-Animate-14B-Diffusers 6 float16" in out
