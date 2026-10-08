"""Catálogo e download dos checkpoints oficiais do SAM 2.1."""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sam2Model:
    key: str
    checkpoint: str
    config: str
    hf_repo: str
    approx_mb: int


# Mesmos arquivos de checkpoints/download_ckpts.sh e build_sam.py do repositório
# oficial (facebookresearch/sam2), versão 2.1.
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODELS = {
    m.key: m
    for m in [
        Sam2Model("tiny", "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml", "facebook/sam2.1-hiera-tiny", 156),
        Sam2Model("small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml", "facebook/sam2.1-hiera-small", 184),
        Sam2Model("base_plus", "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml", "facebook/sam2.1-hiera-base-plus", 324),
        Sam2Model("large", "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml", "facebook/sam2.1-hiera-large", 898),
    ]
}


def get_model(key: str) -> Sam2Model:
    try:
        return SAM2_MODELS[key]
    except KeyError:
        raise ValueError(f"Modelo SAM 2.1 desconhecido: {key!r}. Opções: {', '.join(SAM2_MODELS)}") from None


def checkpoint_path(models_dir: Path, key: str) -> Path:
    return models_dir / "sam2" / get_model(key).checkpoint


def download_urls(model: Sam2Model) -> list[str]:
    return [
        f"{SAM2_BASE_URL}/{model.checkpoint}",
        f"https://huggingface.co/{model.hf_repo}/resolve/main/{model.checkpoint}",
    ]


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stderr.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stderr.flush()
    sys.stderr.write("\n")
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise OSError(f"download incompleto ({done} de {total} bytes)")
    os.replace(tmp, dest)


def download_checkpoint(models_dir: Path, key: str, force: bool = False) -> Path:
    model = get_model(key)
    dest = checkpoint_path(models_dir, key)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in download_urls(model):
        print(f"Baixando {model.checkpoint} (~{model.approx_mb} MB) de {url}", file=sys.stderr)
        try:
            _download(url, dest)
            return dest
        except OSError as exc:
            errors.append(f"{url}: {exc}")
    raise OSError("Não foi possível baixar o checkpoint:\n" + "\n".join(errors))
