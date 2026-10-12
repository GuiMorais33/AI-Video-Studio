#!/usr/bin/env bash
# Instala o AI Video Studio no Linux ou no WSL2 (Ubuntu).
#
#   bash scripts/setup.sh            # usa a GPU NVIDIA se ela, o driver e o espaço em disco permitirem
#   TORCH=cpu bash scripts/setup.sh  # força PyTorch só-CPU (~200 MB)
#   TORCH=cuda bash scripts/setup.sh # força PyTorch com CUDA (~7 GB)
#   SAM2_MODEL=small bash scripts/setup.sh
#
# Tudo fica dentro da pasta do projeto: engine/.venv (Python), engine/.vendor/sam2
# (código oficial da Meta), engine/.vendor/facefusion (refino do rosto, com Python próprio),
# .vendor/node (Node.js, se o do sistema for antigo) e data/models (checkpoints). Nada é pago
# nem enviado.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/node-env.sh
source "$ROOT/scripts/node-env.sh"
# shellcheck source=scripts/torch-choice.sh
source "$ROOT/scripts/torch-choice.sh"
VENV="$ROOT/engine/.venv"
SAM2_DIR="$ROOT/engine/.vendor/sam2"
# Commit fixo do repositório oficial facebookresearch/sam2 (SAM 2.1, dez/2024).
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
# FaceFusion 3.9.1 (commit da tag): troca o rosto do vídeo do Wan pelo das fotos, aqui na CPU.
# O Kaggle bloqueia o FaceFusion, então ele nunca roda lá.
FACEFUSION_DIR="$ROOT/engine/.vendor/facefusion"
FACEFUSION_COMMIT="72470819a0373be3388b3929c8f8f311f418fc3c"
# Só o modo sem interface (headless-run): as versões do requirements.txt do 3.9.1, sem o gradio.
FACEFUSION_PACKAGES=(numpy==2.4.6 onnx==1.23.1 onnxruntime==1.30.0 opencv-python-headless==5.0.0.93
  tqdm==4.70.1 scipy==1.18.1)
# Python do FaceFusion (~0,5 GB) agora e os modelos (~1,7 GB) no primeiro refino do rosto.
FACEFUSION_MIN_FREE_GB=3
TORCH="${TORCH:-auto}"
SAM2_MODEL="${SAM2_MODEL:-tiny}"
# Mínimo para instalar do zero pelo caminho só-CPU: ambiente Python, refino do rosto (com os
# modelos), painel, checkpoint e alguns projetos. Para atualizar uma instalação que já existe,
# 1 GB basta (sem espaço, o FaceFusion fica de fora com um aviso).
MIN_FREE_GB=7

# Temporários no disco do projeto: o /tmp do Ubuntu pode ficar na memória (poucos GB), e o pip
# baixa todos os pacotes para lá antes de instalar (2,8 GB no PyTorch com GPU).
export TMPDIR="$ROOT/.cache/tmp"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }

# FaceFusion com Python próprio: as versões dele (numpy, onnxruntime) não brigam com as do motor.
# Não usa "set -e" aqui dentro (é chamado num if): cada passo devolve o erro explicitamente.
install_facefusion() {
  local dir="$FACEFUSION_DIR" py="$FACEFUSION_DIR/.venv/bin/python"
  # O motor só considera o FaceFusion instalado com este marcador, gravado no fim.
  rm -f "$dir/.venv/.aivs-ok"
  mkdir -p "$dir" || return 1
  [[ -d "$dir/.git" ]] || git init --quiet "$dir" || return 1
  if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$FACEFUSION_COMMIT" ]]; then
    git -C "$dir" fetch --quiet --depth 1 https://github.com/facefusion/facefusion.git "$FACEFUSION_COMMIT" \
      || return 1
    # Os modelos baixados (.assets, fora do git) continuam lá.
    git -C "$dir" checkout --quiet --force FETCH_HEAD || return 1
  fi
  # Python pela metade (ex.: instalação interrompida) não tem pip: recria em vez de reaproveitar.
  "$py" -m pip --version >/dev/null 2>&1 || python3 -m venv --clear "$dir/.venv" || return 1
  "$py" -m pip install --quiet --upgrade pip || return 1
  "$py" -m pip install --quiet --no-cache-dir "${FACEFUSION_PACKAGES[@]}" || return 1
  # Confere que ele abre (importa onnxruntime, OpenCV...).
  (cd "$dir" && "$py" facefusion.py --version) || return 1
  touch "$dir/.venv/.aivs-ok"
}

# Imprime por que este Ubuntu não roda o FaceFusion 3.9.1; nada quando roda.
facefusion_blocker() {
  local py ff
  py="$(python3 -c 'import sys; print(*sys.version_info[:2])')"
  ff="$(ffmpeg -version 2>/dev/null | sed -nE '1s/^ffmpeg version n?([0-9]+)\.([0-9]+).*/\1 \2/p')"
  read -r py_major py_minor <<<"$py"
  if (( py_major < 3 || (py_major == 3 && py_minor < 11) )); then
    echo "o Python $py_major.$py_minor deste Ubuntu é antigo (o FaceFusion pede 3.11 ou mais novo)"
  elif [[ -n "$ff" ]]; then
    read -r ff_major ff_minor <<<"$ff"
    if (( ff_major < 5 || (ff_major == 5 && ff_minor < 1) )); then
      echo "o FFmpeg $ff_major.$ff_minor deste Ubuntu é antigo (o FaceFusion pede 5.1 ou mais novo)"
    fi
  fi
}

facefusion_warning() {
  printf '\nAviso: FaceFusion não instalado: %s.\n' "$1"
  echo "O estúdio funciona, mas o vídeo final sai sem o refino do rosto. ${2:-Rode o instalador de novo depois.}"
}

APT_PACKAGES=(python3 python3-venv python3-pip git ffmpeg curl xz-utils)

missing_tools() {
  local cmd
  for cmd in python3 git ffmpeg ffprobe curl xz; do
    command -v "$cmd" >/dev/null || printf '%s ' "$cmd"
  done
  python3 -c "import venv, ensurepip" 2>/dev/null || printf 'python3-venv '
}

step "Verificando ferramentas do sistema"
missing="$(missing_tools)"
if [[ -n "$missing" ]] && command -v apt-get >/dev/null; then
  echo "Faltando: $missing— instalando com apt (pode pedir a sua senha do Ubuntu)."
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
  missing="$(missing_tools)"
fi
if [[ -n "$missing" ]]; then
  fail "faltando: $missing
No Ubuntu/WSL instale com:
  sudo apt update && sudo apt install -y ${APT_PACKAGES[*]}"
fi
python3 - <<'PY' || fail "é necessário Python 3.10 ou mais novo"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

step "Verificando espaço em disco"
FREE_GB="$(disk_free_gb "$ROOT")"
echo "Livre para o projeto: $FREE_GB GB"
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import torch" 2>/dev/null; then
  MIN_FREE_GB=1
fi
if (( FREE_GB < MIN_FREE_GB )); then
  fail "pouco espaço em disco ($FREE_GB GB livres; são necessários pelo menos $MIN_FREE_GB GB).
No Windows, o Ubuntu guarda os arquivos dentro do disco C: — libere espaço nele e rode de novo."
fi

step "Criando ambiente Python em engine/.venv"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade --quiet pip setuptools wheel

TORCH_AUTO=0
if [[ "$TORCH" == "auto" ]]; then
  TORCH_AUTO=1
  blocker="$(cuda_blocker "$FREE_GB")"
  if [[ -z "$blocker" ]]; then
    TORCH=cuda
  else
    TORCH=cpu
    echo "PyTorch só-CPU: $blocker."
  fi
fi

step "Instalando PyTorch ($TORCH)"
install_torch "$PY" "$TORCH" "$TORCH_AUTO" || fail "não foi possível instalar o PyTorch (veja as mensagens acima)."

step "Instalando SAM 2.1 (repositório oficial da Meta)"
if [[ ! -d "$SAM2_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SAM2_DIR")"
  git clone --quiet https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
git -C "$SAM2_DIR" checkout --quiet "$SAM2_COMMIT" 2>/dev/null \
  || { git -C "$SAM2_DIR" fetch --quiet origin && git -C "$SAM2_DIR" checkout --quiet "$SAM2_COMMIT"; }
# Sem a extensão CUDA opcional (só remove pequenos buracos nas máscaras) e sem
# isolamento de build, para reaproveitar o PyTorch já instalado.
SAM2_BUILD_CUDA=0 "$PY" -m pip install --quiet --no-build-isolation -e "$SAM2_DIR"

step "Instalando o motor do AI Video Studio"
"$PY" -m pip install --quiet -e "$ROOT/engine[dev,kaggle]"

step "Instalando FaceFusion 3.9.1 (refino do rosto, roda neste computador)"
ff_blocker="$(facefusion_blocker)"
if [[ -n "$ff_blocker" ]]; then
  # Ubuntu 22.04: rodar de novo não resolve.
  facefusion_warning "$ff_blocker" "Atualize o Ubuntu para o 24.04 (sudo do-release-upgrade) e rode o instalador de novo."
elif [[ ! -x "$FACEFUSION_DIR/.venv/bin/python" ]] && (( $(disk_free_gb "$ROOT") < FACEFUSION_MIN_FREE_GB )); then
  facefusion_warning "pouco espaço em disco (o refino do rosto precisa de ~$FACEFUSION_MIN_FREE_GB GB: ~0,5 GB agora e ~1,7 GB de modelos no primeiro uso)"
elif install_facefusion; then
  echo "FaceFusion pronto. Os modelos (~1,7 GB) baixam no primeiro refino do rosto."
else
  facefusion_warning "a instalação falhou (veja as mensagens acima)"
fi

step "Baixando checkpoint SAM 2.1 ($SAM2_MODEL)"
"$VENV/bin/studio" models download --model "$SAM2_MODEL"

step "Preparando Node.js 20+ e o painel web"
if ensure_node; then
  echo "Node.js $(node --version)"
  npm --prefix "$ROOT/web" ci --no-fund --no-audit
else
  printf '\nAviso: não foi possível preparar o Node.js 20+; o painel web não foi instalado.\n'
fi

step "Diagnóstico"
"$VENV/bin/studio" diagnose

cat <<EOF

Pronto. Para abrir o estúdio:
  bash scripts/start.sh
e acesse http://localhost:3000 no navegador (no Windows também funciona).

Teste rápido sem interface (clique no centro do quadro 0):
  engine/.venv/bin/studio track SEU_VIDEO.mp4 --point 0.5,0.5
EOF
