#!/usr/bin/env bash
# Instala o AI Video Studio no Linux ou no WSL2 (Ubuntu).
#
#   bash scripts/setup.sh            # detecta GPU NVIDIA; sem GPU instala PyTorch só-CPU
#   TORCH=cpu bash scripts/setup.sh  # força PyTorch só-CPU
#   TORCH=cuda bash scripts/setup.sh # força PyTorch com CUDA
#   SAM2_MODEL=small bash scripts/setup.sh
#
# Tudo fica dentro da pasta do projeto: engine/.venv (Python), engine/.vendor/sam2
# (código oficial da Meta) e data/models (checkpoints). Nada é pago nem enviado.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/engine/.venv"
SAM2_DIR="$ROOT/engine/.vendor/sam2"
# Commit fixo do repositório oficial facebookresearch/sam2 (SAM 2.1, dez/2024).
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
TORCH="${TORCH:-auto}"
SAM2_MODEL="${SAM2_MODEL:-tiny}"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }

step "Verificando ferramentas do sistema"
missing=()
for cmd in python3 git ffmpeg ffprobe; do
  command -v "$cmd" >/dev/null || missing+=("$cmd")
done
if ((${#missing[@]})); then
  fail "faltando: ${missing[*]}
No Ubuntu/WSL instale com:
  sudo apt update && sudo apt install -y python3 python3-venv python3-pip git ffmpeg"
fi
python3 - <<'PY' || fail "é necessário Python 3.10 ou mais novo"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
python3 -c "import venv, ensurepip" 2>/dev/null \
  || fail "módulo venv ausente. Instale com: sudo apt install -y python3-venv"

step "Criando ambiente Python em engine/.venv"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade --quiet pip setuptools wheel

if [[ "$TORCH" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then TORCH=cuda; else TORCH=cpu; fi
fi
step "Instalando PyTorch ($TORCH)"
if [[ "$TORCH" == "cpu" ]]; then
  # Pacote só-CPU (~200 MB) em vez do pacote CUDA padrão (vários GB).
  "$PY" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install --quiet torch torchvision
fi

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
"$PY" -m pip install --quiet -e "$ROOT/engine[dev]"

step "Baixando checkpoint SAM 2.1 ($SAM2_MODEL)"
"$VENV/bin/studio" models download --model "$SAM2_MODEL"

if command -v npm >/dev/null; then
  step "Instalando dependências do painel web"
  npm --prefix "$ROOT/web" install --no-fund --no-audit
else
  printf '\nAviso: npm não encontrado; o painel web não foi instalado (Node.js 20+ necessário).\n'
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
