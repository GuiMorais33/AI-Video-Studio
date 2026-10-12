#!/usr/bin/env bash
# Teste do fluxo de scripts/setup.sh com tudo que é externo simulado (pip, git, npm, GPU, disco).
#   bash scripts/tests/setup.test.sh
# Os programas simulados são escritos em aspas simples de propósito (expandem quando rodam).
# shellcheck disable=SC2016
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
failures=0

check() {  # check <descrição> <comando...>: ok quando o comando dá certo
  local description="$1"
  shift
  if "$@"; then echo "  ok: $description"; else echo "  FALHOU: $description"; failures=$((failures + 1)); fi
}

# Ferramentas simuladas. CALLS registra cada chamada (com o TMPDIR em uso nas do pip).
STUBS="$WORK/stubs"
mkdir -p "$STUBS"
stub() { printf '#!/usr/bin/env bash\n%s\n' "$2" >"$STUBS/$1"; chmod +x "$STUBS/$1"; }
for tool in git ffprobe curl xz npm; do stub "$tool" 'echo "'"$tool"' $*" >>"$CALLS"'; done
# FAKE_FFMPEG: versão do FFmpeg do "Ubuntu" (22.04 tem a 4.4).
stub ffmpeg 'echo "ffmpeg $*" >>"$CALLS"; [[ "$1" == -version ]] && echo "ffmpeg version ${FAKE_FFMPEG:-6.1.1}-3ubuntu5"; exit 0'
# FAKE_PY: versão do Python do "Ubuntu" ("3 10" no 22.04). Criar um venv só fica registrado.
stub python3 'if [[ "$1 $2" == "-m venv" ]]; then echo "python3 $*" >>"$CALLS"; exit 0; fi
if [[ -n "${FAKE_PY:-}" && "$*" == "-c import sys; print(*sys.version_info[:2])" ]]; then echo "$FAKE_PY"; exit 0; fi
exec /usr/bin/python3 "$@"'

stub node 'echo 22'
stub df 'printf "Filesystem 1024-blocks Used Available Capacity Mounted\nfake 1 1 %s 1%% /\n" "$FAKE_FREE_KB"'
stub nvidia-smi '[[ -n "${FAKE_GPU:-}" ]] || exit 9; [[ "$*" == *--query-gpu* ]] && echo "$FAKE_GPU"; exit 0'

not() { ! "$@"; }

run_setup() {  # [FF_VENV=0] run_setup <GB livres> [VAR=valor...]: roda numa cópia do projeto
  local root="$WORK/proj-$RANDOM"
  mkdir -p "$root/engine/.venv/bin" "$root/engine/.vendor/sam2/.git" "$root/web"
  cp -r "$REPO/scripts" "$root/"
  cat >"$root/engine/.venv/bin/python" <<'PYTHON'
#!/usr/bin/env bash
echo "python $* TMPDIR=$TMPDIR" >>"$CALLS"
[[ "$*" == "-m pip install"* && "$*" != *whl/cpu* && "$*" == *" torch torchvision"* && "${FAIL_CUDA:-0}" == 1 ]] && exit 1
[[ "$*" == "-c import torch" && "${FAKE_TORCH_OK:-0}" == 1 ]] && exit 0
[[ "$*" == "-c "* ]] && exit 1
exit 0
PYTHON
  printf '#!/usr/bin/env bash\necho "studio $*" >>"$CALLS"\n' >"$root/engine/.venv/bin/studio"
  chmod +x "$root/engine/.venv/bin/python" "$root/engine/.venv/bin/studio"
  # Python do FaceFusion (FF_VENV=0: ainda não existe). FAIL_FF: o pip não instala;
  # FF_NO_PIP: Python pela metade, sem pip.
  if [[ "${FF_VENV:-1}" == 1 ]]; then
    mkdir -p "$root/engine/.vendor/facefusion/.venv/bin"
    cat >"$root/engine/.vendor/facefusion/.venv/bin/python" <<'PYTHON'
#!/usr/bin/env bash
echo "ffpython $* TMPDIR=$TMPDIR" >>"$CALLS"
[[ "$*" == *"--no-cache-dir"* && "${FAIL_FF:-0}" == 1 ]] && exit 1
[[ "$*" == "-m pip --version" && "${FF_NO_PIP:-0}" == 1 ]] && exit 1
exit 0
PYTHON
    chmod +x "$root/engine/.vendor/facefusion/.venv/bin/python"
  fi
  export CALLS="$root/calls" FAKE_FREE_KB=$(( $1 * 1048576 ))
  : >"$CALLS"
  shift
  env "$@" PATH="$STUBS:/usr/bin:/bin" bash "$root/scripts/setup.sh" >"$root/out" 2>&1
  echo "$?" >"$root/status"
  LAST="$root"
}

echo "Setup com placa NVIDIA moderna e espaço"
run_setup 50 FAKE_GPU="NVIDIA GeForce RTX 3060, 8.6, 581.57"
check "termina com sucesso" grep -qx 0 "$LAST/status"
check "instala o PyTorch com GPU" grep -q '==> Instalando PyTorch (cuda)' "$LAST/out"
check "pip baixa no disco do projeto, sem cache" \
  grep -q "pip install --quiet --no-cache-dir torch torchvision TMPDIR=$LAST/.cache/tmp" "$LAST/calls"
check "apaga os temporários no fim" test ! -e "$LAST/.cache/tmp"
check "segue até o diagnóstico" grep -q '^studio diagnose' "$LAST/calls"
check "baixa o FaceFusion no commit fixo" grep -q "git -C $LAST/engine/.vendor/facefusion fetch --quiet --depth 1 \
https://github.com/facefusion/facefusion.git 72470819a0373be3388b3929c8f8f311f418fc3c" "$LAST/calls"
check "instala as dependências fixas do FaceFusion no Python dele" grep -q "^ffpython -m pip install --quiet \
--no-cache-dir numpy==2.4.6 onnx==1.23.1 onnxruntime==1.30.0 opencv-python-headless==5.0.0.93 tqdm==4.70.1 \
scipy==1.18.1 TMPDIR=$LAST/.cache/tmp" "$LAST/calls"
check "sem a interface (gradio) do FaceFusion" not grep -q gradio "$LAST/calls"
check "confere que o FaceFusion abre" grep -q '^ffpython facefusion.py --version' "$LAST/calls"
check "avisa que os modelos baixam no primeiro uso" grep -q 'FaceFusion pronto' "$LAST/out"
check "marca a instalação como concluída" test -e "$LAST/engine/.vendor/facefusion/.venv/.aivs-ok"
check "reaproveita o Python do FaceFusion que funciona" not grep -q 'python3 -m venv' "$LAST/calls"

echo "Ubuntu 22.04 (Python 3.10): o FaceFusion 3.9.1 não roda"
run_setup 50 FAKE_PY="3 10"
check "termina com sucesso" grep -qx 0 "$LAST/status"
check "explica o motivo" grep -q 'Aviso: FaceFusion não instalado: o Python 3.10 deste Ubuntu é antigo' "$LAST/out"
check "diz que rodar de novo não resolve: atualizar o Ubuntu" grep -q 'Atualize o Ubuntu para o 24.04' "$LAST/out"
check "não baixa nada do FaceFusion" not grep -q 'facefusion.git' "$LAST/calls"

echo "FFmpeg antigo (4.4)"
run_setup 50 FAKE_FFMPEG=4.4.2
check "explica o motivo" grep -q 'Aviso: FaceFusion não instalado: o FFmpeg 4.4 deste Ubuntu é antigo' "$LAST/out"
check "não baixa nada do FaceFusion" not grep -q 'facefusion.git' "$LAST/calls"

echo "Setup quando o FaceFusion não instala"
run_setup 50 FAIL_FF=1
check "termina com sucesso mesmo assim" grep -qx 0 "$LAST/status"
check "avisa que o refino do rosto fica de fora" grep -q 'Aviso: FaceFusion não instalado: a instalação falhou' "$LAST/out"
check "não marca como instalado" test ! -e "$LAST/engine/.vendor/facefusion/.venv/.aivs-ok"
check "segue até o diagnóstico" grep -q '^studio diagnose' "$LAST/calls"

echo "Python do FaceFusion pela metade (sem pip)"
run_setup 50 FF_NO_PIP=1
check "recria o Python do FaceFusion" grep -q "python3 -m venv --clear $LAST/engine/.vendor/facefusion/.venv" "$LAST/calls"

echo "Setup quando o download da versão com GPU falha"
run_setup 50 FAKE_GPU="NVIDIA GeForce RTX 3060, 8.6, 581.57" FAIL_CUDA=1
check "termina com sucesso" grep -qx 0 "$LAST/status"
check "cai para a versão só-CPU" grep -q 'whl/cpu' "$LAST/calls"
check "avisa a troca" grep -q 'não instalou; instalando a versão só-CPU' "$LAST/out"

echo "Setup com placa antiga"
run_setup 50 FAKE_GPU="NVIDIA GeForce GTX 1060 6GB, 6.1, 581.57"
check "instala só-CPU e diz por quê" grep -q 'PyTorch só-CPU: a placa NVIDIA GeForce GTX 1060 6GB é anterior' "$LAST/out"

echo "Setup com TORCH=cpu (enviado pelo instalador do Windows)"
run_setup 50 FAKE_GPU="NVIDIA GeForce RTX 3060, 8.6, 581.57" TORCH=cpu
check "respeita TORCH=cpu" grep -q '==> Instalando PyTorch (cpu)' "$LAST/out"

echo "Setup com pouco espaço"
run_setup 3
check "para com erro" grep -qx 1 "$LAST/status"
check "explica quanto espaço falta" grep -q 'pouco espaço em disco (3 GB livres' "$LAST/out"
check "não chega a instalar nada" not grep -q 'pip install' "$LAST/calls"

echo "Instalação do zero com 6 GB (o refino do rosto precisa de espaço para os modelos)"
run_setup 6
check "para com erro" grep -qx 1 "$LAST/status"
check "pede 7 GB" grep -q 'pouco espaço em disco (6 GB livres; são necessários pelo menos 7 GB' "$LAST/out"

echo "Atualização de uma instalação que já existe, com pouco espaço"
run_setup 3 FAKE_TORCH_OK=1
check "segue (1 GB basta para atualizar)" grep -qx 0 "$LAST/status"
check "fica na só-CPU por falta de espaço para a GPU" grep -q '==> Instalando PyTorch (cpu)' "$LAST/out"
check "atualiza o FaceFusion que já existe" grep -q 'FaceFusion pronto' "$LAST/out"

echo "Atualização sem o FaceFusion e com pouco espaço para ele"
FF_VENV=0 run_setup 2 FAKE_TORCH_OK=1
check "segue" grep -qx 0 "$LAST/status"
check "explica que falta espaço para o refino do rosto" \
  grep -q 'Aviso: FaceFusion não instalado: pouco espaço em disco (o refino do rosto precisa de ~3 GB' "$LAST/out"
check "não baixa o FaceFusion" not grep -q 'facefusion.git' "$LAST/calls"
check "não cria o Python do FaceFusion" test ! -e "$LAST/engine/.vendor/facefusion/.venv"

if (( failures )); then echo "$failures verificação(ões) falharam"; exit 1; fi
echo "Todas as verificações passaram"
