# shellcheck shell=bash
# Escolha do PyTorch (GPU NVIDIA ou só-CPU), usada por scripts/setup.sh.
# Funções puras para poder testar com uma GPU simulada: scripts/tests/torch-choice.test.sh

# O PyTorch com GPU vem do PyPI com CUDA 13 (desde o PyTorch 2.11): ~2,8 GB de download e
# ~7 GB instalado. O CUDA 13 pede driver NVIDIA 580 ou mais novo e placa Turing (RTX 20 /
# GTX 16) ou mais nova.
TORCH_CUDA_MIN_FREE_GB=12
TORCH_CUDA_MIN_DRIVER=580
TORCH_CUDA_MIN_CAPABILITY=7.5
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# GB livres (inteiro, arredondado para baixo) no sistema de arquivos de um caminho.
free_gb() {
  df -Pk "$1" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

# GB livres para o projeto. No WSL, o "df" do Ubuntu mostra o tamanho virtual do disco dele
# (~1 TB); o limite real é o disco do Windows onde ele fica: /mnt/c, ou a letra que o
# instalador do Windows mediu (AIVS_WIN_DRIVE). Vale o menor dos dois.
#   disk_free_gb <pasta do projeto>
disk_free_gb() {
  local free windows drive="${AIVS_WIN_DRIVE:-c}" mounts="${AIVS_WSL_MOUNTS:-/mnt}"
  free="$(free_gb "$1")"
  if [[ -n "${WSL_DISTRO_NAME:-}" && -d "$mounts/$drive" ]]; then
    windows="$(free_gb "$mounts/$drive" 2>/dev/null || true)"
    if [[ "$windows" =~ ^[0-9]+$ ]] && (( windows < free )); then free="$windows"; fi
  fi
  echo "$free"
}

# Imprime por que NÃO usar a GPU; não imprime nada quando a GPU pode ser usada.
#   cuda_blocker <GB livres>
cuda_blocker() {
  local free="$1" info name capability driver
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "nenhuma placa NVIDIA encontrada"
    return
  fi
  info="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  IFS=',' read -r name capability driver <<<"$info"
  name="$(echo "$name" | xargs)"
  capability="$(echo "$capability" | xargs)"
  driver="$(echo "$driver" | xargs)"
  if [[ ! "$capability" =~ ^[0-9]+\.[0-9]+$ || ! "$driver" =~ ^[0-9]+ ]]; then
    echo "não foi possível ler o modelo e o driver da placa NVIDIA"
    return
  fi
  if awk -v c="$capability" -v m="$TORCH_CUDA_MIN_CAPABILITY" 'BEGIN { exit !(c + 0 < m + 0) }'; then
    echo "a placa $name é anterior às RTX 20 / GTX 16 e o PyTorch atual não a suporta"
    return
  fi
  if (( ${driver%%.*} < TORCH_CUDA_MIN_DRIVER )); then
    echo "o driver NVIDIA $driver é antigo (precisa de $TORCH_CUDA_MIN_DRIVER ou mais novo); atualize o driver no Windows e rode o instalador de novo para usar a placa $name"
    return
  fi
  if (( free < TORCH_CUDA_MIN_FREE_GB )); then
    echo "só $free GB livres no disco; a versão para a placa $name precisa de $TORCH_CUDA_MIN_FREE_GB GB"
  fi
}

# Marca, dentro do venv, que a versão com GPU já falhou no modo automático: as próximas
# execuções (atualizações) não baixam de novo ~3 GB. TORCH=cuda tenta mesmo assim.
torch_gpu_failed_marker() {
  echo "$(dirname "$(dirname "$1")")/.torch-gpu-falhou"
}

# Instala o PyTorch. Com auto=1, se a versão com GPU falhar (download, disco), instala a só-CPU.
#   install_torch <python do venv> <cuda|cpu> <auto: 0|1>
install_torch() {
  local py="$1" mode="$2" auto="$3" partial=() marker
  marker="$(torch_gpu_failed_marker "$py")"
  # Sem cache: o pip não guarda uma segunda cópia dos pacotes grandes.
  if [[ "$mode" == "cpu" ]]; then
    "$py" -m pip install --quiet --no-cache-dir torch torchvision --index-url "$TORCH_CPU_INDEX"
    return
  fi
  if "$py" -c "import sys, torch; sys.exit('+cpu' not in torch.__version__)" 2>/dev/null; then
    if (( auto )) && [[ -e "$marker" ]]; then
      echo "Mantendo o PyTorch só-CPU: a versão para a placa NVIDIA já falhou antes neste computador."
      return 0
    fi
    # Agora a placa pode ser usada (driver atualizado, espaço liberado ou TORCH=cuda):
    # troca a só-CPU instalada, que o pip aceitaria como "já instalado".
    "$py" -m pip uninstall --quiet -y torch torchvision
  fi
  if "$py" -m pip install --quiet --no-cache-dir torch torchvision; then
    rm -f "$marker"
    return 0
  fi
  (( auto )) || return 1
  echo "A versão para a placa NVIDIA não instalou; instalando a versão só-CPU."
  touch "$marker"
  # Tira o que ficou pela metade da versão CUDA; senão o pip acha que o torch já está instalado.
  mapfile -t partial < <("$py" -m pip list --format=freeze 2>/dev/null \
    | grep -Eo '^(torch|torchvision|triton|nvidia-[A-Za-z0-9_.-]+|cuda-[A-Za-z0-9_.-]+)==' | sed 's/==$//')
  (( ${#partial[@]} == 0 )) || "$py" -m pip uninstall --quiet -y "${partial[@]}"
  "$py" -m pip install --quiet --no-cache-dir torch torchvision --index-url "$TORCH_CPU_INDEX"
}
