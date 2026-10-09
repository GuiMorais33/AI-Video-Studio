#!/usr/bin/env bash
# Teste da escolha do PyTorch (scripts/torch-choice.sh) com placas NVIDIA simuladas.
#   bash scripts/tests/torch-choice.test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/torch-choice.sh
source "$ROOT/scripts/torch-choice.sh"
STUBS="$(mktemp -d)"
trap 'rm -rf "$STUBS"' EXIT
failures=0

# nvidia-smi falso: responde à consulta de nome, arquitetura e driver.
fake_gpu() {
  cat >"$STUBS/nvidia-smi" <<EOF
#!/usr/bin/env bash
[[ "\$*" == *--query-gpu* ]] && echo "$1"
exit 0
EOF
  chmod +x "$STUBS/nvidia-smi"
}

check() {  # check <descrição> <GB livres> <trecho esperado no motivo; vazio = usa a GPU>
  local got
  got="$(PATH="$STUBS:/usr/bin:/bin" cuda_blocker "$2")"
  if [[ -z "$3" && -z "$got" ]] || [[ -n "$3" && "$got" == *"$3"* ]]; then
    echo "  ok: $1"
  else
    echo "  FALHOU: $1 -> '${got}'"
    failures=$((failures + 1))
  fi
}

echo "Escolha do PyTorch"
rm -f "$STUBS/nvidia-smi"
check "sem placa NVIDIA usa CPU" 50 "nenhuma placa NVIDIA"

fake_gpu "NVIDIA GeForce RTX 3060, 8.6, 581.57"
check "RTX 3060 com driver 581 e espaço usa a GPU" 50 ""
check "pouco espaço usa CPU" 11 "só 11 GB livres"

fake_gpu "NVIDIA GeForce GTX 1060 6GB, 6.1, 581.57"
check "GTX 1060 (Pascal) usa CPU" 50 "anterior às RTX 20"

fake_gpu "NVIDIA GeForce GTX 1650, 7.5, 581.57"
check "GTX 1650 (Turing) usa a GPU" 50 ""

fake_gpu "NVIDIA GeForce RTX 2060, 7.5, 566.36"
check "driver 566 usa CPU e pede atualização" 50 "atualize o driver"

fake_gpu "[N/A], [N/A], [N/A]"
check "consulta ilegível usa CPU" 50 "não foi possível ler"

# python falso do venv: registra as chamadas do pip; a instalação CUDA falha com FAIL_CUDA=1
# e INSTALLED simula a versão do torch já instalada.
cat >"$STUBS/python" <<'PYTHON'
#!/usr/bin/env bash
echo "$*" >>"$CALLS"
case "$*" in
  "-c "*) [[ "${INSTALLED:-}" == *+cpu ]] && exit 0 || exit 1 ;;
  "-m pip list --format=freeze")
    printf 'numpy==2.5.3\ntorch==2.14.1\ntriton==3.8.0\nnvidia-cublas==13.1.1.3\ntorchao==0.9\n' ;;
  "-m pip install"*) [[ "$*" != *whl/cpu* && "${FAIL_CUDA:-0}" == 1 ]] && exit 1 || exit 0 ;;
esac
exit 0
PYTHON
chmod +x "$STUBS/python"
export CALLS="$STUBS/calls"

run_install() {  # run_install <modo> <auto>: status e as chamadas do pip, uma por linha
  : >"$CALLS"
  install_torch "$STUBS/python" "$1" "$2" >/dev/null
  echo "status=$?"
  cat "$CALLS"
}

expect() {  # expect <descrição> <saída> <padrão esperado> [padrão proibido]
  if grep -Eq -- "$3" <<<"$2" && { [[ -z "${4:-}" ]] || ! grep -Eq -- "$4" <<<"$2"; }; then
    echo "  ok: $1"
  else
    echo "  FALHOU: $1"; printf '      %s\n' "${2//$'\n'/$'\n'      }"; failures=$((failures + 1))
  fi
}

echo "Instalação do PyTorch"
out="$(run_install cpu 1)"
expect "só-CPU usa o índice da CPU, sem cache" "$out" "install --quiet --no-cache-dir torch torchvision --index-url .*/whl/cpu"
out="$(FAIL_CUDA=0 run_install cuda 1)"
expect "GPU instala do PyPI e para" "$out" "^status=0" "whl/cpu"
out="$(FAIL_CUDA=1 run_install cuda 1)"
expect "GPU falhou: remove a instalação pela metade" "$out" "uninstall --quiet -y torch triton nvidia-cublas$"
expect "GPU falhou: não remove outros pacotes" "$out" "^status=0" "uninstall.*(numpy|torchao)"
expect "GPU falhou: instala a só-CPU" "$out" "whl/cpu"
out="$(FAIL_CUDA=1 run_install cuda 0)"
expect "GPU forçada que falha devolve erro, sem cair para CPU" "$out" "^status=1" "whl/cpu"
out="$(INSTALLED=2.14.1+cpu run_install cuda 0)"
expect "GPU forçada troca a só-CPU instalada" "$out" "uninstall --quiet -y torch torchvision$"
out="$(INSTALLED=2.14.1+cpu run_install cuda 1)"
expect "automático não troca a só-CPU que já funciona" "$out" "^status=0" "uninstall"

echo "Outros"
if [[ "$(free_gb "$ROOT")" =~ ^[0-9]+$ ]]; then
  echo "  ok: free_gb devolve um inteiro"
else
  echo "  FALHOU: free_gb"; failures=$((failures + 1))
fi

if (( failures )); then echo "$failures verificação(ões) falharam"; exit 1; fi
echo "Todas as verificações passaram"
