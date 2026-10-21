#!/usr/bin/env bash
# Sobe o motor (http://127.0.0.1:8765) e o painel (http://localhost:3000).
# Ctrl+C encerra os dois. Use STUDIO_TRACKER=demo para testar a interface sem o SAM 2.
# O que o motor escreve fica em data/logs/motor.log (o da vez anterior, em motor.anterior.log).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/node-env.sh
source "$ROOT/scripts/node-env.sh"
use_vendored_node
STUDIO="$ROOT/engine/.venv/bin/studio"
[[ -x "$STUDIO" ]] || { echo "Motor não instalado. Rode: bash scripts/setup.sh" >&2; exit 1; }
node_ok || { echo "Node.js 20+ não encontrado. Rode: bash scripts/setup.sh" >&2; exit 1; }
[[ -d "$ROOT/web/node_modules" ]] || { echo "Painel não instalado. Rode: bash scripts/setup.sh" >&2; exit 1; }

ALREADY_RUNNING=3  # studio serve: outro motor já usa a mesma pasta de dados (engine/studio/cli.py)
LOGS="${STUDIO_DATA_DIR:-$ROOT/data}/logs"
mkdir -p "$LOGS"

if "$ROOT/engine/.venv/bin/python" -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)" 2>/dev/null; then
  echo "O AI Video Studio já está aberto (em outra janela ou terminal): use http://localhost:3000"
  exit "$ALREADY_RUNNING"
fi

[[ -f "$LOGS/motor.log" ]] && mv -f "$LOGS/motor.log" "$LOGS/motor.anterior.log"
echo "Registro do motor: $LOGS/motor.log"

# Cada um no seu grupo de processos (setsid): Ctrl+C ou a janela fechando chegam só a este script,
# que encerra os dois do jeito certo. O motor recebe um único SIGTERM e, antes de sair, para o
# FaceFusion e grava na tarefa em andamento que ela foi interrompida.
setsid "$STUDIO" serve --host 127.0.0.1 --port 8765 >>"$LOGS/motor.log" 2>&1 </dev/null &
ENGINE_PID=$!
setsid npm --prefix "$ROOT/web" run dev </dev/null &
PANEL_PID=$!

cleanup() {
  trap - EXIT INT TERM HUP
  kill -TERM -- "-$PANEL_PID" 2>/dev/null || true
  kill -TERM "$ENGINE_PID" 2>/dev/null || true
  wait "$ENGINE_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP  # janela do atalho fechada

code=0
wait -n "$ENGINE_PID" "$PANEL_PID" || code=$?
echo
if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
  if ((code == ALREADY_RUNNING)); then
    echo "O AI Video Studio já está aberto (em outra janela ou terminal): use http://localhost:3000"
    exit "$ALREADY_RUNNING"
  elif ((code == 137)); then
    echo "O motor do estúdio foi encerrado à força pelo sistema, quase sempre por falta de memória."
    echo "Feche outros programas e abra o estúdio de novo. Veja \"Se o estúdio fechar sozinho\" no README."
  else
    echo "O motor do estúdio parou (código $code)."
  fi
  echo "Últimas linhas de $LOGS/motor.log:"
  tail -n 15 "$LOGS/motor.log" || true
else
  echo "O painel parou (código $code). Se a porta 3000 estiver ocupada, feche a outra janela do estúdio."
fi
exit 1
