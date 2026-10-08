#!/usr/bin/env bash
# Sobe o motor (http://127.0.0.1:8765) e o painel (http://localhost:3000).
# Ctrl+C encerra os dois. Use STUDIO_TRACKER=demo para testar a interface sem o SAM 2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/node-env.sh
source "$ROOT/scripts/node-env.sh"
use_vendored_node
STUDIO="$ROOT/engine/.venv/bin/studio"
[[ -x "$STUDIO" ]] || { echo "Motor não instalado. Rode: bash scripts/setup.sh" >&2; exit 1; }
node_ok || { echo "Node.js 20+ não encontrado. Rode: bash scripts/setup.sh" >&2; exit 1; }
[[ -d "$ROOT/web/node_modules" ]] || { echo "Painel não instalado. Rode: bash scripts/setup.sh" >&2; exit 1; }

"$STUDIO" serve --host 127.0.0.1 --port 8765 &
ENGINE_PID=$!
trap 'kill "$ENGINE_PID" 2>/dev/null || true' EXIT INT TERM

npm --prefix "$ROOT/web" run dev
