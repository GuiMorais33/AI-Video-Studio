# shellcheck shell=bash
# Funções compartilhadas para garantir Node.js 20+ (painel Next.js).
# Uso: source scripts/node-env.sh; ensure_node   (ou só use_vendored_node)
#
# Se o sistema não tiver Node 20+ (o Ubuntu traz o 18 no apt), baixa o Node 22
# LTS oficial de nodejs.org para .vendor/node, conferindo o SHA-256, sem sudo.

NODE_MAJOR_MIN=20
NODE_DIR="${ROOT:?ROOT não definido}/.vendor/node"

use_vendored_node() {
  if [[ -x "$NODE_DIR/bin/node" ]]; then
    export PATH="$NODE_DIR/bin:$PATH"
  fi
}

node_ok() {
  command -v node >/dev/null || return 1
  local major
  major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null)" || return 1
  [[ "$major" =~ ^[0-9]+$ ]] && ((major >= NODE_MAJOR_MIN))
}

ensure_node() {
  use_vendored_node
  if node_ok; then
    return 0
  fi
  local arch
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64 | arm64) arch=arm64 ;;
    *) echo "Arquitetura $(uname -m) sem Node.js pronto; instale Node 20+ manualmente." >&2; return 1 ;;
  esac
  local base="https://nodejs.org/dist/latest-v22.x"
  local sums file tmp
  sums="$(curl -fsSL "$base/SHASUMS256.txt")" || { echo "Falha ao consultar nodejs.org" >&2; return 1; }
  file="$(awk -v a="linux-$arch.tar.xz" '$2 ~ a"$" {print $2}' <<<"$sums" | head -1)"
  [[ -n "$file" ]] || { echo "Pacote do Node.js para linux-$arch não encontrado" >&2; return 1; }
  tmp="$(mktemp -d)"
  echo "Baixando $file de nodejs.org..."
  if ! curl -fSL --progress-bar -o "$tmp/$file" "$base/$file"; then
    rm -rf "$tmp"
    return 1
  fi
  if ! (cd "$tmp" && grep " $file\$" <<<"$sums" | sha256sum -c --status); then
    echo "Checksum do Node.js não confere; download descartado." >&2
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$NODE_DIR"
  mkdir -p "$NODE_DIR"
  tar -xJf "$tmp/$file" -C "$NODE_DIR" --strip-components=1
  rm -rf "$tmp"
  use_vendored_node
  node_ok
}
