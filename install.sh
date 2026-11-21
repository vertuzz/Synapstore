#!/bin/sh
# SynapStore installer: puts the `synapstore` binary on PATH, then runs
# `synapstore install` to register the MCP server and memory skill with every
# agent it finds (Claude Code, Codex, OpenCode, Gemini CLI, Cursor, VS Code,
# Windsurf, Claude Desktop, pi).
#
#   curl -fsSL https://raw.githubusercontent.com/vertuzz/synapstore/main/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --embedder multilingual --client claude-code,codex
#
# Arguments are passed to `synapstore install`. Environment:
#   SYNAPSTORE_INSTALL_DIR   where the binary goes (default ~/.local/bin)
#   SYNAPSTORE_VERSION       release tag (default: latest)
#   SYNAPSTORE_NO_SETUP=1    only install the binary
#   SYNAPSTORE_FROM_SOURCE=1 build with cargo instead of downloading
set -eu

REPO="vertuzz/synapstore"
DIR="${SYNAPSTORE_INSTALL_DIR:-$HOME/.local/bin}"
BASE="${SYNAPSTORE_DOWNLOAD_BASE:-https://github.com/$REPO/releases}"

say() { printf 'synapstore-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }

asset_name() {
  os=$(uname -s); arch=$(uname -m)
  case "$arch" in
    x86_64|amd64) arch=x86_64 ;;
    aarch64|arm64) arch=aarch64 ;;
    *) return 1 ;;
  esac
  case "$os" in
    Linux) echo "synapstore-$arch-linux" ;;
    Darwin) echo "synapstore-$arch-macos" ;;
    *) return 1 ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

download() {
  name=$1
  if [ -n "${SYNAPSTORE_VERSION:-}" ]; then url="$BASE/download/$SYNAPSTORE_VERSION/$name.tar.gz"
  else url="$BASE/latest/download/$name.tar.gz"; fi
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  say "downloading $url"
  curl -fsSL --proto '=https,file' "$url" -o "$tmp/a.tar.gz" || return 1
  curl -fsSL --proto '=https,file' "$url.sha256" -o "$tmp/a.sha256" || return 1
  want=$(cut -d' ' -f1 < "$tmp/a.sha256")
  got=$(sha256 "$tmp/a.tar.gz")
  [ "$want" = "$got" ] || die "checksum mismatch for $name.tar.gz"
  tar -xzf "$tmp/a.tar.gz" -C "$tmp"
  mkdir -p "$DIR"
  cp "$tmp/$name/synapstore" "$DIR/synapstore.new"
  chmod 755 "$DIR/synapstore.new"
  mv "$DIR/synapstore.new" "$DIR/synapstore"
}

from_source() {
  command -v cargo >/dev/null 2>&1 || die "no prebuilt binary for this platform and no cargo; install Rust from https://rustup.rs"
  say "building from source with cargo (a few minutes)"
  root=$(mktemp -d)
  cargo install --locked --git "https://github.com/$REPO" --root "$root" synapstore
  mkdir -p "$DIR"
  cp "$root/bin/synapstore" "$DIR/synapstore"
  rm -rf "$root"
}

command -v curl >/dev/null 2>&1 || die "curl is required"
if [ "${SYNAPSTORE_FROM_SOURCE:-}" = 1 ]; then
  from_source
elif name=$(asset_name) && download "$name"; then
  :
else
  say "no prebuilt binary found for $(uname -s)/$(uname -m)"
  from_source
fi

BIN="$DIR/synapstore"
[ -x "$BIN" ] || die "installation failed: $BIN missing"
say "installed $("$BIN" --version) to $BIN"
case ":$PATH:" in
  *":$DIR:"*) ;;
  *) say "add $DIR to your PATH, e.g.: echo 'export PATH=\"$DIR:\$PATH\"' >> ~/.profile" ;;
esac

if [ "${SYNAPSTORE_NO_SETUP:-}" != 1 ]; then
  PATH="$DIR:$PATH" "$BIN" install "$@"
fi
