#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
. "$ROOT/runtime-versions.conf"

RUNTIME_DIR="$ROOT/.runtime"
INSTALL_DEPENDENCIES=1
if [ "${1:-}" = "--runtime-only" ]; then
  INSTALL_DEPENDENCIES=0
elif [ "$#" -gt 0 ]; then
  echo "usage: $0 [--runtime-only]" >&2
  exit 2
fi

if [ -L "$RUNTIME_DIR" ] || [ -L "$RUNTIME_DIR/python" ] || [ -L "$RUNTIME_DIR/node" ]; then
  echo "project runtime roots must be real directories, not symbolic links" >&2
  exit 1
fi

OS=$(uname -s)
ARCH=$(uname -m)
if [ "$INSTALL_DEPENDENCIES" -eq 1 ] && [ "$OS:$ARCH" != "Darwin:arm64" ]; then
  echo "locked dependency installation is currently validated only for macOS arm64; use --runtime-only on other supported platforms" >&2
  exit 2
fi
case "$OS:$ARCH" in
  Darwin:arm64)
    PYTHON_PLATFORM=aarch64-apple-darwin
    PYTHON_SHA256=25baa97c65b3f0aa90e21131b4f9e80aef8899e8144006db8a9d2c1ab9e807e3
    NODE_PLATFORM=darwin-arm64
    NODE_SHA256=a1a54f46a750d2523d628d924aab61758a51c9dad3e0238beb14141be9615dd3
    ;;
  Darwin:x86_64)
    PYTHON_PLATFORM=x86_64-apple-darwin
    PYTHON_SHA256=127053f1736f721e391ddb46f07585d05756e15bb8d757d3bbc0519738998ba1
    NODE_PLATFORM=darwin-x64
    NODE_SHA256=f2879eb810e25993a0578e5d878930266fd2eafcffe9f2839b3d8db354d4879e
    ;;
  Linux:aarch64|Linux:arm64)
    PYTHON_PLATFORM=aarch64-unknown-linux-gnu
    PYTHON_SHA256=11e713ae1f969385907a76533cc554f6b2e3f1a84896009f91c3fc871034a170
    NODE_PLATFORM=linux-arm64
    NODE_SHA256=f44740cd218de8127f1c44c41510a3a740fa5c9c8d1cdce1c3bedada79f3cde7
    ;;
  Linux:x86_64)
    PYTHON_PLATFORM=x86_64-unknown-linux-gnu
    PYTHON_SHA256=506191be3ee7bd190a8834dcdc1b3bc70aab50608deccc711935aa007239cabd
    NODE_PLATFORM=linux-x64
    NODE_SHA256=dbf5b8665dec15e59e6359a517fefb47b23fdb9152d8def975b9bca3dfc6d355
    ;;
  *)
    echo "unsupported standalone runtime platform: $OS $ARCH" >&2
    exit 1
    ;;
esac

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/entelecheia-runtime.XXXXXX")
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT HUP INT TERM

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "neither shasum nor sha256sum is available" >&2
    exit 1
  fi
}

download_verified() {
  url=$1
  destination=$2
  expected=$3
  curl --fail --location --show-error --silent "$url" --output "$destination"
  actual=$(sha256_file "$destination")
  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch for $url" >&2
    exit 1
  fi
}

python_matches() {
  [ -x "$RUNTIME_DIR/python/bin/python3" ] &&
    [ "$("$RUNTIME_DIR/python/bin/python3" -c 'import platform; print(platform.python_version())')" = "$PERSONAGRAPH_PYTHON_VERSION" ]
}

node_matches() {
  [ -x "$RUNTIME_DIR/node/bin/node" ] &&
    [ "$("$RUNTIME_DIR/node/bin/node" --version)" = "v$PERSONAGRAPH_NODE_VERSION" ]
}

mkdir -p "$RUNTIME_DIR"

if ! python_matches; then
  PYTHON_ARCHIVE="cpython-$PERSONAGRAPH_PYTHON_VERSION+$PERSONAGRAPH_PYTHON_STANDALONE_RELEASE-$PYTHON_PLATFORM-install_only_stripped.tar.gz"
  PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PERSONAGRAPH_PYTHON_STANDALONE_RELEASE/$PYTHON_ARCHIVE"
  download_verified "$PYTHON_URL" "$TMP_ROOT/python.tar.gz" "$PYTHON_SHA256"
  mkdir -p "$TMP_ROOT/python-stage"
  tar -xzf "$TMP_ROOT/python.tar.gz" -C "$TMP_ROOT/python-stage"
  [ -x "$TMP_ROOT/python-stage/python/bin/python3" ] || {
    echo "downloaded Python runtime has an invalid layout" >&2
    exit 1
  }
  rm -rf "$RUNTIME_DIR/python"
  mv "$TMP_ROOT/python-stage/python" "$RUNTIME_DIR/python"
fi

if ! node_matches; then
  NODE_ARCHIVE="node-v$PERSONAGRAPH_NODE_VERSION-$NODE_PLATFORM.tar.gz"
  NODE_URL="https://nodejs.org/dist/v$PERSONAGRAPH_NODE_VERSION/$NODE_ARCHIVE"
  download_verified "$NODE_URL" "$TMP_ROOT/node.tar.gz" "$NODE_SHA256"
  mkdir -p "$TMP_ROOT/node-stage"
  tar -xzf "$TMP_ROOT/node.tar.gz" -C "$TMP_ROOT/node-stage" --strip-components=1
  [ -x "$TMP_ROOT/node-stage/bin/node" ] || {
    echo "downloaded Node runtime has an invalid layout" >&2
    exit 1
  }
  rm -rf "$RUNTIME_DIR/node"
  mv "$TMP_ROOT/node-stage" "$RUNTIME_DIR/node"
fi

"$RUNTIME_DIR/python/bin/python3" -m venv --upgrade "$ROOT/.venv"
# CPython's same-version --upgrade leaves an existing interpreter symlink untouched.
# Replace all public venv interpreter links explicitly so no previous provider survives.
ln -sfn "$RUNTIME_DIR/python/bin/python3" "$ROOT/.venv/bin/python3"
ln -sfn python3 "$ROOT/.venv/bin/python"
ln -sfn python3 "$ROOT/.venv/bin/python3.12"

if [ "$INSTALL_DEPENDENCIES" -eq 1 ]; then
  # OmegaConf's antlr runtime is source-only. Bootstrap its build backend from
  # this same hash lock before building anything; do not resolve an isolated backend.
  awk '/^[^[:space:]#]/ { selected = ($0 ~ /^setuptools==/) } selected { print }' \
    "$ROOT/requirements-macos-arm64.lock" | \
    "$ROOT/.venv/bin/python" -I -m pip --isolated --disable-pip-version-check \
      --cache-dir "$RUNTIME_DIR/pip-cache" install --require-hashes \
      --only-binary=:all: --no-deps --index-url https://pypi.org/simple -r /dev/stdin
  "$ROOT/.venv/bin/python" -I -m pip --isolated --disable-pip-version-check \
    --cache-dir "$RUNTIME_DIR/pip-cache" install --require-hashes --only-binary=:all: \
    --no-binary=antlr4-python3-runtime --no-build-isolation \
    --index-url https://pypi.org/simple -r "$ROOT/requirements-macos-arm64.lock"
  # Dependencies and the build backend come from the hash lock, not a second resolver.
  "$ROOT/.venv/bin/python" -I -m pip --isolated --disable-pip-version-check \
    install --no-index --no-deps --no-build-isolation \
    -e "$ROOT[embeddings,documents-layout,eval,dev]"
  "$ROOT/.venv/bin/python" -I -m pip --isolated --disable-pip-version-check check
  export PATH="$RUNTIME_DIR/node/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  export COREPACK_HOME="$RUNTIME_DIR/corepack"
  export PNPM_CONFIG_STORE_DIR="$RUNTIME_DIR/pnpm-store"
  "$RUNTIME_DIR/node/bin/corepack" pnpm@"$PERSONAGRAPH_PNPM_VERSION" \
    --dir "$ROOT/frontend" install --frozen-lockfile
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/verify-runtime-independence.py"
