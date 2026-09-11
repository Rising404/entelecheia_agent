#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
. "$ROOT/runtime-versions.conf"

NODE_BIN="$ROOT/.runtime/node/bin"
COREPACK="$NODE_BIN/corepack"
if [ ! -x "$NODE_BIN/node" ] || [ ! -x "$COREPACK" ]; then
  echo "project Node runtime is missing; run scripts/bootstrap-local-runtime.sh first" >&2
  exit 1
fi

PATH="$NODE_BIN:/usr/bin:/bin:/usr/sbin:/sbin"
COREPACK_HOME="$ROOT/.runtime/corepack"
PNPM_CONFIG_STORE_DIR="$ROOT/.runtime/pnpm-store"
export PATH COREPACK_HOME PNPM_CONFIG_STORE_DIR

exec "$COREPACK" "pnpm@$PERSONAGRAPH_PNPM_VERSION" "$@"
