#!/bin/zsh
# 双击在浏览器里打开 Entelecheia（隐得莱希）（Vite dev，无需 Electron）。
# 后端需另起；Runtime 与依赖由仓库 bootstrap 准备，不依赖全局 Node 或 pnpm。
set -e
setopt NULL_GLOB          # 不匹配的通配符静默展开为空，而非报错中断
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"
REPO_ROOT="${SCRIPT_DIR:h}"
PORT="${PERSONAGRAPH_FRONTEND_PORT:-5174}"
URL="http://127.0.0.1:${PORT}/"

NODE_BIN="$REPO_ROOT/.runtime/node/bin"
if [ ! -x "$NODE_BIN/node" ]; then
  echo "✗ 项目 Node 运行时不存在。请先在仓库根目录运行 scripts/bootstrap-local-runtime.sh。"
  exit 1
fi
export PATH="$NODE_BIN:$PATH"

if [ ! -d "./node_modules" ]; then
  echo "✗ 前端依赖未安装。请先在仓库根目录运行 scripts/bootstrap-local-runtime.sh。"
  exit 1
fi

echo "在浏览器打开 Entelecheia（隐得莱希）— From intent to actuality."
echo "$URL（Ctrl+C 停止）"
open "$URL" >/dev/null 2>&1 || true
exec "$NODE_BIN/node" ./node_modules/vite/bin/vite.js --host 127.0.0.1 --port "$PORT"
