#!/bin/zsh
# 双击启动 Entelecheia（隐得莱希）桌面应用（独立 Electron 窗口，自动构建渲染层 + 拉起后端）。
# Runtime 与依赖由仓库 bootstrap 准备，不依赖全局 Node 或 pnpm。
set -e
setopt NULL_GLOB          # 不匹配的通配符静默展开为空，而非报错中断
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"
REPO_ROOT="${SCRIPT_DIR:h}"

NODE_BIN="$REPO_ROOT/.runtime/node/bin"
if [ ! -x "$NODE_BIN/node" ]; then
  echo "✗ 项目 Node 运行时不存在。请先在仓库根目录运行 scripts/bootstrap-local-runtime.sh。"
  exit 1
fi
export PATH="$NODE_BIN:$PATH"
echo "✓ node: $("$NODE_BIN/node" --version)"

if [ ! -x "./node_modules/.bin/electron" ]; then
  echo "✗ 前端依赖未安装。请先在仓库根目录运行 scripts/bootstrap-local-runtime.sh。"
  exit 1
fi

echo "构建渲染层…"
CI=true "$NODE_BIN/node" ./node_modules/vite/bin/vite.js build
echo "启动 Entelecheia（隐得莱希）桌面应用 — From intent to actuality."
echo "Electron 会自动从仓库 .venv 拉起后端 API…"
exec ./node_modules/.bin/electron .
