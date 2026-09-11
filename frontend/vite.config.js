import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  // 相对资源路径：Electron 用 file:// 加载 dist/index.html，绝对 /assets 会解析到文件系统根 → 白屏。
  base: "./",
  plugins: [vue(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true
  },
  preview: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.spec.js"],
    clearMocks: true
  }
});
