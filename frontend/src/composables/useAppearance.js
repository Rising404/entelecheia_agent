import { onScopeDispose, ref } from "vue";
import { api } from "../api";

// 背景是一次一个：这是单机单用户的外观选择，不是素材库，换就是覆盖。
export function useAppearance({ showError } = {}) {
  const background = ref(null);
  // 已鉴权取回并转成 object URL 的资产地址；渲染层用它，而不是直接引用后端地址。
  const objectUrl = ref("");
  const busy = ref(false);
  // 记住这份 object URL 对应的资产路径。后端在地址里带了版本参数，所以路径相同就是
  // 同一份字节，不必为每次刷新重下一遍几十 MB。
  let loadedAssetPath = "";

  function releaseObjectUrl() {
    if (objectUrl.value) {
      URL.revokeObjectURL(objectUrl.value);
      objectUrl.value = "";
    }
    loadedAssetPath = "";
  }

  async function resolveAsset() {
    const path = background.value?.url || "";
    if (!path) {
      releaseObjectUrl();
      return;
    }
    if (path === loadedAssetPath && objectUrl.value) return;
    try {
      const blob = await api.fetchBackgroundAsset(path);
      releaseObjectUrl();
      objectUrl.value = URL.createObjectURL(blob);
      loadedAssetPath = path;
    } catch (err) {
      // 背景取不到不该挡住整个界面：BackgroundLayer 会退回直接引用后端地址（Electron
      // 那条路仍然有效），再不行就由 body::before 的渐变兜底。
      releaseObjectUrl();
    }
  }

  async function load() {
    try {
      const payload = await api.getBackground();
      background.value = payload?.background || null;
    } catch (err) {
      // 背景读不到不该挡住整个界面，渐变兜底继续用
      background.value = null;
    }
    await resolveAsset();
  }

  async function upload(file) {
    if (!file) return false;
    busy.value = true;
    try {
      const payload = await api.uploadBackground(file);
      background.value = payload?.background || null;
      await resolveAsset();
      return true;
    } catch (err) {
      showError?.(err);
      return false;
    } finally {
      busy.value = false;
    }
  }

  async function clear() {
    busy.value = true;
    try {
      await api.clearBackground();
      background.value = null;
      releaseObjectUrl();
      return true;
    } catch (err) {
      showError?.(err);
      return false;
    } finally {
      busy.value = false;
    }
  }

  // object URL 活到文档结束为止，组件卸载时不主动释放就是泄漏。
  onScopeDispose(releaseObjectUrl);

  return { background, objectUrl, busy, load, upload, clear };
}
