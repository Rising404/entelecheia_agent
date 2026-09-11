import { reactive, ref } from "vue";

/**
 * 已保存的端点配置：列出、增删改、切换，以及按需取回明文。
 *
 * 明文不进普通的列表：设置页每次打开都会拉一次列表，让 key 搭这趟车，
 * 等于让它出现在每一张截图、每一次录屏和每一个渲染进程里。要看就单独取一次，
 * 并且过一会儿自动收回去——因为"看得见"应该是一个瞬间的动作，不是一种状态。
 */
export const REVEAL_TIMEOUT_MS = 30_000;

export function useModelProfiles({ api, showError, refreshConfig }) {
  const profiles = reactive({ model: { profiles: [], active_id: null },
                              vision: { profiles: [], active_id: null } });
  const revealed = reactive({});
  const busy = ref(false);
  const timers = new Map();

  function apply(payload) {
    for (const kind of ["model", "vision"]) {
      const next = payload?.[kind];
      if (!next) continue;
      profiles[kind] = { profiles: next.profiles || [], active_id: next.active_id ?? null };
    }
  }

  async function load() {
    try {
      apply((await api.listModelProfiles()).profiles);
    } catch (err) {
      showError(err);
    }
  }

  async function run(action) {
    busy.value = true;
    try {
      await action();
      await load();
      // 切换或编辑当前配置会改变全局设置，界面上的那份视图要跟着更新。
      await refreshConfig?.();
    } catch (err) {
      showError(err);
    } finally {
      busy.value = false;
    }
  }

  const create = (payload) => run(() => api.createModelProfile(payload));
  const update = ({ id, payload }) => run(() => api.updateModelProfile(id, payload));
  const activate = (id) => run(() => api.activateModelProfile(id));

  function remove(id) {
    if (!globalThis.confirm("删除这份配置？如果它正在使用，删除后将没有生效的配置。")) return;
    hide(id);
    return run(() => api.deleteModelProfile(id));
  }

  async function reveal(id) {
    try {
      const { api_key: key } = await api.revealModelProfileSecret(id);
      revealed[id] = key;
      clearTimeout(timers.get(id));
      timers.set(id, globalThis.setTimeout(() => hide(id), REVEAL_TIMEOUT_MS));
    } catch (err) {
      showError(err);
    }
  }

  function hide(id) {
    clearTimeout(timers.get(id));
    timers.delete(id);
    delete revealed[id];
  }

  async function copy(id) {
    // 取一次明文只为了送进剪贴板，不放进 revealed：复制和"亮在屏幕上"是两件事。
    try {
      const { api_key: key } = await api.revealModelProfileSecret(id);
      await globalThis.navigator?.clipboard?.writeText(key);
    } catch (err) {
      showError(err);
    }
  }

  function hideAll() {
    for (const id of Object.keys(revealed)) hide(id);
  }

  return { profiles, revealed, busy, load, create, update, remove, activate, reveal, hide, hideAll, copy };
}
