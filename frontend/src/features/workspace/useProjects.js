import { computed, ref } from "vue";

// 一个 project 就是"会话绑定的那个目录"。分组不需要用户维护，会话自己说了算；
// 前端只额外记住"新建但还没发过消息"的那一条草稿。
export function useProjects({ apiClient, showError } = {}) {
  if (!apiClient) throw new Error("useProjects requires an injected API client.");
  const projects = ref([]);
  const unbound = ref([]);
  const loading = ref(false);

  // 草稿：还没落库的会话。null 表示当前没有草稿。
  const draft = ref(null);

  const draftActive = computed(() => draft.value !== null);
  const draftWorkingDir = computed(() => draft.value?.workingDir || "");
  const draftAttachments = computed(() => draft.value?.attachments || []);
  const draftDirectoryLocked = computed(() => Boolean(draft.value?.creationRequest));

  // 整份写入而不是"把 A 移到 B 前面"：相对指令要求两边对同一份列表有一致认知，
  // 而列表随时会因为会话状态变化而变。整份落下来的就是用户看到的那个顺序。
  async function reorder(paths) {
    const before = projects.value;
    // 先本地重排，拖完立刻定位；失败再退回去
    const byPath = new Map(projects.value.map((item) => [item.path, item]));
    projects.value = paths.map((path) => byPath.get(path)).filter(Boolean);
    try {
      await apiClient.reorderProjects(paths);
      return true;
    } catch (err) {
      projects.value = before;
      showError?.(err);
      return false;
    }
  }

  async function setPinned(path, pinned) {
    try {
      await apiClient.pinProject(path, pinned);
      // 本地先翻，避免等一次完整重取；顺序交给下一次 load 收敛
      projects.value = projects.value.map(
        (item) => (item.path === path ? { ...item, pinned } : item)
      );
      return true;
    } catch (err) {
      showError?.(err);
      return false;
    }
  }

  async function load(status = "active", query = "") {
    loading.value = true;
    try {
      const payload = await apiClient.listProjects(status, query);
      projects.value = Array.isArray(payload?.projects) ? payload.projects : [];
      unbound.value = Array.isArray(payload?.unbound) ? payload.unbound : [];
    } catch (err) {
      if (showError) showError(err);
    } finally {
      loading.value = false;
    }
  }

  function startDraft(workingDir = "") {
    draft.value = {
      workingDir: String(workingDir || ""), attachments: [], creationRequest: null, session: null
    };
  }

  function setDraftDirectory(workingDir) {
    if (draft.value && !draftDirectoryLocked.value) draft.value.workingDir = String(workingDir || "");
  }

  function setDraftAttachments(attachments) {
    if (draft.value) draft.value.attachments = attachments;
  }

  function discardDraft() {
    draft.value = null;
  }

  function prepareDraftCreation(title) {
    if (!draft.value) throw new Error("没有待创建的会话草稿");
    if (!draft.value.creationRequest) {
      const workingDir = draft.value.workingDir.trim();
      draft.value.creationRequest = Object.freeze({
        client_request_id: crypto.randomUUID(),
        title,
        ...(workingDir ? { working_dir: workingDir } : {})
      });
    }
    return draft.value.creationRequest;
  }

  function resetDraftCreation() {
    if (draft.value && !draft.value.session) draft.value.creationRequest = null;
  }

  // 目录登记成 project 失败不该拦住这一轮对话：会话本身已经绑好了目录，
  // 下次分组时它还是会出现在同一个目录下面，只是名字回到目录名。
  async function remember(path, name) {
    if (!path) return null;
    try {
      const payload = await apiClient.rememberProject({ path, ...(name ? { name } : {}) });
      return payload?.project || null;
    } catch {
      return null;
    }
  }

  async function rename(path, name) {
    try {
      await apiClient.renameProject({ path, name });
      await load();
      return true;
    } catch (err) {
      if (showError) showError(err);
      return false;
    }
  }

  async function forget(path) {
    try {
      await apiClient.forgetProject(path);
      await load();
      return true;
    } catch (err) {
      if (showError) showError(err);
      return false;
    }
  }

  return {
    projects, unbound, loading, draft, draftActive, draftWorkingDir, draftAttachments, draftDirectoryLocked,
    prepareDraftCreation, resetDraftCreation,
    setDraftDirectory, setDraftAttachments, setPinned, reorder,
    load, startDraft, discardDraft, remember, rename, forget
  };
}
