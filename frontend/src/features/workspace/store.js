import { computed, ref } from "vue";
import { defineStore } from "pinia";

import { useWorkspaceContents } from "./useWorkspaceContents";

/**
 * 仅供工作区外壳使用的共享状态。
 *
 * `folder_id` 是历史组织轴，`working_dir` 是本地文件上下文轴。两者刻意保持独立：
 * 文件夹绝不授予文件访问权，工作目录也绝不决定会话在历史中的位置。文档使用
 * 当前会话的挂载；表单草稿仍为功能局部状态。
 */
// Pinia 只拥有工作区投影。外壳负责选择传输方式，并在组装 store 时传入精简的
// 工作区形态客户端。
export function createWorkspaceStore({ client } = {}) {
  if (!client) throw new Error("createWorkspaceStore requires an injected workspace client.");

  return defineStore("workspace", () => createWorkspaceStoreSetup(client));
}

function createWorkspaceStoreSetup(client) {
  const folders = ref([]);
  const sessions = ref([]);
  const selectedFolderId = ref("");
  const selectedSessionId = ref("");
  const {
    workspaceDocuments,
    contentsLoading,
    resetForSession,
    refreshContents
  } = useWorkspaceContents({ client, selectedSessionId });
  const loading = ref(false);
  const error = ref("");
  let refreshGeneration = 0;
  let selectionGeneration = 0;

  const selectedSession = computed(() =>
    sessions.value.find((session) => session.id === selectedSessionId.value) || null
  );

  const selectedFolder = computed(() => findFolder(folders.value, selectedFolderId.value));
  const historyFolderOptions = computed(() => flattenFolderOptions(folders.value));

  async function refresh({ status = "active", query = "" } = {}) {
    const requestGeneration = ++refreshGeneration;
    const selectionGenerationAtStart = selectionGeneration;
    loading.value = true;
    error.value = "";
    try {
      const sessionParams = { status, limit: 200 };
      if (String(query).trim()) sessionParams.query = String(query).trim();
      const [folderResult, sessionResult] = await Promise.all([
        client.listFolders({ status }),
        client.listSessions(sessionParams)
      ]);
      if (requestGeneration !== refreshGeneration) return;
      folders.value = folderResult.folders || [];
      sessions.value = sessionResult.sessions || [];
      if (selectionGenerationAtStart === selectionGeneration) {
        ensureSelectionIsVisible();
      }
    } catch (err) {
      if (requestGeneration === refreshGeneration) {
        error.value = err?.message || String(err);
      }
      throw err;
    } finally {
      if (requestGeneration === refreshGeneration) loading.value = false;
    }
  }

  function selectSession(sessionId = "") {
    if (selectedSessionId.value === sessionId) return;
    selectedSessionId.value = sessionId;
    selectionGeneration += 1;
    resetForSession();
  }

  function selectFolder(folderId = "") {
    selectedFolderId.value = folderId;
  }

  // 工作目录由后端创建并绑定；前端只保留独立的历史放置位置，不猜平台路径。
  async function createFreeSession({ title, folderId = null } = {}) {
    const result = await client.createSession({
      title: String(title || "").trim() || null,
      ...(folderId ? { folder_id: folderId } : {})
    });
    return afterCreate(result.session);
  }

  async function afterCreate(session) {
    await refresh();
    if (session) {
      selectedFolderId.value = session.folder_id || "";
      selectSession(session.id);
    }
    return session || null;
  }

  async function createFolder({ name, parentId = null } = {}) {
    const result = await client.createFolder({
      name: String(name || "").trim(),
      parent_id: parentId || null
    });
    selectedFolderId.value = result.folder?.id || selectedFolderId.value;
    return result.folder || null;
  }

  async function renameFolder(folderId, name) {
    const result = await client.patchFolder(folderId, { name: String(name || "").trim() });
    return result.folder || null;
  }

  async function moveFolder(folderId, parentId = null) {
    const result = await client.patchFolder(folderId, { parent_id: parentId || null });
    return result.folder || null;
  }

  async function setFolderStatus(folderId, status) {
    const result = await client.patchFolder(folderId, { status });
    return result.folder || null;
  }

  async function deleteFolder(folderId) {
    return client.deleteFolder(folderId);
  }

  async function moveSession(sessionId, folderId = null) {
    const result = await client.patchSession(sessionId, { folder_id: folderId || null });
    return result.session || null;
  }

  function ensureSelectionIsVisible() {
    if (selectedFolderId.value && !findFolder(folders.value, selectedFolderId.value)) {
      selectedFolderId.value = "";
    }
    if (selectedSessionId.value && !sessions.value.some((session) => session.id === selectedSessionId.value)) {
      selectSession("");
    }
  }

  return {
    folders,
    sessions,
    selectedFolderId,
    selectedSessionId,
    workspaceDocuments,
    contentsLoading,
    loading,
    error,
    selectedSession,
    selectedFolder,
    historyFolderOptions,
    refresh,
    selectFolder,
    selectSession,
    refreshContents,
    createFreeSession,
    createFolder,
    renameFolder,
    moveFolder,
    setFolderStatus,
    deleteFolder,
    moveSession
  };
}

function findFolder(folders, folderId) {
  for (const folder of folders) {
    if (folder.id === folderId) return folder;
    const nested = findFolder(folder.children || [], folderId);
    if (nested) return nested;
  }
  return null;
}

function flattenFolderOptions(folders, parents = []) {
  return folders.flatMap((folder) => {
    const path = [...parents, folder.name];
    return [
      { id: folder.id, name: folder.name, depth: parents.length, path: path.join(" / ") },
      ...flattenFolderOptions(folder.children || [], path)
    ];
  });
}
