import { computed, ref, watch } from "vue";

function defaultConfirmDelete(entry) {
  return globalThis.confirm(
    `删除「${entry.name}」？将移入系统回收站，可从回收站恢复。`
  );
}

export function useWorkspaceFiles({
  client,
  sessionId,
  workingDir,
  open,
  confirmDelete = defaultConfirmDelete
}) {
  const cwd = ref("");
  const entries = ref([]);
  const loading = ref(false);
  const error = ref("");
  const newName = ref("");
  const newKind = ref("file");
  const openPath = ref("");
  const content = ref("");
  const savingFile = ref(false);
  const hasWorkingDirectory = computed(() => Boolean(workingDir.value));
  let scopeGeneration = 0;
  let loadGeneration = 0;
  let openGeneration = 0;
  let createGeneration = 0;
  let saveGeneration = 0;

  function isCurrentScope(generation, requestSessionId, requestWorkingDir) {
    return (
      generation === scopeGeneration &&
      requestSessionId === sessionId.value &&
      requestWorkingDir === workingDir.value
    );
  }

  function relJoin(directory, name) {
    return directory ? `${directory}/${name}` : name;
  }

  async function load(directory = cwd.value) {
    if (!hasWorkingDirectory.value) return;
    const requestGeneration = ++loadGeneration;
    const requestScopeGeneration = scopeGeneration;
    const requestSessionId = sessionId.value;
    const requestWorkingDir = workingDir.value;
    const isCurrentRequest = () => (
      requestGeneration === loadGeneration &&
      isCurrentScope(requestScopeGeneration, requestSessionId, requestWorkingDir)
    );
    loading.value = true;
    error.value = "";
    try {
      const data = await client.listWorkspaceFiles(requestSessionId, { path: directory });
      if (!isCurrentRequest()) return;
      cwd.value = data.rel_path || "";
      entries.value = data.entries || [];
    } catch (err) {
      if (!isCurrentRequest()) return;
      error.value = err?.error?.details ? err.message : (err?.message || String(err));
    } finally {
      if (isCurrentRequest()) loading.value = false;
    }
  }

  function closeFile() {
    openGeneration += 1;
    openPath.value = "";
    content.value = "";
  }

  function enterDir(entry) {
    closeFile();
    load(entry.rel_path);
  }

  function goUp() {
    closeFile();
    const parent = cwd.value.includes("/")
      ? cwd.value.slice(0, cwd.value.lastIndexOf("/"))
      : "";
    load(parent);
  }

  async function createEntry() {
    const name = newName.value.trim();
    if (!name || !hasWorkingDirectory.value) return;
    const requestGeneration = ++createGeneration;
    const requestScopeGeneration = scopeGeneration;
    const requestSessionId = sessionId.value;
    const requestWorkingDir = workingDir.value;
    const requestPath = relJoin(cwd.value, name);
    const requestKind = newKind.value;
    const isCurrentRequest = () => (
      requestGeneration === createGeneration &&
      isCurrentScope(requestScopeGeneration, requestSessionId, requestWorkingDir)
    );
    error.value = "";
    try {
      await client.createWorkspaceEntry(requestSessionId, {
        path: requestPath,
        kind: requestKind
      });
      if (!isCurrentRequest()) return false;
      newName.value = "";
      await load();
      return true;
    } catch (err) {
      if (isCurrentRequest()) error.value = err?.message || String(err);
      return false;
    }
  }

  async function openFile(entry) {
    const requestGeneration = ++openGeneration;
    const requestScopeGeneration = scopeGeneration;
    const requestSessionId = sessionId.value;
    const requestWorkingDir = workingDir.value;
    const requestPath = entry.rel_path;
    const isCurrentRequest = () => (
      requestGeneration === openGeneration &&
      isCurrentScope(requestScopeGeneration, requestSessionId, requestWorkingDir)
    );
    error.value = "";
    try {
      const data = await client.readWorkspaceFile(requestSessionId, { path: requestPath });
      if (!isCurrentRequest()) return false;
      openPath.value = requestPath;
      content.value = data.content;
      return true;
    } catch (err) {
      if (isCurrentRequest()) error.value = err?.message || String(err);
      return false;
    }
  }

  async function saveFile() {
    if (!openPath.value || !hasWorkingDirectory.value) return;
    const requestGeneration = ++saveGeneration;
    const requestScopeGeneration = scopeGeneration;
    const requestSessionId = sessionId.value;
    const requestWorkingDir = workingDir.value;
    const requestPath = openPath.value;
    const requestContent = content.value;
    const isCurrentRequest = () => (
      requestGeneration === saveGeneration &&
      isCurrentScope(requestScopeGeneration, requestSessionId, requestWorkingDir)
    );
    savingFile.value = true;
    error.value = "";
    try {
      await client.writeWorkspaceFile(requestSessionId, {
        path: requestPath,
        content: requestContent
      });
      if (!isCurrentRequest()) return false;
      await load();
      return true;
    } catch (err) {
      if (isCurrentRequest()) error.value = err?.message || String(err);
      return false;
    } finally {
      if (isCurrentRequest()) savingFile.value = false;
    }
  }

  async function deleteEntry(entry) {
    if (!confirmDelete(entry)) return;
    const requestScopeGeneration = scopeGeneration;
    const requestSessionId = sessionId.value;
    const requestWorkingDir = workingDir.value;
    const requestPath = entry.rel_path;
    const isCurrentRequest = () => (
      isCurrentScope(requestScopeGeneration, requestSessionId, requestWorkingDir)
    );
    error.value = "";
    try {
      await client.deleteWorkspaceEntry(requestSessionId, { path: requestPath });
      if (!isCurrentRequest()) return false;
      if (openPath.value === requestPath) closeFile();
      await load();
      return true;
    } catch (err) {
      if (isCurrentRequest()) error.value = err?.message || String(err);
      return false;
    }
  }

  function resetForSession() {
    scopeGeneration += 1;
    loadGeneration += 1;
    createGeneration += 1;
    saveGeneration += 1;
    loading.value = false;
    savingFile.value = false;
    closeFile();
    cwd.value = "";
    entries.value = [];
    error.value = "";
  }

  watch(
    () => [sessionId.value, workingDir.value],
    () => {
      resetForSession();
      if (open.value && hasWorkingDirectory.value) load("");
    },
    { immediate: true }
  );

  watch(open, (isOpen) => {
    if (isOpen && hasWorkingDirectory.value) load(cwd.value);
  });

  return {
    cwd,
    entries,
    loading,
    error,
    newName,
    newKind,
    openPath,
    content,
    savingFile,
    hasWorkingDirectory,
    load,
    enterDir,
    goUp,
    createEntry,
    openFile,
    closeFile,
    saveFile,
    deleteEntry
  };
}
