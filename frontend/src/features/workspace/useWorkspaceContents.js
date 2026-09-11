import { ref } from "vue";

/**
 * 工作区外壳使用的会话级文档投影。
 *
 * 本模块只拥有活动会话旁展示的聚合视图。历史导航仍拥有选择权；选择变化时，调用方
 * 必须重置该投影，避免旧请求把结果发布到新选择中。
 */
export function useWorkspaceContents({ client, selectedSessionId } = {}) {
  if (!client) throw new Error("useWorkspaceContents requires an injected workspace client.");
  if (!selectedSessionId) throw new Error("useWorkspaceContents requires a selected session ref.");

  const workspaceDocuments = ref([]);
  const contentsLoading = ref(false);
  let contentsGeneration = 0;

  function resetForSession() {
    contentsGeneration += 1;
    workspaceDocuments.value = [];
    contentsLoading.value = false;
  }

  async function refreshContents(sessionId = selectedSessionId.value) {
    // 只有严格匹配的当前 Session 才能向这份共享投影发布数据。
    if (sessionId !== selectedSessionId.value) return;
    const requestGeneration = ++contentsGeneration;
    if (!sessionId) {
      workspaceDocuments.value = [];
      contentsLoading.value = false;
      return;
    }
    contentsLoading.value = true;
    try {
      const documentResult = await client.listDocuments({ session_id: sessionId });
      if (
        requestGeneration !== contentsGeneration ||
        selectedSessionId.value !== sessionId
      ) return;
      workspaceDocuments.value = documentResult.documents || [];
    } finally {
      // 已被取代的请求不能让当前请求看起来已经完成。
      if (
        requestGeneration === contentsGeneration &&
        selectedSessionId.value === sessionId
      ) contentsLoading.value = false;
    }
  }

  return {
    workspaceDocuments,
    contentsLoading,
    resetForSession,
    refreshContents
  };
}
