import { ref, watch } from "vue";

function emptyContext(sessionId = "") {
  return {
    session_id: sessionId,
    user_state: [],
    task_state: [],
    interaction_state: [],
    counts: { user_state: 0, task_state: 0, interaction_state: 0, total: 0 },
    latest_reset: null
  };
}

function browserDownload(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function fallbackRequestId() {
  return `clear-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function useSessionContextFeature({
  api,
  selectedSessionId,
  notice,
  download = browserDownload,
  makeRequestId = () => globalThis.crypto?.randomUUID?.() || fallbackRequestId()
}) {
  const sessionContext = ref(emptyContext());
  const contextLoading = ref(false);
  const contextBusy = ref(false);
  const contextError = ref("");
  const selectedState = ref(null);
  const stateExplanation = ref(null);
  const repairPreview = ref(null);
  const repairApplyEnabled = ref(false);
  const repairPreviewStale = ref(false);
  let loadGeneration = 0;
  let pendingClear = null;

  function resetView(sessionId = "") {
    sessionContext.value = emptyContext(sessionId);
    selectedState.value = null;
    stateExplanation.value = null;
    repairPreview.value = null;
    repairApplyEnabled.value = false;
    repairPreviewStale.value = false;
    contextError.value = "";
    pendingClear = null;
  }

  function captureError(error) {
    contextError.value = error?.message || String(error);
    repairPreviewStale.value = (
      error?.error?.code || error?.payload?.error?.code
    ) === "SESSION_CONTEXT_REPAIR_PREVIEW_STALE";
  }

  async function loadSessionContext() {
    const sessionId = selectedSessionId.value;
    const generation = ++loadGeneration;
    if (!sessionId) {
      resetView();
      return;
    }
    contextLoading.value = true;
    contextError.value = "";
    try {
      const contextPayload = await api.getSessionContext(sessionId);
      if (generation !== loadGeneration || selectedSessionId.value !== sessionId) return;
      sessionContext.value = contextPayload.session_context || emptyContext(sessionId);
      repairPreview.value = null;
      repairApplyEnabled.value = false;
      repairPreviewStale.value = false;
      if (selectedState.value) {
        const stillExists = [
          ...(sessionContext.value.user_state || []),
          ...(sessionContext.value.task_state || []),
          ...(sessionContext.value.interaction_state || [])
        ].some((item) => item.id === selectedState.value.id);
        if (!stillExists) {
          selectedState.value = null;
          stateExplanation.value = null;
        }
      }
    } catch (error) {
      if (generation === loadGeneration) captureError(error);
    } finally {
      if (generation === loadGeneration) contextLoading.value = false;
    }
  }

  async function explainState(state, { includeExcerpt = false } = {}) {
    const sessionId = selectedSessionId.value;
    if (!sessionId || !state) return;
    contextBusy.value = true;
    contextError.value = "";
    try {
      const payload = await api.explainSessionContext(sessionId, {
        domain: state.domain,
        state_type: state.state_type,
        key: state.key,
        include_evidence_excerpt: includeExcerpt
      });
      if (selectedSessionId.value !== sessionId) return;
      selectedState.value = state;
      stateExplanation.value = payload;
    } catch (error) {
      captureError(error);
    } finally {
      contextBusy.value = false;
    }
  }

  async function exportSessionContext({ includeExcerpt = false } = {}) {
    const sessionId = selectedSessionId.value;
    if (!sessionId) return;
    contextBusy.value = true;
    contextError.value = "";
    try {
      const payload = await api.exportSessionContext(sessionId, {
        include_evidence_excerpt: includeExcerpt
      });
      download(
        `session-context-${sessionId}.json`,
        payload.session_context_export || payload
      );
      notice.value = "会话状态已导出";
    } catch (error) {
      captureError(error);
    } finally {
      contextBusy.value = false;
    }
  }

  async function clearSessionContext(reason) {
    const sessionId = selectedSessionId.value;
    const normalizedReason = String(reason || "").trim();
    if (!sessionId || !normalizedReason) return false;
    if (!pendingClear || pendingClear.sessionId !== sessionId || pendingClear.reason !== normalizedReason) {
      pendingClear = { sessionId, reason: normalizedReason, requestId: makeRequestId() };
    }
    contextBusy.value = true;
    contextError.value = "";
    try {
      await api.clearSessionContext(sessionId, {
        confirm: true,
        request_id: pendingClear.requestId,
        reason: normalizedReason
      });
      pendingClear = null;
      await loadSessionContext();
      notice.value = "会话状态已清空；聊天记录仍保留";
      return true;
    } catch (error) {
      captureError(error);
      return false;
    } finally {
      contextBusy.value = false;
    }
  }

  function slotPayload(state) {
    if (!state) return {};
    return { domain: state.domain, state_type: state.state_type, key: state.key };
  }

  async function createCorrection({ state, operation, value }) {
    const sessionId = selectedSessionId.value;
    if (!sessionId || !state) return false;
    contextBusy.value = true;
    contextError.value = "";
    repairPreviewStale.value = false;
    try {
      const payload = await api.createSessionContextCorrection(sessionId, {
        confirm: true,
        ...slotPayload(state),
        operation,
        ...(["set", "append"].includes(operation) ? { value } : {})
      });
      if (selectedSessionId.value !== sessionId) return false;
      repairPreview.value = payload.repair || null;
      repairApplyEnabled.value = payload.apply_enabled === true;
      notice.value = "纠正证据已记录；当前状态尚未替换";
      return true;
    } catch (error) {
      captureError(error);
      return false;
    } finally {
      contextBusy.value = false;
    }
  }

  async function previewRepair(state = null) {
    const sessionId = selectedSessionId.value;
    if (!sessionId) return false;
    contextBusy.value = true;
    contextError.value = "";
    repairPreviewStale.value = false;
    try {
      const payload = await api.previewSessionContextRepair(
        sessionId,
        slotPayload(state)
      );
      if (selectedSessionId.value !== sessionId) return false;
      repairPreview.value = payload.repair || null;
      repairApplyEnabled.value = payload.apply_enabled === true;
      notice.value = "Repair dry-run 已更新";
      return true;
    } catch (error) {
      captureError(error);
      return false;
    } finally {
      contextBusy.value = false;
    }
  }

  async function applyRepair() {
    const sessionId = selectedSessionId.value;
    const preview = repairPreview.value;
    if (!sessionId || !preview?.preview_token || !repairApplyEnabled.value) return false;
    contextBusy.value = true;
    contextError.value = "";
    repairPreviewStale.value = false;
    try {
      await api.applySessionContextRepair(sessionId, {
        confirm: true,
        preview_token: preview.preview_token,
        ...(preview.slot || {})
      });
      repairPreview.value = null;
      repairApplyEnabled.value = false;
      await loadSessionContext();
      notice.value = "Repair 已按预览原子应用";
      return true;
    } catch (error) {
      captureError(error);
      return false;
    } finally {
      contextBusy.value = false;
    }
  }

  watch(selectedSessionId, (sessionId, previousId) => {
    if (sessionId !== previousId) resetView(sessionId || "");
    loadSessionContext();
  }, { immediate: true });

  return {
    sessionContext,
    contextLoading,
    contextBusy,
    contextError,
    selectedState,
    stateExplanation,
    repairPreview,
    repairApplyEnabled,
    repairPreviewStale,
    loadSessionContext,
    explainState,
    exportSessionContext,
    clearSessionContext,
    createCorrection,
    previewRepair,
    applyRepair
  };
}
