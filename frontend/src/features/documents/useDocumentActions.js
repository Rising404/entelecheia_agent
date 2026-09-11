import { watch } from "vue";

function newDocumentIngestJobId() {
  const uuid = globalThis.crypto?.randomUUID?.();
  return uuid
    ? `document-ingest-${uuid}`
    : `document-ingest-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function useDocumentActions({
  api,
  state,
  selectedDocument,
  documentDirty,
  selectedSessionId = null,
  acceptDocumentIngestJob = () => true,
  makeJobId = newDocumentIngestJobId,
  confirmDiscardChanges,
  clearMessage,
  showError
}) {
  const { loading, saving, notice, documents, selectedDocumentId, documentSearch, documentForm, newDocument } = state;
  let intakeGeneration = 0;
  let latestIntakeOperation = 0;
  const pendingIngestRequests = new Map();

  if (selectedSessionId) {
    watch(selectedSessionId, () => {
      intakeGeneration += 1;
    }, { flush: "sync" });
  }

  function hydrateDocumentForm(document) {
    documentForm.title = document?.title || "";
    documentForm.tags = document?.tags || "";
    documentForm.summary = document?.summary || "";
  }

  function selectDocument(documentId) {
    selectedDocumentId.value = documentId;
    hydrateDocumentForm(selectedDocument.value);
    return true;
  }

  async function loadDocuments(preferredId = selectedDocumentId.value) {
    loading.value = true;
    clearMessage();
    try {
      const sessionId = selectedSessionId?.value || "";
      if (!sessionId) {
        documents.value = [];
        selectedDocumentId.value = "";
        hydrateDocumentForm(null);
        return;
      }
      const payload = await api.listDocuments({ session_id: sessionId });
      documents.value = payload.documents || [];
      const nextId = documents.value.some((document) => document.id === preferredId) ? preferredId : documents.value[0]?.id || "";
      if (nextId) selectDocument(nextId);
      else {
        selectedDocumentId.value = "";
        hydrateDocumentForm(null);
      }
    } catch (err) { showError(err); } finally { loading.value = false; }
  }

  function guardedSelectDocument(documentId) {
    if (selectedDocumentId.value === documentId) return true;
    if (!confirmDiscardChanges()) return false;
    return selectDocument(documentId);
  }

  const clearDocumentSearch = () => { documentSearch.value = ""; };

  async function chooseDocumentPath() {
    const picker = globalThis.personagraphDesktop?.chooseDocumentPath;
    if (!picker) return;
    clearMessage();
    try {
      const result = await picker();
      if (result?.path) newDocument.path = result.path;
    } catch (err) { showError(err); }
  }

  async function createDocument() {
    if (!newDocument.path.trim()) return;
    const path = newDocument.path.trim();
    const sessionId = newDocument.session_id.trim();
    if (!sessionId) {
      showError(new Error("请先选择一个当前会话，再收录文档。"));
      return false;
    }
    if (selectedSessionId?.value && selectedSessionId.value !== sessionId) {
      showError(new Error("当前会话已变更，请重新打开文档收录表单。"));
      return false;
    }
    // D0 intake 只持久化延后生成摘要的意图，本身不会生成摘要。
    const withSummary = false;
    const requestSignature = JSON.stringify({ path, sessionId, withSummary });
    if (!pendingIngestRequests.has(requestSignature)) {
      pendingIngestRequests.set(requestSignature, makeJobId());
    }
    const jobId = pendingIngestRequests.get(requestSignature);
    const submissionGeneration = intakeGeneration;
    const operation = ++latestIntakeOperation;
    const stillOwnsForm = () => (
      submissionGeneration === intakeGeneration &&
      (!selectedSessionId || selectedSessionId.value === sessionId) &&
      newDocument.path.trim() === path &&
      newDocument.session_id.trim() === sessionId
    );

    saving.value = true;
    clearMessage();
    try {
      const payload = await api.createDocumentIngestJob({
        job_id: jobId,
        session_id: sessionId,
        path,
        with_summary: withSummary
      });

      const acceptedJob = payload?.job;
      if (
        payload?.accepted !== true ||
        !acceptedJob ||
        acceptedJob.job_id !== jobId ||
        acceptedJob.session_id !== sessionId
      ) {
        throw new Error("文档收录任务未返回有效的持久确认，请使用同一表单重试。");
      }
      if (!stillOwnsForm()) return false;
      if (!acceptDocumentIngestJob(acceptedJob, sessionId)) return false;

      pendingIngestRequests.delete(requestSignature);
      newDocument.path = "";
      newDocument.session_id = "";
      documentSearch.value = "";
      notice.value = payload.replayed
        ? "文档收录任务已在队列中"
        : "文档收录任务已进入队列";
      return true;
    } catch (err) {
      if (
        submissionGeneration === intakeGeneration &&
        (!selectedSessionId || selectedSessionId.value === sessionId)
      ) showError(err);
      return false;
    } finally {
      if (operation === latestIntakeOperation) saving.value = false;
    }
  }

  async function saveDocument() {
    if (!selectedDocument.value || !documentDirty.value) return;
    saving.value = true;
    clearMessage();
    try {
      await api.patchDocument(selectedDocument.value.id, {
        session_id: selectedSessionId?.value || "",
        title: documentForm.title.trim(), tags: documentForm.tags.trim(), summary: documentForm.summary.trim()
      });
      await loadDocuments(selectedDocument.value.id);
      notice.value = "文档信息已保存";
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  async function deleteDocument() {
    if (!selectedDocument.value) return;
    if (!globalThis.confirm(`从工作集删除「${selectedDocument.value.title || selectedDocument.value.id}」？原文件不会被删除。`)) return;
    saving.value = true;
    clearMessage();
    try {
      await api.deleteDocument(selectedDocument.value.id, {
        session_id: selectedSessionId?.value || ""
      });
      await loadDocuments();
      notice.value = "文档索引已删除";
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  return { loadDocuments, selectDocument, guardedSelectDocument, clearDocumentSearch, hydrateDocumentForm, chooseDocumentPath, createDocument, saveDocument, deleteDocument };
}
