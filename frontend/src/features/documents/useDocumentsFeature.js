import { computed, reactive, ref } from "vue";
import { useDocumentActions } from "./useDocumentActions";
import { useDocumentIngestJobs } from "./useDocumentIngestJobs";

function completedIngestNotice(job) {
  const processingStatus = job?.processing_status ?? job?.processing?.status;
  const needsVision = job?.needs_vision ?? job?.processing?.needs_vision;
  if (processingStatus !== "partial") return "文档已收录并完成检索覆盖";
  return needsVision
    ? "文档已收录，部分内容仍需视觉解析"
    : "文档已收录，仍有解析覆盖缺口";
}

/**
 * 分栏侧栏/详情布局的文档功能局部状态。
 *
 * 应用外壳提供共享消息和导航守卫；文档表单、选择、筛选与 CRUD 协调均保留在此处。
 */
export function useDocumentsFeature({
  api,
  loading,
  saving,
  notice,
  selectedSessionId = ref(""),
  onDocumentIngestSucceeded = async () => {},
  confirmDiscardChanges,
  clearMessage,
  showError
}) {
  const documentSearch = ref("");
  const documents = ref([]);
  const selectedDocumentId = ref("");
  const documentForm = reactive({ title: "", tags: "", summary: "" });
  const newDocument = reactive({ path: "", session_id: "" });

  const selectedDocument = computed(() =>
    documents.value.find((document) => document.id === selectedDocumentId.value) || null
  );
  const documentDirty = computed(() => Boolean(selectedDocument.value) && (
    String(documentForm.title || "").trim() !== String(selectedDocument.value?.title || "").trim() ||
    String(documentForm.tags || "").trim() !== String(selectedDocument.value?.tags || "").trim() ||
    String(documentForm.summary || "").trim() !== String(selectedDocument.value?.summary || "").trim()
  ));
  const visibleDocuments = computed(() => {
    const needle = documentSearch.value.trim().toLowerCase();
    if (!needle) return documents.value;
    return documents.value.filter((document) => [document.title, document.summary, document.tags, document.path].some((value) =>
      String(value || "").toLowerCase().includes(needle)
    ));
  });
  const canChooseDocumentPath = computed(() => Boolean(globalThis.personagraphDesktop?.chooseDocumentPath));

  let actions;
  const ingestJobs = useDocumentIngestJobs({
    api,
    selectedSessionId,
    onError: showError,
    onSucceeded: async (job, sessionId) => {
      await actions.loadDocuments(job.document_id || undefined);
      notice.value = completedIngestNotice(job);
      await onDocumentIngestSucceeded(job, sessionId);
    }
  });

  actions = useDocumentActions({
    api,
    state: {
      loading,
      saving,
      notice,
      documents,
      selectedDocumentId,
      documentSearch,
      documentForm,
      newDocument
    },
    selectedDocument,
    documentDirty,
    selectedSessionId,
    acceptDocumentIngestJob: ingestJobs.accept,
    confirmDiscardChanges,
    clearMessage,
    showError
  });

  function discardChanges() {
    actions.hydrateDocumentForm(selectedDocument.value);
  }

  return {
    documentSearch,
    documents,
    selectedDocumentId,
    documentForm,
    newDocument,
    selectedDocument,
    documentDirty,
    visibleDocuments,
    canChooseDocumentPath,
    documentIngestJobs: ingestJobs.jobs,
    documentIngestJobsLoading: ingestJobs.loading,
    documentIngestJobsSupported: ingestJobs.supported,
    startDocumentIngestJobs: ingestJobs.start,
    stopDocumentIngestJobs: ingestJobs.stop,
    reconcileDocumentIngestJobs: ingestJobs.reconcile,
    retryDocumentIngestJob: ingestJobs.retry,
    discardChanges,
    ...actions
  };
}
