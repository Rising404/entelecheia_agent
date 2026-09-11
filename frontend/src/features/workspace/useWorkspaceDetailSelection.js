import { watch } from "vue";

/** 仅在所属功能确认选择后打开工作区详情。 */
export function useWorkspaceDetailSelection({
  workspaceDetailKind,
  selectedSessionId = null,
  guardedSelectDocument
}) {
  let selectionGeneration = 0;

  function invalidateWorkspaceDetailSelection() {
    selectionGeneration += 1;
    workspaceDetailKind.value = null;
  }

  if (selectedSessionId) {
    watch(selectedSessionId, invalidateWorkspaceDetailSelection, { flush: "sync" });
  }

  async function openGuardedDetail(kind, itemId, guardedSelect) {
    const requestGeneration = ++selectionGeneration;
    const owningSessionId = selectedSessionId?.value;
    const selected = await guardedSelect(itemId);
    if (
      !selected ||
      requestGeneration !== selectionGeneration ||
      (selectedSessionId && selectedSessionId.value !== owningSessionId)
    ) return false;
    workspaceDetailKind.value = kind;
    return true;
  }

  const openWorkspaceDocumentDetail = (documentId) => (
    openGuardedDetail("document", documentId, guardedSelectDocument)
  );

  return {
    openWorkspaceDocumentDetail,
    invalidateWorkspaceDetailSelection
  };
}
