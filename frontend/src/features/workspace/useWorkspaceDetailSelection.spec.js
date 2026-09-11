import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";
import { useWorkspaceDetailSelection } from "./useWorkspaceDetailSelection";

function deferred() {
  let resolve;
  const promise = new Promise((onResolve) => { resolve = onResolve; });
  return { promise, resolve };
}

describe("useWorkspaceDetailSelection", () => {
  it("does not expose the old document when its selection guard rejects", async () => {
    const workspaceDetailKind = ref(null);
    const guardedSelectDocument = vi.fn(() => false);
    const selection = useWorkspaceDetailSelection({
      workspaceDetailKind,
      guardedSelectDocument
    });

    await expect(selection.openWorkspaceDocumentDetail("doc-new")).resolves.toBe(false);

    expect(guardedSelectDocument).toHaveBeenCalledWith("doc-new");
    expect(workspaceDetailKind.value).toBeNull();
  });

  it("keeps document detail closed while guarded async selection is pending", async () => {
    const workspaceDetailKind = ref(null);
    const pending = deferred();
    const selection = useWorkspaceDetailSelection({
      workspaceDetailKind,
      guardedSelectDocument: vi.fn(() => pending.promise)
    });

    const opening = selection.openWorkspaceDocumentDetail("doc-new");

    expect(workspaceDetailKind.value).toBeNull();
    pending.resolve(true);
    await expect(opening).resolves.toBe(true);
    expect(workspaceDetailKind.value).toBe("document");
  });

  it("does not let an older guarded selection replace a newer detail", async () => {
    const workspaceDetailKind = ref(null);
    const olderDocument = deferred();
    const currentDocument = deferred();
    const selection = useWorkspaceDetailSelection({
      workspaceDetailKind,
      guardedSelectDocument: vi.fn((documentId) => (
        documentId === "doc-old" ? olderDocument.promise : currentDocument.promise
      ))
    });

    const olderOpening = selection.openWorkspaceDocumentDetail("doc-old");
    const documentOpening = selection.openWorkspaceDocumentDetail("doc-current");
    currentDocument.resolve(true);
    await expect(documentOpening).resolves.toBe(true);
    olderDocument.resolve(true);
    await expect(olderOpening).resolves.toBe(false);

    expect(workspaceDetailKind.value).toBe("document");
  });

  it("invalidates a pending document detail when its owning Session changes", async () => {
    const workspaceDetailKind = ref(null);
    const selectedSessionId = ref("session-a");
    const pendingDocument = deferred();
    const selection = useWorkspaceDetailSelection({
      workspaceDetailKind,
      selectedSessionId,
      guardedSelectDocument: vi.fn(() => pendingDocument.promise)
    });

    const opening = selection.openWorkspaceDocumentDetail("doc-a");
    selectedSessionId.value = "session-b";
    expect(workspaceDetailKind.value).toBeNull();

    pendingDocument.resolve(true);
    await expect(opening).resolves.toBe(false);
    expect(workspaceDetailKind.value).toBeNull();
  });
});
