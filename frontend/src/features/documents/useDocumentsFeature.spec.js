import { describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import { useDocumentsFeature } from "./useDocumentsFeature";

describe("useDocumentsFeature", () => {
  it("keeps document selection, filtering, dirty state, and reset in one local feature", async () => {
    const api = {
      listDocuments: vi.fn().mockResolvedValue({
        documents: [
          { id: "doc-a", title: "实验记录", tags: "实验", summary: "第一轮结果", path: "/tmp/a.md" },
          { id: "doc-b", title: "会议笔记", tags: "会议", summary: "", path: "/tmp/b.md" }
        ]
      })
    };
    const feature = useDocumentsFeature({
      api,
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      selectedSessionId: ref("session-a"),
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn()
    });

    expect(feature.newDocument.with_summary).toBeUndefined();

    await feature.loadDocuments();
    expect(feature.selectedDocument.value?.id).toBe("doc-a");

    feature.documentSearch.value = "会议";
    expect(feature.visibleDocuments.value.map((document) => document.id)).toEqual(["doc-b"]);

    feature.documentForm.title = "实验记录（修订）";
    expect(feature.documentDirty.value).toBe(true);
    feature.discardChanges();
    expect(feature.documentDirty.value).toBe(false);
  });

  it("owns durable intake tracking and refreshes document views once on success", async () => {
    const succeeded = {
      job_id: "job-a",
      session_id: "session-a",
      path: "paper.pdf",
      status: "succeeded",
      stage: "active",
      document_id: "doc-a",
      processing_status: "complete",
      can_retry: false
    };
    const api = {
      listDocumentIngestJobs: vi.fn().mockResolvedValue({ jobs: [succeeded] }),
      listDocuments: vi.fn().mockResolvedValue({ documents: [{ id: "doc-a" }] })
    };
    const notice = ref("");
    const onDocumentIngestSucceeded = vi.fn().mockResolvedValue(undefined);
    const feature = useDocumentsFeature({
      api,
      loading: ref(false),
      saving: ref(false),
      notice,
      selectedSessionId: ref("session-a"),
      onDocumentIngestSucceeded,
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn()
    });

    feature.startDocumentIngestJobs();
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    await nextTick();

    expect(feature.documentIngestJobs.value).toEqual([succeeded]);
    expect(api.listDocuments).toHaveBeenCalledOnce();
    expect(onDocumentIngestSucceeded).toHaveBeenCalledOnce();
    expect(notice.value).toBe("文档已收录并完成检索覆盖");
    feature.stopDocumentIngestJobs();
  });
});
