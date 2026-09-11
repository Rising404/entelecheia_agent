import { reactive, ref } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useDocumentActions } from "./useDocumentActions";

afterEach(() => vi.unstubAllGlobals());

function harness(api, document = null, {
  confirmDiscardChanges = () => true,
  acceptDocumentIngestJob = vi.fn(() => true),
  selectedSessionId = ref("session-a"),
  makeJobId = vi.fn(() => "durable-job-1")
} = {}) {
  const state = {
    loading: ref(false), saving: ref(false), notice: ref(""), documents: ref(document ? [document] : []),
    selectedDocumentId: ref(document?.id || ""), documentSearch: ref(""),
    documentForm: reactive({ title: "", tags: "", summary: "" }),
    newDocument: reactive({ path: "", session_id: "", with_summary: true })
  };
  const showError = vi.fn();
  return {
    state,
    showError,
    actions: useDocumentActions({
      api,
      state,
      selectedDocument: ref(document),
      documentDirty: ref(false),
      selectedSessionId,
      acceptDocumentIngestJob,
      makeJobId,
      confirmDiscardChanges,
      clearMessage: vi.fn(),
      showError
    }),
    selectedSessionId,
    acceptDocumentIngestJob,
    makeJobId
  };
}

describe("useDocumentActions", () => {
  it("reports a rejected guarded selection without changing the current document", () => {
    const current = { id: "d1", title: "旧文档" };
    const { actions, state } = harness({}, current, {
      confirmDiscardChanges: () => false
    });

    expect(actions.guardedSelectDocument("d2")).toBe(false);
    expect(state.selectedDocumentId.value).toBe(current.id);
  });

  it("does nothing when the desktop path picker is unavailable", async () => {
    const { actions, state } = harness({});
    await actions.chooseDocumentPath();
    expect(state.newDocument.path).toBe("");
  });

  it("does not delete the index when confirmation is rejected", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false));
    const api = { deleteDocument: vi.fn() };
    const { actions } = harness(api, { id: "d1", title: "Doc" });
    await actions.deleteDocument();
    expect(api.deleteDocument).not.toHaveBeenCalled();
  });

  it("loads, edits, and deletes documents only through the selected session", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const document = { id: "d1", title: "Doc", tags: "", summary: "before" };
    const api = {
      listDocuments: vi.fn().mockResolvedValue({ documents: [document] }),
      patchDocument: vi.fn().mockResolvedValue({ document }),
      deleteDocument: vi.fn().mockResolvedValue({ deleted: true })
    };
    const selectedDocument = ref(document);
    const state = {
      loading: ref(false), saving: ref(false), notice: ref(""), documents: ref([document]),
      selectedDocumentId: ref("d1"), documentSearch: ref(""),
      documentForm: reactive({ title: "Renamed", tags: "tag", summary: "after" }),
      newDocument: reactive({ path: "", session_id: "", with_summary: true })
    };
    const selectedSessionId = ref("session-a");
    const actions = useDocumentActions({
      api,
      state,
      selectedDocument,
      documentDirty: ref(true),
      selectedSessionId,
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn()
    });

    await actions.loadDocuments();
    expect(api.listDocuments).toHaveBeenCalledWith({ session_id: "session-a" });

    state.documentForm.title = "Renamed";
    state.documentForm.tags = "tag";
    state.documentForm.summary = "after";
    await actions.saveDocument();
    expect(api.patchDocument).toHaveBeenCalledWith("d1", {
      session_id: "session-a",
      title: "Renamed",
      tags: "tag",
      summary: "after"
    });

    await actions.deleteDocument();
    expect(api.deleteDocument).toHaveBeenCalledWith("d1", {
      session_id: "session-a"
    });
  });

  it("does not list documents without a selected session", async () => {
    const api = { listDocuments: vi.fn() };
    const { actions, state } = harness(api, null, {
      selectedSessionId: ref("")
    });

    await actions.loadDocuments();

    expect(api.listDocuments).not.toHaveBeenCalled();
    expect(state.documents.value).toEqual([]);
  });

  it("does not submit document intake without an exact session", async () => {
    const api = { createDocumentIngestJob: vi.fn() };
    const { actions, state, showError } = harness(api);
    state.newDocument.path = "/tmp/paper.pdf";

    await actions.createDocument();

    expect(api.createDocumentIngestJob).not.toHaveBeenCalled();
    expect(showError).toHaveBeenCalledWith(expect.objectContaining({
      message: "请先选择一个当前会话，再收录文档。"
    }));
  });

  it("clears the intake form only after a durable enqueue acknowledgement", async () => {
    const acceptedJob = {
      job_id: "durable-job-1",
      session_id: "session-a",
      status: "queued",
      stage: "parsing"
    };
    const api = {
      createDocumentIngestJob: vi.fn().mockResolvedValue({
        accepted: true,
        replayed: false,
        job: acceptedJob
      }),
      listDocuments: vi.fn()
    };
    const { actions, state, acceptDocumentIngestJob } = harness(api);
    state.newDocument.path = " paper.pdf ";
    state.newDocument.session_id = " session-a ";

    await expect(actions.createDocument()).resolves.toBe(true);

    expect(api.createDocumentIngestJob).toHaveBeenCalledWith({
      job_id: "durable-job-1",
      session_id: "session-a",
      path: "paper.pdf",
      with_summary: false
    });
    expect(acceptDocumentIngestJob).toHaveBeenCalledWith(acceptedJob, "session-a");
    expect(api.listDocuments).not.toHaveBeenCalled();
    expect(state.newDocument.path).toBe("");
    expect(state.newDocument.session_id).toBe("");
    expect(state.notice.value).toBe("文档收录任务已进入队列");
  });

  it("fails closed when the durable endpoint is unavailable", async () => {
    const api = {
      createDocumentIngestJob: vi.fn().mockRejectedValue({
        status: 404,
        message: "not found"
      })
    };
    const { actions, state, showError } = harness(api);
    state.newDocument.path = "paper.pdf";
    state.newDocument.session_id = "session-a";

    await expect(actions.createDocument()).resolves.toBe(false);

    expect(showError).toHaveBeenCalledWith(expect.objectContaining({ status: 404 }));
    expect(state.newDocument.path).toBe("paper.pdf");
  });

  it("retains the durable idempotency key after a timeout or 5xx", async () => {
    const unavailable = { status: 503, message: "unavailable" };
    const api = {
      createDocumentIngestJob: vi.fn()
        .mockRejectedValueOnce(unavailable)
        .mockResolvedValueOnce({
          accepted: true,
          job: {
            job_id: "durable-job-1",
            session_id: "session-a",
            status: "queued",
            stage: "parsing"
          }
        })
    };
    const { actions, state, makeJobId } = harness(api);
    state.newDocument.path = "paper.pdf";
    state.newDocument.session_id = "session-a";

    await expect(actions.createDocument()).resolves.toBe(false);
    expect(state.newDocument.path).toBe("paper.pdf");
    await expect(actions.createDocument()).resolves.toBe(true);
    expect(api.createDocumentIngestJob.mock.calls.map(([payload]) => payload.job_id))
      .toEqual(["durable-job-1", "durable-job-1"]);
    expect(makeJobId).toHaveBeenCalledOnce();
  });

  it("does not clear or acknowledge a malformed durable response", async () => {
    const api = {
      createDocumentIngestJob: vi.fn().mockResolvedValue({ accepted: false })
    };
    const { actions, state, showError } = harness(api);
    state.newDocument.path = "paper.pdf";
    state.newDocument.session_id = "session-a";

    await expect(actions.createDocument()).resolves.toBe(false);

    expect(state.newDocument.path).toBe("paper.pdf");
    expect(showError).toHaveBeenCalled();
  });

  it("does not let an A acknowledgement clear a new A form after an A-B-A switch", async () => {
    let resolveEnqueue;
    const enqueue = new Promise((resolve) => { resolveEnqueue = resolve; });
    const api = { createDocumentIngestJob: vi.fn().mockReturnValue(enqueue) };
    const { actions, state, selectedSessionId, acceptDocumentIngestJob } = harness(api);
    state.newDocument.path = "paper.pdf";
    state.newDocument.session_id = "session-a";

    const staleSubmit = actions.createDocument();
    selectedSessionId.value = "session-b";
    selectedSessionId.value = "session-a";
    state.newDocument.path = "paper.pdf";
    state.newDocument.session_id = "session-a";
    resolveEnqueue({
      accepted: true,
      job: {
        job_id: "durable-job-1",
        session_id: "session-a",
        status: "queued",
        stage: "parsing"
      }
    });

    await expect(staleSubmit).resolves.toBe(false);
    expect(state.newDocument.path).toBe("paper.pdf");
    expect(acceptDocumentIngestJob).not.toHaveBeenCalled();

    await expect(actions.createDocument()).resolves.toBe(true);
    expect(api.createDocumentIngestJob.mock.calls.map(([payload]) => payload.job_id))
      .toEqual(["durable-job-1", "durable-job-1"]);
    expect(state.newDocument.path).toBe("");
  });
});
