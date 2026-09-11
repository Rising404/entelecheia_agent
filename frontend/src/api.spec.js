import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError } from "./api";


function sseResponse(chunks) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    }
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream; charset=utf-8" }
  });
}


afterEach(() => vi.unstubAllGlobals());

describe("retired feature boundary", () => {
  it("does not expose clients for retired global memory, tasks, reminders, or promotions", () => {
    const retiredMethods = [
      "listSessionPromotions",
      "approveSessionPromotion",
      "rejectSessionPromotion",
      "commitSessionPromotion",
      "revokeSessionPromotion",
      "listMemories",
      "pendingMemories",
      "patchMemory",
      "confirmMemory",
      "rejectMemory",
      "archiveMemory",
      "forgetMemory",
      "restoreMemory",
      "listTasks",
      "getTask",
      "createTask",
      "patchTask",
      "completeTask",
      "reopenTask",
      "deleteTask",
      "taskTrail",
      "listReminders",
      "createReminder",
      "patchReminder",
      "cancelReminder"
    ];

    for (const method of retiredMethods) expect(api).not.toHaveProperty(method);
    expect(api.getInSessionTaskDetails).toBeTypeOf("function");
  });
});

describe("getInSessionTaskDetails", () => {
  it("uses the session-scoped read-only task endpoint", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ task: {} }), { status: 200 }));
    vi.stubGlobal("fetch", fetch);

    await api.getInSessionTaskDetails("session/a", "task b");

    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/api/sessions/session%2Fa/insession-tasks/task%20b",
      expect.objectContaining({ headers: expect.objectContaining({ "Content-Type": "application/json" }) })
    );
  });
});

describe("post-commit failure control", () => {
  it("posts the exact explicit command to its Session scope", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{"replayed":false}', { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const command = {
      turn_id: "turn-a", request_id: "request-a", action: "waive", confirm_stale: true,
      expected_window_revision: 5, expected_failed_job_digest: "a".repeat(64), job_ids: ["job-a"]
    };
    await api.controlTurnPostCommitJobs("session/a", command);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/api/sessions/session%2Fa/post-commit/control",
      expect.objectContaining({ method: "POST", body: JSON.stringify(command) })
    );
  });
});

describe("document ingest jobs", () => {
  it("uses session-scoped durable enqueue, list, detail, and retry endpoints", async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ accepted: true, jobs: [], job: {} }), { status: 202 })
    ));
    vi.stubGlobal("fetch", fetch);

    await api.createDocumentIngestJob({
      job_id: "job/a",
      session_id: "session-a",
      path: "paper.pdf",
      with_summary: true
    });
    await api.listDocumentIngestJobs({ session_id: "session/a", status: "running", limit: 20 });
    await api.getDocumentIngestJob("job/a", { session_id: "session/a" });
    await api.retryDocumentIngestJob("job/a", { session_id: "session/a" });

    expect(fetch.mock.calls.map(([url, options]) => [url, options.method, options.body])).toEqual([
      ["http://127.0.0.1:8765/api/document-ingest-jobs", "POST", JSON.stringify({
        job_id: "job/a",
        session_id: "session-a",
        path: "paper.pdf",
        with_summary: true
      })],
      ["http://127.0.0.1:8765/api/document-ingest-jobs?session_id=session%2Fa&status=running&limit=20", undefined, undefined],
      ["http://127.0.0.1:8765/api/document-ingest-jobs/job%2Fa?session_id=session%2Fa", undefined, undefined],
      ["http://127.0.0.1:8765/api/document-ingest-jobs/job%2Fa/retry", "POST", JSON.stringify({ session_id: "session/a" })]
    ]);
  });
});

describe("project document routes", () => {
  it("carries the session locator on list, detail, patch, and delete", async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ documents: [], document: {}, deleted: true }), {
        status: 200
      })
    ));
    vi.stubGlobal("fetch", fetch);

    await api.listDocuments({ session_id: "session/a" });
    await api.patchDocument("doc/a", {
      session_id: "session/a",
      title: "Renamed"
    });
    await api.deleteDocument("doc/a", { session_id: "session/a" });

    expect(fetch.mock.calls.map(([url, options]) => [url, options.method, options.body]))
      .toEqual([
        ["http://127.0.0.1:8765/api/documents?session_id=session%2Fa", undefined, undefined],
        ["http://127.0.0.1:8765/api/documents/doc%2Fa", "PATCH", JSON.stringify({
          session_id: "session/a",
          title: "Renamed"
        })],
        ["http://127.0.0.1:8765/api/documents/doc%2Fa?session_id=session%2Fa", "DELETE", undefined]
      ]);
  });
});

describe("chatTurnStream", () => {
  it("keeps an accepted-but-unfinished stream retryable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      'event: accepted\ndata: {"turn_id":"turn-1","client_request_id":"request-1"}\n\n'
    ])));
    const events = [];

    await expect(api.chatTurnStream({ message: "继续", client_request_id: "request-1" }, (item) => {
      events.push(item);
    })).rejects.toMatchObject({
      name: "ApiError",
      error: { code: "SSE_ENDED_BEFORE_FINAL" }
    });

    expect(events).toEqual([{
      event: "accepted",
      id: null,
      data: { turn_id: "turn-1", client_request_id: "request-1" }
    }]);
  });

  it("returns the server final projection when it arrives", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      'event: accepted\ndata: {"turn_id":"turn-1"}\n\n',
      'event: final\ndata: {"result":{"status":"completed","reply":"完成"}}\n\n'
    ])));

    await expect(api.chatTurnStream({ message: "继续", client_request_id: "request-1" }))
      .resolves.toEqual({ result: { status: "completed", reply: "完成" } });
  });

  it("exposes structured stream errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      'event: error\ndata: {"status":409,"error":{"code":"TURN_IN_PROGRESS","message":"busy"}}\n\n'
    ])));

    await expect(api.chatTurnStream({ message: "继续", client_request_id: "request-1" }))
      .rejects.toBeInstanceOf(ApiError);
  });
});
