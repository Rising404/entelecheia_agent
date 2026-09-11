import { effectScope, nextTick, ref } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDocumentIngestJobs } from "./useDocumentIngestJobs";

function job(jobId, status = "queued", overrides = {}) {
  return {
    job_id: jobId,
    session_id: "session-a",
    path: `${jobId}.pdf`,
    status,
    stage: status === "succeeded" ? "active" : "parsing",
    can_retry: status === "failed",
    document_id: status === "succeeded" ? `doc-${jobId}` : null,
    ...overrides
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

async function settle() {
  for (let index = 0; index < 8; index += 1) await Promise.resolve();
  await nextTick();
}

function fakeTarget() {
  const target = new EventTarget();
  target.document = new EventTarget();
  Object.defineProperty(target.document, "visibilityState", {
    configurable: true,
    writable: true,
    value: "visible"
  });
  return target;
}

afterEach(() => {
  vi.useRealTimers();
});

describe("useDocumentIngestJobs", () => {
  it("loads durable jobs, polls only non-terminal jobs, and refreshes one successful document once", async () => {
    vi.useFakeTimers();
    const queued = job("queued");
    const failed = job("failed", "failed");
    const succeeded = job("queued", "succeeded", {
      processing_status: "partial",
      needs_vision: true
    });
    const api = {
      listDocumentIngestJobs: vi.fn().mockResolvedValue({ jobs: [queued, failed] }),
      getDocumentIngestJob: vi.fn().mockResolvedValue({ job: succeeded })
    };
    const onSucceeded = vi.fn().mockResolvedValue(undefined);
    const tracker = useDocumentIngestJobs({
      api,
      selectedSessionId: ref("session-a"),
      onSucceeded,
      target: fakeTarget(),
      intervalMs: 25
    });

    tracker.start();
    await settle();
    expect(api.listDocumentIngestJobs).toHaveBeenCalledWith({
      session_id: "session-a",
      limit: 50
    });

    await vi.advanceTimersByTimeAsync(25);
    await settle();

    expect(api.getDocumentIngestJob).toHaveBeenCalledTimes(1);
    expect(api.getDocumentIngestJob).toHaveBeenCalledWith("queued", {
      session_id: "session-a"
    });
    expect(tracker.jobs.value.map((item) => item.status)).toEqual(["succeeded", "failed"]);
    expect(onSucceeded).toHaveBeenCalledOnce();
    expect(onSucceeded).toHaveBeenCalledWith(succeeded, "session-a");

    await vi.advanceTimersByTimeAsync(100);
    await settle();
    expect(api.getDocumentIngestJob).toHaveBeenCalledTimes(1);
    expect(onSucceeded).toHaveBeenCalledOnce();
    tracker.stop();
  });

  it("coalesces focus, online, and visibility recovery behind one in-flight read", async () => {
    const target = fakeTarget();
    const initial = deferred();
    const recovered = deferred();
    let activeReads = 0;
    let maximumReads = 0;
    const api = {
      listDocumentIngestJobs: vi.fn()
        .mockImplementationOnce(async () => {
          activeReads += 1;
          maximumReads = Math.max(maximumReads, activeReads);
          const value = await initial.promise;
          activeReads -= 1;
          return value;
        })
        .mockImplementationOnce(async () => {
          activeReads += 1;
          maximumReads = Math.max(maximumReads, activeReads);
          const value = await recovered.promise;
          activeReads -= 1;
          return value;
        })
    };
    const tracker = useDocumentIngestJobs({
      api,
      selectedSessionId: ref("session-a"),
      target
    });

    tracker.start();
    target.dispatchEvent(new Event("focus"));
    target.dispatchEvent(new Event("online"));
    target.document.dispatchEvent(new Event("visibilitychange"));
    await settle();

    expect(api.listDocumentIngestJobs).toHaveBeenCalledTimes(1);
    initial.resolve({ jobs: [] });
    await settle();
    expect(api.listDocumentIngestJobs).toHaveBeenCalledTimes(2);
    recovered.resolve({ jobs: [] });
    await settle();
    expect(api.listDocumentIngestJobs).toHaveBeenCalledTimes(2);
    expect(maximumReads).toBe(1);
    tracker.stop();
  });

  it("keeps polling an automatic retry but never polls a manual terminal failure", async () => {
    vi.useFakeTimers();
    const automaticRetry = job("automatic", "failed", {
      can_retry: false,
      next_retry_at: "2026-08-19T12:00:00Z",
      error: { code: "coverage_pending", message: "覆盖未完成", retryable: true }
    });
    const manualRetry = job("manual", "failed", {
      can_retry: true,
      error: { code: "source_changed", message: "源文件已变化", retryable: true }
    });
    const api = {
      listDocumentIngestJobs: vi.fn().mockResolvedValue({ jobs: [automaticRetry, manualRetry] }),
      getDocumentIngestJob: vi.fn().mockResolvedValue({
        job: { ...automaticRetry, status: "queued", error: null }
      })
    };
    const tracker = useDocumentIngestJobs({
      api,
      selectedSessionId: ref("session-a"),
      target: fakeTarget(),
      intervalMs: 25
    });

    tracker.start();
    await settle();
    await vi.advanceTimersByTimeAsync(25);
    await settle();

    expect(api.getDocumentIngestJob).toHaveBeenCalledOnce();
    expect(api.getDocumentIngestJob).toHaveBeenCalledWith("automatic", {
      session_id: "session-a"
    });
    tracker.stop();
  });

  it("never publishes an A response after switching to B", async () => {
    const oldA = deferred();
    const api = {
      listDocumentIngestJobs: vi.fn()
        .mockReturnValueOnce(oldA.promise)
        .mockResolvedValueOnce({ jobs: [job("b", "queued", { session_id: "session-b" })] })
    };
    const selectedSessionId = ref("session-a");
    const tracker = useDocumentIngestJobs({ api, selectedSessionId, target: fakeTarget() });

    tracker.start();
    selectedSessionId.value = "session-b";
    oldA.resolve({ jobs: [job("old-a")] });
    await settle();
    await settle();

    expect(api.listDocumentIngestJobs.mock.calls).toEqual([
      [{ session_id: "session-a", limit: 50 }],
      [{ session_id: "session-b", limit: 50 }]
    ]);
    expect(tracker.jobs.value.map((item) => item.job_id)).toEqual(["b"]);
    tracker.stop();
  });

  it("uses a generation token when selection returns A to B to A", async () => {
    const oldA = deferred();
    const currentA = deferred();
    const api = {
      listDocumentIngestJobs: vi.fn()
        .mockReturnValueOnce(oldA.promise)
        .mockReturnValueOnce(currentA.promise)
    };
    const selectedSessionId = ref("session-a");
    const tracker = useDocumentIngestJobs({ api, selectedSessionId, target: fakeTarget() });

    tracker.start();
    selectedSessionId.value = "session-b";
    selectedSessionId.value = "session-a";
    oldA.resolve({ jobs: [job("stale-a")] });
    await settle();
    currentA.resolve({ jobs: [job("current-a")] });
    await settle();

    expect(api.listDocumentIngestJobs).toHaveBeenCalledTimes(2);
    expect(api.listDocumentIngestJobs).toHaveBeenLastCalledWith({
      session_id: "session-a",
      limit: 50
    });
    expect(tracker.jobs.value.map((item) => item.job_id)).toEqual(["current-a"]);
    tracker.stop();
  });

  it("invalidates pending reads and removes recovery listeners when its scope is disposed", async () => {
    const pending = deferred();
    const target = fakeTarget();
    const api = { listDocumentIngestJobs: vi.fn().mockReturnValue(pending.promise) };
    const scope = effectScope();
    let tracker;
    scope.run(() => {
      tracker = useDocumentIngestJobs({
        api,
        selectedSessionId: ref("session-a"),
        target
      });
      tracker.start();
    });

    scope.stop();
    pending.resolve({ jobs: [job("too-late")] });
    await settle();
    target.dispatchEvent(new Event("focus"));
    await settle();

    expect(tracker.jobs.value).toEqual([]);
    expect(api.listDocumentIngestJobs).toHaveBeenCalledTimes(1);
  });

  it("treats only an HTTP 404 as endpoint absence", async () => {
    const onError = vi.fn();
    const missing = useDocumentIngestJobs({
      api: { listDocumentIngestJobs: vi.fn().mockRejectedValue({ status: 404 }) },
      selectedSessionId: ref("session-a"),
      onError,
      target: fakeTarget()
    });
    missing.start();
    await settle();
    expect(missing.supported.value).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    missing.stop();

    const unavailable = { status: 503, message: "unavailable" };
    const failing = useDocumentIngestJobs({
      api: { listDocumentIngestJobs: vi.fn().mockRejectedValue(unavailable) },
      selectedSessionId: ref("session-a"),
      onError,
      target: fakeTarget()
    });
    failing.start();
    await settle();
    expect(failing.supported.value).not.toBe(false);
    expect(onError).toHaveBeenCalledWith(unavailable);
    failing.stop();
  });

  it("retries only a retryable job from the exact current session", async () => {
    const failed = job("failed", "failed");
    const queued = { ...failed, status: "queued", can_retry: false };
    const api = {
      listDocumentIngestJobs: vi.fn().mockResolvedValue({ jobs: [failed] }),
      retryDocumentIngestJob: vi.fn().mockResolvedValue({ job: queued })
    };
    const selectedSessionId = ref("session-a");
    const tracker = useDocumentIngestJobs({ api, selectedSessionId, target: fakeTarget() });
    tracker.start();
    await settle();

    await expect(tracker.retry(failed)).resolves.toBe(true);
    expect(api.retryDocumentIngestJob).toHaveBeenCalledWith("failed", {
      session_id: "session-a"
    });
    expect(tracker.jobs.value[0].status).toBe("queued");

    selectedSessionId.value = "session-b";
    await expect(tracker.retry(failed)).resolves.toBe(false);
    expect(api.retryDocumentIngestJob).toHaveBeenCalledTimes(1);
    tracker.stop();
  });
});
