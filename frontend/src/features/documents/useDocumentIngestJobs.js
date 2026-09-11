import { computed, getCurrentScope, onScopeDispose, ref, watch } from "vue";

export const DOCUMENT_INGEST_POLL_INTERVAL_MS = 2_000;
const DOCUMENT_INGEST_JOB_LIMIT = 50;
const PUBLIC_STATUSES = new Set(["queued", "running", "succeeded", "failed"]);
const NON_TERMINAL_STATUSES = new Set(["queued", "running"]);

function isPollableJob(job) {
  if (NON_TERMINAL_STATUSES.has(job?.status)) return true;
  // retryable_failed 会投影为 `failed`，但仍属于存储层自动处理的工作；
  // terminal_failed 是唯一带有 can_retry 的失败状态。
  return Boolean(
    job?.status === "failed" &&
    job?.can_retry === false &&
    job?.error?.retryable === true
  );
}

function isExactEndpointMissing(error) {
  return error?.status === 404;
}

function isJobForSession(value, sessionId) {
  return Boolean(
    value &&
    typeof value === "object" &&
    typeof value.job_id === "string" &&
    value.job_id.trim() &&
    value.session_id === sessionId &&
    PUBLIC_STATUSES.has(value.status)
  );
}

function replaceJob(items, job) {
  const index = items.findIndex((item) => item.job_id === job.job_id);
  if (index < 0) return [job, ...items];
  return items.map((item, itemIndex) => itemIndex === index ? job : item);
}

/**
 * 持久文档收录投影的会话级所有者。
 *
 * 所有读取都经过同一个合并循环，因此 focus/online/timer 的密集信号不会产生重叠请求。
 * 同步 Session watcher 会在旧响应发布前，为 A -> B -> A 的变化推进 generation。
 */
export function useDocumentIngestJobs({
  api,
  selectedSessionId,
  onSucceeded = async () => {},
  onError = () => {},
  target = globalThis,
  intervalMs = DOCUMENT_INGEST_POLL_INTERVAL_MS,
  limit = DOCUMENT_INGEST_JOB_LIMIT
}) {
  const jobs = ref([]);
  const loading = ref(false);
  const supported = ref(null);
  const lastError = ref(null);
  const activeJobs = computed(() => jobs.value.filter(isPollableJob));

  let started = false;
  let timer = null;
  let generation = 0;
  let pendingRead = null;
  let readLoop = null;
  const succeededRefreshes = new Set();

  function exactCurrent(token, sessionId) {
    return started && token === generation && selectedSessionId.value === sessionId;
  }

  function clearTimer() {
    if (timer === null) return;
    globalThis.clearTimeout(timer);
    timer = null;
  }

  function schedulePoll() {
    clearTimer();
    if (!started || supported.value === false || activeJobs.value.length === 0) return;
    timer = globalThis.setTimeout(() => {
      timer = null;
      void requestRead("poll");
    }, intervalMs);
  }

  async function notifySucceeded(published, token, sessionId) {
    for (const job of published) {
      if (job.status !== "succeeded") continue;
      const key = `${sessionId}\u0000${job.job_id}`;
      if (succeededRefreshes.has(key)) continue;
      succeededRefreshes.add(key);
      if (!exactCurrent(token, sessionId)) return;
      try {
        await onSucceeded(job, sessionId);
      } catch (error) {
        if (exactCurrent(token, sessionId)) onError(error);
      }
    }
  }

  async function publishList(payload, token, sessionId) {
    if (!exactCurrent(token, sessionId)) return;
    const next = Array.isArray(payload?.jobs)
      ? payload.jobs.filter((job) => isJobForSession(job, sessionId))
      : [];
    jobs.value = next;
    supported.value = true;
    lastError.value = null;
    await notifySucceeded(next, token, sessionId);
  }

  async function publishJob(candidate, token, sessionId) {
    if (!exactCurrent(token, sessionId) || !isJobForSession(candidate, sessionId)) return;
    jobs.value = replaceJob(jobs.value, candidate);
    supported.value = true;
    lastError.value = null;
    await notifySucceeded([candidate], token, sessionId);
  }

  async function executeList(token, sessionId) {
    if (typeof api?.listDocumentIngestJobs !== "function") {
      if (exactCurrent(token, sessionId)) supported.value = false;
      return;
    }
    try {
      const payload = await api.listDocumentIngestJobs({ session_id: sessionId, limit });
      await publishList(payload, token, sessionId);
    } catch (error) {
      if (!exactCurrent(token, sessionId)) return;
      if (isExactEndpointMissing(error)) {
        supported.value = false;
        jobs.value = [];
        lastError.value = null;
        return;
      }
      lastError.value = error;
      onError(error);
    }
  }

  async function executePoll(token, sessionId) {
    if (typeof api?.getDocumentIngestJob !== "function") return;
    const identifiers = activeJobs.value.map((job) => job.job_id);
    for (const jobId of identifiers) {
      if (!exactCurrent(token, sessionId)) return;
      // 同一轮中的上一条详情可能已让任务进入终态，不要再为已被取代的快照发起冗余读取。
      const current = jobs.value.find((job) => job.job_id === jobId);
      if (!current || !isPollableJob(current)) continue;
      try {
        const payload = await api.getDocumentIngestJob(jobId, { session_id: sessionId });
        await publishJob(payload?.job, token, sessionId);
      } catch (error) {
        if (!exactCurrent(token, sessionId)) return;
        lastError.value = error;
        onError(error);
      }
    }
  }

  function requestRead(mode = "list") {
    if (!started || !selectedSessionId.value || supported.value === false) {
      return Promise.resolve();
    }
    // 列表读取是更强的恢复读取。当 timer/detail 请求仍在执行时，多个信号会合并为
    // 一次待处理列表读取。
    if (mode === "list" || pendingRead === null) pendingRead = mode;
    if (readLoop) return readLoop;

    readLoop = (async () => {
      loading.value = true;
      while (started && pendingRead) {
        const requestedMode = pendingRead;
        pendingRead = null;
        const token = generation;
        const sessionId = selectedSessionId.value;
        if (!sessionId) continue;
        if (requestedMode === "poll") await executePoll(token, sessionId);
        else await executeList(token, sessionId);
      }
    })().finally(() => {
      readLoop = null;
      loading.value = false;
      if (started && pendingRead) void requestRead(pendingRead);
      else schedulePoll();
    });
    return readLoop;
  }

  function accept(job, sessionId = selectedSessionId.value) {
    if (!started || sessionId !== selectedSessionId.value || !isJobForSession(job, sessionId)) {
      return false;
    }
    // 使可能在入队提交前捕获的旧列表失效；它的空投影不能抹掉已经确认的操作。
    generation += 1;
    clearTimer();
    jobs.value = replaceJob(jobs.value, job);
    supported.value = true;
    lastError.value = null;
    const token = generation;
    void notifySucceeded([job], token, sessionId).finally(() => {
      if (exactCurrent(token, sessionId)) schedulePoll();
    });
    return true;
  }

  async function retry(job) {
    const sessionId = selectedSessionId.value;
    const token = generation;
    if (
      !started ||
      !isJobForSession(job, sessionId) ||
      job.status !== "failed" ||
      job.can_retry !== true ||
      typeof api?.retryDocumentIngestJob !== "function"
    ) return false;
    try {
      const payload = await api.retryDocumentIngestJob(job.job_id, { session_id: sessionId });
      if (!exactCurrent(token, sessionId)) return false;
      return accept(payload?.job, sessionId);
    } catch (error) {
      if (exactCurrent(token, sessionId)) {
        lastError.value = error;
        onError(error);
      }
      return false;
    }
  }

  function handleRecovery() {
    if (started) void requestRead("list");
  }

  function handleVisibilityChange() {
    if (target.document?.visibilityState !== "hidden") handleRecovery();
  }

  function start() {
    if (started) return;
    started = true;
    target.addEventListener?.("focus", handleRecovery);
    target.addEventListener?.("online", handleRecovery);
    target.document?.addEventListener?.("visibilitychange", handleVisibilityChange);
    void requestRead("list");
  }

  function stop() {
    if (!started) return;
    started = false;
    generation += 1;
    pendingRead = null;
    clearTimer();
    jobs.value = [];
    loading.value = false;
    target.removeEventListener?.("focus", handleRecovery);
    target.removeEventListener?.("online", handleRecovery);
    target.document?.removeEventListener?.("visibilitychange", handleVisibilityChange);
  }

  const stopSessionWatch = watch(selectedSessionId, (sessionId, previousSessionId) => {
    if (sessionId === previousSessionId) return;
    generation += 1;
    pendingRead = null;
    clearTimer();
    jobs.value = [];
    lastError.value = null;
    supported.value = null;
    if (started && sessionId) void requestRead("list");
  }, { flush: "sync" });

  if (getCurrentScope()) {
    onScopeDispose(() => {
      stop();
      stopSessionWatch();
    });
  }

  return {
    jobs,
    activeJobs,
    loading,
    supported,
    lastError,
    start,
    stop,
    reconcile: () => requestRead("list"),
    accept,
    retry
  };
}
