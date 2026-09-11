import { getCurrentScope, onScopeDispose, ref } from "vue";

const DEFAULT_EVENT_LIMIT = 100;
const MAX_CATCH_UP_PAGES = 5;

function isNonEmptyString(value) {
  return typeof value === "string" && value.length > 0;
}

/**
 * 接受稳定的公开生命周期信封，而非旧工具事件结构。stage 与 status 刻意保留为
 * 开放字符串：新版 Host 的阶段必须落入通用卡片，不能从用户时间线中消失。
 */
function isPublicRuntimeEvent(event) {
  return Boolean(
    event &&
    event.schema_version === 1 &&
    isNonEmptyString(event.event_id) &&
    Number.isInteger(event.sequence) && event.sequence > 0 &&
    isNonEmptyString(event.session_id) &&
    isNonEmptyString(event.turn_id) &&
    isNonEmptyString(event.stage) &&
    isNonEmptyString(event.status) &&
    Number.isFinite(Date.parse(event.occurred_at || "")) &&
    event.prompt_replay === false &&
    (event.error_code === undefined || event.error_code === null || isNonEmptyString(event.error_code)) &&
    (event.retryable === undefined || typeof event.retryable === "boolean")
  );
}

function eventTime(event) {
  const value = Date.parse(event?.occurred_at || "");
  return Number.isFinite(value) ? value : 0;
}

function latestDurableCursor(events) {
  let latest = null;
  for (const event of events || []) {
    if (!latest || event.sequence > latest.sequence) latest = event;
  }
  return latest?.event_id || null;
}

function isMissingEndpoint(error) {
  return error?.status === 404 || error?.payload?.error?.code === "NOT_FOUND";
}

function isInvalidCursor(error) {
  return error?.status === 409 && error?.payload?.error?.code === "RUNTIME_EVENT_CURSOR_NOT_FOUND";
}

export function mergeRuntimeEvents(current, incoming, limit = DEFAULT_EVENT_LIMIT) {
  const merged = new Map();
  for (const event of [...(current || []), ...(incoming || [])]) {
    if (!isPublicRuntimeEvent(event)) continue;
    const previous = merged.get(event.event_id);
    // 一个持久事件 ID 只能对应一个追加序号。忽略格式错误的重复项，避免它重排
    // 可见时间线。
    if (previous && previous.sequence !== event.sequence) continue;
    merged.set(event.event_id, event);
  }
  return [...merged.values()]
    .sort((left, right) => left.sequence - right.sequence || eventTime(left) - eventTime(right) ||
      left.event_id.localeCompare(right.event_id))
    .slice(-Math.max(1, limit));
}

/**
 * 会话作用域内的公开生命周期时间线。
 *
 * 时间线只用于可观测性。当前执行权归 TurnExecutionWindow 所有，因此本模块不会
 * 从已退休的 active-run 端点轮询或推断状态。
 */
export function useRuntimeEvents({
  api,
  onError = () => {},
  limit = DEFAULT_EVENT_LIMIT
}) {
  const events = ref([]);
  const enabled = ref(false);
  const loading = ref(false);
  const lastError = ref(null);
  let generation = 0;
  let currentSessionId = "";

  function resetState() {
    events.value = [];
    enabled.value = false;
    loading.value = false;
    lastError.value = null;
  }

  function clear() {
    generation += 1;
    currentSessionId = "";
    resetState();
  }

  function isCurrent(token, sessionId) {
    return token === generation && sessionId === currentSessionId;
  }

  function reportError(error, token, sessionId) {
    if (!isCurrent(token, sessionId)) return;
    lastError.value = error;
    onError(error);
  }

  function ingest(event) {
    if (!isPublicRuntimeEvent(event)) return;
    if (currentSessionId && event.session_id !== currentSessionId) return;
    if (!currentSessionId) currentSessionId = event.session_id;
    events.value = mergeRuntimeEvents(events.value, [event], limit);
    enabled.value = true;
  }

  async function readCatchUpPage(sessionId, cursor) {
    try {
      return await api.getRuntimeEvents(sessionId, cursor ? { after: cursor, limit } : { limit });
    } catch (error) {
      if (!cursor || !isInvalidCursor(error)) throw error;
      // 保留策略可能删除旧游标。此时只在有界尾部重新锚定一次，绝不让恢复过程
      // 变成无界重放。
      return api.getRuntimeEvents(sessionId, { limit });
    }
  }

  async function catchUp(sessionId = currentSessionId, { token = generation } = {}) {
    if (!sessionId || !isCurrent(token, sessionId) || typeof api?.getRuntimeEvents !== "function") {
      return { enabled: false, events: [] };
    }
    try {
      let cursor = latestDurableCursor(events.value);
      let payload = null;
      for (let page = 0; page < MAX_CATCH_UP_PAGES; page += 1) {
        payload = await readCatchUpPage(sessionId, cursor);
        if (!isCurrent(token, sessionId)) return payload;
        enabled.value = payload?.enabled === true;
        if (!enabled.value) {
          resetState();
          return payload;
        }
        events.value = mergeRuntimeEvents(events.value, payload?.events || [], limit);
        const nextCursor = payload?.next_after || latestDurableCursor(events.value);
        if (!payload?.has_more || !nextCursor || nextCursor === cursor) break;
        cursor = nextCursor;
      }
      return payload || { enabled: enabled.value, events: [] };
    } catch (error) {
      if (!isCurrent(token, sessionId)) return { enabled: false, events: [] };
      if (isMissingEndpoint(error)) {
        resetState();
        return { enabled: false, events: [] };
      }
      reportError(error, token, sessionId);
      return { enabled: enabled.value, events: [] };
    }
  }

  async function load(sessionId) {
    const changedSession = sessionId !== currentSessionId;
    currentSessionId = sessionId || "";
    const requestGeneration = ++generation;
    loading.value = true;
    lastError.value = null;
    if (changedSession) events.value = [];
    if (!sessionId || typeof api?.getRuntimeEvents !== "function") {
      if (isCurrent(requestGeneration, currentSessionId)) resetState();
      return { enabled: false, events: [] };
    }
    try {
      const payload = await api.getRuntimeEvents(sessionId, { limit });
      if (!isCurrent(requestGeneration, sessionId)) return payload;
      enabled.value = payload?.enabled === true;
      events.value = enabled.value
        ? mergeRuntimeEvents(changedSession ? [] : events.value, payload?.events || [], limit)
        : [];
      return payload;
    } catch (error) {
      if (!isCurrent(requestGeneration, sessionId)) return { enabled: false, events: [] };
      events.value = [];
      enabled.value = false;
      if (!isMissingEndpoint(error)) reportError(error, requestGeneration, sessionId);
      return { enabled: false, events: [] };
    } finally {
      if (isCurrent(requestGeneration, sessionId)) loading.value = false;
    }
  }

  async function reconcile(sessionId = currentSessionId) {
    if (!sessionId) return { enabled: false, events: [] };
    if (sessionId !== currentSessionId) return load(sessionId);
    return catchUp(sessionId, { token: generation });
  }

  if (getCurrentScope()) onScopeDispose(clear);

  return {
    events,
    enabled,
    loading,
    lastError,
    clear,
    ingest,
    load,
    catchUp,
    reconcile
  };
}
