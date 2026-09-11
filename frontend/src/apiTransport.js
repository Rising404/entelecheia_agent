const READ_RETRY_DELAY_MS = 120;

function defaultDelay(milliseconds) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, milliseconds));
}

function isRetryableReadFailure(error, options) {
  const method = String(options?.method || "GET").trim().toUpperCase();
  if (!new Set(["GET", "HEAD"]).has(method)) return false;
  if (options?.signal?.aborted || error?.name === "AbortError") return false;
  return error instanceof TypeError;
}

/**
 * 浏览器级网络失败后，重试一次幂等读取。
 *
 * HTTP 响应、写操作、流和显式取消绝不在这里重试。这仅覆盖新 renderer 替换旧页面并立即
 * 恢复会话状态时出现的短暂连接竞态。
 */
export async function fetchWithReadRetry(
  input,
  options = {},
  {
    fetchImpl = globalThis.fetch,
    delay = defaultDelay,
    retryDelayMs = READ_RETRY_DELAY_MS
  } = {}
) {
  try {
    return await fetchImpl(input, options);
  } catch (error) {
    if (!isRetryableReadFailure(error, options)) throw error;
    await delay(retryDelayMs);
    return fetchImpl(input, options);
  }
}
