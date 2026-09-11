import { fetchWithReadRetry } from "./apiTransport";


const API_BASE =
  globalThis.personagraphApiBase ||
  globalThis.personagraphDesktop?.apiBase ||
  "http://127.0.0.1:8765";

export class ApiError extends Error {
  constructor(message, { status = null, path = "", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.payload = payload;
    this.error = payload?.error || null;
  }
}

async function request(path, options = {}) {
  const response = await fetchWithReadRetry(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const text = await response.text();
  const payload = text ? parseJson(text) : {};
  if (!response.ok) {
    throw new ApiError(payload?.error?.message || response.statusText, {
      status: response.status,
      path,
      payload
    });
  }
  return payload;
}

async function requestSse(path, options = {}, onEvent = () => {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...(options.headers || {}) },
    ...options
  });
  const contentType = response.headers.get("Content-Type") || "";
  if (!response.ok || !contentType.includes("text/event-stream") || !response.body) {
    const text = await response.text();
    const payload = text ? parseJson(text) : {};
    throw new ApiError(payload?.error?.message || response.statusText, {
      status: response.status,
      path,
      payload
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const item = parseSseEvent(part);
      if (!item) continue;
      onEvent(item);
      if (item.event === "final") {
        finalPayload = item.data;
      }
      if (item.event === "error") {
        throw new ApiError(item.data?.error?.message || "Stream failed", {
          status: item.data?.status || 500,
          path,
          payload: item.data
        });
      }
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const item = parseSseEvent(buffer);
    if (item) {
      onEvent(item);
      if (item.event === "final") finalPayload = item.data;
    }
  }
  if (finalPayload) return finalPayload;

  // 服务端可能已经持久化接受了 Turn，却在发送最终投影前丢失数据流。若将这种情况
  // 当作 `{}`，编辑器会丢弃客户端请求 ID，导致重试无法保持幂等。因此这里返回明确
  // 错误，由调用方继续保留待处理请求。
  throw new ApiError("连接在本轮结果返回前中断，请重试。", {
    path,
    payload: { error: { code: "SSE_ENDED_BEFORE_FINAL" } }
  });
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function parseSseEvent(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  let id = null;
  const data = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("id:")) {
      id = line.slice(3).trim() || null;
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  if (!data.length) return null;
  return { event, id, data: parseJson(data.join("\n")) };
}

function queryString(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, value);
    }
  });
  const text = query.toString();
  return text ? `?${text}` : "";
}

export const api = {
  base: API_BASE,

  status() {
    return request("/api/status");
  },

  listSessions(params) {
    return request(`/api/sessions${queryString(params)}`);
  },

  getSession(sessionId) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}`);
  },

  controlTurnPostCommitJobs(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/post-commit/control`, {
      method: "POST", body: JSON.stringify(payload)
    });
  },

  getRuntimeEvents(sessionId, params = {}) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/runtime-events${queryString(params)}`
    );
  },

  getInSessionTaskDetails(sessionId, taskId) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/insession-tasks/${encodeURIComponent(taskId)}`
    );
  },

  createSession(payload) {
    return request("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  patchSession(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    });
  },

  listFolders(params) {
    return request(`/api/folders${queryString(params)}`);
  },

  createFolder(payload) {
    return request("/api/folders", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  patchFolder(folderId, payload) {
    return request(`/api/folders/${encodeURIComponent(folderId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    });
  },

  deleteFolder(folderId) {
    return request(`/api/folders/${encodeURIComponent(folderId)}`, {
      method: "DELETE"
    });
  },

  uploadAttachment(sessionId, file) {
    // 直接发送原始二进制而非 multipart：服务端直接读取请求体，并从请求头获取名称，
    // 因而文件名中的任何内容都不会进入路径解析。
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/attachments`, {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        // 请求头采用 latin-1，因此需要对 UTF-8 文件名进行百分号编码。
        "X-Attachment-Filename": encodeURIComponent(file.name || "file")
      },
      body: file
    });
  },
  listAttachments(sessionId) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/attachments`);
  },
  deleteAttachment(sessionId, attachmentId) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/attachments/${encodeURIComponent(attachmentId)}`,
      { method: "DELETE" }
    );
  },
  chatTurn(payload) {
    return request("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  chatTurnStream(payload, onEvent) {
    return requestSse("/api/chat/stream", {
      method: "POST",
      body: JSON.stringify(payload)
    }, onEvent);
  },

  getSessionContext(sessionId, params = {}) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/session-context${queryString(params)}`
    );
  },

  explainSessionContext(sessionId, params) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/session-context/explain${queryString(params)}`
    );
  },

  exportSessionContext(sessionId, params = {}) {
    return request(
      `/api/sessions/${encodeURIComponent(sessionId)}/session-context/export${queryString(params)}`
    );
  },

  clearSessionContext(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/session-context/clear`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  createSessionContextCorrection(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/session-context/corrections`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  previewSessionContextRepair(sessionId, payload = {}) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/session-context/repair/preview`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  applySessionContextRepair(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/session-context/repair/apply`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  // --- 文档工作集（FR-4）---
  listDocuments(params) {
    return request(`/api/documents${queryString(params)}`);
  },
  createDocumentIngestJob(payload) {
    return request("/api/document-ingest-jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  listDocumentIngestJobs(params) {
    return request(`/api/document-ingest-jobs${queryString(params)}`);
  },
  getDocumentIngestJob(jobId, params) {
    return request(
      `/api/document-ingest-jobs/${encodeURIComponent(jobId)}${queryString(params)}`
    );
  },
  retryDocumentIngestJob(jobId, payload) {
    return request(`/api/document-ingest-jobs/${encodeURIComponent(jobId)}/retry`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },
  patchDocument(docId, payload) {
    return request(`/api/documents/${encodeURIComponent(docId)}`, {
      method: "PATCH", body: JSON.stringify(payload)
    });
  },
  detachDocument(docId, payload) {
    return request(`/api/documents/${encodeURIComponent(docId)}/detach`, {
      method: "POST", body: JSON.stringify(payload)
    });
  },
  deleteDocument(docId, params) {
    return request(`/api/documents/${encodeURIComponent(docId)}${queryString(params)}`, {
      method: "DELETE"
    });
  },
  // --- 模型配置（FR-5：设置/密钥）---
  getConfig() {
    return request("/api/config");
  },
  updateConfig(payload) {
    return request("/api/config", { method: "PUT", body: JSON.stringify(payload) });
  },

  // --- 工作区文件管理（FR-8：会话 working_dir 内真实文件，删除→系统回收站）---
  listWorkspaceFiles(sessionId, params) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/files${queryString(params)}`);
  },
  readWorkspaceFile(sessionId, params) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/file${queryString(params)}`);
  },
  createWorkspaceEntry(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/files`, {
      method: "POST", body: JSON.stringify(payload)
    });
  },
  writeWorkspaceFile(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/file`, {
      method: "PUT", body: JSON.stringify(payload)
    });
  },
  deleteWorkspaceEntry(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/files`, {
      method: "DELETE", body: JSON.stringify(payload)
    });
  },

  purgeSession(sessionId) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  },

  emptySessionTrash() {
    return request("/api/sessions/trash/empty", { method: "POST", body: JSON.stringify({}) });
  },

  getBackground() {
    return request("/api/appearance/background");
  },

  // 资产字节要走 fetch，不能直接交给 <img src>。图片/视频元素既拿不到 Electron 在网络层
  // 注入的 token，浏览器也不会为跨源媒体请求带上 Origin，那条路只会拿到 401，再被 ORB
  // 当成"不是图片"拦掉。这里复用与其它请求相同的鉴权路径，字节由调用方转成 object URL。
  async fetchBackgroundAsset(path) {
    const response = await fetchWithReadRetry(`${API_BASE}${path}`);
    if (!response.ok) {
      throw new ApiError(response.statusText, { status: response.status, path });
    }
    return response.blob();
  },

  // 原始字节直传，不走 JSON——一段几十 MB 的视频 base64 之后还要再涨三分之一。
  async uploadBackground(file) {
    const buffer = await file.arrayBuffer();
    return request("/api/appearance/background", {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: buffer
    });
  },

  clearBackground() {
    return request("/api/appearance/background", { method: "DELETE" });
  },

  listProjects(status = "active", query = "") {
    const search = query.trim() ? `&query=${encodeURIComponent(query.trim())}` : "";
    return request(`/api/projects?status=${encodeURIComponent(status)}${search}`);
  },

  reorderProjects(paths) {
    return request("/api/projects/reorder", {
      method: "POST", body: JSON.stringify({ paths })
    });
  },

  pinProject(path, pinned) {
    return request("/api/projects/pin", {
      method: "POST", body: JSON.stringify({ path, pinned })
    });
  },

  rememberProject(payload) {
    return request("/api/projects", { method: "POST", body: JSON.stringify(payload) });
  },

  renameProject(payload) {
    return request("/api/projects", { method: "PATCH", body: JSON.stringify(payload) });
  },

  forgetProject(path) {
    return request("/api/projects/forget", { method: "POST", body: JSON.stringify({ path }) });
  },

  autonameSession(sessionId, payload) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/autoname`, {
      method: "POST", body: JSON.stringify(payload)
    });
  },

  listModelProfiles() {
    return request("/api/model-profiles");
  },

  createModelProfile(payload) {
    return request("/api/model-profiles", { method: "POST", body: JSON.stringify(payload) });
  },

  updateModelProfile(profileId, payload) {
    return request(`/api/model-profiles/${encodeURIComponent(profileId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    });
  },

  deleteModelProfile(profileId) {
    return request(`/api/model-profiles/${encodeURIComponent(profileId)}`, { method: "DELETE" });
  },

  activateModelProfile(profileId) {
    return request(`/api/model-profiles/${encodeURIComponent(profileId)}/activate`, {
      method: "POST",
      body: JSON.stringify({})
    });
  },

  // 明文单独取：列表里从来没有 key，普通的设置页加载因此不带任何密钥。
  revealModelProfileSecret(profileId) {
    return request(`/api/model-profiles/${encodeURIComponent(profileId)}/secret`);
  },
};
