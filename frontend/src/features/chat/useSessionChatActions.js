import { streamEventDetail } from "./presentation";
import {
  RUNTIME_MODE_TASK,
  RUNTIME_MODE_TURN,
  runtimePolicyForMode
} from "../runtime/runtimeRouting";

// 与 input_processing/attachments/storage.py 的 MAX_ATTACHMENTS_PER_TURN 一致。
// 前端写一个更大的数，只会让超出的那几个必然失败。
export const MAX_ATTACHMENTS_PER_TURN = 16;
export const TASK_MODE_SEND_BLOCKED_NOTICE = "任务模式当前仍在开发中，暂不提供消息发送功能。";

function initialDraftTitle(message) {
  const normalized = String(message || "").trim().replace(/\s+/g, " ");
  return Array.from(normalized).slice(0, 24).join("") || "新会话";
}

export function useSessionChatActions({
  api,
  state,
  selectedSession,
  sessionTitleDirty,
  canSendChat,
  canAnswerPendingQuestion = { value: false },
  drafts,
  confirmDiscardChanges,
  clearMessage,
  showError,
  workspace = null,
  projects = null,
  runtimeTimeline = null,
  runtimeMode = { value: RUNTIME_MODE_TURN },
  clearInSessionTaskDetails = () => {},
  loadRelatedInSessionTaskDetails = async () => [],
  clock = () => new Date().toLocaleTimeString()
}) {
  const {
    mode, loading, saving, notice, sessionStatus, sessionSearch, sessions, selectedSessionId,
    sessionDetail, sessionTitleDraft, newSessionTitle, chatInput, chatBusy, chatStreamEvents,
    runtimeEvents, streamingReply, lastTurnResult, pendingClientRequest,
    pendingAttachments, attachmentUploading
  } = state;
  let sessionSelectionGeneration = 0;
  let sessionListGeneration = 0;
  let chatSendGeneration = 0;
  let attachmentUploadGeneration = 0;

  function isCurrentSessionSelection(generation, sessionId) {
    return generation === sessionSelectionGeneration && selectedSessionId.value === sessionId;
  }

  function isCurrentChatSend(generation, sessionId) {
    return generation === chatSendGeneration && selectedSessionId.value === sessionId;
  }

  function isCurrentAttachmentUpload(generation, sessionId) {
    return generation === attachmentUploadGeneration && selectedSessionId.value === sessionId;
  }

  function invalidateSessionScopedRequests() {
    chatSendGeneration += 1;
    attachmentUploadGeneration += 1;
    chatBusy.value = false;
    attachmentUploading.value = false;
  }

  function clearTurnSurface() {
    lastTurnResult.value = null;
    chatStreamEvents.value = [];
    if (runtimeTimeline) runtimeTimeline.clear();
    else runtimeEvents.value = [];
    streamingReply.value = "";
  }

  function clearSessionAuthority({ clearTurn = true } = {}) {
    sessionDetail.value = null;
    sessionTitleDraft.value = "";
    if (clearTurn) clearTurnSurface();
  }

  async function loadSessions(preferredId = selectedSessionId.value) {
    const requestGeneration = ++sessionListGeneration;
    const selectionGenerationAtStart = sessionSelectionGeneration;
    loading.value = true;
    clearMessage();
    try {
      if (workspace) {
        await workspace.refresh({ status: sessionStatus.value, query: sessionSearch.value });
      } else {
        const payload = await api.listSessions({ status: sessionStatus.value, query: sessionSearch.value.trim(), limit: 80 });
        if (requestGeneration !== sessionListGeneration) return;
        sessions.value = payload.sessions || [];
      }
      if (
        requestGeneration !== sessionListGeneration ||
        selectionGenerationAtStart !== sessionSelectionGeneration
      ) return;
      if (projects?.draftActive?.value && !preferredId) return;
      const nextId = sessions.value.some((session) => session.id === preferredId) ? preferredId : sessions.value[0]?.id || "";
      if (nextId) await selectSession(nextId);
      else {
        sessionSelectionGeneration += 1;
        invalidateSessionScopedRequests();
        if (typeof workspace?.selectSession === "function") workspace.selectSession("");
        selectedSessionId.value = "";
        clearSessionAuthority();
        pendingAttachments.value = [];
        pendingClientRequest.value = null;
        clearInSessionTaskDetails();
      }
    } catch (err) {
      if (
        requestGeneration === sessionListGeneration &&
        selectionGenerationAtStart === sessionSelectionGeneration
      ) showError(err);
    } finally {
      if (requestGeneration === sessionListGeneration) loading.value = false;
    }
  }

  const reloadSessionsFromSearch = () => loadSessions("");
  async function guardedReloadSessionsFromSearch() { if (confirmDiscardChanges()) await reloadSessionsFromSearch(); }
  async function clearSessionSearch() { sessionSearch.value = ""; await loadSessions(""); }
  async function guardedClearSessionSearch() { if (confirmDiscardChanges()) await clearSessionSearch(); }

  async function guardedSetSessionStatus(nextStatus) {
    if (sessionStatus.value === nextStatus || !confirmDiscardChanges()) return;
    sessionStatus.value = nextStatus;
    await reloadSessionsFromSearch();
  }

  async function selectSession(sessionId, { resetTurnState = true, loadedDetail = null } = {}) {
    const requestGeneration = ++sessionSelectionGeneration;
    // 聊天交接 authority 前，workspace store 可能已经预选新建的 Session。优先采用
    // 已加载详情的所有者，确保这里仍执行真实的 A -> B 草稿/附件交接。
    const previousSessionId = sessionDetail.value?.session?.id || selectedSessionId.value;
    const changedSession = Boolean(previousSessionId && previousSessionId !== sessionId);
    const needsInitialDraftLoad = !previousSessionId;
    if (changedSession) drafts.save(previousSessionId, chatInput.value);
    if (changedSession || needsInitialDraftLoad) invalidateSessionScopedRequests();
    if (changedSession) {
      pendingAttachments.value = [];
      pendingClientRequest.value = null;
      clearInSessionTaskDetails();
    }
    if (changedSession || needsInitialDraftLoad) {
      clearSessionAuthority({ clearTurn: false });
    }
    clearMessage();
    if (resetTurnState || changedSession || needsInitialDraftLoad) clearTurnSurface();
    // 在较慢的详情或附件读取之前，先限定并清空共享工作区投影。请求 generation 的
    // 竞态保护由 store 负责。
    if (typeof workspace?.selectSession === "function") workspace.selectSession(sessionId);
    selectedSessionId.value = sessionId;
    if (changedSession || needsInitialDraftLoad) chatInput.value = drafts.load(sessionId);
    const workspaceContentsPromise = Promise.resolve(
      workspace ? workspace.refreshContents(sessionId) : undefined
    ).catch((err) => {
      if (isCurrentSessionSelection(requestGeneration, sessionId)) showError(err);
    });
    // 已上传但未发送的文件保存在服务端，因此重新打开或切换窗口时必须恢复它们；
    // 否则文件虽在磁盘和数据库中，用户却无法发送或移除。
    if (changedSession || needsInitialDraftLoad) {
      await restoreStagedAttachments(sessionId, requestGeneration);
    }
    if (!isCurrentSessionSelection(requestGeneration, sessionId)) {
      await workspaceContentsPromise;
      return;
    }
    try {
      const detail = loadedDetail || await api.getSession(sessionId);
      if (!isCurrentSessionSelection(requestGeneration, sessionId)) return;
      sessionDetail.value = detail;
      sessionTitleDraft.value = sessionDetail.value?.session?.title || "";
      if (resetTurnState && runtimeTimeline) {
        await runtimeTimeline.load(sessionId);
      }
    } catch (err) {
      if (isCurrentSessionSelection(requestGeneration, sessionId)) showError(err);
    } finally {
      await workspaceContentsPromise;
    }
  }

  async function guardedSelectSession(sessionId) {
    if (saving.value || attachmentUploading.value) return false;
    const detailSessionId = sessionDetail.value?.session?.id || "";
    if (selectedSessionId.value === sessionId && detailSessionId === sessionId) return true;
    if (!confirmDiscardChanges()) return false;
    projects?.discardDraft();
    await selectSession(sessionId);
    return sessionDetail.value?.session?.id === sessionId;
  }

  // `folderId` 仅控制历史放置位置；工作目录由后端固定分配。
  async function createSession({ folderId = null } = {}) {
    if (!newSessionTitle.value.trim()) return;
    saving.value = true;
    clearMessage();
    try {
      const title = newSessionTitle.value.trim();
      let session;
      if (workspace) {
        session = await workspace.createFreeSession({ title, folderId });
      } else {
        session = (await api.createSession({
          title,
          ...(folderId ? { folder_id: folderId } : {})
        })).session;
      }
      if (!session) return;
      newSessionTitle.value = "";
      mode.value = "chat";
      sessionStatus.value = "active";
      sessionSearch.value = "";
      await loadSessions(session.id);
      notice.value = "会话已创建";
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  async function saveSessionTitle() {
    if (!selectedSession.value?.id || !sessionTitleDirty.value) return;
    saving.value = true;
    clearMessage();
    try {
      await api.patchSession(selectedSession.value.id, { title: sessionTitleDraft.value.trim() || "未命名会话" });
      await loadSessions(selectedSession.value.id);
      notice.value = "会话名已保存";
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  async function transitionSession(statusAction, message, { status = null, clearSearch = false } = {}) {
    if (!selectedSession.value) return;
    const id = selectedSession.value.id;
    saving.value = true;
    clearMessage();
    try {
      await api.patchSession(id, { status_action: statusAction });
      if (status) sessionStatus.value = status;
      if (clearSearch) sessionSearch.value = "";
      await loadSessions(status === "active" ? id : undefined);
      notice.value = message;
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  function archiveSession() {
    // 归档不像删除那样不可逆，所以问法也不同：只说清它会变成什么样，
    // 而不是吓唬人。真正需要"想清楚再点"的是彻底删除那一条。
    const title = selectedSession.value?.title || "未命名会话";
    if (!globalThis.confirm(`归档「${title}」？归档后不能再对话，可以随时取消归档。`)) return;
    return transitionSession("archive", "会话已归档");
  }
  const unarchiveSession = () => transitionSession("unarchive", "会话已恢复", { status: "active", clearSearch: true });
  const restoreTrashedSession = () => transitionSession("restore", "会话已从回收站恢复", { status: "active", clearSearch: true });

  async function renameSession(sessionId, title) {
    // 按 id 改名，不经过标题草稿：右键菜单可以作用在没被选中的会话上，
    // 而草稿只属于当前选中的那一个。
    const trimmed = String(title || "").trim() || "未命名会话";
    if (!sessionId) return;
    saving.value = true;
    clearMessage();
    try {
      await api.patchSession(sessionId, { title: trimmed });
      await loadSessions(selectedSessionId.value || sessionId);
      if (selectedSessionId.value === sessionId) sessionTitleDraft.value = trimmed;
      notice.value = "会话名已保存";
    } catch (err) { showError(err); } finally { saving.value = false; }
  }

  async function trashSession() {
    if (!selectedSession.value || !globalThis.confirm(`将会话「${selectedSession.value.title || "未命名会话"}」移入回收站？`)) return;
    await transitionSession("trash", "会话已移入回收站", { status: "trashed" });
  }

  async function sendChatWithStreamFallback(payload, { shouldPublish = () => true } = {}) {
    try {
      return await api.chatTurnStream(payload, (item) => {
        if (!shouldPublish()) return;
        if (item.event === "answer_start") streamingReply.value = "";
        if (item.event === "delta") streamingReply.value += String(item.data?.text || "");
        if (
          item.event === "answer_discard" ||
          item.event === "error" ||
          (item.event === "final" && item.data?.result?.status === "incomplete")
        ) streamingReply.value = "";
        if (item.event === "runtime_event" && item.data?.runtime_event) {
          if (runtimeTimeline) runtimeTimeline.ingest(item.data.runtime_event);
          else runtimeEvents.value = [...runtimeEvents.value.slice(-19), item.data.runtime_event];
          return;
        }
        if (!["answer_start", "delta", "answer_complete", "answer_discard"].includes(item.event)) {
          chatStreamEvents.value = [...chatStreamEvents.value.slice(-5), { event: item.event, at: clock(), detail: streamEventDetail(item) }];
        }
      });
    } catch (err) {
      if (err?.status !== 404 && err?.payload?.error?.code !== "NOT_FOUND") throw err;
      if (shouldPublish()) {
        chatStreamEvents.value = [{ event: "fallback", at: clock(), detail: "stream endpoint unavailable" }];
        streamingReply.value = "";
      }
      return api.chatTurn(payload);
    }
  }

  function makeClientRequestId() {
    if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID();
    return `turn-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  function requestMatches(candidate, sessionId, message, attachmentIds, runtimePolicy) {
    return Boolean(
      candidate &&
      candidate.sessionId === sessionId &&
      candidate.message === message &&
      candidate.attachmentIds.length === attachmentIds.length &&
      candidate.attachmentIds.every((attachmentId, index) => attachmentId === attachmentIds[index]) &&
      candidate.runtimePolicy?.schema_version === runtimePolicy.schema_version &&
      candidate.runtimePolicy?.l1_enabled === runtimePolicy.l1_enabled &&
      candidate.runtimePolicy?.l2_enabled === runtimePolicy.l2_enabled
    );
  }

  function getClientRequestId(sessionId, message, attachmentIds, runtimePolicy) {
    const previous = pendingClientRequest.value;
    if (requestMatches(previous, sessionId, message, attachmentIds, runtimePolicy)) {
      return previous.clientRequestId;
    }
    const clientRequestId = makeClientRequestId();
    pendingClientRequest.value = {
      sessionId,
      message,
      attachmentIds: [...attachmentIds],
      runtimePolicy: { ...runtimePolicy },
      clientRequestId
    };
    return clientRequestId;
  }

  async function restoreStagedAttachments(
    sessionId,
    requestGeneration = sessionSelectionGeneration
  ) {
    if (!api.listAttachments) return;
    try {
      const { attachments = [] } = await api.listAttachments(sessionId);
      if (!isCurrentSessionSelection(requestGeneration, sessionId)) return;
      pendingAttachments.value = attachments.map((item) => ({
        ...item,
        // 列表携带已存储事实；原则上，可读性应采用与上传响应相同的方式推导。
        readable: ["image", "text", "document"].includes(item.kind)
      }));
    } catch (err) {
      // 刷新暂存附件绝不能阻塞打开会话。
      if (isCurrentSessionSelection(requestGeneration, sessionId)) showError(err);
    }
  }

  async function attachFiles(files) {
    if (!files?.length) return;
    if (saving.value || attachmentUploading.value || chatBusy.value) return false;
    // 草稿阶段只保留 File 对象；首次发送前仍可修改目录，不能因选附件提前绑定。
    if (projects?.draftActive?.value) {
      const current = projects.draftAttachments.value;
      const room = MAX_ATTACHMENTS_PER_TURN - current.length;
      const accepted = Array.from(files).slice(0, room).map((file) => ({
        attachment_id: `draft-${globalThis.crypto.randomUUID()}`,
        original_name: file.name,
        name: file.name,
        size_bytes: file.size,
        file
      }));
      projects.setDraftAttachments([...current, ...accepted]);
      if (files.length > room) showError(new Error(`一条消息最多带 ${MAX_ATTACHMENTS_PER_TURN} 个附件。`));
      return files.length <= room;
    }
    const sessionId = selectedSessionId.value;
    if (!sessionId) return;

    // 与 store 中的 MAX_ATTACHMENTS_PER_TURN 保持一致。这里提前检查，拖入二十个文件时
    // 会在上传前说明哪些超额，而不是等到第十七个请求才发现上限。
    const room = MAX_ATTACHMENTS_PER_TURN - pendingAttachments.value.length;
    if (room <= 0) {
      showError(new Error(`一条消息最多带 ${MAX_ATTACHMENTS_PER_TURN} 个附件，请先移除一些。`));
      return false;
    }
    const accepted = Array.from(files).slice(0, room);
    const rejectedCount = files.length - accepted.length;

    const requestGeneration = ++attachmentUploadGeneration;
    attachmentUploading.value = true;
    clearMessage();
    const failures = [];
    try {
      for (const file of accepted) {
        try {
          const { attachment } = await api.uploadAttachment(sessionId, file);
          if (!isCurrentAttachmentUpload(requestGeneration, sessionId)) return false;
          // 立即上传、发送时才绑定，可以避免大文件阻塞消息，并允许用户事先移除。
          pendingAttachments.value = [...pendingAttachments.value, attachment];
        } catch (err) {
          // 一个文件被拒绝不能导致后续文件被静默丢弃；拖入的目录中包含单个超大文件
          // 是常见情况。
          failures.push({ name: file?.name || "文件", err });
        }
      }
      if (!isCurrentAttachmentUpload(requestGeneration, sessionId)) return false;
      if (rejectedCount > 0) {
        showError(new Error(
          `一条消息最多带 ${MAX_ATTACHMENTS_PER_TURN} 个附件；后 ${rejectedCount} 个未上传。`
        ));
      } else if (failures.length) {
        const names = failures.map((item) => item.name).join("、");
        showError(new Error(`${failures.length} 个文件未能上传：${names}`));
      }
      return failures.length === 0 && rejectedCount === 0;
    } catch (err) {
      if (isCurrentAttachmentUpload(requestGeneration, sessionId)) showError(err);
      return false;
    } finally {
      if (isCurrentAttachmentUpload(requestGeneration, sessionId)) {
        attachmentUploading.value = false;
      }
    }
  }

  async function removePendingAttachment(attachmentId) {
    if (saving.value || attachmentUploading.value || chatBusy.value) return;
    if (projects?.draftActive?.value) {
      projects.setDraftAttachments(projects.draftAttachments.value.filter(
        (item) => item.attachment_id !== attachmentId
      ));
      return;
    }
    const removed = pendingAttachments.value.find((item) => item.attachment_id === attachmentId);
    // 先从编辑器移除以即时响应点击，再从服务端丢弃；缺少后一步，文件会在下次打开
    // 会话时重新出现。
    pendingAttachments.value = pendingAttachments.value.filter(
      (item) => item.attachment_id !== attachmentId
    );
    if (removed?.file || !api.deleteAttachment || !selectedSessionId.value) return;
    try {
      await api.deleteAttachment(selectedSessionId.value, attachmentId);
    } catch (err) {
      showError(err);
    }
  }

  // 目录只在创建时选择；后续改名、配置变更均不能移动该 Session。
  async function materializeDraft(firstMessage = "") {
    const draft = projects?.draft?.value;
    if (!draft) return null;
    if (!draft.session) {
      const request = projects.prepareDraftCreation(initialDraftTitle(firstMessage));
      try {
        const created = await api.createSession(request);
        if (projects.draft.value !== draft) return null;
        if (!created?.session?.id) throw new Error("创建响应缺少会话信息，请重试确认结果");
        draft.session = created.session;
      } catch (err) {
        // 只有服务端确认未提交，才允许修改路径。传输失败仍重放冻结请求。
        if (projects.draft.value === draft && err?.error?.details?.creation_not_committed === true) {
          projects.resetDraftCreation();
        }
        throw err;
      }
    }
    const session = draft.session;
    // 精确详情加载成功前不切换选择、不清输入、不转交附件；列表刷新不是创建依赖。
    const detail = await api.getSession(session.id);
    if (projects.draft.value !== draft) return null;
    if (detail?.session?.id !== session.id) throw new Error("会话详情尚未确认，请重试");
    sessions.value = [detail.session, ...sessions.value.filter((item) => item.id !== session.id)];
    await selectSession(session.id, { loadedDetail: detail });
    if (projects.draft.value !== draft || selectedSession.value?.id !== session.id) return null;
    pendingAttachments.value = [...pendingAttachments.value, ...draft.attachments];
    chatInput.value = firstMessage;
    projects.discardDraft();
    if (session.working_dir) await projects.remember(session.working_dir);
    return session;
  }

  async function uploadPendingFiles(sessionId) {
    const requestGeneration = ++attachmentUploadGeneration;
    attachmentUploading.value = true;
    try {
      for (const item of pendingAttachments.value.filter((entry) => entry.file)) {
        const { attachment } = await api.uploadAttachment(sessionId, item.file);
        if (!isCurrentAttachmentUpload(requestGeneration, sessionId)) return false;
        pendingAttachments.value = pendingAttachments.value.map(
          (entry) => entry.attachment_id === item.attachment_id ? attachment : entry
        );
      }
      return true;
    } finally {
      if (isCurrentAttachmentUpload(requestGeneration, sessionId)) attachmentUploading.value = false;
    }
  }

  // 名字从这轮对话里取，而不是让用户在还什么都没说的时候先想一个。
  // 失败就保持原样——没名字的会话仍然能用，起名不该反过来影响这轮结果。
  async function autonameIfFirstTurn(sessionId, userText, assistantText) {
    const expectedTitle = sessions.value.find((item) => item.id === sessionId)?.title;
    try {
      const payload = await api.autonameSession(sessionId, {
        user_text: userText,
        assistant_text: String(assistantText || "")
      });
      const title = payload?.title;
      if (!title) return;
      const currentTitle = sessions.value.find((item) => item.id === sessionId)?.title;
      if (currentTitle !== expectedTitle) return;
      // 先问"用户改过标题吗"，再改 sessionDetail：脏不脏是拿输入框和它比出来的，
      // 顺序反了就永远读到脏，输入框再也追不上自动起的名字。
      const userEditedTitle = sessionTitleDirty.value;
      sessions.value = sessions.value.map(
        (item) => (item.id === sessionId ? { ...item, title } : item)
      );
      if (sessionDetail.value?.session?.id === sessionId) {
        sessionDetail.value = {
          ...sessionDetail.value,
          session: { ...sessionDetail.value.session, title }
        };
        if (!userEditedTitle) sessionTitleDraft.value = title;
      }
    } catch { /* 起名是装饰，不是这轮的交付物 */ }
  }

  async function sendChat({ answerPendingQuestion = false } = {}) {
    if (saving.value || chatBusy.value || attachmentUploading.value) return false;
    // 先把这句话抓在手里：草稿落库会切到新会话，切会话时输入框会被换成
    // 那个会话自己的草稿（新会话是空的），晚一步读就只剩空字符串了。
    const message = chatInput.value.trim();
    if (!message) return;
    // 任务模式已经可以被选择和展示，但前端尚未开放真实执行。守在所有发送路径
    // 汇合的动作边界，避免按钮、Enter、待回答问题或程序调用中的任意一条漏发。
    // 这也必须早于草稿会话落库：被阻止的尝试不应产生任何持久化副作用。
    if (runtimeMode.value === RUNTIME_MODE_TASK) {
      clearMessage();
      notice.value = TASK_MODE_SEND_BLOCKED_NOTICE;
      return false;
    }
    // 在草稿实体化前捕获：加载新建 Session 可能刷新 UI 默认值，但本次 AcceptedTurn
    // 必须保留用户按下发送时选择的模式。
    const runtimePolicy = runtimePolicyForMode(runtimeMode.value);
    if (projects?.draftActive?.value) {
      saving.value = true;
      try {
        if (!(await materializeDraft(message))) return false;
      } catch (err) { showError(err); return false; } finally { saving.value = false; }
    }
    const allowed = answerPendingQuestion
      ? canAnswerPendingQuestion.value
      : canSendChat.value;
    if (!selectedSession.value || !allowed) return;
    const sessionId = selectedSession.value.id;
    if (selectedSessionId.value !== sessionId) return;
    const requestGeneration = ++chatSendGeneration;
    const isCurrentRequest = () => isCurrentChatSend(requestGeneration, sessionId);
    chatInput.value = "";
    chatBusy.value = true;
    // 传输结果不明时的重试复用客户端请求 ID；消息、附件集合或 Runtime 模式发生变化时，
    // 则刻意创建新的 Turn。
    lastTurnResult.value = null;
    clearInSessionTaskDetails();
    chatStreamEvents.value = [];
    if (runtimeTimeline) runtimeTimeline.clear();
    else runtimeEvents.value = [];
    streamingReply.value = "";
    clearMessage();
    try {
      if (!(await uploadPendingFiles(sessionId))) return false;
      const attachmentIds = pendingAttachments.value.map((item) => item.attachment_id);
      const hadPriorTurns = Boolean(sessionDetail.value?.turns?.length);
      const payload = await sendChatWithStreamFallback({
        session_id: sessionId,
        message,
        attachment_ids: attachmentIds,
        runtime_policy: runtimePolicy,
        client_request_id: getClientRequestId(
          sessionId,
          message,
          attachmentIds,
          runtimePolicy
        )
      }, { shouldPublish: isCurrentRequest });
      if (!isCurrentRequest()) return false;
      lastTurnResult.value = payload.result || null;
      // Task 卡片只是普通回复的尽力而为只读装饰，刻意不影响 Turn 交付、重试或编辑器。
      void loadRelatedInSessionTaskDetails(
        sessionId,
        lastTurnResult.value?.related_insession_task_ids
      );
      pendingClientRequest.value = null;
      drafts.clear(sessionId);
      // 仅在发送成功后清空；Turn 失败时继续暂存文件，让用户无需重新上传即可重试。
      pendingAttachments.value = [];
      const wasFirstTurn = !hadPriorTurns;
      await selectSession(sessionId, { resetTurnState: false });
      if (!isCurrentRequest()) return false;
      streamingReply.value = "";
      if (wasFirstTurn) void autonameIfFirstTurn(sessionId, message, payload.result?.reply);
      notice.value = lastTurnResult.value?.status === "incomplete" ? "本轮未完成，已记录中断状态" : "回复已返回";
      return true;
    } catch (err) {
      if (!isCurrentRequest()) return false;
      showError(err);
      streamingReply.value = "";
      chatInput.value = message;
      return false;
    } finally {
      if (isCurrentRequest()) chatBusy.value = false;
    }
  }

  const sendPendingUserAnswer = () => sendChat({ answerPendingQuestion: true });

  // 光标位置插一个换行。Shift+Enter 浏览器自己会插，但 Ctrl/Cmd+Enter 在 textarea
  // 里默认什么也不做，所以那两个组合得自己来。
  function insertNewlineAtCursor(element) {
    const text = chatInput.value ?? "";
    const start = element?.selectionStart ?? text.length;
    const end = element?.selectionEnd ?? start;
    chatInput.value = `${text.slice(0, start)}\n${text.slice(end)}`;
    queueMicrotask(() => {
      if (!element) return;
      // v-model 回写之后再放光标，否则会被覆盖回原位
      element.selectionStart = element.selectionEnd = start + 1;
    });
  }

  function handleChatComposerKeydown(event) {
    if (event.key !== "Enter") return;

    // 输入法组词过程中的 Enter 是在选字，不是发送。中文输入下不判这个会非常难用：
    // 打一半拼音按回车选词，消息就发出去了。
    if (event.isComposing || event.keyCode === 229) return;

    // Ctrl / Cmd / Shift + Enter 换行
    if (event.ctrlKey || event.metaKey || event.shiftKey) {
      if (event.shiftKey && !event.ctrlKey && !event.metaKey) return; // 浏览器默认就是换行
      event.preventDefault();
      insertNewlineAtCursor(event.target);
      return;
    }

    // 单独的 Enter 发送
    event.preventDefault();
    if (chatBusy.value || !chatInput.value.trim()) return;
    if (canAnswerPendingQuestion.value) sendPendingUserAnswer();
    else if (canSendChat.value) sendChat();
  }

  async function reconcileRuntimeState() {
    const sessionId = selectedSessionId.value;
    if (!sessionId || !runtimeTimeline) return { enabled: false, events: [] };
    const reconcile = runtimeTimeline.reconcile || runtimeTimeline.load;
    if (typeof reconcile !== "function") return { enabled: false, events: [] };
    return reconcile.call(runtimeTimeline, sessionId);
  }

  return {
    renameSession,
    loadSessions, reloadSessionsFromSearch, guardedReloadSessionsFromSearch, clearSessionSearch,
    guardedClearSessionSearch, guardedSetSessionStatus, selectSession, guardedSelectSession, createSession,
    saveSessionTitle, archiveSession, unarchiveSession, trashSession, restoreTrashedSession, sendChat,
    sendPendingUserAnswer,
    materializeDraft,
    attachFiles, removePendingAttachment, restoreStagedAttachments,
    handleChatComposerKeydown, sendChatWithStreamFallback, reconcileRuntimeState
  };
}
