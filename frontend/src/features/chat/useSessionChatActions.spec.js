import { computed, ref } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useProjects } from "../workspace/useProjects";
import {
  TASK_MODE_SEND_BLOCKED_NOTICE,
  useSessionChatActions
} from "./useSessionChatActions";

afterEach(() => vi.unstubAllGlobals());

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

function harness(
  api,
  session = { id: "s1", status: "active" },
  runtimeTimeline = null,
  actionOverrides = {}
) {
  const state = {
    mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""),
    sessionStatus: ref("active"), sessionSearch: ref(""), sessions: ref([session]), selectedSessionId: ref(session.id),
    sessionDetail: ref({ session, turns: [] }), sessionTitleDraft: ref(session.title || ""), newSessionTitle: ref(""),
    chatInput: ref(""), chatBusy: ref(false), chatStreamEvents: ref([]), runtimeEvents: ref([]), streamingReply: ref(""), lastTurnResult: ref(null),
    pendingClientRequest: ref(null),
    pendingAttachments: ref([]), attachmentUploading: ref(false)
  };
  const drafts = { load: vi.fn(() => ""), save: vi.fn(), clear: vi.fn() };
  const showError = vi.fn();
  const selectedSession = computed(() => state.sessionDetail.value?.session || null);
  const actions = useSessionChatActions({
    api,
    state,
    selectedSession,
    sessionTitleDirty: ref(false),
    canSendChat: ref(true),
    drafts,
    confirmDiscardChanges: () => true,
    clearMessage: vi.fn(),
    showError,
    loadReferenceChoices: vi.fn(),
    runtimeTimeline,
    clock: () => "12:00",
    ...actionOverrides
  });
  return { actions, state, drafts, showError };
}

describe("useSessionChatActions", () => {
  it("does not rename an uncreated Session draft", async () => {
    const api = { patchSession: vi.fn() };
    const { actions } = harness(api, { id: "", title: "新会话" }, null, {
      sessionTitleDirty: ref(true)
    });

    await actions.saveSessionTitle();

    expect(api.patchSession).not.toHaveBeenCalled();
  });

  it("does not publish a late stream or successful Turn into another Session", async () => {
    const sessionA = { id: "s1", status: "active" };
    const sessionB = { id: "s2", status: "active" };
    const turnA = deferred();
    let publishAEvent;
    const api = {
      chatTurnStream: vi.fn((_payload, onEvent) => {
        publishAEvent = onEvent;
        return turnA.promise;
      }),
      getSession: vi.fn(async (sessionId) => ({
        session: sessionId === sessionA.id ? sessionA : sessionB,
        turns: []
      }))
    };
    const { actions, state } = harness(api, sessionA);
    state.chatInput.value = "message for A";

    const sendingA = actions.sendChat();
    await actions.selectSession(sessionB.id);
    state.pendingAttachments.value = [{ attachment_id: "b-file", name: "b.txt" }];

    publishAEvent({ event: "delta", data: { text: "stale A stream" } });
    publishAEvent({
      event: "runtime_event",
      data: { runtime_event: { event_id: "stale-a-runtime" } }
    });
    turnA.resolve({ result: { status: "completed", reply: "stale A result" } });
    await sendingA;

    expect(state.selectedSessionId.value).toBe(sessionB.id);
    expect(state.sessionDetail.value?.session?.id).toBe(sessionB.id);
    expect(state.streamingReply.value).toBe("");
    expect(state.runtimeEvents.value).toEqual([]);
    expect(state.lastTurnResult.value).toBeNull();
    expect(state.pendingAttachments.value).toEqual([{ attachment_id: "b-file", name: "b.txt" }]);
    expect(state.notice.value).toBe("");
    expect(state.chatBusy.value).toBe(false);
  });

  it("does not let a stale send failure restore input or settle a newer send", async () => {
    const sessionA = { id: "s1", status: "active" };
    const sessionB = { id: "s2", status: "active" };
    const turnA = deferred();
    const turnB = deferred();
    const api = {
      chatTurnStream: vi.fn((payload) => (
        payload.session_id === sessionA.id ? turnA.promise : turnB.promise
      )),
      getSession: vi.fn(async (sessionId) => ({
        session: sessionId === sessionA.id ? sessionA : sessionB,
        turns: []
      }))
    };
    const { actions, state, showError } = harness(api, sessionA);
    state.chatInput.value = "message for A";
    const sendingA = actions.sendChat();

    await actions.selectSession(sessionB.id);
    state.chatInput.value = "message for B";
    const sendingB = actions.sendChat();
    expect(state.chatBusy.value).toBe(true);

    turnA.reject(new Error("late A failure"));
    await sendingA;

    expect(state.selectedSessionId.value).toBe(sessionB.id);
    expect(state.chatInput.value).toBe("");
    expect(state.chatBusy.value).toBe(true);
    expect(showError).not.toHaveBeenCalled();

    turnB.resolve({ result: { status: "completed", reply: "current B result" } });
    await sendingB;
    expect(state.lastTurnResult.value?.reply).toBe("current B result");
    expect(state.chatBusy.value).toBe(false);
  });

  it("does not append a stale upload or let it settle the current Session upload", async () => {
    const sessionA = { id: "s1", status: "active" };
    const sessionB = { id: "s2", status: "active" };
    const uploadA = deferred();
    const uploadB = deferred();
    const api = {
      uploadAttachment: vi.fn((sessionId) => (
        sessionId === sessionA.id ? uploadA.promise : uploadB.promise
      )),
      getSession: vi.fn(async (sessionId) => ({
        session: sessionId === sessionA.id ? sessionA : sessionB,
        turns: []
      }))
    };
    const { actions, state } = harness(api, sessionA);

    const attachingA = actions.attachFiles([{ name: "a.txt" }]);
    await actions.selectSession(sessionB.id);
    const attachingB = actions.attachFiles([{ name: "b.txt" }]);
    expect(state.attachmentUploading.value).toBe(true);

    uploadA.resolve({ attachment: { attachment_id: "a-file", name: "a.txt" } });
    await attachingA;

    expect(state.pendingAttachments.value).toEqual([]);
    expect(state.attachmentUploading.value).toBe(true);

    uploadB.resolve({ attachment: { attachment_id: "b-file", name: "b.txt" } });
    await attachingB;
    expect(state.pendingAttachments.value).toEqual([{ attachment_id: "b-file", name: "b.txt" }]);
    expect(state.attachmentUploading.value).toBe(false);
  });

  it("does not let a pending Session-list refresh undo a manual selection", async () => {
    const sessionA = { id: "s1", status: "active" };
    const sessionB = { id: "s2", status: "active" };
    const sessionList = deferred();
    const api = {
      listSessions: vi.fn(() => sessionList.promise),
      getSession: vi.fn(async (sessionId) => ({
        session: sessionId === sessionA.id ? sessionA : sessionB,
        turns: []
      }))
    };
    const { actions, state } = harness(api, sessionA);
    state.sessions.value = [sessionA, sessionB];

    const refreshing = actions.loadSessions();
    await actions.selectSession(sessionB.id);
    sessionList.resolve({ sessions: [sessionA, sessionB] });
    await refreshing;

    expect(state.selectedSessionId.value).toBe(sessionB.id);
    expect(state.sessionDetail.value?.session?.id).toBe(sessionB.id);
  });

  it("lets only the latest Session-list search publish or settle loading", async () => {
    const oldList = deferred();
    const currentList = deferred();
    const baselineSession = { id: "baseline", status: "active" };
    const oldSession = { id: "old", status: "active" };
    const currentSession = { id: "current", status: "active" };
    const api = {
      listSessions: vi.fn()
        .mockReturnValueOnce(oldList.promise)
        .mockReturnValueOnce(currentList.promise),
      getSession: vi.fn(async (sessionId) => ({
        session: sessionId === oldSession.id ? oldSession : currentSession,
        turns: []
      }))
    };
    const { actions, state } = harness(api, baselineSession);

    state.sessionSearch.value = "old query";
    const oldRefresh = actions.loadSessions("");
    state.sessionSearch.value = "current query";
    const currentRefresh = actions.loadSessions("");

    oldList.resolve({ sessions: [oldSession] });
    await oldRefresh;
    expect(state.sessions.value).toEqual([baselineSession]);
    expect(state.loading.value).toBe(true);

    currentList.resolve({ sessions: [currentSession] });
    await currentRefresh;
    expect(state.sessions.value).toEqual([currentSession]);
    expect(state.selectedSessionId.value).toBe(currentSession.id);
    expect(state.loading.value).toBe(false);
  });

  it("switches chat authority when workspace creation preselects the new shared Session id", async () => {
    const sessionA = { id: "s1", status: "active" };
    const sessionB = { id: "s2", status: "active" };
    let state;
    const workspace = {
      createFreeSession: vi.fn(async () => {
        state.selectedSessionId.value = sessionB.id;
        state.sessions.value = [sessionA, sessionB];
        return sessionB;
      }),
      refresh: vi.fn().mockResolvedValue(undefined),
      selectSession: vi.fn((sessionId) => { state.selectedSessionId.value = sessionId; }),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const api = {
      getSession: vi.fn().mockResolvedValue({ session: sessionB, turns: [] }),
      listAttachments: vi.fn().mockResolvedValue({
        attachments: [{ attachment_id: "b-file", name: "b.txt", kind: "text" }]
      })
    };
    const result = harness(api, sessionA, null, { workspace });
    ({ state } = result);
    state.newSessionTitle.value = "new B";
    state.chatInput.value = "draft for A";
    state.pendingAttachments.value = [{ attachment_id: "a-file", name: "a.txt" }];
    state.pendingClientRequest.value = { sessionId: sessionA.id, clientRequestId: "request-a" };

    await result.actions.createSession();

    expect(result.drafts.save).toHaveBeenCalledWith(sessionA.id, "draft for A");
    expect(result.drafts.load).toHaveBeenCalledWith(sessionB.id);
    expect(state.sessionDetail.value?.session?.id).toBe(sessionB.id);
    expect(state.chatInput.value).toBe("");
    expect(state.pendingAttachments.value.map((item) => item.attachment_id)).toEqual(["b-file"]);
    expect(state.pendingClientRequest.value).toBeNull();
  });

  it("submits a pending-question response through the ordinary chat endpoint", async () => {
    const session = { id: "s1", status: "active" };
    const api = {
      chatTurnStream: vi.fn().mockResolvedValue({
        result: { status: "completed", reply: "已记录" }
      }),
      getSession: vi.fn().mockResolvedValue({
        session,
        turns: [],
        pending_user_questions: [{
          insession_task_id: "task-trip",
          question: "哪天出发？"
        }]
      })
    };
    const { actions, state } = harness(api, session, null, {
      canSendChat: ref(false),
      canAnswerPendingQuestion: ref(true)
    });
    state.chatInput.value = "20 号";

    await actions.sendPendingUserAnswer();

    expect(api.chatTurnStream).toHaveBeenCalledOnce();
    expect(api.chatTurnStream.mock.calls[0][0]).toMatchObject({
      session_id: "s1",
      message: "20 号"
    });
    expect(state.sessionDetail.value.pending_user_questions).toHaveLength(1);
  });

  it("falls back only when the stream endpoint is missing", async () => {
    const api = {
      chatTurnStream: vi.fn().mockRejectedValue({ status: 404 }),
      chatTurn: vi.fn().mockResolvedValue({ result: { response: "ok" } })
    };
    const { actions, state } = harness(api);
    const result = await actions.sendChatWithStreamFallback({ message: "hi" });
    expect(result.result.response).toBe("ok");
    expect(api.chatTurn).toHaveBeenCalledOnce();
    expect(state.chatStreamEvents.value[0].event).toBe("fallback");
  });

  it("does not hide non-404 stream failures behind fallback", async () => {
    const api = { chatTurnStream: vi.fn().mockRejectedValue({ status: 500 }), chatTurn: vi.fn() };
    const { actions } = harness(api);
    await expect(actions.sendChatWithStreamFallback({ message: "hi" })).rejects.toEqual({ status: 500 });
    expect(api.chatTurn).not.toHaveBeenCalled();
  });

  it("reuses a client request id when a transport failure leaves acceptance ambiguous", async () => {
    const session = { id: "s1", status: "active" };
    const api = {
      chatTurnStream: vi.fn()
        .mockRejectedValueOnce(new Error("connection lost"))
        .mockResolvedValueOnce({ result: { status: "completed", reply: "已完成" } }),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const { actions, state } = harness(api, session);
    state.chatInput.value = "继续这个任务";

    await actions.sendChat();
    await actions.sendChat();

    expect(api.chatTurnStream).toHaveBeenCalledTimes(2);
    expect(api.chatTurnStream.mock.calls[0][0].client_request_id).toEqual(
      api.chatTurnStream.mock.calls[1][0].client_request_id
    );
    expect(api.chatTurnStream.mock.calls[0][0].runtime_policy).toEqual({
      schema_version: 1,
      l1_enabled: true,
      l2_enabled: false
    });
    expect(state.pendingClientRequest.value).toBeNull();
  });

  it("creates a new request identity when the Runtime policy changes", async () => {
    const session = { id: "s1", status: "active" };
    const runtimeMode = ref("turn");
    const api = {
      chatTurnStream: vi.fn().mockRejectedValue(new Error("connection lost")),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const { actions, state } = harness(api, session, null, { runtimeMode });
    state.chatInput.value = "检查当前目录";

    await actions.sendChat();
    runtimeMode.value = "direct";
    await actions.sendChat();

    const [first, second] = api.chatTurnStream.mock.calls.map(([payload]) => payload);
    expect(first.client_request_id).not.toBe(second.client_request_id);
    expect(first.runtime_policy).toMatchObject({ l1_enabled: true, l2_enabled: false });
    expect(second.runtime_policy).toMatchObject({ l1_enabled: false, l2_enabled: false });
  });

  it("blocks every task-mode send before creating a Session or calling the Runtime", async () => {
    const runtimeMode = ref("task");
    const api = {
      createSession: vi.fn(),
      chatTurnStream: vi.fn(),
      chatTurn: vi.fn()
    };
    const projects = {
      draftActive: ref(true)
    };
    const { actions, state } = harness(api, undefined, null, { runtimeMode, projects });
    state.chatInput.value = "执行这个长期任务";
    state.pendingAttachments.value = [{ attachment_id: "att-1", name: "plan.md" }];
    state.pendingClientRequest.value = { sessionId: "s1", clientRequestId: "keep-me" };

    const result = await actions.sendChat();

    expect(result).toBe(false);
    expect(state.notice.value).toBe(TASK_MODE_SEND_BLOCKED_NOTICE);
    expect(state.chatInput.value).toBe("执行这个长期任务");
    expect(state.pendingAttachments.value).toEqual([{ attachment_id: "att-1", name: "plan.md" }]);
    expect(state.pendingClientRequest.value).toEqual({ sessionId: "s1", clientRequestId: "keep-me" });
    expect(state.chatBusy.value).toBe(false);
    expect(api.createSession).not.toHaveBeenCalled();
    expect(api.chatTurnStream).not.toHaveBeenCalled();
    expect(api.chatTurn).not.toHaveBeenCalled();

    const keyboardEvent = {
      key: "Enter",
      isComposing: false,
      keyCode: 13,
      ctrlKey: false,
      metaKey: false,
      shiftKey: false,
      preventDefault: vi.fn()
    };
    actions.handleChatComposerKeydown(keyboardEvent);
    expect(keyboardEvent.preventDefault).toHaveBeenCalledOnce();
    expect(state.notice.value).toBe(TASK_MODE_SEND_BLOCKED_NOTICE);
    expect(api.chatTurnStream).not.toHaveBeenCalled();

    await actions.sendPendingUserAnswer();
    expect(state.notice.value).toBe(TASK_MODE_SEND_BLOCKED_NOTICE);
    expect(api.chatTurnStream).not.toHaveBeenCalled();
  });


  it("renders delta text incrementally without filling the status-event rail", async () => {
    const api = {
      chatTurnStream: vi.fn(async (_payload, onEvent) => {
        onEvent({ event: "answer_start", data: { generation_id: "g1" } });
        onEvent({ event: "delta", data: { generation_id: "g1", text: "逐字" } });
        onEvent({ event: "delta", data: { generation_id: "g1", text: "回复" } });
        onEvent({ event: "answer_complete", data: { generation_id: "g1" } });
        return { result: { response: "逐字回复" } };
      })
    };
    const { actions, state } = harness(api);
    await actions.sendChatWithStreamFallback({ message: "hi" });
    expect(state.streamingReply.value).toBe("逐字回复");
    expect(state.chatStreamEvents.value).toEqual([]);
  });

  it("replaces a non-executable preview when the same turn restarts its control generation", async () => {
    const api = {
      chatTurnStream: vi.fn(async (_payload, onEvent) => {
        onEvent({ event: "answer_start", data: { generation_id: "turn-1:0" } });
        onEvent({ event: "delta", data: { generation_id: "turn-1:0", text: "动作预览：将读取记忆" } });
        onEvent({ event: "answer_complete", data: { generation_id: "turn-1:0" } });

        // 运行时会将预览视为控制输出而拒绝，再重试同一回合；其 JSON 工具命令会在上游被刻意丢弃。
        onEvent({ event: "answer_start", data: { generation_id: "turn-1:0" } });
        onEvent({ event: "answer_discard", data: { generation_id: "turn-1:0" } });

        onEvent({ event: "answer_start", data: { generation_id: "turn-1:1" } });
        onEvent({ event: "delta", data: { generation_id: "turn-1:1", text: "资料已读取。" } });
        onEvent({ event: "answer_complete", data: { generation_id: "turn-1:1" } });
        return { result: { response: "资料已读取。" } };
      })
    };
    const { actions, state } = harness(api);

    await actions.sendChatWithStreamFallback({ message: "读取我的记忆" });

    expect(state.streamingReply.value).toBe("资料已读取。");
    expect(state.streamingReply.value).not.toContain("动作预览");
    expect(state.chatStreamEvents.value).toEqual([]);
  });

  it("keeps typed runtime events separate from answer and generic status events", async () => {
    const runtimeEvent = {
      schema_version: 1, event_id: "evt_1", sequence: 1, session_id: "s1", turn_id: "turn_1",
      stage: "TOOL", status: "started", occurred_at: "2026-07-14T00:00:00Z", prompt_replay: false,
      operation_id: "operation_1"
    };
    const api = {
      chatTurnStream: vi.fn(async (_payload, onEvent) => {
        onEvent({ event: "runtime_event", data: { runtime_event: runtimeEvent } });
        return { result: { response: "ok" } };
      })
    };
    const { actions, state } = harness(api);
    await actions.sendChatWithStreamFallback({ message: "hi" });
    expect(state.runtimeEvents.value).toEqual([runtimeEvent]);
    expect(state.chatStreamEvents.value).toEqual([]);
  });

  it("clears a streamed draft and records only the incomplete Turn result", async () => {
    const session = { id: "s1", status: "active" };
    const result = { status: "incomplete", end_reason: "provider_unavailable", error_code: "MODEL_TIMEOUT" };
    const api = {
      chatTurnStream: vi.fn(async (_payload, onEvent) => {
        onEvent({ event: "answer_start", data: { generation_id: "g1" } });
        onEvent({ event: "delta", data: { generation_id: "g1", text: "不应保留的草稿" } });
        onEvent({ event: "final", data: { result } });
        return { result };
      }),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const { actions, state } = harness(api, session);
    state.chatInput.value = "查询状态";

    await actions.sendChat();

    expect(state.streamingReply.value).toBe("");
    expect(state.lastTurnResult.value).toEqual(result);
    expect(state.notice.value).toContain("未完成");
  });

  it("starts related in-session task reads without holding the completed Turn", async () => {
    const session = { id: "s1", status: "active" };
    const loadRelatedInSessionTaskDetails = vi.fn().mockResolvedValue([]);
    const api = {
      chatTurnStream: vi.fn().mockResolvedValue({
        result: {
          status: "completed",
          reply: "已返回",
          related_insession_task_ids: ["insession_task_alpha"]
        }
      }),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const { actions, state } = harness(api, session, null, { loadRelatedInSessionTaskDetails });
    state.chatInput.value = "继续整理";

    await actions.sendChat();

    expect(loadRelatedInSessionTaskDetails).toHaveBeenCalledWith("s1", ["insession_task_alpha"]);
    expect(state.chatBusy.value).toBe(false);
    expect(state.notice.value).toBe("回复已返回");
  });

  it("does not trash a session when confirmation is rejected", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false));
    const api = { patchSession: vi.fn() };
    const { actions } = harness(api);
    await actions.trashSession();
    expect(api.patchSession).not.toHaveBeenCalled();
  });



  it("does not carry staged attachments into another session", async () => {
    const api = {
      getSession: vi.fn().mockResolvedValue({
        session: { id: "s2", status: "active" },
        turns: []
      })
    };
    const { actions, state } = harness(api);
    state.pendingAttachments.value = [{ attachment_id: "att_1", name: "a.png" }];

    await actions.selectSession("s2");

    // 附件属于上传时所在的会话；若将其带入新会话，就会把文件附到用户从未选择的对话中。
    expect(state.pendingAttachments.value).toEqual([]);
  });

  it("switches and refreshes the workspace scope before session detail can fail", async () => {
    const detail = deferred();
    const api = { getSession: vi.fn(() => detail.promise) };
    const workspace = {
      selectSession: vi.fn(),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const { actions, state, drafts, showError } = harness(api, undefined, null, { workspace });
    state.sessionTitleDraft.value = "旧会话标题";
    state.chatInput.value = "旧会话草稿";
    state.lastTurnResult.value = { status: "completed", reply: "旧回复" };
    state.chatStreamEvents.value = [{ event: "old" }];
    state.runtimeEvents.value = [{ event_id: "old" }];
    state.streamingReply.value = "旧流式回复";

    const switching = actions.selectSession("s2");

    expect(workspace.selectSession).toHaveBeenCalledWith("s2");
    expect(workspace.refreshContents).toHaveBeenCalledWith("s2");
    expect(state.selectedSessionId.value).toBe("s2");
    expect(state.sessionDetail.value).toBeNull();
    expect(state.sessionTitleDraft.value).toBe("");
    expect(state.chatInput.value).toBe("");
    expect(state.lastTurnResult.value).toBeNull();
    expect(state.chatStreamEvents.value).toEqual([]);
    expect(state.runtimeEvents.value).toEqual([]);
    expect(state.streamingReply.value).toBe("");
    expect(drafts.save).toHaveBeenCalledWith("s1", "旧会话草稿");
    expect(drafts.load).toHaveBeenCalledWith("s2");

    await Promise.resolve();
    detail.reject(new Error("session detail unavailable"));
    await switching;

    expect(state.sessionDetail.value).toBeNull();
    expect(state.sessionTitleDraft.value).toBe("");
    expect(showError).toHaveBeenCalledWith(expect.objectContaining({
      message: "session detail unavailable"
    }));
  });

  it("does not let an older session detail request overwrite the latest selection", async () => {
    const formerDetail = deferred();
    const currentDetail = deferred();
    const api = {
      getSession: vi.fn((sessionId) => (
        sessionId === "s2" ? formerDetail.promise : currentDetail.promise
      ))
    };
    const workspace = {
      selectSession: vi.fn(),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const { actions, state } = harness(api, undefined, null, { workspace });

    const formerSwitch = actions.selectSession("s2");
    const currentSwitch = actions.selectSession("s3");
    currentDetail.resolve({ session: { id: "s3", title: "当前会话" }, turns: [] });
    await currentSwitch;
    formerDetail.resolve({ session: { id: "s2", title: "旧会话" }, turns: [] });
    await formerSwitch;

    expect(state.selectedSessionId.value).toBe("s3");
    expect(state.sessionDetail.value.session.id).toBe("s3");
    expect(state.sessionTitleDraft.value).toBe("当前会话");
  });

  it("uses a request generation so an A-B-A switch rejects the oldest A detail", async () => {
    const olderA = deferred();
    const sessionB = deferred();
    const currentA = deferred();
    const api = {
      getSession: vi.fn()
        .mockImplementationOnce(() => olderA.promise)
        .mockImplementationOnce(() => sessionB.promise)
        .mockImplementationOnce(() => currentA.promise)
    };
    const workspace = {
      selectSession: vi.fn(),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const initialA = {
      id: "s1",
      title: "初始 A",
      status: "active"
    };
    const { actions, state } = harness(api, initialA, null, { workspace });

    const olderASelection = actions.selectSession("s1");
    const bSelection = actions.selectSession("s2");
    await Promise.resolve();
    const currentASelection = actions.selectSession("s1");
    await Promise.resolve();
    expect(api.getSession).toHaveBeenCalledTimes(3);

    currentA.resolve({ session: { ...initialA, title: "当前 A" }, turns: [] });
    await currentASelection;
    sessionB.resolve({ session: { id: "s2", title: "B" }, turns: [] });
    await bSelection;
    olderA.resolve({ session: { ...initialA, title: "过期 A" }, turns: [] });
    await olderASelection;

    expect(state.selectedSessionId.value).toBe("s1");
    expect(state.sessionDetail.value.session.title).toBe("当前 A");
    expect(state.sessionTitleDraft.value).toBe("当前 A");
  });

  it("allows the selected sidebar item to retry when its detail is still absent", async () => {
    const session = { id: "s2", title: "重试成功", status: "active" };
    const api = {
      getSession: vi.fn()
        .mockRejectedValueOnce(new Error("temporary detail failure"))
        .mockResolvedValueOnce({ session, turns: [] })
    };
    const workspace = {
      selectSession: vi.fn(),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const { actions, state } = harness(api, undefined, null, { workspace });

    await actions.selectSession(session.id);
    expect(state.selectedSessionId.value).toBe(session.id);
    expect(state.sessionDetail.value).toBeNull();

    await actions.guardedSelectSession(session.id);

    expect(api.getSession).toHaveBeenCalledTimes(2);
    expect(state.sessionDetail.value?.session?.id).toBe(session.id);
  });

  it("does not expose a post-creation workspace binding action", () => {
    const result = harness({}, { id: "s1", status: "active", working_dir: "/workspace/fixed" });

    expect(result.actions).not.toHaveProperty("bindWorkingDirectory");
  });

  it("stages an upload immediately and keeps it until the turn succeeds", async () => {
    const session = { id: "s1", status: "active" };
    const api = {
      uploadAttachment: vi.fn().mockResolvedValue({
        attachment: { attachment_id: "att_1", name: "shot.png", size_bytes: 10, kind: "image", readable: true }
      }),
      chatTurn: vi.fn().mockRejectedValue(new Error("provider down")),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const { actions, state } = harness(api, session);

    await actions.attachFiles([{ name: "shot.png", type: "image/png" }]);
    expect(api.uploadAttachment).toHaveBeenCalledTimes(1);
    expect(state.pendingAttachments.value).toHaveLength(1);

    state.chatInput.value = "看看这个";
    await actions.sendChat();

    // 发送失败不能丢弃已上传文件，用户应能直接重试而无需再次选择文件。
    expect(state.pendingAttachments.value).toHaveLength(1);
  });

  it("discards a staged attachment on the server, not only in the composer", async () => {
    const api = { deleteAttachment: vi.fn().mockResolvedValue({ deleted: true }) };
    const { actions, state } = harness(api);
    state.pendingAttachments.value = [
      { attachment_id: "att_1", name: "a.png" },
      { attachment_id: "att_2", name: "b.png" }
    ];

    await actions.removePendingAttachment("att_1");

    expect(state.pendingAttachments.value.map((item) => item.attachment_id)).toEqual(["att_2"]);
    // 若只在本地移除，文件会在下次加载会话时重新出现。
    expect(api.deleteAttachment).toHaveBeenCalledWith("s1", "att_1");
  });

  it("recovers uploads that were staged but never sent", async () => {
    const session = { id: "s1", status: "active" };
    const api = {
      getSession: vi.fn().mockResolvedValue({ session, turns: [] }),
      listAttachments: vi.fn().mockResolvedValue({
        attachments: [
          { attachment_id: "att_1", name: "left.png", size_bytes: 10, kind: "image" },
          { attachment_id: "att_2", name: "left.mp3", size_bytes: 10, kind: "audio" }
        ]
      })
    };
    const { actions, state } = harness(api, session);
    state.selectedSessionId.value = null;
    state.sessionDetail.value = null;

    await actions.selectSession("s1");

    expect(api.listAttachments).toHaveBeenCalledWith("s1");
    expect(state.pendingAttachments.value.map((item) => item.name)).toEqual(["left.png", "left.mp3"]);
    // 重新推导可读性，确保编辑器以同样方式标记音频。
    expect(state.pendingAttachments.value.map((item) => item.readable)).toEqual([true, false]);
  });

  it("keeps the current composer draft when refreshing the same session", async () => {
    const session = { id: "s1", status: "active" };
    const api = { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) };
    const { actions, state, drafts } = harness(api, session);
    state.chatInput.value = "不要在归档移动时丢失我正在写的内容";

    await actions.selectSession("s1");

    expect(state.chatInput.value).toBe("不要在归档移动时丢失我正在写的内容");
    expect(drafts.load).not.toHaveBeenCalled();
  });

  it("catches up the public event timeline without invoking a control endpoint", async () => {
    const api = { chatTurnStream: vi.fn() };
    const runtimeTimeline = {
      reconcile: vi.fn().mockResolvedValue({ enabled: true, events: [] })
    };
    const { actions } = harness(api, undefined, runtimeTimeline);

    await actions.reconcileRuntimeState();

    expect(runtimeTimeline.reconcile).toHaveBeenCalledWith("s1");
    expect(api.chatTurnStream).not.toHaveBeenCalled();
  });

  // --- 一次拖进来很多个 -----------------------------------------------------------

  it("stops at the per-message ceiling and says which files did not go", async () => {
    const api = {
      uploadAttachment: vi.fn((_sessionId, file) => Promise.resolve({
        attachment: { attachment_id: file.name, name: file.name, kind: "image", readable: true }
      }))
    };
    const { actions, state, showError } = harness(api);
    const files = Array.from({ length: 20 }, (_, index) =>
      new File(["x"], `f${index}.png`, { type: "image/png" }));

    await actions.attachFiles(files);

    expect(state.pendingAttachments.value).toHaveLength(16);
    expect(showError).toHaveBeenCalledTimes(1);
    expect(String(showError.mock.calls[0][0].message)).toContain("后 4 个未上传");
  });

  it("keeps the files after a rejected one instead of abandoning the batch", async () => {
    // 拖一个文件夹进来、其中一个超大，是很平常的事。
    const api = { uploadAttachment: vi.fn() };
    const { actions, state, showError } = harness(api);
    let call = 0;
    api.uploadAttachment.mockImplementation((_sessionId, file) => {
      call += 1;
      if (call === 2) return Promise.reject(new Error("too large"));
      return Promise.resolve({ attachment: { attachment_id: file.name, name: file.name, kind: "image" } });
    });

    await actions.attachFiles([
      new File(["a"], "a.png", { type: "image/png" }),
      new File(["b"], "big.png", { type: "image/png" }),
      new File(["c"], "c.png", { type: "image/png" })
    ]);

    expect(state.pendingAttachments.value.map((item) => item.name)).toEqual(["a.png", "c.png"]);
    expect(String(showError.mock.calls[0][0].message)).toContain("big.png");
  });

  it("refuses outright once the message is already full", async () => {
    const api = { uploadAttachment: vi.fn() };
    const { actions, state, showError } = harness(api);
    state.pendingAttachments.value = Array.from({ length: 16 }, (_, index) => ({
      attachment_id: `a${index}`, name: `a${index}.png`
    }));

    await actions.attachFiles([new File(["x"], "one-more.png", { type: "image/png" })]);

    expect(api.uploadAttachment).not.toHaveBeenCalled();
    expect(String(showError.mock.calls[0][0].message)).toContain("最多带 16 个附件");
  });
});

describe("useSessionChatActions 的草稿会话", () => {
  // 草稿：还没落库的会话。它只有在真的发出第一条消息时才变成一条会话记录。
  function draftHarness({
    autonameTitle = "季度销售汇总",
    titleDirty = false
  } = {}) {
    const created = { id: "s-new", status: "active", working_dir: "/private/tmp/report" };
    const api = {
      createSession: vi.fn(async () => ({ session: created })),
      listSessions: vi.fn(async () => ({ sessions: [created] })),
      getSession: vi.fn(async () => ({ session: created, turns: [] })),
      chatTurnStream: vi.fn(async () => ({ result: { status: "complete", reply: "好的" } })),
      autonameSession: vi.fn(async () => ({ session_id: created.id, title: autonameTitle }))
    };
    const projects = useProjects({ apiClient: api });
    projects.startDraft();
    vi.spyOn(projects, "remember").mockResolvedValue(null);
    vi.spyOn(projects, "discardDraft");
    const built = harness(api, { id: "", status: "active", title: "新会话" }, null, {
      projects,
      sessionTitleDirty: ref(titleDirty)
    });
    return { ...built, api, projects, created };
  }

  it("keeps attachments local until the first message fixes the directory and title", async () => {
    const { actions, state, api, projects } = draftHarness();
    const file = new File(["report"], "report.pdf", { type: "application/pdf" });
    api.uploadAttachment = vi.fn(async () => ({ attachment: { attachment_id: "uploaded-1", name: file.name } }));
    await actions.attachFiles([file]);
    expect(api.createSession).not.toHaveBeenCalled();
    expect(api.uploadAttachment).not.toHaveBeenCalled();
    expect(projects.draftAttachments.value).toHaveLength(1);
    projects.setDraftDirectory("/chosen/research");
    state.chatInput.value = "整理上传的报告";
    await actions.sendChat();
    expect(api.createSession).toHaveBeenCalledWith({
      title: "整理上传的报告", working_dir: "/chosen/research", client_request_id: expect.any(String)
    });
    expect(api.uploadAttachment).toHaveBeenCalledWith("s-new", file);
    expect(api.chatTurnStream).toHaveBeenCalledWith(expect.objectContaining({ attachment_ids: ["uploaded-1"] }), expect.anything());
  });

  it("retries a failed draft upload without creating another session or omitting the file", async () => {
    const { actions, state, api } = draftHarness();
    const file = new File(["report"], "report.pdf");
    api.uploadAttachment = vi.fn()
      .mockRejectedValueOnce(new Error("upload failed"))
      .mockResolvedValueOnce({ attachment: { attachment_id: "uploaded-1" } });
    await actions.attachFiles([file]);
    state.chatInput.value = "整理上传的报告";
    expect(await actions.sendChat()).toBe(false);
    expect(api.chatTurnStream).not.toHaveBeenCalled();
    expect(state.pendingAttachments.value[0].file).toBe(file);
    expect(state.chatInput.value).toBe("整理上传的报告");
    expect(await actions.sendChat()).toBe(true);
    expect(api.createSession).toHaveBeenCalledTimes(1);
  });

  it("removes draft attachments without sending a delete request", async () => {
    const { actions, api, projects } = draftHarness();
    api.deleteAttachment = vi.fn();
    await actions.attachFiles([new File(["a"], "a.pdf")]);
    await actions.removePendingAttachment(projects.draftAttachments.value[0].attachment_id);
    expect(projects.draftAttachments.value).toEqual([]);
    expect(api.deleteAttachment).not.toHaveBeenCalled();
  });

  it("reuses the frozen creation request after a lost response", async () => {
    const { actions, state, api, projects, created } = draftHarness();
    api.createSession.mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce({ session: created });
    projects.setDraftDirectory("/chosen/research");
    state.chatInput.value = "整理报告";
    expect(await actions.sendChat()).toBe(false);
    projects.setDraftDirectory("/must-not-switch");
    state.chatInput.value = "整理报告，并列出摘要";
    expect(await actions.sendChat()).toBe(true);
    expect(api.createSession.mock.calls[1][0]).toEqual(api.createSession.mock.calls[0][0]);
    expect(api.createSession.mock.calls[0][0].client_request_id).toEqual(expect.any(String));
    expect(api.createSession.mock.calls[0][0].working_dir).toBe("/chosen/research");
    expect(api.chatTurnStream.mock.calls[0][0].message).toBe("整理报告，并列出摘要");
  });

  it("keeps the created Session and attachments until its detail can be loaded", async () => {
    const { actions, state, api, projects } = draftHarness();
    const file = new File(["report"], "report.pdf");
    api.getSession.mockRejectedValueOnce(new Error("detail unavailable"));
    api.uploadAttachment = vi.fn(async () => ({ attachment: { attachment_id: "uploaded-1" } }));
    await actions.attachFiles([file]);
    state.chatInput.value = "整理报告";
    expect(await actions.sendChat()).toBe(false);
    expect(projects.draftActive.value).toBe(true);
    expect(projects.draftAttachments.value[0].file).toBe(file);
    expect(state.chatInput.value).toBe("整理报告");
    expect(state.sessionDetail.value.session.id).toBe("");
    expect(api.uploadAttachment).not.toHaveBeenCalled();
    expect(await actions.sendChat()).toBe(true);
    expect(api.createSession).toHaveBeenCalledTimes(1);
    expect(api.uploadAttachment).toHaveBeenCalledWith("s-new", file);
  });

  it("does not require a list refresh to finish creating and sending", async () => {
    const { actions, state, api } = draftHarness();
    api.listSessions.mockRejectedValue(new Error("list unavailable"));
    state.chatInput.value = "整理报告";
    expect(await actions.sendChat()).toBe(true);
    expect(api.createSession).toHaveBeenCalledTimes(1);
  });

  it("allows editing the directory after an explicitly uncommitted rejection", async () => {
    const { actions, state, api, projects } = draftHarness();
    api.createSession.mockRejectedValueOnce(Object.assign(new Error("invalid directory"), {
      error: { details: { creation_not_committed: true } }
    }));
    projects.setDraftDirectory("/missing");
    state.chatInput.value = "整理报告";
    expect(await actions.sendChat()).toBe(false);
    projects.setDraftDirectory("/fixed");
    expect(await actions.sendChat()).toBe(true);
    expect(api.createSession.mock.calls[1][0].working_dir).toBe("/fixed");
    expect(api.createSession.mock.calls[1][0].client_request_id)
      .not.toBe(api.createSession.mock.calls[0][0].client_request_id);
  });

  it("discards a draft when selecting an existing session", async () => {
    const { actions, api, projects } = draftHarness();
    await actions.guardedSelectSession("s-existing");
    expect(projects.draftActive.value).toBe(false);
    expect(api.createSession).not.toHaveBeenCalled();
  });

  it("a list refresh does not replace the unsent draft with an existing session", async () => {
    const { actions, state, api, projects } = draftHarness();
    await actions.loadSessions("");
    expect(projects.draftActive.value).toBe(true);
    expect(state.sessionDetail.value.session.id).toBe("");
    expect(api.getSession).not.toHaveBeenCalled();
  });

  it("发送第一条消息时才建会话，不发送路径并按后端回传目录登记项目", async () => {
    const { actions, state, api, projects } = draftHarness();
    state.chatInput.value = "帮我汇总三份销售报表";
    await actions.sendChat();

    expect(api.createSession).toHaveBeenCalledWith({ title: "帮我汇总三份销售报表", client_request_id: expect.any(String) });
    // 后端会把 /tmp 解析成 /private/tmp。登记要用会话回传的那个，
    // 否则项目和它自己的会话会分成两组。
    expect(projects.remember).toHaveBeenCalledWith("/private/tmp/report");
  });

  it("把首条消息作为创建期会话名并采用后端回传的默认目录", async () => {
    const { actions, state, api, projects } = draftHarness();
    state.chatInput.value = "  分析本季度\n销售数据  ";

    await actions.sendChat();

    expect(api.createSession).toHaveBeenCalledWith({ title: "分析本季度 销售数据", client_request_id: expect.any(String) });
    expect(projects.remember).toHaveBeenCalledWith("/private/tmp/report");
  });

  it("这句话不会在切到新会话时被输入框的重置吃掉", async () => {
    const { actions, state, api } = draftHarness();
    state.chatInput.value = "帮我汇总三份销售报表";
    await actions.sendChat();

    expect(api.chatTurnStream).toHaveBeenCalledWith(
      expect.objectContaining({ message: "帮我汇总三份销售报表" }),
      expect.anything()
    );
  });

  it("首轮结束后按对话内容起名，并让标题输入框跟上", async () => {
    const { actions, state, api } = draftHarness();
    state.chatInput.value = "帮我汇总三份销售报表";
    await actions.sendChat();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.autonameSession).toHaveBeenCalledWith("s-new", expect.objectContaining({
      user_text: "帮我汇总三份销售报表"
    }));
    expect(state.sessionDetail.value.session.title).toBe("季度销售汇总");
    // 输入框必须一起跟上，不然它和标题对不上，会一直显示"有未保存改动"。
    expect(state.sessionTitleDraft.value).toBe("季度销售汇总");
  });

  it("起名失败不影响这一轮的结果", async () => {
    const { actions, state, api, showError } = draftHarness();
    api.autonameSession = vi.fn(async () => { throw new Error("模型没返回"); });
    state.chatInput.value = "帮我汇总三份销售报表";
    const sent = await actions.sendChat();
    await Promise.resolve();
    await Promise.resolve();

    // 名字是装饰，不是这轮的交付物：起不出来就保持无名，回复照常返回。
    expect(sent).toBe(true);
    expect(showError).not.toHaveBeenCalled();
    expect(state.lastTurnResult.value?.status).toBe("complete");
  });
});
