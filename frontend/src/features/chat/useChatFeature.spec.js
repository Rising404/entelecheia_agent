import { afterEach, describe, expect, it, vi } from "vitest";
import { effectScope, ref } from "vue";
import { flushPromises } from "@vue/test-utils";
import { useChatFeature } from "./useChatFeature";
import { useProjects } from "../workspace/useProjects";

const turnRuntimeRouting = {
  schema_version: 2,
  source: "session_default",
  policy: { schema_version: 2, l1_enabled: true },
  allowed_processing_levels: ["L0", "L1"],
  available_processing_levels: ["L0", "L1"]
};

afterEach(() => {
  localStorage.clear();
  vi.clearAllTimers();
  vi.useRealTimers();
});

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

describe("useChatFeature", () => {
  it("does not inherit a previous session window when the selected draft has no id", () => {
    const scope = effectScope();
    const feature = scope.run(() => useChatFeature({
      api: {}, mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""),
      runtimeRoutingStatus: ref(turnRuntimeRouting),
      sessions: ref([]), selectedSessionId: ref(""), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn()
    }));
    try {
      feature.sessionDetail.value = { session: { id: "", title: "新会话", status: "active" } };
      feature.lastTurnResult.value = {
        session_id: "old-session", window_state: "post_commit_pending", window_revision: 8
      };
      expect(feature.turnWindow.value.state).toBe("empty");
      expect(feature.canSendChat.value).toBe(true);
    } finally { scope.stop(); }
  });

  it("clears turn state and ignores late session reads when opening or discarding a draft", async () => {
    const scope = effectScope();
    const pending = deferred();
    const projects = useProjects({ apiClient: {} });
    const feature = scope.run(() => useChatFeature({
      api: { getSession: vi.fn(() => pending.promise) },
      projects, mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""),
      runtimeRoutingStatus: ref(turnRuntimeRouting),
      sessions: ref([]), selectedSessionId: ref("old-session"), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn()
    }));
    try {
      feature.sessionDetail.value = { session: { id: "old-session", title: "旧会话", status: "active" } };
      const selecting = feature.selectSession("old-session");
      feature.lastTurnResult.value = {
        session_id: "old-session", window_state: "post_commit_pending", window_revision: 8
      };
      feature.streamingReply.value = "旧回复";
      feature.pendingAttachments.value = [{ attachment_id: "old-attachment" }];
      feature.startDraft("/workspace/new");

      expect(feature.lastTurnResult.value).toBeNull();
      expect(feature.streamingReply.value).toBe("");
      expect(feature.pendingAttachments.value).toEqual([]);
      expect(feature.sessionTitleDirty.value).toBe(false);
      expect(projects.draftWorkingDir.value).toBe("/workspace/new");
      expect(feature.canSendChat.value).toBe(true);
      pending.resolve({ session: { id: "old-session", status: "active" }, turns: [] });
      await selecting;
      expect(feature.selectedSession.value.id).toBe("");
      feature.discardDraft();
      expect(projects.draftActive.value).toBe(false);
      expect(feature.sessionDetail.value).toBeNull();
      expect(feature.lastTurnResult.value).toBeNull();
    } finally { scope.stop(); }
  });

  it("starts in the L1-backed turn mode before routing projections arrive", () => {
    const feature = useChatFeature({
      api: {},
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace: null,
      sessions: ref([]),
      selectedSessionId: ref(""),
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn(),
      loadReferenceChoices: vi.fn()
    });

    expect(feature.runtimeMode.value).toBe("turn");
  });

  it("preserves a per-Turn mode choice across same-Session detail refreshes", async () => {
    const session = {
      id: "session-a",
      title: "单轮选择",
      status: "active",
    };
    const routing = {
      schema_version: 2,
      source: "session_default",
      policy: { schema_version: 2, l1_enabled: false },
      allowed_processing_levels: ["L0"],
      available_processing_levels: ["L0", "L1"]
    };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [], runtime_routing: routing }) },
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace: null,
      sessions: ref([session]),
      selectedSessionId: ref(session.id),
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn(),
      loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = { session, turns: [], runtime_routing: routing };
    await flushPromises();
    expect(feature.runtimeMode.value).toBe("turn");

    // 用一个仍然可选的模式来验"刷新不覆盖用户选择"这件事本身。
    // "direct" 已不再提供，见 RuntimeModeControl.spec 的对应用例。
    feature.runtimeMode.value = "turn";
    feature.sessionDetail.value = {
      session,
      turns: [{ role: "assistant", content: "后台刷新" }],
      runtime_routing: routing
    };
    await flushPromises();

    expect(feature.runtimeMode.value).toBe("turn");
  });

  it("withdraws the old Session authority while a replacement detail is pending or failed", async () => {
    const sessionA = {
      id: "session-a",
      title: "A",
      status: "active",
      working_dir: "/workspace/a"
    };
    const sessionB = {
      id: "session-b",
      title: "B",
      status: "active",
      working_dir: "/workspace/b"
    };
    const detailB = deferred();
    const showError = vi.fn();
    const api = {
      getSession: vi.fn((sessionId) => (
        sessionId === sessionA.id
          ? Promise.resolve({ session: sessionA, turns: [{ role: "assistant", content: "A turn" }] })
          : detailB.promise
      ))
    };
    const workspace = {
      selectSession: vi.fn(),
      refreshContents: vi.fn().mockResolvedValue(undefined)
    };
    const selectedSessionId = ref(sessionA.id);
    const feature = useChatFeature({
      api,
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace,
      sessions: ref([sessionA, sessionB]),
      selectedSessionId,
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError,
      loadReferenceChoices: vi.fn()
    });
    await feature.selectSession(sessionA.id);
    expect(feature.selectedSession.value?.working_dir).toBe("/workspace/a");

    const switching = feature.selectSession(sessionB.id);

    expect(selectedSessionId.value).toBe(sessionB.id);
    expect(feature.selectedSession.value).toBeNull();
    expect(feature.turns.value).toEqual([]);
    expect(feature.sessionTitleDraft.value).toBe("");
    // App 依据这份 authority 同时控制工作目录文件面和新文档收录。null 表示两者都不会
    // 意外复用 Session A。
    expect(feature.selectedSession.value?.working_dir).toBeUndefined();

    await Promise.resolve();
    detailB.reject(new Error("detail unavailable"));
    await switching;

    expect(feature.selectedSession.value).toBeNull();
    expect(showError).toHaveBeenCalledWith(expect.objectContaining({
      message: "detail unavailable"
    }));
  });

  it("projects only safe incomplete Turn fields into chat state", () => {
    const session = { id: "session-a", title: "隐私测试", status: "active" };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) },
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace: null,
      sessions: ref([session]),
      selectedSessionId: ref(session.id),
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn(),
      loadReferenceChoices: vi.fn()
    });
    feature.lastTurnResult.value = {
      status: "incomplete",
      end_reason: "provider_unavailable",
      error_code: "MODEL_TIMEOUT",
      reply: "this draft must not be rendered",
      diagnostics: { internal: "must not cross the UI boundary" }
    };

    expect(feature.incompleteTurn.value).toEqual({
      endReason: "provider_unavailable",
      errorCode: "MODEL_TIMEOUT"
    });

    feature.lastTurnResult.value = { status: "completed", reply: "正式回复" };
    expect(feature.incompleteTurn.value).toBeNull();
  });

  it("says which stage a running Turn is in, and when nothing has changed", async () => {
    // 停滞 Turn 与正常运行 Turn 曾显示同一句话，用户只能等待并猜测来区分。
    vi.useFakeTimers();
    const session = { id: "session-a", title: "分析材料", status: "active" };
    // composable 现在会在 Turn 运行时轮询，因此 stub 必须像真实端点一样响应；若重载时
    // 丢失 Window，就会终止正在测试的状态。
    const running = {
      session, turns: [],
      turn_window: {
        turn_id: "turn-1", window_state: "active", window_revision: 7, stage: "TOOL"
      }
    };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue(running) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = running;
    await flushPromises();

    expect(feature.turnWindow.value.stage).toBe("TOOL");
    expect(feature.chatReadOnlyReason.value).toContain("正在调用能力");
    expect(feature.chatReadOnlyReason.value).not.toContain("没有任何变化");

    vi.advanceTimersByTime(125_000);
    await flushPromises();
    expect(feature.chatReadOnlyReason.value).toContain("已 2 分钟没有任何变化");
  });

  it("stops counting once the Window moves on", async () => {
    vi.useFakeTimers();
    const session = { id: "session-a", title: "分析材料", status: "active" };
    const window_ = (revision, stage) => ({
      session, turns: [],
      turn_window: { turn_id: "turn-1", window_state: "active", window_revision: revision, stage }
    });
    const latest = { value: window_(7, "TOOL") };
    const feature = useChatFeature({
      api: { getSession: vi.fn(() => Promise.resolve(latest.value)) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = latest.value;
    await flushPromises();
    vi.advanceTimersByTime(90_000);
    await flushPromises();
    expect(feature.chatReadOnlyReason.value).toContain("没有任何变化");

    // 推进到下一阶段正是它仍在运行的证据。
    latest.value = window_(8, "RESPONSE");
    feature.sessionDetail.value = latest.value;
    await flushPromises();
    expect(feature.chatReadOnlyReason.value).toContain("正在交付回复");
    expect(feature.chatReadOnlyReason.value).not.toContain("没有任何变化");
  });

  it("holds the composer while an authoritative post-commit window is pending", () => {
    const session = { id: "session-a", title: "整理上下文", status: "active" };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = {
      session,
      turns: [],
      runtime_routing: turnRuntimeRouting
    };
    feature.lastTurnResult.value = {
      session_id: session.id,
      turn_id: "turn-1",
      status: "completed",
      reply: "正式交付",
      window_state: "post_commit_pending",
      window_revision: 2
    };

    expect(feature.turnWindow.value).toEqual({
      state: "post_commit_pending",
      revision: 2,
      stage: null,
      interruptionReason: null,
      postCommitStatus: "pending",
      postCommitErrorCodes: []
    });
    expect(feature.canSendChat.value).toBe(false);
    expect(feature.chatInputPlaceholder.value).toBe("正在整理本轮上下文");
    expect(feature.chatReadOnlyReason.value).toBe("正在整理本轮上下文");

    feature.sessionDetail.value = {
      session,
      turns: [],
      runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "empty", window_revision: 3 }
    };

    expect(feature.turnWindow.value).toEqual({
      state: "empty", revision: 3, stage: null, interruptionReason: null
    });
    expect(feature.canSendChat.value).toBe(true);
  });


  it("keeps session selection, title dirty state, and title reset within chat", async () => {
    const session = { id: "session-a", title: "项目讨论", status: "active" };
    const api = {
      listSessions: vi.fn().mockResolvedValue({ sessions: [session] }),
      getSession: vi.fn().mockResolvedValue({ session, turns: [] })
    };
    const sessions = ref([]);
    const selectedSessionId = ref("");
    const workspace = {
      refresh: vi.fn().mockImplementation(async () => { sessions.value = [session]; }),
      refreshContents: vi.fn()
    };
    const feature = useChatFeature({
      api,
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace,
      sessions,
      selectedSessionId,
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn(),
      loadReferenceChoices: vi.fn()
    });

    await feature.loadSessions();
    expect(feature.selectedSession.value?.id).toBe("session-a");
    feature.sessionTitleDraft.value = "项目讨论（修订）";
    expect(feature.sessionTitleDirty.value).toBe(true);
    feature.discardChanges();
    expect(feature.sessionTitleDirty.value).toBe(false);
  });

  it("restores the selected session public runtime timeline", async () => {
    const session = { id: "session-a", title: "工具任务", status: "active" };
    const persisted = {
      schema_version: 1,
      event_id: "evt-1",
      sequence: 1,
      session_id: session.id,
      turn_id: "turn-1",
      stage: "TOOL",
      status: "completed",
      occurred_at: "2026-07-15T00:00:00Z",
      prompt_replay: false,
      operation_id: "operation-1"
    };
    const api = {
      getSession: vi.fn().mockResolvedValue({ session, turns: [] }),
      getRuntimeEvents: vi.fn().mockResolvedValue({ enabled: true, events: [persisted] })
    };
    const sessions = ref([session]);
    const feature = useChatFeature({
      api,
      mode: ref("chat"),
      loading: ref(false),
      saving: ref(false),
      notice: ref(""),
      workspace: null,
      sessions,
      selectedSessionId: ref(session.id),
      confirmDiscardChanges: () => true,
      clearMessage: vi.fn(),
      showError: vi.fn(),
      loadReferenceChoices: vi.fn()
    });

    await feature.selectSession(session.id);

    expect(api.getRuntimeEvents).toHaveBeenCalledWith(session.id, { limit: 100 });
    expect(feature.runtimeEvents.value).toEqual([persisted]);
  });

  it("blocks only the current composer while the active Turn is running", () => {
    const session = { id: "session-a", title: "后台任务", status: "active" };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });

    feature.chatBusy.value = true;

    expect(feature.canSendChat.value).toBe(false);
    expect(feature.chatInputPlaceholder.value).toBe("当前回合正在运行");
    expect(feature.chatReadOnlyReason.value).toContain("当前回合正在运行");
  });

  it("shows a safe interruption marker after reloading the session", () => {
    const session = { id: "session-a", title: "异常恢复", status: "active" };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = {
      session,
      turns: [{ turn_idx: 0, role: "user", content: "上一轮输入" }],
      runtime_routing: turnRuntimeRouting,
      turn_window: {
        window_state: "interrupted",
        window_revision: 4,
        interruption_reason: "MODEL_TIMEOUT"
      }
    };

    expect(feature.incompleteTurn.value).toEqual({
      endReason: "unknown",
      errorCode: "MODEL_TIMEOUT"
    });
    expect(feature.canSendChat.value).toBe(true);
  });

  it("shows a safe post-commit failure category and does not keep polling it", () => {
    const session = { id: "session-a", title: "摘要失败", status: "active" };
    const feature = useChatFeature({
      api: { getSession: vi.fn().mockResolvedValue({ session, turns: [] }) },
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = {
      session,
      turns: [],
      turn_window: {
        window_state: "post_commit_pending",
        window_revision: 5,
        post_commit_status: "failed",
        post_commit_error_codes: ["SUMMARY_MODEL_TIMEOUT"]
      }
    };

    expect(feature.canSendChat.value).toBe(false);
    expect(feature.chatInputPlaceholder.value).toBe("会话摘要更新失败");
    expect(feature.chatReadOnlyReason.value).toBe(
      "本轮回复已保存，但会话摘要更新失败（SUMMARY_MODEL_TIMEOUT）；当前会话已暂停。"
    );
  });

  it("identifies index failures from the actual failed jobs instead of calling them summary failures", () => {
    const session = { id: "session-a", status: "active" };
    const feature = useChatFeature({
      api: {}, mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""),
      workspace: null, sessions: ref([session]), selectedSessionId: ref(session.id),
      confirmDiscardChanges: () => true, clearMessage: vi.fn(), showError: vi.fn()
    });
    feature.sessionDetail.value = {
      session, turns: [],
      turn_window: {
        window_state: "post_commit_pending", window_revision: 5,
        post_commit_status: "failed", post_commit_error_codes: ["POST_COMMIT_HANDLER_UNAVAILABLE"]
      },
      post_commit: { failed_jobs: [{ job_kind: "session_retrieval_index" }] }
    };
    expect(feature.chatReadOnlyReason.value).toContain("历史检索索引更新失败");
    expect(feature.chatReadOnlyReason.value).not.toContain("摘要");
    expect(feature.chatInputPlaceholder.value).toBe("历史检索索引更新失败");
  });

  it("shows the accepted user message before the answer stream completes", async () => {
    vi.useFakeTimers();
    const session = { id: "session-a", title: "发送中", status: "active" };
    const stream = deferred();
    const userTurn = { turn_idx: 0, role: "user", content: "请读取本轮附件" };
    let storedDetail = {
      session,
      turns: [userTurn],
      runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "active", window_revision: 1 }
    };
    const api = {
      chatTurnStream: vi.fn(() => stream.promise),
      getSession: vi.fn(async () => storedDetail)
    };
    const feature = useChatFeature({
      api,
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn()
    });
    feature.sessionDetail.value = {
      session, turns: [], runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "empty", window_revision: 0 }
    };
    feature.chatInput.value = userTurn.content;
    const sending = feature.sendChat();
    await flushPromises();

    await vi.advanceTimersByTimeAsync(2_000);
    expect(feature.chatBusy.value).toBe(true);
    expect(feature.turns.value).toEqual([userTurn]);
    expect(feature.turnWindow.value.state).toBe("active");

    storedDetail = {
      ...storedDetail,
      turns: [userTurn, { turn_idx: 1, role: "assistant", content: "已读取" }],
      turn_window: { window_state: "empty", window_revision: 2 }
    };
    stream.resolve({ result: { status: "completed", reply: "已读取" } });
    await sending;
    expect(feature.turns.value).toEqual(storedDetail.turns);
    expect(feature.chatBusy.value).toBe(false);
    const readsAfterCompletion = api.getSession.mock.calls.length;
    await vi.advanceTimersByTimeAsync(4_000);
    expect(api.getSession).toHaveBeenCalledTimes(readsAfterCompletion);
  });

  it("discards an old send refresh and continues refreshing the new session", async () => {
    vi.useFakeTimers();
    const sessionA = { id: "session-a", status: "active" };
    const sessionB = { id: "session-b", status: "active" };
    const streamA = deferred();
    const streamB = deferred();
    const oldRefresh = deferred();
    let detailB = {
      session: sessionB, turns: [{ turn_idx: 0, role: "user", content: "B 的消息" }],
      runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "empty", window_revision: 0 }
    };
    const api = {
      chatTurnStream: vi.fn((payload) => payload.session_id === sessionA.id ? streamA.promise : streamB.promise),
      getSession: vi.fn((id) => id === sessionA.id ? oldRefresh.promise : Promise.resolve(detailB))
    };
    const feature = useChatFeature({
      api,
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([sessionA, sessionB]), selectedSessionId: ref(sessionA.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn()
    });
    feature.sessionDetail.value = {
      session: sessionA, turns: [], runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "empty", window_revision: 0 }
    };
    feature.chatInput.value = "A 的消息";
    const sending = feature.sendChat();
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2_000);
    expect(api.getSession).toHaveBeenCalledWith(sessionA.id);

    await feature.selectSession(sessionB.id);
    feature.chatInput.value = "B 的下一条消息";
    const sendingB = feature.sendChat();
    await flushPromises();
    oldRefresh.resolve({
      session: sessionA, turns: [{ turn_idx: 0, role: "user", content: "A 的消息" }],
      turn_window: { window_state: "active", window_revision: 1 }
    });
    streamA.resolve({ result: { status: "completed", reply: "A 的回复" } });
    await sending;
    await flushPromises();
    expect(feature.selectedSession.value.id).toBe(sessionB.id);
    expect(feature.turns.value).toEqual(detailB.turns);
    expect(feature.chatBusy.value).toBe(true);

    detailB = {
      ...detailB,
      turns: [...detailB.turns, { turn_idx: 1, role: "user", content: "B 的下一条消息" }],
      turn_window: { window_state: "active", window_revision: 1 }
    };
    await vi.advanceTimersByTimeAsync(2_000);
    expect(feature.turns.value).toEqual(detailB.turns);
    detailB = { ...detailB, turn_window: { window_state: "empty", window_revision: 2 } };
    streamB.resolve({ result: { status: "completed", reply: "B 的回复" } });
    await sendingB;
    expect(feature.chatBusy.value).toBe(false);
  });

  it("does not restart send refreshes after their Vue scope is disposed", async () => {
    vi.useFakeTimers();
    const session = { id: "session-a", status: "active" };
    const refresh = deferred();
    const api = { getSession: vi.fn(() => refresh.promise) };
    const scope = effectScope();
    const feature = scope.run(() => useChatFeature({
      api,
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn()
    }));
    const initialDetail = {
      session, turns: [], runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "empty", window_revision: 0 }
    };
    feature.sessionDetail.value = initialDetail;
    feature.chatBusy.value = true;
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2_000);
    expect(api.getSession).toHaveBeenCalledTimes(1);
    scope.stop();

    refresh.resolve({ ...initialDetail, turns: [{ turn_idx: 0, role: "user", content: "迟到的消息" }] });
    await flushPromises();
    await vi.advanceTimersByTimeAsync(4_000);
    expect(feature.turns.value).toEqual([]);
    expect(api.getSession).toHaveBeenCalledTimes(1);
  });

  it("refreshes a pending execution window until the durable slot is released", async () => {
    vi.useFakeTimers();
    const session = { id: "session-a", title: "摘要完成", status: "active" };
    const api = {
      getSession: vi.fn().mockResolvedValue({
        session,
        turns: [],
        runtime_routing: turnRuntimeRouting,
        turn_window: { window_state: "empty", window_revision: 6 }
      })
    };
    const feature = useChatFeature({
      api,
      mode: ref("chat"), loading: ref(false), saving: ref(false), notice: ref(""), workspace: null,
      sessions: ref([session]), selectedSessionId: ref(session.id), confirmDiscardChanges: () => true,
      clearMessage: vi.fn(), showError: vi.fn(), loadReferenceChoices: vi.fn()
    });
    feature.sessionDetail.value = {
      session,
      turns: [],
      runtime_routing: turnRuntimeRouting,
      turn_window: { window_state: "post_commit_pending", window_revision: 5 }
    };

    await vi.advanceTimersByTimeAsync(2_000);
    await flushPromises();

    expect(api.getSession).toHaveBeenCalledWith(session.id);
    expect(feature.turnWindow.value).toEqual({
      state: "empty", revision: 6, stage: null, interruptionReason: null
    });
    expect(feature.canSendChat.value).toBe(true);
  });


});
