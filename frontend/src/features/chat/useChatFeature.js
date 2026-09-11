import { computed, ref, watch } from "vue";
import { useChatDrafts } from "./useChatDrafts";
import { useChatRuntimeDecorations } from "./useChatRuntimeDecorations";
import {
  TASK_MODE_SEND_BLOCKED_NOTICE,
  useSessionChatActions
} from "./useSessionChatActions";
import { useTurnWindowRefresh } from "./useTurnWindowRefresh";
import { usePostCommitRecovery } from "./usePostCommitRecovery";
import { postCommitFailureLabel } from "./presentation";
import { useRuntimeEvents } from "../runtime/useRuntimeEvents";
import { stageText } from "../../shared/runtimeStages";
import {
  RUNTIME_MODE_TASK,
  RUNTIME_MODE_TURN,
  runtimeModeAvailable,
  runtimeModeFromProjection,
  safeRuntimeMode
} from "../runtime/runtimeRouting";

const TURN_WINDOW_STATES = new Set(["empty", "active", "post_commit_pending", "interrupted"]);
const POST_COMMIT_STATES = new Set(["pending", "failed"]);

function readTurnWindowProjection(value) {
  if (!value || typeof value !== "object") return null;
  const state = value.window_state ?? value.state;
  if (!TURN_WINDOW_STATES.has(state)) return null;
  const revision = value.window_revision ?? value.revision;
  const projection = {
    state,
    revision: Number.isInteger(revision) && revision >= 0 ? revision : null,
    // Window 已经报告当前阶段；若丢弃该信息，正在运行的 Turn 将无法与停滞 Turn 区分。
    stage: typeof value.stage === "string" && value.stage.trim() ? value.stage : null,
    interruptionReason: typeof value.interruption_reason === "string"
      ? value.interruption_reason
      : null
  };
  if (state !== "post_commit_pending") return projection;
  const postCommitStatus = POST_COMMIT_STATES.has(value.post_commit_status)
    ? value.post_commit_status
    : "pending";
  const postCommitErrorCodes = Array.isArray(value.post_commit_error_codes)
    ? [...new Set(value.post_commit_error_codes.filter(
      (code) => typeof code === "string" && code.trim()
    ))].sort()
    : [];
  return { ...projection, postCommitStatus, postCommitErrorCodes };
}

function turnWindowBlocksNewInput(state) {
  // 此处刻意不阻塞被中断的 Turn：下一条普通用户提示才是触发服务端审计的权威信号。
  return state === "active" || state === "post_commit_pending";
}

/** 会话树和聊天工作区共享的会话/聊天局部状态。 */
export function useChatFeature({
  api,
  mode,
  loading,
  saving,
  notice,
  workspace,
  projects = null,
  sessions,
  selectedSessionId,
  confirmDiscardChanges,
  confirmPostCommitWaiver,
  clearMessage,
  showError,
  runtimeRoutingStatus = null
}) {
  const sessionStatus = ref("active");
  const sessionSearch = ref("");
  const sessionDetail = ref(null);
  const sessionTitleDraft = ref("");
  const newSessionTitle = ref("");
  const chatInput = ref("");
  const chatBusy = ref(false);
  const pendingAttachments = ref([]);
  const attachmentUploading = ref(false);
  const chatStreamEvents = ref([]);
  const postCommitRecovery = usePostCommitRecovery({
    api, selectedSessionId, sessionDetail, saving, chatBusy, attachmentUploading,
    notice, clearMessage, showError, confirmWaiver: confirmPostCommitWaiver
  });

  const runtimeTimeline = useRuntimeEvents({
    api,
    onError: showError
  });
  const runtimeEvents = runtimeTimeline.events;
  const {
    inSessionTaskDetails,
    clearInSessionTaskDetails,
    loadRelatedInSessionTaskDetails
  } = useChatRuntimeDecorations({
    api,
    selectedSessionId
  });

  const streamingReply = ref("");
  const lastTurnResult = ref(null);
  const pendingClientRequest = ref(null);
  const runtimeModeValue = ref(RUNTIME_MODE_TURN);
  const runtimeModeTouched = ref(false);
  const runtimeMode = computed({
    get: () => runtimeModeValue.value,
    set: (value) => {
      runtimeModeTouched.value = true;
      runtimeModeValue.value = value;
    }
  });
  const drafts = useChatDrafts();

  const selectedSession = computed(() => sessionDetail.value?.session || null);
  const runtimeRouting = computed(() =>
    sessionDetail.value?.runtime_routing
    || runtimeRoutingStatus?.value
    || null
  );
  const runtimeModeIsAvailable = computed(() =>
    runtimeModeAvailable(runtimeMode.value, runtimeRouting.value)
  );
  const turns = computed(() => sessionDetail.value?.turns || []);
  const pendingUserQuestions = computed(() => {
    const values = sessionDetail.value?.pending_user_questions;
    if (!Array.isArray(values)) return [];
    return values.filter((item) =>
      item &&
      typeof item === "object" &&
      typeof item.insession_task_id === "string" &&
      item.insession_task_id.trim() &&
      typeof item.question === "string" &&
      item.question.trim()
    );
  });
  const turnWindow = computed(() => {
    // 将来 Session 重载可能会直接提供这份投影；在此之前，最近一次权威 Turn 结果是
    // 唯一可用的窗口来源。
    const sessionProjection = readTurnWindowProjection(
      sessionDetail.value?.turn_window ?? sessionDetail.value?.turnWindow
    );
    if (sessionProjection) return sessionProjection;

    const result = lastTurnResult.value;
    if (
      result?.session_id &&
      selectedSession.value?.id &&
      result.session_id !== selectedSession.value.id
    ) return { state: "empty", revision: null, stage: null, interruptionReason: null };
    return readTurnWindowProjection(result) || {
      state: "empty", revision: null, stage: null, interruptionReason: null
    };
  });
  const incompleteTurn = computed(() => {
    const result = lastTurnResult.value;
    if (result?.status === "incomplete") {
      return {
        endReason: typeof result.end_reason === "string" ? result.end_reason : null,
        errorCode: typeof result.error_code === "string" ? result.error_code : null
      };
    }
    // 应用重启后，内存中不再有最近的 Turn 结果。持久 Window 仍带有安全停止标记，
    // 因此应展示中性的中断卡片，而不是在对话记录中静默留下孤立的用户消息。
    if (turnWindow.value.state !== "interrupted") return null;
    const marker = turnWindow.value.interruptionReason;
    return {
      endReason: ["user_paused", "host_stopped", "process_lost"].includes(marker)
        ? marker
        : "unknown",
      errorCode: marker
    };
  });
  const shouldRefreshTurnWindow = computed(() =>
    // 刷新运行中 Turn 的原因与刷新收尾中 Turn 相同：Window 由服务端拥有；若不轮询，
    // 即便等待的 Turn 已推进或终止，编辑器仍会长时间保持禁用。
    turnWindow.value.state === "active"
    || (
      turnWindow.value.state === "post_commit_pending"
      && turnWindow.value.postCommitStatus !== "failed"
    )
  );

  // Window 保持完全相同外观的时长。这里刻意不把它说成“Turn 已运行多久”：payload
  // 没有开始时间，虚构一个数字会造成误导。它只能如实说明 UI 观察期间没有变化。
  const turnWindowSignature = computed(() => [
    turnWindow.value.state,
    turnWindow.value.revision,
    turnWindow.value.stage
  ].join("|"));
  const unchangedSince = ref(Date.now());
  const now = ref(Date.now());
  let unchangedTimer = null;
  watch(turnWindowSignature, () => { unchangedSince.value = Date.now(); });
  watch(
    () => turnWindow.value.state === "active",
    (running) => {
      if (unchangedTimer !== null) {
        globalThis.clearInterval(unchangedTimer);
        unchangedTimer = null;
      }
      if (!running) return;
      unchangedTimer = globalThis.setInterval(() => { now.value = Date.now(); }, 1_000);
    },
    { immediate: true }
  );
  const turnWindowUnchangedSeconds = computed(() =>
    turnWindow.value.state === "active"
      ? Math.max(0, Math.floor((now.value - unchangedSince.value) / 1000))
      : 0
  );
  useTurnWindowRefresh({
    api,
    selectedSessionId,
    sessionDetail,
    shouldRefresh: shouldRefreshTurnWindow
  });
  const sessionTitleDirty = computed(() => Boolean(selectedSession.value) &&
    String(sessionTitleDraft.value || "").trim() !== String(selectedSession.value?.title || "").trim()
  );
  const currentChatDraftLength = computed(() => chatInput.value.trim().length);
  const canSendChat = computed(() =>
    selectedSession.value?.status === "active" &&
    runtimeModeIsAvailable.value &&
    !chatBusy.value &&
    !postCommitRecovery.postCommitRecoveryBusy.value &&
    !turnWindowBlocksNewInput(turnWindow.value.state)
  );
  const canChangeRuntimeMode = computed(() =>
    selectedSession.value?.status === "active" &&
    !chatBusy.value &&
    !turnWindowBlocksNewInput(turnWindow.value.state)
  );
  const canAnswerPendingQuestion = computed(() =>
    pendingUserQuestions.value.length > 0 &&
    selectedSession.value?.status === "active" &&
    runtimeModeIsAvailable.value &&
    !chatBusy.value &&
    turnWindow.value.state !== "post_commit_pending"
  );
  const chatReadOnlyReason = computed(() => {
    if (chatBusy.value) return "当前回合正在运行，完成后才能发送下一条消息。";
    if (
      turnWindow.value.state === "post_commit_pending"
      && turnWindow.value.postCommitStatus === "failed"
    ) {
      const code = turnWindow.value.postCommitErrorCodes?.[0];
      const label = postCommitFailureLabel(
        sessionDetail.value?.post_commit?.failed_jobs,
        turnWindow.value.postCommitErrorCodes
      );
      return code
        ? `本轮回复已保存，但${label}更新失败（${code}）；当前会话已暂停。`
        : `本轮回复已保存，但${label}更新失败；当前会话已暂停。`;
    }
    if (turnWindow.value.state === "post_commit_pending") return "正在整理本轮上下文";
    if (turnWindow.value.state === "active") return runningTurnNotice.value;
    if (!runtimeModeIsAvailable.value) return "本轮运行模式当前不可用。";
    if (!selectedSession.value || selectedSession.value.status === "active") return "";
    if (selectedSession.value.status === "archived") return "归档会话为只读；取消归档后可继续对话。";
    if (selectedSession.value.status === "trashed") return "回收站会话为只读；恢复后可继续对话。";
    return "当前会话为只读。";
  });
  const runningTurnNotice = computed(() => {
    // 阶段代码是日志标识符，应出现在开发者视图，而非用户等待时阅读的句子中。
    // 未知代码直接省略，不展示原始值。
    const stage = stageText(turnWindow.value.stage);
    const head = stage
      ? `正在${stage}，完成后才能发送下一条消息。`
      : "当前回合仍在运行，完成后才能发送下一条消息。";
    // 只有等待足够久、值得提示时才展示；一开始就不断提醒的计数器只会让用户学会忽略它。
    const seconds = turnWindowUnchangedSeconds.value;
    if (seconds < 60) return head;
    const minutes = Math.floor(seconds / 60);
    return `${head} 已 ${minutes} 分钟没有任何变化。`;
  });

  const chatInputPlaceholder = computed(() => {
    if (chatBusy.value) return "当前回合正在运行";
    if (
      turnWindow.value.state === "post_commit_pending"
      && turnWindow.value.postCommitStatus === "failed"
    ) return `${postCommitFailureLabel(
      sessionDetail.value?.post_commit?.failed_jobs,
      turnWindow.value.postCommitErrorCodes
    )}更新失败`;
    if (turnWindow.value.state === "post_commit_pending") return "正在整理本轮上下文";
    if (turnWindow.value.state === "active") return "当前回合仍在运行";
    if (selectedSession.value?.status !== "active") return "归档或回收站会话为只读";
    // 发送时还会在公共动作边界再次拦截；这里先把状态说清楚，避免暗示它只是
    // “不稳定”但仍可真实执行。
    if (runtimeModeValue.value === RUNTIME_MODE_TASK) return TASK_MODE_SEND_BLOCKED_NOTICE;
    return "输入消息";
  });

  const actions = useSessionChatActions({
    api,
    state: {
      mode,
      loading,
      saving,
      notice,
      sessionStatus,
      sessionSearch,
      sessions,
      selectedSessionId,
      sessionDetail,
      sessionTitleDraft,
      newSessionTitle,
      chatInput,
      pendingAttachments,
      attachmentUploading,
      chatBusy,
      chatStreamEvents,
      runtimeEvents,
      streamingReply,
      lastTurnResult,
      pendingClientRequest
    },
    selectedSession,
    sessionTitleDirty,
    canSendChat,
    canAnswerPendingQuestion,
    drafts: {
      load: drafts.load,
      save: drafts.save,
      clear: drafts.clear
    },
    confirmDiscardChanges,
    clearMessage,
    showError,
    workspace,
    projects,
    runtimeTimeline,
    runtimeMode,
    clearInSessionTaskDetails,
    loadRelatedInSessionTaskDetails
  });

  watch(chatInput, (value) => {
    drafts.save(selectedSessionId.value, value);
  });

  let runtimeModeOwner = null;
  watch(
    () => sessionDetail.value,
    (detail) => {
      const owner = detail?.session?.id ?? null;
      if (owner !== runtimeModeOwner) {
        runtimeModeOwner = owner;
        runtimeModeTouched.value = false;
      }
      const routing = detail?.runtime_routing;
      if (routing) {
        if (
          !runtimeModeTouched.value
          || !runtimeModeAvailable(runtimeModeValue.value, routing)
        ) {
          runtimeModeValue.value = safeRuntimeMode(
            routing,
            runtimeModeFromProjection(routing)
          );
        }
        return;
      }
      // 草稿 Session 尚无持久策略。其首个 Turn 使用当前 Host 默认值，并受已发布能力门约束。
      if (
        detail?.session?.id === ""
        && runtimeRoutingStatus?.value
        && (
          !runtimeModeTouched.value
          || !runtimeModeAvailable(
            runtimeModeValue.value,
            runtimeRoutingStatus.value
          )
        )
      ) {
        runtimeModeValue.value = safeRuntimeMode(
          runtimeRoutingStatus.value,
          runtimeModeFromProjection(runtimeRoutingStatus.value)
        );
        return;
      }
      if (
        detail?.session
        && runtimeRouting.value
        && (
          !runtimeModeTouched.value
          || !runtimeModeAvailable(runtimeModeValue.value, runtimeRouting.value)
        )
      ) {
        runtimeModeValue.value = safeRuntimeMode(
          runtimeRouting.value,
          runtimeModeFromProjection(runtimeRouting.value)
        );
      }
    },
    { immediate: true, flush: "sync" }
  );

  watch(
    () => runtimeRoutingStatus?.value,
    (routing) => {
      if (!routing || sessionDetail.value?.runtime_routing) return;
      if (sessionDetail.value?.session?.id === "" && !runtimeModeTouched.value) {
        runtimeModeValue.value = safeRuntimeMode(
          routing,
          runtimeModeFromProjection(routing)
        );
        return;
      }
      if (!runtimeModeAvailable(runtimeModeValue.value, routing)) {
        runtimeModeValue.value = safeRuntimeMode(
          routing,
          runtimeModeFromProjection(routing)
        );
      }
    },
    { deep: true }
  );

  function clearCurrentChatDraft() {
    chatInput.value = "";
    drafts.clear(selectedSessionId.value);
    notice.value = "已清空当前草稿";
  }

  function discardChanges() {
    sessionTitleDraft.value = selectedSession.value?.title || "";
  }

  return {
    sessionStatus,
    sessionSearch,
    sessions,
    selectedSessionId,
    sessionDetail,
    materializeDraft: actions.materializeDraft,
    sessionTitleDraft,
    newSessionTitle,
    chatInput,
    pendingAttachments,
    attachmentUploading,
    chatBusy,
    chatStreamEvents,
    runtimeEvents,
    inSessionTaskDetails,
    streamingReply,
    lastTurnResult,
    turnWindow,
    incompleteTurn,
    chatDraftSessionIds: drafts.draftSessionIds,
    refreshChatDraftSessionIds: drafts.refreshIds,
    selectedSession,
    runtimeMode,
    runtimeRouting,
    canChangeRuntimeMode,
    turns,
    pendingUserQuestions,
    sessionTitleDirty,
    currentChatDraftLength,
    canSendChat,
    canAnswerPendingQuestion,
    chatReadOnlyReason,
    chatInputPlaceholder,
    clearCurrentChatDraft,
    discardChanges,
    ...postCommitRecovery,
    ...actions
  };
}
