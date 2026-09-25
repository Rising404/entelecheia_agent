<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import {
  ChevronLeft,
  ChevronRight,
  FileText,
  FolderOpen,
  Plus,
  Settings,
  XCircle
} from "@lucide/vue";
import { api } from "./api";
import { useSystemStatus } from "./composables/useSystemStatus";
import { usePersistentRef } from "./composables/usePersistentRef";
import { useModelProfiles } from "./composables/useModelProfiles";
import { useConfig } from "./composables/useConfig";
import { useUnsavedGuard } from "./composables/useUnsavedGuard";
import { useRuntimeRecoverySignals } from "./composables/useRuntimeRecoverySignals";
import SessionTree from "./features/workspace/SessionTree.vue";
import { createWorkspaceStore } from "./features/workspace/store";
import { useProjects } from "./features/workspace/useProjects";
import { useWorkspaceDetailSelection } from "./features/workspace/useWorkspaceDetailSelection";
import WorkspaceContextPanel from "./features/workspace/WorkspaceContextPanel.vue";
import WorkspaceFiles from "./features/workspace/WorkspaceFiles.vue";
import ChatWorkspace from "./features/chat/ChatWorkspace.vue";
import { useChatFeature } from "./features/chat/useChatFeature";
import SessionContextPanel from "./features/session-context/SessionContextPanel.vue";
import { useSessionContextFeature } from "./features/session-context/useSessionContextFeature";
import DocumentDetailPanel from "./features/documents/DocumentDetailPanel.vue";
import { useDocumentsFeature } from "./features/documents/useDocumentsFeature";
import SettingsPanel from "./features/settings/SettingsPanel.vue";
import SessionContextMenu from "./features/workspace/SessionContextMenu.vue";
import BackgroundLayer from "./components/BackgroundLayer.vue";
import { useAppearance } from "./composables/useAppearance";
import { useBackgroundBox } from "./composables/useBackgroundBox";
import RightInspector from "./features/inspector/RightInspector.vue";
import AppNav from "./components/AppNav.vue";
import WelcomeGuide from "./components/WelcomeGuide.vue";
import FeedbackBanners from "./components/FeedbackBanners.vue";

const mode = ref("chat");
const loading = ref(false);
const saving = ref(false);
const error = ref("");
const notice = ref("");
const debugOpen = ref(false);
const apiWakeBusy = ref(false);
const canWakeLocalApi = computed(() => (
  typeof globalThis.personagraphDesktop?.ensureApiSidecar === "function"
));
const workspaceClient = {
  listFolders: (...args) => api.listFolders(...args),
  listSessions: (...args) => api.listSessions(...args),
  listDocuments: (...args) => api.listDocuments(...args),
  createSession: (...args) => api.createSession(...args),
  patchSession: (...args) => api.patchSession(...args),
  createFolder: (...args) => api.createFolder(...args),
  patchFolder: (...args) => api.patchFolder(...args),
  deleteFolder: (...args) => api.deleteFolder(...args)
};
const workspaceFilesClient = {
  listWorkspaceFiles: (...args) => api.listWorkspaceFiles(...args),
  createWorkspaceEntry: (...args) => api.createWorkspaceEntry(...args),
  readWorkspaceFile: (...args) => api.readWorkspaceFile(...args),
  writeWorkspaceFile: (...args) => api.writeWorkspaceFile(...args),
  deleteWorkspaceEntry: (...args) => api.deleteWorkspaceEntry(...args)
};
// 用户按过一次就该一直生效，所以这一个记得住。
const developerMode = usePersistentRef("developerMode", false);
// 设置分节。放在左栏而不是堆成一长页：几组配置之间没有先后关系，
// 竖着排在一起只会让人每次都从头滚一遍找目标。
const SETTINGS_SECTIONS = [
  { key: "model", label: "任务模型配置" },
  { key: "vision", label: "视觉模型配置" },
  { key: "tiers", label: "分档模型配置" },
  { key: "workspace", label: "工作目录" },
  { key: "appearance", label: "外观" }
];
const settingsSection = usePersistentRef("settings.section", "model");
// 面板通透度：0 = 完全不透（看不到背景图），1 = 完全撤下（背景图全露，但小字会看不清）
const sheetTranslucency = usePersistentRef("appearance.sheetTranslucency", 0.45);
// 背景铺在哪：整个窗口 / 对话区 / 对话区但不含输入框
const backgroundScope = usePersistentRef("appearance.backgroundScope", "content");
useBackgroundBox(backgroundScope);
watch(backgroundScope, (value) => {
  // 贴合对话区时跟着纸面一起圆角，铺满窗口时不需要
  document.documentElement.style.setProperty(
    "--bg-radius", value === "window" ? "0px" : "14px"
  );
}, { immediate: true });
watch(sheetTranslucency, (value) => {
  const clamped = Math.min(1, Math.max(0, Number(value) || 0));
  document.documentElement.style.setProperty("--sheet-alpha", String(1 - clamped));
}, { immediate: true });

// refreshConfig 用箭头包一层：loadConfig 在下面才声明，直接传引用会踩进暂时性死区。
const modelProfiles = useModelProfiles({ api, showError, refreshConfig: () => loadConfig() });
// 离开设置页时把露出来的密钥收回去：显示是一个动作，不该跨页面留着。
watch(mode, (next, previous) => {
  if (previous === "settings") modelProfiles.hideAll();
  if (next === "settings") modelProfiles.load();
});

// 会话右键菜单。右键先选中再操作：既有的归档/回收站动作都作用于当前选中的会话，
// 让菜单去改那套语义，比让它先把会话选中要绕得多。
const sessionTreeRef = ref(null);
const sessionMenu = ref(null);
const sessionMenuTarget = computed(() =>
  sessions.value.find((item) => item.id === sessionMenu.value?.sessionId) || null
);

function openSessionMenu({ sessionId, x, y }) {
  if (selectedSessionId.value !== sessionId) guardedSelectSession(sessionId);
  sessionMenu.value = { sessionId, x, y };
}

function closeSessionMenu() {
  sessionMenu.value = null;
}

async function purgeFromMenu() {
  const target = sessionMenuTarget.value;
  if (!target) return;
  // 这一步没有回收站兜底，所以问一次，并且把名字念出来。
  if (!globalThis.confirm(`彻底删除「${target.title || "未命名会话"}」？此操作不可撤销。`)) return;
  try {
    await api.purgeSession(target.id);
    await loadSessions();
  } catch (err) {
    showError(err);
  }
}

async function emptyTrash() {
  if (!globalThis.confirm("清空回收站？其中的会话将被彻底删除，不可撤销。")) return;
  try {
    const { purged } = await api.emptySessionTrash();
    notice.value = `已彻底删除 ${purged} 个会话`;
    await loadSessions();
  } catch (err) {
    showError(err);
  }
}

// 中间那栏认的是 sessionDetail，不是 selectedSessionId。草稿还没有会话，
// 就先给它一个空壳：id 留空，凡是按 id 取数据的地方都会自己短路，
// 输入框却已经能用了——不然用户要在"还看着上一条会话"的界面里写第一句话。
const DRAFT_TITLE = "新会话";

function draftSessionDetail() {
  return {
    session: {
      id: "",
      title: DRAFT_TITLE,
      status: "active",
      working_dir: null
    },
    turns: [],
    pending_user_questions: []
  };
}

const directoryPickerAvailable = typeof globalThis.personagraphDesktop?.chooseDirectory === "function";

function startDraft(workingDir = "") {
  if (saving.value || chatBusy.value || !confirmDiscardChanges()) return;
  projectsStore.startDraft(workingDir);
  selectedSessionId.value = "";
  sessionDetail.value = draftSessionDetail();
  // 标题也要跟着换：留着上一条会话的标题，未保存提示会立刻误报一次。
  sessionTitleDraft.value = DRAFT_TITLE;
  chatInput.value = "";
  mode.value = "chat";
}

function discardDraft() {
  if (saving.value || !confirmDiscardChanges()) return;
  projectsStore.discardDraft();
  sessionDetail.value = null;
  sessionTitleDraft.value = "";
  chatInput.value = "";
}

async function chooseDraftDirectory() {
  const draft = projectsStore.draft.value;
  if (!draft || saving.value || projectsStore.draftDirectoryLocked.value || !directoryPickerAvailable) return;
  try {
    const result = await globalThis.personagraphDesktop.chooseDirectory();
    if (result?.path && projectsStore.draft.value === draft && !saving.value) {
      projectsStore.setDraftDirectory(result.path);
    }
  } catch (err) { showError(err); }
}

async function chooseDefaultDirectory() {
  if (!directoryPickerAvailable || configSaving.value) return;
  try {
    const result = await globalThis.personagraphDesktop.chooseDirectory();
    if (result?.path) configForm.default_projects_dir = result.path;
  } catch (err) { showError(err); }
}

async function renameProject({ path, name }) {
  if (await projectsStore.rename(path, name)) notice.value = "项目已改名";
}

async function forgetProject(path) {
  if (!globalThis.confirm("不再在列表里显示这个项目？目录和其中的会话都不会被删除。")) return;
  if (await projectsStore.forget(path)) notice.value = "已从列表移除";
}

function renameFromMenu() {
  const target = sessionMenuTarget.value;
  if (target) sessionTreeRef.value?.startRename(target.id, target.title || "");
}
// 右栏整列只服务于排查，默认不存在。栅格跟着少一列，否则会留下一条 340px 的空白。
//
// 四种组合逐个写全，不用模板字符串拼：Tailwind 是扫源码里的字面类名生成 CSS 的，
// 拼出来的类名它看不见，运行时就没有对应样式。
const GRID_COLUMNS = {
  "open|dev": "grid-cols-[320px_minmax(0,1fr)_340px] max-[1080px]:grid-cols-[280px_minmax(0,1fr)]",
  "open|plain": "grid-cols-[320px_minmax(0,1fr)] max-[1080px]:grid-cols-[280px_minmax(0,1fr)]",
  "collapsed|dev": "grid-cols-[56px_minmax(0,1fr)_340px] max-[1080px]:grid-cols-[56px_minmax(0,1fr)]",
  "collapsed|plain": "grid-cols-[56px_minmax(0,1fr)] max-[1080px]:grid-cols-[56px_minmax(0,1fr)]"
};
const gridColumns = computed(() => GRID_COLUMNS[
  `${historyPanelOpen.value ? "open" : "collapsed"}|${developerMode.value ? "dev" : "plain"}`
]);
const historyPanelOpen = ref(true);
const localDirectoryOpen = ref(true);
// 系统状态 + 模型配置：逻辑抽到 composables（重构 2026-07-11）
const {
  systemStatus,
  statusLoaded,
  apiReachable,
  apiOk,
  modelConfigured,
  loadStatus
} = useSystemStatus();
const runtimeRoutingStatus = computed(() => systemStatus.value?.runtime_routing || null);
const { configForm, configHasKey,
  configHasVisionKey, configLegacyOfficeStatus,
        tierForm, tierEffective, tierLoaded, updateTier,
        configLoading, configSaving, configMessage,
        loadConfig, saveConfig } = useConfig(loadStatus);
const workspaceStore = createWorkspaceStore({ client: workspaceClient })();
const projectsStore = useProjects({ apiClient: api, showError: (err) => showError(err) });
const appearance = useAppearance({ showError: (err) => showError(err) });
const {
  folders,
  sessions,
  selectedFolderId,
  historyFolderOptions,
  selectedSessionId,
  workspaceDocuments,
  contentsLoading
} = storeToRefs(workspaceStore);

const {
  sessionStatus,
  sessionSearch,
  sessionDetail,
  sessionTitleDraft,
  newSessionTitle,
  chatInput,
  pendingAttachments,
  attachmentUploading,
  attachFiles,
  removePendingAttachment,
  chatBusy,
  chatStreamEvents,
  runtimeEvents,
  inSessionTaskDetails,
  streamingReply,
  incompleteTurn,
  chatDraftSessionIds,
  refreshChatDraftSessionIds,
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
  postCommitFailure,
  postCommitRecoveryBusy,
  canRecoverPostCommit,
  retryPostCommit,
  waivePostCommit,
  clearCurrentChatDraft,
  discardChanges: discardChatChanges,
  loadSessions,
  guardedReloadSessionsFromSearch,
  guardedClearSessionSearch,
  guardedSetSessionStatus,
  guardedSelectSession,
  saveSessionTitle,
  renameSession,
  archiveSession,
  unarchiveSession,
  trashSession,
  restoreTrashedSession,
  sendChat,
  sendPendingUserAnswer,
  handleChatComposerKeydown,
  reconcileRuntimeState
} = useChatFeature({
  api,
  mode,
  loading,
  saving,
  notice,
  workspace: workspaceStore,
  projects: projectsStore,
  sessions,
  selectedSessionId,
  // 守卫与引用刷新函数在下方声明。
  confirmDiscardChanges: () => confirmDiscardChanges(),
  confirmPostCommitWaiver: (message) => globalThis.confirm(message),
  clearMessage,
  showError,
  runtimeRoutingStatus
});

const {
  sessionContext,
  contextLoading,
  contextBusy,
  contextError,
  selectedState: selectedSessionState,
  stateExplanation,
  repairPreview,
  repairApplyEnabled,
  repairPreviewStale,
  loadSessionContext,
  explainState,
  exportSessionContext,
  clearSessionContext,
  createCorrection,
  previewRepair,
  applyRepair
} = useSessionContextFeature({
  api,
  selectedSessionId,
  notice
});

watch(() => turns.value.length, (count, previousCount) => {
  if (count !== previousCount && selectedSessionId.value) loadSessionContext();
});

const {
  documentForm,
  newDocument,
  selectedDocument,
  documentDirty,
  canChooseDocumentPath,
  documentIngestJobs,
  documentIngestJobsLoading,
  startDocumentIngestJobs,
  stopDocumentIngestJobs,
  reconcileDocumentIngestJobs,
  retryDocumentIngestJob,
  discardChanges: discardDocumentChanges,
  loadDocuments,
  guardedSelectDocument,
  chooseDocumentPath,
  createDocument,
  saveDocument,
  deleteDocument
} = useDocumentsFeature({
  api,
  loading,
  saving,
  notice,
  selectedSessionId,
  onDocumentIngestSucceeded: (_job, sessionId) => workspaceStore.refreshContents(sessionId),
  // 守卫在下方创建；此闭包只会在初始化完成后运行。
  confirmDiscardChanges: () => confirmDiscardChanges(),
  clearMessage,
  showError
});

const debugPayload = computed(() => ({
  apiBase: api.base,
  mode: mode.value,
  selectedSession: selectedSession.value,
  sessionContext: sessionContext.value,
  selectedDocument: selectedDocument.value
}));
const {
  hasUnsavedChanges,
  unsavedSummary,
  isInteractionBusy,
  confirmDiscardChanges,
  discardDraftsFromBanner,
  handleBeforeUnload
} = useUnsavedGuard({
  dirtySources: {
    "会话名称": sessionTitleDirty,
    "未发送的新会话": computed(() => projectsStore.draftActive.value && (
      Boolean(chatInput.value.trim()) || projectsStore.draftAttachments.value.length > 0
    )),
    "文档信息": documentDirty
  },
  busySignals: [saving, chatBusy, contextBusy],
  clearMessage,
  discard: resetUnsavedChanges,
  notice
});
const workspaceTitle = computed(() => mode.value === "settings" ? "设置" : "工作区");
const runtimeRecoverySignals = useRuntimeRecoverySignals({
  reconcile: reconcileRuntimeState,
  onError: showError
});
// 搜索词也要透给 project 列表：之前只有会话走了查询，project 树是另一条不带查询的
// 请求，所以搜东西的时候树纹丝不动。
watch(
  [sessions, sessionStatus, sessionSearch],
  () => { void projectsStore.load(sessionStatus.value, sessionSearch.value); },
  { deep: false }
);

onMounted(async () => {
  appearance.load();
  globalThis.addEventListener?.("beforeunload", handleBeforeUnload);
  refreshChatDraftSessionIds();
  startDocumentIngestJobs();
  await refreshAll();
  await projectsStore.load(sessionStatus.value, sessionSearch.value);
  runtimeRecoverySignals.start();
});

onBeforeUnmount(() => {
  globalThis.removeEventListener?.("beforeunload", handleBeforeUnload);
  runtimeRecoverySignals.stop();
  stopDocumentIngestJobs();
});

async function guardedRefreshAll() {
  if (isInteractionBusy.value) {
    error.value = "当前有保存或对话运行中，完成后再刷新。";
    return;
  }
  if (!confirmDiscardChanges()) return;
  await refreshAll();
}

async function refreshAll() {
  await Promise.allSettled([
    loadStatus(),
    loadSessions(),
    loadSessionContext(),
    loadDocuments(),
    reconcileDocumentIngestJobs()
  ]);
}

async function wakeLocalApi() {
  if (!canWakeLocalApi.value || apiWakeBusy.value) return;
  apiWakeBusy.value = true;
  clearMessage();
  try {
    const result = await globalThis.personagraphDesktop.ensureApiSidecar();
    if (!result?.ok) {
      throw new Error(
        result?.mode === "starting"
          ? "本地服务仍在启动，请稍后再试"
          : "本地服务启动失败，请查看应用日志"
      );
    }
    await refreshAll();
    if (!apiReachable.value) {
      throw new Error("本地服务已启动，但前端仍未连接成功");
    }
    notice.value = "本地服务已唤醒";
  } catch (err) {
    showError(err);
  } finally {
    apiWakeBusy.value = false;
  }
}

async function runWorkspaceHistoryAction(successMessage, action) {
  saving.value = true;
  clearMessage();
  try {
    await action();
    await loadSessions(selectedSessionId.value);
    notice.value = successMessage;
  } catch (err) {
    showError(err);
  } finally {
    saving.value = false;
  }
}

function createWorkspaceFolder({ name, parentId }) {
  return runWorkspaceHistoryAction("文件夹已创建", () => workspaceStore.createFolder({ name, parentId }));
}

function renameWorkspaceFolder({ folderId, name }) {
  return runWorkspaceHistoryAction("文件夹名称已更新", () => workspaceStore.renameFolder(folderId, name));
}

function moveWorkspaceFolder({ folderId, parentId }) {
  return runWorkspaceHistoryAction("文件夹位置已更新", () => workspaceStore.moveFolder(folderId, parentId));
}

function setWorkspaceFolderStatus({ folderId, status }) {
  const message = status === "active" ? "文件夹已恢复" : status === "archived" ? "文件夹已归档" : "文件夹已移入回收站";
  return runWorkspaceHistoryAction(message, () => workspaceStore.setFolderStatus(folderId, status));
}

function deleteWorkspaceFolder(folderId) {
  const folder = workspaceStore.selectedFolder;
  if (!globalThis.confirm(`删除空文件夹「${folder?.name || "未命名文件夹"}」？`)) return;
  return runWorkspaceHistoryAction("空文件夹已删除", async () => {
    await workspaceStore.deleteFolder(folderId);
    workspaceStore.selectFolder("");
  });
}

async function moveSelectedSessionToHistory({ sessionId, folderId }) {
  if (!selectedSession.value || selectedSession.value.id !== sessionId) return;
  if ((selectedSession.value.folder_id || null) === folderId) return;
  if (!confirmDiscardChanges()) return;
  const message = folderId ? "会话已移动到历史文件夹" : "会话已移至未归档会话";
  return runWorkspaceHistoryAction(message, async () => {
    const session = await workspaceStore.moveSession(sessionId, folderId);
    workspaceStore.selectFolder(folderId || "");
    return session;
  });
}

function clearMessage() {
  error.value = "";
  notice.value = "";
}

function showError(err) {
  const base = err?.message || String(err);
  // 后端在 error.details.hint 里给可操作提示（如收录失败的原因与建议），一并展示。
  const hint = err?.error?.details?.hint || err?.payload?.error?.details?.hint || "";
  error.value = hint ? `${base}：${hint}` : base;
}


// 弹层内文档动作后回刷工作区面板（否则摘要显示旧状态）
async function refreshWorkspaceContents() {
  if (selectedSession.value) await workspaceStore.refreshContents(selectedSession.value.id);
}
async function overlaySaveDocument() { await saveDocument(); await refreshWorkspaceContents(); }
async function overlayDeleteDocument() {
  await deleteDocument();
  workspaceDetailKind.value = null;
  await refreshWorkspaceContents();
}
async function overlayCreateDocument() {
  if (!await createDocument()) return;
  workspaceDetailKind.value = null;
}
async function retryWorkspaceDocumentIngest(job) {
  clearMessage();
  if (await retryDocumentIngestJob(job)) notice.value = "文档收录任务已重新排队";
}

// 文档详情留在当前工作区内，不单独占导航栏。
const workspaceDetailKind = ref(null);   // 可选值：'document' | 'add-document' | null
const {
  openWorkspaceDocumentDetail,
  invalidateWorkspaceDetailSelection
} = useWorkspaceDetailSelection({
  workspaceDetailKind,
  selectedSessionId,
  guardedSelectDocument
});

function openWorkspaceDocuments() {
  if (!selectedSession.value) return;
  invalidateWorkspaceDetailSelection();
  newDocument.session_id = selectedSession.value.id;
  workspaceDetailKind.value = "add-document";
}

function closeWorkspaceDetail() {
  if (!confirmDiscardChanges()) return;
  invalidateWorkspaceDetailSelection();
}

function switchMode(nextMode) {
  if (mode.value === nextMode) return;
  if (!confirmDiscardChanges()) return;
  mode.value = nextMode;
  if (nextMode === "settings") loadConfig();
}

function resetUnsavedChanges() {
  discardChatChanges();
  discardDocumentChanges();
  if (projectsStore.draftActive.value) {
    projectsStore.setDraftAttachments([]);
    chatInput.value = "";
  }
}

function updateDraftField(target, { field, value }) {
  if (!Object.prototype.hasOwnProperty.call(target, field)) return;
  target[field] = value;
}

</script>

<template>
  <BackgroundLayer
    :background="appearance.background.value"
    :object-url="appearance.objectUrl.value"
  />
  <main class="h-screen overflow-hidden text-ink max-[780px]:h-auto max-[780px]:overflow-visible">
    <div
      class="grid h-screen grid-rows-[minmax(0,1fr)] max-[780px]:block max-[780px]:h-auto"
      :class="gridColumns"
    >
      <aside
        class="shell-dark flex h-screen flex-col border-r border-glass-line max-[780px]:h-auto max-[780px]:border-b max-[780px]:border-r-0"
        :class="historyPanelOpen ? 'p-4' : 'p-2'"
      >
        <div class="min-h-0 flex-1 overflow-y-auto">
        <div class="mb-3 flex" :class="historyPanelOpen ? 'justify-end' : 'justify-center'">
          <button
            class="icon-btn"
            type="button"
            :title="historyPanelOpen ? '收起历史栏' : '展开历史栏'"
            :aria-label="historyPanelOpen ? '收起历史栏' : '展开历史栏'"
            :aria-expanded="historyPanelOpen"
            @click="historyPanelOpen = !historyPanelOpen"
          >
            <ChevronLeft v-if="historyPanelOpen" :size="18" />
            <ChevronRight v-else :size="18" />
          </button>
        </div>
        <div v-show="historyPanelOpen">
        <AppNav :title="workspaceTitle" />

        <nav v-if="mode === 'settings'" class="space-y-1" aria-label="设置分节">
          <button
            v-for="section in SETTINGS_SECTIONS"
            :key="section.key"
            class="task-row"
            :class="{ active: settingsSection === section.key }"
            type="button"
            @click="settingsSection = section.key"
          >
            <span class="min-w-0 truncate font-medium">{{ section.label }}</span>
          </button>
        </nav>

        <SessionTree
          v-if="mode === 'chat'"
          ref="sessionTreeRef"
          @session-context="openSessionMenu"
          @empty-trash="emptyTrash"
          @rename-session="({ sessionId, title }) => renameSession(sessionId, title)"
          v-model:search="sessionSearch"
          :status="sessionStatus"
          :projects="projectsStore.projects.value"
          :unbound="projectsStore.unbound.value"
          :selected-id="selectedSessionId"
          :draft-active="projectsStore.draftActive.value"
          :draft-working-dir="projectsStore.draftWorkingDir.value"
          :draft-directory-locked="projectsStore.draftDirectoryLocked.value"
          :directory-picker-available="directoryPickerAvailable"
          :draft-session-ids="chatDraftSessionIds"
          :loading="loading"
          :saving="saving"
          @start-draft="startDraft"
          @choose-draft-directory="chooseDraftDirectory"
          @update-draft-directory="projectsStore.setDraftDirectory($event)"
          @discard-draft="discardDraft"
          @rename-project="renameProject"
          @pin-project="projectsStore.setPinned($event.path, $event.pinned)"
          @reorder-projects="projectsStore.reorder($event)"
          @set-status="guardedSetSessionStatus"
          @search="guardedReloadSessionsFromSearch"
          @clear-search="guardedClearSessionSearch"
          @select-session="guardedSelectSession"
        />

        </div>
        </div>

        <!-- 设置不是一个"工作台",是一个偶尔要去的地方。它固定在底部，
             不跟会话列表一起滚走，也不占据顶部的注意力。 -->
        <div class="mt-3 shrink-0 border-t border-line pt-3">
          <button
            class="seg w-full"
            :class="{ active: mode === 'settings' }"
            type="button"
            :title="mode === 'settings' ? '返回工作区' : '设置'"
            @click="switchMode(mode === 'settings' ? 'chat' : 'settings')"
          >
            <Settings :size="15" />
            <span v-if="historyPanelOpen">{{ mode === 'settings' ? '返回工作区' : '设置' }}</span>
          </button>
        </div>
      </aside>

      <section class="sheet m-4 flex min-h-0 min-w-0 flex-col overflow-y-auto p-5 max-[780px]:m-3 max-[780px]:overflow-visible max-[780px]:p-4">
        <FeedbackBanners
          :mode="mode"
          :error="error"
          :notice="notice"
          :status-loaded="statusLoaded"
          :api-reachable="apiReachable"
          :can-wake-api="canWakeLocalApi"
          :waking-api="apiWakeBusy"
          :model-configured="modelConfigured"
          :has-unsaved-changes="hasUnsavedChanges"
          :unsaved-summary="unsavedSummary"
          @discard="discardDraftsFromBanner"
          @open-settings="switchMode('settings')"
          @wake-api="wakeLocalApi"
        />

        <ChatWorkspace
          v-if="mode === 'chat' && selectedSession"
          v-model:title="sessionTitleDraft"
          v-model:input="chatInput"
          v-model:runtime-mode="runtimeMode"
          :runtime-routing="runtimeRouting"
          :can-change-runtime-mode="canChangeRuntimeMode"
          :pending-attachments="projectsStore.draftActive.value ? projectsStore.draftAttachments.value : pendingAttachments"
          :attachment-uploading="attachmentUploading"
          @attach-files="attachFiles"
          @remove-attachment="removePendingAttachment"
          :session="selectedSession"
          :title-dirty="sessionTitleDirty"
          :saving="saving"
          :turns="turns"
          :pending-user-questions="pendingUserQuestions"
          :incomplete-turn="incompleteTurn"
          :chat-busy="chatBusy"
          :stream-events="chatStreamEvents"
          :runtime-events="runtimeEvents"
          :insession-task-details="inSessionTaskDetails"
          :streaming-reply="streamingReply"
          :read-only-reason="chatReadOnlyReason"
          :post-commit-failure="postCommitFailure"
          :post-commit-recovery-busy="postCommitRecoveryBusy"
          :can-recover-post-commit="canRecoverPostCommit"
          @retry-post-commit="retryPostCommit"
          @waive-post-commit="waivePostCommit"
          :can-send="canSendChat"
          :can-answer-pending-question="canAnswerPendingQuestion"
          :input-placeholder="chatInputPlaceholder"
          :draft-length="currentChatDraftLength"
          :history-folders="historyFolderOptions"
          @save-title="saveSessionTitle"
          @archive="archiveSession"
          @unarchive="unarchiveSession"
          @trash="trashSession"
          @restore="restoreTrashedSession"
          @clear-draft="clearCurrentChatDraft"
          @move-session-history="moveSelectedSessionToHistory"
          @send="sendChat"
          @answer-pending-question="sendPendingUserAnswer"
          :developer-mode="developerMode"
          @composer-keydown="handleChatComposerKeydown"
        />

        <SettingsPanel
        v-model:developer-mode="developerMode"
          v-else-if="mode === 'settings'"
          :form="configForm"
          :directory-picker-available="directoryPickerAvailable"
          @choose-default-directory="chooseDefaultDirectory"
          :tier-form="tierForm"
          :tier-effective="tierEffective"
          :tier-loaded="tierLoaded"
          @update-tier="updateTier"
          :has-key="configHasKey"
          :has-vision-key="configHasVisionKey"
          :section="settingsSection"
          :background="appearance.background.value"
          :background-busy="appearance.busy.value"
          :sheet-translucency="sheetTranslucency"
          :background-scope="backgroundScope"
          @update:background-scope="backgroundScope = $event"
          @update:sheet-translucency="sheetTranslucency = $event"
          @background-pick="appearance.upload($event)"
          @background-clear="appearance.clear()"
          :profiles="modelProfiles.profiles"
          :revealed="modelProfiles.revealed"
          :profiles-busy="modelProfiles.busy.value"
          @profile-create="modelProfiles.create"
          @profile-update="modelProfiles.update"
          @profile-delete="modelProfiles.remove"
          @profile-activate="modelProfiles.activate"
          @profile-reveal="modelProfiles.reveal"
          @profile-hide="modelProfiles.hide"
          @profile-copy="modelProfiles.copy"
          :legacy-office-status="configLegacyOfficeStatus"
          :model-configured="modelConfigured"
          :model-control="systemStatus?.model_control || null"
          :runtime-events-status="systemStatus?.runtime_events || null"
          :loading="configLoading"
          :saving="configSaving"
          :message="configMessage"
          @update-field="updateDraftField(configForm, $event)"
          @save="saveConfig({ section: settingsSection })"
        />

        <WelcomeGuide
          v-else-if="mode === 'chat'"
          :model-configured="modelConfigured"
          @go-settings="switchMode('settings')"
        />

      </section>

      <RightInspector
        v-if="developerMode"
        :mode="mode"
        :system-status="systemStatus"
        :api-ok="apiOk"
        :loading="loading"
        :saving="saving"
        :chat-busy="chatBusy"
        :turns-count="turns.length"
        v-model:debug-open="debugOpen"
        :developer-mode="developerMode"
        :debug-payload="debugPayload"
        @refresh="guardedRefreshAll"
      >
        <template #local-directory>
          <WorkspaceFiles
            v-if="selectedSession"
            v-model:open="localDirectoryOpen"
            :session-id="selectedSession.id"
            :working-dir="selectedSession.working_dir || ''"
            :workspace-files-client="workspaceFilesClient"
          />
          <WorkspaceContextPanel
            v-if="selectedSession"
            class="mt-3"
            :documents="workspaceDocuments"
            :document-ingest-jobs="documentIngestJobs"
            :document-ingest-jobs-loading="documentIngestJobsLoading"
            :loading="contentsLoading"
            :saving="saving"
            :can-add-document="Boolean(selectedSession.working_dir)"
            @add-document="openWorkspaceDocuments"
            @open-document="openWorkspaceDocumentDetail"
            @retry-document-ingest="retryWorkspaceDocumentIngest"
          />
        </template>
        <template #chat-controls>
          <SessionContextPanel
            v-if="selectedSession"
            :session-context="sessionContext"
            :selected-state="selectedSessionState"
            :explanation="stateExplanation"
            :loading="contextLoading"
            :busy="contextBusy"
            :error="contextError"
            :repair-preview="repairPreview"
            :repair-apply-enabled="repairApplyEnabled"
            :repair-preview-stale="repairPreviewStale"
            @refresh="loadSessionContext"
            @explain="explainState($event.state, { includeExcerpt: $event.includeExcerpt })"
            @export="exportSessionContext($event)"
            @clear="clearSessionContext"
            @create-correction="createCorrection"
            @preview-repair="previewRepair"
            @apply-repair="applyRepair"
          />
        </template>
      </RightInspector>
    </div>

    <!-- 工作区内联文档详情：不单独占导航栏。 -->
    <SessionContextMenu
      v-if="sessionMenu"
      :session="sessionMenuTarget"
      :x="sessionMenu.x"
      :y="sessionMenu.y"
      :busy="saving"
      @close="closeSessionMenu"
      @rename="renameFromMenu"
      @archive="archiveSession"
      @unarchive="unarchiveSession"
      @trash="trashSession"
      @restore="restoreTrashedSession"
      @purge="purgeFromMenu"
    />

    <div v-if="workspaceDetailKind" class="ws-detail-backdrop" @click.self="closeWorkspaceDetail">
      <div class="ws-detail-panel">
        <button class="ws-detail-close" type="button" title="关闭" @click="closeWorkspaceDetail">
          <XCircle :size="20" />
        </button>

        <DocumentDetailPanel
          v-if="workspaceDetailKind === 'document' && selectedDocument"
          :document="selectedDocument"
          :form="documentForm"
          :dirty="documentDirty"
          :saving="saving"
          @update-field="updateDraftField(documentForm, $event)"
          @save="overlaySaveDocument"
          @delete="overlayDeleteDocument"
        />

        <form v-else-if="workspaceDetailKind === 'add-document'" class="space-y-3" @submit.prevent="overlayCreateDocument">
          <h2 class="flex items-center gap-2 text-lg font-semibold"><FileText :size="18" />收录文档到当前工作区</h2>
          <p class="text-sm text-ink-3">任务确认后会在后台解析、分块并建立检索覆盖；关闭窗口或重启应用不会丢失进度。当前收录不会自动生成摘要。</p>
          <div class="flex gap-2">
            <input v-model="newDocument.path" class="field flex-1" placeholder="文档路径" />
            <button v-if="canChooseDocumentPath" class="cmd" type="button" @click="chooseDocumentPath"><FolderOpen :size="15" />选择</button>
          </div>
          <button class="cmd primary" type="submit" :disabled="saving || !newDocument.path.trim()"><Plus :size="15" />提交收录任务</button>
        </form>
      </div>
    </div>
  </main>
</template>

<style scoped>
.ws-detail-backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 3rem 1rem;
  background: rgba(20, 22, 18, 0.35);
  overflow: auto;
}
.ws-detail-panel {
  position: relative;
  width: 100%;
  max-width: 640px;
  border-radius: 0.75rem;
  background: #fbfaf6;
  padding: 1.5rem;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
}
.ws-detail-close {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  color: #74766f;
}
.ws-detail-close:hover { color: #8f2d1c; }
</style>
