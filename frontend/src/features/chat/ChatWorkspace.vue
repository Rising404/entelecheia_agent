<script setup>
import { computed, ref } from "vue";
import { AlertTriangle, Archive, FolderOpen, MessageSquare, Paperclip, RotateCcw, Save, Send, Trash2, X, XCircle } from "@lucide/vue";
import InSessionTaskCards from "../runtime/InSessionTaskCards.vue";
import SessionHistoryPlacement from "../workspace/SessionHistoryPlacement.vue";
import PendingUserQuestionDialog from "./PendingUserQuestionDialog.vue";
import RuntimeModeControl from "../runtime/RuntimeModeControl.vue";
import { roleText } from "./presentation";
import { sessionStatusText } from "../../shared/presentation";
import {
  MAX_ATTACHMENTS_PER_TURN,
  TASK_MODE_SEND_BLOCKED_NOTICE
} from "./useSessionChatActions";

const props = defineProps({
  session: { type: Object, required: true },
  title: { type: String, default: "" },
  titleDirty: Boolean,
  saving: Boolean,
  turns: { type: Array, default: () => [] },
  incompleteTurn: { type: Object, default: null },
  chatBusy: Boolean,
  pendingAttachments: { type: Array, default: () => [] },
  attachmentUploading: Boolean,
  streamEvents: { type: Array, default: () => [] },
  runtimeEvents: { type: Array, default: () => [] },
  insessionTaskDetails: { type: Array, default: () => [] },
  streamingReply: { type: String, default: "" },
  readOnlyReason: { type: String, default: "" },
  postCommitFailure: { type: Object, default: null },
  postCommitRecoveryBusy: Boolean,
  canRecoverPostCommit: Boolean,
  input: { type: String, default: "" },
  canSend: Boolean,
  pendingUserQuestions: { type: Array, default: () => [] },
  canAnswerPendingQuestion: Boolean,
  inputPlaceholder: { type: String, default: "" },
  draftLength: { type: Number, default: 0 },
  developerMode: Boolean,
  historyFolders: { type: Array, default: () => [] },
  runtimeMode: { type: String, default: "turn" },
  runtimeRouting: { type: Object, default: null },
  canChangeRuntimeMode: Boolean
});

const incompleteReasonText = computed(() => ({
  user_paused: "本轮已暂停，尚未形成正式回复。",
  host_stopped: "运行在宿主停止前中断，尚未形成正式回复。",
  provider_unavailable: "模型服务暂时不可用，尚未形成正式回复。",
  module_error: "处理过程中出现模块异常，尚未形成正式回复。",
  persistence_error: "结果保存未完成，尚未形成正式回复。",
  process_lost: "运行进程意外结束，尚未形成正式回复。",
  unknown: "本轮未能正常结束，尚未形成正式回复。"
}[props.incompleteTurn?.endReason] || "本轮未能形成正式回复。"));
const taskModeSendBlocked = computed(() => props.runtimeMode === "task");

const emit = defineEmits([
  "update:title",
  "update:input",
  "update:runtime-mode",
  "save-title",
  "archive",
  "unarchive",
  "trash",
  "restore",
  "move-session-history",
  "clear-draft",
  "attach-files",
  "remove-attachment",
  "send",
  "answer-pending-question",
  "retry-post-commit",
  "waive-post-commit",
  "composer-keydown"
]);

function formatAttachmentSize(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function onPickFiles(event) {
  const files = Array.from(event.target.files || []);
  // 重置输入框，确保再次选择同一文件时仍会触发 change 事件。
  event.target.value = "";
  if (files.length) emit("attach-files", files);
}

// 拖拽进来的文件。用计数而不是布尔量：拖过子元素时浏览器会先发子元素的
// dragenter、再发父元素的 dragleave，用布尔量会让提示在拖动途中闪烁。
const dragDepth = ref(0);
const draggingFiles = computed(() => dragDepth.value > 0);
const acceptsDrop = computed(() => !props.chatBusy && !props.attachmentUploading && props.canSend);

function carriesFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

function onDragEnter(event) {
  if (!carriesFiles(event)) return;
  dragDepth.value += 1;
}

function onDragOver(event) {
  if (!carriesFiles(event)) return;
  // 不 preventDefault 的话，浏览器会用自己的默认行为打开这个文件。
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = acceptsDrop.value ? "copy" : "none";
  }
}

function onDragLeave(event) {
  if (!carriesFiles(event)) return;
  dragDepth.value = Math.max(0, dragDepth.value - 1);
}

function onDrop(event) {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth.value = 0;
  if (!acceptsDrop.value) return;
  const files = Array.from(event.dataTransfer?.files || []);
  if (files.length) emit("attach-files", files);
}
</script>

<template>
  <div
    class="relative flex h-full min-h-0 flex-col gap-4 max-[780px]:h-auto"
    @dragenter="onDragEnter"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
  >
    <div
      v-if="draggingFiles"
      class="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-lg border-2 border-dashed"
      :class="acceptsDrop
        ? 'border-accent-solid bg-accent-bg/85 text-accent'
        : 'border-warn-line bg-warn-bg/85 text-warn'"
      data-testid="attachment-drop-overlay"
    >
      <div class="flex items-center gap-2 text-sm font-medium">
        <Paperclip :size="18" />
        <span>{{ acceptsDrop ? `松手上传，最多 ${MAX_ATTACHMENTS_PER_TURN} 个附件` : "当前无法添加附件" }}</span>
      </div>
    </div>
    <header class="flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
      <div class="min-w-[260px] flex-1">
        <input :value="title" class="title-input" @input="$emit('update:title', $event.target.value)" />
        <!-- 会话状态、工作目录、历史归档：都是"这个会话是什么"的元信息，
             不是对话本身。默认收起，需要时随开发者视图一起出现。 -->
        <div v-if="developerMode" class="mt-2 flex flex-wrap items-center gap-2 text-sm text-ink-3">
          <span class="rounded-md border border-line bg-surface px-2 py-1">{{ sessionStatusText(session.status) }}</span>
          <span class="flex max-w-full items-center gap-1 rounded-md border border-line bg-surface px-2 py-1" :title="session.working_dir || '未绑定工作目录'">
            <FolderOpen :size="14" />
            <span class="truncate">{{ session.working_dir || '未绑定工作目录' }}</span>
          </span>
          <SessionHistoryPlacement
            :session="session"
            :folders="historyFolders"
            :disabled="saving || chatBusy"
            @move="$emit('move-session-history', $event)"
          />
        </div>
      </div>
      <!-- 重命名、归档、回收站都移到了左侧会话列表的右键菜单：
           要对哪个会话做事，手已经指在那一行上了。 -->
    </header>

    <section class="message-stream max-[780px]:min-h-64 max-[780px]:flex-none">
      <article v-for="turn in turns" :key="turn.turn_idx" class="message-card" :class="turn.role">
        <div class="mb-1 text-xs font-semibold text-ink-3">{{ roleText(turn.role) }}</div>
        <p class="whitespace-pre-wrap">{{ turn.content }}</p>
      </article>
      <article v-if="streamingReply" class="message-card assistant" aria-live="polite" aria-label="模型正在回复">
        <div class="mb-1 text-xs font-semibold text-ink-3">助手</div>
        <p class="whitespace-pre-wrap">{{ streamingReply }}<span class="ml-0.5 inline-block animate-pulse">▍</span></p>
      </article>

      <article v-if="incompleteTurn" class="message-card border-warn-line bg-warn-bg text-warn" aria-live="polite" data-testid="incomplete-turn-status">
        <div class="mb-1 text-xs font-semibold">本轮未完成</div>
        <p class="whitespace-pre-wrap">{{ incompleteReasonText }}</p>
        <code v-if="incompleteTurn.errorCode" class="mt-2 block text-xs text-warn">{{ incompleteTurn.errorCode }}</code>
      </article>

      <div v-if="turns.length === 0 && !incompleteTurn && !chatBusy && !streamingReply" class="empty min-h-60"><MessageSquare :size="30" /><span>开始一轮真实会话</span></div>
    </section>

    <InSessionTaskCards :tasks="insessionTaskDetails" />

    <div v-if="readOnlyReason" class="flex items-center gap-2 rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-sm text-warn"><AlertTriangle :size="16" /><span>{{ readOnlyReason }}</span></div>
    <div v-if="postCommitFailure" class="flex flex-wrap items-center gap-2" data-testid="post-commit-recovery">
      <span class="text-sm text-ink-3">失败项：{{ postCommitFailure.label }}。可以重试，或确认跳过这些失败项。</span>
      <button
        class="cmd" type="button" data-testid="retry-post-commit"
        :disabled="!canRecoverPostCommit || postCommitRecoveryBusy || saving || chatBusy"
        @click="$emit('retry-post-commit')"
      ><RotateCcw :size="16" />{{ postCommitRecoveryBusy ? "处理中" : "重试失败项" }}</button>
      <button
        class="cmd" type="button" data-testid="waive-post-commit"
        :disabled="!canRecoverPostCommit || postCommitRecoveryBusy || saving || chatBusy"
        @click="$emit('waive-post-commit')"
      >确认跳过…</button>
    </div>

    <RuntimeModeControl
      :model-value="runtimeMode"
      :routing="runtimeRouting"
      :disabled="!canChangeRuntimeMode"
      @update:model-value="$emit('update:runtime-mode', $event)"
    />

    <PendingUserQuestionDialog
      v-if="pendingUserQuestions.length"
      :questions="pendingUserQuestions"
      :input="input"
      :can-answer="canAnswerPendingQuestion"
      :busy="chatBusy"
      @update:input="$emit('update:input', $event)"
      @send="$emit('answer-pending-question')"
      @composer-keydown="$emit('composer-keydown', $event)"
    />

    <form v-else class="composer" @submit.prevent="$emit('send')">
      <textarea
        :value="input"
        class="composer-input col-span-2"
        rows="3"
        :disabled="!canSend"
        :placeholder="inputPlaceholder"
        @input="$emit('update:input', $event.target.value)"
        @keydown="$emit('composer-keydown', $event)"
      />
      <ul v-if="pendingAttachments.length" class="col-span-2 flex flex-wrap gap-2">
        <li
          v-for="item in pendingAttachments"
          :key="item.attachment_id"
          class="flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs"
          :class="item.readable === false ? 'border-warn-line bg-warn-bg text-warn' : 'border-line bg-surface'"
        >
          <Paperclip :size="13" />
          <span class="max-w-52 truncate">{{ item.name }}</span>
          <span class="text-ink-3">{{ formatAttachmentSize(item.size_bytes) }}</span>
          <span v-if="item.readable === false" title="该类型暂不支持读取内容">· 仅记录</span>
          <button type="button" class="text-ink-3 hover:text-danger-text" :disabled="chatBusy"
                  @click="$emit('remove-attachment', item.attachment_id)"><X :size="12" /></button>
        </li>
      </ul>
      <div class="col-span-2 flex flex-wrap items-center gap-2">
        <label class="cmd cursor-pointer" :class="{ 'opacity-50': chatBusy || attachmentUploading }">
          <Paperclip :size="16" />{{ attachmentUploading ? "上传中" : "附件" }}
          <input type="file" multiple class="hidden" :disabled="chatBusy || attachmentUploading"
                 @change="onPickFiles" />
        </label>
        <span v-if="pendingAttachments.length" class="self-center text-xs text-ink-3">
          {{ pendingAttachments.length }} / {{ MAX_ATTACHMENTS_PER_TURN }}
        </span>
        <span class="min-w-0 flex-1" />
        <button class="cmd" type="button" :disabled="!draftLength" @click="$emit('clear-draft')"><XCircle :size="16" />清空草稿</button>
        <span v-if="draftLength" class="self-center text-xs text-ink-3">草稿 {{ draftLength }} 字</span>
        <!-- 保持按钮可点击，才能在用户尝试发送时解释原因；真正的零副作用拦截位于
             sendChat 公共边界，title 只负责提前说明当前状态。 -->
        <button
          class="cmd primary"
          type="submit"
          :disabled="chatBusy || !canSend || !input.trim()"
          :title="taskModeSendBlocked ? TASK_MODE_SEND_BLOCKED_NOTICE : undefined"
        >
          <Send :size="16" />{{ chatBusy ? "发送中" : "发送" }}
        </button>
      </div>
    </form>
  </div>
</template>
