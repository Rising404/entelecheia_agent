<script setup>
import { AlertTriangle, CheckCircle2, Power, RotateCcw } from "@lucide/vue";
import { outcomeIssueCode, outcomeIssueText, outcomePrimaryIssue, outcomeRetryText, outcomeStatusText } from "../features/chat/presentation";

defineProps({
  mode: { type: String, required: true },
  error: { type: String, default: "" },
  errorOutcome: { type: Object, default: null },
  notice: { type: String, default: "" },
  statusLoaded: Boolean,
  apiReachable: { type: Boolean, default: true },
  canWakeApi: Boolean,
  wakingApi: Boolean,
  modelConfigured: Boolean,
  hasUnsavedChanges: Boolean,
  unsavedSummary: { type: String, default: "" }
});

defineEmits(["discard", "open-settings", "wake-api"]);
</script>

<template>
  <div
    v-if="statusLoaded && !apiReachable"
    class="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-sm text-warn"
    role="alert"
    data-testid="api-offline-banner"
  >
    <span class="flex min-w-0 items-center gap-2">
      <AlertTriangle class="shrink-0" :size="17" />
      本地服务未连接。桌面版可以在这里直接启动，不需要打开命令行。
    </span>
    <button
      v-if="canWakeApi"
      class="cmd"
      type="button"
      :disabled="wakingApi"
      @click="$emit('wake-api')"
    >
      <Power :size="15" />{{ wakingApi ? "正在启动…" : "唤醒本地服务" }}
    </button>
  </div>
  <div v-if="error" class="mb-3 flex items-start gap-2 rounded-md border border-danger-line bg-danger-bg px-3 py-2 text-sm text-danger">
    <AlertTriangle class="mt-0.5 shrink-0" :size="17" />
    <div class="min-w-0">
      <div class="break-words">{{ error }}</div>
      <div v-if="errorOutcome" class="mt-1 space-y-1 text-xs text-danger">
        <div>{{ outcomeStatusText(errorOutcome.status) }}<span v-if="outcomeRetryText(errorOutcome)"> · {{ outcomeRetryText(errorOutcome) }}</span></div>
        <template v-if="outcomePrimaryIssue(errorOutcome)">
          <div class="break-words">{{ outcomeIssueText(outcomePrimaryIssue(errorOutcome)) }}</div>
          <div class="break-words">{{ outcomeIssueCode(outcomePrimaryIssue(errorOutcome)) }}</div>
        </template>
      </div>
    </div>
  </div>
  <div
    v-if="notice"
    class="mb-3 flex items-center gap-2 rounded-md border border-accent-line bg-accent-bg px-3 py-2 text-sm text-accent"
    role="status"
    aria-live="polite"
  >
    <CheckCircle2 :size="17" />{{ notice }}
  </div>
  <button v-if="mode === 'chat' && apiReachable && !modelConfigured" type="button" class="mb-3 flex w-full items-center gap-2 rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-left text-sm text-warn hover:bg-warn-bg-strong" @click="$emit('open-settings')"><AlertTriangle class="shrink-0" :size="17" /><span>模型未配置，对话不会有真实回复。点此前往设置填写 Provider 与 API Key。</span></button>
  <div v-if="hasUnsavedChanges" class="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-sm text-warn">
    <span>有未保存改动：{{ unsavedSummary }}。保存后生效，切换或筛选前会要求确认。</span>
    <button class="cmd" type="button" @click="$emit('discard')"><RotateCcw :size="15" />丢弃草稿</button>
  </div>
</template>
