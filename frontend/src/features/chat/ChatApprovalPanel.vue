<script setup>
import { CheckCircle2, FileText, XCircle } from "@lucide/vue";
import {
  committedWriteMessage,
  committedWriteStatusText,
  committedWrites,
  emptyReviewResultText,
  fileWritePreview,
  outcomeCardClass,
  outcomeIssueCode,
  outcomeIssueText,
  outcomePartialText,
  outcomePrimaryIssue,
  outcomeRetryText,
  outcomeStatusText,
  outcomeWarnings,
  reviewItemTitle,
  reviewItems,
  reviewPayloadFields,
  reviewPolicyText,
  reviewRiskText,
  reviewSideEffectText,
  resumeDecisionText
} from "./presentation";

defineProps({
  pendingReview: { type: Object, default: null },
  resumeResult: { type: Object, default: null },
  decisionApproved: { type: Boolean, default: null },
  recoveryOutcome: { type: Object, default: null },
  saving: Boolean,
  chatBusy: Boolean
});

defineEmits(["decide"]);
</script>

<template>
  <article v-if="pendingReview" class="approval-card">
    <header class="mb-2 flex items-center justify-between gap-2">
      <div>
        <div class="text-sm font-semibold">需要确认</div>
        <div class="text-xs text-ink-3">{{ pendingReview.reason || "本轮包含受保护操作" }}</div>
      </div>
      <span class="rounded bg-warn-bg-strong px-2 py-1 text-xs text-warn">pending</span>
    </header>
    <p class="mb-2 rounded-md border border-warn-line bg-warn-bg px-2 py-1 text-xs text-warn">
      当前轮次等待确认，处理审批前不会发送新的消息。
    </p>
    <div v-if="pendingReview.reply" class="mb-2 rounded-md border border-line bg-surface px-3 py-2 text-sm">
      <div class="mb-1 text-xs font-semibold text-ink-3">执行前说明</div>
      <p class="whitespace-pre-wrap break-words">{{ pendingReview.reply }}</p>
    </div>
    <div class="space-y-2">
      <pre v-if="reviewItems(pendingReview).length === 0" class="review-pre">{{ pendingReview }}</pre>
      <div v-for="(item, index) in reviewItems(pendingReview)" :key="index" class="rounded-md border border-line bg-surface p-2 text-sm">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="font-medium">{{ reviewItemTitle(item, index) }}</div>
          <div class="flex flex-wrap gap-1 text-xs">
            <span class="rounded bg-shade px-2 py-0.5 text-ink-3">{{ reviewRiskText(item.risk_level) }}</span>
            <span class="rounded bg-shade px-2 py-0.5 text-ink-3">{{ reviewSideEffectText(item) }}</span>
            <span class="rounded bg-warn-bg-strong px-2 py-0.5 text-warn">{{ reviewPolicyText(item) }}</span>
          </div>
        </div>
        <dl v-if="reviewPayloadFields(item).length" class="mt-2 grid gap-2 text-xs">
          <div v-for="field in reviewPayloadFields(item)" :key="field.key" class="rounded border border-line bg-sunken px-2 py-1">
            <dt class="font-semibold text-ink-3">{{ field.label }}</dt>
            <dd class="mt-0.5 whitespace-pre-wrap break-words text-ink">{{ field.value }}</dd>
          </div>
        </dl>
        <div v-if="fileWritePreview(item)" class="mt-2 overflow-hidden rounded-md border border-line bg-sunken text-xs">
          <div class="flex flex-wrap items-center justify-between gap-2 border-b border-line px-2 py-1.5 text-ink-2">
            <span class="flex min-w-0 items-center gap-1 font-semibold"><FileText :size="14" /><span class="truncate">{{ fileWritePreview(item).path }}</span></span>
            <span class="rounded bg-shade px-2 py-0.5 text-ink-3">{{ fileWritePreview(item).modeText }}</span>
          </div>
          <div class="max-h-64 overflow-auto bg-ink py-2 font-mono text-[12px] leading-relaxed text-canvas">
            <div v-for="(line, lineIndex) in fileWritePreview(item).lines" :key="`${lineIndex}-${line}`" class="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-2 px-2">
              <span class="select-none text-right text-accent-line">+{{ lineIndex + 1 }}</span>
              <span class="whitespace-pre-wrap break-words text-accent-line">{{ line || " " }}</span>
            </div>
            <div v-if="fileWritePreview(item).omitted" class="px-2 pt-1 text-warn-line">还有 {{ fileWritePreview(item).omitted }} 行未展示</div>
            <div v-if="fileWritePreview(item).lineCount === 0" class="px-2 text-warn-line">待写入内容为空</div>
          </div>
        </div>
        <details class="mt-2"><summary class="cursor-pointer text-xs font-medium text-ink-3">原始详情</summary><pre class="review-pre">{{ item }}</pre></details>
      </div>
    </div>
    <div class="mt-3 flex gap-2">
      <button class="cmd" type="button" :disabled="saving || chatBusy" @click="$emit('decide', true)"><CheckCircle2 :size="16" />批准</button>
      <button class="cmd danger" type="button" :disabled="saving || chatBusy" @click="$emit('decide', false)"><XCircle :size="16" />拒绝</button>
    </div>
  </article>

  <article v-if="resumeResult" class="approval-card">
    <header class="mb-2 flex items-center justify-between gap-2">
      <div>
        <div class="text-sm font-semibold">审批结果</div>
        <div class="text-xs text-ink-3">{{ committedWrites(resumeResult).length ? `处理 ${committedWrites(resumeResult).length} 项受保护操作` : "没有提交操作" }}</div>
      </div>
      <span class="rounded px-2 py-1 text-xs" :class="decisionApproved === false ? 'bg-danger-bg text-danger' : 'bg-shade text-ink-3'">{{ resumeDecisionText(decisionApproved) }}</span>
    </header>
    <div v-if="committedWrites(resumeResult).length" class="space-y-2">
      <div v-for="(write, index) in committedWrites(resumeResult)" :key="`${write.tool_id}-${index}`" class="rounded-md border border-line bg-surface p-2 text-sm">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="font-medium">{{ reviewItemTitle(write, index) }}</div>
          <span class="rounded px-2 py-0.5 text-xs" :class="write.ok ? 'bg-accent-bg text-accent' : 'bg-danger-bg text-danger'">{{ committedWriteStatusText(write) }}</span>
        </div>
        <p class="mt-1 break-words text-xs text-ink-3">{{ committedWriteMessage(write) }}</p>
        <details class="mt-2"><summary class="cursor-pointer text-xs font-medium text-ink-3">原始结果</summary><pre class="review-pre">{{ write }}</pre></details>
      </div>
    </div>
    <div v-else class="rounded-md border border-line bg-surface p-2 text-sm">
      <p class="text-ink-2">{{ emptyReviewResultText(decisionApproved) }}</p>
      <details class="mt-2"><summary class="cursor-pointer text-xs font-medium text-ink-3">原始结果</summary><pre class="review-pre">{{ resumeResult }}</pre></details>
    </div>
  </article>

  <article v-if="recoveryOutcome" class="approval-card" :class="outcomeCardClass(recoveryOutcome)">
    <header class="mb-2 flex flex-wrap items-start justify-between gap-2">
      <div>
        <div class="text-sm font-semibold">{{ outcomeStatusText(recoveryOutcome.status) }}</div>
        <div v-if="outcomeRetryText(recoveryOutcome)" class="text-xs text-ink-3">{{ outcomeRetryText(recoveryOutcome) }}</div>
      </div>
      <span class="rounded bg-surface/70 px-2 py-1 text-xs text-ink-3">{{ recoveryOutcome.status }}</span>
    </header>
    <p v-if="outcomePartialText(recoveryOutcome)" class="mb-2 text-sm text-ink-2">{{ outcomePartialText(recoveryOutcome) }}</p>
    <div v-if="outcomePrimaryIssue(recoveryOutcome)" class="border-t border-line pt-2 text-sm">
      <div class="font-medium">{{ outcomeIssueText(outcomePrimaryIssue(recoveryOutcome)) }}</div>
      <div v-if="outcomeIssueCode(outcomePrimaryIssue(recoveryOutcome))" class="mt-1 text-xs text-ink-3">{{ outcomeIssueCode(outcomePrimaryIssue(recoveryOutcome)) }}</div>
    </div>
    <details v-if="outcomeWarnings(recoveryOutcome).length > 1" class="mt-2">
      <summary class="cursor-pointer text-xs font-medium text-ink-3">更多提示</summary>
      <ul class="mt-2 space-y-1 text-xs text-ink-2"><li v-for="warning in outcomeWarnings(recoveryOutcome).slice(1)" :key="warning.code">{{ outcomeIssueText(warning) }}</li></ul>
    </details>
  </article>
</template>
