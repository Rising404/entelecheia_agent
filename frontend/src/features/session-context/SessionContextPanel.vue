<script setup>
import { computed, ref, watch } from "vue";
import {
  Download,
  Eraser,
  Eye,
  Pencil,
  RefreshCw,
  Wrench,
  XCircle
} from "@lucide/vue";
import {
  formatContextValue,
  SESSION_CONTEXT_GROUPS
} from "./presentation";

const props = defineProps({
  sessionContext: { type: Object, required: true },
  selectedState: { type: Object, default: null },
  explanation: { type: Object, default: null },
  loading: Boolean,
  busy: Boolean,
  error: { type: String, default: "" },
  repairPreview: { type: Object, default: null },
  repairApplyEnabled: Boolean,
  repairPreviewStale: Boolean
});

const emit = defineEmits([
  "refresh",
  "explain",
  "export",
  "clear",
  "create-correction",
  "preview-repair",
  "apply-repair"
]);

const exportWithExcerpt = ref(false);
const clearOpen = ref(false);
const clearReason = ref("");
const clearAcknowledged = ref(false);
const clearSubmitted = ref(false);
const correctionOpen = ref(false);
const correctionOperation = ref("set");
const correctionValue = ref("");
const correctionAcknowledged = ref(false);
const repairAcknowledged = ref(false);
const correctionSubmitted = ref(false);

const stateRows = computed(() => SESSION_CONTEXT_GROUPS.flatMap(([key, label]) =>
  (props.sessionContext?.[key] || []).map((item) => ({ ...item, groupLabel: label }))
));
const evidenceRows = computed(() => props.explanation?.explanation?.evidence || []);
const correctionOperations = computed(() => props.selectedState?.correction_operations || []);
const repairChangeCount = computed(() => {
  const changes = props.repairPreview?.changes || {};
  return (changes.added?.length || 0) + (changes.removed?.length || 0) + (changes.changed?.length || 0);
});

watch(() => props.selectedState?.id, () => {
  correctionOpen.value = false;
  correctionAcknowledged.value = false;
  repairAcknowledged.value = false;
});
watch(() => props.repairPreview?.preview_token, () => {
  repairAcknowledged.value = false;
});
watch(() => props.busy, (busy) => {
  if (busy) return;
  if (clearSubmitted.value) {
    clearSubmitted.value = false;
    if (!props.error) closeClear();
  }
  if (correctionSubmitted.value) {
    correctionSubmitted.value = false;
    if (!props.error) {
      correctionOpen.value = false;
      correctionAcknowledged.value = false;
    }
  }
});

function submitClear() {
  const reason = clearReason.value.trim();
  if (!reason || !clearAcknowledged.value) return;
  clearSubmitted.value = true;
  emit("clear", reason);
}

function closeClear() {
  if (props.busy) return;
  clearOpen.value = false;
  clearReason.value = "";
  clearAcknowledged.value = false;
}

function openCorrection() {
  if (!props.selectedState || correctionOperations.value.length === 0) return;
  correctionOperation.value = correctionOperations.value.includes("set")
    ? "set"
    : correctionOperations.value[0];
  correctionValue.value = typeof props.selectedState.value_json === "string"
    ? props.selectedState.value_json
    : JSON.stringify(props.selectedState.value_json, null, 2);
  correctionAcknowledged.value = false;
  correctionOpen.value = true;
}

function parsedCorrectionValue() {
  const text = correctionValue.value.trim();
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function submitCorrection() {
  if (!props.selectedState || !correctionAcknowledged.value) return;
  if (["set", "append"].includes(correctionOperation.value) && !correctionValue.value.trim()) return;
  correctionSubmitted.value = true;
  emit("create-correction", {
    state: props.selectedState,
    operation: correctionOperation.value,
    value: parsedCorrectionValue()
  });
}
</script>

<template>
  <section class="mt-4 border-t border-line pt-4" data-testid="session-context-panel">
    <div class="mb-3 flex items-center justify-between gap-2">
      <div>
        <h2 class="text-sm font-semibold">会话状态</h2>
        <p class="mt-0.5 text-xs text-ink-3">{{ sessionContext.counts?.total || 0 }} 条状态</p>
      </div>
      <button class="icon-btn" type="button" title="刷新会话状态" :disabled="loading || busy" @click="$emit('refresh')"><RefreshCw :size="15" /></button>
    </div>

    <div v-if="error" class="mb-3 rounded-md border border-danger-line bg-danger-bg p-2 text-xs text-danger">
      <p>{{ error }}</p>
    </div>

    <div class="space-y-3">
      <div v-if="loading" class="empty min-h-24">读取中</div>
      <div v-else-if="stateRows.length === 0" class="empty min-h-24">当前没有会话状态</div>
      <div v-else class="space-y-1.5">
        <button
          v-for="item in stateRows"
          :key="item.id"
          class="w-full rounded-md border px-2.5 py-2 text-left text-xs"
          :class="selectedState?.id === item.id ? 'border-accent-solid bg-accent-bg' : 'border-line bg-surface'"
          type="button"
          @click="$emit('explain', { state: item, includeExcerpt: false })"
        >
          <span class="flex items-center justify-between gap-2"><strong class="truncate">{{ item.key }}</strong><span>{{ item.groupLabel }}</span></span>
          <span class="mt-1 block line-clamp-2 text-ink-3">{{ formatContextValue(item.value_json) }}</span>
        </button>
      </div>

      <div v-if="selectedState && explanation" class="border-t border-line pt-3 text-xs">
        <div class="mb-2 flex items-center justify-between gap-2"><strong>证据依据</strong><span>{{ explanation.explanation?.support_status === 'complete' ? '完整' : '不完整' }}</span></div>
        <div v-for="item in evidenceRows" :key="item.id" class="mb-2 rounded-md border border-line bg-surface p-2">
          <p class="break-all font-mono text-[11px] text-ink-3">{{ item.id }}</p>
          <p v-if="item.content_excerpt" class="mt-1 whitespace-pre-wrap">{{ item.content_excerpt }}</p>
          <p v-else-if="item.status === 'missing'" class="mt-1 text-danger">证据缺失</p>
          <p v-else class="mt-1 text-ink-3">摘录已隐藏</p>
        </div>
        <button class="cmd w-full justify-center" type="button" :disabled="busy" @click="$emit('explain', { state: selectedState, includeExcerpt: true })"><Eye :size="15" />显示证据摘录</button>
        <div class="mt-2 grid grid-cols-2 gap-2">
          <button class="cmd justify-center" type="button" :disabled="busy || correctionOperations.length === 0" @click="openCorrection"><Pencil :size="15" />纠正状态</button>
          <button class="cmd justify-center" type="button" :disabled="busy" @click="$emit('preview-repair', selectedState)"><Wrench :size="15" />预览修复</button>
        </div>
        <p v-if="correctionOperations.length === 0" class="mt-2 text-ink-3">该类型不接受用户显式纠正</p>
      </div>

      <div v-if="correctionOpen && selectedState" class="border-t border-line pt-3 text-xs" data-testid="correction-form">
        <div class="mb-2 flex items-center justify-between"><strong>记录纠正证据</strong><button class="mini-btn" type="button" title="关闭纠正表单" :disabled="busy" @click="correctionOpen = false"><XCircle :size="14" /></button></div>
        <select v-model="correctionOperation" class="field mb-2 w-full" aria-label="纠正操作">
          <option v-for="operation in correctionOperations" :key="operation" :value="operation">{{ operation }}</option>
        </select>
        <textarea v-if="['set', 'append'].includes(correctionOperation)" v-model="correctionValue" class="slot-textarea min-h-24" placeholder="字符串或 JSON 值" />
        <label class="mt-2 flex items-start gap-2"><input v-model="correctionAcknowledged" class="mt-0.5" type="checkbox" /><span>确认把这次修改记录为用户纠正 evidence；应用 Repair 前当前状态不会改变</span></label>
        <button class="cmd primary mt-2 w-full justify-center" type="button" :disabled="busy || !correctionAcknowledged || (['set', 'append'].includes(correctionOperation) && !correctionValue.trim())" @click="submitCorrection"><Pencil :size="15" />记录并生成预览</button>
      </div>

      <div v-if="repairPreview" class="border-t border-line pt-3 text-xs" data-testid="repair-preview">
        <div class="mb-2 flex items-center justify-between gap-2"><strong>Repair dry-run</strong><span>{{ repairChangeCount }} 项变化</span></div>
        <div class="space-y-2">
          <div v-for="change in repairPreview.changes?.changed || []" :key="change.id" class="rounded-md border border-line bg-surface p-2">
            <p class="mb-1 break-all font-mono text-[11px]">{{ change.id }}</p>
            <p class="text-danger">原：{{ formatContextValue(change.before?.value_json) }}</p>
            <p class="mt-1 text-accent">新：{{ formatContextValue(change.after?.value_json) }}</p>
          </div>
          <div v-for="item in repairPreview.changes?.added || []" :key="`added-${item.id}`" class="rounded-md border border-accent-line bg-accent-bg p-2">新增 · {{ item.key }}：{{ formatContextValue(item.value_json) }}</div>
          <div v-for="item in repairPreview.changes?.removed || []" :key="`removed-${item.id}`" class="rounded-md border border-danger-line bg-danger-bg p-2">移除 · {{ item.key }}：{{ formatContextValue(item.value_json) }}</div>
        </div>
        <p class="mt-2 text-ink-3">revision {{ repairPreview.context_revision }} · {{ repairPreview.candidate_source }}</p>
        <p v-if="repairPreviewStale" class="mt-2 font-semibold text-danger">预览已过期，请重新生成</p>
        <p v-else-if="!repairApplyEnabled" class="mt-2 text-warn">controlled replace 当前被 feature flag 禁用</p>
        <label v-else class="mt-2 flex items-start gap-2"><input v-model="repairAcknowledged" class="mt-0.5" type="checkbox" /><span>确认仅应用这份预览；任何新会话写入都会使 token 失效</span></label>
        <div class="mt-2 grid grid-cols-2 gap-2">
          <button class="cmd justify-center" type="button" :disabled="busy" @click="$emit('preview-repair', selectedState)"><RefreshCw :size="15" />重新预览</button>
          <button class="cmd primary justify-center" type="button" :disabled="busy || !repairApplyEnabled || repairPreviewStale || !repairAcknowledged" @click="$emit('apply-repair')"><Wrench :size="15" />应用 Repair</button>
        </div>
      </div>

      <div v-if="sessionContext.latest_reset" class="border-t border-line pt-3 text-xs text-ink-3">
        <p class="font-semibold text-ink-2">最近清空</p>
        <p class="mt-1">{{ sessionContext.latest_reset.created_at }}</p>
        <p class="mt-1 line-clamp-2">{{ sessionContext.latest_reset.reason }}</p>
      </div>

      <div class="border-t border-line pt-3">
        <label class="mb-2 flex items-center gap-2 text-xs"><input v-model="exportWithExcerpt" type="checkbox" />导出中包含证据摘录</label>
        <div class="grid grid-cols-2 gap-2">
          <button class="cmd justify-center" type="button" :disabled="busy" @click="$emit('export', { includeExcerpt: exportWithExcerpt })"><Download :size="15" />导出</button>
          <button class="cmd danger justify-center" type="button" :disabled="busy" @click="clearOpen = true"><Eraser :size="15" />清空状态</button>
        </div>
        <button class="cmd mt-2 w-full justify-center" type="button" :disabled="busy" @click="$emit('preview-repair', null)"><Wrench :size="15" />检查全会话修复</button>
      </div>
    </div>

    <div v-if="clearOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4" role="dialog" aria-modal="true" aria-label="清空会话状态">
      <div class="w-full max-w-md rounded-md border border-line bg-sunken p-5 shadow-xl">
        <h2 class="text-base font-semibold">清空会话状态</h2>
        <p class="mt-2 text-sm text-ink-2">状态和变更记录会被清除；聊天记录和证据仍会保留。</p>
        <label class="mt-4 block text-sm font-medium">原因<input v-model="clearReason" class="field mt-1 w-full" placeholder="为什么要重新开始" /></label>
        <label class="mt-3 flex items-start gap-2 text-sm"><input v-model="clearAcknowledged" class="mt-1" type="checkbox" /><span>我理解聊天记录和证据不会被本操作删除</span></label>
        <div class="mt-5 flex justify-end gap-2">
          <button class="cmd" type="button" :disabled="busy" @click="closeClear">取消</button>
          <button class="cmd danger" type="button" :disabled="busy || !clearReason.trim() || !clearAcknowledged" @click="submitClear"><Eraser :size="15" />确认清空</button>
        </div>
      </div>
    </div>
  </section>
</template>
