<script setup>
import { computed } from "vue";
import { STAGE_LABELS } from "../../shared/runtimeStages";

const props = defineProps({
  event: { type: Object, required: true }
});


const STATUS_LABELS = {
  started: "开始",
  completed: "完成",
  failed: "未完成",
  blocked: "受阻",
  paused: "已暂停"
};

const tone = computed(() => ({
  completed: "border-accent-line bg-accent-bg text-accent",
  failed: "border-danger-line bg-danger-bg text-danger",
  blocked: "border-warn-line bg-warn-bg text-warn",
  paused: "border-warn-line bg-warn-bg text-warn",
  started: "border-accent-line bg-accent-bg text-accent"
}[props.event.status] || "border-line bg-surface text-ink-3"));

const stageText = computed(() => STAGE_LABELS[props.event.stage] || `运行阶段：${props.event.stage}`);
const statusText = computed(() => STATUS_LABELS[props.event.status] || `状态：${props.event.status}`);
</script>

<template>
  <article
    class="flex min-w-0 items-start gap-2 rounded-md border px-3 py-2 text-xs"
    :class="tone"
    data-testid="runtime-activity-card"
  >
    <span class="mt-1 h-2 w-2 shrink-0 rounded-full bg-current opacity-70" />
    <div class="min-w-0 flex-1">
      <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
        <strong class="truncate font-semibold">{{ stageText }}</strong>
        <span>{{ statusText }}</span>
      </div>
      <p v-if="event.error_code" class="mt-1 truncate opacity-80">错误代码：{{ event.error_code }}</p>
    </div>
  </article>
</template>
