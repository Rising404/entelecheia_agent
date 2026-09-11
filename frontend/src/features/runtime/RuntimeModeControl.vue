<script setup>
import { computed } from "vue";
import { ListTodo, Zap } from "@lucide/vue";
import {
  RUNTIME_MODE_TASK,
  RUNTIME_MODE_TURN,
  runtimeModeAvailable
} from "./runtimeRouting";

const props = defineProps({
  modelValue: { type: String, required: true },
  routing: { type: Object, default: null },
  disabled: Boolean
});

const emit = defineEmits(["update:modelValue"]);

const options = computed(() => [
  {
    value: RUNTIME_MODE_TURN,
    label: "单轮",
    icon: Zap,
    title: runtimeModeAvailable(RUNTIME_MODE_TURN, props.routing)
      ? "本轮允许模型在直接回答与单轮工具执行之间选择（L0/L1）"
      : "L1 Runtime 当前未启用"
  },
  {
    value: RUNTIME_MODE_TASK,
    label: "任务",
    icon: ListTodo,
    title: "任务模式仍在开发中；当前可查看界面，但不会发送消息或启动真实任务"
  }
].map((option) => ({
  ...option,
  available: runtimeModeAvailable(option.value, props.routing)
})));

function selectMode(option) {
  if (props.disabled || !option.available || option.value === props.modelValue) return;
  emit("update:modelValue", option.value);
}
</script>

<template>
  <div class="runtime-mode-control" data-testid="runtime-mode-control">
    <span class="runtime-mode-label">本轮</span>
    <div class="runtime-mode-segments" role="radiogroup" aria-label="本轮运行模式">
      <button
        v-for="option in options"
        :key="option.value"
        class="runtime-mode-option"
        :class="{ active: option.value === modelValue }"
        type="button"
        role="radio"
        :aria-checked="option.value === modelValue"
        :aria-label="option.label"
        :title="option.title"
        :disabled="disabled || !option.available"
        @click="selectMode(option)"
      >
        <component :is="option.icon" :size="14" />
        <span>{{ option.label }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.runtime-mode-control {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.runtime-mode-label {
  flex: none;
  color: var(--color-ink-3);
  font-size: 12px;
  font-weight: 650;
}

.runtime-mode-segments {
  display: grid;
  min-width: min(100%, 252px);
  grid-template-columns: repeat(3, minmax(68px, 1fr));
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 6px;
  background: var(--color-surface);
}

.runtime-mode-option {
  display: inline-flex;
  min-height: 34px;
  min-width: 0;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 0 9px;
  color: var(--color-ink-3);
  font-size: 13px;
  font-weight: 650;
}

.runtime-mode-option + .runtime-mode-option {
  border-left: 1px solid var(--color-line);
}

.runtime-mode-option.active {
  background: var(--color-accent-solid);
  color: var(--color-on-solid);
}

.runtime-mode-option:disabled:not(.active) {
  opacity: 0.42;
}
</style>
