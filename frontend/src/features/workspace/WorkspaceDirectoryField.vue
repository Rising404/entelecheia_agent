<script setup>
import { FolderOpen } from "@lucide/vue";

defineProps({
  modelValue: { type: String, default: "" },
  label: { type: String, default: "会话工作目录" },
  placeholder: { type: String, default: "留空使用默认目录" },
  disabled: Boolean,
  pickerAvailable: Boolean
});
defineEmits(["update:modelValue", "choose"]);
</script>

<template>
  <label class="block min-w-0 text-sm">
    <span class="mb-1 block font-medium">{{ label }}</span>
    <span class="flex min-w-0 gap-1">
      <input
        class="field min-w-0 flex-1"
        :aria-label="label"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        @input="$emit('update:modelValue', $event.target.value)"
      />
      <button
        v-if="pickerAvailable"
        class="icon-btn shrink-0"
        type="button"
        title="选择本机目录"
        aria-label="选择本机目录"
        :disabled="disabled"
        @click="$emit('choose')"
      ><FolderOpen :size="17" /></button>
    </span>
  </label>
</template>
