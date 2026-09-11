<script setup>
import { computed } from "vue";
import { Archive } from "@lucide/vue";

const props = defineProps({
  session: { type: Object, required: true },
  folders: { type: Array, default: () => [] },
  disabled: Boolean
});

const emit = defineEmits(["move"]);

const currentFolderPath = computed(() => {
  if (!props.session.folder_id) return "未归档会话";
  return props.folders.find((folder) => folder.id === props.session.folder_id)?.path || "文件夹不可见";
});

function move(event) {
  const folderId = String(event.target.value || "") || null;
  if ((props.session.folder_id || null) === folderId) return;
  emit("move", { sessionId: props.session.id, folderId });
}
</script>

<template>
  <label class="flex max-w-full items-center gap-1 rounded-md border border-line bg-surface px-2 py-1" :title="`历史归档：${currentFolderPath}`">
    <Archive :size="14" />
    <span class="sr-only">历史归档</span>
    <select
      class="min-w-0 max-w-48 bg-transparent text-sm outline-none"
      aria-label="移动会话到历史文件夹"
      :value="session.folder_id || ''"
      :disabled="disabled"
      @change="move"
    >
      <option value="">未归档会话</option>
      <option v-for="folder in folders" :key="folder.id" :value="folder.id">
        {{ '　'.repeat(folder.depth) }}{{ folder.name }}
      </option>
    </select>
  </label>
</template>
