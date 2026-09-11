<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Archive, Pencil, RotateCcw, Trash2 } from "@lucide/vue";

const props = defineProps({
  session: { type: Object, default: null },
  x: { type: Number, default: 0 },
  y: { type: Number, default: 0 },
  busy: Boolean
});

const emit = defineEmits(["close", "rename", "archive", "unarchive", "trash", "restore", "purge"]);

const root = ref(null);
const position = ref({ left: 0, top: 0 });

// 能做什么取决于会话现在在哪一格。回收站里的会话不能改名也不能归档——
// 把做不了的事灰着摆在那里，只会让人反复去点。
const items = computed(() => {
  const status = props.session?.status;
  if (status === "trashed") {
    return [
      { key: "restore", label: "恢复", icon: RotateCcw },
      { key: "purge", label: "彻底删除", icon: Trash2, danger: true }
    ];
  }
  return [
    { key: "rename", label: "重命名", icon: Pencil },
    status === "archived"
      ? { key: "unarchive", label: "取消归档", icon: RotateCcw }
      : { key: "archive", label: "归档", icon: Archive },
    { key: "trash", label: "移到回收站", icon: Trash2, danger: true }
  ];
});

function reposition() {
  const element = root.value;
  if (!element) return;
  const { width, height } = element.getBoundingClientRect();
  // 贴着视口边缘弹出时翻到另一侧，否则菜单会有一半在屏幕外。
  const left = Math.min(props.x, globalThis.innerWidth - width - 8);
  const top = Math.min(props.y, globalThis.innerHeight - height - 8);
  position.value = { left: Math.max(8, left), top: Math.max(8, top) };
}

function onGlobalPointer(event) {
  if (!root.value?.contains(event.target)) emit("close");
}

function onGlobalKey(event) {
  if (event.key === "Escape") emit("close");
}

onMounted(() => {
  reposition();
  // 捕获阶段监听：菜单外的按钮可能自己 stopPropagation，冒泡阶段就收不到了。
  globalThis.addEventListener("pointerdown", onGlobalPointer, true);
  globalThis.addEventListener("keydown", onGlobalKey);
});
onBeforeUnmount(() => {
  globalThis.removeEventListener("pointerdown", onGlobalPointer, true);
  globalThis.removeEventListener("keydown", onGlobalKey);
});
watch(() => [props.x, props.y, props.session?.id], reposition);

function choose(key) {
  emit(key);
  emit("close");
}
</script>

<template>
  <div
    ref="root"
    class="glass fixed z-50 min-w-40 rounded-md border border-glass-line py-1 shadow-lg"
    :style="{ left: `${position.left}px`, top: `${position.top}px` }"
    role="menu"
    data-testid="session-context-menu"
  >
    <p class="truncate px-3 pb-1 pt-0.5 text-xs text-ink-3">{{ session?.title || "未命名会话" }}</p>
    <button
      v-for="item in items"
      :key="item.key"
      class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-sunken disabled:opacity-50"
      :class="item.danger ? 'text-danger-text' : ''"
      type="button"
      role="menuitem"
      :disabled="busy"
      @click="choose(item.key)"
    >
      <component :is="item.icon" :size="15" />{{ item.label }}
    </button>
  </div>
</template>
