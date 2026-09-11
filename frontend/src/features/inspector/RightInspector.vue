<script setup>
import { Bug, RefreshCw } from "@lucide/vue";
import { computed } from "vue";

const props = defineProps({
  mode: { type: String, required: true },
  systemStatus: { type: Object, default: null },
  apiOk: Boolean,
  developerMode: Boolean,
  loading: Boolean,
  saving: Boolean,
  chatBusy: Boolean,
  turnsCount: { type: Number, default: 0 },
  debugOpen: Boolean,
  debugPayload: { type: Object, default: () => ({}) }
});

defineEmits([
  "refresh",
  "update:debug-open"
]);

// 收起来的时候仍然要能看出有没有事。一个把告警一起藏掉的折叠，
// 比不折叠更糟——它让"没看见"和"没问题"变成同一件事。
const componentList = computed(() => Object.entries(props.systemStatus?.components || {}));
const failing = computed(() => componentList.value.filter(([, item]) => !item?.ok));
const connectionSummary = computed(() => {
  if (!componentList.value.length) return "尚未检查";
  if (!failing.value.length) return `${componentList.value.length} 项正常`;
  const names = failing.value.map(([key, item]) => item?.label || key);
  return `${names.length} 项异常：${names.join("、")}`;
});
</script>

<template>
  <aside class="h-screen overflow-y-auto border-l border-line bg-sunken p-4 max-[1080px]:col-span-2 max-[1080px]:h-auto max-[1080px]:border-l-0 max-[1080px]:border-t">
    <section v-if="developerMode" class="mb-4">
      <div class="mb-2 flex items-center justify-between gap-2">
        <div class="flex min-w-0 items-center gap-2">
          <h2 class="text-sm font-semibold">连接</h2>
          <span
            class="truncate text-xs"
            :class="failing.length ? 'text-danger-text' : 'text-ink-3'"
          >{{ connectionSummary }}</span>
        </div>
        <div class="flex shrink-0 items-center gap-2">
          <button class="icon-btn" type="button" title="重连 API" :disabled="loading || saving || chatBusy" @click="$emit('refresh')"><RefreshCw :size="15" /></button>
          <span class="rounded-md px-2 py-1 text-xs" :class="apiOk ? 'bg-accent-bg-strong text-accent' : 'bg-danger-bg-strong text-danger'">{{ apiOk ? "api" : "离线" }}</span>
        </div>
      </div>
      <div class="space-y-2 text-sm">
        <div v-for="(component, key) in systemStatus?.components || {}" :key="key" class="rounded-md border border-line bg-surface p-2">
          <div class="flex items-center justify-between gap-2">
            <span class="font-medium">{{ component.label || key }}</span>
            <span :class="component.ok ? 'text-accent-text' : 'text-danger-text'">{{ component.ok ? "正常" : "异常" }}</span>
          </div>
          <p class="mt-1 line-clamp-2 text-xs text-ink-3">{{ component.detail || component.error || "-" }}</p>
        </div>
      </div>
    </section>

    <section v-if="mode === 'chat'" class="mb-4">
      <h2 class="mb-2 text-sm font-semibold">会话</h2>
      <div class="rounded-md border border-line bg-surface p-3 text-sm">
        <div class="flex justify-between gap-2"><span>消息</span><strong>{{ turnsCount }}</strong></div>
      </div>
      <div class="mt-4">
        <slot name="local-directory" />
      </div>
      <slot name="chat-controls" />
    </section>

    <section v-if="developerMode">
      <button class="debug-toggle" type="button" @click="$emit('update:debug-open', !debugOpen)"><Bug :size="16" />Debug</button>
      <template v-if="debugOpen">
        <pre class="debug-pre">{{ debugPayload }}</pre>
      </template>
    </section>
  </aside>
</template>
