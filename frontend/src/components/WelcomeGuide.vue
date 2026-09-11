<script setup>
import { computed } from "vue";
import { Check, Settings, FolderPlus, MessageSquare } from "@lucide/vue";

const props = defineProps({
  modelConfigured: Boolean
});

defineEmits(["go-settings"]);

// 三步里同一时刻只有一步是"当前"。之前三步长得一模一样、前面都挂一个空心圆，
// 看上去像三个没选中的单选框——既没说清顺序，也没说清现在该做哪一件。
const steps = computed(() => {
  const done = props.modelConfigured;
  return [
    { key: "model", icon: Settings, label: "配置模型",
      hint: "填写 Provider 与 API Key，对话才会有真实回复",
      state: done ? "done" : "current", action: done ? null : "前往设置" },
    { key: "workspace", icon: FolderPlus, label: "新建工作区会话",
      hint: "选择本机目录，或使用默认目录开始会话",
      state: done ? "current" : "upcoming" },
    { key: "chat", icon: MessageSquare, label: "开始对话",
      hint: "把文件拖进输入框，或者直接提问",
      state: "upcoming" }
  ];
});
</script>

<template>
  <div class="mx-auto flex h-full min-h-0 max-w-lg flex-col justify-center gap-8 px-4">
    <div>
      <p class="text-xs font-semibold tracking-[0.18em] text-ink-3">Entelecheia（隐得莱希）</p>
      <p class="mt-1 text-sm text-ink-3">From intent to actuality.</p>
      <h1 class="mt-2 text-2xl font-semibold text-ink">欢迎，开始你的第一个工作区</h1>
    </div>

    <ol class="relative">
      <li
        v-for="(step, index) in steps"
        :key="step.key"
        class="relative flex gap-4 pb-1"
      >
        <!-- 连接线：把三步串成一条序列，而不是三个各自独立的盒子 -->
        <span
          v-if="index < steps.length - 1"
          aria-hidden="true"
          class="absolute left-[13px] top-8 bottom-0 w-px"
          :class="step.state === 'done' ? 'bg-accent-line' : 'bg-line'"
        />

        <span
          class="relative z-10 mt-1 flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-semibold"
          :class="{
            'border-accent-line bg-accent-bg text-accent': step.state === 'done',
            'border-accent-solid bg-accent-solid text-white': step.state === 'current',
            'border-line bg-canvas text-ink-3': step.state === 'upcoming'
          }"
        >
          <Check v-if="step.state === 'done'" :size="15" />
          <template v-else>{{ index + 1 }}</template>
        </span>

        <!-- 只有当前步是实体卡片；已完成和未开始的不占视觉重量 -->
        <div
          class="min-w-0 flex-1 rounded-lg px-3 pb-5 pt-1.5"
          :class="step.state === 'current'
            ? '-mt-0.5 border border-line bg-surface pb-3 pt-3 shadow-[0_1px_2px_rgba(34,35,32,0.05)]'
            : ''"
        >
          <div
            class="flex items-center gap-2 font-medium"
            :class="step.state === 'upcoming' ? 'text-ink-3' : 'text-ink'"
          >
            <component :is="step.icon" :size="15" />{{ step.label }}
          </div>
          <p class="mt-1 text-xs text-ink-3">{{ step.hint }}</p>
          <button
            v-if="step.action"
            class="cmd primary mt-3"
            type="button"
            @click="$emit('go-settings')"
          >
            <Settings :size="15" />{{ step.action }}
          </button>
        </div>
      </li>
    </ol>
  </div>
</template>
