<script setup>
import { computed } from "vue";
import { ListChecks, Target } from "@lucide/vue";
import { inSessionTaskStatusText, isInSessionTaskDetail } from "./insessionTaskDetails";

const props = defineProps({
  tasks: { type: Array, default: () => [] }
});

const visibleTasks = computed(() => props.tasks.filter(isInSessionTaskDetail));
</script>

<template>
  <section
    v-if="visibleTasks.length"
    class="rounded-lg border border-accent-line bg-accent-bg p-4"
    aria-label="关联的会话内任务"
    data-testid="insession-task-cards"
  >
    <div class="flex items-center gap-2 text-xs font-semibold text-accent-text">
      <ListChecks :size="15" />关联的会话内任务
    </div>
    <div class="mt-3 grid gap-3">
      <article
        v-for="task in visibleTasks"
        :key="task.insession_task_id"
        class="rounded-md border border-accent-line bg-surface p-3"
        data-testid="insession-task-card"
      >
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 class="flex items-center gap-1 text-sm font-semibold text-accent"><Target :size="15" />{{ task.title }}</h2>
            <p class="mt-1 text-sm text-accent-text">{{ inSessionTaskStatusText(task.status) }}</p>
          </div>
          <span
            v-if="task.current_graph_revision !== null"
            class="rounded-full border border-accent-line bg-surface px-2 py-1 text-xs text-accent-text"
          >版本 {{ task.current_graph_revision }}</span>
          <span
            v-else
            class="rounded-full border border-accent-line bg-accent-bg px-2 py-1 text-xs text-ink-3"
          >任务图尚未建立</span>
        </div>

        <ul v-if="task.nodes.length" class="mt-3 space-y-1.5 text-sm">
          <li v-for="node in task.nodes" :key="node.insession_task_node_id" class="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-ink">
            <span>{{ node.title }}</span>
            <span class="text-xs text-ink-3">{{ inSessionTaskStatusText(node.status) }}</span>
          </li>
        </ul>
        <p class="mt-3 text-xs text-ink-3">关联 {{ task.related_turn_count }} 个会话回合 · 仅展示已持久化状态</p>
      </article>
    </div>
  </section>
</template>
