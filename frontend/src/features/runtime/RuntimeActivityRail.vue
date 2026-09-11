<script setup>
import { computed } from "vue";
import RuntimeEventCard from "./RuntimeEventCard.vue";

const props = defineProps({
  events: { type: Array, default: () => [] },
  running: Boolean
});

const activities = computed(() => {
  const latest = new Map();
  for (const event of props.events) {
    if (event?.schema_version !== 1) continue;
    const activityKey = [
      event.turn_id,
      event.stage,
      event.operation_id || event.attempt_id || event.event_id
    ].join(":");
    latest.set(activityKey, { ...event, activityKey });
  }
  return [...latest.values()].slice(-6).reverse();
});
</script>

<template>
  <section v-if="activities.length" class="space-y-2" aria-label="运行动态">
    <div class="flex items-center justify-between text-xs text-ink-3">
      <span class="font-semibold text-ink-2">运行动态</span>
      <span>{{ running ? "实时更新" : "活动记录" }}</span>
    </div>
    <div class="grid gap-2 md:grid-cols-2">
      <RuntimeEventCard
        v-for="activity in activities"
        :key="activity.activityKey"
        :event="activity"
      />
    </div>
  </section>
</template>
