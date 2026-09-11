<script setup>
import { HelpCircle, Send } from "@lucide/vue";

defineProps({
  questions: { type: Array, default: () => [] },
  input: { type: String, default: "" },
  canAnswer: Boolean,
  busy: Boolean
});

defineEmits(["update:input", "send", "composer-keydown"]);
</script>

<template>
  <section
    v-if="questions.length"
    class="fixed bottom-6 left-1/2 z-50 w-[min(42rem,calc(100vw-2rem))] -translate-x-1/2 rounded-xl border border-warn-line bg-warn-bg-glass p-4 backdrop-blur-xl backdrop-saturate-150 shadow-[0_18px_48px_rgba(52,45,28,0.24)]"
    role="dialog"
    aria-modal="false"
    aria-labelledby="pending-question-title"
    data-testid="pending-user-question-dialog"
  >
    <div class="flex items-start gap-3">
      <span class="mt-0.5 rounded-full bg-warn-bg-strong p-2 text-warn"><HelpCircle :size="18" /></span>
      <div class="min-w-0 flex-1">
        <h2 id="pending-question-title" class="font-semibold text-ink">需要你补充信息</h2>
        <p class="mt-1 text-xs leading-5 text-warn-text">
          问题会一直保留到对应任务真正续接。你也可以提出疑问或转向其他任务；系统不会把“点击发送”当作已回答的证明。
        </p>
      </div>
    </div>

    <ol class="mt-3 space-y-2">
      <li
        v-for="(question, index) in questions"
        :key="question.question_ref || `${question.insession_task_id || 'task'}-${index}`"
        class="rounded-lg border border-warn-line bg-surface px-3 py-2"
      >
        <div v-if="question.task_title" class="text-xs font-semibold text-warn-text">{{ question.task_title }}</div>
        <p class="mt-0.5 whitespace-pre-wrap text-sm text-ink">{{ question.question }}</p>
      </li>
    </ol>

    <form class="mt-3 grid grid-cols-[1fr_auto] gap-2" @submit.prevent="$emit('send')">
      <textarea
        :value="input"
        rows="3"
        class="min-h-20 resize-y rounded-lg border border-line-strong bg-surface px-3 py-2 text-sm outline-none focus:border-line-strong"
        :disabled="!canAnswer"
        placeholder="回答、追问，或输入你现在真正想做的事……"
        @input="$emit('update:input', $event.target.value)"
        @keydown="$emit('composer-keydown', $event)"
      />
      <button
        class="cmd primary self-end"
        type="submit"
        :disabled="busy || !canAnswer || !input.trim()"
      >
        <Send :size="16" />{{ busy ? "发送中" : "发送" }}
      </button>
    </form>
  </section>
</template>
