<script setup>
import { computed, ref, watch } from "vue";
import { Check, Copy } from "@lucide/vue";
import { renderReplyMarkdown } from "./markdown";
import { roleText } from "./presentation";

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, default: "" },
  streaming: Boolean
});

const isReply = computed(() => props.role === "assistant");
const renderedReply = computed(() => isReply.value ? renderReplyMarkdown(props.content) : "");
const copyState = ref("idle");
const copyLabel = computed(() => ({
  idle: "复制", pending: "复制中", copied: "已复制", failed: "重试复制"
}[copyState.value]));

watch(() => props.content, () => { copyState.value = "idle"; });

async function copyMarkdown() {
  const original = props.content;
  copyState.value = "pending";
  try {
    // 复制持久消息原文，不能从渲染后的 DOM 提取，否则会丢失 Markdown 标记。
    await navigator.clipboard.writeText(original);
    if (props.content === original) copyState.value = "copied";
  } catch {
    if (props.content === original) copyState.value = "failed";
  }
}
</script>

<template>
  <article class="message-card" :class="role" :aria-live="streaming ? 'polite' : undefined">
    <header class="mb-2 flex items-center justify-between gap-4 text-xs text-ink-3">
      <span class="font-semibold">{{ roleText(role) }}</span>
      <button
        v-if="isReply && !streaming && content"
        class="reply-copy"
        type="button"
        title="复制 Markdown 原文"
        aria-label="复制 Markdown 原文"
        :disabled="copyState === 'pending'"
        @click="copyMarkdown"
      >
        <Check v-if="copyState === 'copied'" :size="13" aria-hidden="true" />
        <Copy v-else :size="13" aria-hidden="true" />
        <span role="status">{{ copyLabel }}</span>
      </button>
    </header>
    <p v-if="copyState === 'failed'" class="mb-2 text-xs text-warn" role="alert">复制失败，请重试或检查剪贴板权限。</p>
    <div v-if="isReply" class="markdown-body" v-html="renderedReply" />
    <p v-else class="whitespace-pre-wrap break-words">{{ content }}</p>
    <span v-if="streaming" class="ml-0.5 inline-block animate-pulse" aria-hidden="true">▍</span>
  </article>
</template>

<style scoped>
.reply-copy {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  border-radius: 4px;
  padding: 4px 6px;
  cursor: pointer;
}
.reply-copy:hover { background: var(--color-surface); }
.reply-copy:focus-visible { outline: 2px solid var(--color-accent-solid); outline-offset: 2px; }
.reply-copy:disabled { cursor: wait; opacity: 0.6; }

.markdown-body { min-width: 0; line-height: 1.65; overflow-wrap: anywhere; }
.markdown-body :deep(> * + *) { margin-top: 0.75em; }
.markdown-body :deep(h1), .markdown-body :deep(h2), .markdown-body :deep(h3),
.markdown-body :deep(h4), .markdown-body :deep(h5), .markdown-body :deep(h6) {
  margin-top: 1.1em;
  margin-bottom: 0.4em;
  font-weight: 700;
  line-height: 1.35;
}
.markdown-body :deep(> :first-child) { margin-top: 0; }
.markdown-body :deep(h1) { font-size: 1.5em; }
.markdown-body :deep(h2) { font-size: 1.3em; }
.markdown-body :deep(h3) { font-size: 1.15em; }
.markdown-body :deep(strong) { font-weight: 700; }
.markdown-body :deep(em) { font-style: italic; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 1.5em; }
.markdown-body :deep(ul) { list-style: disc; }
.markdown-body :deep(ol) { list-style: decimal; }
.markdown-body :deep(li + li), .markdown-body :deep(li > p + p) { margin-top: 0.3em; }
.markdown-body :deep(blockquote) {
  border-left: 3px solid var(--color-accent-line);
  padding-left: 0.9em;
  color: var(--color-ink-2);
}
.markdown-body :deep(a) { color: var(--color-accent); text-decoration: underline; }
.markdown-body :deep(code) {
  border-radius: 4px;
  padding: 0.1em 0.3em;
  background: var(--color-shade);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.9em;
}
.markdown-body :deep(pre) {
  max-width: 100%;
  overflow-x: auto;
  border-radius: 6px;
  padding: 12px;
  background: var(--color-code-bg);
  color: var(--color-code-ink);
}
.markdown-body :deep(pre code) { padding: 0; background: transparent; white-space: pre; }
.markdown-body :deep(table) { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
.markdown-body :deep(th), .markdown-body :deep(td) { border: 1px solid var(--color-line); padding: 6px 10px; }
.markdown-body :deep(th) { background: var(--color-surface); font-weight: 600; }
.markdown-body :deep(hr) { border: 0; border-top: 1px solid var(--color-line); }
</style>
