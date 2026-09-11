<script setup>
import { Save, Trash2 } from "@lucide/vue";
defineProps({
  document: { type: Object, required: true },
  form: { type: Object, required: true },
  dirty: Boolean,
  saving: Boolean
});

defineEmits(["update-field", "save", "delete"]);

function processingLabel(status) {
  if (status === "complete") return "文本覆盖完整";
  if (status === "partial") return "部分可读";
  return "覆盖未知";
}
</script>

<template>
  <div class="space-y-4">
    <header class="flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
      <div class="min-w-[260px] flex-1">
        <input :value="form.title" class="title-input" placeholder="文档标题" @input="$emit('update-field', { field: 'title', value: $event.target.value })" />
        <div class="mt-2 flex flex-wrap items-center gap-2 text-sm text-ink-3">
          <span class="rounded-md border border-line bg-surface px-2 py-1">{{ document.mime || "document" }}</span>
          <span class="rounded-md border border-line bg-surface px-2 py-1">{{ document.n_chunks || 0 }} chunks</span>
          <span class="rounded-md border border-line bg-surface px-2 py-1">{{ document.added_at || "-" }}</span>
          <span class="rounded-md border border-line bg-surface px-2 py-1">{{ processingLabel(document.processing_status) }}</span>
        </div>
      </div>
      <div class="flex flex-wrap gap-2">
        <button class="cmd" type="button" :disabled="saving || !dirty" @click="$emit('save')"><Save :size="16" />保存</button>
        <button class="cmd danger" type="button" :disabled="saving" @click="$emit('delete')"><Trash2 :size="16" />删除索引</button>
      </div>
    </header>

    <section v-if="document.processing_status === 'partial'" class="rounded-md border border-warn-line bg-warn-bg p-3 text-sm text-warn">
      <p class="font-semibold">
        {{ document.needs_vision ? "仍有内容需要视觉解析" : "文档仍有解析覆盖缺口" }}，当前结论不得覆盖这些缺口。
      </p>
      <ul v-if="document.diagnostics?.length" class="mt-2 list-disc space-y-1 pl-5 text-xs">
        <li v-for="(diagnostic, index) in document.diagnostics" :key="`${diagnostic.code || 'gap'}-${index}`">
          {{ diagnostic.at || "位置未知" }} · {{ diagnostic.detail || diagnostic.code || "未解析内容" }}
        </li>
      </ul>
    </section>
    <section v-else-if="document.processing_status === 'legacy_unknown'" class="rounded-md border border-line bg-sunken p-3 text-sm text-ink-3">
      此文档来自旧索引，尚无可验证的解析覆盖记录；重新收录后才能用于严格证据引用。
    </section>

    <section class="grid grid-cols-[minmax(0,1fr)_260px] gap-3 max-[980px]:grid-cols-1">
      <div class="slot-panel">
        <h2 class="mb-2 text-sm font-semibold">摘要</h2>
        <textarea :value="form.summary" class="document-textarea" data-testid="document-summary" placeholder="摘要可留空" @input="$emit('update-field', { field: 'summary', value: $event.target.value })" />
      </div>
      <div class="slot-panel">
        <h2 class="mb-2 text-sm font-semibold">标签与来源</h2>
        <label class="mb-2 block text-sm">
          <span class="mb-1 block text-xs text-ink-3">tags</span>
          <input :value="form.tags" class="field w-full" placeholder="逗号分隔" @input="$emit('update-field', { field: 'tags', value: $event.target.value })" />
        </label>
        <dl class="document-meta">
          <div><dt>ID</dt><dd>{{ document.id }}</dd></div>
          <div><dt>task_id</dt><dd>{{ document.task_id || "-" }}</dd></div>
          <div><dt>path</dt><dd>{{ document.path || "-" }}</dd></div>
        </dl>
      </div>
    </section>
  </div>
</template>
