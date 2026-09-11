<script setup>
import { FileText, LoaderCircle } from "@lucide/vue";
defineProps({
  documents: { type: Array, default: () => [] },
  documentIngestJobs: { type: Array, default: () => [] },
  documentIngestJobsLoading: Boolean,
  loading: Boolean,
  saving: Boolean,
  canAddDocument: { type: Boolean, default: true }
});

const emit = defineEmits([
  "add-document",
  "open-document",
  "retry-document-ingest"
]);

const stageLabels = {
  parsing: "正在解析文档",
  chunked: "已生成稳定分块",
  indexing: "正在建立检索索引",
  coverage_ready: "正在校验检索覆盖",
  active: "检索版本已激活"
};

function ingestStatusLabel(job) {
  if (job.status === "queued") return "等待后台处理";
  if (job.status === "running") return stageLabels[job.stage] || "正在后台处理";
  if (job.status === "failed") {
    return job.can_retry ? "收录失败" : "暂时失败，等待自动重试";
  }
  const processingStatus = job.processing_status ?? job.processing?.status;
  const needsVision = job.needs_vision ?? job.processing?.needs_vision;
  if (processingStatus === "partial") {
    return needsVision ? "已收录，需补充视觉解析" : "已收录，部分内容可读";
  }
  return "已收录并可检索";
}

function ingestErrorMessage(job) {
  const message = job.error?.message || job.error_message || "后台处理未完成，可重试此任务。";
  const hint = job.error?.hint;
  return hint && hint !== message ? `${message}：${hint}` : message;
}

</script>

<template>
  <section class="space-y-3 rounded-md border border-line bg-surface p-3 text-sm">
    <div>
      <div class="mb-2 flex items-center gap-2 font-semibold"><FileText :size="16" />引入的外部文档</div>
      <div v-if="loading" class="flex items-center gap-1 text-xs text-ink-3"><LoaderCircle class="animate-spin" :size="14" />加载中</div>
      <ul v-else-if="documents.length" class="max-h-40 space-y-1 overflow-auto text-ink-2">
        <li v-for="document in documents" :key="document.id">
          <button type="button" class="ctx-item" :title="document.title || document.path || document.id" @click="emit('open-document', document.id)">
            <span class="truncate">{{ document.title || document.path || document.id }}</span>
            <span v-if="document.processing_status === 'partial'" class="ctx-badge coverage-gap">{{ document.needs_vision ? "需补图" : "部分可读" }}</span>
            <span v-else-if="document.processing_status === 'legacy_unknown'" class="ctx-badge unknown">覆盖未知</span>
          </button>
        </li>
      </ul>
      <p v-else class="text-xs text-ink-3">还没有引入外部文档。</p>
      <div v-if="documentIngestJobs.length || documentIngestJobsLoading" class="mt-3 border-t border-line pt-2">
        <div class="mb-1 flex items-center justify-between gap-2 text-xs font-semibold text-ink-2">
          <span>后台收录任务</span>
          <LoaderCircle v-if="documentIngestJobsLoading" class="animate-spin" :size="13" />
        </div>
        <ul class="max-h-44 space-y-1.5 overflow-auto">
          <li v-for="job in documentIngestJobs" :key="job.job_id" class="ingest-job">
            <div class="flex min-w-0 items-center justify-between gap-2">
              <span class="truncate" :title="job.path || job.job_id">{{ job.path || job.job_id }}</span>
              <span class="ctx-badge" :class="{ 'coverage-gap': job.status === 'failed' || job.processing_status === 'partial' }">
                {{ ingestStatusLabel(job) }}
              </span>
            </div>
            <div v-if="job.status === 'failed'" class="mt-1 flex items-start justify-between gap-2">
              <p class="min-w-0 text-xs text-danger">{{ ingestErrorMessage(job) }}</p>
              <button
                v-if="job.can_retry"
                class="ingest-retry shrink-0 text-xs font-medium text-accent underline underline-offset-2"
                type="button"
                :disabled="saving"
                @click="emit('retry-document-ingest', job)"
              >重试</button>
            </div>
          </li>
        </ul>
      </div>
      <button class="cmd mt-2" type="button" :disabled="saving || !canAddDocument" @click="emit('add-document')"><FileText :size="15" />收录工作目录文档</button>
      <p v-if="!canAddDocument" class="mt-1 text-xs text-warn">当前会话没有可用的本地工作目录。</p>
    </div>
  </section>
</template>

<style scoped>
.ctx-item {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  border-radius: 0.375rem;
  padding: 0.25rem 0.5rem;
  text-align: left;
  transition: background-color 0.12s;
}
.ctx-item:hover { background-color: #f2f1ea; }
.ctx-badge {
  flex-shrink: 0;
  border-radius: 0.375rem;
  background: #e1efdc;
  padding: 0 0.375rem;
  font-size: 0.68rem;
  color: #315d33;
}
.ctx-badge.overdue { background: #f2d6cc; color: #8f2d1c; }
.ctx-badge.coverage-gap { background: #fff0c7; color: #7a5412; }
.ctx-badge.unknown { background: #ece9df; color: #686b62; }
.ingest-job {
  border-radius: 0.375rem;
  background: #f7f5ee;
  padding: 0.375rem 0.5rem;
  color: #4f544b;
}
</style>
