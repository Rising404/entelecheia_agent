<script setup>
import { toRef } from "vue";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  XCircle
} from "@lucide/vue";
import { useWorkspaceFiles } from "./useWorkspaceFiles";

const props = defineProps({
  sessionId: { type: String, required: true },
  workingDir: { type: String, default: "" },
  open: { type: Boolean, default: true },
  workspaceFilesClient: { type: Object, required: true }
});

const emit = defineEmits(["update:open"]);

const {
  cwd,
  entries,
  loading,
  error,
  newName,
  newKind,
  openPath,
  content,
  savingFile,
  hasWorkingDirectory,
  load,
  enterDir,
  goUp,
  createEntry,
  openFile,
  closeFile,
  saveFile,
  deleteEntry
} = useWorkspaceFiles({
  client: props.workspaceFilesClient,
  sessionId: toRef(props, "sessionId"),
  workingDir: toRef(props, "workingDir"),
  open: toRef(props, "open")
});
</script>

<template>
  <section class="rounded-md border border-line bg-surface p-3 text-sm" data-testid="workspace-files-panel">
    <div class="flex items-center justify-between gap-2">
      <div class="flex min-w-0 items-center gap-1 font-semibold">
        <FolderOpen :size="16" /><span class="truncate">本地目录</span>
      </div>
      <div class="flex items-center gap-1">
        <button v-if="open && hasWorkingDirectory" class="icon-btn" type="button" title="刷新本地目录" :disabled="loading" @click="load()"><RefreshCw :size="15" /></button>
        <button class="icon-btn" type="button" :title="open ? '折叠本地目录面板' : '展开本地目录面板'" @click="emit('update:open', !open)">
          <ChevronDown v-if="open" :size="15" /><ChevronRight v-else :size="15" />
        </button>
      </div>
    </div>

    <p class="mt-1 truncate text-xs text-ink-3" :title="workingDir || '未绑定工作目录'">{{ workingDir || '未绑定工作目录' }}</p>

    <template v-if="open">
      <div v-if="!hasWorkingDirectory" class="mt-3 rounded border border-dashed border-line bg-sunken p-3 text-xs text-ink-3">
        <p>此会话没有可用的本地工作目录；历史归档不受影响。</p>
      </div>

      <template v-else>
        <div class="mt-2 flex items-center gap-2 text-xs text-ink-3">
          <button v-if="cwd" class="icon-btn" type="button" title="上一级" @click="goUp"><ChevronLeft :size="15" /></button>
          <span class="truncate">/{{ cwd }}</span>
        </div>

        <form class="mt-2 flex gap-2" @submit.prevent="createEntry">
          <select v-model="newKind" class="field" aria-label="类型" style="max-width:5.5rem">
            <option value="file">文件</option>
            <option value="dir">文件夹</option>
          </select>
          <input v-model="newName" class="field min-w-0 flex-1" placeholder="名称" />
          <button class="icon-btn primary" type="submit" :title="newKind==='dir' ? '新建文件夹' : '新建文件'" :disabled="!newName.trim()">
            <FolderPlus v-if="newKind==='dir'" :size="16" /><Plus v-else :size="16" />
          </button>
        </form>

        <div v-if="error" class="mt-2 rounded border border-danger-line bg-danger-bg px-2 py-1 text-xs text-danger">{{ error }}</div>

        <ul v-if="entries.length" class="mt-2 max-h-56 space-y-0.5 overflow-auto">
          <li v-for="entry in entries" :key="entry.rel_path" class="flex items-center gap-1">
            <button type="button" class="ctx-item flex-1" :title="entry.name" @click="entry.kind==='dir' ? enterDir(entry) : openFile(entry)">
              <span class="flex min-w-0 items-center gap-2">
                <Folder v-if="entry.kind==='dir'" :size="15" class="text-ink-3" />
                <FileText v-else :size="15" class="text-ink-3" />
                <span class="truncate">{{ entry.name }}</span>
              </span>
              <span v-if="entry.kind==='file' && entry.size != null" class="shrink-0 text-xs text-ink-3">{{ entry.size }}B</span>
            </button>
            <button type="button" class="icon-btn" title="删除到回收站" @click="deleteEntry(entry)"><Trash2 :size="14" /></button>
          </li>
        </ul>
        <p v-else-if="!loading" class="mt-2 text-xs text-ink-3">此目录为空。</p>

        <div v-if="openPath" class="mt-3 border-t border-line pt-2">
          <div class="mb-1 flex items-center justify-between">
            <span class="truncate text-xs font-medium">编辑：{{ openPath }}</span>
            <div class="flex gap-1">
              <button class="cmd primary" type="button" :disabled="savingFile" @click="saveFile"><Save :size="14" />{{ savingFile ? '保存中' : '保存' }}</button>
              <button class="icon-btn" type="button" title="关闭" @click="closeFile"><XCircle :size="15" /></button>
            </div>
          </div>
          <textarea v-model="content" class="field w-full font-mono text-xs" rows="10" spellcheck="false" />
        </div>
      </template>
    </template>
  </section>
</template>
