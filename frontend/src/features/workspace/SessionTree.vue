<script setup>
import { computed, nextTick, ref } from "vue";
import WorkspaceDirectoryField from "./WorkspaceDirectoryField.vue";
import {
  ChevronDown,
  ChevronRight,
  FolderOpen,
  MessageSquare,
  Pencil,
  Plus,
  Trash2,
  XCircle
, Pin } from "@lucide/vue";

const props = defineProps({
  status: { type: String, default: "active" },
  search: { type: String, default: "" },
  projects: { type: Array, default: () => [] },
  unbound: { type: Array, default: () => [] },
  selectedId: { type: String, default: "" },
  draftActive: Boolean,
  draftWorkingDir: { type: String, default: "" },
  draftDirectoryLocked: Boolean,
  directoryPickerAvailable: Boolean,
  draftSessionIds: { type: Object, default: () => new Set() },
  loading: Boolean,
  saving: Boolean
});

const emit = defineEmits([
  "empty-trash",
  "session-context",
  "rename-session",
  "update:search",
  "set-status",
  "search",
  "clear-search",
  "select-session",
  "start-draft",
  "choose-draft-directory",
  "update-draft-directory",
  "discard-draft",
  "rename-project",
  "pin-project",
  "reorder-projects"
]);

const collapsedPaths = ref(new Set());
const canCreateInCurrentStatus = computed(() => props.status === "active");
const sessionCount = computed(
  () => props.unbound.length + props.projects.reduce((total, item) => total + item.sessions.length, 0)
);

// 搜索时全部展开：折叠状态是浏览用的，不该把命中的结果藏起来。
function isCollapsed(path) {
  return collapsedPaths.value.has(path) && !props.search.trim();
}

function toggleProject(path) {
  const next = new Set(collapsedPaths.value);
  if (next.has(path)) next.delete(path);
  else next.add(path);
  collapsedPaths.value = next;
}

// —— 项目拖拽排序 ——
//
// 用原生 HTML5 拖放，不引库：一个侧栏里的十几个条目不值得为它加一个依赖。
// 重命名进行中时禁止拖，否则拖动会打断输入框的焦点。
const draggingPath = ref("");
const dropTargetPath = ref("");

function onProjectDragStart(project, event) {
  if (renamingProjectPath.value) { event.preventDefault(); return; }
  draggingPath.value = project.path;
  event.dataTransfer.effectAllowed = "move";
  // Firefox 要求设过数据才会真正开始拖
  event.dataTransfer.setData("text/plain", project.path);
}

function onProjectDragOver(project) {
  if (!draggingPath.value || project.path === draggingPath.value) return;
  dropTargetPath.value = project.path;
}

function onProjectDrop(project) {
  const from = draggingPath.value;
  const to = project.path;
  onProjectDragEnd();
  if (!from || from === to) return;
  const order = props.projects.map((item) => item.path);
  const fromIndex = order.indexOf(from);
  const toIndex = order.indexOf(to);
  if (fromIndex < 0 || toIndex < 0) return;
  order.splice(toIndex, 0, ...order.splice(fromIndex, 1));
  emit("reorder-projects", order);
}

function onProjectDragEnd() {
  draggingPath.value = "";
  dropTargetPath.value = "";
}

// —— 项目重命名（就地编辑）——
//
// 原来走 window.prompt。浏览器里能用，但 Electron 的渲染进程根本没实现它——按下去
// 静默无事发生，桌面端因此完全按不动。会话那边早就是就地编辑了，这里对齐。
const renamingProjectPath = ref("");
const projectNameDraft = ref("");
const projectNameInputs = new Map();

function registerProjectNameInput(path, element) {
  if (element) projectNameInputs.set(path, element);
  else projectNameInputs.delete(path);
}

function startProjectRename(project) {
  renamingProjectPath.value = project.path;
  projectNameDraft.value = project.name || "";
  nextTick(() => projectNameInputs.get(project.path)?.select?.());
}

function cancelProjectRename() {
  renamingProjectPath.value = "";
  projectNameDraft.value = "";
}

function commitProjectRename(project) {
  const name = projectNameDraft.value.trim();
  const path = project.path;
  cancelProjectRename();
  if (!name || name === project.name) return;
  emit("rename-project", { path, name });
}

// —— 会话重命名（就地编辑）——
const renamingId = ref("");
const renameDraft = ref("");
const renameInputs = new Map();

function registerRenameInput(sessionId, element) {
  if (element) renameInputs.set(sessionId, element);
  else renameInputs.delete(sessionId);
}

function startRename(sessionId, currentTitle) {
  renamingId.value = sessionId;
  renameDraft.value = currentTitle || "";
  nextTick(() => renameInputs.get(sessionId)?.select?.());
}

function cancelRename() {
  renamingId.value = "";
  renameDraft.value = "";
}

function commitRename(sessionId) {
  const title = renameDraft.value.trim();
  cancelRename();
  if (!title) return;
  emit("rename-session", { sessionId, title });
}

defineExpose({ startRename });

function lastActiveText(session) {
  const stamp = session.last_active_at || session.updated_at || session.created_at;
  if (!stamp) return "尚无对话";
  const at = new Date(stamp);
  if (Number.isNaN(at.getTime())) return "尚无对话";
  const minutes = Math.floor((Date.now() - at.getTime()) / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  if (minutes < 60 * 24) return `${Math.floor(minutes / 60)} 小时前`;
  if (minutes < 60 * 24 * 7) return `${Math.floor(minutes / 1440)} 天前`;
  return at.toLocaleDateString();
}
</script>

<template>
  <button
    class="cmd mb-3 w-full justify-center"
    type="button"
    :disabled="saving || !canCreateInCurrentStatus"
    @click="emit('start-draft')"
  ><Plus :size="16" />新建会话</button>

  <section
    v-if="draftActive"
    class="mb-3 space-y-2 rounded-md border border-warn-line bg-warn-bg p-2 text-xs"
  >
    <div class="flex items-center justify-between gap-2">
      <span class="font-semibold text-warn">新会话</span>
      <button class="icon-btn" type="button" title="放弃这个新会话" @click="emit('discard-draft')">
        <XCircle :size="15" />
      </button>
    </div>
    <WorkspaceDirectoryField
      :model-value="draftWorkingDir"
      :disabled="saving || draftDirectoryLocked"
      :picker-available="directoryPickerAvailable"
      @update:model-value="emit('update-draft-directory', $event)"
      @choose="emit('choose-draft-directory')"
    />
  </section>

  <div class="mb-3 flex rounded-md border border-line bg-surface p-1">
    <button v-for="item in ['active', 'archived', 'trashed']" :key="item" class="seg" :class="{ active: status === item }" type="button" @click="emit('set-status', item)">
      {{ item === "active" ? "活跃" : item === "archived" ? "归档" : "回收站" }}
    </button>
  </div>

  <form class="mb-3 flex gap-2" @submit.prevent="emit('search')">
    <label class="min-w-0 flex-1">
      <input :value="search" class="field w-full" placeholder="搜索标题或历史" @input="emit('update:search', $event.target.value)" />
    </label>
    <button
      v-if="status === 'trashed' && sessionCount"
      class="icon-btn danger"
      type="button"
      title="清空回收站"
      :disabled="loading"
      @click="emit('empty-trash')"
    ><Trash2 :size="18" /></button>
    <button v-if="search.trim()" class="icon-btn" type="button" title="清空搜索" :disabled="loading" @click="emit('clear-search')"><XCircle :size="18" /></button>
  </form>

  <div class="space-y-1">
    <template v-for="project in projects" :key="project.path">
      <div
        class="group flex items-center gap-1 rounded-md transition-[opacity,box-shadow] duration-100"
        :class="{
          'opacity-40': draggingPath === project.path,
          'shadow-[inset_0_2px_0_var(--color-accent-solid)]': dropTargetPath === project.path
        }"
        :draggable="!renamingProjectPath"
        @dragstart="onProjectDragStart(project, $event)"
        @dragover.prevent="onProjectDragOver(project)"
        @drop.prevent="onProjectDrop(project)"
        @dragend="onProjectDragEnd"
      >
        <span v-if="renamingProjectPath === project.path" class="min-w-0 flex-1" @click.stop>
          <input
            :ref="(element) => registerProjectNameInput(project.path, element)"
            v-model="projectNameDraft"
            class="field w-full text-sm"
            aria-label="重命名项目"
            @keydown.enter.prevent="commitProjectRename(project)"
            @keydown.esc.prevent="cancelProjectRename"
            @blur="commitProjectRename(project)"
          />
        </span>
        <button v-else class="task-row flex-1" type="button" :title="project.path" @click="toggleProject(project.path)">
          <span class="flex min-w-0 items-center gap-2">
            <ChevronRight v-if="isCollapsed(project.path)" :size="15" />
            <ChevronDown v-else :size="15" />
            <FolderOpen :size="17" />
            <span class="flex min-w-0 flex-col items-start">
              <span class="w-full truncate text-left">{{ project.name }}</span>
              <!-- 项目名取的是路径最后一段，不同目录经常同名（多个 .../workspace）。
                   把完整路径显示出来，否则用户只能靠名字猜，很容易认错项目。 -->
              <span class="project-path text-[11px] leading-tight text-ink-3"><bdi>{{ project.path }}</bdi></span>
            </span>
          </span>
          <span class="text-xs text-ink-3">{{ project.sessions.length }}</span>
        </button>
        <button v-if="canCreateInCurrentStatus" class="icon-btn" type="button" title="在此项目新建会话" :disabled="saving" @click="emit('start-draft', project.path)"><Plus :size="14" /></button>
        <button class="icon-btn opacity-0 group-hover:opacity-100" type="button" title="重命名项目" @click="startProjectRename(project)"><Pencil :size="14" /></button>
        <button
          class="icon-btn"
          :class="project.pinned ? 'text-accent-text' : 'opacity-0 group-hover:opacity-100'"
          type="button"
          :title="project.pinned ? '取消置顶' : '置顶这个项目'"
          @click="emit('pin-project', { path: project.path, pinned: !project.pinned })"
        ><Pin :size="14" /></button>
      </div>

      <button
        v-for="session in (isCollapsed(project.path) ? [] : project.sessions)"
        :key="session.id"
        class="task-row ml-5 w-[calc(100%-1.25rem)]"
        :class="{ active: selectedId === session.id }"
        type="button"
        @click="emit('select-session', session.id)"
        @contextmenu.prevent="emit('session-context', { sessionId: session.id, x: $event.clientX, y: $event.clientY })"
      >
        <span v-if="renamingId === session.id" class="min-w-0 flex-1" @click.stop>
          <input
            :ref="(element) => registerRenameInput(session.id, element)"
            v-model="renameDraft"
            class="field w-full text-sm"
            aria-label="重命名会话"
            @keydown.enter.prevent="commitRename(session.id)"
            @keydown.esc.prevent="cancelRename"
            @blur="commitRename(session.id)"
          />
        </span>
        <span v-else class="min-w-0"><span class="flex min-w-0 items-center gap-2"><span class="block truncate font-medium">{{ session.title || '未命名会话' }}</span><span v-if="draftSessionIds.has(session.id)" class="shrink-0 rounded bg-warn-bg-strong px-1.5 py-0.5 text-xs text-warn">草稿</span></span><span class="mt-1 block text-xs text-ink-3">{{ lastActiveText(session) }}</span><span v-if="session.match_snippet" class="mt-1 block truncate text-xs text-warn-text">{{ session.match_snippet }}</span></span>
        <MessageSquare class="mt-0.5 text-ink-3" :size="17" />
      </button>

      <div v-if="!isCollapsed(project.path) && !project.sessions.length" class="ml-5 py-1 text-xs text-ink-3">还没有会话</div>
    </template>

    <div v-if="unbound.length" class="pt-2 text-xs text-ink-3">未归入项目</div>
    <button
      v-for="session in unbound"
      :key="session.id"
      class="task-row"
      :class="{ active: selectedId === session.id }"
      type="button"
      @click="emit('select-session', session.id)"
      @contextmenu.prevent="emit('session-context', { sessionId: session.id, x: $event.clientX, y: $event.clientY })"
    >
      <span v-if="renamingId === session.id" class="min-w-0 flex-1" @click.stop>
        <input
          :ref="(element) => registerRenameInput(session.id, element)"
          v-model="renameDraft"
          class="field w-full text-sm"
          aria-label="重命名会话"
          @keydown.enter.prevent="commitRename(session.id)"
          @keydown.esc.prevent="cancelRename"
          @blur="commitRename(session.id)"
        />
      </span>
      <span v-else class="min-w-0"><span class="flex min-w-0 items-center gap-2"><span class="block truncate font-medium">{{ session.title || '未命名会话' }}</span><span v-if="draftSessionIds.has(session.id)" class="shrink-0 rounded bg-warn-bg-strong px-1.5 py-0.5 text-xs text-warn">草稿</span></span><span class="mt-1 block text-xs text-ink-3">{{ lastActiveText(session) }}</span><span v-if="session.match_snippet" class="mt-1 block truncate text-xs text-warn-text">{{ session.match_snippet }}</span></span>
      <MessageSquare class="mt-0.5 text-ink-3" :size="17" />
    </button>

    <div v-if="!loading && !sessionCount && !draftActive" class="empty">暂无会话</div>
  </div>
</template>


<style scoped>
/* 路径从左侧截断：区分度在尾部（.../test_a/workspace 与 .../test_b/workspace），
 * 普通的右侧省略号只会留下一模一样的 /private/var/folders/... 前缀。
 *
 * 容器 direction: rtl 把省略号挪到左边；内层 <bdi> 隔离出独立的方向上下文，让路径
 * 本身照常从左往右渲染。不能用 unicode-bidi: plaintext——它按首个强字符判定方向，
 * 路径以 / 开头、随后是拉丁字母，会被判成 LTR，省略号又被推回右边。 */
.project-path {
  max-width: 100%;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  direction: rtl;
}
</style>
