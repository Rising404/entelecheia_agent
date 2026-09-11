<script setup>
import { computed, ref, watch } from "vue";
import { Check, Copy, Eye, EyeOff, KeyRound, Pencil, Plus, Trash2 } from "@lucide/vue";

const props = defineProps({
  kind: { type: String, required: true },
  title: { type: String, required: true },
  profiles: { type: Array, default: () => [] },
  activeId: { type: String, default: null },
  busy: Boolean,
  // 明文由外层按需取回，取到之后才放进这里；平时它是空的。
  revealed: { type: Object, default: () => ({}) }
});

const emit = defineEmits([
  "create", "update", "delete", "activate",
  "reveal", "hide", "copy", "load-secret", "purge"
]);

const editingId = ref("");
const draft = ref(null);
// 每次进出编辑都回到遮住的状态：显示是一次动作，不是这个表单的属性。
const draftKeyVisible = ref(false);

// 外层取回明文后写进这里；组件自己不去调接口。
watch(() => props.revealed[editingId.value], (value) => {
  if (value && draft.value && !draft.value.api_key) draft.value.api_key = value;
});

function blank() {
  return {
    name: "", provider: "", request_dialect: "auto",
    base_url: "", model: "", api_key: "",
    quota: {
      requests_per_minute: "",
      tokens_per_minute: "",
      tokens_per_week: "",
      max_in_flight: "",
      quota_group: ""
    }
  };
}

function startCreate() {
  draftKeyVisible.value = false;
  editingId.value = "new";
  draft.value = blank();
}

function startEdit(profile) {
  draftKeyVisible.value = false;
  editingId.value = profile.id;
  draft.value = {
    name: profile.name, provider: profile.provider,
    request_dialect: profile.request_dialect || "auto",
    base_url: profile.base_url, model: profile.model, api_key: "",
    quota: {
      requests_per_minute: profile.quota?.requests_per_minute ?? "",
      tokens_per_minute: profile.quota?.tokens_per_minute ?? "",
      tokens_per_week: profile.quota?.tokens_per_week ?? "",
      max_in_flight: profile.quota?.max_in_flight ?? "",
      quota_group: profile.quota?.quota_group ?? ""
    }
  };
  // 已配置的 key 取回来填上：看不见现有值，就没法判断存的是不是自己以为的那个。
  // 取的时机是"点了编辑"这个明确动作，不是每次列出配置——后者才是会把密钥
  // 带进每一次页面加载的那条路。
  if (profile.has_api_key) emit("load-secret", profile.id);
}

function cancel() {
  draftKeyVisible.value = false;
  editingId.value = "";
  draft.value = null;
}

// 复制不要求先显示：想拿走它的人不必先把它亮在屏幕上。
const copiedId = ref("");
async function copy(profileId) {
  emit("copy", profileId);
  copiedId.value = profileId;
  globalThis.setTimeout(() => {
    if (copiedId.value === profileId) copiedId.value = "";
  }, 2_000);
}

function submit() {
  const payload = { ...draft.value };
  if (props.kind === "model") {
    payload.quota = Object.fromEntries(
      Object.entries(draft.value.quota).map(([key, raw]) => {
        const text = String(raw ?? "").trim();
        if (key === "quota_group") return [key, text || null];
        return [key, text ? Number(text) : null];
      })
    );
  } else {
    delete payload.quota;
  }
  if (editingId.value === "new") emit("create", { kind: props.kind, ...payload });
  else emit("update", { id: editingId.value, payload });
  cancel();
}

const canSubmit = computed(() => {
  const value = draft.value;
  if (!value) return false;
  const filled = ["name", "provider", "base_url", "model"].every((key) => value[key]?.trim());
  const quotaValid = props.kind !== "model" || [
    "requests_per_minute", "tokens_per_minute", "tokens_per_week", "max_in_flight"
  ].every((key) => {
    const raw = String(value.quota?.[key] ?? "").trim();
    return !raw || (Number.isSafeInteger(Number(raw)) && Number(raw) > 0);
  });
  // 新建必须给 key；编辑时留空表示保留原来的。
  return filled && quotaValid && (editingId.value !== "new" || Boolean(value.api_key.trim()));
});

// 任务模型的 provider 只决定协议外壳；request_dialect 再决定该厂商真正
// 接受的 token、thinking、effort 与结构化输出字段。视觉模型的 provider
// 只是另一个适配器标识，两者同名但不是一回事。
const MODEL_PROVIDERS = [
  { value: "anthropic-compatible", label: "Anthropic 兼容" },
  { value: "openai-compatible", label: "OpenAI 兼容" },
  { value: "mock", label: "mock（离线，不真实调用）" }
];
const REQUEST_DIALECTS = [
  { value: "auto", label: "自动识别（推荐）" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "generic", label: "通用兼容接口" }
];
const providerIsFixed = computed(() => props.kind === "model");

const QUOTA_FIELDS = [
  { key: "requests_per_minute", label: "每分钟请求数（RPM）", placeholder: "例如 10" },
  { key: "tokens_per_minute", label: "每分钟 Token（TPM）", placeholder: "例如 100000" },
  { key: "tokens_per_week", label: "每周 Token", placeholder: "例如 1000000000" },
  { key: "max_in_flight", label: "同时在途请求", placeholder: "例如 2" }
];

const FIELDS = computed(() => [
  { key: "name", label: "名称", placeholder: props.kind === "vision" ? "例如 Highland 视觉" : "例如 DeepSeek 主力" },
  { key: "base_url", label: "Base URL", placeholder: "https://…" },
  {
    key: "model",
    label: "Model",
    placeholder: props.kind === "vision" ? "qwen3-vl-30b-a3b-instruct" : "deepseek-v4-pro"
  }
]);
</script>

<template>
  <section class="space-y-3" :data-testid="`profile-list-${kind}`">
    <div>
      <h3 class="text-sm font-medium">{{ title }}</h3>
    </div>

    <ul class="space-y-2">
      <li
        v-for="profile in profiles"
        :key="profile.id"
        class="rounded-md border bg-surface p-3"
        :class="profile.id === activeId ? 'border-accent-solid' : 'border-line'"
      >
        <template v-if="editingId === profile.id">
          <div class="space-y-2">
            <label class="block">
              <span class="mb-1 block text-xs text-ink-3">Provider</span>
              <select v-if="providerIsFixed" v-model="draft.provider" class="field w-full">
                <option value="" disabled>请选择</option>
                <option v-for="item in MODEL_PROVIDERS" :key="item.value" :value="item.value">{{ item.label }}</option>
              </select>
              <input v-else v-model="draft.provider" class="field w-full" placeholder="例如 highland" />
            </label>
            <label v-if="providerIsFixed" class="block">
              <span class="mb-1 block text-xs text-ink-3">请求方言</span>
              <select
                v-model="draft.request_dialect"
                class="field w-full"
                data-testid="request-dialect"
              >
                <option v-for="item in REQUEST_DIALECTS" :key="item.value" :value="item.value">{{ item.label }}</option>
              </select>
            </label>
            <label v-for="field in FIELDS" :key="field.key" class="block">
              <span class="mb-1 block text-xs text-ink-3">{{ field.label }}</span>
              <input v-model="draft[field.key]" class="field w-full" :placeholder="field.placeholder" />
            </label>
            <fieldset v-if="providerIsFixed" class="space-y-2 rounded border border-line p-2">
              <legend class="px-1 text-xs text-ink-3">共享 API 配额（留空表示不限）</legend>
              <label v-for="field in QUOTA_FIELDS" :key="field.key" class="block">
                <span class="mb-1 block text-xs text-ink-3">{{ field.label }}</span>
                <input
                  v-model="draft.quota[field.key]"
                  class="field w-full"
                  type="number"
                  min="1"
                  step="1"
                  :data-testid="`quota-${field.key}`"
                  :placeholder="field.placeholder"
                />
              </label>
              <label class="block">
                <span class="mb-1 block text-xs text-ink-3">共享配额组（可选）</span>
                <input
                  v-model="draft.quota.quota_group"
                  class="field w-full"
                  data-testid="quota-quota_group"
                  placeholder="多个 Key 共用额度时填写同一名称"
                />
              </label>
            </fieldset>
            <label class="block">
              <span class="mb-1 block text-xs text-ink-3">API Key</span>
              <div class="relative">
                <input
                  v-model="draft.api_key"
                  :type="draftKeyVisible ? 'text' : 'password'"
                  class="field with-icon-right w-full"
                  :placeholder="profile.has_api_key ? '留空则不修改' : '输入 API Key'"
                  autocomplete="off"
                />
                <button
                  class="absolute right-2 top-2.5 text-ink-3 hover:text-ink"
                  type="button"
                  :title="draftKeyVisible ? '隐藏' : '显示'"
                  @click="draftKeyVisible = !draftKeyVisible"
                >
                  <EyeOff v-if="draftKeyVisible" :size="15" /><Eye v-else :size="15" />
                </button>
              </div>
            </label>
            <div class="flex gap-2">
              <button class="cmd primary" type="button" :disabled="!canSubmit || busy" @click="submit">保存</button>
              <button class="cmd" type="button" @click="cancel">取消</button>
            </div>
          </div>
        </template>

        <template v-else>
          <div class="flex items-start justify-between gap-2">
            <div class="min-w-0">
              <p class="flex items-center gap-2 font-medium">
                <span class="truncate">{{ profile.name }}</span>
                <span
                  v-if="profile.id === activeId"
                  class="shrink-0 rounded bg-accent-bg-strong px-1.5 py-0.5 text-xs text-accent"
                >使用中</span>
              </p>
              <p class="mt-1 truncate text-xs text-ink-3">
                {{ profile.provider }} · {{ profile.request_dialect || "auto" }} · {{ profile.model }}
              </p>
              <p class="truncate text-xs text-ink-3">{{ profile.base_url }}</p>
              <p v-if="profile.kind === 'model'" class="truncate text-xs text-ink-3">
                配额：{{ profile.quota?.requests_per_minute ?? "不限" }} RPM ·
                {{ profile.quota?.tokens_per_minute ?? "不限" }} TPM ·
                在途 {{ profile.quota?.max_in_flight ?? "不限" }}
              </p>
              <p class="mt-1 flex items-center gap-1 text-xs text-ink-3">
                <KeyRound :size="12" />
                <span v-if="revealed[profile.id]" class="font-mono">{{ revealed[profile.id] }}</span>
                <span v-else>{{ profile.has_api_key ? "已配置" : "未配置" }}</span>
                <button
                  v-if="profile.has_api_key"
                  class="ml-1 text-ink-3 hover:text-ink"
                  type="button"
                  :title="revealed[profile.id] ? '隐藏' : '显示密钥'"
                  @click="revealed[profile.id] ? emit('hide', profile.id) : emit('reveal', profile.id)"
                >
                  <EyeOff v-if="revealed[profile.id]" :size="13" /><Eye v-else :size="13" />
                </button>
                <button
                  v-if="profile.has_api_key"
                  class="text-ink-3 hover:text-ink"
                  type="button"
                  :title="copiedId === profile.id ? '已复制' : '复制密钥'"
                  @click="copy(profile.id)"
                >
                  <Copy :size="13" />
                </button>
                <span v-if="copiedId === profile.id" class="text-accent-text">已复制</span>
              </p>
            </div>
            <div class="flex shrink-0 gap-1">
              <button
                v-if="profile.id !== activeId"
                class="icon-btn"
                type="button"
                title="启用这份配置"
                :disabled="busy"
                @click="emit('activate', profile.id)"
              ><Check :size="15" /></button>
              <button class="icon-btn" type="button" title="编辑" :disabled="busy" @click="startEdit(profile)"><Pencil :size="15" /></button>
              <button class="icon-btn" type="button" title="删除" :disabled="busy" @click="emit('delete', profile.id)"><Trash2 :size="15" /></button>
            </div>
          </div>
        </template>
      </li>
    </ul>

    <div v-if="editingId === 'new'" class="space-y-2 rounded-md border border-line bg-surface p-3">
      <label class="block">
        <span class="mb-1 block text-xs text-ink-3">Provider</span>
        <select v-if="providerIsFixed" v-model="draft.provider" class="field w-full">
          <option value="" disabled>请选择</option>
          <option v-for="item in MODEL_PROVIDERS" :key="item.value" :value="item.value">{{ item.label }}</option>
        </select>
        <input v-else v-model="draft.provider" class="field w-full" placeholder="例如 highland" />
      </label>
      <label v-if="providerIsFixed" class="block">
        <span class="mb-1 block text-xs text-ink-3">请求方言</span>
        <select
          v-model="draft.request_dialect"
          class="field w-full"
          data-testid="request-dialect"
        >
          <option v-for="item in REQUEST_DIALECTS" :key="item.value" :value="item.value">{{ item.label }}</option>
        </select>
      </label>
      <label v-for="field in FIELDS" :key="field.key" class="block">
        <span class="mb-1 block text-xs text-ink-3">{{ field.label }}</span>
        <input v-model="draft[field.key]" class="field w-full" :placeholder="field.placeholder" />
      </label>
      <fieldset v-if="providerIsFixed" class="space-y-2 rounded border border-line p-2">
        <legend class="px-1 text-xs text-ink-3">共享 API 配额（留空表示不限）</legend>
        <label v-for="field in QUOTA_FIELDS" :key="field.key" class="block">
          <span class="mb-1 block text-xs text-ink-3">{{ field.label }}</span>
          <input
            v-model="draft.quota[field.key]"
            class="field w-full"
            type="number"
            min="1"
            step="1"
            :data-testid="`quota-${field.key}`"
            :placeholder="field.placeholder"
          />
        </label>
        <label class="block">
          <span class="mb-1 block text-xs text-ink-3">共享配额组（可选）</span>
          <input
            v-model="draft.quota.quota_group"
            class="field w-full"
            data-testid="quota-quota_group"
            placeholder="多个 Key 共用额度时填写同一名称"
          />
        </label>
      </fieldset>
      <label class="block">
        <span class="mb-1 block text-xs text-ink-3">API Key</span>
        <div class="relative">
          <input
            v-model="draft.api_key"
            :type="draftKeyVisible ? 'text' : 'password'"
            class="field with-icon-right w-full"
            placeholder="输入 API Key"
            autocomplete="off"
          />
          <button
            class="absolute right-2 top-2.5 text-ink-3 hover:text-ink"
            type="button"
            :title="draftKeyVisible ? '隐藏' : '显示'"
            @click="draftKeyVisible = !draftKeyVisible"
          >
            <EyeOff v-if="draftKeyVisible" :size="15" /><Eye v-else :size="15" />
          </button>
        </div>
      </label>
      <div class="flex gap-2">
        <button class="cmd primary" type="button" :disabled="!canSubmit || busy" @click="submit">保存</button>
        <button class="cmd" type="button" @click="cancel">取消</button>
      </div>
    </div>

    <button v-else class="cmd" type="button" :disabled="busy" @click="startCreate"><Plus :size="15" />新建配置</button>
  </section>
</template>
