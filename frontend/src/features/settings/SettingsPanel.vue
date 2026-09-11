<script setup>
import { computed } from "vue";
import { AlertTriangle, KeyRound, Save, Settings } from "@lucide/vue";
import ModelProfileList from "./ModelProfileList.vue";
import WorkspaceDirectoryField from "../workspace/WorkspaceDirectoryField.vue";
import { MODEL_TIERS } from "../../composables/useConfig";

const props = defineProps({
  developerMode: Boolean,
  directoryPickerAvailable: Boolean,
  hasVisionKey: Boolean,
  section: { type: String, default: "model" },
  background: { type: Object, default: null },
  backgroundBusy: Boolean,
  sheetTranslucency: { type: Number, default: 0.45 },
  backgroundScope: { type: String, default: "content" },
  profiles: { type: Object, default: () => ({}) },
  revealed: { type: Object, default: () => ({}) },
  profilesBusy: Boolean,
  form: { type: Object, required: true },
  tierForm: { type: Object, default: () => ({}) },
  tierEffective: { type: Array, default: () => [] },
  tierLoaded: Boolean,
  hasKey: Boolean,
  legacyOfficeStatus: {
    type: Object,
    default: () => ({ configured: false, available: false })
  },
  modelConfigured: Boolean,
  modelControl: { type: Object, default: null },
  runtimeEventsStatus: { type: Object, default: null },
  loading: Boolean,
  saving: Boolean,
  message: { type: String, default: "" }
});

const BACKGROUND_SCOPES = [
  { value: "content", label: "对话区",
    hint: "图完整落在右侧这一块里，不会被左边栏切掉" },
  { value: "content-no-composer", label: "对话区，不含输入框",
    hint: "底部输入框那一条不铺背景，打字时视线更干净" },
  { value: "window", label: "整个窗口",
    hint: "铺满窗口。左边四分之一会被侧栏挡住" }
];

const SECTION_TITLES = {
  model: "任务模型配置",
  vision: "视觉模型配置",
  tiers: "分档模型配置",
  workspace: "工作目录",
  appearance: "外观"
};

const REASONING_EFFORT_OPTIONS = [
  { value: "", label: "由模型决定（默认）" },
  { value: "none", label: "none · 最低延迟" },
  { value: "minimal", label: "minimal" },
  { value: "low", label: "low" },
  { value: "medium", label: "medium · 平衡" },
  { value: "high", label: "high" },
  { value: "xhigh", label: "xhigh" },
  { value: "max", label: "max · 最高投入" }
];

// 标题跟着分节走。原来固定写"模型设置"，切到外观也还是那五个字，
// 加上下面几块内容没做分节过滤，看起来就像根本没切过去。
const sectionTitle = computed(() => SECTION_TITLES[props.section] || "设置");

const tierRows = computed(() => {
  const effective = Object.fromEntries(
    (props.tierEffective || []).map((t) => [t.tier, t])
  );
  const profiles = props.profiles?.model?.profiles || [];
  return MODEL_TIERS.map((tier) => {
    const form = props.tierForm?.[tier.key] || {
      profile_id: "",
      thinking: false,
      reasoning_effort: ""
    };
    const live = effective[tier.key] || null;
    const selected = form.profile_id
      ? profiles.find((profile) => profile.id === form.profile_id)
      : null;
    // 选项刚改但尚未保存时，优先按草稿 profile 的 provider 切换控件；
    // “跟随全局”则按全局表单。这样不会让用户先保存一次才看到正确的控件。
    const provider = selected?.provider
      || (form.profile_id ? live?.provider : (props.form?.provider || live?.provider));
    const dialect = selected
      ? concreteDialect(selected.provider, selected.base_url, selected.request_dialect)
      : (form.profile_id
          ? live?.request_dialect
          : concreteDialect(
              props.form?.provider || live?.provider,
              props.form?.base_url || live?.base_url,
              props.form?.request_dialect || live?.request_dialect
            ));
    const reasoningControl = (
      form.profile_id && !selected && !dialect && live?.reasoning_control
        ? live.reasoning_control
        : reasoningControlForDialect(dialect, provider)
    );
    const liveOptions = live?.request_dialect === dialect
      ? (live?.reasoning_effort_options || [])
      : [];
    return {
      ...tier,
      form,
      live,
      reasoningControl,
      reasoningEffortOptions: liveOptions.length
        ? liveOptions
        : reasoningOptionsForDialect(dialect)
    };
  });
});

function concreteDialect(provider, baseUrl, configured) {
  const choice = String(configured || "auto").trim().toLowerCase();
  const normalized = String(provider || "").trim().toLowerCase();
  if (choice && choice !== "auto") {
    if (choice.includes("deepseek")) return "deepseek";
    if (choice.includes("openai")) return "openai";
    if (choice.includes("anthropic")) return "anthropic";
    if (choice.includes("generic")) return "generic";
  }
  const host = String(baseUrl || "").toLowerCase();
  if (normalized === "deepseek" || host.includes("deepseek.com")) return "deepseek";
  if (host.includes("api.openai.com")) return "openai";
  if (normalized === "anthropic" || host.includes("api.anthropic.com")) return "anthropic";
  // 兼容展示尚未提供 dialect 字段的旧服务端响应。新服务端始终返回明确方言，
  // 对未知代理则继续采取保守策略。
  if (!host && (configured === undefined || configured === null)) {
    if (normalized === "openai-compatible") return "openai";
    if (normalized === "anthropic-compatible") return "legacy-toggle";
  }
  return "generic";
}

function reasoningControlForDialect(dialect, provider) {
  if (dialect === "deepseek") return "toggle_effort";
  if (dialect === "anthropic") return "toggle_independent_effort";
  if (dialect === "legacy-toggle") return "toggle";
  if (dialect === "openai") return "effort";
  // 旧服务端尚不返回 request_dialect。保留其最后已知 UI 状态，但不能假装每个
  // 兼容代理都支持这些控制项。
  if (!dialect && String(provider || "").toLowerCase() === "openai-compatible") {
    return "none";
  }
  return "none";
}

function reasoningOptionsForDialect(dialect) {
  if (dialect === "deepseek") return ["low", "high", "max"];
  if (dialect === "anthropic") return ["low", "medium", "high", "xhigh", "max"];
  if (dialect === "openai") return ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
  return [];
}

function effortOptions(values) {
  const allowed = new Set(["", ...(values || [])]);
  return REASONING_EFFORT_OPTIONS.filter((option) => allowed.has(option.value));
}

// 某档没选配置时会静默继承全局那一份。只显示一个空下拉框，看起来像这一档
// 没在跑；所以未选中时要把实际会调用的端点写出来。指向已删除配置的情况
// 单独说，否则用户不会知道自己选的东西已经不在了。
function tierEndpointNote(live) {
  if (!live) return "";
  if (live.origin === "profile") return `当前：${live.model}`;
  if (live.origin === "stale_profile") {
    return `选中的配置已不存在，正在改用全局配置：${live.model || "未配置"}`;
  }
  return `未单独配置，跟随全局：${live.model || "未配置"}`;
}

const emit = defineEmits([
  "choose-default-directory",
  "update-tier",
  "update:background-scope",
  "update:sheet-translucency",
  "background-pick",
  "background-clear",
  "update:developer-mode",
  "profile-create",
  "profile-update",
  "profile-delete",
  "profile-activate",
  "profile-reveal",
  "profile-hide",
  "profile-copy",
  "update-field",
  "save"
]);

function changeTierProfile(tier, event) {
  const profileId = event.target.value;
  emit("update-tier", {
    tier: tier.key,
    field: "profile_id",
    value: profileId
  });

  const selected = profileId
    ? (props.profiles?.model?.profiles || [])
        .find((profile) => profile.id === profileId)
    : null;
  const dialect = selected
    ? concreteDialect(
        selected.provider,
        selected.base_url,
        selected.request_dialect
      )
    : concreteDialect(
        props.form?.provider || tier.live?.provider,
        props.form?.base_url || tier.live?.base_url,
        props.form?.request_dialect || tier.live?.request_dialect
      );
  const currentEffort = String(tier.form.reasoning_effort || "");
  if (
    currentEffort
    && !reasoningOptionsForDialect(dialect).includes(currentEffort)
  ) {
    // profile 切换与本次重置由同一个同步 UI 操作发出，避免父级表单随后提交一个
    // 在新方言下拉框中不可见的值。
    emit("update-tier", {
      tier: tier.key,
      field: "reasoning_effort",
      value: ""
    });
  }
}
</script>

<template>
  <div class="max-w-xl space-y-4">
    <div class="flex items-center gap-2">
      <Settings :size="20" class="text-ink-3" />
      <h2 class="text-lg font-semibold">{{ sectionTitle }}</h2>
    </div>
    <div v-if="!modelConfigured && section !== 'appearance' && section !== 'workspace'" class="flex items-start gap-2 rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-sm text-warn">
      <AlertTriangle class="mt-0.5 shrink-0" :size="17" />
      <span>当前模型未配置，对话不会有真实回复。填写下方 Provider 与 API Key 并保存即可开始对话。</span>
    </div>

    <div v-if="developerMode && modelControl && section === 'model'" class="rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink-3">
      <div class="font-medium text-ink-2">工具控制通道：{{ modelControl.effective === 'native' ? '原生 tool use' : '兼容 JSON' }}</div>
      <div class="mt-1 text-xs">
        配置 {{ modelControl.configured }} · {{ modelControl.endpoint_host || '本地' }} ·
        {{ modelControl.native_probe_verified ? '原生能力已验证' : '尚无原生能力验证，保持兼容路径' }}
      </div>
    </div>

    <div
      v-if="developerMode && runtimeEventsStatus && section === 'model'"
      class="rounded-md border px-3 py-2 text-sm"
      :class="runtimeEventsStatus.degraded ? 'border-warn-line bg-warn-bg text-warn' : 'border-line bg-surface text-ink-3'"
      data-testid="runtime-events-status"
    >
      <div class="flex items-center gap-2 font-medium text-ink-2">
        <AlertTriangle v-if="runtimeEventsStatus.degraded" :size="15" />
        <span>活动恢复：{{ runtimeEventsStatus.enabled ? (runtimeEventsStatus.degraded ? '日志异常，实时显示仍可用' : '已启用') : '未启用' }}</span>
      </div>
      <div class="mt-1 text-xs">
        {{ runtimeEventsStatus.database_present ? '事件库已创建' : '尚未创建事件库' }} ·
        写入失败 {{ runtimeEventsStatus.append_failures_total || 0 }} ·
        清理失败 {{ runtimeEventsStatus.prune_failures_total || 0 }}
      </div>
    </div>

    <ModelProfileList
      v-if="section === 'model'"
      kind="model"
      title="已保存的任务模型"
      :profiles="profiles.model?.profiles || []"
      :active-id="profiles.model?.active_id || null"
      :revealed="revealed"
      :busy="profilesBusy"
      @create="$emit('profile-create', $event)"
      @update="$emit('profile-update', $event)"
      @delete="$emit('profile-delete', $event)"
      @activate="$emit('profile-activate', $event)"
      @reveal="$emit('profile-reveal', $event)"
      @hide="$emit('profile-hide', $event)"
      @copy="$emit('profile-copy', $event)"
      @load-secret="$emit('profile-reveal', $event)"
    />

    <div v-if="section === 'model' && developerMode" class="space-y-3">
      <div v-if="developerMode" class="border-t border-line pt-3">
        <label class="block">
          <span class="mb-1 block text-sm font-medium">LibreOffice executable</span>
          <input
            data-testid="legacy-office-soffice"
            :value="form.legacy_office_soffice"
            type="text"
            class="field w-full"
            :placeholder="legacyOfficeStatus.configured ? '已配置（留空则不修改）' : '输入 soffice 的绝对路径'"
            autocomplete="off"
            spellcheck="false"
            @input="$emit('update-field', { field: 'legacy_office_soffice', value: $event.target.value })"
          />
          <span
            data-testid="legacy-office-status"
            class="mt-1 block text-xs text-ink-3"
          >
            {{ legacyOfficeStatus.available
              ? '旧版 .doc/.ppt 转换已配置且当前可用。'
              : legacyOfficeStatus.configured
                ? '已配置，但当前主机不可用。'
                : '未配置；旧版 .doc/.ppt 转换不可用。' }}
          </span>
        </label>
      </div>
    </div>

      <ModelProfileList
        v-if="section === 'vision'"
        kind="vision"
        title="已保存的视觉模型"
        :profiles="profiles.vision?.profiles || []"
        :active-id="profiles.vision?.active_id || null"
        :revealed="revealed"
        :busy="profilesBusy"
        @create="$emit('profile-create', $event)"
        @update="$emit('profile-update', $event)"
        @delete="$emit('profile-delete', $event)"
        @activate="$emit('profile-activate', $event)"
        @reveal="$emit('profile-reveal', $event)"
        @hide="$emit('profile-hide', $event)"
        @copy="$emit('profile-copy', $event)"
        @load-secret="$emit('profile-reveal', $event)"
      />

    <div v-if="section === 'tiers'" class="space-y-4">
      <div>
        <h3 class="text-sm font-semibold">按调用位置分开配置</h3>
        <p class="mt-1 text-xs text-ink-3">
          一次任务会分别经过建图、执行、验证和质量门，它们的开销和风险并不一样。
          每一档可以各选一份已保存的任务模型；不选就跟随全局那一份。
          几档指向同一份配置就是复用，不必重复填。
        </p>
      </div>

      <p
        v-if="!tierLoaded"
        data-testid="tier-config-unavailable"
        class="rounded-md border border-warn-line bg-warn-bg px-3 py-2 text-xs text-warn"
      >档位配置尚未可靠读取。请重新读取配置后再修改或保存，当前显示值仅供参考。</p>

      <div
        v-for="tier in tierRows"
        :key="tier.key"
        class="border-t border-line pt-3"
        :data-testid="`tier-${tier.key}`"
      >
        <div class="text-sm font-medium">{{ tier.label }}</div>
        <p class="mt-0.5 text-xs text-ink-3">{{ tier.hint }}</p>

        <label class="mt-2 block">
          <span class="mb-1 block text-xs text-ink-3">使用的模型配置</span>
          <select
            :data-testid="`tier-${tier.key}-profile`"
            :value="tier.form.profile_id"
            class="field w-full"
            :disabled="!tierLoaded"
            @change="changeTierProfile(tier, $event)"
          >
            <option value="">跟随全局配置</option>
            <option
              v-for="profile in (profiles.model?.profiles || [])"
              :key="profile.id"
              :value="profile.id"
            >{{ profile.name }} · {{ profile.model }}</option>
          </select>
          <span
            :data-testid="`tier-${tier.key}-endpoint`"
            class="mt-1 block text-xs text-ink-3"
          >{{ tierEndpointNote(tier.live) }}</span>
        </label>

        <label
          v-if="tier.reasoningControl === 'toggle' || tier.reasoningControl === 'toggle_effort' || tier.reasoningControl === 'toggle_independent_effort'"
          class="mt-2 flex items-start gap-2"
        >
          <input
            :data-testid="`tier-${tier.key}-thinking`"
            type="checkbox"
            :checked="tier.form.thinking"
            class="mt-0.5"
            :disabled="!tierLoaded"
            @change="$emit('update-tier', { tier: tier.key, field: 'thinking', value: $event.target.checked })"
          />
          <span class="text-xs">
            <span class="font-medium">开启深度思考</span>
            <span class="mt-0.5 block text-ink-3">
              {{ tier.reasoningControl === 'toggle_independent_effort'
                ? '思考开关与推理强度相互独立。'
                : '关闭优先降低延迟；开启后还可选择推理强度。' }}
            </span>
          </span>
        </label>

        <label
          v-if="tier.reasoningControl === 'effort' || tier.reasoningControl === 'toggle_effort' || tier.reasoningControl === 'toggle_independent_effort'"
          class="mt-2 block"
        >
          <span class="mb-1 block text-xs font-medium">推理强度</span>
          <select
            :data-testid="`tier-${tier.key}-reasoning-effort`"
            :value="tier.form.reasoning_effort || ''"
            class="field w-full"
            :disabled="!tierLoaded || (tier.reasoningControl === 'toggle_effort' && !tier.form.thinking)"
            @change="$emit('update-tier', { tier: tier.key, field: 'reasoning_effort', value: $event.target.value })"
          >
            <option
              v-for="option in effortOptions(tier.reasoningEffortOptions)"
              :key="option.value"
              :value="option.value"
            >{{ option.label }}</option>
          </select>
          <span class="mt-1 block text-xs text-ink-3">
            留空使用端点默认值；可选档位来自当前厂商接口，具体模型若不支持会明确报错。
          </span>
        </label>

        <p
          v-if="tier.reasoningControl === 'none'"
          :data-testid="`tier-${tier.key}-reasoning-unavailable`"
          class="mt-2 text-xs text-ink-3"
        >当前端点没有可配置的推理强度。</p>
      </div>

      <div class="border-t border-line pt-3">
        <p class="mb-2 text-xs text-ink-3">
          下面新建或修改的配置，会立刻出现在上面每一档的下拉里。
          几档指向同一份就是复用，删掉正在被某档使用的那份，该档会退回跟随全局。
        </p>
        <ModelProfileList
          kind="model"
          title="模型配置"
          :profiles="profiles.model?.profiles || []"
          :active-id="profiles.model?.active_id || null"
          :revealed="revealed"
          :busy="profilesBusy"
          @create="$emit('profile-create', $event)"
          @update="$emit('profile-update', $event)"
          @delete="$emit('profile-delete', $event)"
          @activate="$emit('profile-activate', $event)"
          @reveal="$emit('profile-reveal', $event)"
          @hide="$emit('profile-hide', $event)"
          @copy="$emit('profile-copy', $event)"
          @load-secret="$emit('profile-reveal', $event)"
        />
      </div>
    </div>

    <div v-if="section === 'workspace'" class="space-y-3">
      <WorkspaceDirectoryField
        label="新会话默认根目录"
        placeholder="本机绝对路径"
        :model-value="form.default_projects_dir || ''"
        :picker-available="directoryPickerAvailable"
        :disabled="saving || loading || form.default_projects_dir_managed"
        @update:model-value="$emit('update-field', { field: 'default_projects_dir', value: $event })"
        @choose="$emit('choose-default-directory')"
      />
      <p v-if="form.default_projects_dir_managed" class="text-xs text-ink-3">当前由启动环境变量指定。</p>
    </div>

    <div v-if="section === 'appearance'" class="space-y-4">
      <div>
        <h3 class="text-sm font-semibold">界面背景</h3>
        <p class="mt-1 text-xs text-ink-3">
          面板的毛玻璃糊的是它背后的东西。默认那层渐变几乎没有细节可糊，磨砂感因此很弱；
          换成照片或视频，模糊才有真正的结构可以推开。不设置就继续用渐变。
        </p>
      </div>

      <div v-if="background" class="flex items-center gap-3 rounded-md border border-line bg-surface px-3 py-2 text-sm">
        <span class="font-medium">{{ background.kind === "video" ? "视频背景" : "图片背景" }}</span>
        <span class="text-xs text-ink-3">{{ background.media_type }}</span>
        <span v-if="background.size_bytes" class="text-xs text-ink-3">
          {{ Math.round(background.size_bytes / 1024 / 1024 * 10) / 10 }}MB
        </span>
        <button
          class="cmd danger ml-auto"
          type="button"
          :disabled="backgroundBusy"
          @click="$emit('background-clear')"
        >移除</button>
      </div>

      <label class="inline-flex cursor-pointer items-center gap-2">
        <span class="cmd" :class="{ 'opacity-60': backgroundBusy }">
          {{ backgroundBusy ? "处理中…" : (background ? "换一张" : "选择图片或视频") }}
        </span>
        <input
          type="file"
          class="sr-only"
          accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/webm"
          :disabled="backgroundBusy"
          @change="$emit('background-pick', $event.target.files?.[0] || null); $event.target.value = ''"
        />
      </label>

      <div v-if="background" class="space-y-2 border-t border-line pt-4">
        <h3 class="text-sm font-semibold">背景铺在哪</h3>
        <div class="space-y-1.5">
          <label
            v-for="option in BACKGROUND_SCOPES"
            :key="option.value"
            class="flex cursor-pointer items-start gap-2"
          >
            <input
              type="radio"
              class="mt-1"
              name="background-scope"
              :value="option.value"
              :checked="backgroundScope === option.value"
              @change="$emit('update:background-scope', option.value)"
            />
            <span>
              <span class="block text-sm">{{ option.label }}</span>
              <span class="mt-0.5 block text-xs text-ink-3">{{ option.hint }}</span>
            </span>
          </label>
        </div>
      </div>

      <div class="space-y-2 border-t border-line pt-4">
        <div class="flex items-baseline justify-between">
          <h3 class="text-sm font-semibold">面板通透度</h3>
          <span class="text-xs text-ink-3">{{ Math.round(sheetTranslucency * 100) }}%</span>
        </div>
        <p class="text-xs text-ink-3">
          面板越透，背景图越清楚，但压在上面的字越难认。这个取舍没有唯一答案，自己拖到舒服为止。
        </p>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          class="w-full"
          :value="sheetTranslucency"
          aria-label="面板通透度"
          @input="$emit('update:sheet-translucency', Number($event.target.value))"
        />
        <p class="text-xs" :class="sheetTranslucency > 0.35 ? 'text-warn-text' : 'text-ink-3'">
          <template v-if="sheetTranslucency > 0.35">
            超过 35% 之后，深色图上最亮的那些区域（比如城市灯光、窗户高光）会把小字压到看不清。
            偶尔拉高看一眼背景没问题，长期阅读建议收回 35% 以内。
          </template>
          <template v-else>当前在安全范围内：任何位置的小字都还清楚。</template>
        </p>
      </div>

      <p class="text-xs text-ink-3">
        支持 JPG / PNG / WebP / GIF / MP4 / WebM，单个不超过 48MB。
        视频会静音循环播放——但它每一帧都要让上层玻璃重糊一次，笔记本用电池时会有明显开销，
        静态图没有这个问题。
      </p>
    </div>


    <div v-if="section === 'model'" class="space-y-3">
      <label class="flex items-start gap-2">
        <input
          type="checkbox"
          class="mt-1"
          :checked="developerMode"
          @change="$emit('update:developer-mode', $event.target.checked)"
        />
        <span>
          <span class="block text-sm font-medium">开发者视图</span>
          <span class="mt-1 block text-xs text-ink-3">

          </span>
        </span>
      </label>
    </div>

    <div v-if="section !== 'appearance'" class="flex items-center gap-3">
      <button
        data-testid="settings-save"
        class="cmd primary"
        type="button"
        :disabled="saving || loading || (section === 'tiers' && !tierLoaded) || (section === 'workspace' && form.default_projects_dir_managed)"
        @click="$emit('save')"
      ><Save :size="15" />{{ saving ? "保存中…" : "保存" }}</button>
      <span v-if="message" class="text-sm text-ink-3">{{ message }}</span>
    </div>
  </div>
</template>
