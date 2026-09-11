import { reactive, ref } from "vue";
import { api } from "../api";

// 六档调用点位（后端 runtime_config.TIER_SETTING_NAMES）。共享 Router 先决定
// processing level；随后进入 L2 图链或独立 L1 链。
export const MODEL_TIERS = [
  { key: "router", label: "入口路由",
    hint: "共享意图分类器；只决定 processing level 与任务关联，不属于 L1/L2 执行链。" },
  { key: "architect", label: "建图",
    hint: "在任务图的可能形状里搜索。L2 中关掉思考风险最高的一档。" },
  { key: "attempt", label: "执行",
    hint: "生成节点内容与最终交付正文。" },
  { key: "node_verification", label: "节点验证",
    hint: "判定单个节点做完了没有。判错了后面还有质量门兜着。" },
  { key: "final_gate", label: "最终质量门",
    hint: "决定这份交付能不能给你。它后面没有任何兜底。" },
  { key: "l1", label: "L1",
    hint: "L1 链路的每一步。首次调用同时出计划，性质接近建图。" }
];

// 清除本档、回到全局配置。空串在后端表示"不改该字段"，所以清除需要一个
// 真的写得进去的值。与后端 model_tiers.CLEARED_PROFILE_ID 对应。
export const TIER_CLEARED = "-";

/**
 * 运行时配置（FR-5）。API Key 与本地 executable 路径永不回填。
 * @param {Function} [refreshStatus] 保存成功后调用，用于刷新 model_configured（联动横幅）。
 */
export function useConfig(refreshStatus) {
  const configForm = reactive({
    provider: "mock",
    request_dialect: "auto",
    base_url: "",
    model: "",
    api_key: "",
    vision_provider: "",
    vision_base_url: "",
    vision_model: "",
    vision_api_key: "",
    legacy_office_soffice: "",
    default_projects_dir: "",
    default_projects_dir_managed: false
  });
  // 每档保存端点指针、思考开关与推理强度。实际能显示和发送哪些控制由
  // 所选 profile 的请求方言决定，不能只看 Anthropic/OpenAI 协议外壳。
  const tierForm = reactive(Object.fromEntries(
    MODEL_TIERS.map((t) => [t.key, {
      profile_id: "",
      thinking: false,
      reasoning_effort: ""
    }])
  ));
  // 后端算出来的**实际生效**端点，含未选档位继承来的那一份。只读，不参与提交。
  const tierEffective = ref([]);
  // 没成功读到过就不提交分档字段。否则一次读取失败之后保存设置，会拿一份
  // 全空的默认表单把用户已有的分档选择和开关全部覆盖掉。
  const tierLoaded = ref(false);

  const configHasKey = ref(false);
  const configHasVisionKey = ref(false);
  const configLegacyOfficeStatus = ref({ configured: false, available: false });
  const configLoading = ref(false);
  const configSaving = ref(false);
  const configMessage = ref("");

  function applyModelTiers(modelTiers) {
    if (!Array.isArray(modelTiers)) return false;
    tierEffective.value = modelTiers;
    for (const { key } of MODEL_TIERS) {
      tierForm[key].profile_id = "";
      tierForm[key].thinking = false;
      tierForm[key].reasoning_effort = "";
    }
    for (const tier of modelTiers) {
      const row = tierForm[tier.tier];
      if (!row) continue;
      // 只有真正选了 profile 才回填下拉框；继承和失效指针都留空，
      // 让"当前生效"那一行去说明实际在用什么。
      row.profile_id = tier.origin === "profile" ? (tier.profile_id || "") : "";
      row.thinking = tier.thinking_enabled === true;
      row.reasoning_effort = tier.reasoning_effort || "";
    }
    tierLoaded.value = true;
    return true;
  }

  async function loadConfig() {
    tierLoaded.value = false;
    configLoading.value = true;
    configMessage.value = "";
    try {
      const data = await api.getConfig();
      const c = data.config || {};
      configForm.default_projects_dir = c.default_projects_dir || "";
      configForm.default_projects_dir_managed = c.default_projects_dir_managed === true;
      // 不把 mock 当占位推给用户：未配置时选择器默认落到 deepseek（实际值仍是后端返回的，保存才改）
      configForm.provider = c.provider && c.provider !== "mock" ? c.provider : "deepseek";
      configForm.request_dialect = c.request_dialect || "auto";
      configForm.base_url = c.base_url || "";
      configForm.model = c.model || "";
      configForm.api_key = "";           // 永不回填明文 key
      configForm.vision_provider = c.vision_provider || "";
      configForm.vision_base_url = c.vision_base_url || "";
      configForm.vision_model = c.vision_model || "";
      configForm.vision_api_key = "";     // 同样不回填
      configForm.legacy_office_soffice = ""; // 主机本地路径也只写不读
      configHasKey.value = c.has_key === true;
      configHasVisionKey.value = c.has_vision_key === true;
      if (!applyModelTiers(data.model_tiers)) tierEffective.value = [];
      configLegacyOfficeStatus.value = {
        configured: c.legacy_office_soffice?.configured === true,
        available: c.legacy_office_soffice?.available === true
      };
    } catch (err) {
      configMessage.value = `读取配置失败：${err.message}`;
    } finally {
      configLoading.value = false;
    }
  }

  async function saveConfig({ section = "model" } = {}) {
    configSaving.value = true;
    configMessage.value = "";
    try {
      let payload = {
        provider: configForm.provider,
        request_dialect: configForm.request_dialect,
        vision_provider: configForm.vision_provider,
        vision_base_url: configForm.vision_base_url,
        vision_model: configForm.vision_model,
        base_url: configForm.base_url,
        model: configForm.model
      };
      for (const { key } of (tierLoaded.value ? MODEL_TIERS : [])) {
        const row = tierForm[key];
        // 没选 profile 时提交清除记号而不是空串，否则后端会理解成"不改"，
        // 用户就取消不掉一个已经选过的档。
        payload[`tier_${key}_profile_id`] = row.profile_id || TIER_CLEARED;
        payload[`tier_${key}_thinking`] = row.thinking === true;
        // 空串不能清掉已经保存的值，因为后端统一把空字段理解成“不改”。
        // auto 是显式的 provider-default 记号，可可靠覆盖之前的 effort。
        payload[`tier_${key}_reasoning_effort`] = row.reasoning_effort || "auto";
      }
      if (configForm.api_key.trim()) payload.api_key = configForm.api_key.trim();
      // 留空表示"不改"，不是"清空"——把空串发过去会让人以为能这样删掉 key。
      if (configForm.vision_api_key.trim()) {
        payload.vision_api_key = configForm.vision_api_key.trim();
      }
      if (configForm.legacy_office_soffice.trim()) {
        payload.legacy_office_soffice = configForm.legacy_office_soffice.trim();
      }
      if (section === "workspace") {
        payload = { default_projects_dir: configForm.default_projects_dir.trim() };
      }
      const data = await api.updateConfig(payload);
      applyModelTiers(data.model_tiers);
      configHasKey.value = data.config?.has_key === true;
      configHasVisionKey.value = data.config?.has_vision_key === true;
      configLegacyOfficeStatus.value = {
        configured: data.config?.legacy_office_soffice?.configured === true,
        available: data.config?.legacy_office_soffice?.available === true
      };
      configForm.api_key = "";
      configForm.legacy_office_soffice = "";
      configForm.default_projects_dir = data.config?.default_projects_dir || configForm.default_projects_dir;
      configMessage.value = data.model_configured ? "已保存，模型已就绪，可以开始对话。" : "已保存。";
      try {
        await refreshStatus?.();          // 刷新 model_configured，联动横幅消失
      } catch {
        // 配置已经成功保存；附带的状态刷新不能把已提交结果改写成保存失败。
      }
    } catch (err) {
      configMessage.value = `保存失败：${err.message}`;
    } finally {
      configSaving.value = false;
    }
  }

  function updateTier({ tier, field, value }) {
    const row = tierForm[tier];
    if (row && field in row) row[field] = value;
  }

  return {
    configHasVisionKey,
    tierForm, tierEffective, tierLoaded, updateTier,
    configForm, configHasKey, configLegacyOfficeStatus,
    configLoading, configSaving, configMessage,
    loadConfig, saveConfig
  };
}
