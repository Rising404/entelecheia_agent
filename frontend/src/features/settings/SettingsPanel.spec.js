import { mount } from "@vue/test-utils";
import { reactive } from "vue";
import { describe, expect, it } from "vitest";

import SettingsPanel from "./SettingsPanel.vue";

describe("SettingsPanel", () => {

  it("shows safe runtime journal health without raw errors", () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "mock", api_key: "", base_url: "", model: "" },
        modelConfigured: false,
        // 运行日志健康只服务于排查，普通视图里不出现。
        developerMode: true,
        runtimeEventsStatus: {
          enabled: true,
          database_present: true,
          degraded: true,
          append_failures_total: 2,
          prune_failures_total: 1
        }
      }
    });

    const status = wrapper.get('[data-testid="runtime-events-status"]');
    expect(status.text()).toContain("日志异常，实时显示仍可用");
    expect(status.text()).toContain("写入失败 2");
    expect(status.text()).not.toContain("sqlite");
  });

  it("shows legacy Office availability without receiving the stored path", async () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: {
          provider: "deepseek",
          api_key: "",
          base_url: "",
          model: "",
          legacy_office_soffice: ""
        },
        legacyOfficeStatus: { configured: true, available: true },
        modelConfigured: true,
        developerMode: true
      }
    });

    const input = wrapper.get('[data-testid="legacy-office-soffice"]');
    expect(input.element.value).toBe("");
    expect(input.attributes("placeholder")).toContain("已配置");
    expect(wrapper.get('[data-testid="legacy-office-status"]').text()).toContain(
      "当前可用"
    );

    await input.setValue("/Applications/LibreOffice.app/Contents/MacOS/soffice");
    expect(wrapper.emitted("update-field")?.at(-1)).toEqual([
      {
        field: "legacy_office_soffice",
        value: "/Applications/LibreOffice.app/Contents/MacOS/soffice"
      }
    ]);
  });

  it("keeps operator-only settings out of the ordinary view", () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "deepseek", api_key: "", base_url: "", model: "" },
        modelConfigured: true,
        modelControl: { effective: "native", configured: "native", endpoint_host: "x" },
        runtimeEventsStatus: { enabled: true, database_present: true, degraded: false },
        legacyOfficeStatus: { configured: true, available: true }
      }
    });

    expect(wrapper.text()).not.toContain("工具控制通道");
    expect(wrapper.text()).not.toContain("活动恢复");
    expect(wrapper.find('[data-testid="legacy-office-soffice"]').exists()).toBe(false);
  });

  it("shows one section at a time", () => {
    // 几组配置之间没有先后关系；堆成一长页只会让人每次从头滚一遍找目标。
    const form = {
      provider: "deepseek", api_key: "", base_url: "", model: "",
      vision_provider: "", vision_base_url: "", vision_model: "", vision_api_key: ""
    };
    const model = mount(SettingsPanel, { props: { form, section: "model", modelConfigured: true } });
    expect(model.find('[data-testid="profile-list-model"]').exists()).toBe(true);
    expect(model.find('[data-testid="profile-list-vision"]').exists()).toBe(false);

    const vision = mount(SettingsPanel, { props: { form, section: "vision", modelConfigured: true } });
    expect(vision.find('[data-testid="profile-list-vision"]').exists()).toBe(true);
    expect(vision.find('[data-testid="profile-list-model"]').exists()).toBe(false);
  });

  it("shows graded reasoning effort for an OpenAI-compatible tier", async () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "anthropic-compatible" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: true,
        profiles: {
          model: {
            profiles: [{
              id: "mp-openai",
              name: "OpenAI",
              provider: "openai-compatible",
              model: "gpt-5.6-sol"
            }]
          }
        },
        tierForm: {
          architect: {
            profile_id: "mp-openai",
            thinking: false,
            reasoning_effort: "high"
          }
        },
        tierEffective: [{
          tier: "architect",
          provider: "anthropic-compatible",
          reasoning_control: "toggle",
          model: "previous-model",
          origin: "profile"
        }]
      }
    });

    const effort = wrapper.get('[data-testid="tier-architect-reasoning-effort"]');
    expect(effort.element.value).toBe("high");
    expect(wrapper.find('[data-testid="tier-architect-thinking"]').exists()).toBe(false);

    await effort.setValue("xhigh");
    expect(wrapper.emitted("update-tier")?.at(-1)).toEqual([
      { tier: "architect", field: "reasoning_effort", value: "xhigh" }
    ]);
  });

  it("clears a stale effort immediately when the selected profile cannot use it", async () => {
    const tierForm = reactive({
      architect: {
        profile_id: "mp-openai",
        thinking: false,
        reasoning_effort: "minimal"
      }
    });
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "openai-compatible" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: true,
        profiles: {
          model: {
            profiles: [
              {
                id: "mp-openai",
                name: "OpenAI",
                provider: "openai-compatible",
                request_dialect: "openai",
                base_url: "https://api.openai.com/v1",
                model: "gpt-test"
              },
              {
                id: "mp-anthropic",
                name: "Anthropic",
                provider: "anthropic-compatible",
                request_dialect: "anthropic",
                base_url: "https://api.anthropic.com",
                model: "claude-test"
              }
            ]
          }
        },
        tierForm,
        tierEffective: [],
        onUpdateTier({ tier, field, value }) {
          tierForm[tier][field] = value;
        }
      }
    });

    await wrapper.get('[data-testid="tier-architect-profile"]')
      .setValue("mp-anthropic");

    expect(tierForm.architect.profile_id).toBe("mp-anthropic");
    expect(tierForm.architect.reasoning_effort).toBe("");
    expect(wrapper.get('[data-testid="tier-architect-reasoning-effort"]')
      .element.value).toBe("");
  });

  it("keeps the legacy toggle-only server response usable", async () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "anthropic-compatible" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: true,
        tierForm: {
          attempt: { profile_id: "", thinking: false, reasoning_effort: "" }
        },
        tierEffective: [{
          tier: "attempt",
          provider: "anthropic-compatible",
          reasoning_control: "toggle",
          model: "deepseek-v4-pro",
          origin: "global"
        }]
      }
    });

    const thinking = wrapper.get('[data-testid="tier-attempt-thinking"]');
    expect(wrapper.find('[data-testid="tier-attempt-reasoning-effort"]').exists()).toBe(false);

    await thinking.setValue(true);
    expect(wrapper.emitted("update-tier")?.at(-1)).toEqual([
      { tier: "attempt", field: "thinking", value: true }
    ]);
  });

  it("keeps native Anthropic effort editable while thinking is disabled", () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "anthropic-compatible" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: true,
        tierForm: {
          final_gate: {
            profile_id: "",
            thinking: false,
            reasoning_effort: "medium"
          }
        },
        tierEffective: [{
          tier: "final_gate",
          provider: "anthropic-compatible",
          request_dialect: "anthropic-native",
          reasoning_control: "toggle_independent_effort",
          reasoning_effort_options: ["low", "medium", "high", "xhigh", "max"],
          model: "claude",
          origin: "global"
        }]
      }
    });

    expect(wrapper.get('[data-testid="tier-final_gate-thinking"]').element.checked)
      .toBe(false);
    expect(wrapper.get('[data-testid="tier-final_gate-reasoning-effort"]').attributes("disabled"))
      .toBeUndefined();
    expect(wrapper.get('[data-testid="tier-final_gate-reasoning-effort"]').element.value)
      .toBe("medium");
  });

  it("does not offer a fake reasoning control for endpoints without one", () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "mock" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: true,
        tierForm: {
          final_gate: { profile_id: "", thinking: false, reasoning_effort: "" }
        },
        tierEffective: [{
          tier: "final_gate",
          provider: "mock",
          reasoning_control: "none",
          model: "mock-structured",
          origin: "global"
        }]
      }
    });

    expect(wrapper.find('[data-testid="tier-final_gate-thinking"]').exists()).toBe(false);
    expect(wrapper.find('[data-testid="tier-final_gate-reasoning-effort"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="tier-final_gate-reasoning-unavailable"]').text())
      .toContain("没有可配置的推理强度");
  });

  it("disables tier edits after a configuration reload fails", () => {
    const wrapper = mount(SettingsPanel, {
      props: {
        form: { provider: "openai-compatible" },
        section: "tiers",
        modelConfigured: true,
        tierLoaded: false,
        tierForm: {
          architect: {
            profile_id: "mp-stale",
            thinking: false,
            reasoning_effort: "high"
          }
        },
        tierEffective: [{
          tier: "architect",
          provider: "openai-compatible",
          reasoning_control: "effort",
          model: "stale-model",
          origin: "profile"
        }]
      }
    });

    expect(wrapper.get('[data-testid="tier-config-unavailable"]').text())
      .toContain("重新读取配置");
    expect(wrapper.get('[data-testid="tier-architect-profile"]').attributes("disabled"))
      .toBeDefined();
    expect(wrapper.get('[data-testid="tier-architect-reasoning-effort"]').attributes("disabled"))
      .toBeDefined();
    expect(wrapper.get('[data-testid="settings-save"]').attributes("disabled"))
      .toBeDefined();
  });
});
