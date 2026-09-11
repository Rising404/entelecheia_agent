import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { useConfig } from "./useConfig";

afterEach(() => vi.restoreAllMocks());

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

function tier({
  key = "architect",
  origin = "profile",
  profileId = "profile-old",
  thinking = true,
  provider = "anthropic-compatible",
  reasoningControl = "toggle",
  reasoningEffort = null
} = {}) {
  return {
    tier: key,
    origin,
    profile_id: profileId,
    thinking_enabled: thinking,
    provider,
    reasoning_control: reasoningControl,
    reasoning_effort: reasoningEffort
  };
}

function configResponse(modelTiers) {
  return {
    config: {
      provider: "deepseek",
      request_dialect: "deepseek",
      base_url: "https://model.example",
      model: "model-a",
      has_key: true,
      has_vision_key: false
    },
    model_tiers: modelTiers
  };
}

describe("useConfig", () => {
  it("saves the default root without touching model configuration", async () => {
    const update = vi.spyOn(api, "updateConfig").mockResolvedValue({ config: { default_projects_dir: "/chosen/default" } });
    const config = useConfig();
    config.configForm.default_projects_dir = "/chosen/default";
    await config.saveConfig({ section: "workspace" });
    expect(update).toHaveBeenCalledWith({ default_projects_dir: "/chosen/default" });
  });
  it("withdraws stale tier submission authority while reload is pending and after it fails", async () => {
    const reload = deferred();
    vi.spyOn(api, "getConfig")
      .mockResolvedValueOnce(configResponse([tier()]))
      .mockReturnValueOnce(reload.promise);
    const update = vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_configured: true
    });
    const config = useConfig();

    await config.loadConfig();
    expect(config.tierForm.architect.profile_id).toBe("profile-old");
    expect(config.tierLoaded.value).toBe(true);

    const loading = config.loadConfig();
    expect(config.tierLoaded.value).toBe(false);
    await config.saveConfig();
    expect(update.mock.calls[0][0]).not.toHaveProperty("tier_architect_profile_id");
    expect(update.mock.calls[0][0]).not.toHaveProperty("tier_architect_thinking");
    expect(update.mock.calls[0][0]).not.toHaveProperty("tier_architect_reasoning_effort");

    reload.reject(new Error("configuration unavailable"));
    await loading;
    expect(config.tierLoaded.value).toBe(false);
    await config.saveConfig();
    expect(update.mock.calls[1][0]).not.toHaveProperty("tier_architect_profile_id");
    expect(update.mock.calls[1][0]).not.toHaveProperty("tier_architect_thinking");
    expect(update.mock.calls[1][0]).not.toHaveProperty("tier_architect_reasoning_effort");
  });

  it("keeps a successful save result when the follow-up status refresh fails", async () => {
    vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_configured: true
    });
    const refreshStatus = vi.fn().mockRejectedValue(new Error("status unavailable"));
    const config = useConfig(refreshStatus);

    await config.saveConfig();

    expect(refreshStatus).toHaveBeenCalledOnce();
    expect(config.configMessage.value).toBe("已保存，模型已就绪，可以开始对话。");
    expect(config.configMessage.value).not.toContain("保存失败");
  });

  it("round-trips the global request dialect instead of resetting it", async () => {
    vi.spyOn(api, "getConfig").mockResolvedValue(configResponse([]));
    const update = vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_tiers: [],
      model_configured: true
    });
    const config = useConfig();

    await config.loadConfig();
    expect(config.configForm.request_dialect).toBe("deepseek");
    await config.saveConfig();

    expect(update.mock.calls[0][0].request_dialect).toBe("deepseek");
  });

  it("applies model tiers from a successful save to both effective and editable views", async () => {
    vi.spyOn(api, "getConfig").mockResolvedValue(
      configResponse([tier({ profileId: "profile-old", thinking: false })])
    );
    const confirmed = tier({ profileId: "profile-confirmed", thinking: true });
    vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_tiers: [confirmed],
      model_configured: true
    });
    const config = useConfig();
    await config.loadConfig();
    config.updateTier({ tier: "architect", field: "profile_id", value: "profile-draft" });

    await config.saveConfig();

    expect(config.tierEffective.value).toEqual([confirmed]);
    expect(config.tierForm.architect).toEqual({
      profile_id: "profile-confirmed",
      thinking: true,
      reasoning_effort: ""
    });
  });

  it("loads and saves the graded reasoning effort independently of the legacy switch", async () => {
    vi.spyOn(api, "getConfig").mockResolvedValue(configResponse([
      tier({
        provider: "openai-compatible",
        reasoningControl: "effort",
        reasoningEffort: "high"
      })
    ]));
    const update = vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_tiers: [],
      model_configured: true
    });
    const config = useConfig();

    await config.loadConfig();
    expect(config.tierForm.architect.reasoning_effort).toBe("high");
    config.updateTier({ tier: "architect", field: "reasoning_effort", value: "xhigh" });

    await config.saveConfig();

    expect(update.mock.calls[0][0].tier_architect_reasoning_effort).toBe("xhigh");
  });

  it("submits auto to clear a previously selected reasoning effort", async () => {
    vi.spyOn(api, "getConfig").mockResolvedValue(configResponse([
      tier({
        provider: "openai-compatible",
        reasoningControl: "effort",
        reasoningEffort: "high"
      })
    ]));
    const update = vi.spyOn(api, "updateConfig").mockResolvedValue({
      config: { has_key: true, has_vision_key: false },
      model_tiers: [],
      model_configured: true
    });
    const config = useConfig();

    await config.loadConfig();
    config.updateTier({ tier: "architect", field: "reasoning_effort", value: "" });
    await config.saveConfig();

    expect(update.mock.calls[0][0].tier_architect_reasoning_effort).toBe("auto");
  });
});
