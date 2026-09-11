import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import FeedbackBanners from "./FeedbackBanners.vue";

describe("FeedbackBanners", () => {
  it("delegates unsaved discard and settings navigation to the shell", async () => {
    const wrapper = mount(FeedbackBanners, {
      props: { mode: "chat", modelConfigured: false, hasUnsavedChanges: true, unsavedSummary: "任务详情" }
    });

    await wrapper.findAll("button")[0].trigger("click");
    await wrapper.findAll("button")[1].trigger("click");
    expect(wrapper.emitted("open-settings")).toHaveLength(1);
    expect(wrapper.emitted("discard")).toHaveLength(1);
  });

  it("announces a newly rendered notice to assistive technology", () => {
    const wrapper = mount(FeedbackBanners, {
      props: {
        mode: "chat",
        modelConfigured: true,
        notice: "任务模式当前仍在开发中，暂不提供消息发送功能。"
      }
    });

    const notice = wrapper.get('[role="status"]');
    expect(notice.attributes("aria-live")).toBe("polite");
    expect(notice.text()).toContain("任务模式当前仍在开发中");
  });

  it("offers one-click sidecar wake only when the desktop API is unreachable", async () => {
    const wrapper = mount(FeedbackBanners, {
      props: {
        mode: "chat",
        statusLoaded: true,
        apiReachable: false,
        canWakeApi: true,
        wakingApi: false,
        modelConfigured: false
      }
    });

    expect(wrapper.get('[data-testid="api-offline-banner"]').text()).toContain("不需要打开命令行");
    const button = wrapper.get('[data-testid="api-offline-banner"] button');
    expect(button.text()).toContain("唤醒本地服务");
    await button.trigger("click");
    expect(wrapper.emitted("wake-api")).toHaveLength(1);
    expect(wrapper.text()).not.toContain("模型未配置，对话不会有真实回复");
  });

  it("does not offer a process-launch button in the browser fallback", () => {
    const wrapper = mount(FeedbackBanners, {
      props: {
        mode: "chat",
        statusLoaded: true,
        apiReachable: false,
        canWakeApi: false
      }
    });

    expect(wrapper.find('[data-testid="api-offline-banner"] button').exists()).toBe(false);
  });

  it("renders bounded runtime failure guidance from the stable outcome code", () => {
    const wrapper = mount(FeedbackBanners, {
      props: {
        mode: "chat",
        error: "工具任务未在安全边界内完成，已中断本轮。",
        errorOutcome: {
          status: "failed",
          error: {
            domain: "runtime",
            code: "TOOL_LOOP_LIMIT_REACHED",
            details: { max_rounds: 10 }
          },
          retry: { action: "retry_turn", label: "调整任务后重试" }
        }
      }
    });

    expect(wrapper.text()).toContain("工具在 10 轮调用后仍未收敛");
    expect(wrapper.text()).toContain("不会自动重放");
    expect(wrapper.text()).toContain("runtime / TOOL_LOOP_LIMIT_REACHED");
    expect(wrapper.text()).toContain("调整任务后重试");
  });

  it("renders a safe actionable context-limit message instead of the internal error", () => {
    const wrapper = mount(FeedbackBanners, {
      props: {
        mode: "chat",
        error: "上下文超过安全预算，已在模型调用前停止",
        errorOutcome: {
          status: "failed",
          error: {
            domain: "context",
            code: "CONTEXT_BUDGET_EXCEEDED",
            message: "private prompt assembly failure",
            details: { guard_limit: 1_000, estimated_tokens: 1_250 }
          },
          retry: { action: "ask_user", label: "检查配置或输入" }
        }
      }
    });

    expect(wrapper.text()).toContain("模型尚未被调用");
    expect(wrapper.text()).toContain("缩短输入或拆分任务");
    expect(wrapper.text()).not.toContain("private prompt assembly failure");
  });
});
