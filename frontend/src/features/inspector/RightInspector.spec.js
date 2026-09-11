import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import RightInspector from "./RightInspector.vue";

describe("RightInspector", () => {
  it("emits refresh and debug intentions without owning state", async () => {
    const wrapper = mount(RightInspector, { props: { mode: "chat", apiOk: true, developerMode: true } });
    await wrapper.get("button[title='重连 API']").trigger("click");
    await wrapper.get("button.debug-toggle").trigger("click");
    expect(wrapper.emitted("refresh")).toHaveLength(1);
    expect(wrapper.emitted("update:debug-open")?.[0]).toEqual([true]);
  });

  it("keeps refresh disabled while a protected turn is busy", () => {
    const wrapper = mount(RightInspector, { props: { mode: "chat", chatBusy: true, developerMode: true } });
    expect(wrapper.get("button[title='重连 API']").attributes("disabled")).toBeDefined();
  });

  it("keeps the local-directory surface separate from chat controls", () => {
    const wrapper = mount(RightInspector, {
      props: { mode: "chat", apiOk: true },
      slots: {
        "local-directory": '<div data-testid="local-directory-slot">本地目录面板</div>',
        "chat-controls": '<div data-testid="chat-controls-slot">上下文治理</div>'
      }
    });

    expect(wrapper.get('[data-testid="local-directory-slot"]').text()).toBe("本地目录面板");
    expect(wrapper.get('[data-testid="chat-controls-slot"]').text()).toBe("上下文治理");
  });

  it("keeps internal stream stages out of the ordinary chat inspector", () => {
    const wrapper = mount(RightInspector, {
      props: {
        mode: "chat",
        apiOk: true,
        chatStreamEvents: [
          { at: "12:00", event: "tool_started", detail: "internal tool call" }
        ]
      }
    });

    expect(wrapper.text()).not.toContain("internal tool call");
    expect(wrapper.text()).not.toContain("本轮状态");
  });

  it("keeps developer surfaces out of the ordinary view", () => {
    // 连接检查和 Debug 服务于排查，不服务于使用。默认视图里它们不该占位置，
    // 也不该只是折叠起来——一个收起的开发者面板仍然在告诉用户这里有他要管的东西。
    const wrapper = mount(RightInspector, {
      props: { mode: "chat", apiOk: true, systemStatus: { components: { api: { label: "HTTP API", ok: true } } } }
    });

    expect(wrapper.text()).not.toContain("连接");
    expect(wrapper.text()).not.toContain("HTTP API");
    expect(wrapper.find("button.debug-toggle").exists()).toBe(false);
    expect(wrapper.find("button[title='重连 API']").exists()).toBe(false);
  });

  it("brings them back when the developer view is on", () => {
    const wrapper = mount(RightInspector, {
      props: {
        mode: "chat", apiOk: true, developerMode: true,
        systemStatus: { components: { api: { label: "HTTP API", ok: true } } }
      }
    });

    expect(wrapper.text()).toContain("HTTP API");
    expect(wrapper.find("button.debug-toggle").exists()).toBe(true);
  });
});
