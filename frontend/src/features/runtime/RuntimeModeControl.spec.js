import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import RuntimeModeControl from "./RuntimeModeControl.vue";

describe("RuntimeModeControl", () => {
  it("emits exactly one of the three Runtime modes", async () => {
    const wrapper = mount(RuntimeModeControl, {
      props: {
        modelValue: "turn",
        routing: { available_processing_levels: ["L0", "L1", "L2"] }
      }
    });

    await wrapper.get('button[aria-label="任务"]').trigger("click");

    expect(wrapper.emitted("update:modelValue")).toEqual([["task"]]);
    expect(wrapper.get('button[aria-label="单轮"]').attributes("aria-checked")).toBe("true");
    expect(wrapper.get('button[aria-label="任务"]').attributes("title")).toContain("仍在开发中");
  });

  it("disables only the gated L1 choice when the Host does not advertise it", () => {
    const wrapper = mount(RuntimeModeControl, {
      props: {
        modelValue: "task",
        routing: { available_processing_levels: ["L0", "L2"] }
      }
    });

    expect(wrapper.get('button[aria-label="单轮"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('button[aria-label="任务"]').attributes("disabled")).toBeUndefined();
  });

  it("offers only the two modes that let the model choose", () => {
    // "本轮仅允许直接回答"把选择权从模型手里拿走了；单轮与任务本来就都是
    // "在直接回答与升级之间由模型决定"，所以只留这两个。
    const wrapper = mount(RuntimeModeControl, {
      props: {
        modelValue: "task",
        routing: { available_processing_levels: ["L0", "L1", "L2"] }
      }
    });

    const labels = wrapper.findAll("button").map((b) => b.attributes("aria-label"));
    expect(labels).toEqual(["单轮", "任务"]);
  });
});
