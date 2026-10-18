import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import RuntimeModeControl from "./RuntimeModeControl.vue";

describe("RuntimeModeControl", () => {
  it("offers only the supported single-Turn mode even with stale capabilities", async () => {
    const wrapper = mount(RuntimeModeControl, {
      props: { modelValue: "turn", routing: { available_processing_levels: ["L0", "L1"] } }
    });
    expect(wrapper.findAll("button").map((b) => b.attributes("aria-label"))).toEqual(["单轮"]);
    expect(wrapper.get("button").attributes("aria-checked")).toBe("true");
    await wrapper.get("button").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });
  it("disables the single-Turn choice when L1 is not advertised", () => {
    const wrapper = mount(RuntimeModeControl, {
      props: { modelValue: "turn", routing: { available_processing_levels: ["L0"] } }
    });
    expect(wrapper.get("button").attributes("disabled")).toBeDefined();
  });
  it("can select L0/L1 from a stale mode", async () => {
    const wrapper = mount(RuntimeModeControl, {
      props: { modelValue: "task", routing: { available_processing_levels: ["L0", "L1"] } }
    });
    await wrapper.get("button").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["turn"]]);
  });
});
