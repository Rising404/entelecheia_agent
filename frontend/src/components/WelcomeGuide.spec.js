import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import WelcomeGuide from "./WelcomeGuide.vue";

describe("WelcomeGuide", () => {
  it("prompts to configure the model and emits go-settings when unconfigured", async () => {
    const wrapper = mount(WelcomeGuide, { props: { modelConfigured: false } });
    // 断言的是"有没有那条出路"，不是那句话怎么写的。
    const button = wrapper.get("button");
    await button.trigger("click");
    expect(wrapper.emitted("go-settings")).toBeTruthy();
  });

  it("shows the model as ready and hides the settings button when configured", () => {
    const wrapper = mount(WelcomeGuide, { props: { modelConfigured: true } });
    expect(wrapper.find("button").exists()).toBe(false);
  });
});
