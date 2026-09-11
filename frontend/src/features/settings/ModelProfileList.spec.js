import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ModelProfileList from "./ModelProfileList.vue";

const profile = (overrides = {}) => ({
  id: "mp_1", kind: "model", name: "deepseek 主力", provider: "deepseek",
  request_dialect: "deepseek",
  base_url: "https://api.deepseek.com/anthropic", model: "deepseek-v4-pro",
  quota: {
    requests_per_minute: 10,
    tokens_per_minute: 100000,
    tokens_per_week: 1000000000,
    max_in_flight: 2,
    quota_group: "deepseek-account"
  },
  has_api_key: true, active: false, ...overrides
});

const props = (overrides = {}) => ({
  kind: "model", title: "已保存的对话模型", profiles: [profile()], activeId: "mp_1", ...overrides
});

describe("ModelProfileList", () => {
  it("shows what a saved profile is without showing its key", () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    expect(wrapper.text()).toContain("deepseek-v4-pro");
    expect(wrapper.text()).toContain("已配置");
    expect(wrapper.text()).toContain("使用中");
  });

  it("asks for a key only when one is wanted", async () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.get("button[title='显示密钥']").trigger("click");
    expect(wrapper.emitted("reveal")?.[0]).toEqual(["mp_1"]);

    // 明文由外层取回后传进来；组件自己不持有它。
    const revealing = mount(ModelProfileList, {
      props: props({ revealed: { mp_1: "sk-live" } })
    });
    expect(revealing.text()).toContain("sk-live");
    await revealing.get("button[title='隐藏']").trigger("click");
    expect(revealing.emitted("hide")?.[0]).toEqual(["mp_1"]);
  });

  it("does not offer to activate the one already in use", () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    expect(wrapper.find("button[title='启用这份配置']").exists()).toBe(false);

    const other = mount(ModelProfileList, { props: props({ activeId: "mp_other" }) });
    expect(other.find("button[title='启用这份配置']").exists()).toBe(true);
  });

  it("requires a key when creating but not when editing", async () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.findAll("button").find((b) => b.text().includes("新建配置")).trigger("click");

    const save = () => wrapper.findAll("button").find((b) => b.text() === "保存");
    // 任务模型的 provider 是枚举，所以这里是 select 而不是输入框。
    await wrapper.get("select").setValue("anthropic-compatible");
    const text = wrapper.findAll("input[type='text'], input:not([type])");
    for (const [index, value] of ["名字", "https://x", "m"].entries()) {
      await text[index].setValue(value);
    }
    // 四项齐了但没有 key：新建不允许，因为一份连不上的配置存了也没用。
    expect(save().attributes("disabled")).toBeDefined();
    await wrapper.get("input[type='password']").setValue("sk-new");
    expect(save().attributes("disabled")).toBeUndefined();
  });

  it("never prefills the key when editing, so saving cannot erase it", async () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.get("button[title='编辑']").trigger("click");
    const key = wrapper.get("input[type='password']");
    expect(key.element.value).toBe("");
    expect(key.attributes("placeholder")).toContain("留空则不修改");
  });

  it("edits and submits the request dialect independently from the provider", async () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.get("button[title='编辑']").trigger("click");

    const dialect = wrapper.get("select[data-testid='request-dialect']");
    expect(dialect.element.value).toBe("deepseek");
    await dialect.setValue("openai");
    await wrapper.findAll("button").find((button) => button.text() === "保存").trigger("click");

    expect(wrapper.emitted("update")?.[0]).toEqual([{
      id: "mp_1",
      payload: expect.objectContaining({ request_dialect: "openai" })
    }]);
  });

  it("defaults a newly-created task model to automatic dialect detection", async () => {
    const wrapper = mount(ModelProfileList, {
      props: props({ profiles: [], activeId: null })
    });
    await wrapper.findAll("button").find((button) => button.text().includes("新建配置")).trigger("click");

    expect(wrapper.get("select[data-testid='request-dialect']").element.value).toBe("auto");
  });

  it("edits quota as numbers and keeps an optional shared group", async () => {
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.get("button[title='编辑']").trigger("click");

    expect(wrapper.get("input[data-testid='quota-requests_per_minute']").element.value).toBe("10");
    await wrapper.get("input[data-testid='quota-requests_per_minute']").setValue("12");
    await wrapper.get("input[data-testid='quota-quota_group']").setValue("shared-sjtu");
    await wrapper.findAll("button").find((button) => button.text() === "保存").trigger("click");

    expect(wrapper.emitted("update")?.[0]).toEqual([{
      id: "mp_1",
      payload: expect.objectContaining({
        quota: {
          requests_per_minute: 12,
          tokens_per_minute: 100000,
          tokens_per_week: 1000000000,
          max_in_flight: 2,
          quota_group: "shared-sjtu"
        }
      })
    }]);
  });

  it("omits model quota fields entirely for vision profiles", async () => {
    const wrapper = mount(ModelProfileList, {
      props: { kind: "vision", title: "视觉", profiles: [], activeId: null }
    });
    await wrapper.findAll("button").find((button) => button.text().includes("新建配置")).trigger("click");
    const inputs = wrapper.findAll("input");
    await inputs[0].setValue("highland");
    await inputs[1].setValue("视觉配置");
    await inputs[2].setValue("https://vision.invalid/v1");
    await inputs[3].setValue("vision-model");
    await wrapper.get("input[type='password']").setValue("vision-key");
    await wrapper.findAll("button").find((button) => button.text() === "保存").trigger("click");

    expect(wrapper.emitted("create")?.[0][0]).not.toHaveProperty("quota");
  });

  it("renders an empty list without inventing anything to say", () => {
    const wrapper = mount(ModelProfileList, { props: props({ profiles: [], activeId: null }) });
    expect(wrapper.findAll("li")).toHaveLength(0);
    expect(wrapper.findAll("button").some((b) => b.text().includes("新建配置"))).toBe(true);
  });

  it("asks for a provider the way each kind actually needs", () => {
    // 任务模型的 provider 决定走哪套适配器，后端按枚举校验；
    // 视觉模型的 provider 只是个名字，写什么都行。
    const model = mount(ModelProfileList, { props: props({ profiles: [], activeId: null }) });
    model.findAll("button").find((b) => b.text().includes("新建配置")).trigger("click");

    const vision = mount(ModelProfileList, {
      props: { kind: "vision", title: "视觉", profiles: [], activeId: null }
    });
    return vision.findAll("button").find((b) => b.text().includes("新建配置")).trigger("click")
      .then(() => model.vm.$nextTick())
      .then(() => {
        expect(model.find("select").exists()).toBe(true);
        expect(vision.find("select").exists()).toBe(false);
      });
  });

  it("lets the key being typed be checked, then puts it back", async () => {
    // 粘贴一个 key 之后想核对一眼是常事；但"显示"是一次动作，
    // 不是这个表单的属性——重新进入编辑时必须回到遮住。
    const wrapper = mount(ModelProfileList, { props: props() });
    await wrapper.get("button[title='编辑']").trigger("click");
    expect(wrapper.get("input[type='password']").exists()).toBe(true);

    await wrapper.get("button[title='显示']").trigger("click");
    expect(wrapper.find("input[type='password']").exists()).toBe(false);

    await wrapper.findAll("button").find((b) => b.text() === "取消").trigger("click");
    await wrapper.get("button[title='编辑']").trigger("click");
    expect(wrapper.get("input[type='password']").exists()).toBe(true);
  });
});
