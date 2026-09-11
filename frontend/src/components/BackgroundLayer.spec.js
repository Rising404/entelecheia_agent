import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import BackgroundLayer from "./BackgroundLayer.vue";

const image = { kind: "image", media_type: "image/jpeg", url: "/api/appearance/background/asset?v=1" };
const video = { kind: "video", media_type: "video/mp4", url: "/api/appearance/background/asset?v=2" };

function render(props) {
  return mount(BackgroundLayer, { props });
}

describe("BackgroundLayer", () => {
  it("优先用已鉴权取回的 object URL", () => {
    const wrapper = render({ background: image, objectUrl: "blob:stub-1" });
    expect(wrapper.get("img").attributes("src")).toBe("blob:stub-1");
  });

  it("字节还没取回时什么都不渲染，绝不直接引用后端地址", () => {
    // 直接引用只会换来一次 401 并被 ORB 拦掉，等于每次加载都发一个注定失败的请求。
    const wrapper = render({ background: image, objectUrl: "" });
    expect(wrapper.find(".bg-layer").exists()).toBe(false);
    expect(wrapper.find("img").exists()).toBe(false);
  });

  it("视频背景同样走 object URL", () => {
    const wrapper = render({ background: video, objectUrl: "blob:stub-2" });
    const element = wrapper.get("video");
    expect(element.attributes("src")).toBe("blob:stub-2");
    // 带声音的自动播放会被浏览器拦掉，整块背景就黑了。muted 由 Vue 作为 DOM 属性写入，
    // 不出现在 attributes 里，所以断言元素本身。
    expect(element.element.muted).toBe(true);
    expect(element.attributes("playsinline")).toBeDefined();
  });

  it("没有背景时什么都不渲染，交给渐变兜底", () => {
    const wrapper = render({ background: null, objectUrl: "" });
    expect(wrapper.find(".bg-layer").exists()).toBe(false);
  });
});
