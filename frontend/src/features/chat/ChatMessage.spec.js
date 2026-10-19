import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import ChatMessage from "./ChatMessage.vue";

afterEach(() => vi.unstubAllGlobals());

function reply(content, extra = {}) {
  return mount(ChatMessage, { props: { role: "assistant", content, ...extra } });
}

describe("ChatMessage", () => {
  it("renders document answers with headings, emphasis, lists, quotes, tables and escaped code", () => {
    const wrapper = reply([
      "# 检索结果", "", "## 证据", "", "**三类**，*见原文*，~~旧结论~~。", "",
      "- document_01.pdf", "- 物理第 3 页", "", "1. 定位", "2. 核实", "",
      "> 原文摘录", "", "| 类别 | 数量 |", "| --- | ---: |", "| 全部 | 3 |", "",
      "使用 `count`。", "", "```html", '<img src="x" onerror="alert(1)">', "```"
    ].join("\n"));
    expect(wrapper.get("h1").text()).toBe("检索结果");
    expect(wrapper.get("h2").text()).toBe("证据");
    expect(wrapper.get("strong").text()).toBe("三类");
    expect(wrapper.get("em").text()).toBe("见原文");
    expect(wrapper.get("s").text()).toBe("旧结论");
    expect(wrapper.findAll("ul > li")).toHaveLength(2);
    expect(wrapper.findAll("ol > li")).toHaveLength(2);
    expect(wrapper.get("blockquote").text()).toBe("原文摘录");
    expect(wrapper.findAll("th").map((cell) => cell.text())).toEqual(["类别", "数量"]);
    expect(wrapper.findAll("td").map((cell) => cell.text())).toEqual(["全部", "3"]);
    expect(wrapper.get("p > code").text()).toBe("count");
    expect(wrapper.get("pre code").text()).toContain('<img src="x" onerror="alert(1)">');
    expect(wrapper.find("img").exists()).toBe(false);
  });

  it("keeps HTML inert and never loads image resources", () => {
    const wrapper = reply([
      '<script>alert(1)</script>', '<img src="file:///private/data" onerror="alert(1)">',
      '<iframe src="https://example.com"></iframe>', '<svg onload="alert(1)"></svg>',
      "", "![外部图片](https://example.com/tracker.png)",
      "![<img onerror=alert(1)>](https://example.com/image.png)"
    ].join("\n"));
    expect(wrapper.find("script, img, iframe, svg[onload]").exists()).toBe(false);
    expect(wrapper.get(".markdown-body").text()).toContain('<script>alert(1)</script>');
    expect(wrapper.get(".markdown-body").text()).toContain("外部图片");
  });

  it("only permits web and email links, opening them outside the chat page", () => {
    const wrapper = reply([
      "[来源](https://example.com/report?q=1&n=2)", "[邮箱](mailto:reader@example.com)",
      "[脚本](javascript:alert%281%29)", "[编码脚本](jav&#x61;script:alert%281%29)",
      "[数据](data:text/html,test)", "[文件](file:///private/data)",
      "[相对文件](../secret.txt)", "[其他程序](custom:open)"
    ].join("\n"));
    const links = wrapper.findAll("a");
    expect(links.map((link) => link.attributes("href"))).toEqual([
      "https://example.com/report?q=1&n=2", "mailto:reader@example.com"
    ]);
    for (const link of links) {
      expect(link.attributes("target")).toBe("_blank");
      expect(link.attributes("rel")).toBe("noopener noreferrer");
    }
  });

  it("copies the exact Markdown source, including whitespace and CRLF, and reports success", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    const original = "## 结论\r\n\r\n**三类**  \r\n`原文`\r\n";
    const wrapper = reply(original);
    await wrapper.get('[aria-label="复制 Markdown 原文"]').trigger("click");
    await flushPromises();
    expect(writeText).toHaveBeenCalledExactlyOnceWith(original);
    expect(wrapper.get('[role="status"]').text()).toBe("已复制");
    await wrapper.setProps({ content: "新的回复" });
    expect(wrapper.get('[role="status"]').text()).toBe("复制");
  });

  it("reports clipboard denial and allows retry without claiming success", async () => {
    const writeText = vi.fn().mockRejectedValueOnce(new Error("denied")).mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    const wrapper = reply("**保留原文**");
    await wrapper.get("button").trigger("click");
    await flushPromises();
    expect(wrapper.get('[role="alert"]').text()).toContain("复制失败");
    expect(wrapper.text()).not.toContain("已复制");
    await wrapper.get("button").trigger("click");
    await flushPromises();
    expect(wrapper.find('[role="alert"]').exists()).toBe(false);
    expect(wrapper.get('[role="status"]').text()).toBe("已复制");
    expect(writeText).toHaveBeenLastCalledWith("**保留原文**");
  });

  it("reports an unavailable clipboard instead of silently succeeding", async () => {
    vi.stubGlobal("navigator", {});
    const wrapper = reply("答复");
    await wrapper.get("button").trigger("click");
    await flushPromises();
    expect(wrapper.get('[role="alert"]').text()).toContain("复制失败");
  });

  it("updates partial Markdown while streaming and only offers copying after completion", async () => {
    const wrapper = reply("## 回复\n\n```python\nprint(", { streaming: true });
    expect(wrapper.get("h2").text()).toBe("回复");
    expect(wrapper.get("pre code").text()).toContain("print(");
    expect(wrapper.find("button").exists()).toBe(false);
    await wrapper.setProps({ content: "## 回复\n\n```python\nprint(3)\n```", streaming: false });
    expect(wrapper.get("pre code").element.textContent).toBe("print(3)\n");
    expect(wrapper.find('[aria-label="复制 Markdown 原文"]').exists()).toBe(true);
  });

  it.each(["user", "system", "tool"])("keeps %s messages as literal text", (role) => {
    const wrapper = reply("# 原文\n**内容** <b>标签</b>", { role });
    expect(wrapper.find("h1, strong, b, button").exists()).toBe(false);
    expect(wrapper.get("p").text()).toBe("# 原文\n**内容** <b>标签</b>");
    if (role === "user") expect(wrapper.get("header").text()).toBe("用户");
  });
});
