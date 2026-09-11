import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import DocumentDetailPanel from "./DocumentDetailPanel.vue";

describe("DocumentDetailPanel", () => {
  it("does not enable save until the shell marks the draft dirty", async () => {
    const wrapper = mount(DocumentDetailPanel, {
      props: {
        document: { id: "d1" },
        form: { title: "Doc", summary: "", tags: "" },
        dirty: false
      }
    });
    const save = wrapper.findAll("button").find((button) => button.text().includes("保存"));
    expect(save.attributes("disabled")).toBeDefined();
    await wrapper.get("textarea").setValue("Summary");
    expect(wrapper.emitted("update-field")?.[0]).toEqual([{ field: "summary", value: "Summary" }]);
  });

  it("shows durable partial-coverage diagnostics instead of implying full understanding", () => {
    const wrapper = mount(DocumentDetailPanel, {
      props: {
        document: {
          id: "d1",
          processing_status: "partial",
          needs_vision: true,
          diagnostics: [
            { code: "page_needs_vision", at: "p4", detail: "figure not interpreted" }
          ]
        },
        form: { title: "Paper", summary: "", tags: "" },
        dirty: false
      }
    });

    expect(wrapper.text()).toContain("部分可读");
    expect(wrapper.text()).toContain("仍有内容需要视觉解析");
    expect(wrapper.text()).toContain("p4");
    expect(wrapper.text()).toContain("figure not interpreted");
  });

  it("does not describe a non-visual partial diagnostic as a vision problem", () => {
    const wrapper = mount(DocumentDetailPanel, {
      props: {
        document: {
          id: "d1",
          processing_status: "partial",
          needs_vision: false,
          diagnostics: [{ code: "fallback_reader_used", detail: "fallback parser" }]
        },
        form: { title: "Paper", summary: "", tags: "" },
        dirty: false
      }
    });

    expect(wrapper.text()).toContain("文档仍有解析覆盖缺口");
    expect(wrapper.text()).not.toContain("需要视觉解析");
  });

  it("explains that a legacy document has unknown coverage", () => {
    const wrapper = mount(DocumentDetailPanel, {
      props: {
        document: { id: "d1", processing_status: "legacy_unknown", diagnostics: null, needs_vision: null },
        form: { title: "Legacy", summary: "", tags: "" },
        dirty: false
      }
    });

    expect(wrapper.text()).toContain("覆盖未知");
    expect(wrapper.text()).toContain("尚无可验证的解析覆盖记录");
  });
});
