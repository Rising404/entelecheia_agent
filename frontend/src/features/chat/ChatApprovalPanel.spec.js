import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ChatApprovalPanel from "./ChatApprovalPanel.vue";

describe("ChatApprovalPanel", () => {
  it("emits an explicit decision and never executes it itself", async () => {
    const wrapper = mount(ChatApprovalPanel, {
      props: {
        pendingReview: {
          reason: "protected write",
          items: [{ tool_id: "file_write", risk_level: "high", args: { path: "note.md", content: "draft" } }]
        }
      }
    });
    await wrapper.findAll("button").find((button) => button.text().includes("批准")).trigger("click");
    expect(wrapper.emitted("decide")?.[0]).toEqual([true]);
  });

  it("blocks both decisions while the turn is busy", () => {
    const wrapper = mount(ChatApprovalPanel, { props: { pendingReview: { items: [] }, chatBusy: true } });
    const decisions = wrapper.findAll("button").filter((button) => ["批准", "拒绝"].some((text) => button.text().includes(text)));
    expect(decisions).toHaveLength(2);
    expect(decisions.every((button) => button.attributes("disabled") !== undefined)).toBe(true);
  });

  it("shows the recovered pre-execution explanation with the operation payload", () => {
    const wrapper = mount(ChatApprovalPanel, {
      props: {
        pendingReview: {
          schema_version: 1,
          review_id: "review-1",
          reply: "我准备更新项目文档，请确认。",
          pending_writes: [{ tool_id: "file_write", payload: { path: "note.md", content: "draft" } }]
        }
      }
    });

    expect(wrapper.text()).toContain("执行前说明");
    expect(wrapper.text()).toContain("我准备更新项目文档，请确认。");
    expect(wrapper.text()).toContain("note.md");
  });
});
