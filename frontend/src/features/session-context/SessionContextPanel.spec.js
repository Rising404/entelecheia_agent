import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import SessionContextPanel from "./SessionContextPanel.vue";

const state = {
  id: "state-1",
  domain: "user",
  state_type: "temporary_preference",
  key: "response_depth",
  value_json: "detailed"
};

const context = {
  user_state: [state],
  task_state: [],
  interaction_state: [],
  counts: { total: 1 },
  latest_reset: null
};

describe("SessionContextPanel", () => {
  it("requests redacted explanation first and makes excerpt reveal explicit", async () => {
    const wrapper = mount(SessionContextPanel, {
      props: {
        sessionContext: context,
        selectedState: state,
        explanation: {
          explanation: {
            support_status: "complete",
            evidence: [{ id: "turn:s1:0", status: "available" }]
          }
        }
      }
    });
    await wrapper.findAll("button").find((button) => button.text().includes("response_depth")).trigger("click");
    await wrapper.findAll("button").find((button) => button.text().includes("显示证据摘录")).trigger("click");
    expect(wrapper.emitted("explain")?.[0][0]).toMatchObject({ state, includeExcerpt: false });
    expect(wrapper.emitted("explain")?.[1][0]).toMatchObject({ state, includeExcerpt: true });
  });

  it("requires reason and preservation acknowledgement before clear", async () => {
    const wrapper = mount(SessionContextPanel, { props: { sessionContext: context } });
    expect(wrapper.text()).not.toContain("长期提升");
    expect(wrapper.text()).not.toContain("长期记忆");
    await wrapper.findAll("button").find((button) => button.text().includes("清空状态")).trigger("click");
    const confirmButton = wrapper.findAll("button").find((button) => button.text().includes("确认清空"));
    expect(confirmButton.attributes("disabled")).toBeDefined();
    await wrapper.get("input[placeholder='为什么要重新开始']").setValue("wrong context");
    await wrapper.find("[role='dialog'] input[type='checkbox']").setValue(true);
    expect(confirmButton.attributes("disabled")).toBeUndefined();
    await confirmButton.trigger("click");
    expect(wrapper.emitted("clear")?.[0]).toEqual(["wrong context"]);
  });

  it("records a correction only after evidence acknowledgement", async () => {
    const correctable = { ...state, correction_operations: ["set", "retract"] };
    const wrapper = mount(SessionContextPanel, {
      props: {
        sessionContext: { ...context, user_state: [correctable] },
        selectedState: correctable,
        explanation: { explanation: { support_status: "complete", evidence: [] } }
      }
    });
    await wrapper.findAll("button").find((button) => button.text().includes("纠正状态")).trigger("click");
    const form = wrapper.get("[data-testid='correction-form']");
    await form.get("textarea[placeholder='字符串或 JSON 值']").setValue('{"depth":"brief"}');
    const recordButton = form.findAll("button").find((button) => button.text().includes("记录并生成预览"));
    expect(recordButton.attributes("disabled")).toBeDefined();
    await form.get("input[type='checkbox']").setValue(true);
    await recordButton.trigger("click");
    expect(wrapper.emitted("create-correction")?.[0][0]).toEqual({
      state: correctable,
      operation: "set",
      value: { depth: "brief" }
    });
  });

  it("shows backend diff and keeps apply behind flag plus confirmation", async () => {
    const preview = {
      preview_token: "repair-preview:abc",
      context_revision: 8,
      candidate_source: "stored",
      changes: {
        changed: [{
          id: "state-1",
          before: { value_json: "detailed" },
          after: { value_json: "brief" }
        }],
        added: [],
        removed: []
      }
    };
    const wrapper = mount(SessionContextPanel, {
      props: {
        sessionContext: context,
        selectedState: state,
        explanation: { explanation: { support_status: "complete", evidence: [] } },
        repairPreview: preview,
        repairApplyEnabled: true
      }
    });
    expect(wrapper.get("[data-testid='repair-preview']").text()).toContain("原：detailed");
    expect(wrapper.get("[data-testid='repair-preview']").text()).toContain("新：brief");
    const applyButton = wrapper.findAll("button").find((button) => button.text().includes("应用 Repair"));
    expect(applyButton.attributes("disabled")).toBeDefined();
    await wrapper.get("[data-testid='repair-preview'] input[type='checkbox']").setValue(true);
    await applyButton.trigger("click");
    expect(wrapper.emitted("apply-repair")).toHaveLength(1);
  });
});
