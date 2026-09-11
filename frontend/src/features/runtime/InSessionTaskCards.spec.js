import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import InSessionTaskCards from "./InSessionTaskCards.vue";

const task = {
  insession_task_id: "insession_task_alpha",
  title: "整理论文分析",
  status: "active",
  current_graph_revision: 2,
  nodes: [
    {
      insession_task_node_id: "insession_task_alpha",
      parent_insession_task_node_id: null,
      node_kind: "root",
      ordinal: 0,
      title: "整理论文分析",
      status: "active"
    },
    {
      insession_task_node_id: "insession_task_node_methods",
      parent_insession_task_node_id: "insession_task_alpha",
      node_kind: "subtask",
      ordinal: 1,
      title: "分析方法",
      status: "awaiting_user"
    }
  ],
  related_turn_count: 2
};

describe("InSessionTaskCards", () => {
  it("renders a compact read-only summary of related task state", () => {
    const wrapper = mount(InSessionTaskCards, { props: { tasks: [task] } });

    const card = wrapper.get('[data-testid="insession-task-card"]');
    expect(card.text()).toContain("整理论文分析");
    expect(card.text()).toContain("版本 2");
    expect(card.text()).toContain("分析方法");
    expect(card.text()).toContain("等待你的信息");
    expect(card.text()).toContain("关联 2 个会话回合");
    expect(card.findAll("button")).toHaveLength(0);
  });

  it("hides malformed details rather than rendering expanded state", () => {
    const wrapper = mount(InSessionTaskCards, {
      props: { tasks: [{ ...task, source_anchors: [{ excerpt: "private" }] }] }
    });

    expect(wrapper.find('[data-testid="insession-task-cards"]').exists()).toBe(false);
    expect(wrapper.text()).not.toContain("private");
  });

  it("renders a graphless Task shell without inventing a revision or node", () => {
    const wrapper = mount(InSessionTaskCards, {
      props: {
        tasks: [{ ...task, current_graph_revision: null, nodes: [] }]
      }
    });

    const card = wrapper.get('[data-testid="insession-task-card"]');
    expect(card.text()).toContain("整理论文分析");
    expect(card.text()).toContain("任务图尚未建立");
    expect(card.text()).not.toContain("版本 null");
    expect(card.text()).not.toContain("分析方法");
    expect(card.findAll("button")).toHaveLength(0);
  });
});
