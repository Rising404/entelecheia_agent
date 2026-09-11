import { describe, expect, it } from "vitest";

import {
  inSessionTaskStatusText,
  isInSessionTaskDetail,
  taskDetailFromResponse
} from "./insessionTaskDetails";

const task = {
  insession_task_id: "insession_task_alpha",
  title: "整理论文分析",
  status: "active",
  current_graph_revision: 1,
  nodes: [{
    insession_task_node_id: "insession_task_alpha",
    parent_insession_task_node_id: null,
    node_kind: "root",
    ordinal: 0,
    title: "整理论文分析",
    status: "active"
  }],
  related_turn_count: 1
};

describe("insessionTaskDetails", () => {
  it("accepts only the narrow public Task projection", () => {
    expect(isInSessionTaskDetail(task)).toBe(true);
    expect(taskDetailFromResponse({ task })).toEqual(task);
  });

  it("accepts a graphless Task shell and rejects mixed shell/graph shapes", () => {
    const shell = {
      ...task,
      current_graph_revision: null,
      nodes: []
    };

    expect(isInSessionTaskDetail(shell)).toBe(true);
    expect(taskDetailFromResponse({ task: shell })).toEqual(shell);
    expect(isInSessionTaskDetail({ ...shell, nodes: task.nodes })).toBe(false);
    expect(isInSessionTaskDetail({ ...task, nodes: [] })).toBe(false);
  });

  it("rejects unexpected fields instead of rendering a broader server object", () => {
    expect(isInSessionTaskDetail({ ...task, source_anchors: [] })).toBe(false);
    expect(taskDetailFromResponse({ task: { ...task, objective: "private" } })).toBeNull();
  });

  it("uses a safe label for known and unknown task statuses", () => {
    expect(inSessionTaskStatusText("awaiting_user")).toBe("等待你的信息");
    expect(inSessionTaskStatusText("unknown")).toBe("状态不可用");
  });
});
