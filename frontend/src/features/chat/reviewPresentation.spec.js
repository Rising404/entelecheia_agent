import { describe, expect, it } from "vitest";

import {
  emptyReviewResultText,
  fileWritePreview,
  outcomeIssueText,
  outcomePartialText,
  reviewPayloadFields,
  resumeDecisionText,
  summarizeCommittedResult
} from "./presentation";


describe("chat presentation", () => {
  it("orders payload fields and caps file previews", () => {
    const item = {
      tool_id: "file_write",
      payload: {
        source: "agent",
        path: "/tmp/report.md",
        content: Array.from({ length: 45 }, (_, index) => `line ${index + 1}`).join("\n")
      }
    };
    expect(reviewPayloadFields(item).map((field) => field.key)).toEqual(["content", "path", "source"]);
    const preview = fileWritePreview(item);
    expect(preview.lines).toHaveLength(40);
    expect(preview.omitted).toBe(5);
  });

  it("keeps review and partial outcome wording stable", () => {
    expect(resumeDecisionText(true)).toBe("已批准");
    expect(emptyReviewResultText(false)).toContain("没有提交任何写入");
    expect(outcomePartialText({ partial: { failed_tools: ["a", "b"] } })).toBe("2 个工具未成功执行。");
  });

  it("hides runtime-only scope and does not special-case retired task results", () => {
    const fields = reviewPayloadFields({
      tool_id: "file_write",
      payload: {
        content: "验收记录",
        path: "report.md",
        _workspace_session_id: "session-internal"
      }
    });
    expect(fields.map((field) => field.key)).toEqual(["content", "path"]);
    expect(summarizeCommittedResult({
      written: true,
      created: true,
      task: { id: "task-1", name: "验收任务" }
    })).toBe("操作已完成");
  });

  it("explains bounded timeout, context, and tool-loop failures without leaking internals", () => {
    expect(outcomeIssueText({
      code: "MODEL_CALL_TIMEOUT",
      details: { timeout_s: 60 }
    })).toBe("模型等待超过 60 秒，本轮已停止；确认网络或服务状态后可重试。");
    expect(outcomeIssueText({
      code: "TOOL_LOOP_LIMIT_REACHED",
      details: { max_rounds: 10 }
    })).toBe("工具在 10 轮调用后仍未收敛，本轮已安全停止；不会自动重放。");
    expect(outcomeIssueText({ code: "TOOL_LOOP_LIMIT_REACHED", details: { max_rounds: 1000 } }))
      .toBe("工具调用达到安全上限，本轮已安全停止；不会自动重放。");
    expect(outcomeIssueText({
      code: "CONTEXT_BUDGET_EXCEEDED",
      message: "private assembly detail",
      details: { guard_limit: 1_000, estimated_tokens: 1_250 }
    })).toBe("当前输入与必要上下文超过安全预算，模型尚未被调用；请缩短输入或拆分任务后再发送。");
  });
});
