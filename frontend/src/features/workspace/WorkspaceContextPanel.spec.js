import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import WorkspaceContextPanel from "./WorkspaceContextPanel.vue";

describe("WorkspaceContextPanel", () => {
  it("exposes document intake without the retired workspace task controls", async () => {
    const wrapper = mount(WorkspaceContextPanel, {
      props: { documents: [] }
    });

    await wrapper.get("button.cmd").trigger("click");
    expect(wrapper.emitted("add-document")).toHaveLength(1);
    expect(wrapper.text()).not.toContain("工作区任务");
    expect(wrapper.find("form").exists()).toBe(false);
  });

  it("marks partial and legacy document coverage in the visible workspace list", () => {
    const wrapper = mount(WorkspaceContextPanel, {
      props: {
        documents: [
          { id: "partial", title: "带图论文", processing_status: "partial", needs_vision: true },
          { id: "legacy", title: "旧索引", processing_status: "legacy_unknown", needs_vision: null }
        ]
      }
    });

    expect(wrapper.text()).toContain("需补图");
    expect(wrapper.text()).toContain("覆盖未知");
  });

  it("does not offer document intake before the session has a working directory", () => {
    const wrapper = mount(WorkspaceContextPanel, {
      props: { documents: [], canAddDocument: false }
    });

    // 保留的是"为什么现在不能做"这条状态提示，去掉的是解释收录机制的说明。
    expect(wrapper.text()).toContain("当前会话没有可用的本地工作目录");
    expect(wrapper.get("button.cmd").attributes("disabled")).toBeDefined();
  });

  it("shows durable intake progress, safe coverage, and an explicit retry action", async () => {
    const wrapper = mount(WorkspaceContextPanel, {
      props: {
        documents: [],
        documentIngestJobs: [
          {
            job_id: "running",
            session_id: "session-a",
            path: "paper.pdf",
            status: "running",
            stage: "indexing",
            can_retry: false
          },
          {
            job_id: "partial",
            session_id: "session-a",
            path: "figures.pdf",
            status: "succeeded",
            stage: "active",
            processing_status: "partial",
            needs_vision: true,
            can_retry: false
          },
          {
            job_id: "failed",
            session_id: "session-a",
            path: "broken.pdf",
            status: "failed",
            stage: "parsing",
            can_retry: true,
            error: { code: "DOCUMENT_PARSE_FAILED", message: "文件格式无法解析" }
          }
        ]
      }
    });

    expect(wrapper.text()).toContain("正在建立检索索引");
    expect(wrapper.text()).toContain("已收录，需补充视觉解析");
    expect(wrapper.text()).toContain("文件格式无法解析");

    await wrapper.get("button.ingest-retry").trigger("click");
    expect(wrapper.emitted("retry-document-ingest")).toEqual([[
      expect.objectContaining({ job_id: "failed" })
    ]]);
  });
});
