import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ChatWorkspace from "./ChatWorkspace.vue";

describe("ChatWorkspace", () => {
  it("offers explicit failed-task intentions without changing the saved reply", async () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        turns: [{ turn_idx: 1, role: "assistant", content: "已保存的答复" }],
        postCommitFailure: { label: "历史检索索引" }, canRecoverPostCommit: true
      }
    });
    expect(wrapper.get('[data-testid="post-commit-recovery"]').text()).toContain("历史检索索引");
    await wrapper.get('[data-testid="retry-post-commit"]').trigger("click");
    await wrapper.get('[data-testid="waive-post-commit"]').trigger("click");
    expect(wrapper.emitted("retry-post-commit")).toEqual([[]]);
    expect(wrapper.emitted("waive-post-commit")).toEqual([[]]);
    expect(wrapper.text()).toContain("已保存的答复");
    await wrapper.setProps({ postCommitRecoveryBusy: true });
    expect(wrapper.get('[data-testid="retry-post-commit"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('[data-testid="waive-post-commit"]').attributes("disabled")).toBeDefined();
    await wrapper.setProps({ postCommitFailure: null });
    expect(wrapper.find('[data-testid="post-commit-recovery"]').exists()).toBe(false);
  });

  it("does not expose or forward a legacy persona_id through composer intentions", async () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", persona_id: "legacy-persona-id", status: "active" },
        title: "Session",
        titleDirty: true,
        input: "hello",
        canSend: true
      }
    });
    await wrapper.get("input.title-input").setValue("Renamed");
    await wrapper.get("form.composer").trigger("submit");
    expect(wrapper.emitted("update:title")?.[0]).toEqual(["Renamed"]);
    expect(wrapper.emitted("send")).toHaveLength(1);
    expect(wrapper.emitted("send")?.[0]).toEqual([]);
    expect(wrapper.text()).not.toContain("legacy-persona-id");

    // 归档与回收站已经移到左侧会话列表的右键菜单，中间栏不再提供。
    expect(wrapper.findAll("button").some((button) => button.text().includes("归档"))).toBe(false);
    expect(wrapper.findAll("button").some((button) => button.text().includes("回收站"))).toBe(false);
  });

  it("keeps the composer disabled when the shell denies sending", () => {
    const wrapper = mount(ChatWorkspace, {
      props: { session: { id: "s1", status: "active" }, input: "blocked", canSend: false }
    });
    expect(wrapper.get("textarea.composer-input").attributes("disabled")).toBeDefined();
    expect(wrapper.findAll("button").find((button) => button.text().includes("发送")).attributes("disabled")).toBeDefined();
  });

  it("keeps task-mode Send clickable for feedback and explains that it is blocked", async () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        input: "start the task",
        canSend: true,
        runtimeMode: "task"
      }
    });

    const sendButton = wrapper.findAll("button").find((button) => button.text().includes("发送"));
    expect(sendButton.attributes("disabled")).toBeUndefined();
    expect(sendButton.attributes("title")).toBe("任务模式当前仍在开发中，暂不提供消息发送功能。");

    // Vue Test Utils 不执行 submit 按钮的浏览器默认行为，直接触发表单提交来验证
    // 这颗可点击按钮连接到的组件事件契约。
    await wrapper.get("form.composer").trigger("submit");
    expect(wrapper.emitted("send")).toHaveLength(1);
  });

  it("replaces the ordinary composer with a durable pending-question dialog", async () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        input: "20 号",
        pendingUserQuestions: [{
          insession_task_id: "task-trip",
          question: "你希望哪天出发？"
        }],
        canAnswerPendingQuestion: true
      }
    });

    expect(wrapper.find('[data-testid="pending-user-question-dialog"]').exists()).toBe(true);
    expect(wrapper.find("form.composer").exists()).toBe(false);
    await wrapper.get('[data-testid="pending-user-question-dialog"] form').trigger("submit");
    expect(wrapper.emitted("answer-pending-question")).toHaveLength(1);
  });

  it("shows staged attachments and marks the ones that cannot be read", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        canSend: true,
        pendingAttachments: [
          { attachment_id: "att_1", name: "shot.png", size_bytes: 2048, kind: "image", readable: true },
          { attachment_id: "att_2", name: "voice.mp3", size_bytes: 4096, kind: "audio", readable: false }
        ]
      }
    });

    const text = wrapper.text();
    expect(text).toContain("shot.png");
    expect(text).toContain("voice.mp3");
    // 不可读文件不仅要在模型 manifest 中标记，也要在编辑器中标记，避免回答让用户意外。
    expect(text).toContain("仅记录");
  });

  it("emits the removal of a staged attachment without handling it locally", async () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        canSend: true,
        pendingAttachments: [
          { attachment_id: "att_1", name: "shot.png", size_bytes: 2048, kind: "image", readable: true }
        ]
      }
    });

    await wrapper.get("li button").trigger("click");

    expect(wrapper.emitted("remove-attachment")?.[0]).toEqual(["att_1"]);
  });

  it("delegates history-folder moves to the shell without handling persistence", async () => {
    // 这个下拉在会话元信息那一行里，默认视图已经把整行收起来了。
    // 移动会话到文件夹因此暂时只在开发者视图下可达。
    const wrapper = mount(ChatWorkspace, {
      props: {
        developerMode: true,
        session: { id: "s1", status: "active", folder_id: "folder-a" },
        historyFolders: [
          { id: "folder-a", name: "项目", depth: 0, path: "项目" },
          { id: "folder-b", name: "研究", depth: 0, path: "研究" }
        ]
      }
    });

    await wrapper.get("select[aria-label='移动会话到历史文件夹']").setValue("folder-b");
    expect(wrapper.emitted("move-session-history")).toEqual([[{ sessionId: "s1", folderId: "folder-b" }]]);
  });

  it("shows the backend workspace as read-only metadata without bind or switch controls", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        developerMode: true,
        session: { id: "s1", status: "active", working_dir: "/workspace/fixed" }
      }
    });

    expect(wrapper.text()).toContain("/workspace/fixed");
    expect(wrapper.findAll("button").some((button) => /绑定目录|更换目录/.test(button.text()))).toBe(false);
    expect(wrapper.emitted("bind-working-directory")).toBeUndefined();
  });

  it("releases the fixed desktop height when the workspace becomes one column", () => {
    const wrapper = mount(ChatWorkspace, {
      props: { session: { id: "s1", status: "active" } }
    });

    expect(wrapper.classes()).toContain("max-[780px]:h-auto");
    expect(wrapper.get(".message-stream").classes()).toContain("max-[780px]:flex-none");
    expect(wrapper.get(".message-stream").classes()).toContain("max-[780px]:min-h-64");
  });

  it("shows a provisional assistant message while delta text is arriving", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        streamingReply: "正在逐字回复"
      }
    });
    expect(wrapper.get('[aria-label="模型正在回复"]').text()).toContain("正在逐字回复");
  });

  it("does not expose internal runtime lifecycle cards in the conversation", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        runtimeEvents: [
          { schema_version: 1, event_id: "evt_1", sequence: 1, session_id: "s1", turn_id: "turn_1", stage: "TOOL", status: "started", occurred_at: "2026-08-11T00:00:00Z", prompt_replay: false, operation_id: "op_1" },
          { schema_version: 1, event_id: "evt_2", sequence: 2, session_id: "s1", turn_id: "turn_1", stage: "TOOL", status: "completed", occurred_at: "2026-08-11T00:00:01Z", prompt_replay: false, operation_id: "op_1" }
        ]
      }
    });
    expect(wrapper.findAll('[data-testid="runtime-activity-card"]')).toHaveLength(0);
    expect(wrapper.text()).not.toContain("调用能力");
  });

  it("shows an incomplete Turn as a sanitized status card instead of a draft", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        incompleteTurn: {
          endReason: "provider_unavailable",
          errorCode: "MODEL_TIMEOUT"
        }
      }
    });

    expect(wrapper.get('[data-testid="incomplete-turn-status"]').text()).toContain("本轮未完成");
    expect(wrapper.text()).toContain("模型服务暂时不可用");
    expect(wrapper.text()).toContain("MODEL_TIMEOUT");
  });

  it("renders related in-session tasks as a read-only card", () => {
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        insessionTaskDetails: [{
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
        }]
      }
    });

    expect(wrapper.get('[data-testid="insession-task-card"]').text()).toContain("整理论文分析");
    expect(wrapper.get('[data-testid="insession-task-card"]').findAll("button")).toHaveLength(0);
  });


  // --- 把文件拖进来 -------------------------------------------------------------

  function fileDrag(type, files = [new File(["x"], "a.png", { type: "image/png" })]) {
    const transfer = { types: ["Files"], files };
    return [type, { dataTransfer: transfer }];
  }

  const chatProps = (extra = {}) => ({
    session: { id: "s1", status: "active" },
    canSend: true,
    ...extra
  });

  it("offers to take files that are dragged over the conversation", async () => {
    const wrapper = mount(ChatWorkspace, { props: chatProps() });
    await wrapper.get("div.relative").trigger(...fileDrag("dragenter"));

    const overlay = wrapper.get('[data-testid="attachment-drop-overlay"]');
    expect(overlay.text()).toContain("松手上传");
    expect(overlay.text()).toContain("16");
  });

  it("says so rather than accepting a file it could not send", async () => {
    const wrapper = mount(ChatWorkspace, { props: chatProps({ canSend: false }) });
    await wrapper.get("div.relative").trigger(...fileDrag("dragenter"));
    expect(wrapper.get('[data-testid="attachment-drop-overlay"]').text()).toContain("当前无法添加附件");

    await wrapper.get("div.relative").trigger(...fileDrag("drop"));
    expect(wrapper.emitted("attach-files")).toBeUndefined();
  });

  it("hands every dropped file to the shell at once", async () => {
    const wrapper = mount(ChatWorkspace, { props: chatProps() });
    const files = [
      new File(["a"], "a.png", { type: "image/png" }),
      new File(["b"], "b.pdf", { type: "application/pdf" })
    ];
    await wrapper.get("div.relative").trigger(...fileDrag("drop", files));
    expect(wrapper.emitted("attach-files")?.[0][0]).toHaveLength(2);
  });

  it("ignores a drag that carries no files", async () => {
    const wrapper = mount(ChatWorkspace, { props: chatProps() });
    await wrapper.get("div.relative").trigger("dragenter", {
      dataTransfer: { types: ["text/plain"], files: [] }
    });
    expect(wrapper.find('[data-testid="attachment-drop-overlay"]').exists()).toBe(false);
  });

  it("keeps the hint up while the pointer crosses inner elements", async () => {
    // 浏览器会先发子元素的 dragenter、再发父元素的 dragleave。用布尔量记状态，
    // 提示会在拖动途中闪烁；这里用计数，所以只有真正离开才收起。
    const wrapper = mount(ChatWorkspace, { props: chatProps() });
    const root = wrapper.get("div.relative");
    await root.trigger(...fileDrag("dragenter"));
    await root.trigger(...fileDrag("dragenter"));
    await root.trigger(...fileDrag("dragleave"));
    expect(wrapper.find('[data-testid="attachment-drop-overlay"]').exists()).toBe(true);

    await root.trigger(...fileDrag("dragleave"));
    expect(wrapper.find('[data-testid="attachment-drop-overlay"]').exists()).toBe(false);
  });

  it("shows how much of the per-message allowance is used", () => {
    const wrapper = mount(ChatWorkspace, {
      props: chatProps({
        pendingAttachments: [
          { attachment_id: "a1", name: "a.png", size_bytes: 10, readable: true }
        ]
      })
    });
    expect(wrapper.text()).toContain("1 / 16");
  });

  it("keeps session metadata out of the ordinary view", () => {
    // 状态、工作目录、历史归档说的是"这个会话是什么"，不是对话内容。
    const wrapper = mount(ChatWorkspace, {
      props: {
        session: { id: "s1", status: "active" },
        canSend: true
      }
    });
    expect(wrapper.text()).not.toContain("未绑定工作目录");
  });
});
