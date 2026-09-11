import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import PendingUserQuestionDialog from "./PendingUserQuestionDialog.vue";

describe("PendingUserQuestionDialog", () => {
  it("shows durable questions and submits through the ordinary composer contract", async () => {
    const wrapper = mount(PendingUserQuestionDialog, {
      props: {
        questions: [{
          question_ref: "question-opaque",
          insession_task_id: "task-1",
          task_title: "准备行程",
          question: "你希望哪天出发？"
        }],
        input: "20 号",
        canAnswer: true
      }
    });

    expect(wrapper.text()).toContain("准备行程");
    expect(wrapper.text()).toContain("你希望哪天出发？");
    expect(wrapper.get('[data-testid="pending-user-question-dialog"]').classes()).toContain("fixed");
    await wrapper.get("textarea").setValue("我先想问一下费用");
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("update:input")?.at(-1)).toEqual(["我先想问一下费用"]);
    expect(wrapper.emitted("send")).toHaveLength(1);
  });

  it("renders multiple independent pending tasks without choosing one", () => {
    const wrapper = mount(PendingUserQuestionDialog, {
      props: {
        questions: [
          { question_ref: "q1", task_title: "任务 A", question: "问题 A" },
          { question_ref: "q2", task_title: "任务 B", question: "问题 B" }
        ],
        canAnswer: true
      }
    });

    expect(wrapper.findAll("ol li")).toHaveLength(2);
    expect(wrapper.text()).toContain("问题 A");
    expect(wrapper.text()).toContain("问题 B");
  });

  it("does not disappear locally or claim an answer when sending is unavailable", () => {
    const wrapper = mount(PendingUserQuestionDialog, {
      props: {
        questions: [{ question_ref: "q1", question: "请补充" }],
        input: "回答",
        canAnswer: false
      }
    });

    expect(wrapper.get("textarea").attributes("disabled")).toBeDefined();
    expect(wrapper.get("button[type='submit']").attributes("disabled")).toBeDefined();
    expect(wrapper.find('[data-testid="pending-user-question-dialog"]').exists()).toBe(true);
  });
});
