import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import RuntimeActivityRail from "./RuntimeActivityRail.vue";

function runtimeEvent(eventId, sequence, overrides = {}) {
  return {
    schema_version: 1,
    event_id: eventId,
    sequence,
    session_id: "s1",
    turn_id: "turn-1",
    stage: "TOOL",
    status: "started",
    occurred_at: "2026-07-15T00:00:00Z",
    error_code: null,
    retryable: false,
    insession_task_id: null,
    work_run_id: null,
    attempt_id: "attempt-1",
    operation_id: null,
    prompt_replay: false,
    ...overrides
  };
}

describe("RuntimeActivityRail", () => {
  it("folds the same stage and attempt to its latest public lifecycle status", () => {
    const wrapper = mount(RuntimeActivityRail, {
      props: {
        events: [
          runtimeEvent("evt-1", 1),
          runtimeEvent("evt-2", 2, { status: "completed" })
        ],
        running: true
      }
    });

    expect(wrapper.findAll('[data-testid="runtime-activity-card"]')).toHaveLength(1);
    expect(wrapper.text()).toContain("调用能力");
    expect(wrapper.text()).toContain("完成");
    expect(wrapper.text()).toContain("实时更新");
  });

  it("keeps identical attempt ids separate across turns", () => {
    const wrapper = mount(RuntimeActivityRail, {
      props: {
        events: [
          runtimeEvent("evt-1", 1, { turn_id: "turn-1", status: "completed" }),
          runtimeEvent("evt-2", 2, { turn_id: "turn-2" })
        ]
      }
    });

    expect(wrapper.findAll('[data-testid="runtime-activity-card"]')).toHaveLength(2);
    expect(wrapper.text()).toContain("活动记录");
  });

  it("renders an unknown public stage and status as a generic safe card", () => {
    const wrapper = mount(RuntimeActivityRail, {
      props: {
        events: [
          runtimeEvent("evt-unknown", 1, {
            stage: "FUTURE_STAGE",
            status: "future_status",
            error_code: "FUTURE_ERROR"
          })
        ]
      }
    });

    expect(wrapper.find('[data-testid="runtime-activity-card"]').exists()).toBe(true);
    expect(wrapper.text()).toContain("运行阶段：FUTURE_STAGE");
    expect(wrapper.text()).toContain("状态：future_status");
    expect(wrapper.text()).toContain("错误代码：FUTURE_ERROR");
  });
});
