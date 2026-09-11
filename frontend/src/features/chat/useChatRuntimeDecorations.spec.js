import { afterEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";

import { useChatRuntimeDecorations } from "./useChatRuntimeDecorations";

afterEach(() => {
  vi.clearAllMocks();
});

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

function task(taskId) {
  return {
    insession_task_id: taskId,
    title: `Task ${taskId}`,
    status: "active",
    current_graph_revision: null,
    nodes: [],
    related_turn_count: 0
  };
}

function harness(api, selectedId = "") {
  const selectedSessionId = ref(selectedId);
  return {
    ...useChatRuntimeDecorations({
      api,
      selectedSessionId
    }),
    selectedSessionId
  };
}

describe("useChatRuntimeDecorations", () => {
  it("deduplicates, caps, validates, and silently tolerates related task reads", async () => {
    const ids = ["task-0", "task-0", ...Array.from({ length: 25 }, (_, index) => `task-${index + 1}`)];
    const api = {
      getInSessionTaskDetails: vi.fn((sessionId, taskId) => {
        expect(sessionId).toBe("session-a");
        if (taskId === "task-1") return Promise.reject(new Error("unavailable"));
        if (taskId === "task-2") return Promise.resolve({ task: task("another-task") });
        return Promise.resolve({ task: task(taskId) });
      })
    };
    const decorations = harness(api, "session-a");

    const details = await decorations.loadRelatedInSessionTaskDetails("session-a", ids);

    expect(api.getInSessionTaskDetails).toHaveBeenCalledTimes(24);
    expect(api.getInSessionTaskDetails).not.toHaveBeenCalledWith("session-a", "task-24");
    expect(api.getInSessionTaskDetails).not.toHaveBeenCalledWith("session-a", "task-25");
    expect(details.map((item) => item.insession_task_id)).toEqual(
      ["task-0", ...Array.from({ length: 21 }, (_, index) => `task-${index + 3}`)]
    );
    expect(decorations.inSessionTaskDetails.value).toEqual(details);
  });

  it("clearing related task details invalidates an in-flight decoration read", async () => {
    const response = deferred();
    const api = {
      getInSessionTaskDetails: vi.fn(() => response.promise)
    };
    const decorations = harness(api, "session-a");

    const pending = decorations.loadRelatedInSessionTaskDetails("session-a", ["task-0"]);
    await Promise.resolve();
    decorations.clearInSessionTaskDetails();
    response.resolve({ task: task("task-0") });

    await expect(pending).resolves.toEqual([]);
    expect(decorations.inSessionTaskDetails.value).toEqual([]);
  });
});
