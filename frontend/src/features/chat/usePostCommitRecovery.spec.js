import { describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { flushPromises } from "@vue/test-utils";
import { usePostCommitRecovery } from "./usePostCommitRecovery";

function detail(id = "session-a") {
  return {
    session: { id, status: "active" },
    turns: [{ role: "assistant", content: "已保存的答复" }],
    turn_window: { turn_id: "turn-a", window_state: "post_commit_pending", window_revision: 5 },
    post_commit: {
      turn_id: "turn-a", window_revision: 5, failed_job_digest: "a".repeat(64),
      failed_jobs: [
        { job_id: "index-a", job_kind: "session_retrieval_index", reason_code: "INDEX_FAILURE" },
        { job_id: "summary-a", job_kind: "session_summary", reason_code: "SUMMARY_FAILURE" }
      ]
    }
  };
}

function setup(overrides = {}) {
  const state = {
    api: {
      controlTurnPostCommitJobs: vi.fn().mockResolvedValue({ replayed: false }),
      getSession: vi.fn().mockResolvedValue({
        ...detail(), post_commit: null,
        turn_window: { window_state: "empty", window_revision: 7 }
      })
    },
    selectedSessionId: ref("session-a"), sessionDetail: ref(detail()),
    saving: ref(false), chatBusy: ref(false), attachmentUploading: ref(false),
    notice: ref(""), clearMessage: vi.fn(), showError: vi.fn(),
    confirmWaiver: vi.fn().mockResolvedValue(true),
    createRequestId: () => "request-a",
    ...overrides
  };
  return { ...state, ...usePostCommitRecovery(state) };
}

function deferred() {
  let resolve;
  const promise = new Promise((accept) => { resolve = accept; });
  return { promise, resolve };
}

describe("usePostCommitRecovery", () => {
  it("retries exactly the displayed failed set then reads the authoritative state", async () => {
    const view = setup();
    expect(view.postCommitFailure.value.label).toBe("历史检索索引和会话摘要");
    await view.retryPostCommit();
    expect(view.api.controlTurnPostCommitJobs).toHaveBeenCalledWith("session-a", {
      turn_id: "turn-a", request_id: "request-a", action: "retry",
      expected_window_revision: 5, expected_failed_job_digest: "a".repeat(64),
      job_ids: ["index-a", "summary-a"]
    });
    expect(view.api.getSession).toHaveBeenCalledWith("session-a");
    expect(view.sessionDetail.value.turns[0].content).toBe("已保存的答复");
    expect(view.postCommitFailure.value).toBeNull();
    expect(view.postCommitRecoveryBusy.value).toBe(false);
    expect(view.confirmWaiver).not.toHaveBeenCalled();
  });

  it("requires separate confirmation before skipping and explains lost derived context", async () => {
    const view = setup();
    await view.waivePostCommit();
    expect(view.confirmWaiver).toHaveBeenCalledWith(expect.stringContaining("可能不完整"));
    expect(view.confirmWaiver).toHaveBeenCalledWith(expect.stringContaining("不会删除已保存的答复"));
    expect(view.api.controlTurnPostCommitJobs.mock.calls[0][1]).toMatchObject({
      action: "waive", confirm_stale: true
    });
  });

  it("does not mutate or fetch after the user cancels skipping", async () => {
    const view = setup({ confirmWaiver: vi.fn().mockResolvedValue(false) });
    await view.waivePostCommit();
    expect(view.api.controlTurnPostCommitJobs).not.toHaveBeenCalled();
    expect(view.api.getSession).not.toHaveBeenCalled();
    expect(view.postCommitRecoveryBusy.value).toBe(false);
  });

  it("does not reuse confirmation after the failed set or Session changes", async () => {
    for (const changeSession of [false, true]) {
      const confirmation = deferred();
      const view = setup({ confirmWaiver: () => confirmation.promise });
      const running = view.waivePostCommit();
      if (changeSession) {
        view.selectedSessionId.value = "session-b";
        view.sessionDetail.value = detail("session-b");
      } else {
        view.sessionDetail.value.post_commit.failed_job_digest = "b".repeat(64);
      }
      confirmation.resolve(true);
      await running;
      expect(view.api.controlTurnPostCommitJobs).not.toHaveBeenCalled();
    }
  });

  it("does not overwrite the new Session when the old request completes", async () => {
    const pending = deferred();
    const view = setup();
    view.api.controlTurnPostCommitJobs.mockReturnValue(pending.promise);
    const running = view.retryPostCommit();
    view.selectedSessionId.value = "session-b";
    view.sessionDetail.value = detail("session-b");
    pending.resolve({ replayed: false });
    await running;
    expect(view.sessionDetail.value.session.id).toBe("session-b");
    expect(view.notice.value).toBe("");
    expect(view.api.getSession).not.toHaveBeenCalled();
  });

  it("does not roll back a newer window with an older in-flight GET response", async () => {
    const response = deferred();
    const view = setup();
    view.api.getSession.mockReturnValue(response.promise);
    const running = view.retryPostCommit();
    await flushPromises();
    view.sessionDetail.value = {
      ...detail(), post_commit: null,
      turn_window: { window_state: "empty", window_revision: 7 }
    };
    response.resolve(detail());
    await running;
    expect(view.sessionDetail.value.turn_window.window_state).toBe("empty");
  });

  it("refreshes after conflict without automatically replaying the command", async () => {
    const view = setup();
    const error = Object.assign(new Error("状态已经变化"), { status: 409 });
    view.api.controlTurnPostCommitJobs.mockRejectedValue(error);
    await view.retryPostCommit();
    expect(view.api.controlTurnPostCommitJobs).toHaveBeenCalledTimes(1);
    expect(view.api.getSession).toHaveBeenCalledOnce();
    expect(view.showError).toHaveBeenCalledWith(error);
  });

  it("blocks double submission while the first control is pending", async () => {
    const pending = deferred();
    const view = setup();
    view.api.controlTurnPostCommitJobs.mockReturnValue(pending.promise);
    const running = view.retryPostCommit();
    await view.retryPostCommit();
    await view.waivePostCommit();
    expect(view.api.controlTurnPostCommitJobs).toHaveBeenCalledTimes(1);
    expect(view.confirmWaiver).not.toHaveBeenCalled();
    pending.resolve({ replayed: false });
    await running;
  });

  it.each(["archived", "mismatched-session", "mismatched-turn", "pending", "bad-digest", "old-window", "busy"])(
    "fails closed for %s", async (condition) => {
      const view = setup();
      if (condition === "archived") view.sessionDetail.value.session.status = "archived";
      if (condition === "mismatched-session") view.selectedSessionId.value = "session-b";
      if (condition === "mismatched-turn") view.sessionDetail.value.turn_window.turn_id = "turn-b";
      if (condition === "pending") view.sessionDetail.value.post_commit.failed_jobs = [];
      if (condition === "bad-digest") view.sessionDetail.value.post_commit.failed_job_digest = null;
      if (condition === "old-window") view.sessionDetail.value.turn_window.window_revision = 6;
      if (condition === "busy") view.chatBusy.value = true;
      expect(view.canRecoverPostCommit.value).toBe(false);
      await view.retryPostCommit();
      await view.waivePostCommit();
      await flushPromises();
      expect(view.api.controlTurnPostCommitJobs).not.toHaveBeenCalled();
    }
  );
});
