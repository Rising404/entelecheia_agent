import { describe, expect, it, vi } from "vitest";

import { mergeRuntimeEvents, useRuntimeEvents } from "./useRuntimeEvents";

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
    attempt_id: null,
    operation_id: null,
    prompt_replay: false,
    ...overrides
  };
}

describe("useRuntimeEvents", () => {
  it("deduplicates by event id, preserves the durable sequence, and bounds the cache", () => {
    const stored = runtimeEvent("evt-1", 1);
    const malformedDuplicate = runtimeEvent("evt-1", 99, { stage: "RESPONSE" });
    const merged = mergeRuntimeEvents(
      [stored],
      [malformedDuplicate, runtimeEvent("evt-2", 2), runtimeEvent("evt-3", 3)],
      2
    );

    expect(merged.map((event) => event.event_id)).toEqual(["evt-2", "evt-3"]);
    expect(mergeRuntimeEvents([stored], [malformedDuplicate])[0].sequence).toBe(1);
  });

  it("loads the public lifecycle projection without legacy visibility or tool data", async () => {
    const api = {
      getRuntimeEvents: vi.fn().mockResolvedValue({
        enabled: true,
        events: [runtimeEvent("evt-1", 1)]
      })
    };
    const timeline = useRuntimeEvents({ api });

    await timeline.load("s1");

    expect(api.getRuntimeEvents).toHaveBeenCalledWith("s1", { limit: 100 });
    expect(timeline.enabled.value).toBe(true);
    expect(timeline.events.value.map((event) => event.event_id)).toEqual(["evt-1"]);
  });

  it("keeps an unknown stage and status so the generic UI can render it", () => {
    const timeline = useRuntimeEvents({ api: {} });

    timeline.ingest(runtimeEvent("evt-future", 7, {
      stage: "FUTURE_STAGE",
      status: "future_status"
    }));

    expect(timeline.enabled.value).toBe(true);
    expect(timeline.events.value).toHaveLength(1);
    expect(timeline.events.value[0]).toMatchObject({
      stage: "FUTURE_STAGE",
      status: "future_status"
    });
  });

  it("rejects the retired public event shape instead of reading kind, data, or visibility", () => {
    const legacy = {
      schema_version: 1,
      event_id: "evt-legacy",
      sequence: 1,
      session_id: "s1",
      kind: "tool.running",
      status: "running",
      summary: "执行中",
      occurred_at: "2026-07-15T00:00:00Z",
      prompt_replay: false,
      visibility: "user",
      data: { tool_id: "file_read" }
    };

    expect(mergeRuntimeEvents([], [legacy])).toEqual([]);
  });

  it("does not let a stale session load overwrite a newer selection", async () => {
    let resolveFirst;
    const first = new Promise((resolve) => { resolveFirst = resolve; });
    const api = {
      getRuntimeEvents: vi.fn()
        .mockReturnValueOnce(first)
        .mockResolvedValueOnce({ enabled: true, events: [runtimeEvent("evt-new", 2, { session_id: "new-session" })] })
    };
    const timeline = useRuntimeEvents({ api });

    const staleLoad = timeline.load("old-session");
    await timeline.load("new-session");
    resolveFirst({ enabled: true, events: [runtimeEvent("evt-old", 1, { session_id: "old-session" })] });
    await staleLoad;

    expect(timeline.events.value.map((event) => event.event_id)).toEqual(["evt-new"]);
  });

  it("catches up from the latest durable cursor without polling retired active-run state", async () => {
    const api = {
      getRuntimeEvents: vi.fn()
        .mockResolvedValueOnce({ enabled: true, events: [runtimeEvent("evt-1", 1)] })
        .mockResolvedValueOnce({
          enabled: true,
          events: [runtimeEvent("evt-2", 2)],
          next_after: "evt-2",
          has_more: false
        }),
      getActiveRun: vi.fn()
    };
    const timeline = useRuntimeEvents({ api });

    await timeline.load("s1");
    await timeline.catchUp("s1");

    expect(api.getRuntimeEvents.mock.calls).toEqual([
      ["s1", { limit: 100 }],
      ["s1", { after: "evt-1", limit: 100 }]
    ]);
    expect(api.getActiveRun).not.toHaveBeenCalled();
    expect(timeline.events.value.map((event) => event.event_id)).toEqual(["evt-1", "evt-2"]);
  });

  it("re-anchors once at the bounded tail when retention removed the cursor", async () => {
    const cursorError = {
      status: 409,
      payload: { error: { code: "RUNTIME_EVENT_CURSOR_NOT_FOUND" } }
    };
    const api = {
      getRuntimeEvents: vi.fn()
        .mockResolvedValueOnce({ enabled: true, events: [runtimeEvent("evt-old", 1)] })
        .mockRejectedValueOnce(cursorError)
        .mockResolvedValueOnce({ enabled: true, events: [runtimeEvent("evt-new", 9)] })
    };
    const timeline = useRuntimeEvents({ api });

    await timeline.load("s1");
    await timeline.catchUp("s1");

    expect(api.getRuntimeEvents.mock.calls).toEqual([
      ["s1", { limit: 100 }],
      ["s1", { after: "evt-old", limit: 100 }],
      ["s1", { limit: 100 }]
    ]);
    expect(timeline.events.value.map((event) => event.event_id)).toEqual(["evt-old", "evt-new"]);
  });

  it("cancels a stale catch-up page after a session switch", async () => {
    let resolveCatchUp;
    const catchUpPage = new Promise((resolve) => { resolveCatchUp = resolve; });
    const api = {
      getRuntimeEvents: vi.fn()
        .mockResolvedValueOnce({ enabled: true, events: [runtimeEvent("evt-1", 1)] })
        .mockReturnValueOnce(catchUpPage)
        .mockResolvedValueOnce({
          enabled: true,
          events: [runtimeEvent("evt-s2", 1, { session_id: "s2" })]
        })
    };
    const timeline = useRuntimeEvents({ api });

    await timeline.load("s1");
    const staleCatchUp = timeline.catchUp("s1");
    await timeline.load("s2");
    resolveCatchUp({ enabled: true, events: [runtimeEvent("evt-stale", 2)] });
    await staleCatchUp;

    expect(timeline.events.value.map((event) => event.event_id)).toEqual(["evt-s2"]);
  });

  it("treats a missing endpoint as a safe disabled fallback", async () => {
    const onError = vi.fn();
    const timeline = useRuntimeEvents({
      api: { getRuntimeEvents: vi.fn().mockRejectedValue({ status: 404 }) },
      onError
    });

    await timeline.load("s1");

    expect(timeline.events.value).toEqual([]);
    expect(timeline.enabled.value).toBe(false);
    expect(onError).not.toHaveBeenCalled();
  });
});
