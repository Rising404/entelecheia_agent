import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";

import { useSessionContextFeature } from "./useSessionContextFeature";

const empty = {
  session_id: "s1",
  user_state: [],
  task_state: [],
  interaction_state: [],
  counts: { total: 0 }
};

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

function harness(overrides = {}) {
  const api = {
    getSessionContext: vi.fn().mockResolvedValue({ session_context: empty }),
    explainSessionContext: vi.fn(),
    exportSessionContext: vi.fn(),
    clearSessionContext: vi.fn(),
    createSessionContextCorrection: vi.fn(),
    previewSessionContextRepair: vi.fn(),
    applySessionContextRepair: vi.fn(),
    ...overrides
  };
  const selectedSessionId = ref("s1");
  const notice = ref("");
  const download = vi.fn();
  const makeRequestId = vi.fn()
    .mockReturnValueOnce("request-1")
    .mockReturnValueOnce("request-2");
  const feature = useSessionContextFeature({
    api, selectedSessionId, notice, download, makeRequestId
  });
  return { api, selectedSessionId, notice, download, makeRequestId, feature };
}

describe("useSessionContextFeature", () => {
  it("loads only the selected session's transient context", async () => {
    const state = { ...empty, counts: { total: 1 }, user_state: [{ id: "state-1" }] };
    const { api, feature } = harness({
      getSessionContext: vi.fn().mockResolvedValue({ session_context: state })
    });
    await settle();
    expect(api.getSessionContext).toHaveBeenCalledWith("s1");
    expect(feature.sessionContext.value.counts.total).toBe(1);
    expect(feature).not.toHaveProperty("promotions");
  });

  it("keeps evidence excerpts opt-in and downloads the versioned export", async () => {
    const explanation = { explanation: { evidence: [{ id: "turn:s1:0" }] } };
    const exported = { schema_version: 2, session_id: "s1" };
    const { api, download, feature } = harness({
      explainSessionContext: vi.fn().mockResolvedValue(explanation),
      exportSessionContext: vi.fn().mockResolvedValue({ session_context_export: exported })
    });
    await settle();
    const state = { id: "state-1", domain: "user", state_type: "temporary_preference", key: "depth" };
    await feature.explainState(state);
    await feature.exportSessionContext({ includeExcerpt: true });
    expect(api.explainSessionContext).toHaveBeenCalledWith("s1", {
      domain: "user", state_type: "temporary_preference", key: "depth",
      include_evidence_excerpt: false
    });
    expect(api.exportSessionContext).toHaveBeenCalledWith("s1", {
      include_evidence_excerpt: true
    });
    expect(download).toHaveBeenCalledWith("session-context-s1.json", exported);
  });

  it("reuses a clear idempotency key after failure and rotates it when reason changes", async () => {
    const clearSessionContext = vi.fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error("offline again"));
    const { api, makeRequestId, feature } = harness({ clearSessionContext });
    await settle();

    expect(await feature.clearSessionContext("start fresh")).toBe(false);
    expect(await feature.clearSessionContext("start fresh")).toBe(true);
    expect(clearSessionContext.mock.calls[0][1].request_id).toBe("request-1");
    expect(clearSessionContext.mock.calls[1][1].request_id).toBe("request-1");
    expect(await feature.clearSessionContext("new boundary")).toBe(false);
    expect(clearSessionContext.mock.calls[2][1].request_id).toBe("request-2");
    expect(makeRequestId).toHaveBeenCalledTimes(2);
    expect(api.getSessionContext).toHaveBeenCalled();
  });

  it("records correction evidence without changing view and retains the backend preview", async () => {
    const preview = {
      preview_token: "repair-preview:abc",
      slot: { domain: "user", state_type: "temporary_preference", key: "depth" },
      changes: { changed: [] }
    };
    const { api, feature, notice } = harness({
      createSessionContextCorrection: vi.fn().mockResolvedValue({
        correction_evidence: { id: "event-1" },
        repair: preview,
        apply_enabled: false
      })
    });
    await settle();
    const state = {
      id: "state-1", domain: "user", state_type: "temporary_preference", key: "depth"
    };

    expect(await feature.createCorrection({ state, operation: "set", value: "brief" })).toBe(true);

    expect(api.createSessionContextCorrection).toHaveBeenCalledWith("s1", {
      confirm: true,
      domain: "user",
      state_type: "temporary_preference",
      key: "depth",
      operation: "set",
      value: "brief"
    });
    expect(feature.repairPreview.value).toEqual(preview);
    expect(feature.repairApplyEnabled.value).toBe(false);
    expect(notice.value).toContain("尚未替换");
  });

  it("applies only a backend-enabled preview with literal confirmation and token", async () => {
    const preview = {
      preview_token: "repair-preview:abc",
      slot: { domain: "user", state_type: "temporary_preference", key: "depth" },
      changes: { changed: [] }
    };
    const { api, feature } = harness({
      previewSessionContextRepair: vi.fn().mockResolvedValue({ repair: preview, apply_enabled: true }),
      applySessionContextRepair: vi.fn().mockResolvedValue({ repair: { applied: true } })
    });
    await settle();
    await feature.previewRepair({ domain: "user", state_type: "temporary_preference", key: "depth" });
    expect(await feature.applyRepair()).toBe(true);
    expect(api.applySessionContextRepair).toHaveBeenCalledWith("s1", {
      confirm: true,
      preview_token: "repair-preview:abc",
      domain: "user",
      state_type: "temporary_preference",
      key: "depth"
    });
  });

  it("marks a backend stale-preview conflict without discarding the diff", async () => {
    const preview = { preview_token: "repair-preview:old", slot: null, changes: {} };
    const stale = {
      message: "preview stale",
      error: { code: "SESSION_CONTEXT_REPAIR_PREVIEW_STALE", details: {} }
    };
    const { feature } = harness({
      previewSessionContextRepair: vi.fn().mockResolvedValue({ repair: preview, apply_enabled: true }),
      applySessionContextRepair: vi.fn().mockRejectedValue(stale)
    });
    await settle();
    await feature.previewRepair();
    expect(await feature.applyRepair()).toBe(false);
    expect(feature.repairPreview.value).toEqual(preview);
    expect(feature.repairPreviewStale.value).toBe(true);
  });
});
