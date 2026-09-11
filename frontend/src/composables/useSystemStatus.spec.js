import { describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { useSystemStatus } from "./useSystemStatus";

describe("useSystemStatus", () => {
  it("keeps API reachability separate from component health", async () => {
    vi.spyOn(api, "status").mockResolvedValueOnce({
      ok: false,
      model_configured: false,
      components: { api: { ok: true }, gateway: { ok: false } }
    });
    const status = useSystemStatus();

    await status.loadStatus();

    expect(status.statusLoaded.value).toBe(true);
    expect(status.apiReachable.value).toBe(true);
    expect(status.apiOk.value).toBe(false);
  });

  it("marks a transport failure as unreachable", async () => {
    vi.spyOn(api, "status").mockRejectedValueOnce(new TypeError("offline"));
    const status = useSystemStatus();

    await status.loadStatus();

    expect(status.statusLoaded.value).toBe(true);
    expect(status.apiReachable.value).toBe(false);
    expect(status.systemStatus.value.components.api.detail).toBe("offline");
  });
});
