import { describe, expect, it, vi } from "vitest";

import { fetchWithReadRetry } from "./apiTransport";


describe("fetchWithReadRetry", () => {
  it("retries one idempotent read after a browser network failure", async () => {
    const response = { ok: true };
    const fetchImpl = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(response);
    const delay = vi.fn().mockResolvedValue(undefined);

    await expect(fetchWithReadRetry("/api/sessions", {}, { fetchImpl, delay }))
      .resolves.toBe(response);

    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(delay).toHaveBeenCalledWith(120);
  });

  it("does not retry mutations, explicit aborts or non-network failures", async () => {
    const cases = [
      { options: { method: "POST" }, error: new TypeError("Failed to fetch") },
      { options: { signal: { aborted: true } }, error: new TypeError("Failed to fetch") },
      { options: {}, error: new Error("application failure") }
    ];

    for (const item of cases) {
      const fetchImpl = vi.fn().mockRejectedValue(item.error);
      await expect(fetchWithReadRetry("/api/test", item.options, {
        fetchImpl,
        delay: vi.fn()
      })).rejects.toBe(item.error);
      expect(fetchImpl).toHaveBeenCalledTimes(1);
    }
  });

  it("surfaces a second read failure without looping", async () => {
    const second = new TypeError("still offline");
    const fetchImpl = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(second);

    await expect(fetchWithReadRetry("/api/status", {}, {
      fetchImpl,
      delay: vi.fn().mockResolvedValue(undefined)
    })).rejects.toBe(second);

    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
