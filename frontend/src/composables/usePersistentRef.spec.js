import { beforeEach, describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { usePersistentRef } from "./usePersistentRef";

describe("usePersistentRef", () => {
  beforeEach(() => localStorage.clear());

  it("starts from the default when nothing was ever chosen", () => {
    expect(usePersistentRef("panel", true).value).toBe(true);
  });

  it("remembers a choice across a fresh read", async () => {
    const first = usePersistentRef("panel", true);
    first.value = false;
    await nextTick();
    expect(usePersistentRef("panel", true).value).toBe(false);
  });

  it("falls back rather than failing on a corrupted value", () => {
    localStorage.setItem("personagraph.ui.panel", "{not json");
    expect(usePersistentRef("panel", true).value).toBe(true);
  });

  it("keeps a stored false, which is the whole point of hiding something", () => {
    localStorage.setItem("personagraph.ui.panel", "false");
    expect(usePersistentRef("panel", true).value).toBe(false);
  });

  it("does not let unavailable storage break the ref", async () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      get() { throw new Error("blocked"); }
    });
    try {
      const state = usePersistentRef("panel", true);
      state.value = false;
      await nextTick();
      expect(state.value).toBe(false);
    } finally {
      if (original) Object.defineProperty(globalThis, "localStorage", original);
    }
  });
});
