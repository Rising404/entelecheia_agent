import { describe, expect, it } from "vitest";

import { useChatDrafts } from "./useChatDrafts";


function createStorage() {
  const values = new Map();
  return {
    get length() { return values.size; },
    getItem(key) { return values.get(key) || null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
    key(index) { return [...values.keys()][index] || null; }
  };
}


describe("useChatDrafts", () => {
  it("persists, enumerates, and clears per-session drafts", () => {
    const storage = createStorage();
    const drafts = useChatDrafts({ storage, prefix: "draft." });

    drafts.save("s1", "hello");
    drafts.save("s2", "world");

    expect(drafts.load("s1")).toBe("hello");
    expect([...drafts.draftSessionIds.value]).toEqual(["s1", "s2"]);
    expect(drafts.has("s2")).toBe(true);

    drafts.clear("s1");
    expect(drafts.load("s1")).toBe("");
    expect(drafts.has("s1")).toBe(false);
  });

  it("degrades when storage is unavailable", () => {
    const drafts = useChatDrafts({ storage: null });
    drafts.save("s1", "hello");
    drafts.refreshIds();
    expect(drafts.load("s1")).toBe("");
    expect([...drafts.draftSessionIds.value]).toEqual([]);
  });
});
