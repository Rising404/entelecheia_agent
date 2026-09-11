import { describe, expect, it } from "vitest";

import {
  formatContextValue
} from "./presentation";

describe("session context presentation", () => {
  it("formats structured values without losing fields", () => {
    expect(formatContextValue({ depth: "detailed" })).toContain('"depth": "detailed"');
    expect(formatContextValue("brief")).toBe("brief");
  });
});
