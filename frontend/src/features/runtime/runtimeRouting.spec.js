import { describe, expect, it } from "vitest";

import {
  RUNTIME_MODE_DIRECT,
  RUNTIME_MODE_TURN,
  runtimeModeAvailable,
  runtimeModeFromProjection,
  runtimeModeFromPolicy,
  runtimePolicyForMode,
  safeRuntimeMode
} from "./runtimeRouting";

describe("runtime routing contract", () => {
  it("preserves L0/L1 policies", () => {
    expect(runtimePolicyForMode(RUNTIME_MODE_DIRECT)).toEqual({
      schema_version: 2, l1_enabled: false
    });
    expect(runtimePolicyForMode(RUNTIME_MODE_TURN)).toEqual({
      schema_version: 2, l1_enabled: true
    });
  });


  it("uses the published capability gate when choosing a safe mode", () => {
    const l1Unavailable = {
      available_processing_levels: ["L0"],
      default_policy: { l1_enabled: true }
    };
    expect(runtimeModeAvailable(RUNTIME_MODE_TURN, l1Unavailable)).toBe(false);
    expect(runtimeModeFromProjection(l1Unavailable)).toBe(RUNTIME_MODE_TURN);
    expect(safeRuntimeMode(l1Unavailable, RUNTIME_MODE_TURN)).toBe(RUNTIME_MODE_TURN);
  });
});
