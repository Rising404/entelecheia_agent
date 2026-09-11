import { describe, expect, it } from "vitest";

import {
  RUNTIME_MODE_DIRECT,
  RUNTIME_MODE_TASK,
  RUNTIME_MODE_TURN,
  runtimeModeAvailable,
  runtimeModeFromProjection,
  runtimeModeFromPolicy,
  runtimePolicyForMode,
  safeRuntimeMode
} from "./runtimeRouting";

describe("runtime routing contract", () => {
  it("maps the three UI modes to mutually exclusive backend policies", () => {
    expect(runtimePolicyForMode(RUNTIME_MODE_DIRECT)).toEqual({
      schema_version: 1, l1_enabled: false, l2_enabled: false
    });
    expect(runtimePolicyForMode(RUNTIME_MODE_TURN)).toEqual({
      schema_version: 1, l1_enabled: true, l2_enabled: false
    });
    expect(runtimePolicyForMode(RUNTIME_MODE_TASK)).toEqual({
      schema_version: 1, l1_enabled: false, l2_enabled: true
    });
  });

  it("refuses to interpret an invalid double-enabled policy", () => {
    expect(runtimeModeFromPolicy({ l1_enabled: true, l2_enabled: true }, "invalid"))
      .toBe("invalid");
  });

  it("uses the published capability gate when choosing a safe mode", () => {
    const l1Unavailable = {
      available_processing_levels: ["L0", "L2"],
      default_policy: { l1_enabled: true, l2_enabled: false }
    };
    expect(runtimeModeAvailable(RUNTIME_MODE_TURN, l1Unavailable)).toBe(false);
    expect(runtimeModeFromProjection(l1Unavailable)).toBe(RUNTIME_MODE_TURN);
    expect(safeRuntimeMode(l1Unavailable, RUNTIME_MODE_TURN)).toBe(RUNTIME_MODE_TASK);
  });
});
