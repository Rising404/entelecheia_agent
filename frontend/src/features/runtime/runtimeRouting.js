export const RUNTIME_MODE_DIRECT = "direct";
export const RUNTIME_MODE_TURN = "turn";

// L0 remains a recognized policy projection; the UI uses the L0/L1 single-Turn mode.
export const RUNTIME_MODES = Object.freeze([RUNTIME_MODE_TURN]);

const POLICY_BY_MODE = Object.freeze({
  [RUNTIME_MODE_DIRECT]: Object.freeze({
    schema_version: 2,
    l1_enabled: false
  }),
  [RUNTIME_MODE_TURN]: Object.freeze({
    schema_version: 2,
    l1_enabled: true
  })
});

export function runtimePolicyForMode(mode) {
  const policy = POLICY_BY_MODE[mode];
  if (!policy) throw new TypeError(`Unknown Runtime mode: ${String(mode)}`);
  return { ...policy };
}

export function runtimeModeFromPolicy(value, fallback = RUNTIME_MODE_TURN) {
  if (!value || typeof value !== "object") return fallback;
  if (value.l1_enabled === true) return RUNTIME_MODE_TURN;
  if (value.l1_enabled === false) return RUNTIME_MODE_DIRECT;
  return fallback;
}

export function runtimeModeFromProjection(value, fallback = RUNTIME_MODE_TURN) {
  return runtimeModeFromPolicy(value?.policy ?? value?.default_policy, fallback);
}

export function runtimeModeAvailable(mode, projection) {
  // Direct remains a stored projection, not a selectable UI mode.
  if (mode === RUNTIME_MODE_DIRECT) return false;
  const levels = projection?.available_processing_levels;
  if (!Array.isArray(levels)) return false;
  if (mode === RUNTIME_MODE_TURN) return levels.includes("L1");
  return false;
}

export function safeRuntimeMode(projection, preferred = RUNTIME_MODE_TURN) {
  if (runtimeModeAvailable(preferred, projection)) return preferred;
  const projected = runtimeModeFromProjection(projection, RUNTIME_MODE_TURN);
  if (runtimeModeAvailable(projected, projection)) return projected;
  return RUNTIME_MODE_TURN;
}
