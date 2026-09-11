export const RUNTIME_MODE_DIRECT = "direct";
export const RUNTIME_MODE_TURN = "turn";
export const RUNTIME_MODE_TASK = "task";

// 用户可选的模式只有两个。`direct` 保留为一个可被识别的**投影值**——后端仍然
// 能表达 l1/l2 全关，历史会话里也可能存着它——但它不再出现在选择器里：
// "要不要直接回答"本来就该由模型在一轮之内决定，而不是由用户提前锁死。
export const RUNTIME_MODES = Object.freeze([
  RUNTIME_MODE_TURN,
  RUNTIME_MODE_TASK
]);

const POLICY_BY_MODE = Object.freeze({
  [RUNTIME_MODE_DIRECT]: Object.freeze({
    schema_version: 1,
    l1_enabled: false,
    l2_enabled: false
  }),
  [RUNTIME_MODE_TURN]: Object.freeze({
    schema_version: 1,
    l1_enabled: true,
    l2_enabled: false
  }),
  [RUNTIME_MODE_TASK]: Object.freeze({
    schema_version: 1,
    l1_enabled: false,
    l2_enabled: true
  })
});

export function runtimePolicyForMode(mode) {
  const policy = POLICY_BY_MODE[mode];
  if (!policy) throw new TypeError(`Unknown Runtime mode: ${String(mode)}`);
  return { ...policy };
}

export function runtimeModeFromPolicy(value, fallback = RUNTIME_MODE_TURN) {
  if (!value || typeof value !== "object") return fallback;
  if (value.l1_enabled === true && value.l2_enabled === false) return RUNTIME_MODE_TURN;
  if (value.l1_enabled === false && value.l2_enabled === true) return RUNTIME_MODE_TASK;
  if (value.l1_enabled === false && value.l2_enabled === false) return RUNTIME_MODE_DIRECT;
  return fallback;
}

export function runtimeModeFromProjection(value, fallback = RUNTIME_MODE_TURN) {
  return runtimeModeFromPolicy(value?.policy ?? value?.default_policy, fallback);
}

export function runtimeModeAvailable(mode, projection) {
  // 不再可选。历史会话若存着它，会被 safeRuntimeMode 归到"任务"。
  if (mode === RUNTIME_MODE_DIRECT) return false;
  const levels = projection?.available_processing_levels;
  if (!Array.isArray(levels)) return false;
  if (mode === RUNTIME_MODE_TURN) return levels.includes("L1");
  if (mode === RUNTIME_MODE_TASK) return levels.includes("L2");
  return false;
}

export function safeRuntimeMode(projection, preferred = RUNTIME_MODE_TURN) {
  if (runtimeModeAvailable(preferred, projection)) return preferred;
  const projected = runtimeModeFromProjection(projection, RUNTIME_MODE_TURN);
  if (runtimeModeAvailable(projected, projection)) return projected;
  // 兜底是"任务"而不是"直接"：L2 是后端始终提供的持久通道，而"直接"已经不是
  // 一个可选项，落到它身上会让选择器显示不出当前状态。
  return RUNTIME_MODE_TASK;
}
