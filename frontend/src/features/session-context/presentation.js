export const SESSION_CONTEXT_GROUPS = [
  ["user_state", "用户"],
  ["task_state", "任务"],
  ["interaction_state", "交互"]
];

export function formatContextValue(value) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "-";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
