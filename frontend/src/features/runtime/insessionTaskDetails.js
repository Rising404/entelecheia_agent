const TASK_STATUSES = new Set([
  "proposed",
  "active",
  "awaiting_user",
  "waiting_external",
  "interrupted",
  "blocked",
  "cancelled",
  "completed"
]);
const NODE_KINDS = new Set(["root", "subtask"]);
const TASK_KEYS = [
  "insession_task_id",
  "title",
  "status",
  "current_graph_revision",
  "nodes",
  "related_turn_count"
];
const NODE_KEYS = [
  "insession_task_node_id",
  "parent_insession_task_node_id",
  "node_kind",
  "ordinal",
  "title",
  "status"
];

function isExactRecord(value, keys) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

function isNonemptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function isTaskNode(value) {
  return isExactRecord(value, NODE_KEYS) &&
    isNonemptyString(value.insession_task_node_id) &&
    (value.parent_insession_task_node_id === null || isNonemptyString(value.parent_insession_task_node_id)) &&
    NODE_KINDS.has(value.node_kind) &&
    Number.isInteger(value.ordinal) && value.ordinal >= 0 &&
    isNonemptyString(value.title) && TASK_STATUSES.has(value.status);
}

/**
 * 在精简的公开 Task 投影进入前端状态前进行校验。会话边界已经由请求 URL 表达，
 * 因此该投影有意不携带会话 ID。
 */
export function isInSessionTaskDetail(value) {
  if (!isExactRecord(value, TASK_KEYS)) {
    return false;
  }
  if (!isNonemptyString(value.insession_task_id) ||
      !isNonemptyString(value.title) ||
      !TASK_STATUSES.has(value.status) ||
      !Array.isArray(value.nodes) ||
      !value.nodes.every(isTaskNode) ||
      !Number.isInteger(value.related_turn_count) ||
      value.related_turn_count < 0) {
    return false;
  }
  const isShell = value.current_graph_revision === null && value.nodes.length === 0;
  const hasGraph = Number.isInteger(value.current_graph_revision) &&
    value.current_graph_revision >= 1 && value.nodes.length > 0;
  return isShell || hasGraph;
}

/** 从端点固定的响应信封中提取一条安全的公开任务。 */
export function taskDetailFromResponse(payload) {
  return isInSessionTaskDetail(payload?.task) ? payload.task : null;
}

export function inSessionTaskStatusText(status) {
  return {
    proposed: "已提议",
    active: "处理中",
    awaiting_user: "等待你的信息",
    waiting_external: "等待外部状态",
    interrupted: "已中断",
    blocked: "已阻塞",
    cancelled: "已取消",
    completed: "已完成"
  }[status] || "状态不可用";
}
