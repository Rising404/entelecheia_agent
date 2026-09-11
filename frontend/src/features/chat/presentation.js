import { AGENT_NAME } from "../../shared/identity";

const REVIEW_PAYLOAD_KEYS = [
  "name",
  "content",
  "text",
  "path",
  "dest",
  "task_id",
  "due_at",
  "deadline",
  "category",
  "source"
];

const REVIEW_PAYLOAD_LABELS = {
  name: "任务名称",
  content: "内容",
  text: "文本",
  path: "路径",
  dest: "目标",
  task_id: "任务",
  due_at: "时间",
  deadline: "截止",
  category: "分类",
  source: "来源"
};

const REVIEW_TOOL_LABELS = {
  file_read: "读取文件",
  file_list: "查看目录",
  file_search: "搜索文件",
  file_write: "写入文件",
  file_download: "下载文件",
  web_search: "网络搜索",
  web_fetch: "网页读取",
  web_scrape: "网页抓取"
};

const REVIEW_FILE_PREVIEW_LINES = 40;

export function roleText(role) {
  return { user: "你", assistant: AGENT_NAME, system: "系统", tool: "工具" }[role] || role;
}

export function postCommitFailureLabel(jobs = [], errorCodes = []) {
  const kinds = jobs.map((job) => job?.job_kind);
  // 旧的 Turn 投影只有安全原因码；不能将所有后台失败都误称为摘要失败。
  const labels = kinds.length
    ? kinds.map((kind) => ({
      session_retrieval_index: "历史检索索引", session_summary: "会话摘要"
    }[kind] || "后台整理任务"))
    : errorCodes.map((code) => {
      if (code.startsWith("SESSION_RETRIEVAL_")) return "历史检索索引";
      if (code.startsWith("SUMMARY_")) return "会话摘要";
      return "后台整理任务";
    });
  return [...new Set(labels)].join("和") || "后台整理任务";
}

export function streamEventText(event) {
  return { accepted: "已接收", running: "运行中", final: "已完成", error: "错误", fallback: "兼容模式" }[event] || event;
}

export function streamEventDetail(item) {
  if (item.event === "running") return item.data?.stage || "run_turn";
  if (item.event === "final") return item.data?.result?.needs_review ? "needs_review" : "final";
  if (item.event === "error") {
    const code = item.data?.error?.code || "error";
    const retry = outcomeRetryText(item.data?.outcome);
    return retry ? `${code} · ${retry}` : code;
  }
  return item.data?.ok ? "ok" : "";
}

export function outcomeWarnings(outcome) {
  return Array.isArray(outcome?.warnings) ? outcome.warnings : [];
}

export function outcomeStatusText(status) {
  return {
    completed: "已完成",
    partial: "部分完成",
    needs_review: "等待审批",
    failed: "运行失败",
    cancelled: "已取消"
  }[status] || status || "未知状态";
}

export function outcomeRetryText(outcome) {
  return outcome?.retry?.label || outcome?.retry?.action || "";
}

export function outcomeIssueText(issue) {
  const code = issue?.code;
  if (code === "MODEL_CALL_TIMEOUT") {
    const timeoutSeconds = boundedPositiveInteger(issue?.details?.timeout_s);
    return timeoutSeconds
      ? `模型等待超过 ${timeoutSeconds} 秒，本轮已停止；确认网络或服务状态后可重试。`
      : "模型等待超时，本轮已停止；确认网络或服务状态后可重试。";
  }
  if (code === "CONTEXT_BUDGET_EXCEEDED") {
    return "当前输入与必要上下文超过安全预算，模型尚未被调用；请缩短输入或拆分任务后再发送。";
  }
  if (code === "TOOL_LOOP_LIMIT_REACHED") {
    const maxRounds = boundedPositiveInteger(issue?.details?.max_rounds);
    return maxRounds
      ? `工具在 ${maxRounds} 轮调用后仍未收敛，本轮已安全停止；不会自动重放。`
      : "工具调用达到安全上限，本轮已安全停止；不会自动重放。";
  }
  if (code === "TOOL_LOOP_NO_PROGRESS") {
    return "工具调用没有产生新的进展，本轮已安全停止；不会自动重放。";
  }
  if (code === "TOOL_LOOP_TOKEN_BUDGET_REACHED") {
    return "工具调用达到本轮预算边界，已安全停止；不会自动重放。";
  }
  return issue?.message || issue?.code || "运行提示";
}

function boundedPositiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 && number <= 100 ? number : null;
}

export function outcomeIssueCode(issue) {
  return [issue?.domain, issue?.code].filter(Boolean).join(" / ");
}

export function outcomePrimaryIssue(outcome) {
  return outcome?.error || outcomeWarnings(outcome)[0] || null;
}

export function outcomePartialText(outcome) {
  const partial = outcome?.partial || {};
  if (partial.long_output?.truncated) {
    return `已生成 ${partial.long_output.segment_count || "-"} 段，输出可能尚未完整。`;
  }
  if (partial.pending_review?.pending_write_count) {
    return `有 ${partial.pending_review.pending_write_count} 项写入等待确认。`;
  }
  if (partial.failed_tools?.length) return `${partial.failed_tools.length} 个工具未成功执行。`;
  if (partial.review?.approved === false) return "用户拒绝审批，本轮写入未提交。";
  return "";
}

export function outcomeCardClass(outcome) {
  if (outcome?.status === "failed" || outcome?.status === "cancelled") {
    return "border-danger-line bg-danger-bg";
  }
  if (outcome?.status === "partial" || outcomeWarnings(outcome).length) {
    return "border-warn-line bg-warn-bg";
  }
  return "border-line bg-sunken";
}

export function reviewItems(review) {
  const writes = review?.pending_writes || review?.writes || review?.items || [];
  return Array.isArray(writes) ? writes : [];
}

export function reviewItemTitle(item, index) {
  const id = item?.tool_id || item?.type || "";
  return REVIEW_TOOL_LABELS[id] || id || `写入 ${index + 1}`;
}

export function reviewRiskText(value) {
  return { low: "低风险", medium: "中风险", high: "高风险", blocked: "已阻断" }[value] || value || "未知风险";
}

export function reviewPolicyText(item) {
  const reason = item?.policy?.reason;
  return {
    confirmation_required: "需要确认",
    sensitive_external_input_confirmation_required: "敏感外发需确认",
    allowed: "已允许",
    blocked_risk: "风险阻断"
  }[reason] || reason || "等待决定";
}

export function reviewSideEffectText(item) {
  if (item?.side_effect === true) return "会修改状态";
  if (item?.side_effect === false) return "只读工具";
  return item?.requires_confirmation ? "需确认" : "未声明副作用";
}

export function reviewPayloadFields(item) {
  const payload = item?.payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  return Object.entries(payload)
    .filter(([key, value]) => !key.startsWith("_") && value !== undefined && value !== null && value !== "")
    .sort(([left], [right]) => payloadKeyRank(left) - payloadKeyRank(right))
    .slice(0, 6)
    .map(([key, value]) => ({
      key,
      label: REVIEW_PAYLOAD_LABELS[key] || key,
      value: summarizeReviewValue(value)
    }));
}

function payloadKeyRank(key) {
  const index = REVIEW_PAYLOAD_KEYS.indexOf(key);
  return index === -1 ? REVIEW_PAYLOAD_KEYS.length : index;
}

function summarizeReviewValue(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (!text) return "";
  return text.length > 180 ? `${text.slice(0, 180)}...` : text;
}

export function fileWritePreview(item) {
  if (item?.tool_id !== "file_write" || !item?.payload) return null;
  const content = String(item.payload.content || "");
  const lines = content ? content.split(/\r?\n/) : [];
  const mode = String(item.payload.mode || "overwrite");
  return {
    path: String(item.payload.path || "未指定路径"),
    mode,
    modeText: mode === "append" ? "追加到文件末尾" : "覆盖或新建文件",
    lineCount: lines.length,
    charCount: content.length,
    lines: lines.slice(0, REVIEW_FILE_PREVIEW_LINES),
    omitted: Math.max(0, lines.length - REVIEW_FILE_PREVIEW_LINES)
  };
}

export function committedWrites(result) {
  const writes = result?.committed_writes || [];
  return Array.isArray(writes) ? writes : [];
}

export function committedWriteStatusText(write) {
  return write?.ok ? "已提交" : "未提交";
}

export function committedWriteMessage(write) {
  if (!write) return "";
  if (write.ok) return summarizeCommittedResult(write.result);
  const error = write.error || {};
  return error.message || error.code || "提交失败";
}

export function summarizeCommittedResult(result) {
  if (!result || typeof result !== "object") return "操作已完成";
  if (result.path) return `${result.path}${result.mode ? ` · ${result.mode}` : ""}`;
  if (result.reason) return result.reason;
  return "操作已完成";
}

export function resumeDecisionText(approved) {
  if (approved === true) return "已批准";
  if (approved === false) return "已拒绝";
  return "resume";
}

export function emptyReviewResultText(approved) {
  if (approved === false) return "你已拒绝本次审批，没有提交任何写入。";
  if (approved === true) return "审批已继续，但后端没有返回已提交写入。";
  return "没有提交写入。";
}
