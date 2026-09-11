import { ref } from "vue";
import { taskDetailFromResponse } from "../runtime/insessionTaskDetails";

function relatedTaskIds(value) {
  if (!Array.isArray(value)) return [];
  const unique = new Set();
  for (const taskId of value) {
    if (typeof taskId === "string" && taskId.trim()) unique.add(taskId);
  }
  return [...unique].slice(0, 24);
}

/**
 * 当前已确认 Session 的只读 Task 装饰。
 *
 * 这些请求不会取得会话 authority、改变 Turn 状态或延长 composer busy，并独立
 * 丢弃属于旧 Session 的过期结果。
 */
export function useChatRuntimeDecorations({
  api,
  selectedSessionId
}) {
  const inSessionTaskDetails = ref([]);
  let inSessionTaskDetailsGeneration = 0;

  function clearInSessionTaskDetails() {
    inSessionTaskDetailsGeneration += 1;
    inSessionTaskDetails.value = [];
  }

  async function loadRelatedInSessionTaskDetails(sessionId, taskIds) {
    const generation = ++inSessionTaskDetailsGeneration;
    const ids = relatedTaskIds(taskIds);
    inSessionTaskDetails.value = [];
    if (!sessionId || !ids.length || typeof api?.getInSessionTaskDetails !== "function") return [];

    // 这项只读装饰绝不能让已完成的聊天 Turn 继续显示忙碌，也不能在对话界面暴露
    // 端点或存储错误。
    const results = await Promise.allSettled(
      ids.map((taskId) => Promise.resolve().then(
        () => api.getInSessionTaskDetails(sessionId, taskId)
      ))
    );
    if (generation !== inSessionTaskDetailsGeneration || selectedSessionId.value !== sessionId) return [];

    const details = results.flatMap((result, index) => {
      if (result.status !== "fulfilled") return [];
      const task = taskDetailFromResponse(result.value);
      return task?.insession_task_id === ids[index] ? [task] : [];
    });
    inSessionTaskDetails.value = details;
    return details;
  }

  return {
    inSessionTaskDetails,
    clearInSessionTaskDetails,
    loadRelatedInSessionTaskDetails
  };
}
