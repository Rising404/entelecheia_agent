/** 运行阶段的说法。代号是给日志看的，这里是给人看的。 */
export const STAGE_LABELS = {
  INGRESS: "检查输入",
  CLASSIFY: "理解请求",
  SUPERVISOR: "选择处理路径",
  L0_GENERATE: "生成回复",
  L2_UNDERSTAND: "分析任务",
  L2_PLAN: "制定计划",
  TRANSITION_GUARD: "检查执行条件",
  TOOL: "调用能力",
  OBSERVATION: "记录观察结果",
  VERIFICATION: "验证结果",
  PERSIST: "保存结果",
  RESPONSE: "交付回复"
};

export function stageText(stage) {
  return STAGE_LABELS[stage] || "";
}
