/**
 * 会话状态的说法，和任务状态分开。
 *
 * 会话的 `active` 只表示「没被归档或删掉」，而任务的 `active` 表示「正在做」。
 * 共用一张表时，一个刚建好、什么都没发生的会话会显示成「进行中」——
 * 而同一屏上「当前回合仍在运行」用的也是这个词，于是分不清哪个是哪个。
 */
export function sessionStatusText(value) {
  return { active: "活跃", archived: "已归档", trashed: "回收站" }[value] || value;
}
