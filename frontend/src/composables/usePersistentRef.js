import { ref, watch } from "vue";

const PREFIX = "personagraph.ui.";

/**
 * 一个记得住自己的 ref，用于用户主动做过的界面选择。
 *
 * 只用于"用户按过一次就该一直生效"的偏好——收起某个面板、拖过的栏宽。
 * 不用于服务端拥有的状态：那种东西存在本地只会和真相分叉。
 *
 * 读不出来就用默认值。一个损坏的偏好不值得让界面起不来。
 */
export function usePersistentRef(key, fallback, { parse, serialize } = {}) {
  const storageKey = PREFIX + key;
  const decode = parse || ((raw) => JSON.parse(raw));
  const encode = serialize || ((value) => JSON.stringify(value));

  let initial = fallback;
  try {
    const raw = globalThis.localStorage?.getItem(storageKey);
    if (raw !== null && raw !== undefined) {
      const parsed = decode(raw);
      if (parsed !== undefined && parsed !== null) initial = parsed;
    }
  } catch {
    initial = fallback;
  }

  const state = ref(initial);
  watch(state, (value) => {
    try {
      globalThis.localStorage?.setItem(storageKey, encode(value));
    } catch {
      // 存不下（隐私模式、配额满）不该影响本次会话里的界面行为。
    }
  }, { deep: false });

  return state;
}
