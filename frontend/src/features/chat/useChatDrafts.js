import { ref } from "vue";

const DEFAULT_PREFIX = "personagraph.chatDraft.";

export function useChatDrafts({ storage = globalThis.sessionStorage, prefix = DEFAULT_PREFIX } = {}) {
  const draftSessionIds = ref(new Set());

  function keyFor(sessionId) {
    return sessionId ? `${prefix}${sessionId}` : "";
  }

  function load(sessionId) {
    const key = keyFor(sessionId);
    if (!key || !storage) return "";
    try {
      return storage.getItem(key) || "";
    } catch {
      return "";
    }
  }

  function refreshIds() {
    const ids = new Set();
    if (!storage) {
      draftSessionIds.value = ids;
      return;
    }
    try {
      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index) || "";
        if (key.startsWith(prefix) && storage.getItem(key)) {
          ids.add(key.slice(prefix.length));
        }
      }
    } catch {
      // 草稿标记尽力而为，绝不能阻塞会话列表。
    }
    draftSessionIds.value = ids;
  }

  function has(sessionId) {
    return draftSessionIds.value.has(sessionId);
  }

  function save(sessionId, value) {
    const key = keyFor(sessionId);
    if (!key || !storage) return;
    const text = String(value || "");
    try {
      if (text) storage.setItem(key, text);
      else storage.removeItem(key);
    } catch {
      // 浏览器存储不可用时，聊天仍须可用。
    }
    refreshIds();
  }

  function clear(sessionId) {
    save(sessionId, "");
  }

  return { draftSessionIds, load, refreshIds, has, save, clear };
}
