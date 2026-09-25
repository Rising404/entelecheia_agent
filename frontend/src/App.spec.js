import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, shallowMount } from "@vue/test-utils";
import { createPinia } from "pinia";
import App from "./App.vue";
import { api } from "./api";
import SessionTree from "./features/workspace/SessionTree.vue";
import ChatWorkspace from "./features/chat/ChatWorkspace.vue";

vi.mock("./api", () => ({
  api: Object.fromEntries([
    "status", "getBackground", "listFolders", "listSessions", "listProjects",
    "getSession", "listAttachments", "getRuntimeEvents", "getSessionContext",
    "listDocuments", "listDocumentIngestJobs", "patchSession"
  ].map((name) => [name, vi.fn()]))
}));

let wrapper;

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
});

describe("App session rename wiring", () => {
  it.each(["selected", "other"])("renames the emitted %s session and refreshes without changing selection", async (targetId) => {
    let sessions = [
      { id: "selected", title: "当前会话", status: "active" },
      { id: "other", title: "另一会话", status: "active" }
    ];
    api.status.mockResolvedValue({ ok: true, model_configured: true });
    api.getBackground.mockResolvedValue({ background: null });
    api.listFolders.mockResolvedValue({ folders: [] });
    api.listSessions.mockImplementation(async () => ({ sessions: sessions.map((item) => ({ ...item })) }));
    api.listProjects.mockImplementation(async () => ({ projects: [], unbound: sessions.map((item) => ({ ...item })) }));
    api.getSession.mockImplementation(async (id) => ({
      session: { ...sessions.find((item) => item.id === id) },
      turns: [], turn_window: { window_state: "empty", window_revision: 0 }
    }));
    api.listAttachments.mockResolvedValue({ attachments: [] });
    api.getRuntimeEvents.mockResolvedValue({ enabled: true, events: [] });
    api.getSessionContext.mockResolvedValue({});
    api.listDocuments.mockResolvedValue({ documents: [] });
    api.listDocumentIngestJobs.mockResolvedValue({ jobs: [] });
    api.patchSession.mockImplementation(async (id, update) => {
      sessions = sessions.map((item) => item.id === id ? { ...item, ...update } : item);
      return { session: sessions.find((item) => item.id === id) };
    });
    const handleError = vi.fn();
    wrapper = shallowMount(App, {
      global: { plugins: [createPinia()], config: { errorHandler: handleError } }
    });
    await flushPromises();
    const tree = wrapper.findComponent(SessionTree);
    expect(tree.props("selectedId")).toBe("selected");
    api.listSessions.mockClear();

    tree.vm.$emit("rename-session", { sessionId: targetId, title: "  已改名  " });
    await flushPromises();

    expect(api.patchSession).toHaveBeenCalledExactlyOnceWith(targetId, { title: "已改名" });
    expect(api.listSessions).toHaveBeenCalledOnce();
    expect(tree.props("unbound").find((item) => item.id === targetId).title).toBe("已改名");
    expect(tree.props("selectedId")).toBe("selected");
    expect(wrapper.findComponent(ChatWorkspace).props("title")).toBe(
      targetId === "selected" ? "已改名" : "当前会话"
    );
    expect(handleError).not.toHaveBeenCalled();
  });
});
