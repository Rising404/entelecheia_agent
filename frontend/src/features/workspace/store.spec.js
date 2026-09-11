import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

import { createWorkspaceStore } from "./store";

const api = {
  listFolders: vi.fn(),
  listSessions: vi.fn(),
  listDocuments: vi.fn(),
  createSession: vi.fn(),
  patchSession: vi.fn(),
  createFolder: vi.fn(),
  patchFolder: vi.fn(),
  deleteFolder: vi.fn()
};

function deferred() {
  let resolve;
  const promise = new Promise((next) => { resolve = next; });
  return { promise, resolve };
}

describe("WorkspaceStore", () => {
  let useWorkspaceStore;

  beforeEach(() => {
    setActivePinia(createPinia());
    useWorkspaceStore = createWorkspaceStore({ client: api });
    vi.clearAllMocks();
    api.listFolders.mockResolvedValue({
      folders: [
        { id: "folder-a", name: "项目 A", children: [] },
        { id: "folder-b", name: "项目 B", children: [] }
      ]
    });
    api.listSessions.mockResolvedValue({
      sessions: [
        { id: "session-a", folder_id: "folder-a", working_dir: "/tmp/shared" },
        { id: "session-b", folder_id: "folder-b", working_dir: "/tmp/shared" },
        { id: "session-none", folder_id: null, working_dir: null }
      ]
    });
    api.listDocuments.mockResolvedValue({ documents: [{ id: "doc-a", title: "会议记录" }] });
    api.patchSession.mockResolvedValue({ session: { id: "session-a", folder_id: "folder-a" } });
  });

  it("requires the shell to inject a workspace client", () => {
    expect(() => createWorkspaceStore()).toThrow("requires an injected workspace client");
  });

  it("loads folders and sessions as independent history and local-context axes", async () => {
    const store = useWorkspaceStore();

    await store.refresh();
    store.selectFolder("folder-a");

    expect(api.listFolders).toHaveBeenCalledWith({ status: "active" });
    expect(api.listSessions).toHaveBeenCalledWith({ status: "active", limit: 200 });
    expect(store.selectedFolder?.name).toBe("项目 A");
    expect(store.sessions.filter((session) => session.working_dir === "/tmp/shared").map((session) => session.folder_id))
      .toEqual(["folder-a", "folder-b"]);
    expect(store.historyFolderOptions).toEqual([
      { id: "folder-a", name: "项目 A", depth: 0, path: "项目 A" },
      { id: "folder-b", name: "项目 B", depth: 0, path: "项目 B" }
    ]);
    expect("workspaceGroups" in store).toBe(false);
  });

  it("lets only the latest workspace-list search publish or settle loading", async () => {
    const store = useWorkspaceStore();
    const oldFolders = deferred();
    const oldSessions = deferred();
    const currentFolders = deferred();
    const currentSessions = deferred();
    api.listFolders
      .mockReturnValueOnce(oldFolders.promise)
      .mockReturnValueOnce(currentFolders.promise);
    api.listSessions
      .mockReturnValueOnce(oldSessions.promise)
      .mockReturnValueOnce(currentSessions.promise);

    const oldRefresh = store.refresh({ query: "old query" });
    const currentRefresh = store.refresh({ query: "current query" });

    oldFolders.resolve({ folders: [{ id: "old-folder", name: "Old", children: [] }] });
    oldSessions.resolve({ sessions: [{ id: "old-session" }] });
    await oldRefresh;

    expect(store.folders).toEqual([]);
    expect(store.sessions).toEqual([]);
    expect(store.loading).toBe(true);

    currentFolders.resolve({ folders: [{ id: "current-folder", name: "Current", children: [] }] });
    currentSessions.resolve({ sessions: [{ id: "current-session" }] });
    await currentRefresh;

    expect(store.folders.map((folder) => folder.id)).toEqual(["current-folder"]);
    expect(store.sessions.map((session) => session.id)).toEqual(["current-session"]);
    expect(store.loading).toBe(false);
  });

  it("does not clear a Session selected after a workspace-list refresh started", async () => {
    const store = useWorkspaceStore();
    const folders = deferred();
    const sessions = deferred();
    api.listFolders.mockReturnValue(folders.promise);
    api.listSessions.mockReturnValue(sessions.promise);
    store.selectSession("session-a");

    const refreshing = store.refresh();
    store.selectSession("session-b");
    folders.resolve({ folders: [] });
    sessions.resolve({ sessions: [{ id: "session-a" }] });
    await refreshing;

    expect(store.selectedSessionId).toBe("session-b");
  });

  it("preserves history placement while leaving workspace assignment to the backend", async () => {
    const store = useWorkspaceStore();
    api.createSession.mockResolvedValue({
      session: { id: "session-new", folder_id: "folder-a", working_dir: "/tmp/project" }
    });

    const session = await store.createFreeSession({
      title: "论文整理",
      folderId: "folder-a"
    });

    expect(api.createSession).toHaveBeenCalledWith({
      title: "论文整理",
      folder_id: "folder-a"
    });
    expect(session?.id).toBe("session-new");
    expect(session?.working_dir).toBe("/tmp/project");
    expect(store.selectedFolderId).toBe("folder-a");
    expect(store.selectedSessionId).toBe("session-new");
    expect(store).not.toHaveProperty("chooseDirectory");
    expect(store).not.toHaveProperty("createSessionInDirectory");
    expect(store).not.toHaveProperty("bindWorkingDirectory");
  });

  it("uses the backend-returned default workspace without sending a path", async () => {
    const store = useWorkspaceStore();
    api.createSession.mockResolvedValue({
      session: {
        id: "session-new",
        folder_id: "folder-b",
        working_dir: "/default/2026-09-02_随手聊聊"
      }
    });

    const session = await store.createFreeSession({ title: "随手聊聊", folderId: "folder-b" });

    expect(api.createSession).toHaveBeenCalledWith({
      title: "随手聊聊",
      folder_id: "folder-b"
    });
    expect(session?.working_dir).toBe("/default/2026-09-02_随手聊聊");
  });

  it("loads only the selected session's document aggregate", async () => {
    const store = useWorkspaceStore();
    store.selectSession("session-a");

    await store.refreshContents();

    expect(api.listDocuments).toHaveBeenCalledWith({ session_id: "session-a" });
    expect(store.workspaceDocuments.map((document) => document.id)).toEqual(["doc-a"]);
    expect(store).not.toHaveProperty("workspaceTasks");
  });

  it("clears the former session contents as soon as the selected session changes", async () => {
    const store = useWorkspaceStore();
    store.selectSession("session-a");
    await store.refreshContents();
    expect(store.workspaceDocuments.map((document) => document.id)).toEqual(["doc-a"]);

    store.selectSession("session-b");

    expect(store.workspaceDocuments).toEqual([]);
    expect(store.contentsLoading).toBe(false);
  });

  it("lets only the latest session request publish contents or finish loading", async () => {
    const store = useWorkspaceStore();
    const docsA = deferred();
    const docsB = deferred();
    api.listDocuments.mockImplementation(({ session_id: sessionId }) => (
      sessionId === "session-a" ? docsA.promise : docsB.promise
    ));

    store.selectSession("session-a");
    const formerLoad = store.refreshContents();
    store.selectSession("session-b");
    const currentLoad = store.refreshContents();

    expect(store.workspaceDocuments).toEqual([]);
    expect(store.contentsLoading).toBe(true);

    docsA.resolve({ documents: [{ id: "doc-a" }] });
    await formerLoad;

    expect(store.workspaceDocuments).toEqual([]);
    expect(store.contentsLoading).toBe(true);

    docsB.resolve({ documents: [{ id: "doc-b" }] });
    await currentLoad;

    expect(store.workspaceDocuments.map((document) => document.id)).toEqual(["doc-b"]);
    expect(store.contentsLoading).toBe(false);
  });

  it("uses a request generation when selection returns from A through B to A", async () => {
    const store = useWorkspaceStore();
    const oldDocsA = deferred();
    const docsB = deferred();
    const currentDocsA = deferred();
    api.listDocuments
      .mockReturnValueOnce(oldDocsA.promise)
      .mockReturnValueOnce(docsB.promise)
      .mockReturnValueOnce(currentDocsA.promise);

    store.selectSession("session-a");
    const oldALoad = store.refreshContents();
    store.selectSession("session-b");
    const bLoad = store.refreshContents();
    store.selectSession("session-a");
    const currentALoad = store.refreshContents();

    currentDocsA.resolve({ documents: [{ id: "current-doc-a" }] });
    await currentALoad;
    docsB.resolve({ documents: [{ id: "doc-b" }] });
    await bLoad;
    oldDocsA.resolve({ documents: [{ id: "old-doc-a" }] });
    await oldALoad;

    expect(store.selectedSessionId).toBe("session-a");
    expect(store.workspaceDocuments.map((document) => document.id)).toEqual(["current-doc-a"]);
    expect(store.contentsLoading).toBe(false);
  });

  it("does not repopulate contents after the session selection is cleared", async () => {
    const store = useWorkspaceStore();
    const documents = deferred();
    api.listDocuments.mockReturnValue(documents.promise);

    store.selectSession("session-a");
    const formerLoad = store.refreshContents();
    store.selectSession("");
    documents.resolve({ documents: [{ id: "doc-a" }] });
    await formerLoad;

    expect(store.workspaceDocuments).toEqual([]);
    expect(store.contentsLoading).toBe(false);
  });

  it("forwards history-folder mutations through the workspace API facade", async () => {
    const store = useWorkspaceStore();
    api.createFolder.mockResolvedValue({ folder: { id: "folder-new", name: "新项目" } });
    api.patchFolder.mockResolvedValue({ folder: { id: "folder-new", name: "改名后" } });
    api.deleteFolder.mockResolvedValue({ ok: true });

    await store.createFolder({ name: "新项目", parentId: "folder-a" });
    await store.renameFolder("folder-new", "改名后");
    await store.moveFolder("folder-new", null);
    await store.setFolderStatus("folder-new", "archived");
    await store.deleteFolder("folder-new");

    expect(api.createFolder).toHaveBeenCalledWith({ name: "新项目", parent_id: "folder-a" });
    expect(api.patchFolder).toHaveBeenNthCalledWith(1, "folder-new", { name: "改名后" });
    expect(api.patchFolder).toHaveBeenNthCalledWith(2, "folder-new", { parent_id: null });
    expect(api.patchFolder).toHaveBeenNthCalledWith(3, "folder-new", { status: "archived" });
    expect(api.deleteFolder).toHaveBeenCalledWith("folder-new");
  });

  it("moves a session's history placement without changing its working directory", async () => {
    const store = useWorkspaceStore();
    api.patchSession.mockResolvedValue({
      session: { id: "session-a", folder_id: "folder-b", working_dir: "/tmp/shared" }
    });

    const session = await store.moveSession("session-a", "folder-b");

    expect(api.patchSession).toHaveBeenCalledWith("session-a", { folder_id: "folder-b" });
    expect(session).toEqual({ id: "session-a", folder_id: "folder-b", working_dir: "/tmp/shared" });
  });
});
