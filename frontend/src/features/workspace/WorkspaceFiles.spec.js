import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import WorkspaceFiles from "./WorkspaceFiles.vue";

const api = {
  listWorkspaceFiles: vi.fn(),
  createWorkspaceEntry: vi.fn(),
  readWorkspaceFile: vi.fn(),
  writeWorkspaceFile: vi.fn(),
  deleteWorkspaceEntry: vi.fn()
};

function mountWorkspaceFiles(props) {
  return mount(WorkspaceFiles, { props: { workspaceFilesClient: api, ...props } });
}

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
}

describe("WorkspaceFiles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listWorkspaceFiles.mockResolvedValue({ rel_path: "", entries: [] });
  });

  it("shows an unavailable state without querying files or offering rebinding", () => {
    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "", open: true
    });

    expect(wrapper.text()).toContain("未绑定工作目录");
    expect(wrapper.text()).toContain("历史归档不受影响");
    expect(api.listWorkspaceFiles).not.toHaveBeenCalled();
    expect(wrapper.findAll("button").some((button) => button.text().includes("绑定工作目录"))).toBe(false);
    expect(wrapper.emitted("bind-working-directory")).toBeUndefined();
  });

  it("does not load files while independently collapsed and emits only its own open state", async () => {
    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "/tmp/project", open: false
    });

    expect(api.listWorkspaceFiles).not.toHaveBeenCalled();
    await wrapper.get("button[title='展开本地目录面板']").trigger("click");
    expect(wrapper.emitted("update:open")).toEqual([[true]]);
  });

  it("loads only the bound current session when the panel is open", async () => {
    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "/tmp/project", open: true
    });
    await flushPromises();

    expect(api.listWorkspaceFiles).toHaveBeenCalledWith("session-a", { path: "" });
    expect(wrapper.text()).toContain("/tmp/project");
  });

  it("does not let a slower former session overwrite the current directory view", async () => {
    let resolveFormer;
    api.listWorkspaceFiles
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFormer = resolve; }))
      .mockResolvedValueOnce({ rel_path: "", entries: [{ name: "current.txt", kind: "file", rel_path: "current.txt", size: 1 }] });

    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "/tmp/a", open: true
    });
    await wrapper.setProps({ sessionId: "session-b", workingDir: "/tmp/b" });
    await flushPromises();
    resolveFormer({ rel_path: "", entries: [{ name: "former.txt", kind: "file", rel_path: "former.txt", size: 1 }] });
    await flushPromises();

    expect(wrapper.text()).toContain("current.txt");
    expect(wrapper.text()).not.toContain("former.txt");
  });

  it("does not expose a late former-session file read in the current Session editor", async () => {
    const formerRead = deferred();
    api.listWorkspaceFiles.mockImplementation(async (sessionId) => ({
      rel_path: "",
      entries: sessionId === "session-a"
        ? [{ name: "former.txt", kind: "file", rel_path: "former.txt", size: 3 }]
        : []
    }));
    api.readWorkspaceFile.mockReturnValue(formerRead.promise);

    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "/tmp/a", open: true
    });
    await flushPromises();
    await wrapper.get("button[title='former.txt']").trigger("click");

    await wrapper.setProps({ sessionId: "session-b", workingDir: "/tmp/b" });
    await flushPromises();
    formerRead.resolve({ content: "secret content from A" });
    await flushPromises();

    expect(wrapper.text()).not.toContain("编辑：former.txt");
    expect(wrapper.find("textarea").exists()).toBe(false);
    expect(wrapper.find("button.cmd.primary").exists()).toBe(false);
    expect(api.writeWorkspaceFile).not.toHaveBeenCalled();
  });

  it("does not let a former-session save error settle the current Session save", async () => {
    const saveA = deferred();
    const saveB = deferred();
    api.listWorkspaceFiles.mockImplementation(async (sessionId) => ({
      rel_path: "",
      entries: [{
        name: sessionId === "session-a" ? "a.txt" : "b.txt",
        kind: "file",
        rel_path: sessionId === "session-a" ? "a.txt" : "b.txt",
        size: 1
      }]
    }));
    api.readWorkspaceFile.mockImplementation(async (sessionId) => ({ content: `content ${sessionId}` }));
    api.writeWorkspaceFile.mockImplementation((sessionId) => (
      sessionId === "session-a" ? saveA.promise : saveB.promise
    ));

    const wrapper = mountWorkspaceFiles({
      sessionId: "session-a", workingDir: "/tmp/a", open: true
    });
    await flushPromises();
    await wrapper.get("button[title='a.txt']").trigger("click");
    await flushPromises();
    await wrapper.get("textarea").setValue("edited A");
    await wrapper.get("button.cmd.primary").trigger("click");
    expect(api.writeWorkspaceFile).toHaveBeenCalledWith("session-a", {
      path: "a.txt",
      content: "edited A"
    });

    await wrapper.setProps({ sessionId: "session-b", workingDir: "/tmp/b" });
    await flushPromises();
    await wrapper.get("button[title='b.txt']").trigger("click");
    await flushPromises();
    await wrapper.get("textarea").setValue("edited B");
    await wrapper.get("button.cmd.primary").trigger("click");
    expect(api.writeWorkspaceFile).toHaveBeenCalledWith("session-b", {
      path: "b.txt",
      content: "edited B"
    });

    saveA.reject(new Error("late A save failure"));
    await flushPromises();
    expect(wrapper.text()).not.toContain("late A save failure");
    expect(wrapper.text()).toContain("编辑：b.txt");
    expect(wrapper.get("button.cmd.primary").text()).toContain("保存中");

    saveB.resolve({ ok: true });
    await flushPromises();
    expect(wrapper.get("button.cmd.primary").text()).toContain("保存");
    expect(wrapper.get("button.cmd.primary").text()).not.toContain("保存中");
  });
});
