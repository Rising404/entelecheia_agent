import { describe, expect, it, vi } from "vitest";

import { useProjects } from "./useProjects";

function makeClient() {
  return {
    forgetProject: vi.fn(),
    listProjects: vi.fn().mockResolvedValue({ projects: [], unbound: [] }),
    pinProject: vi.fn(),
    rememberProject: vi.fn(),
    renameProject: vi.fn(),
    reorderProjects: vi.fn()
  };
}

describe("useProjects", () => {
  it("keeps a selected path only on the uncreated draft", () => {
    const projects = useProjects({ apiClient: makeClient() });
    projects.startDraft("/chosen/project");
    expect(projects.draftWorkingDir.value).toBe("/chosen/project");
    projects.setDraftDirectory("/another/project");
    expect(projects.draftWorkingDir.value).toBe("/another/project");
    projects.discardDraft();
    projects.setDraftDirectory("/must-not-bind");
    expect(projects.draft.value).toBeNull();
  });
  it("requires its API client to be injected", () => {
    expect(() => useProjects()).toThrow("requires an injected API client");
  });

  it("freezes creation parameters until a confirmed rejection or draft disposal", () => {
    const projects = useProjects({ apiClient: makeClient() });
    projects.startDraft("/selected");
    const first = projects.prepareDraftCreation("first");
    projects.setDraftDirectory("/ignored");
    expect(projects.draftDirectoryLocked.value).toBe(true);
    expect(projects.prepareDraftCreation("changed")).toBe(first);
    expect(projects.draftWorkingDir.value).toBe("/selected");
    projects.resetDraftCreation();
    projects.setDraftDirectory("/corrected");
    const second = projects.prepareDraftCreation("changed");
    expect(second.working_dir).toBe("/corrected");
    expect(second.client_request_id).not.toBe(first.client_request_id);
    projects.draft.value.session = { id: "created" };
    projects.resetDraftCreation();
    expect(projects.prepareDraftCreation("ignored")).toBe(second);
    projects.discardDraft();
    expect(projects.draftDirectoryLocked.value).toBe(false);
  });

  it("loads projects through the injected client", async () => {
    const apiClient = makeClient();
    apiClient.listProjects.mockResolvedValue({
      projects: [{ path: "/tmp/project", name: "project" }],
      unbound: [{ id: "session-a" }]
    });
    const projects = useProjects({ apiClient });

    await projects.load("archived", "draft");

    expect(apiClient.listProjects).toHaveBeenCalledWith("archived", "draft");
    expect(projects.projects.value).toEqual([{ path: "/tmp/project", name: "project" }]);
    expect(projects.unbound.value).toEqual([{ id: "session-a" }]);
  });

  it("rolls back an optimistic reorder when the injected client rejects it", async () => {
    const apiClient = makeClient();
    apiClient.reorderProjects.mockRejectedValue(new Error("offline"));
    const showError = vi.fn();
    const projects = useProjects({ apiClient, showError });
    projects.projects.value = [{ path: "/a" }, { path: "/b" }];

    const result = await projects.reorder(["/b", "/a"]);

    expect(result).toBe(false);
    expect(projects.projects.value).toEqual([{ path: "/a" }, { path: "/b" }]);
    expect(showError).toHaveBeenCalledOnce();
  });
});
