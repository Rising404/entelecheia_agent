import { effectScope, ref } from "vue";
import { flushPromises } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

import { useWorkspaceFiles } from "./useWorkspaceFiles";

function makeClient() {
  return {
    listWorkspaceFiles: vi.fn().mockResolvedValue({ rel_path: "", entries: [] }),
    createWorkspaceEntry: vi.fn().mockResolvedValue({ ok: true }),
    readWorkspaceFile: vi.fn(),
    writeWorkspaceFile: vi.fn(),
    deleteWorkspaceEntry: vi.fn().mockResolvedValue({ ok: true })
  };
}

function createController({ client = makeClient(), confirmDelete = vi.fn(() => true) } = {}) {
  const scope = effectScope();
  const controller = scope.run(() => useWorkspaceFiles({
    client,
    sessionId: ref("session-a"),
    workingDir: ref("/tmp/project"),
    open: ref(true),
    confirmDelete
  }));
  return { client, confirmDelete, controller, scope };
}

describe("useWorkspaceFiles", () => {
  it("uses only its injected client and refreshes after creating an entry", async () => {
    const { client, controller, scope } = createController();
    await flushPromises();

    controller.newName.value = "notes.md";
    await controller.createEntry();

    expect(client.createWorkspaceEntry).toHaveBeenCalledWith("session-a", {
      path: "notes.md",
      kind: "file"
    });
    expect(client.listWorkspaceFiles).toHaveBeenCalledTimes(2);
    scope.stop();
  });

  it("keeps deletion confirmation and refresh inside the controller", async () => {
    const { client, confirmDelete, controller, scope } = createController();
    await flushPromises();
    const entry = { name: "notes.md", rel_path: "notes.md" };

    await controller.deleteEntry(entry);

    expect(confirmDelete).toHaveBeenCalledWith(entry);
    expect(client.deleteWorkspaceEntry).toHaveBeenCalledWith("session-a", {
      path: "notes.md"
    });
    expect(client.listWorkspaceFiles).toHaveBeenCalledTimes(2);
    scope.stop();
  });
});
