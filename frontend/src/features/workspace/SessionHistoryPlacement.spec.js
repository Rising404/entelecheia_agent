import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import SessionHistoryPlacement from "./SessionHistoryPlacement.vue";

const folders = [
  { id: "folder-a", name: "项目", depth: 0, path: "项目" },
  { id: "folder-b", name: "研究", depth: 1, path: "项目 / 研究" }
];

describe("SessionHistoryPlacement", () => {
  it("renders history folders without inferring anything from the working directory", () => {
    const wrapper = mount(SessionHistoryPlacement, {
      props: {
        session: { id: "session-a", folder_id: "folder-b", working_dir: "/tmp/shared" },
        folders
      }
    });

    expect(wrapper.get("label").attributes("title")).toBe("历史归档：项目 / 研究");
    expect(wrapper.get("select").element.value).toBe("folder-b");
    expect(wrapper.text()).toContain("项目");
    expect(wrapper.text()).toContain("研究");
  });

  it("emits a normalized placement intent and ignores the current value", async () => {
    const wrapper = mount(SessionHistoryPlacement, {
      props: { session: { id: "session-a", folder_id: "folder-a" }, folders }
    });

    await wrapper.get("select").setValue("folder-a");
    expect(wrapper.emitted("move")).toBeUndefined();

    await wrapper.get("select").setValue("");
    expect(wrapper.emitted("move")).toEqual([[{ sessionId: "session-a", folderId: null }]]);
  });
});
