import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SessionTree from "./SessionTree.vue";

const projects = [
  {
    path: "/tmp/report",
    name: "report",
    pinned: false,
    sessions: [
      { id: "a-session", title: "季度分析", status: "active", working_dir: "/tmp/report" },
      { id: "b-session", title: "补充材料", status: "active", working_dir: "/tmp/report" }
    ]
  },
  { path: "/tmp/empty", name: "空项目", pinned: false, sessions: [] }
];

const unbound = [
  { id: "free-session", title: "随手问问", status: "active", working_dir: null }
];

function build(props = {}) {
  return mount(SessionTree, {
    props: { projects, unbound, draftSessionIds: new Set(), ...props }
  });
}

describe("SessionTree", () => {
  it("groups sessions by the directory they are bound to", () => {
    const tree = build();
    const text = tree.text();
    expect(text).toContain("report");
    expect(text).toContain("季度分析");
    expect(text).toContain("补充材料");
    // 没绑目录的会话仍然可见，只是单独一组——否则它就打不开了。
    expect(text).toContain("未归入项目");
    expect(text).toContain("随手问问");
  });

  it("collapses a project without hiding it, and keeps an empty project listed", async () => {
    const tree = build();
    const header = tree.findAll("button").find((button) => button.text().includes("report"));
    await header.trigger("click");
    expect(tree.text()).toContain("report");
    expect(tree.text()).not.toContain("季度分析");
    expect(tree.text()).toContain("空项目");
  });

  it("starts a draft rather than creating a session up front", async () => {
    const tree = build();
    await tree.findAll("button").find((button) => button.text().includes("新建会话")).trigger("click");
    expect(tree.emitted("start-draft")).toHaveLength(1);
    expect(tree.emitted("create")).toBeUndefined();
  });

  it("lets an uncreated draft select a directory", async () => {
    const tree = build({ draftActive: true, directoryPickerAvailable: true });
    await tree.get('[title="选择本机目录"]').trigger("click");
    expect(tree.emitted("choose-draft-directory")).toHaveLength(1);
    await tree.get('[aria-label="会话工作目录"]').setValue("/chosen/root");
    expect(tree.emitted("update-draft-directory")[0]).toEqual(["/chosen/root"]);
  });

  it("starts a new draft inside an existing project", async () => {
    const tree = build();
    await tree.findAll('[title="在此项目新建会话"]')[0].trigger("click");
    expect(tree.emitted("start-draft")[0]).toEqual(["/tmp/report"]);
  });

  it("locks directory controls while creation is awaiting confirmation", async () => {
    const tree = build({ draftActive: true, draftDirectoryLocked: true, directoryPickerAvailable: true });
    expect(tree.get('[aria-label="会话工作目录"]').element.disabled).toBe(true);
    expect(tree.get('[title="选择本机目录"]').element.disabled).toBe(true);
    await tree.get('[title="选择本机目录"]').trigger("click");
    expect(tree.emitted("choose-draft-directory")).toBeUndefined();
  });

  it("offers pinning instead of the old forget button", () => {
    // 空 project 现在不会出现在列表里——存在与否完全由"有没有会话绑着它"决定，
    // 所以"不再列出"这个按钮没有存在理由了，换成置顶。
    const tree = build();
    const titles = tree.findAll("button").map((button) => button.attributes("title"));
    expect(titles).not.toContain("不再列出这个项目");
    expect(titles.filter((title) => title === "置顶这个项目").length).toBeGreaterThan(0);
  });

  it("emits the flipped pin state so one click toggles", async () => {
    const tree = build();
    const pin = tree.findAll("button").find((b) => b.attributes("title") === "置顶这个项目");
    await pin.trigger("click");
    expect(tree.emitted("pin-project")[0][0]).toMatchObject({ pinned: true });
  });
});

describe("SessionTree 项目拖拽排序", () => {
  function dragTree() {
    return build({
      projects: [
        { path: "/a", name: "甲", pinned: false, sessions: [] },
        { path: "/b", name: "乙", pinned: false, sessions: [] },
        { path: "/c", name: "丙", pinned: false, sessions: [] }
      ]
    });
  }

  it("把拖动项插到落点位置，并回传完整顺序", async () => {
    // 回传整份顺序而不是"移到谁前面"：相对指令要求两边对同一份列表有一致认知，
    // 而列表随时会因为会话状态变化而变。
    const tree = dragTree();
    const rows = tree.findAll('[draggable="true"]');
    await rows[2].trigger("dragstart", { dataTransfer: { setData() {}, effectAllowed: "" } });
    await rows[0].trigger("drop");
    expect(tree.emitted("reorder-projects")[0][0]).toEqual(["/c", "/a", "/b"]);
  });

  it("拖到自己身上不发事件", async () => {
    const tree = dragTree();
    const rows = tree.findAll('[draggable="true"]');
    await rows[1].trigger("dragstart", { dataTransfer: { setData() {}, effectAllowed: "" } });
    await rows[1].trigger("drop");
    expect(tree.emitted("reorder-projects")).toBeUndefined();
  });

  it("每个项目都显示完整路径，避免同名目录被认错", () => {
    const text = build().text();

    // 名字取自路径末段，不同目录经常同名；路径必须可见，而不是只躲在 title 里。
    expect(text).toContain("/tmp/report");
    expect(text).toContain("/tmp/empty");
  });

});
