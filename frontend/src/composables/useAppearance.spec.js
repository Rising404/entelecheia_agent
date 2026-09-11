import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { effectScope } from "vue";

import { api } from "../api";
import { useAppearance } from "./useAppearance";

let created;
let revoked;

beforeEach(() => {
  created = 0;
  revoked = [];
  // jsdom 没有实现 object URL，这里给出可观测的替身，好断言创建与释放成对发生。
  globalThis.URL.createObjectURL = vi.fn(() => `blob:stub-${(created += 1)}`);
  globalThis.URL.revokeObjectURL = vi.fn((value) => revoked.push(value));
});

afterEach(() => vi.restoreAllMocks());

function backgroundPayload(path) {
  return { background: { kind: "image", media_type: "image/jpeg", url: path } };
}

describe("useAppearance", () => {
  it("把资产取回为 object URL，而不是让渲染层直接引用后端地址", async () => {
    const asset = vi
      .spyOn(api, "fetchBackgroundAsset")
      .mockResolvedValue(new Blob(["x"], { type: "image/jpeg" }));
    vi.spyOn(api, "getBackground").mockResolvedValue(
      backgroundPayload("/api/appearance/background/asset?v=1")
    );

    const appearance = useAppearance();
    await appearance.load();

    expect(asset).toHaveBeenCalledWith("/api/appearance/background/asset?v=1");
    expect(appearance.objectUrl.value).toBe("blob:stub-1");
  });

  it("同一份资产不重复下载", async () => {
    const asset = vi
      .spyOn(api, "fetchBackgroundAsset")
      .mockResolvedValue(new Blob(["x"], { type: "image/jpeg" }));
    vi.spyOn(api, "getBackground").mockResolvedValue(
      backgroundPayload("/api/appearance/background/asset?v=1")
    );

    const appearance = useAppearance();
    await appearance.load();
    await appearance.load();

    expect(asset).toHaveBeenCalledTimes(1);
    expect(appearance.objectUrl.value).toBe("blob:stub-1");
  });

  it("换了背景就释放旧的 object URL", async () => {
    vi.spyOn(api, "fetchBackgroundAsset").mockResolvedValue(
      new Blob(["x"], { type: "image/jpeg" })
    );
    vi.spyOn(api, "getBackground")
      .mockResolvedValueOnce(backgroundPayload("/api/appearance/background/asset?v=1"))
      .mockResolvedValueOnce(backgroundPayload("/api/appearance/background/asset?v=2"));

    const appearance = useAppearance();
    await appearance.load();
    await appearance.load();

    expect(revoked).toEqual(["blob:stub-1"]);
    expect(appearance.objectUrl.value).toBe("blob:stub-2");
  });

  it("取字节失败不抛错，只是留空让渲染层退回兜底", async () => {
    vi.spyOn(api, "fetchBackgroundAsset").mockRejectedValue(new Error("401"));
    vi.spyOn(api, "getBackground").mockResolvedValue(
      backgroundPayload("/api/appearance/background/asset?v=1")
    );

    const appearance = useAppearance();
    await expect(appearance.load()).resolves.toBeUndefined();

    expect(appearance.objectUrl.value).toBe("");
    expect(appearance.background.value).not.toBeNull();
  });

  it("清空背景时释放 object URL", async () => {
    vi.spyOn(api, "fetchBackgroundAsset").mockResolvedValue(
      new Blob(["x"], { type: "image/jpeg" })
    );
    vi.spyOn(api, "getBackground").mockResolvedValue(
      backgroundPayload("/api/appearance/background/asset?v=1")
    );
    vi.spyOn(api, "clearBackground").mockResolvedValue({});

    const appearance = useAppearance();
    await appearance.load();
    await appearance.clear();

    expect(revoked).toEqual(["blob:stub-1"]);
    expect(appearance.objectUrl.value).toBe("");
  });

  it("作用域销毁时释放 object URL，不留悬挂引用", async () => {
    vi.spyOn(api, "fetchBackgroundAsset").mockResolvedValue(
      new Blob(["x"], { type: "image/jpeg" })
    );
    vi.spyOn(api, "getBackground").mockResolvedValue(
      backgroundPayload("/api/appearance/background/asset?v=1")
    );

    const scope = effectScope();
    let appearance;
    scope.run(() => {
      appearance = useAppearance();
    });
    await appearance.load();
    scope.stop();

    expect(revoked).toEqual(["blob:stub-1"]);
  });
});
