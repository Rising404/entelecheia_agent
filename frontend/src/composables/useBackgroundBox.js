import { onBeforeUnmount, onMounted, watch } from "vue";

// 背景该铺在哪一块。
//
// 之前是铺满整个视口，结果左边四分之一被不透明的侧栏盖死——等于选了张图然后裁掉
// 一块。这里把目标区域的实际位置量出来写成 CSS 变量，让背景层自己贴上去。
//
// 量而不是算：侧栏能收起、窗口能缩放、780px 以下还会变成上下堆叠。把布局规则在
// 这里抄一遍，迟早会和真实布局对不上；量出来的框永远是对的。
export function useBackgroundBox(scope) {
  let frame = 0;
  let observer = null;

  function apply() {
    const root = document.documentElement;
    if (scope.value === "window") {
      for (const name of ["top", "left", "width", "height"]) {
        root.style.removeProperty(`--bg-${name}`);
      }
      return;
    }

    const sheet = document.querySelector(".sheet");
    if (!sheet) return;
    const box = sheet.getBoundingClientRect();
    let bottom = box.bottom;

    if (scope.value === "content-no-composer") {
      // 输入区在纸面底部。不含它的时候，背景到它的上沿为止。
      const composer = sheet.querySelector(".composer");
      if (composer) {
        const composerBox = composer.getBoundingClientRect();
        if (composerBox.height > 0) bottom = composerBox.top - 10;
      }
    }

    root.style.setProperty("--bg-left", `${Math.round(box.left)}px`);
    root.style.setProperty("--bg-top", `${Math.round(box.top)}px`);
    root.style.setProperty("--bg-width", `${Math.round(box.width)}px`);
    root.style.setProperty("--bg-height", `${Math.round(Math.max(0, bottom - box.top))}px`);
  }

  function schedule() {
    // 合并连发：一次窗口缩放会打出很多个回调。
    //
    // 不用 requestAnimationFrame——页面不可见时它根本不触发，尺寸变化会全部积压，
    // 切回前台才补上，期间背景框停在旧位置。这里用微任务，和可见性无关。
    if (frame) return;
    frame = 1;
    queueMicrotask(() => {
      frame = 0;
      apply();
    });
  }

  onMounted(() => {
    apply();
    observer = new ResizeObserver(schedule);
    const sheet = document.querySelector(".sheet");
    if (sheet) observer.observe(sheet);
    // 内容变化会顶动输入区的位置，纸面本身的尺寸却不一定变
    const app = document.querySelector("#app");
    if (app) observer.observe(app);
    globalThis.addEventListener?.("resize", schedule);
  });

  onBeforeUnmount(() => {
    observer?.disconnect();
    globalThis.removeEventListener?.("resize", schedule);
    frame = 0;
  });

  watch(scope, schedule);

  return { refresh: schedule };
}
