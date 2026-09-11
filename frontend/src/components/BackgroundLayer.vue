<script setup>
import { computed } from "vue";

const props = defineProps({
  // 后端返回的 { kind, media_type, url }，没设置背景时为 null
  background: { type: Object, default: null },
  // 已鉴权取回的 object URL；见 useAppearance
  objectUrl: { type: String, default: "" }
});

// 只认 object URL。<img>/<video> 的 src 带不上鉴权——图片元素拿不到 Electron 在网络层
// 注入的 token，浏览器也不会为跨源媒体请求带 Origin——所以直接引用后端地址只会换来一次
// 401，再被 ORB 拦掉。字节没到位时就不渲染，由 body::before 的渐变兜底。
const src = computed(() => props.objectUrl);
const isVideo = computed(() => props.background?.kind === "video");
</script>

<template>
  <!--
    玻璃底层。

    毛玻璃糊的是它背后的东西，而渐变几乎没有细节可糊——这就是之前"看不出磨砂"的
    原因。换成照片或视频之后，模糊才有真正的结构可以推开。

    没设置背景时这里什么都不渲染，body::before 的渐变继续兜底。

    视频要静音且 playsinline：带声音的自动播放会被浏览器直接拦掉，整块背景就黑了。
  -->
  <div v-if="src" class="bg-layer" aria-hidden="true">
    <video
      v-if="isVideo"
      :src="src"
      autoplay
      muted
      loop
      playsinline
      disablepictureinpicture
    />
    <img v-else :src="src" alt="" />
  </div>
</template>

<style scoped>
/* 默认铺满视口；给了 --bg-* 就贴到那一块去（见 useBackgroundBox）。
 *
 * 用 fixed + 变量而不是把层塞进纸面里：纸面的玻璃靠 backdrop-filter 糊背后的东西，
 * 而子元素在它前面，糊不到。层必须留在纸面外面、位置上和它对齐。 */
.bg-layer {
  position: fixed;
  top: var(--bg-top, 0px);
  left: var(--bg-left, 0px);
  width: var(--bg-width, 100vw);
  height: var(--bg-height, 100vh);
  z-index: -1;
  overflow: hidden;
  border-radius: var(--bg-radius, 0px);
  pointer-events: none;
}

.bg-layer video,
.bg-layer img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  /* 背景本身先压暗一点点：原图对比度太高时，压在玻璃后面会把上层文字顶得没法看。
   * 真正的可读性还是靠玻璃那层的白度，这里只是给它留出余量。 */
  filter: saturate(1.05) brightness(1.02);
}
</style>
