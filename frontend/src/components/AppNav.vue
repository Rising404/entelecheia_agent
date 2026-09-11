<script setup>

defineProps({
  title: { type: String, required: true }
});
</script>

<template>
  <header class="mb-4">
    <div>
      <p class="text-xs font-semibold uppercase tracking-[0.18em] text-ink-3">Entelecheia</p>
      <h1 class="text-2xl font-semibold">{{ title }}</h1>
    </div>
  </header>


</template>
