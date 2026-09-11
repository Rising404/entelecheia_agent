import { computed, ref } from "vue";
import { api } from "../api";

/** 系统连接状态 + 模型是否已配置（供设置/横幅/连接指示消费）。 */
export function useSystemStatus() {
  const systemStatus = ref(null);
  const statusLoaded = ref(false);
  const apiReachable = ref(false);
  const apiOk = computed(() => systemStatus.value?.ok === true);
  const modelConfigured = computed(() => systemStatus.value?.model_configured === true);

  async function loadStatus() {
    try {
      systemStatus.value = await api.status();
      apiReachable.value = true;
    } catch (err) {
      apiReachable.value = false;
      systemStatus.value = {
        ok: false,
        service: "personagraph-api",
        components: {
          api: { ok: false, label: "HTTP API", detail: err.message }
        }
      };
    } finally {
      statusLoaded.value = true;
    }
  }

  return {
    systemStatus,
    statusLoaded,
    apiReachable,
    apiOk,
    modelConfigured,
    loadStatus
  };
}
