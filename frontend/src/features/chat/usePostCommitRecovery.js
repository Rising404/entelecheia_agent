import { computed, ref } from "vue";
import { postCommitFailureLabel } from "./presentation";

/** 显式恢复失败的后台任务；确认与写入属于动作层，GET 只刷新服务端权威状态。 */
export function usePostCommitRecovery({
  api, selectedSessionId, sessionDetail, saving, chatBusy, attachmentUploading,
  notice, clearMessage, showError, confirmWaiver = () => false,
  createRequestId = () => globalThis.crypto.randomUUID()
}) {
  const postCommitRecoveryBusy = ref(false);
  const postCommitFailure = computed(() => {
    const detail = sessionDetail.value;
    if (!selectedSessionId.value || detail?.session?.id !== selectedSessionId.value) return null;
    const state = detail?.post_commit;
    const window = detail?.turn_window;
    if (
      window?.window_state !== "post_commit_pending"
      || !state?.turn_id
      || window.turn_id !== state.turn_id
      || !Number.isInteger(state.window_revision) || state.window_revision < 1
      || window.window_revision !== state.window_revision
      || typeof state.failed_job_digest !== "string"
      || !/^[0-9a-f]{64}$/.test(state.failed_job_digest)
      || !Array.isArray(state.failed_jobs) || !state.failed_jobs.length
      || state.failed_jobs.some((job) => typeof job?.job_id !== "string" || !job.job_id)
    ) return null;
    return {
      sessionId: detail.session.id, turnId: state.turn_id,
      revision: state.window_revision, digest: state.failed_job_digest,
      jobIds: state.failed_jobs.map((job) => job.job_id),
      label: postCommitFailureLabel(state.failed_jobs)
    };
  });
  const otherwiseBusy = () => saving.value || chatBusy.value || attachmentUploading.value;
  const canRecoverPostCommit = computed(() => Boolean(
    postCommitFailure.value && sessionDetail.value?.session?.status === "active"
    && !otherwiseBusy() && !postCommitRecoveryBusy.value
    && typeof api.controlTurnPostCommitJobs === "function"
  ));

  function ownsSession(id) {
    return selectedSessionId.value === id && sessionDetail.value?.session?.id === id;
  }

  async function refreshSession(id) {
    if (!ownsSession(id)) return;
    const detail = await api.getSession(id);
    if (!ownsSession(id) || detail?.session?.id !== id) return;
    const currentRevision = sessionDetail.value?.turn_window?.window_revision;
    const receivedRevision = detail.turn_window?.window_revision;
    // 另一次权威刷新可能先完成；晚到的 GET 不应重新锁住已释放的窗口。
    if (Number.isInteger(currentRevision) && Number.isInteger(receivedRevision)
      && receivedRevision < currentRevision) return;
    sessionDetail.value = detail;
  }

  function stillCurrent(snapshot) {
    const current = postCommitFailure.value;
    return current && !otherwiseBusy() && sessionDetail.value?.session?.status === "active"
      && JSON.stringify(current) === JSON.stringify(snapshot);
  }

  async function control(action) {
    if (!canRecoverPostCommit.value) return;
    const snapshot = postCommitFailure.value;
    postCommitRecoveryBusy.value = true;
    try {
      if (action === "waive") {
        const confirmed = await confirmWaiver(
          `确认跳过本轮失败的${snapshot.label}更新？后续上下文或历史检索可能不完整；不会删除已保存的答复。`
        );
        if (confirmed !== true) return;
      }
      // 确认对话框可能停留很久；会话或失败集合变化后，旧确认绝不能挪作新授权。
      if (!stillCurrent(snapshot)) return;
      clearMessage();
      const command = {
        turn_id: snapshot.turnId, request_id: createRequestId(), action,
        expected_window_revision: snapshot.revision,
        expected_failed_job_digest: snapshot.digest, job_ids: snapshot.jobIds
      };
      if (action === "waive") command.confirm_stale = true;
      await api.controlTurnPostCommitJobs(snapshot.sessionId, command);
      await refreshSession(snapshot.sessionId);
      if (ownsSession(snapshot.sessionId)) {
        notice.value = action === "retry" ? "已提交失败任务重试" : "已确认跳过失败任务";
      }
    } catch (error) {
      // 包括状态冲突和响应丢失：只重新读取，绝不推断写入失败后自动重复命令。
      if (ownsSession(snapshot.sessionId)) {
        try { await refreshSession(snapshot.sessionId); } catch { /* 保留原操作错误。 */ }
        if (ownsSession(snapshot.sessionId)) showError(error);
      }
    } finally {
      postCommitRecoveryBusy.value = false;
    }
  }

  return {
    postCommitFailure, postCommitRecoveryBusy, canRecoverPostCommit,
    retryPostCommit: () => control("retry"),
    waivePostCommit: () => control("waive")
  };
}
