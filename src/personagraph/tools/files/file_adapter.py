"""将文件访问和中性 ingestion 服务投影到两个批量工具，不直接访问存储。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import time

from personagraph.workspace.files.access import AuthorizedFileSource, FileAccess, FileAccessError
from personagraph.workspace.ingestion.contracts import FilePreparationResult, FilePreparationStatus
from ..execution import ToolBusinessFailure
from ..execution_context import current_tool_execution
from .file_tools import (
    CHECK_FILES_STATE_TOOL_ID, PREPARE_FILES_TOOL_ID, FILE_TOOLS_CONTRACT_VERSION,
    MAX_FILES, build_file_state_registration, derive_file_preparation_tool_request_id,
)


@dataclass(frozen=True, slots=True)
class FileStateRuntime:
    access: FileAccess
    effect_scope: str
    check_state: Callable[[AuthorizedFileSource], FilePreparationResult]
    prepare: Callable[..., FilePreparationResult]

    @property
    def registrations(self):
        return (
            build_file_state_registration(
                tool_id=CHECK_FILES_STATE_TOOL_ID, handler=self.check_files_state,
                effect_scope=self.effect_scope,
            ),
            build_file_state_registration(
                tool_id=PREPARE_FILES_TOOL_ID, handler=self.prepare_files,
                effect_scope=self.effect_scope,
            ),
        )

    def check_files_state(self, payload: dict) -> dict:
        return self._run(payload, self.check_state)

    def prepare_files(self, payload: dict) -> dict:
        return self._run(payload, self.prepare, await_preparation=True)

    def _run(self, payload: dict, operation: Callable, *, await_preparation: bool = False) -> dict:
        targets = _targets(payload)
        control = current_tool_execution() if await_preparation else None
        prepared = []
        for index, target in enumerate(targets):
            source = None
            kwargs = {}
            if control is not None:
                control.checkpoint()
                kwargs["checkpoint"] = control.checkpoint
                if control.logical_tool_call_id:
                    # 同一持久调用的同一输入项绑定原 request；恢复不得选择新的源版本。
                    kwargs["request_id"] = derive_file_preparation_tool_request_id(
                        control.logical_tool_call_id, index,
                    )
            try:
                if "path" in target:
                    source = self.access.resolve_path(target["path"])
                    if target.get("file_version_id") not in {None, source.file_version_id}:
                        raise FileAccessError("file_version_changed")
                else:
                    source = self.access.resolve_file(
                        file_id=target["file_id"], file_version_id=target.get("file_version_id"),
                        allow_changed=True,
                    )
                state = _invoke(operation, source, **kwargs)
            except FileAccessError as exc:
                state = _access_failure(exc)
            prepared.append((source, state, kwargs))
        if await_preparation:
            # 先提交整批，不逐文件等待后台；离线 owner 仍可沿用同步写后读。
            # 仅复查已有任务，沿用第一次冻结来源，不重新选择路径的最新版本。
            for index, (source, state, kwargs) in enumerate(prepared):
                if source is None or state.status is not FilePreparationStatus.PENDING or not state.operation_id:
                    continue
                if control is None or control.deadline_monotonic is None:
                    raise ToolBusinessFailure(
                        "file_preparation_deadline_required",
                        "Waiting for file preparation requires the Host execution deadline.",
                    )
                control.checkpoint()
                settled = _invoke(
                    self.prepare, source,
                    pending_wait_seconds=max(0.0, control.deadline_monotonic - time.monotonic()),
                    **kwargs,
                )
                control.checkpoint()
                # 等待观察本身的幂等重入，不改变首次提交是否复用任务的语义。
                prepared[index] = (source, replace(settled, replayed=state.replayed), kwargs)
        results = [_project(index, source, state) for index, (source, state, _) in enumerate(prepared)]
        return {
            "contract_version": FILE_TOOLS_CONTRACT_VERSION, "results": results,
            "ready_indices": [item["input_index"] for item in results if item["status"] == "ready"],
            "not_ready_indices": [item["input_index"] for item in results if item["status"] != "ready"],
        }


def _invoke(operation, source, **kwargs) -> FilePreparationResult:
    try:
        state = operation(source, **kwargs)
    except FileAccessError as exc:
        return _access_failure(exc)
    if not isinstance(state, FilePreparationResult):
        raise TypeError("file service returned an invalid result")
    return state


def _access_failure(error: FileAccessError) -> FilePreparationResult:
    return FilePreparationResult(
        status=(FilePreparationStatus.STALE if error.reason_code in {
            "file_version_changed", "file_content_changed",
        } else FilePreparationStatus.BLOCKED), reason_code=error.reason_code,
    )


def _project(index, source, state):
    status = {
        FilePreparationStatus.READY: "ready", FilePreparationStatus.STALE: "changed",
        FilePreparationStatus.BLOCKED: "unavailable", FilePreparationStatus.PENDING: "pending",
    }[state.status]
    if state.reason_code in {"file_not_prepared", "image_not_prepared"}:
        status = "not_ingested"
    elif state.reason_code in {"file_mount_required", "image_visual_ready"}:
        status = "partial"
    return {
        "input_index": index, "status": status,
        "file_id": state.file_id or (source.file_id if source else None),
        "file_version_id": state.file_version_id or (source.file_version_id if source else None),
        "file_name": source.file_name if source else None,
        "relative_path": source.relative_path if source else None,
        "document_id": state.document_id, "document_version_id": state.document_version_id,
        "reason_code": state.reason_code, "reused": state.replayed,
    }


def _targets(payload):
    targets = payload.get("files")
    if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_FILES:
        raise ToolBusinessFailure("invalid_files", "files must contain 1-64 path or File ID targets.")
    for target in targets:
        if (
            not isinstance(target, Mapping)
            or set(target) - {"path", "file_id", "file_version_id"}
            or (("path" in target) == ("file_id" in target))
            or any(not isinstance(value, str) or not value.strip() for value in target.values())
        ):
            raise ToolBusinessFailure("invalid_file_target", "Provide exactly one path or file_id per target.")
    return targets
