"""只通过绑定的窄只读端口回查结果；模型无法指定 Session/Run。"""

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from personagraph.persistent_turn_content.tool_results import (
    ToolHistoryError,
    ToolHistoryReadPort,
    project_tool_result_content,
)
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..execution import ToolBusinessFailure
from ..policy import AuthorityFacts, ScopeGrant
from ..catalog.binding import ToolBinding
from ..registration import ToolRegistration
from ...persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ..model_interface.results import project_tool_result
from .catalog import bind_tool_history_registrations
from .definitions import (
    TOOL_HISTORY_TOOL_IDS,
    ListToolResultsInput,
    ReadToolResultInput,
    build_tool_history_registration,
)


@dataclass(frozen=True, slots=True)
class ToolHistoryRuntime:
    port: ToolHistoryReadPort
    effect_scope: str

    def __post_init__(self) -> None:
        if not self.effect_scope.strip() or self.effect_scope == "*":
            raise ValueError("tool history runtime requires an exact execution scope")

    @property
    def registrations(self) -> tuple[ToolRegistration, ...]:
        return tuple(
            build_tool_history_registration(
                tool_id=tool_id,
                effect_scope=self.effect_scope,
                handler=self.list_results
                if tool_id == "list_tool_results"
                else self.read_result,
            )
            for tool_id in TOOL_HISTORY_TOOL_IDS
        )

    @property
    def bindings(self) -> tuple[ToolBinding, ...]:
        return bind_tool_history_registrations(
            self.registrations, authority_sha256=self.port.authority_sha256
        )

    @property
    def authority(self) -> AuthorityFacts:
        return AuthorityFacts(
            grants=(
                ScopeGrant(
                    EffectResource.RUNTIME_STATE,
                    EffectAction.READ,
                    EffectScopeKind.EXECUTION,
                    self.effect_scope,
                ),
            )
        )

    def list_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._read(payload, ListToolResultsInput, self.port.list_results)

    def read_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        def read(**arguments):
            record = self.port.read_result(call_ref=arguments.pop("call_ref"))
            content = {
                "result": project_tool_result(record.source.tool_id, record.outcome.get("result")),
                "error": record.outcome.get("error"),
            }
            page = project_tool_result_content(source=record.source, content=content, **arguments)
            if record.source.tool_id in EXECUTION_FINDINGS_TOOL_IDS and content["result"] is not None:
                page = page.model_copy(update={"partial": True, "source_content_compacted": True})
            return page

        return self._read(payload, ReadToolResultInput, read)

    @staticmethod
    def _read(
        payload: dict[str, Any],
        contract: type[BaseModel],
        read: Callable[..., BaseModel],
    ) -> dict[str, Any]:
        try:
            request = contract.model_validate(payload)
        except ValidationError:
            raise ToolBusinessFailure(
                "invalid_history_request",
                "历史读取参数无效；仅可使用当前工具声明的字段和范围。",
            ) from None
        try:
            return read(**request.model_dump()).model_dump(mode="json")
        except ToolHistoryError as exc:
            raise ToolBusinessFailure(str(exc), str(exc)) from None
