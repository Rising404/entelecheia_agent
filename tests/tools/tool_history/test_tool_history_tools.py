"""工具历史只通过注入的绑定端口读取，模型无法选择 owner。"""

from types import SimpleNamespace

import pytest

from personagraph.persistent_turn_content.tool_results import (
    ToolResultHistoryPage,
    ToolHistoryError,
)
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.schema_validation import ToolSchemaCompiler
from personagraph.tools.tool_history import ToolHistoryRuntime, TOOL_HISTORY_TOOL_IDS


def _runtime(**overrides):
    calls = []
    port = SimpleNamespace(
        authority_sha256="a" * 64,
        list_results=lambda **kw: (
            calls.append(kw)
            or ToolResultHistoryPage(
                results=(),
                offset=kw["offset"],
                total_results=0,
                next_offset=None,
                partial=False,
            )
        ),
        read_result=lambda **kw: (_ for _ in ()).throw(
            ToolHistoryError("tool_result_unavailable")
        ),
        **overrides,
    )
    return ToolHistoryRuntime(port=port, effect_scope="l1-run:a"), calls


def test_history_tools_use_bound_port_and_stable_catalog_definitions():
    runtime, calls = _runtime()
    assert tuple(reg.tool_id for reg in runtime.registrations) == TOOL_HISTORY_TOOL_IDS
    result = runtime.registrations[0].handler({})
    assert calls == [{"offset": 0, "limit": 20}]
    assert result["results"] == []
    seeds = {
        seed.definition.identity.tool_id: seed.definition
        for seed in build_production_default_catalog_seeds()
    }
    for registration, binding in zip(
        runtime.registrations, runtime.bindings, strict=True
    ):
        definition = seeds[registration.tool_id]
        assert (
            BoundToolRegistration(definition, binding).descriptor()
            == registration.descriptor()
        )
        ToolSchemaCompiler().compile(
            registration.spec.output_schema, role="output"
        ).validate(
            result
            if registration.tool_id == "list_tool_results"
            else {
                "contract_version": "tool-result-content-page-v1",
                "source": {
                    "tool_call_id": "call",
                    "tool_id": "read_pdf_text",
                    "status": "failed",
                    "tool_result_id": "l1result_" + "b" * 64,
                    "result_sha256": "a" * 64,
                },
                "path": "", "kind": "object", "value": {}, "expand_paths": [],
                "offset": 0,
                "total_items": 0,
                "next_offset": None,
                "partial": False,
            },
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "other"},
        {"l1_turn_run_id": "other"},
        {"limit": 101},
        {"offset": True},
    ],
)
def test_history_owner_and_invalid_list_parameters_never_reach_port(payload):
    runtime, calls = _runtime()
    with pytest.raises(ToolBusinessFailure):
        runtime.registrations[0].handler(payload)
    assert calls == []


def test_unknown_history_result_has_public_bounded_failure():
    runtime, _ = _runtime()
    with pytest.raises(ToolBusinessFailure, match="tool_result_unavailable"):
        runtime.registrations[1].handler({"tool_result_id": "l1result_" + "b" * 64})


@pytest.mark.parametrize("effect_scope", ["", " ", "*"])
def test_history_runtime_rejects_unbound_effect_scope(effect_scope):
    runtime, _ = _runtime()
    with pytest.raises(ValueError, match="exact execution scope"):
        ToolHistoryRuntime(port=runtime.port, effect_scope=effect_scope)
