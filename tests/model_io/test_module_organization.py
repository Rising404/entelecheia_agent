from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


def test_retired_model_facades_are_absent() -> None:
    for retired_name in (
        "personagraph.output_language",
        "personagraph.model_io.profiles",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(retired_name)


def test_gateway_implementation_is_split_by_behavioral_seam() -> None:
    from personagraph.model_io import gateway

    assert gateway.ModelResult.__module__ == "personagraph.model_io.contracts"
    assert gateway.chat.__module__ == "personagraph.model_io.gateway"
    assert (
        gateway.complete_structured.__module__
        == "personagraph.model_io.structured_calls"
    )
    assert (
        gateway.anthropic_compatible_chat.__module__
        == "personagraph.model_io.provider_anthropic"
    )
    assert (
        gateway.openai_compatible_chat.__module__
        == "personagraph.model_io.provider_openai"
    )
    assert gateway.complete_structured.prepare is gateway.prepare_complete_structured


def test_structured_output_repair_is_owned_by_model_io() -> None:
    from personagraph.model_io import output_repair_contracts
    from personagraph.model_io import prepared_structured_provider
    from personagraph.model_io import structured_output_repair

    assert output_repair_contracts.RuntimeModelOutputRepairFeedback.__module__ == (
        "personagraph.model_io.output_repair_contracts"
    )
    assert prepared_structured_provider.prepare_structured_request.__module__ == (
        "personagraph.model_io.prepared_structured_provider"
    )
    assert structured_output_repair.project_validation_error_issues.__module__ == (
        "personagraph.model_io.structured_output_repair"
    )


def test_model_request_seams_are_owned_by_model_io_without_runtime_back_edges() -> None:
    from personagraph.model_io import output_validation
    from personagraph.model_io import prepared_request_contracts
    from personagraph.runtime.model_calls import requests as model_requests

    assert output_validation.ModelOutputValidationError.__module__ == (
        "personagraph.model_io.output_validation"
    )
    assert output_validation.build_model_output_repair_feedback.__module__ == (
        "personagraph.model_io.output_validation"
    )
    assert prepared_request_contracts.PreparedModelRequest.__module__ == (
        "personagraph.model_io.prepared_request_contracts"
    )
    assert prepared_request_contracts.QuotaPreparedModelRequest.__module__ == (
        "personagraph.model_io.prepared_request_contracts"
    )
    for module in (output_validation, prepared_request_contracts):
        module_path = Path(str(module.__file__))
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("personagraph.runtime")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                assert not (
                    imported_module.startswith("personagraph.runtime")
                    or (node.level > 0 and imported_module == "runtime")
                    or (node.level > 0 and imported_module.startswith("runtime."))
                )

    assert not hasattr(model_requests, "PreparedModelRequest")
    assert not hasattr(model_requests, "QuotaPreparedModelRequest")
    assert not hasattr(model_requests, "ModelOutputValidationError")
