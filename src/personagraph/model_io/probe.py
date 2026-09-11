from __future__ import annotations

import json

from .gateway import (
    DEFAULT_MODEL_TIMEOUT_S,
    ModelGatewayError,
    anthropic_compatible_chat,
    openai_compatible_chat,
)
from .dialects import resolve_request_dialect
from ..configuration.app_settings import get_setting, resolve_global_model_endpoint
from .capabilities import record_native_probe
from .contracts import ToolCallBatch


PROBE_TOOL = "personagraph_native_tool_probe"


def run_native_tool_probe() -> dict:
    endpoint = resolve_global_model_endpoint()
    provider = endpoint.provider
    base_url = endpoint.base_url
    model = endpoint.model
    request_dialect = resolve_request_dialect(
        provider,
        base_url,
        get_setting("request_dialect", "auto"),
        legacy_provider=endpoint.raw_provider,
    ).value
    passed = False
    evidence: dict = {"request_sent": False, "observed_kind": None, "provider_call_id": False}
    try:
        gateway = (
            openai_compatible_chat
            if provider == "openai-compatible"
            else anthropic_compatible_chat
        )
        result = gateway(
            [{"role": "user", "content": "Run the declared capability probe tool exactly once with nonce r1."}],
            temperature=0.0,
            max_tokens=128,
            timeout_s=DEFAULT_MODEL_TIMEOUT_S,
            tools=[{
                "id": PROBE_TOOL,
                "description": "A no-side-effect protocol capability probe.",
                "input_schema": {
                    "type": "object",
                    "properties": {"nonce": {"type": "string"}},
                    "required": ["nonce"],
                },
            }],
            control_transport="native",
            tool_choice={"type": "tool", "name": PROBE_TOOL},
            purpose="capability_probe",
        )
        evidence["request_sent"] = True
        evidence["observed_kind"] = result.output.kind if result.output else None
        if isinstance(result.output, ToolCallBatch) and len(result.output.calls) == 1:
            call = result.output.calls[0]
            passed = call.tool_name == PROBE_TOOL and call.arguments.get("nonce") == "r1"
            evidence["provider_call_id"] = bool(call.call_id)
            evidence["argument_round_trip"] = call.arguments.get("nonce") == "r1"
    except ModelGatewayError as exc:
        evidence.update({
            "request_sent": True,
            "error_code": exc.code,
            "status_code": exc.details.get("status_code"),
            "exception_type": exc.details.get("exception_type"),
            "retryable": exc.retryable,
        })
    record = record_native_probe(
        provider=provider,
        base_url=base_url,
        model=model,
        request_dialect=request_dialect,
        passed=passed,
        evidence=evidence,
    )
    return {
        "passed": passed,
        "provider": provider,
        "endpoint_host": record["endpoint_host"],
        "model": model,
        "request_dialect": request_dialect,
        "verified_at": record["verified_at"],
        "evidence": evidence,
    }


def main() -> None:
    result = run_native_tool_probe()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
