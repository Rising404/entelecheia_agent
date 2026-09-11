"""Model-visible schema for the safe visual provider failure diagnostics."""


def failure_diagnostics_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "phase": {"enum": ["request", "response_decode", "response_validate"]},
            "exception_type": {"type": ["string", "null"]},
            "cause_type": {"type": ["string", "null"]},
            "http_status": {"type": ["integer", "null"], "minimum": 100, "maximum": 599},
            "retry_after_s": {"type": ["number", "null"], "minimum": 0},
            "elapsed_ms": {"type": "integer", "minimum": 0},
            "timeout_s": {"type": "number", "exclusiveMinimum": 0},
            "completion_uncertain": {"type": "boolean"},
        },
    }
