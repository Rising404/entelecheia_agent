"""运行通知只描述确定性终止事实，不冒充已审查的任务答案。"""

import pytest

from personagraph.persistent_turn_content.delivery import build_l1_terminal_notification


@pytest.mark.parametrize("code", [
    "VERIFICATION_FAILED", "MODEL_OUTPUT_INVALID", "TURN_DEADLINE_EXCEEDED",
    "MODEL_CALL_TIMEOUT", "MODEL_BAD_RESPONSE", "MODEL_CALL_FAILED",
    "CONTEXT_BUDGET_EXCEEDED",
])
def test_known_terminal_failure_has_stable_nonempty_notice(code):
    notice = build_l1_terminal_notification(code)
    assert notice is not None
    assert notice.failure_code == code
    assert notice.reply.startswith("本轮未完成")
    assert notice == build_l1_terminal_notification(code)


@pytest.mark.parametrize("code", ["INTERNAL_FAILURE", "PERSIST_FAILED", "CANCELLED", "secret-stack"])
def test_unknown_or_unsafe_terminal_state_is_not_publishable(code):
    assert build_l1_terminal_notification(code) is None


def test_notice_maps_provider_code_without_exposing_raw_diagnostics():
    notice = build_l1_terminal_notification("MODEL_CALL_TIMEOUT")
    assert notice.error_code == "MODEL_TIMEOUT"
    assert "MODEL_CALL_TIMEOUT" not in notice.reply


@pytest.mark.parametrize("code", ["VERIFICATION_FAILED", "MODEL_OUTPUT_INVALID", "MODEL_BAD_RESPONSE"])
def test_error_code_alone_does_not_claim_attempt_or_repair_budget_exhaustion(code):
    notice = build_l1_terminal_notification(code)
    assert notice is not None
    assert "上限" not in notice.reply
    assert "额度" not in notice.reply
    assert "用尽" not in notice.reply
