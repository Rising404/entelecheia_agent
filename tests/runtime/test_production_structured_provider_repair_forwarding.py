from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import pytest

from personagraph.l2.auxiliary_execution.planning import model_provider as planning
from personagraph.l2.auxiliary_execution.verification import model_provider as semantic
from personagraph.l2.task_execution.delivery import model_provider as delivery
from personagraph.l2.task_execution.work_run import model_providers as work_run
from personagraph.l2.task_execution.work_run.model_profile import (
    WorkRunStructuredModelProfile,
)


_REPAIR_MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "user"},
    {"role": "assistant", "content": '{"bad":true}'},
    {"role": "user", "content": "请重新生成完整 JSON。"},
]


def _assert_repair_messages_reach_gateway(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: ModuleType,
    provider_factory: Callable[[], object],
    mock_builder_name: str,
) -> None:
    captured: list[dict[str, object]] = []
    sentinel = object()

    monkeypatch.setattr(
        module,
        "effective_model_tier_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        mock_builder_name,
        lambda *_args, **_kwargs: {"mock": True},
    )

    def prepare_gateway(
        _system_prompt: str,
        _user_content: str,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, "prepare_complete_structured", prepare_gateway)
    provider = provider_factory()
    prepare = getattr(provider, "prepare")

    assert (
        prepare(
            "system",
            "user",
            purpose="repair-forwarding-test",
            repair_messages=_REPAIR_MESSAGES,
        )
        is sentinel
    )
    assert len(captured) == 1
    assert captured[0]["repair_messages"] is _REPAIR_MESSAGES


@pytest.mark.parametrize(
    ("module", "provider_factory", "mock_builder_name"),
    [
        (
            work_run,
            lambda: work_run.build_attempt_structured_provider(
                WorkRunStructuredModelProfile(
                    attempt_max_output_tokens=64,
                    verification_max_output_tokens=64,
                    timeout_s=10.0,
                )
            ),
            "_mock_attempt_decision",
        ),
        (
            work_run,
            lambda: work_run.build_verification_structured_provider(
                WorkRunStructuredModelProfile(
                    attempt_max_output_tokens=64,
                    verification_max_output_tokens=64,
                    timeout_s=10.0,
                )
            ),
            "_mock_verification",
        ),
        (
            planning,
            planning.build_auxiliary_architect_structured_provider,
            "build_mock_auxiliary_architect_proposal",
        ),
        (
            semantic,
            semantic.build_auxiliary_semantic_structured_provider,
            "build_mock_auxiliary_semantic_result",
        ),
        (
            delivery,
            delivery.build_task_delivery_validation_structured_provider,
            "_mock_pass",
        ),
    ],
)
def test_production_structured_provider_forwards_repair_messages(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    provider_factory: Callable[[], object],
    mock_builder_name: str,
) -> None:
    _assert_repair_messages_reach_gateway(
        monkeypatch,
        module=module,
        provider_factory=provider_factory,
        mock_builder_name=mock_builder_name,
    )
