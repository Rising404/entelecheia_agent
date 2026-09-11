"""Runtime Turn linkage for generic model and tool trajectory steps."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from personagraph.model_io import gateway as models
from personagraph.trajectory import StepKind, StepOutcome, TrajectoryStore
from personagraph.trajectory import recorder as recorder_module
from personagraph.trajectory.scope import (
    current_turn_linkage,
    resolve_turn_linkage,
    turn_linkage_scope,
)


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "stop_reason": "end_turn",
        }


class _FakeClient:
    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def post(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse()


@pytest.fixture
def store(tmp_path, monkeypatch) -> TrajectoryStore:
    from personagraph.trajectory import store as store_module

    isolated = TrajectoryStore(tmp_path / "trajectory.sqlite")
    monkeypatch.setattr(store_module, "_ACTIVE", isolated)
    return isolated


def _configure_provider(monkeypatch) -> None:
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")
    monkeypatch.setattr(models.httpx, "Client", lambda *_a, **_k: _FakeClient())


def _tool_registration():
    from personagraph.tools.contracts import (
        ToolSourceDescriptor,
        ToolSourceKind,
        ToolSpec,
    )
    from personagraph.tools.effects import (
        EffectAction,
        EffectDescriptor,
        EffectResource,
        EffectScopeKind,
        ToolEffectProfile,
    )
    from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration

    return ToolRegistration(
        spec=ToolSpec(
            tool_id="scope_probe",
            contract_version="1.0.0",
            name="scope_probe",
            description="A trajectory linkage probe.",
            input_schema={"type": "object", "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            catalog_tags=("read",),
        ),
        implementation_version="impl-1",
        source=ToolSourceDescriptor(ToolSourceKind.LOCAL, "test"),
        handler=lambda _payload: {"ok": True},
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.MEMORY,
                    EffectAction.READ,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(),
    )


def test_generic_provider_and_tool_steps_inherit_the_active_turn(
    store,
    monkeypatch,
) -> None:
    from personagraph.tools.execution import ResolvedInvocation, ToolExecutor

    _configure_provider(monkeypatch)
    with turn_linkage_scope(session_id="session-1", turn_id="turn-1"):
        model_result = models.anthropic_compatible_chat(
            [{"role": "user", "content": "probe"}],
            model_call_id="model-1",
        )
        ToolExecutor().execute(
            ResolvedInvocation(registration=_tool_registration(), arguments={})
        )

    model_step = store.steps_for_model_call(model_result.model_call_id)[0]
    tool_step = next(
        step
        for step in store.steps_for_turn("turn-1")
        if step.kind is StepKind.TOOL_CALL
    )
    assert (model_step.session_id, model_step.turn_id) == ("session-1", "turn-1")
    assert (tool_step.session_id, tool_step.turn_id) == ("session-1", "turn-1")
    assert current_turn_linkage() is None


def test_complete_explicit_linkage_cannot_override_the_active_turn(store) -> None:
    with turn_linkage_scope(session_id="scoped-session", turn_id="scoped-turn"):
        recorder_module.record_tool_call(
            tool_id="explicit_probe",
            arguments={},
            status="succeeded",
            result={"ok": True},
            error_code=None,
            duration_ms=1,
            session_id="explicit-session",
            turn_id="explicit-turn",
            store=store,
        )

    assert store.steps_for_session("explicit-session") == ()
    steps = store.steps_for_turn("scoped-turn")
    assert len(steps) == 1
    assert steps[0].kind is StepKind.RECORDING_FAILURE
    assert steps[0].outcome is StepOutcome.FAILED
    assert steps[0].purpose == "record_tool_call"
    assert steps[0].parts == ()


def test_explicit_linkage_matching_the_active_turn_is_allowed(store) -> None:
    with turn_linkage_scope(session_id="scoped-session", turn_id="scoped-turn"):
        recorder_module.record_tool_call(
            tool_id="explicit_probe",
            arguments={},
            status="succeeded",
            result={"ok": True},
            error_code=None,
            duration_ms=1,
            session_id="scoped-session",
            turn_id="scoped-turn",
            step_id="matching-explicit-step",
            store=store,
        )

    step = store.get("matching-explicit-step")
    assert step is not None
    assert (step.session_id, step.turn_id) == ("scoped-session", "scoped-turn")


@pytest.mark.parametrize(
    ("session_id", "turn_id"),
    (("other-session", None), (None, "other-turn")),
)
def test_partial_explicit_linkage_cannot_mix_two_turn_scopes(
    session_id,
    turn_id,
) -> None:
    with turn_linkage_scope(session_id="scoped-session", turn_id="scoped-turn"):
        with pytest.raises(ValueError, match="conflicts with the active Turn scope"):
            resolve_turn_linkage(session_id=session_id, turn_id=turn_id)


def test_conflicted_partial_linkage_is_fail_safe_for_the_business_call(store) -> None:
    with turn_linkage_scope(session_id="scoped-session", turn_id="scoped-turn"):
        recorder_module.record_tool_call(
            tool_id="conflicted-probe",
            arguments={},
            status="succeeded",
            result={"ok": True},
            error_code=None,
            duration_ms=1,
            session_id="other-session",
            step_id="conflicted-step",
            store=store,
        )

    assert store.get("conflicted-step") is None


def test_turn_linkage_propagates_only_when_a_worker_copies_context() -> None:
    with turn_linkage_scope(session_id="session-1", turn_id="turn-1"):
        inherited = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            copied = executor.submit(inherited.run, current_turn_linkage).result()
            background = executor.submit(current_turn_linkage).result()

    assert copied is not None
    assert (copied.session_id, copied.turn_id) == ("session-1", "turn-1")
    assert background is None


def test_background_recording_without_a_turn_scope_remains_unlinked(store) -> None:
    recorder_module.record_tool_call(
        tool_id="background_probe",
        arguments={},
        status="succeeded",
        result={"ok": True},
        error_code=None,
        duration_ms=1,
        step_id="background-step",
        store=store,
    )

    step = store.get("background-step")
    assert step is not None
    assert step.session_id is None
    assert step.turn_id is None
