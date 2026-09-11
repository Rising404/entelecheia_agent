from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from personagraph.context_budget import (
    ContextBlockAdmissionReason,
    ContextBlockStability,
    ContextBudgetEncodingError,
    ContextBudgetExceeded,
    ContextBudgetMeasurementError,
    IncrementalContextBlockBudget,
    IncrementalContextBudgetEstimator,
    MandatoryContextBlockBudgetExceeded,
)


@dataclass
class CountingCounter:
    name: str = "counting"
    kind: str = "tokenizer"
    requested: str = "test"
    model: str | None = "test-model"
    fallback_reason: str | None = None
    calls: int = 0

    def count_text(self, text: str) -> int:
        self.calls += 1
        return len(text)


def _measure(
    estimator: IncrementalContextBudgetEstimator,
    *,
    block_id: str,
    content: str,
    mandatory: bool = False,
):
    return estimator.measure_text(
        block_id=block_id,
        content=content,
        stability=ContextBlockStability.VOLATILE,
        mandatory=mandatory,
        priority=10,
        source_revision="r1",
    )


def test_unchanged_content_reuses_measurement_without_retaining_content() -> None:
    counter = CountingCounter()
    estimator = IncrementalContextBudgetEstimator(counter)

    first = _measure(estimator, block_id="query", content="secret user input")
    second = _measure(estimator, block_id="query-copy", content="secret user input")

    assert counter.calls == 1
    assert first.content_sha256 == second.content_sha256
    assert first.estimated_tokens == len("secret user input")
    assert estimator.cache_info().model_dump() == {
        "schema_version": 1,
        "hits": 1,
        "misses": 1,
        "size": 1,
        "capacity": 4096,
    }
    dumped = first.model_dump_json()
    assert "secret user input" not in dumped


def test_changed_content_misses_and_lru_capacity_is_bounded() -> None:
    counter = CountingCounter()
    estimator = IncrementalContextBudgetEstimator(counter, max_cache_entries=1)

    _measure(estimator, block_id="a", content="alpha")
    _measure(estimator, block_id="b", content="beta")
    _measure(estimator, block_id="a-again", content="alpha")

    assert counter.calls == 3
    assert estimator.cache_info().size == 1
    assert estimator.cache_info().misses == 3


def test_projection_bill_separates_mandatory_optional_and_overhead() -> None:
    estimator = IncrementalContextBudgetEstimator(CountingCounter())
    mandatory = _measure(
        estimator,
        block_id="current-user",
        content="12345",
        mandatory=True,
    )
    optional = _measure(
        estimator,
        block_id="evidence",
        content="123",
        mandatory=False,
    )

    estimate = estimator.estimate_projection(
        (mandatory, optional),
        projection_epoch="epoch-7",
        purpose="runtime_l1_step",
        projection_generation=2,
        fixed_overhead_tokens=4,
        fixed_overhead_utf8_bytes=9,
    )

    assert estimate.mandatory_tokens == 5
    assert estimate.optional_tokens == 3
    assert estimate.total_estimated_tokens == 12
    assert estimate.total_serialized_utf8_bytes == 17
    assert estimate.projection_epoch == "epoch-7"
    assert estimate.projection_generation == 2


def test_projection_rejects_duplicate_block_ids() -> None:
    estimator = IncrementalContextBudgetEstimator(CountingCounter())
    first = _measure(estimator, block_id="same", content="one")
    second = _measure(estimator, block_id="same", content="two")

    with pytest.raises(ValidationError, match="duplicate block_id"):
        estimator.estimate_projection(
            (first, second),
            projection_epoch="epoch",
            purpose="runtime_l1_step",
            projection_generation=0,
        )


def test_invalid_utf8_text_fails_before_counting() -> None:
    counter = CountingCounter()
    estimator = IncrementalContextBudgetEstimator(counter)

    with pytest.raises(ContextBudgetEncodingError):
        _measure(estimator, block_id="bad", content="\ud800")

    assert counter.calls == 0


def test_counter_must_return_a_non_negative_integer() -> None:
    counter = CountingCounter()
    counter.count_text = lambda _text: -1  # type: ignore[method-assign]
    estimator = IncrementalContextBudgetEstimator(counter)

    with pytest.raises(ContextBudgetMeasurementError, match="non-negative integer"):
        _measure(estimator, block_id="bad-counter", content="text")


def test_first_level_budget_admits_exact_boundary_and_rejects_optional_overflow() -> None:
    estimator = IncrementalContextBudgetEstimator(CountingCounter())
    first = _measure(
        estimator,
        block_id="mandatory",
        content="12345",
        mandatory=True,
    )
    optional = _measure(
        estimator,
        block_id="optional",
        content="1234",
        mandatory=False,
    )
    ledger = IncrementalContextBlockBudget(
        input_budget_tokens=10,
        fixed_overhead_tokens=5,
    )

    admitted = ledger.try_admit(first)
    rejected = ledger.try_admit(optional)

    assert admitted.accepted is True
    assert admitted.reason is ContextBlockAdmissionReason.ADMITTED
    assert admitted.remaining_tokens == 0
    assert rejected.accepted is False
    assert rejected.reason is ContextBlockAdmissionReason.INSUFFICIENT_BUDGET
    assert rejected.used_tokens_before == rejected.used_tokens_after == 10
    snapshot = ledger.snapshot()
    assert snapshot.admitted_blocks == (first,)
    assert snapshot.used_tokens == 10
    assert snapshot.remaining_tokens == 0


def test_mandatory_overflow_fails_without_changing_budget_state() -> None:
    estimator = IncrementalContextBudgetEstimator(CountingCounter())
    mandatory = _measure(
        estimator,
        block_id="mandatory",
        content="123456",
        mandatory=True,
    )
    ledger = IncrementalContextBlockBudget(
        input_budget_tokens=10,
        fixed_overhead_tokens=5,
    )

    with pytest.raises(MandatoryContextBlockBudgetExceeded) as caught:
        ledger.try_admit(mandatory)

    assert isinstance(caught.value, ContextBudgetExceeded)
    assert caught.value.block_id == "mandatory"
    assert caught.value.limit == 10
    assert caught.value.estimated_tokens == 11
    assert caught.value.degraded == ()
    assert caught.value.stage == "block_admission"
    assert ledger.snapshot().used_tokens == 5
    assert ledger.snapshot().admitted_blocks == ()


def test_first_level_budget_rejects_duplicate_block_ids() -> None:
    estimator = IncrementalContextBudgetEstimator(CountingCounter())
    block = _measure(estimator, block_id="same", content="12")
    ledger = IncrementalContextBlockBudget(input_budget_tokens=10)

    assert ledger.try_admit(block).accepted is True
    with pytest.raises(ValueError, match="duplicate context block id"):
        ledger.try_admit(block)


def test_counter_fingerprint_failure_is_typed() -> None:
    class MissingFingerprintCounter:
        def count_text(self, text: str) -> int:
            return len(text)

    with pytest.raises(ContextBudgetMeasurementError):
        IncrementalContextBudgetEstimator(MissingFingerprintCounter())  # type: ignore[arg-type]
