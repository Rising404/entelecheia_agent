from personagraph.context_budget import ContextBudgetExceeded


def test_context_budget_error_shape_is_stable() -> None:
    error = ContextBudgetExceeded(
        limit=20,
        estimated_tokens=21,
        degraded=("drop_window",),
    )

    assert error.limit == 20
    assert error.estimated_tokens == 21
    assert error.degraded == ("drop_window",)
    assert error.stage == "block_assembly"
    assert str(error) == "Prompt context exceeds hard limit: estimated=21, limit=20"
