from personagraph.session.context.catalog import all_policies, policies_for_domain
from personagraph.session.context.models import (
    SessionDomain,
    SourceKind,
)


def test_catalog_has_unique_domain_type_pairs_and_stable_domain_groups():
    policies = all_policies()
    keys = [(policy.domain, policy.state_type) for policy in policies]
    assert len(keys) == len(set(keys))
    assert len(policies_for_domain(SessionDomain.USER)) == 6
    assert len(policies_for_domain(SessionDomain.TASK)) == 11
    assert len(policies_for_domain(SessionDomain.INTERACTION)) == 4


def test_only_short_lived_low_risk_inferences_auto_apply():
    inferred_auto = [
        policy for policy in all_policies()
        if SourceKind.INFERRED in policy.auto_apply_sources
    ]
    assert {(p.domain, p.state_type) for p in inferred_auto} == {
        (SessionDomain.USER, "affect_signal"),
        (SessionDomain.TASK, "task_focus"),
        (SessionDomain.INTERACTION, "referent"),
    }
    assert all(policy.min_inferred_confidence is not None for policy in inferred_auto)


def test_assistant_source_is_isolated_to_assistant_commitments():
    assistant_policies = [
        policy for policy in all_policies()
        if SourceKind.ASSISTANT in policy.allowed_sources
    ]
    assert [(p.domain, p.state_type) for p in assistant_policies] == [
        (SessionDomain.INTERACTION, "assistant_commitment")
    ]
