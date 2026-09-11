from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.deps import StoreDeps
from personagraph.l2.work_run import (
    AcceptanceUpdate,
    AcceptanceVerificationFeedback,
    AttemptDecision,
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    OutputWindowFormat,
    SubmitOutputWindowAction,
    TaskNodeVerificationRequestStatus,
    VerificationVerdict,
    WorkRunStatus,
)


_USER_TEXT = "请分析论文并给出一份可执行计划"


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_task_shell() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-shell",
        source="auxiliary_graph_v2_test",
        user_text=_USER_TEXT,
        lease_owner="auxiliary-graph-v2-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-shell-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "paper",
                        "title": "论文分析",
                        "objective": "分析论文并形成执行计划",
                        "source_excerpt": _USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return (
        session_id,
        turn_id,
        applied.created_insession_task_ids_by_local_key["paper"],
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _revision_proposal(
    *,
    reason: str = "initial",
    observe_executor: str = "host_primitive",
) -> auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord:
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="source_understood",
        criterion="输出必须受任务创建来源约束",
        source_anchor_ids=("task_creation_source",),
    )
    return auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
        revision_reason=reason,
        terminal_node_key="synthesize",
        nodes=(
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="observe",
                node_kind="observe",
                executor_kind=observe_executor,
                title="读取材料",
                objective="读取并核对材料",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="planning_context_v1",
                capability_profile_id="readonly_documents_v1",
            ),
            auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                local_node_key="synthesize",
                node_kind="synthesize",
                executor_kind="terminal_planner",
                title="形成任务图",
                objective="根据已核对材料形成任务图提案",
                source_anchor_ids=("task_creation_source",),
                acceptance_criteria=(acceptance,),
                output_contract="task_graph_revision_proposal_v2",
                capability_profile_id=None,
            ),
        ),
        edges=(
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="observe",
                consumer_node_key="synthesize",
            ),
        ),
    )


def _commit_initial(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    apply_id: str,
    graph_id: str,
    goal_id: str,
    authority_context: dict[str, object] | None = None,
) -> auxiliary_graphs.AuxiliaryGraphRevisionCommitResult:
    return auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id=apply_id,
        goal_objective="形成可执行且受来源约束的任务图",
        proposal=_revision_proposal(),
        authority_context=authority_context or {"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=graph_id,
        goal_id=goal_id,
    )


def _replace_v60_tables_with_known_drift_shapes() -> None:
    with store._connect() as conn:
        output_sql = str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='insession_work_run_output_windows'"
            ).fetchone()[0]
        )
        drifted_output_sql = output_sql.replace(
            "ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            "ON DELETE RESTRICT",
        )
        assert drifted_output_sql != output_sql
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA legacy_alter_table=ON")
        conn.execute("DROP TABLE insession_work_run_budget_charges")
        conn.execute(
            """
            CREATE TABLE insession_work_run_budget_charges (
                budget_charge_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                work_run_revision_before INTEGER NOT NULL,
                work_run_revision_after INTEGER NOT NULL,
                window_state_version_before INTEGER NOT NULL,
                window_state_version_after INTEGER NOT NULL,
                active_seconds_delta REAL NOT NULL,
                active_seconds_before REAL NOT NULL,
                active_seconds_after REAL NOT NULL,
                disposition TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(work_run_id, checkpoint_id),
                FOREIGN KEY(session_id, work_run_id)
                    REFERENCES insession_work_runs(session_id, work_run_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(turn_id, work_run_id)
                    REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                    ON DELETE RESTRICT
            )
            """
        )
        conn.execute("DROP TABLE insession_work_run_output_windows")
        conn.execute(drifted_output_sql)
        conn.execute("PRAGMA legacy_alter_table=OFF")
        conn.execute("PRAGMA foreign_keys=ON")


def test_details_project_the_exact_persisted_goal_objective() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    result = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-goal-objective-revision",
        goal_objective="  Preserve this exact durable planning objective.  ",
        proposal=_revision_proposal(),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-goal-objective-graph",
        goal_id="aux-v2-goal-objective-goal",
    )
    assert result.status == "applied"

    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )

    assert details is not None
    assert details.goal_objective == (
        "Preserve this exact durable planning objective."
    )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT objective FROM insession_auxiliary_graph_goals "
            "WHERE goal_id=?",
            (details.goal_id,),
        ).fetchone()[0] == details.goal_objective


def test_revision_commit_is_atomic_cas_bound_and_exactly_replayable() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    proposal = _revision_proposal()
    kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "insession_task_id": task_id,
        "expected_task_state_version": 1,
        "expected_base_task_graph_revision": None,
        "expected_control_state_version": None,
        "expected_current_auxiliary_graph_revision": None,
        "apply_id": "aux-v2-revision-1",
        "goal_objective": "形成可执行且受来源约束的任务图",
        "proposal": proposal,
        "authority_context": {"anchors": []},
        "budget_profile": {"profile_id": "planning-test-v1"},
        "auxiliary_graph_id": "aux-v2-graph",
        "goal_id": "aux-v2-goal",
    }
    applied = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **kwargs,
    )
    assert applied.status == "applied"
    assert applied.committed_auxiliary_graph_revision == 1
    assert applied.control_state_version == 2

    replayed = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **kwargs,
    )
    assert replayed == applied.model_copy(update={"status": "replayed"})

    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.auxiliary_graph_revision == 1
    assert current.parent_auxiliary_graph_revision is None
    assert current.budget_usage["auxiliary_graph_revisions"] == 1
    assert current.budget_usage["current_graph_nodes"] == 2
    assert current.budget_usage["current_graph_depth"] == 2
    assert len(current.nodes) == 2
    assert len(current.edges) == 1
    assert current.revision is not None
    assert current.revision.structure_sha256 == applied.structure_sha256
    assert current.revision.nodes[-1].executor_kind.value == "terminal_planner"
    assert current.revision.nodes[-1].capability_profile_id is None
    assert current.revision.nodes[0].input_resource_aliases == ()
    assert current.revision.edges[0].required is True
    assert current.goal == planning_store.get_current_auxiliary_planning_goal(
        session_id=session_id,
        insession_task_id=task_id,
    )

    revised = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **(
            kwargs
            | {
                "expected_control_state_version": 2,
                "expected_current_auxiliary_graph_revision": 1,
                "apply_id": "aux-v2-revision-2",
                "proposal": _revision_proposal(reason="evidence_changed"),
            }
        ),
    )
    assert revised.committed_auxiliary_graph_revision == 2
    assert revised.control_state_version == 3
    assert revised.authority_snapshot_id != applied.authority_snapshot_id

    # 重放旧回执时验证其不可变修订版，而不是错误要求它仍为当前指针。
    replayed_again = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        **kwargs,
    )
    assert replayed_again.status == "replayed"
    assert replayed_again.committed_auxiliary_graph_revision == 1

    with pytest.raises(work_run_store.WorkExecutionRevisionConflict):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            **(
                kwargs
                | {
                    "expected_control_state_version": 2,
                    "expected_current_auxiliary_graph_revision": 1,
                    "apply_id": "aux-v2-stale-revision",
                    "proposal": _revision_proposal(reason="manual_replan"),
                }
            ),
        )
    with pytest.raises(auxiliary_graphs.AuxiliaryGraphApplyIdCollision):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            **(kwargs | {"goal_objective": "复用 apply_id 的伪造负载"}),
        )

    with store._connect() as conn:
        states = {
            int(row["auxiliary_graph_revision"]): str(row["status"])
            for row in conn.execute(
                "SELECT auxiliary_graph_revision, status FROM "
                "insession_auxiliary_graph_revision_states_v2 "
                "WHERE auxiliary_graph_id='aux-v2-graph'"
            ).fetchall()
        }
        assert states == {1: "superseded", 2: "active"}
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_graph_revision_apply_receipts_v2"
            ).fetchone()[0]
        ) == 2
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_auxiliary_goal_budget_charges"
            ).fetchone()[0]
        ) == 2
        assert [
            int(row["state_version"])
            for row in conn.execute(
                "SELECT state_version FROM "
                "insession_auxiliary_goal_budget_snapshots "
                "WHERE goal_id='aux-v2-goal' ORDER BY state_version"
            ).fetchall()
        ] == [1, 2, 3]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_revision_lineage_advances_node_definition_without_completion_carry() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    first = _commit_initial(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        apply_id="aux-v2-lineage-1",
        graph_id="aux-v2-lineage-graph",
        goal_id="aux-v2-lineage-goal",
    )
    before = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert before is not None
    before_by_key = {node.local_node_key: node for node in before.nodes}
    base = _revision_proposal(reason="evidence_changed")
    revised_proposal = base.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(
                    update={
                        "title": f"{node.title}（修订）",
                        "origin_node_alias": node.local_node_key,
                    }
                )
                for node in base.nodes
            )
        }
    )

    revised = auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=first.control_state_version,
        expected_current_auxiliary_graph_revision=1,
        apply_id="aux-v2-lineage-2",
        goal_objective="形成可执行且受来源约束的任务图",
        proposal=revised_proposal,
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-lineage-graph",
        goal_id="aux-v2-lineage-goal",
    )

    assert revised.committed_auxiliary_graph_revision == 2
    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None and current.revision is not None
    current_by_key = {node.local_node_key: node for node in current.nodes}
    for key, node in current_by_key.items():
        previous = before_by_key[key]
        assert node.auxiliary_node_id == previous.auxiliary_node_id
        assert node.node_revision == previous.node_revision + 1
        assert node.status == "proposed"
        assert node.origin_node_ref is not None
        assert node.origin_node_ref.node_id == previous.auxiliary_node_id
        assert node.origin_node_ref.node_revision == previous.node_revision
    assert current.budget is not None
    assert current.budget.usage.distinct_auxiliary_nodes == 2
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_node_completion_carries_v2 "
                "WHERE auxiliary_graph_id='aux-v2-lineage-graph'"
            ).fetchone()[0]
        ) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("origin_aliases", (("missing", "synthesize"), ("observe", "observe")))
def test_revision_rejects_unknown_or_duplicate_origin_aliases_atomically(
    origin_aliases: tuple[str, str],
) -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    first = _commit_initial(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        apply_id="aux-v2-bad-lineage-1",
        graph_id="aux-v2-bad-lineage-graph",
        goal_id="aux-v2-bad-lineage-goal",
    )
    base = _revision_proposal(reason="manual_replan")
    invalid = base.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"origin_node_alias": origin})
                for node, origin in zip(base.nodes, origin_aliases, strict=True)
            )
        }
    )

    with pytest.raises(auxiliary_graphs.AuxiliaryGraphPersistenceError):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            store._deps(),
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            expected_task_state_version=1,
            expected_base_task_graph_revision=None,
            expected_control_state_version=first.control_state_version,
            expected_current_auxiliary_graph_revision=1,
            apply_id=f"aux-v2-bad-lineage-{origin_aliases[0]}",
            goal_objective="形成可执行且受来源约束的任务图",
            proposal=invalid,
            authority_context={"anchors": []},
            budget_profile={"profile_id": "planning-test-v1"},
            auxiliary_graph_id="aux-v2-bad-lineage-graph",
            goal_id="aux-v2-bad-lineage-goal",
        )

    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_graph_revision_snapshots "
                "WHERE auxiliary_graph_id='aux-v2-bad-lineage-graph'"
            ).fetchone()[0]
        ) == 1
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_goal_budget_charges "
                "WHERE goal_id='aux-v2-bad-lineage-goal'"
            ).fetchone()[0]
        ) == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_mid_commit_failure_rolls_back_every_authority_row() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    colliding_ids = StoreDeps(
        init_db=store.init_db,
        connect=store._connect,
        now=store._now,
        new_id=lambda: "collision",
    )
    with pytest.raises(sqlite3.IntegrityError):
        auxiliary_graphs.commit_auxiliary_graph_revision(
            colliding_ids,
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            expected_task_state_version=1,
            expected_base_task_graph_revision=None,
            expected_control_state_version=None,
            expected_current_auxiliary_graph_revision=None,
            apply_id="aux-v2-failing-commit",
            goal_objective="该事务必须完整回滚",
            proposal=_revision_proposal(),
            authority_context={"anchors": []},
            budget_profile={"profile_id": "planning-test-v1"},
            auxiliary_graph_id="aux-v2-failing-graph",
            goal_id="aux-v2-failing-goal",
        )

    with store._connect() as conn:
        for table in (
            "insession_auxiliary_graph_v2_containers",
            "insession_auxiliary_graph_goals",
            "insession_auxiliary_authorization_manifests",
            "insession_auxiliary_goal_budgets",
            "insession_auxiliary_goal_budget_snapshots",
            "insession_auxiliary_authority_snapshots",
            "insession_auxiliary_authority_anchors",
            "insession_auxiliary_graph_revision_snapshots",
            "insession_auxiliary_node_definitions_v2",
            "insession_auxiliary_goal_budget_charges",
            "insession_auxiliary_graph_revision_apply_receipts_v2",
        ):
            assert int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            ) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_composite_fk_rejects_cross_goal_pointer_splice() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-fk-commit",
        goal_objective="验证组合外键",
        proposal=_revision_proposal(),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-fk-graph",
        goal_id="aux-v2-fk-goal",
    )
    with pytest.raises(sqlite3.IntegrityError):
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_auxiliary_graph_v2_containers "
                "SET current_goal_id='forged-goal' "
                "WHERE auxiliary_graph_id='aux-v2-fk-graph'"
            )

    current = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.goal_id == "aux-v2-fk-goal"


def test_loader_fails_closed_on_immutable_structure_tamper() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-tamper-commit",
        goal_objective="验证不可变结构散列",
        proposal=_revision_proposal(),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-tamper-graph",
        goal_id="aux-v2-tamper-goal",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_node_definitions_v2 SET title=? "
            "WHERE auxiliary_graph_id='aux-v2-tamper-graph' "
            "AND node_kind='observe'",
            ("篡改后的标题",),
        )

    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="node definition is corrupt",
    ):
        auxiliary_graphs.get_auxiliary_graph_for_task(
            store._deps(),
            session_id=session_id,
            insession_task_id=task_id,
        )


def test_revision_authority_hash_is_composite_fk_bound() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    _commit_initial(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        apply_id="aux-v2-authority-fk",
        graph_id="aux-v2-authority-fk-graph",
        goal_id="aux-v2-authority-fk-goal",
    )

    with pytest.raises(sqlite3.IntegrityError):
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_auxiliary_graph_revision_snapshots "
                "SET authority_snapshot_sha256=? "
                "WHERE auxiliary_graph_id='aux-v2-authority-fk-graph'",
                ("0" * 64,),
            )


def test_execution_frontier_selects_only_dependency_ready_nodes() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    _commit_initial(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        apply_id="aux-v2-frontier-commit",
        graph_id="aux-v2-frontier-graph",
        goal_id="aux-v2-frontier-goal",
    )

    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )

    assert frontier.recoverable == ()
    assert [item.ordinal for item in frontier.ready_fresh] == [0]
    assert [item.executor_kind.value for item in frontier.ready_fresh] == [
        "host_primitive"
    ]
    assert frontier.ready_fresh[0].dependency_completion_ids == ()
    assert frontier.completed_node_refs == ()
    assert frontier.blocking_node_refs == ()


def test_exact_replay_fails_closed_on_historical_budget_charge_tamper() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "insession_task_id": task_id,
        "expected_task_state_version": 1,
        "expected_base_task_graph_revision": None,
        "expected_control_state_version": None,
        "expected_current_auxiliary_graph_revision": None,
        "apply_id": "aux-v2-replay-budget-tamper",
        "goal_objective": "形成可执行且受来源约束的任务图",
        "proposal": _revision_proposal(),
        "authority_context": {"anchors": []},
        "budget_profile": {"profile_id": "planning-test-v1"},
        "auxiliary_graph_id": "aux-v2-replay-budget-tamper-graph",
        "goal_id": "aux-v2-replay-budget-tamper-goal",
    }
    auxiliary_graphs.commit_auxiliary_graph_revision(store._deps(), **kwargs)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_goal_budget_charges "
            "SET budget_snapshot_after_json='{}' "
            "WHERE charge_key='aux-v2-replay-budget-tamper'"
        )

    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="replay lost immutable revision authority",
    ):
        auxiliary_graphs.commit_auxiliary_graph_revision(store._deps(), **kwargs)


def test_rejects_external_evidence_masquerading_as_authorization() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    forged_anchor = {
        "anchor_id": "forged_authorization_anchor",
        "projection_alias": "forged_authorization",
        "authority_class": "authorization",
        "origin_kind": "retrieved_source_unit",
        "origin_id": "untrusted-retrieval",
        "content_sha256": "1" * 64,
        "item_ordinal": 0,
        "projection_sha256": "2" * 64,
        "freshness_binding_sha256": "3" * 64,
    }
    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="authority anchor is not a formal authority anchor",
    ):
        _commit_initial(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            apply_id="aux-v2-forged-authorization",
            graph_id="aux-v2-forged-authorization-graph",
            goal_id="aux-v2-forged-authorization-goal",
            authority_context={"anchors": [forged_anchor]},
        )

    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_auxiliary_graph_v2_containers"
            ).fetchone()[0]
        ) == 0


def test_session_purge_removes_aggregate_before_restricted_turns() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-purge-commit",
        goal_objective="验证完整会话清理",
        proposal=_revision_proposal(),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-purge-graph",
        goal_id="aux-v2-purge-goal",
    )

    assert store.purge_session(session_id) is True
    assert store.get_session(session_id) is None
    with store._connect() as conn:
        for table in (
            "insession_auxiliary_graph_v2_containers",
            "insession_auxiliary_graph_goals",
            "insession_auxiliary_goal_budgets",
            "insession_auxiliary_goal_budget_snapshots",
            "insession_auxiliary_authority_snapshots",
            "insession_auxiliary_authority_anchors",
            "insession_auxiliary_graph_revision_snapshots",
            "insession_auxiliary_graph_revision_states_v2",
            "insession_auxiliary_node_definitions_v2",
            "insession_auxiliary_graph_revision_nodes_v2",
            "insession_auxiliary_graph_edges",
            "insession_auxiliary_node_states_v2",
            "insession_auxiliary_goal_budget_charges",
            "insession_auxiliary_graph_revision_apply_receipts_v2",
        ):
            assert int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            ) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_current_work_run_verification_uses_completion_as_terminal_identity() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v63-executable-graph",
        goal_objective="执行一个受精确版本约束的调查节点",
        proposal=_revision_proposal(observe_executor="model_work_run"),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v63-executable",
        goal_id="aux-v63-executable-goal",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    node = details.nodes[0]
    subject = AuxiliaryNodeSubject(
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        node_id=node.auxiliary_node_id,
        node_revision=node.node_revision,
    )
    with store._connect() as conn:
        task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )
    created = work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=task_version,
        expected_node_state_version=node.state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="aux-v63-create-run",
        work_run_id="aux-v63-run",
    )
    findings = store.get_execution_findings_ledger_for_owner(
        owner_kind="work_run",
        execution_owner_id=created.work_run_id,
    )
    assert findings is not None
    assert findings.ledger.originating_turn_id == turn_id
    active_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert active_frontier.ready_fresh == ()
    assert len(active_frontier.recoverable) == 1
    assert active_frontier.recoverable[0].work_run_id == "aux-v63-run"
    assert active_frontier.recoverable[0].subject == subject
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="aux-v63-run",
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id="aux-v63-start-attempt",
        catalog_snapshot={"revision": 1, "tools": []},
        attempt_id="aux-v63-attempt",
    )
    submitted = work_run_store.commit_work_run_output_action(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="aux-v63-run",
        attempt_id="aux-v63-attempt",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="source_understood",
                    model_claimed_satisfied=True,
                ),
            ),
            action=SubmitOutputWindowAction(
                content="材料已读取，并保留了精确来源约束。",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_output_revision=started.output_window_revision,
        expected_window_revision=started.window_state_version,
        apply_id="aux-v63-submit-output",
        active_seconds_delta=1,
    )
    prepared_mutation = verification_store.prepare_auxiliary_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="aux-v63-run",
        expected_work_run_revision=submitted.work_run_revision,
        expected_progress_revision=submitted.acceptance_progress_revision,
        expected_output_revision=submitted.output_window_revision,
        expected_window_revision=submitted.window_state_version,
        apply_id="aux-v63-prepare-verification",
        verification_request_id="aux-v63-verification",
    )
    prepared = verification_store.get_prepared_auxiliary_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="aux-v63-verification",
    )
    request = prepared.record.request
    result = NodeVerificationResult(
        verification_request_id=request.verification_request_id,
        verification_request_revision=request.revision,
        work_run_id=request.work_run_id,
        locked_work_run_revision=request.locked_work_run_revision,
        submitted_attempt_id=request.submitted_attempt_id,
        acceptance_progress_revision=request.acceptance_progress_revision,
        subject=request.subject,
        output_revision=request.output_revision,
        acceptance_results=(
            AcceptanceVerificationFeedback(
                acceptance_id="source_understood",
                verdict=VerificationVerdict.PASSED,
                finding="输出满足精确绑定的 Acceptance。",
            ),
        ),
        all_pass=True,
    )
    settle_kwargs = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": "aux-v63-run",
        "verification_request_id": "aux-v63-verification",
        "result": result,
        "expected_work_run_revision": prepared_mutation.work_run_revision,
        "expected_verification_request_revision": 1,
        "expected_window_revision": prepared_mutation.window_state_version,
        "apply_id": "aux-v63-settle-verification",
        "active_seconds_delta": 1,
        "completion_id": "aux-v63-completion",
    }
    settled = verification_store.commit_auxiliary_node_verification_result(**settle_kwargs)
    assert settled.all_pass is True
    assert settled.auxiliary_completion_id == "aux-v63-completion"
    replayed = verification_store.commit_auxiliary_node_verification_result(**settle_kwargs)
    assert replayed.status == "replayed"

    completed_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert completed_frontier.recoverable == ()
    assert [item.executor_kind.value for item in completed_frontier.ready_fresh] == [
        "terminal_planner"
    ]
    assert completed_frontier.ready_fresh[0].dependency_completion_ids == (
        "aux-v63-completion",
    )
    assert [item.node_id for item in completed_frontier.completed_node_refs] == [
        subject.node_id
    ]

    record = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="aux-v63-run",
    )
    assert record.auxiliary_node_completion_id == "aux-v63-completion"
    with store._connect() as conn:
        registry = conn.execute(
            "SELECT subject.subject_contract_version "
            "FROM insession_work_runs AS run "
            "JOIN insession_execution_subjects AS subject "
            "ON subject.execution_subject_id=run.execution_subject_id "
            "WHERE run.work_run_id='aux-v63-run'"
        ).fetchone()
        assert registry is not None
        assert str(registry[0]) == "auxiliary_node_v2"
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2"
            ).fetchone()[0]
        ) == 1
        verification_receipt = conn.execute(
            "SELECT result_json FROM insession_work_run_apply_receipts "
            "WHERE apply_id='aux-v63-settle-verification'"
        ).fetchone()
        assert verification_receipt is not None
        assert "terminal_receipt_id" not in json.loads(str(verification_receipt[0]))
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_execution_subjects"
            ).fetchone()[0]
        ) == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_verification_hard_budget_interrupts_the_whole_current_aggregate() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-verification-hard-graph",
        goal_objective="验证 verification 硬预算会原子中断整个聚合",
        proposal=_revision_proposal(observe_executor="model_work_run"),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "verification-hard-budget"},
        auxiliary_graph_id="aux-verification-hard",
        goal_id="aux-verification-hard-goal",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    node = details.nodes[0]
    subject = AuxiliaryNodeSubject(
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        node_id=node.auxiliary_node_id,
        node_revision=node.node_revision,
    )
    with store._connect() as conn:
        task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )
    created = work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=task_version,
        expected_node_state_version=node.state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="aux-verification-hard-create",
        work_run_id="aux-verification-hard-run",
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id="aux-verification-hard-start",
        catalog_snapshot={"revision": 1, "tools": []},
        attempt_id="aux-verification-hard-attempt",
    )
    submitted = work_run_store.commit_work_run_output_action(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        attempt_id="aux-verification-hard-attempt",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="source_understood",
                    model_claimed_satisfied=True,
                ),
            ),
            action=SubmitOutputWindowAction(
                content="该输出将在 verification 结算时触及硬预算。",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_output_revision=started.output_window_revision,
        expected_window_revision=started.window_state_version,
        apply_id="aux-verification-hard-submit",
        active_seconds_delta=1,
    )
    prepared_mutation = verification_store.prepare_auxiliary_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        expected_work_run_revision=submitted.work_run_revision,
        expected_progress_revision=submitted.acceptance_progress_revision,
        expected_output_revision=submitted.output_window_revision,
        expected_window_revision=submitted.window_state_version,
        apply_id="aux-verification-hard-prepare",
        verification_request_id="aux-verification-hard-request",
    )
    prepared = verification_store.get_prepared_auxiliary_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="aux-verification-hard-request",
    )
    request = prepared.record.request
    result = NodeVerificationResult(
        verification_request_id=request.verification_request_id,
        verification_request_revision=request.revision,
        work_run_id=request.work_run_id,
        locked_work_run_revision=request.locked_work_run_revision,
        submitted_attempt_id=request.submitted_attempt_id,
        acceptance_progress_revision=request.acceptance_progress_revision,
        subject=request.subject,
        output_revision=request.output_revision,
        acceptance_results=(
            AcceptanceVerificationFeedback(
                acceptance_id="source_understood",
                verdict=VerificationVerdict.PASSED,
                finding="语义结果有效，但硬预算优先终结执行。",
            ),
        ),
        all_pass=True,
    )
    with store._connect() as conn:
        before = conn.execute(
            "SELECT task.state_version, node.state_version, goal.state_version, "
            "revision.state_version "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS revision "
            "ON revision.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND revision.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, subject.node_id),
        ).fetchone()
        assert before is not None

    settled = verification_store.commit_auxiliary_node_verification_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        verification_request_id=request.verification_request_id,
        result=result,
        expected_work_run_revision=prepared_mutation.work_run_revision,
        expected_verification_request_revision=request.revision,
        expected_window_revision=prepared_mutation.window_state_version,
        apply_id="aux-verification-hard-settle",
        active_seconds_delta=900,
    )

    assert settled.work_run_status is WorkRunStatus.FAILED
    assert settled.work_run_reason == "work_run_limit_reached"
    assert settled.verification_request_status is (
        TaskNodeVerificationRequestStatus.INTERRUPTED
    )
    assert settled.all_pass is None
    assert settled.auxiliary_completion_id is None
    replayed = verification_store.commit_auxiliary_node_verification_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        verification_request_id=request.verification_request_id,
        result=result,
        expected_work_run_revision=prepared_mutation.work_run_revision,
        expected_verification_request_revision=request.revision,
        expected_window_revision=prepared_mutation.window_state_version,
        apply_id="aux-verification-hard-settle",
        active_seconds_delta=900,
    )
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == settled
    with store._connect() as conn:
        aggregate = conn.execute(
            "SELECT task.current_status, node.status, goal.status, revision.status, "
            "task.state_version, node.state_version, goal.state_version, "
            "revision.state_version "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS revision "
            "ON revision.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND revision.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, subject.node_id),
        ).fetchone()
        assert tuple(aggregate[:4]) == (
            "interrupted",
            "interrupted",
            "interrupted",
            "interrupted",
        )
        assert tuple(aggregate[4:]) == tuple(int(value) + 1 for value in before)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges "
            "WHERE budget_charge_id='aux-verification-hard-settle'"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v63_execution_subject_guard_rejects_raw_version_splice() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v63-splice-graph",
        goal_objective="验证执行主体不可拼接",
        proposal=_revision_proposal(observe_executor="model_work_run"),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v63-splice",
        goal_id="aux-v63-splice-goal",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    node = details.nodes[0]
    with store._connect() as conn:
        task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )
    work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            node_id=node.auxiliary_node_id,
            node_revision=node.node_revision,
        ),
        expected_task_state_version=task_version,
        expected_node_state_version=node.state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id="aux-v63-splice-create",
        work_run_id="aux-v63-splice-run",
    )
    with pytest.raises(sqlite3.IntegrityError, match="execution subject"):
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_work_runs SET auxiliary_node_id='forged-node' "
                "WHERE work_run_id='aux-v63-splice-run'"
            )


def test_v63_create_rejects_host_primitive_and_stale_revision() -> None:
    session_id, turn_id, task_id = _seed_task_shell()
    first = _commit_initial(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        apply_id="aux-v63-create-guards-initial",
        graph_id="aux-v63-create-guards",
        goal_id="aux-v63-create-guards-goal",
    )
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    primitive = details.nodes[0]
    with store._connect() as conn:
        task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )
    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="executor",
    ):
        work_run_store.create_auxiliary_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=AuxiliaryNodeSubject(
                task_id=task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                auxiliary_graph_revision=details.auxiliary_graph_revision,
                node_id=primitive.auxiliary_node_id,
                node_revision=primitive.node_revision,
            ),
            expected_task_state_version=task_version,
            expected_node_state_version=primitive.state_version,
            expected_window_revision=_window_revision(session_id),
            apply_id="aux-v63-host-primitive-create",
        )

    old_terminal = details.nodes[-1]
    old_subject = AuxiliaryNodeSubject(
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        node_id=old_terminal.auxiliary_node_id,
        node_revision=old_terminal.node_revision,
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=task_version,
        expected_base_task_graph_revision=None,
        expected_control_state_version=first.control_state_version,
        expected_current_auxiliary_graph_revision=1,
        apply_id="aux-v63-create-guards-revision-2",
        goal_objective="形成可执行且受来源约束的任务图",
        proposal=_revision_proposal(reason="evidence_changed"),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v63-create-guards",
        goal_id="aux-v63-create-guards-goal",
    )
    with pytest.raises(
        auxiliary_graphs.AuxiliaryGraphPersistenceError,
        match="current active",
    ):
        work_run_store.create_auxiliary_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=old_subject,
            expected_task_state_version=task_version,
            expected_node_state_version=old_terminal.state_version,
            expected_window_revision=_window_revision(session_id),
            apply_id="aux-v63-stale-create",
        )
