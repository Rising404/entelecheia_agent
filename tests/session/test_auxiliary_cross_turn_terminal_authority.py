from __future__ import annotations

import hashlib
import json
from pathlib import Path

from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
)
from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainPorts,
    AuxiliaryProductionChainRequest,
    AuxiliaryProductionChainStatus,
    run_auxiliary_to_verified_delivery,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from tests.documents._authority import bound_project_document_authority


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_cross_turn_task() -> tuple[str, str, str, str]:
    session_id = store.create_session("Entelecheia")
    created = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="cross-turn-terminal-create",
        source="cross_turn_terminal_authority_test",
        user_text="创建材料分析任务",
        lease_owner="cross-turn-terminal-test",
    )
    creation_turn_id = str(created["turn"]["turn_id"])
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=creation_turn_id,
        apply_id="cross-turn-terminal-create-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "analysis",
                        "title": "材料分析",
                        "objective": "理解材料并输出有依据的结论",
                        "source_excerpt": "创建材料分析任务",
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=int(window["state_version"]),
    )
    task_id = applied.created_insession_task_ids_by_local_key["analysis"]
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=creation_turn_id,
        expected_window_revision=int(window["state_version"]),
        processing_level="L2",
        assistant_content="任务已建立。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=creation_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )

    invoked = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="cross-turn-terminal-execute",
        source="cross_turn_terminal_authority_test",
        user_text="继续执行已有任务",
        lease_owner="cross-turn-terminal-test",
    )
    invocation_turn_id = str(invoked["turn"]["turn_id"])
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=invocation_turn_id,
        apply_id="cross-turn-terminal-link-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": "继续执行已有任务",
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=int(window["state_version"]),
    )
    return session_id, creation_turn_id, invocation_turn_id, task_id


def test_cross_turn_terminal_seal_commit_preserves_creation_anchor_provenance(
    tmp_path: Path,
) -> None:
    session_id, creation_turn_id, invocation_turn_id, task_id = (
        _seed_cross_turn_task()
    )
    def emit(_event):
        return None

    with bound_project_document_authority(tmp_path):
        result = run_auxiliary_to_verified_delivery(
            AuxiliaryProductionChainRequest(
                session_id=session_id,
                turn_id=invocation_turn_id,
                task_id=task_id,
                max_auxiliary_effect_steps=8,
            ),
            ports=AuxiliaryProductionChainPorts(
                auxiliary=AuxiliaryApplicationPorts(
                    model_ledger_store=store,
                    emit=emit,
                ),
                delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
            ),
        )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    details = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None
    assert details.current_graph_revision == 1
    with store._connect() as conn:
        revision = conn.execute(
            "SELECT source_turn_id, source_anchors_json FROM "
            "insession_task_graph_revisions WHERE insession_task_id=? "
            "AND graph_revision=1",
            (task_id,),
        ).fetchone()
        finish = conn.execute(
            "SELECT validation_context_sha256, receipt_json FROM "
            "insession_auxiliary_v2_finish_gate_receipts "
            "WHERE insession_task_id=?",
            (task_id,),
        ).fetchone()
    assert revision is not None and finish is not None
    committed_anchors = json.loads(str(revision["source_anchors_json"]))
    creation_anchor = next(
        item
        for item in committed_anchors
        if item["anchor_id"] == "task_creation_source"
    )
    assert revision["source_turn_id"] == invocation_turn_id
    assert creation_anchor["source_turn_id"] == creation_turn_id

    finish_receipt = json.loads(str(finish["receipt_json"]))
    sealed_context = finish_receipt["validation_context"]
    sealed_creation = next(
        item
        for item in sealed_context["source_anchors"]
        if item["anchor_id"] == "task_creation_source"
    )
    assert sealed_context["source_turn_id"] == invocation_turn_id
    assert sealed_creation["source_turn_id"] == creation_turn_id
    assert _canonical_sha256(sealed_context) == str(
        finish["validation_context_sha256"]
    )
