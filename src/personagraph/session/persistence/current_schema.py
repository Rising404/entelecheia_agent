"""当前 sessions.sqlite schema 基线。

本模块只包含声明式 DDL。历史升级代码刻意不在运行时中存在：不受支持的数据库改为以
关闭方式失败。
"""

from __future__ import annotations


CURRENT_SCHEMA_SQL = r"""
CREATE TABLE execution_finding_entry_revisions (
            entry_revision_id TEXT PRIMARY KEY,
            ledger_id TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL CHECK(entry_revision >= 1),
            sequence INTEGER NOT NULL CHECK(sequence >= 1),
            operation TEXT NOT NULL CHECK(operation IN (
                'record', 'supersede', 'retract'
            )),
            kind TEXT NOT NULL CHECK(kind IN ('finding', 'decision', 'gap')),
            claim TEXT NOT NULL CHECK(length(claim) > 0),
            source_refs_json TEXT NOT NULL,
            scope_keys_json TEXT NOT NULL,
            entry_state TEXT NOT NULL CHECK(entry_state IN (
                'active', 'retracted'
            )),
            writer_unit_id TEXT NOT NULL,
            writer_tool_call_id TEXT NOT NULL,
            mutation_id TEXT NOT NULL,
            supersedes_entry_revision_id TEXT,
            revision_reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(ledger_id, entry_id, entry_revision),
            UNIQUE(ledger_id, sequence),
            CHECK(
                (
                    operation = 'record'
                    AND entry_revision = 1
                    AND supersedes_entry_revision_id IS NULL
                    AND revision_reason IS NULL
                    AND entry_state = 'active'
                )
                OR
                (
                    operation = 'supersede'
                    AND entry_revision > 1
                    AND supersedes_entry_revision_id IS NOT NULL
                    AND revision_reason IS NULL
                    AND entry_state = 'active'
                )
                OR
                (
                    operation = 'retract'
                    AND entry_revision > 1
                    AND supersedes_entry_revision_id IS NOT NULL
                    AND revision_reason IS NOT NULL
                    AND entry_state = 'retracted'
                )
            ),
            FOREIGN KEY(ledger_id)
                REFERENCES execution_findings_ledgers(ledger_id)
                ON DELETE CASCADE,
            FOREIGN KEY(supersedes_entry_revision_id)
                REFERENCES execution_finding_entry_revisions(entry_revision_id)
                ON DELETE RESTRICT
        );

CREATE TABLE execution_findings_ledgers (
            ledger_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            originating_turn_id TEXT NOT NULL,
            owner_kind TEXT NOT NULL CHECK(owner_kind IN (
                'l1_turn_run', 'work_run'
            )),
            execution_owner_id TEXT NOT NULL,
            l1_turn_run_id TEXT,
            work_run_id TEXT,
            status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
            revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
            quota_json TEXT NOT NULL CHECK(length(quota_json) > 0),
            quota_hash TEXT NOT NULL CHECK(length(quota_hash) = 64),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            closed_at TEXT,
            UNIQUE(session_id, ledger_id),
            CHECK(
                (
                    owner_kind = 'l1_turn_run'
                    AND l1_turn_run_id = execution_owner_id
                    AND work_run_id IS NULL
                )
                OR
                (
                    owner_kind = 'work_run'
                    AND work_run_id = execution_owner_id
                    AND l1_turn_run_id IS NULL
                )
            ),
            CHECK(
                (status = 'open' AND closed_at IS NULL)
                OR (status = 'closed' AND closed_at IS NOT NULL)
            ),
            FOREIGN KEY(session_id, originating_turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE,
            FOREIGN KEY(session_id, originating_turn_id, l1_turn_run_id)
                REFERENCES l1_turn_runs(session_id, turn_id, l1_turn_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(session_id, work_run_id)
                REFERENCES insession_work_runs(session_id, work_run_id)
                ON DELETE CASCADE
        );

CREATE TABLE execution_findings_mutations (
            ledger_id TEXT NOT NULL,
            mutation_id TEXT NOT NULL,
            request_json TEXT NOT NULL CHECK(length(request_json) > 0),
            request_hash TEXT NOT NULL CHECK(length(request_hash) = 64),
            expected_revision INTEGER NOT NULL CHECK(expected_revision >= 0),
            applied_revision INTEGER NOT NULL CHECK(applied_revision >= 1),
            result_json TEXT NOT NULL CHECK(length(result_json) > 0),
            result_hash TEXT NOT NULL CHECK(length(result_hash) = 64),
            created_at TEXT NOT NULL,
            PRIMARY KEY(ledger_id, mutation_id),
            UNIQUE(ledger_id, applied_revision),
            CHECK(applied_revision = expected_revision + 1),
            FOREIGN KEY(ledger_id)
                REFERENCES execution_findings_ledgers(ledger_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_active_task_graph_execution_replan_requests (
                request_id TEXT PRIMARY KEY,
                request_sha256 TEXT NOT NULL CHECK(
                    length(request_sha256)=64
                    AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(base_graph_revision>=1),
                target_graph_revision INTEGER NOT NULL CHECK(
                    target_graph_revision=base_graph_revision+1
                ),
                activated_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id),
                FOREIGN KEY(request_id)
                    REFERENCES insession_task_graph_execution_replan_requests(
                        request_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_active_task_graph_revision_triggers (
                trigger_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(
                    base_graph_revision >= 1
                ),
                target_graph_revision INTEGER NOT NULL CHECK(
                    target_graph_revision=base_graph_revision+1
                ),
                activated_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id),
                FOREIGN KEY(trigger_id)
                    REFERENCES insession_task_graph_revision_triggers(trigger_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_active_replan_triggers (
                trigger_id TEXT PRIMARY KEY,
                semantic_settlement_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                source_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    source_auxiliary_graph_revision >= 1
                ),
                activated_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id),
                UNIQUE(auxiliary_graph_id, goal_id),
                FOREIGN KEY(trigger_id)
                    REFERENCES insession_auxiliary_replan_triggers(trigger_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(semantic_settlement_id)
                    REFERENCES insession_auxiliary_semantic_quorum_settlements(
                        settlement_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_authority_anchors (
                authority_snapshot_id TEXT NOT NULL,
                anchor_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                projection_alias TEXT NOT NULL CHECK(
                    length(projection_alias) BETWEEN 1 AND 64
                    AND substr(projection_alias, 1, 1) GLOB '[a-z]'
                    AND projection_alias NOT GLOB '*[^a-z0-9_]*'
                ),
                authority_class TEXT NOT NULL CHECK(authority_class IN (
                    'authorization', 'evidence', 'gap'
                )),
                origin_kind TEXT NOT NULL CHECK(origin_kind IN (
                    'user_instruction_span', 'user_answer_span',
                    'prior_task_state', 'retrieved_source_unit',
                    'workspace_resource', 'tool_result', 'visual_unit',
                    'memory_record', 'artifact', 'primitive_result',
                    'gap_observation'
                )),
                origin_id TEXT NOT NULL CHECK(length(origin_id) BETWEEN 1 AND 500),
                source_revision INTEGER CHECK(
                    source_revision IS NULL OR source_revision >= 1
                ),
                content_sha256 TEXT NOT NULL CHECK(
                    length(content_sha256)=64
                    AND content_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                item_ordinal INTEGER NOT NULL CHECK(
                    item_ordinal BETWEEN 0 AND 1000000
                ),
                span_start INTEGER CHECK(span_start IS NULL OR span_start >= 0),
                span_end INTEGER CHECK(span_end IS NULL OR span_end >= 1),
                projection_sha256 TEXT NOT NULL CHECK(
                    length(projection_sha256)=64
                    AND projection_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                freshness_binding_sha256 TEXT NOT NULL CHECK(
                    length(freshness_binding_sha256)=64
                    AND freshness_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                disclosure_receipt_id TEXT,
                anchor_json TEXT NOT NULL,
                anchor_sha256 TEXT NOT NULL CHECK(
                    length(anchor_sha256)=64
                    AND anchor_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                PRIMARY KEY(authority_snapshot_id, anchor_id),
                UNIQUE(authority_snapshot_id, ordinal),
                UNIQUE(authority_snapshot_id, projection_alias),
                UNIQUE(authority_snapshot_id, anchor_id, authority_class),
                FOREIGN KEY(authority_snapshot_id)
                    REFERENCES insession_auxiliary_authority_snapshots(
                        authority_snapshot_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                CHECK(
                    (span_start IS NULL AND span_end IS NULL)
                    OR (span_start IS NOT NULL AND span_end IS NOT NULL
                        AND span_end > span_start)
                ),
                CHECK(
                    origin_kind NOT IN (
                        'user_instruction_span', 'user_answer_span'
                    ) OR (span_start IS NOT NULL AND span_end IS NOT NULL)
                ),
                CHECK(
                    origin_kind<>'prior_task_state'
                    OR source_revision IS NOT NULL
                ),
                CHECK(
                    (authority_class='authorization' AND origin_kind IN (
                        'user_instruction_span', 'user_answer_span',
                        'prior_task_state'
                    )) OR (authority_class<>'authorization' AND origin_kind NOT IN (
                        'user_instruction_span', 'user_answer_span',
                        'prior_task_state'
                    ))
                ),
                CHECK(
                    (authority_class='gap' AND origin_kind='gap_observation')
                    OR (authority_class<>'gap'
                        AND origin_kind<>'gap_observation')
                )
            );

CREATE TABLE insession_auxiliary_authority_snapshots (
                authority_snapshot_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                contract_version TEXT NOT NULL CHECK(
                    contract_version='planning-authority-snapshot-v1'
                ),
                snapshot_json TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL CHECK(
                    length(snapshot_sha256)=64
                    AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(auxiliary_graph_id, goal_id, authority_snapshot_id),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, snapshot_sha256
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_authorization_manifests (
                authorization_manifest_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                contract_version TEXT NOT NULL CHECK(
                    contract_version='planning-authorization-manifest-v1'
                ),
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL CHECK(
                    length(manifest_sha256)=64
                    AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    auxiliary_graph_id, goal_id,
                    authorization_manifest_id, manifest_sha256
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_capability_catalog_items (
                capability_catalog_snapshot_id TEXT NOT NULL,
                projection_sha256 TEXT NOT NULL,
                capability_alias TEXT NOT NULL CHECK(
                    length(capability_alias) BETWEEN 1 AND 64
                    AND substr(capability_alias, 1, 1) GLOB '[a-z]'
                    AND capability_alias NOT GLOB '*[^a-z0-9_]*'
                ),
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                descriptor_json TEXT NOT NULL,
                descriptor_sha256 TEXT NOT NULL CHECK(
                    length(descriptor_sha256)=64
                    AND descriptor_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                PRIMARY KEY(
                    capability_catalog_snapshot_id, projection_sha256,
                    capability_alias
                ),
                UNIQUE(
                    capability_catalog_snapshot_id, projection_sha256, ordinal
                ),
                FOREIGN KEY(
                    capability_catalog_snapshot_id, projection_sha256
                ) REFERENCES insession_auxiliary_capability_catalog_projections(
                    capability_catalog_snapshot_id, projection_sha256
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_capability_catalog_projections (
                capability_catalog_snapshot_id TEXT NOT NULL,
                capability_catalog_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(capability_catalog_snapshot_sha256)=64
                    AND capability_catalog_snapshot_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                projection_sha256 TEXT NOT NULL CHECK(
                    length(projection_sha256)=64
                    AND projection_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                projection_json TEXT NOT NULL,
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(
                    capability_catalog_snapshot_id, projection_sha256
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    capability_catalog_snapshot_id,
                    capability_catalog_snapshot_sha256, projection_sha256
                ),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_context_verification_receipts (
                verification_receipt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                source_observation_id TEXT,
                source_work_run_id TEXT,
                source_attempt_id TEXT,
                disposition TEXT NOT NULL CHECK(disposition IN (
                    'pass', 'partial', 'fail'
                )),
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    verification_receipt_id, receipt_sha256
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    verification_receipt_id, receipt_sha256
                ),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    source_observation_id
                ) REFERENCES insession_auxiliary_observations(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    observation_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(source_work_run_id, source_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    source_work_run_id
                ) REFERENCES insession_work_runs(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    work_run_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (source_observation_id IS NOT NULL
                        AND source_work_run_id IS NULL
                        AND source_attempt_id IS NULL)
                    OR (source_observation_id IS NULL
                        AND source_work_run_id IS NOT NULL
                        AND source_attempt_id IS NOT NULL)
                )
            );

CREATE TABLE insession_auxiliary_goal_budget_charges (
                budget_charge_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                budget_ledger_id TEXT NOT NULL,
                charge_kind TEXT NOT NULL CHECK(charge_kind IN (
                    'graph_revision', 'work_run', 'attempt', 'tool_call',
                    'logical_model_call', 'physical_model_attempt',
                    'retrieval', 'token_cost', 'autonomous_turn', 'extension'
                )),
                charge_key TEXT NOT NULL,
                budget_state_version_before INTEGER NOT NULL CHECK(
                    budget_state_version_before >= 1
                ),
                budget_state_version_after INTEGER NOT NULL CHECK(
                    budget_state_version_after=budget_state_version_before+1
                ),
                usage_before_json TEXT NOT NULL,
                usage_before_sha256 TEXT NOT NULL CHECK(
                    length(usage_before_sha256)=64
                    AND usage_before_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                usage_after_json TEXT NOT NULL,
                usage_after_sha256 TEXT NOT NULL CHECK(
                    length(usage_after_sha256)=64
                    AND usage_after_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_snapshot_after_json TEXT NOT NULL,
                budget_snapshot_after_sha256 TEXT NOT NULL CHECK(
                    length(budget_snapshot_after_sha256)=64
                    AND budget_snapshot_after_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                delta_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL CHECK(
                    length(payload_sha256)=64
                    AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                previous_charge_sha256 TEXT CHECK(
                    previous_charge_sha256 IS NULL
                    OR (length(previous_charge_sha256)=64
                        AND previous_charge_sha256 NOT GLOB '*[^0-9a-f]*')
                ),
                charge_sha256 TEXT NOT NULL CHECK(
                    length(charge_sha256)=64
                    AND charge_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(goal_id, charge_kind, charge_key),
                UNIQUE(goal_id, budget_state_version_after),
                UNIQUE(goal_id, charge_sha256),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) REFERENCES insession_auxiliary_goal_budgets(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    goal_id, budget_ledger_id, budget_state_version_after,
                    budget_snapshot_after_sha256
                ) REFERENCES insession_auxiliary_goal_budget_snapshots(
                    goal_id, budget_ledger_id, state_version, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(goal_id, previous_charge_sha256)
                    REFERENCES insession_auxiliary_goal_budget_charges(
                        goal_id, charge_sha256
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED,
                CHECK(
                    (budget_state_version_before=1
                        AND previous_charge_sha256 IS NULL)
                    OR (budget_state_version_before>1
                        AND previous_charge_sha256 IS NOT NULL)
                )
            );

CREATE TABLE insession_auxiliary_goal_budget_extensions (
                extension_receipt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                budget_ledger_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                approved_turn_id TEXT NOT NULL,
                authority_snapshot_id TEXT NOT NULL,
                authorization_anchor_id TEXT NOT NULL,
                authorization_anchor_class TEXT NOT NULL DEFAULT 'authorization'
                    CHECK(authorization_anchor_class='authorization'),
                reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 2000),
                profile_before_json TEXT NOT NULL,
                profile_before_sha256 TEXT NOT NULL CHECK(
                    length(profile_before_sha256)=64
                    AND profile_before_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                profile_after_json TEXT NOT NULL,
                profile_after_sha256 TEXT NOT NULL CHECK(
                    length(profile_after_sha256)=64
                    AND profile_after_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                previous_extension_receipt_sha256 TEXT CHECK(
                    previous_extension_receipt_sha256 IS NULL
                    OR (length(previous_extension_receipt_sha256)=64
                        AND previous_extension_receipt_sha256
                            NOT GLOB '*[^0-9a-f]*')
                ),
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(goal_id, ordinal),
                UNIQUE(goal_id, receipt_sha256),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) REFERENCES insession_auxiliary_goal_budgets(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(authority_snapshot_id, authorization_anchor_id)
                    REFERENCES insession_auxiliary_authority_anchors(
                        authority_snapshot_id, anchor_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    authority_snapshot_id, authorization_anchor_id,
                    authorization_anchor_class
                ) REFERENCES insession_auxiliary_authority_anchors(
                    authority_snapshot_id, anchor_id, authority_class
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id
                ) REFERENCES insession_auxiliary_authority_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(goal_id, previous_extension_receipt_sha256)
                    REFERENCES insession_auxiliary_goal_budget_extensions(
                        goal_id, receipt_sha256
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(session_id, approved_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (ordinal=0 AND previous_extension_receipt_sha256 IS NULL)
                    OR (ordinal>0
                        AND previous_extension_receipt_sha256 IS NOT NULL)
                )
            );

CREATE TABLE insession_auxiliary_goal_budget_snapshots (
                goal_id TEXT NOT NULL,
                budget_ledger_id TEXT NOT NULL,
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL CHECK(
                    length(snapshot_sha256)=64
                    AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(goal_id, budget_ledger_id, state_version),
                UNIQUE(goal_id, budget_ledger_id, state_version, snapshot_sha256),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) REFERENCES insession_auxiliary_goal_budgets(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_goal_budgets (
                goal_id TEXT PRIMARY KEY,
                budget_ledger_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                contract_version TEXT NOT NULL CHECK(
                    contract_version IN (
                        'auxiliary-goal-budget-v1',
                        'planning-episode-budget-v1'
                    )
                ),
                profile_json TEXT NOT NULL,
                profile_sha256 TEXT NOT NULL CHECK(
                    length(profile_sha256)=64
                    AND profile_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                usage_json TEXT NOT NULL,
                usage_sha256 TEXT NOT NULL CHECK(
                    length(usage_sha256)=64
                    AND usage_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                extensions_json TEXT NOT NULL,
                extensions_sha256 TEXT NOT NULL CHECK(
                    length(extensions_sha256)=64
                    AND extensions_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                snapshot_json TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL CHECK(
                    length(snapshot_sha256)=64
                    AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                counter_completeness TEXT NOT NULL CHECK(
                    counter_completeness IN ('complete', 'legacy_partial')
                ),
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id, auxiliary_graph_id, goal_id),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    budget_ledger_id
                ),
                UNIQUE(goal_id, budget_ledger_id),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_graph_edges (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL,
                dependency_auxiliary_node_id TEXT NOT NULL,
                dependency_node_revision INTEGER NOT NULL,
                consumer_auxiliary_node_id TEXT NOT NULL,
                consumer_node_revision INTEGER NOT NULL,
                required INTEGER NOT NULL DEFAULT 1 CHECK(required=1),
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                created_at TEXT NOT NULL,
                PRIMARY KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    dependency_auxiliary_node_id, consumer_auxiliary_node_id
                ),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    consumer_auxiliary_node_id, ordinal
                ),
                CHECK(dependency_auxiliary_node_id <> consumer_auxiliary_node_id),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    dependency_auxiliary_node_id, dependency_node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    consumer_auxiliary_node_id, consumer_node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_graph_goals (
                goal_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_ordinal INTEGER NOT NULL CHECK(goal_ordinal >= 1),
                base_task_graph_revision INTEGER CHECK(
                    base_task_graph_revision IS NULL
                    OR base_task_graph_revision >= 1
                ),
                target_task_graph_revision INTEGER NOT NULL CHECK(
                    target_task_graph_revision >= 1
                ),
                creation_turn_id TEXT NOT NULL,
                objective TEXT NOT NULL CHECK(length(objective) >= 1),
                authorization_manifest_id TEXT NOT NULL,
                authorization_manifest_sha256 TEXT NOT NULL CHECK(
                    length(authorization_manifest_sha256)=64
                    AND authorization_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_ledger_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN (
                    'active', 'waiting_user', 'waiting_authorization',
                    'waiting_external', 'interrupted', 'proposal_ready',
                    'gapped_ready', 'committed', 'failed', 'cancelled',
                    'superseded', 'budget_exhausted'
                )),
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(auxiliary_graph_id, goal_ordinal),
                UNIQUE(auxiliary_graph_id, goal_id),
                UNIQUE(session_id, insession_task_id, auxiliary_graph_id, goal_id),
                CHECK(
                    (base_task_graph_revision IS NULL
                        AND target_task_graph_revision=1)
                    OR target_task_graph_revision=base_task_graph_revision+1
                ),
                FOREIGN KEY(session_id, auxiliary_graph_id)
                    REFERENCES insession_auxiliary_graph_v2_containers(
                        session_id, auxiliary_graph_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, creation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, base_task_graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, goal_id, authorization_manifest_id,
                    authorization_manifest_sha256
                ) REFERENCES insession_auxiliary_authorization_manifests(
                    auxiliary_graph_id, goal_id, authorization_manifest_id,
                    manifest_sha256
                ) ON DELETE NO ACTION ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(goal_id, budget_ledger_id)
                    REFERENCES insession_auxiliary_goal_budgets(
                        goal_id, budget_ledger_id
                    ) ON DELETE NO ACTION ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED
            );

CREATE TABLE insession_auxiliary_graph_revision_apply_receipts_v2 (
                apply_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL CHECK(operation IN (
                    'initialize_goal_revision', 'append_goal_revision',
                    'supersede_goal', 'seal_observation',
                    'seal_authority_snapshot', 'charge_goal_budget'
                )),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                invocation_turn_id TEXT NOT NULL,
                expected_control_state_version INTEGER CHECK(
                    expected_control_state_version IS NULL
                    OR expected_control_state_version >= 1
                ),
                committed_control_state_version INTEGER NOT NULL CHECK(
                    committed_control_state_version >= 2
                ),
                expected_current_auxiliary_graph_revision INTEGER CHECK(
                    expected_current_auxiliary_graph_revision IS NULL
                    OR expected_current_auxiliary_graph_revision >= 1
                ),
                committed_auxiliary_graph_revision INTEGER CHECK(
                    committed_auxiliary_graph_revision IS NULL
                    OR committed_auxiliary_graph_revision >= 1
                ),
                expected_goal_state_version INTEGER CHECK(
                    expected_goal_state_version IS NULL
                    OR expected_goal_state_version >= 1
                ),
                committed_goal_state_version INTEGER NOT NULL CHECK(
                    committed_goal_state_version >= 1
                ),
                expected_budget_state_version INTEGER CHECK(
                    expected_budget_state_version IS NULL
                    OR expected_budget_state_version >= 1
                ),
                committed_budget_state_version INTEGER NOT NULL CHECK(
                    committed_budget_state_version >= 2
                ),
                budget_ledger_id TEXT NOT NULL,
                committed_budget_snapshot_json TEXT NOT NULL,
                committed_budget_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(committed_budget_snapshot_sha256)=64
                    AND committed_budget_snapshot_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                payload_sha256 TEXT NOT NULL CHECK(
                    length(payload_sha256)=64
                    AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_json TEXT NOT NULL,
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, committed_auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    goal_id, budget_ledger_id,
                    committed_budget_state_version,
                    committed_budget_snapshot_sha256
                ) REFERENCES insession_auxiliary_goal_budget_snapshots(
                    goal_id, budget_ledger_id, state_version, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (expected_control_state_version IS NULL
                        AND expected_current_auxiliary_graph_revision IS NULL)
                    OR (expected_control_state_version IS NOT NULL
                        AND expected_current_auxiliary_graph_revision IS NOT NULL)
                ),
                CHECK(
                    (expected_goal_state_version IS NULL
                        AND expected_budget_state_version IS NULL)
                    OR (expected_goal_state_version IS NOT NULL
                        AND expected_budget_state_version IS NOT NULL)
                ),
                CHECK(
                    operation NOT IN (
                        'initialize_goal_revision', 'append_goal_revision'
                    ) OR (
                        operation='initialize_goal_revision' AND (
                            (
                                expected_control_state_version IS NULL
                                AND expected_current_auxiliary_graph_revision
                                    IS NULL
                                AND expected_goal_state_version IS NULL
                                AND expected_budget_state_version IS NULL
                                AND committed_control_state_version=2
                                AND committed_auxiliary_graph_revision=1
                                AND committed_goal_state_version=1
                                AND committed_budget_state_version=2
                            ) OR (
                                expected_control_state_version IS NOT NULL
                                AND expected_current_auxiliary_graph_revision
                                    IS NOT NULL
                                AND expected_goal_state_version IS NULL
                                AND expected_budget_state_version IS NULL
                                AND committed_control_state_version=
                                    expected_control_state_version+1
                                AND committed_auxiliary_graph_revision=
                                    expected_current_auxiliary_graph_revision+1
                                AND committed_goal_state_version=1
                                AND committed_budget_state_version=2
                            )
                        ) OR (
                            operation='append_goal_revision'
                            AND expected_control_state_version IS NOT NULL
                            AND expected_current_auxiliary_graph_revision
                                IS NOT NULL
                            AND expected_goal_state_version IS NOT NULL
                            AND expected_budget_state_version IS NOT NULL
                            AND committed_control_state_version=
                                expected_control_state_version+1
                            AND committed_auxiliary_graph_revision=
                                expected_current_auxiliary_graph_revision+1
                            AND committed_goal_state_version=
                                expected_goal_state_version+1
                            AND committed_budget_state_version=
                                expected_budget_state_version+1
                        )
                    )
                )
            );

CREATE TABLE insession_auxiliary_graph_revision_nodes_v2 (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL,
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                required INTEGER NOT NULL CHECK(required IN (0, 1)),
                local_node_key TEXT NOT NULL CHECK(length(local_node_key) >= 1),
                carried_completion_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ),
                UNIQUE(auxiliary_graph_id, auxiliary_graph_revision, ordinal),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision, local_node_key
                ),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id
                ),
                FOREIGN KEY(auxiliary_graph_id, auxiliary_graph_revision)
                    REFERENCES insession_auxiliary_graph_revision_snapshots(
                        auxiliary_graph_id, auxiliary_graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(auxiliary_graph_id, auxiliary_node_id, node_revision)
                    REFERENCES insession_auxiliary_node_definitions_v2(
                        auxiliary_graph_id, auxiliary_node_id, node_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, required
                ) REFERENCES insession_auxiliary_node_definitions_v2(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, required
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision, carried_completion_id
                )
                    REFERENCES insession_auxiliary_node_completion_carries_v2(
                        auxiliary_graph_id, target_auxiliary_graph_revision,
                        target_auxiliary_node_id, target_node_revision,
                        carry_receipt_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED
            );

CREATE TABLE insession_auxiliary_graph_revision_snapshots (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                parent_auxiliary_graph_revision INTEGER CHECK(
                    parent_auxiliary_graph_revision IS NULL
                    OR parent_auxiliary_graph_revision >= 1
                ),
                base_task_graph_revision INTEGER CHECK(
                    base_task_graph_revision IS NULL
                    OR base_task_graph_revision >= 1
                ),
                source_turn_id TEXT NOT NULL,
                reason TEXT NOT NULL CHECK(reason IN (
                    'initial', 'resource_changed', 'evidence_changed',
                    'user_response', 'node_failed', 'verification_failed',
                    'external_resumed', 'authority_changed', 'manual_replan'
                )),
                authority_snapshot_id TEXT NOT NULL,
                authority_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(authority_snapshot_sha256)=64
                    AND authority_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                terminal_auxiliary_node_id TEXT NOT NULL,
                structure_contract_version TEXT NOT NULL CHECK(
                    structure_contract_version='auxiliary-graph-revision-v2'
                ),
                structure_sha256 TEXT NOT NULL CHECK(
                    length(structure_sha256)=64
                    AND structure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                PRIMARY KEY(auxiliary_graph_id, auxiliary_graph_revision),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id, structure_sha256
                ),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision, goal_id
                ),
                CHECK(
                    (auxiliary_graph_revision=1
                        AND parent_auxiliary_graph_revision IS NULL
                        AND reason='initial')
                    OR (auxiliary_graph_revision>1
                        AND parent_auxiliary_graph_revision=
                            auxiliary_graph_revision-1
                        AND reason<>'initial')
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, authority_snapshot_sha256
                ) REFERENCES insession_auxiliary_authority_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, source_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, base_task_graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, parent_auxiliary_graph_revision
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) ON DELETE NO ACTION ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    terminal_auxiliary_node_id
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id
                ) ON DELETE NO ACTION ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED
            );

CREATE TABLE insession_auxiliary_graph_revision_states_v2 (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'active', 'waiting_user', 'waiting_authorization',
                    'waiting_external', 'interrupted', 'proposal_ready',
                    'gapped_ready', 'committed', 'failed', 'cancelled',
                    'superseded', 'budget_exhausted'
                )),
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(auxiliary_graph_id, auxiliary_graph_revision),
                FOREIGN KEY(auxiliary_graph_id, auxiliary_graph_revision)
                    REFERENCES insession_auxiliary_graph_revision_snapshots(
                        auxiliary_graph_id, auxiliary_graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_graph_v2_containers (
                auxiliary_graph_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL UNIQUE,
                current_goal_id TEXT,
                current_auxiliary_graph_revision INTEGER CHECK(
                    current_auxiliary_graph_revision IS NULL
                    OR current_auxiliary_graph_revision >= 1
                ),
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, auxiliary_graph_id),
                UNIQUE(auxiliary_graph_id, insession_task_id),
                CHECK(
                    (current_goal_id IS NULL
                        AND current_auxiliary_graph_revision IS NULL)
                    OR (current_goal_id IS NOT NULL
                        AND current_auxiliary_graph_revision IS NOT NULL)
                ),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, current_auxiliary_graph_revision,
                    current_goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision, goal_id
                ) ON DELETE NO ACTION ON UPDATE RESTRICT
                    DEFERRABLE INITIALLY DEFERRED
            );

CREATE TABLE insession_auxiliary_node_completion_carries_v2 (
                carry_receipt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                source_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    source_auxiliary_graph_revision >= 1
                ),
                source_auxiliary_node_id TEXT NOT NULL,
                source_node_revision INTEGER NOT NULL CHECK(
                    source_node_revision >= 1
                ),
                source_completion_id TEXT NOT NULL,
                target_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    target_auxiliary_graph_revision >= 2
                ),
                target_auxiliary_node_id TEXT NOT NULL,
                target_node_revision INTEGER NOT NULL CHECK(
                    target_node_revision >= 1
                ),
                definition_sha256 TEXT NOT NULL CHECK(
                    length(definition_sha256)=64
                    AND definition_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                dependency_closure_sha256 TEXT NOT NULL CHECK(
                    length(dependency_closure_sha256)=64
                    AND dependency_closure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                context_artifact_manifest_sha256 TEXT NOT NULL CHECK(
                    length(context_artifact_manifest_sha256)=64
                    AND context_artifact_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                authority_snapshot_id TEXT NOT NULL,
                authority_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(authority_snapshot_sha256)=64
                    AND authority_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                capability_catalog_snapshot_id TEXT NOT NULL,
                capability_catalog_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(capability_catalog_snapshot_sha256)=64
                    AND capability_catalog_snapshot_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                freshness_manifest_sha256 TEXT NOT NULL CHECK(
                    length(freshness_manifest_sha256)=64
                    AND freshness_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    auxiliary_graph_id, target_auxiliary_graph_revision,
                    target_auxiliary_node_id, target_node_revision
                ),
                UNIQUE(
                    auxiliary_graph_id, target_auxiliary_graph_revision,
                    target_auxiliary_node_id, target_node_revision,
                    carry_receipt_id
                ),
                UNIQUE(source_completion_id, target_auxiliary_graph_revision),
                FOREIGN KEY(
                    auxiliary_graph_id, source_auxiliary_graph_revision,
                    source_auxiliary_node_id, source_node_revision,
                    source_completion_id
                ) REFERENCES insession_auxiliary_node_completions_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision, completion_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    source_auxiliary_graph_revision, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    target_auxiliary_graph_revision, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, source_auxiliary_node_id,
                    source_node_revision, definition_sha256
                ) REFERENCES insession_auxiliary_node_definitions_v2(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, definition_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, target_auxiliary_node_id,
                    target_node_revision, definition_sha256
                ) REFERENCES insession_auxiliary_node_definitions_v2(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, definition_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, target_auxiliary_graph_revision,
                    target_auxiliary_node_id, target_node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, authority_snapshot_sha256
                ) REFERENCES insession_auxiliary_authority_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    target_auxiliary_graph_revision>
                        source_auxiliary_graph_revision
                )
            );

CREATE TABLE insession_auxiliary_node_completions_v2 (
                completion_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                work_run_id TEXT NOT NULL UNIQUE,
                verification_request_id TEXT NOT NULL UNIQUE,
                submitted_attempt_id TEXT NOT NULL,
                output_revision INTEGER NOT NULL CHECK(output_revision >= 1),
                completion_json TEXT NOT NULL,
                completion_sha256 TEXT NOT NULL CHECK(
                    length(completion_sha256)=64
                    AND completion_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision, completion_id
                ),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    work_run_id
                ) REFERENCES insession_work_runs(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    work_run_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, verification_request_id)
                    REFERENCES insession_work_run_verification_requests(
                        work_run_id, verification_request_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, submitted_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, output_revision)
                    REFERENCES insession_work_run_output_windows(
                        work_run_id, output_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_node_definitions_v2 (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                node_kind TEXT NOT NULL CHECK(node_kind IN (
                    'observe', 'analyze', 'clarify', 'validate', 'synthesize'
                )),
                executor_kind TEXT NOT NULL CHECK(executor_kind IN (
                    'host_primitive', 'model_work_run', 'user_gate',
                    'terminal_planner'
                )),
                title TEXT NOT NULL CHECK(length(title) >= 1),
                objective TEXT NOT NULL CHECK(length(objective) >= 1),
                source_anchor_ids_json TEXT NOT NULL,
                acceptance_criteria_json TEXT NOT NULL,
                output_contract TEXT NOT NULL CHECK(length(output_contract) >= 1),
                capability_profile_id TEXT CHECK(
                    capability_profile_id IS NULL
                    OR length(capability_profile_id) >= 1
                ),
                input_resource_aliases_json TEXT NOT NULL,
                required INTEGER NOT NULL CHECK(required IN (0, 1)),
                semantic_fingerprint TEXT NOT NULL CHECK(
                    length(semantic_fingerprint)=64
                    AND semantic_fingerprint NOT GLOB '*[^0-9a-f]*'
                ),
                origin_auxiliary_node_id TEXT,
                origin_node_revision INTEGER CHECK(
                    origin_node_revision IS NULL OR origin_node_revision >= 1
                ),
                definition_sha256 TEXT NOT NULL CHECK(
                    length(definition_sha256)=64
                    AND definition_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                PRIMARY KEY(auxiliary_graph_id, auxiliary_node_id, node_revision),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, definition_sha256
                ),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_node_id,
                    node_revision, required
                ),
                CHECK(
                    (node_revision=1 AND origin_auxiliary_node_id IS NULL
                        AND origin_node_revision IS NULL)
                    OR (node_revision>1
                        AND origin_auxiliary_node_id=auxiliary_node_id
                        AND origin_node_revision=node_revision-1)
                ),
                FOREIGN KEY(auxiliary_graph_id)
                    REFERENCES insession_auxiliary_graph_v2_containers(
                        auxiliary_graph_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, origin_auxiliary_node_id,
                    origin_node_revision
                ) REFERENCES insession_auxiliary_node_definitions_v2(
                    auxiliary_graph_id, auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_node_retrieval_capability_plans (
            session_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            auxiliary_graph_id TEXT NOT NULL,
            auxiliary_graph_revision INTEGER NOT NULL
                CHECK(auxiliary_graph_revision >= 1),
            plan_sha256 TEXT NOT NULL CHECK(length(plan_sha256) = 64),
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (
                session_id,
                insession_task_id,
                auxiliary_graph_id,
                auxiliary_graph_revision
            ),
            FOREIGN KEY (auxiliary_graph_id, auxiliary_graph_revision)
                REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT
        );

CREATE TABLE insession_auxiliary_node_retrieval_catalog_bindings (
            session_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            auxiliary_graph_id TEXT NOT NULL,
            auxiliary_graph_revision INTEGER NOT NULL
                CHECK(auxiliary_graph_revision >= 1),
            auxiliary_node_id TEXT NOT NULL,
            node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
            plan_sha256 TEXT NOT NULL CHECK(length(plan_sha256) = 64),
            catalog_snapshot_sha256 TEXT NOT NULL
                CHECK(length(catalog_snapshot_sha256) = 64),
            catalog_snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (
                session_id,
                insession_task_id,
                auxiliary_graph_id,
                auxiliary_graph_revision,
                auxiliary_node_id,
                node_revision
            ),
            FOREIGN KEY (
                session_id,
                insession_task_id,
                auxiliary_graph_id,
                auxiliary_graph_revision
            ) REFERENCES insession_auxiliary_node_retrieval_capability_plans(
                session_id,
                insession_task_id,
                auxiliary_graph_id,
                auxiliary_graph_revision
            ) ON DELETE CASCADE ON UPDATE RESTRICT,
            FOREIGN KEY (
                auxiliary_graph_id,
                auxiliary_graph_revision,
                auxiliary_node_id,
                node_revision
            ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                auxiliary_graph_id,
                auxiliary_graph_revision,
                auxiliary_node_id,
                node_revision
            ) ON DELETE CASCADE ON UPDATE RESTRICT
        );

CREATE TABLE insession_auxiliary_node_states_v2 (
                auxiliary_graph_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL,
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'proposed', 'active', 'waiting_user',
                    'waiting_authorization', 'waiting_external', 'interrupted',
                    'completed', 'failed', 'cancelled', 'superseded'
                )),
                state_version INTEGER NOT NULL CHECK(state_version >= 1),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_observation_gaps (
                observation_gap_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                source_type TEXT,
                reason_code TEXT NOT NULL CHECK(length(reason_code) >= 1),
                blocking INTEGER NOT NULL CHECK(blocking IN (0, 1)),
                known_count INTEGER CHECK(known_count IS NULL OR known_count >= 0),
                created_at TEXT NOT NULL,
                UNIQUE(observation_id, ordinal),
                FOREIGN KEY(observation_id)
                    REFERENCES insession_auxiliary_observations(observation_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_observation_items (
                observation_item_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                source_type TEXT NOT NULL CHECK(length(source_type) >= 1),
                source_unit_id TEXT NOT NULL CHECK(length(source_unit_id) >= 1),
                source_revision TEXT NOT NULL CHECK(length(source_revision) >= 1),
                indexed_content_hash TEXT NOT NULL CHECK(
                    length(indexed_content_hash)=64
                    AND indexed_content_hash NOT GLOB '*[^0-9a-f]*'
                ),
                excerpt TEXT NOT NULL CHECK(length(excerpt) >= 1),
                excerpt_sha256 TEXT NOT NULL CHECK(
                    length(excerpt_sha256)=64
                    AND excerpt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                citation_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(observation_id, ordinal),
                UNIQUE(
                    observation_id, source_type, source_unit_id,
                    source_revision, indexed_content_hash
                ),
                FOREIGN KEY(observation_id)
                    REFERENCES insession_auxiliary_observations(observation_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_observations (
                observation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL,
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL,
                work_run_id TEXT,
                attempt_id TEXT,
                tool_result_id TEXT,
                observation_kind TEXT NOT NULL CHECK(observation_kind IN (
                    'task_graph_context', 'directory', 'document', 'visual',
                    'sandbox', 'user_answer', 'host_fact'
                )),
                outcome TEXT NOT NULL CHECK(outcome IN (
                    'success', 'no_match', 'partial', 'blocked', 'failed',
                    'stale'
                )),
                request_fingerprint TEXT NOT NULL CHECK(
                    length(request_fingerprint)=64
                    AND request_fingerprint NOT GLOB '*[^0-9a-f]*'
                ),
                data_version TEXT,
                snapshot_json TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL CHECK(
                    length(snapshot_sha256)=64
                    AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(auxiliary_graph_id, goal_id, observation_id),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    observation_id
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, tool_result_id)
                    REFERENCES insession_work_run_tool_results(
                        work_run_id, tool_result_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (work_run_id IS NULL AND attempt_id IS NULL
                        AND tool_result_id IS NULL)
                    OR (work_run_id IS NOT NULL AND attempt_id IS NOT NULL)
                ),
                CHECK(tool_result_id IS NULL OR work_run_id IS NOT NULL)
            );

CREATE TABLE insession_auxiliary_planning_context_artifacts (
                artifact_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                producer_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    producer_auxiliary_graph_revision >= 1
                ),
                producer_auxiliary_node_id TEXT NOT NULL,
                producer_node_revision INTEGER NOT NULL CHECK(
                    producer_node_revision >= 1
                ),
                producer_work_run_id TEXT,
                producer_attempt_id TEXT,
                producer_tool_result_ids_json TEXT NOT NULL,
                producer_primitive_call_id TEXT,
                scope_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(scope_snapshot_sha256)=64
                    AND scope_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                facts_json TEXT NOT NULL,
                constraints_json TEXT NOT NULL,
                conflicts_json TEXT NOT NULL,
                gaps_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                freshness_manifest_sha256 TEXT NOT NULL CHECK(
                    length(freshness_manifest_sha256)=64
                    AND freshness_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                verification_receipt_id TEXT NOT NULL,
                verification_receipt_sha256 TEXT NOT NULL CHECK(
                    length(verification_receipt_sha256)=64
                    AND verification_receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                artifact_json TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL CHECK(
                    length(artifact_sha256)=64
                    AND artifact_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id,
                    goal_id, artifact_id, artifact_sha256
                ),
                UNIQUE(artifact_id, producer_work_run_id),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    producer_auxiliary_graph_revision,
                    producer_auxiliary_node_id, producer_node_revision,
                    verification_receipt_id, verification_receipt_sha256
                ) REFERENCES insession_auxiliary_context_verification_receipts(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    verification_receipt_id, receipt_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    producer_auxiliary_graph_revision,
                    producer_auxiliary_node_id, producer_node_revision,
                    producer_primitive_call_id
                ) REFERENCES insession_auxiliary_observations(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    observation_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, producer_auxiliary_graph_revision,
                    producer_auxiliary_node_id, producer_node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, producer_auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(producer_work_run_id, producer_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    producer_auxiliary_graph_revision,
                    producer_auxiliary_node_id, producer_node_revision,
                    producer_work_run_id
                ) REFERENCES insession_work_runs(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    work_run_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (producer_work_run_id IS NOT NULL
                        AND producer_attempt_id IS NOT NULL
                        AND producer_primitive_call_id IS NULL)
                    OR (producer_work_run_id IS NULL
                        AND producer_attempt_id IS NULL
                        AND producer_primitive_call_id IS NOT NULL
                        AND producer_tool_result_ids_json='[]')
                )
            );

CREATE TABLE insession_auxiliary_planning_context_tool_results (
                artifact_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                work_run_id TEXT NOT NULL,
                tool_result_id TEXT NOT NULL,
                PRIMARY KEY(artifact_id, ordinal),
                UNIQUE(artifact_id, tool_result_id),
                FOREIGN KEY(artifact_id)
                    REFERENCES insession_auxiliary_planning_context_artifacts(
                        artifact_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(artifact_id, work_run_id)
                    REFERENCES insession_auxiliary_planning_context_artifacts(
                        artifact_id, producer_work_run_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, tool_result_id)
                    REFERENCES insession_work_run_tool_results(
                        work_run_id, tool_result_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_planning_primitive_invocations (
                primitive_call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                invocation_turn_id TEXT NOT NULL,
                primitive_kind TEXT NOT NULL CHECK(primitive_kind IN (
                    'resource_perception'
                )),
                planned_artifact_id TEXT NOT NULL UNIQUE,
                planned_verification_receipt_id TEXT NOT NULL UNIQUE,
                authority_snapshot_id TEXT NOT NULL,
                scope_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(scope_snapshot_sha256)=64
                    AND scope_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                expected_task_state_version INTEGER NOT NULL CHECK(
                    expected_task_state_version >= 1
                ),
                expected_node_state_version INTEGER NOT NULL CHECK(
                    expected_node_state_version >= 1
                ),
                expected_control_state_version INTEGER NOT NULL CHECK(
                    expected_control_state_version >= 1
                ),
                expected_goal_state_version INTEGER NOT NULL CHECK(
                    expected_goal_state_version >= 1
                ),
                expected_revision_state_version INTEGER NOT NULL CHECK(
                    expected_revision_state_version >= 1
                ),
                expected_budget_state_version INTEGER NOT NULL CHECK(
                    expected_budget_state_version >= 1
                ),
                budget_ledger_id TEXT NOT NULL,
                authority_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(authority_snapshot_sha256)=64
                    AND authority_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                structure_sha256 TEXT NOT NULL CHECK(
                    length(structure_sha256)=64
                    AND structure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(budget_snapshot_sha256)=64
                    AND budget_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                logical_request_json TEXT NOT NULL CHECK(
                    length(logical_request_json)>0
                ),
                logical_request_sha256 TEXT NOT NULL CHECK(
                    length(logical_request_sha256)=64
                    AND logical_request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                state_guard_sha256 TEXT NOT NULL CHECK(
                    length(state_guard_sha256)=64
                    AND state_guard_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                invocation_json TEXT NOT NULL CHECK(length(invocation_json)>0),
                invocation_sha256 TEXT NOT NULL CHECK(
                    length(invocation_sha256)=64
                    AND invocation_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                status TEXT NOT NULL CHECK(status IN ('reserved', 'settled')),
                settled_observation_id TEXT UNIQUE,
                settled_artifact_id TEXT UNIQUE,
                settlement_sha256 TEXT CHECK(
                    settlement_sha256 IS NULL OR (
                        length(settlement_sha256)=64
                        AND settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                    )
                ),
                reserved_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ),
                UNIQUE(primitive_call_id, invocation_sha256),
                CHECK(
                    (status='reserved' AND settled_observation_id IS NULL
                        AND settled_artifact_id IS NULL
                        AND settlement_sha256 IS NULL AND settled_at IS NULL)
                    OR
                    (status='settled'
                        AND settled_observation_id=primitive_call_id
                        AND settled_artifact_id=planned_artifact_id
                        AND settlement_sha256 IS NOT NULL
                        AND settled_at IS NOT NULL)
                ),
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, authority_snapshot_sha256
                ) REFERENCES insession_auxiliary_authority_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id, structure_sha256
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id, structure_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    goal_id, budget_ledger_id, expected_budget_state_version,
                    budget_snapshot_sha256
                ) REFERENCES insession_auxiliary_goal_budget_snapshots(
                    goal_id, budget_ledger_id, state_version, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(settled_observation_id)
                    REFERENCES insession_auxiliary_observations(observation_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(settled_artifact_id)
                    REFERENCES insession_auxiliary_planning_context_artifacts(
                        artifact_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_replan_trigger_applications (
                apply_id TEXT PRIMARY KEY,
                trigger_id TEXT NOT NULL UNIQUE,
                trigger_receipt_sha256 TEXT NOT NULL CHECK(
                    length(trigger_receipt_sha256)=64
                    AND trigger_receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                source_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    source_auxiliary_graph_revision >= 1
                ),
                applied_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    applied_auxiliary_graph_revision=
                        source_auxiliary_graph_revision+1
                ),
                applied_structure_sha256 TEXT NOT NULL CHECK(
                    length(applied_structure_sha256)=64
                    AND applied_structure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                revision_reason TEXT NOT NULL CHECK(
                    revision_reason='verification_failed'
                ),
                consumed_turn_id TEXT NOT NULL,
                command_json TEXT NOT NULL CHECK(length(command_json)>0),
                command_sha256 TEXT NOT NULL CHECK(
                    length(command_sha256)=64
                    AND command_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                FOREIGN KEY(trigger_id)
                    REFERENCES insession_auxiliary_replan_triggers(trigger_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, applied_auxiliary_graph_revision
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, consumed_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_replan_triggers (
                trigger_id TEXT PRIMARY KEY,
                create_apply_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                source_auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    source_auxiliary_graph_revision >= 1
                ),
                source_structure_sha256 TEXT NOT NULL CHECK(
                    length(source_structure_sha256)=64
                    AND source_structure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                source_authority_snapshot_id TEXT NOT NULL,
                source_authority_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(source_authority_snapshot_sha256)=64
                    AND source_authority_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_authority_projection_sha256 TEXT NOT NULL CHECK(
                    length(semantic_authority_projection_sha256)=64
                    AND semantic_authority_projection_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(semantic_prompt_payload_sha256)=64
                    AND semantic_prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                task_graph_proposal_sha256 TEXT NOT NULL CHECK(
                    length(task_graph_proposal_sha256)=64
                    AND task_graph_proposal_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_settlement_id TEXT NOT NULL UNIQUE,
                semantic_settlement_sha256 TEXT NOT NULL CHECK(
                    length(semantic_settlement_sha256)=64
                    AND semantic_settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_result_sha256s_json TEXT NOT NULL CHECK(
                    length(semantic_result_sha256s_json)>0
                ),
                budget_ledger_id TEXT NOT NULL,
                budget_state_version INTEGER NOT NULL CHECK(
                    budget_state_version >= 1
                ),
                budget_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(budget_snapshot_sha256)=64
                    AND budget_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                evidence_epoch_sha256 TEXT NOT NULL CHECK(
                    length(evidence_epoch_sha256)=64
                    AND evidence_epoch_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_disposition TEXT NOT NULL CHECK(
                    semantic_disposition IN ('revise', 'blocked')
                ),
                trigger_reason TEXT NOT NULL CHECK(trigger_reason IN (
                    'semantic_revision_required',
                    'semantic_evidence_blocked'
                )),
                revision_reason TEXT NOT NULL CHECK(
                    revision_reason='verification_failed'
                ),
                created_turn_id TEXT NOT NULL,
                command_json TEXT NOT NULL CHECK(length(command_json)>0),
                command_sha256 TEXT NOT NULL CHECK(
                    length(command_sha256)=64
                    AND command_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id,
                    goal_id, source_auxiliary_graph_revision, trigger_id
                ),
                CHECK(
                    (semantic_disposition='revise'
                        AND trigger_reason='semantic_revision_required')
                    OR (semantic_disposition='blocked'
                        AND trigger_reason='semantic_evidence_blocked')
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(auxiliary_graph_id, source_auxiliary_graph_revision)
                    REFERENCES insession_auxiliary_graph_revision_snapshots(
                        auxiliary_graph_id, auxiliary_graph_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    goal_id, budget_ledger_id, budget_state_version,
                    budget_snapshot_sha256
                ) REFERENCES insession_auxiliary_goal_budget_snapshots(
                    goal_id, budget_ledger_id, state_version, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(semantic_settlement_id)
                    REFERENCES insession_auxiliary_semantic_quorum_settlements(
                        settlement_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_semantic_quorum_reviewers (
                settlement_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                frozen_prompt_payload_sha256 TEXT NOT NULL,
                review_policy_sha256 TEXT NOT NULL CHECK(
                    length(review_policy_sha256)=64
                    AND review_policy_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                reviewer_ordinal INTEGER NOT NULL CHECK(
                    reviewer_ordinal BETWEEN 1 AND 2
                ),
                required_reviewer_count INTEGER NOT NULL CHECK(
                    required_reviewer_count BETWEEN 1 AND 2
                    AND reviewer_ordinal <= required_reviewer_count
                ),
                verification_request_id TEXT NOT NULL,
                request_binding_sha256 TEXT NOT NULL,
                verification_result_id TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                PRIMARY KEY(settlement_id, reviewer_ordinal),
                UNIQUE(settlement_id, verification_request_id),
                UNIQUE(settlement_id, verification_result_id),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, settlement_id,
                    frozen_prompt_payload_sha256, review_policy_sha256,
                    required_reviewer_count
                )
                    REFERENCES insession_auxiliary_semantic_quorum_settlements(
                        session_id, insession_task_id, auxiliary_graph_id, goal_id,
                        auxiliary_graph_revision, settlement_id,
                        frozen_prompt_payload_sha256, review_policy_sha256,
                        required_reviewer_count
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    verification_request_id, request_binding_sha256,
                    frozen_prompt_payload_sha256, review_policy_sha256,
                    verification_result_id,
                    result_sha256, reviewer_ordinal, required_reviewer_count
                ) REFERENCES insession_auxiliary_semantic_verification_results(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    verification_request_id, request_binding_sha256,
                    prompt_payload_sha256, review_policy_sha256,
                    verification_result_id,
                    result_sha256, reviewer_ordinal, required_reviewer_count
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_semantic_quorum_settlements (
                settlement_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                required_reviewer_count INTEGER NOT NULL CHECK(
                    required_reviewer_count BETWEEN 1 AND 2
                ),
                frozen_prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(frozen_prompt_payload_sha256)=64
                    AND frozen_prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                review_policy_json TEXT NOT NULL,
                review_policy_sha256 TEXT NOT NULL CHECK(
                    length(review_policy_sha256)=64
                    AND review_policy_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                policy_source_sha256 TEXT NOT NULL CHECK(
                    length(policy_source_sha256)=64
                    AND policy_source_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                distinct_document_count INTEGER NOT NULL CHECK(
                    distinct_document_count BETWEEN 0 AND 1024
                ),
                has_visual_input INTEGER NOT NULL CHECK(
                    has_visual_input IN (0, 1)
                ),
                requires_protected_effect INTEGER NOT NULL CHECK(
                    requires_protected_effect IN (0, 1)
                ),
                modifies_executed_task_graph INTEGER NOT NULL CHECK(
                    modifies_executed_task_graph IN (0, 1)
                ),
                request_ids_json TEXT NOT NULL,
                result_ids_json TEXT NOT NULL,
                host_disposition TEXT NOT NULL CHECK(host_disposition IN (
                    'pass', 'revise', 'blocked'
                )),
                settlement_json TEXT NOT NULL,
                settlement_sha256 TEXT NOT NULL CHECK(
                    length(settlement_sha256)=64
                    AND settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    auxiliary_graph_id, goal_id, auxiliary_graph_revision,
                    settlement_id
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, frozen_prompt_payload_sha256
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, settlement_id,
                    frozen_prompt_payload_sha256, review_policy_sha256,
                    required_reviewer_count
                ),
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_semantic_request_context_artifacts (
                verification_request_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                artifact_alias TEXT NOT NULL CHECK(
                    length(artifact_alias) BETWEEN 1 AND 64
                    AND substr(artifact_alias, 1, 1) GLOB '[a-z]'
                    AND artifact_alias NOT GLOB '*[^a-z0-9_]*'
                ),
                artifact_id TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL CHECK(
                    length(artifact_sha256)=64
                    AND artifact_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                projection_json TEXT NOT NULL,
                projection_sha256 TEXT NOT NULL CHECK(
                    length(projection_sha256)=64
                    AND projection_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                PRIMARY KEY(verification_request_id, ordinal),
                UNIQUE(verification_request_id, artifact_alias),
                UNIQUE(verification_request_id, artifact_id),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    verification_request_id
                )
                    REFERENCES insession_auxiliary_semantic_verification_requests(
                        session_id, insession_task_id, auxiliary_graph_id, goal_id,
                        verification_request_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    artifact_id, artifact_sha256
                ) REFERENCES insession_auxiliary_planning_context_artifacts(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    artifact_id, artifact_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_semantic_verification_requests (
                verification_request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_graph_structure_sha256 TEXT NOT NULL CHECK(
                    length(auxiliary_graph_structure_sha256)=64
                    AND auxiliary_graph_structure_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                logical_call_id TEXT NOT NULL,
                verification_profile_id TEXT NOT NULL,
                reviewer_ordinal INTEGER NOT NULL CHECK(
                    reviewer_ordinal BETWEEN 1 AND 2
                ),
                required_reviewer_count INTEGER NOT NULL CHECK(
                    required_reviewer_count BETWEEN 1 AND 2
                    AND reviewer_ordinal <= required_reviewer_count
                ),
                authority_snapshot_id TEXT NOT NULL,
                authority_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(authority_snapshot_sha256)=64
                    AND authority_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_ledger_id TEXT NOT NULL,
                budget_state_version INTEGER NOT NULL CHECK(
                    budget_state_version >= 1
                ),
                budget_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(budget_snapshot_sha256)=64
                    AND budget_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                capability_catalog_snapshot_id TEXT NOT NULL,
                capability_catalog_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(capability_catalog_snapshot_sha256)=64
                    AND capability_catalog_snapshot_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                capability_projection_sha256 TEXT NOT NULL CHECK(
                    length(capability_projection_sha256)=64
                    AND capability_projection_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                prompt_payload_json TEXT NOT NULL,
                prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(prompt_payload_sha256)=64
                    AND prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                review_policy_json TEXT NOT NULL,
                review_policy_sha256 TEXT NOT NULL CHECK(
                    length(review_policy_sha256)=64
                    AND review_policy_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                policy_source_sha256 TEXT NOT NULL CHECK(
                    length(policy_source_sha256)=64
                    AND policy_source_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                distinct_document_count INTEGER NOT NULL CHECK(
                    distinct_document_count BETWEEN 0 AND 1024
                ),
                has_visual_input INTEGER NOT NULL CHECK(
                    has_visual_input IN (0, 1)
                ),
                requires_protected_effect INTEGER NOT NULL CHECK(
                    requires_protected_effect IN (0, 1)
                ),
                modifies_executed_task_graph INTEGER NOT NULL CHECK(
                    modifies_executed_task_graph IN (0, 1)
                ),
                task_graph_proposal_sha256 TEXT NOT NULL CHECK(
                    length(task_graph_proposal_sha256)=64
                    AND task_graph_proposal_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                blocking_gap_aliases_json TEXT NOT NULL,
                non_blocking_gap_aliases_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    verification_request_id, binding_sha256, logical_call_id,
                    verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    verification_request_id
                ),
                UNIQUE(
                    session_id, verification_request_id, binding_sha256,
                    logical_call_id, verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ),
                UNIQUE(
                    verification_request_id, binding_sha256,
                    prompt_payload_sha256, review_policy_sha256, logical_call_id,
                    verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, verification_request_id,
                    binding_sha256, prompt_payload_sha256,
                    review_policy_sha256, logical_call_id,
                    verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id,
                    auxiliary_graph_structure_sha256
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, goal_id, structure_sha256
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, authority_snapshot_sha256
                ) REFERENCES insession_auxiliary_authority_snapshots(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    authority_snapshot_id, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    goal_id, budget_ledger_id, budget_state_version,
                    budget_snapshot_sha256
                ) REFERENCES insession_auxiliary_goal_budget_snapshots(
                    goal_id, budget_ledger_id, state_version, snapshot_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    capability_catalog_snapshot_id,
                    capability_catalog_snapshot_sha256,
                    capability_projection_sha256
                ) REFERENCES insession_auxiliary_capability_catalog_projections(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    capability_catalog_snapshot_id,
                    capability_catalog_snapshot_sha256, projection_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_semantic_verification_results (
                verification_result_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                verification_request_id TEXT NOT NULL,
                request_binding_sha256 TEXT NOT NULL,
                prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(prompt_payload_sha256)=64
                    AND prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                review_policy_sha256 TEXT NOT NULL CHECK(
                    length(review_policy_sha256)=64
                    AND review_policy_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                logical_call_id TEXT NOT NULL,
                verification_profile_id TEXT NOT NULL,
                reviewer_ordinal INTEGER NOT NULL CHECK(
                    reviewer_ordinal BETWEEN 1 AND 2
                ),
                required_reviewer_count INTEGER NOT NULL CHECK(
                    required_reviewer_count BETWEEN 1 AND 2
                    AND reviewer_ordinal <= required_reviewer_count
                ),
                items_json TEXT NOT NULL,
                host_disposition TEXT NOT NULL CHECK(host_disposition IN (
                    'pass', 'revise', 'blocked'
                )),
                result_json TEXT NOT NULL,
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(verification_request_id),
                UNIQUE(verification_request_id, verification_result_id),
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    verification_request_id, request_binding_sha256,
                    prompt_payload_sha256, review_policy_sha256,
                    verification_result_id,
                    result_sha256, reviewer_ordinal, required_reviewer_count
                ),
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    verification_request_id, request_binding_sha256,
                    prompt_payload_sha256, review_policy_sha256, logical_call_id,
                    verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ) REFERENCES insession_auxiliary_semantic_verification_requests(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision,
                    verification_request_id, binding_sha256,
                    prompt_payload_sha256, review_policy_sha256, logical_call_id,
                    verification_profile_id, reviewer_ordinal,
                    required_reviewer_count
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE "insession_auxiliary_v2_execution_apply_receipts" (
                apply_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL CHECK(operation IN (
                    'commit_waiting_user_attempt',
                    'continue_waiting_user_and_start_attempt',
                    'resume_active_attempt',
                    'resume_verification'
                )),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                execution_subject_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                invocation_turn_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL CHECK(
                    length(payload_sha256)=64
                    AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_json TEXT NOT NULL CHECK(length(result_json)>0),
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(session_id, work_run_id, apply_id),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, work_run_id, execution_subject_id
                ) REFERENCES insession_work_runs(
                    session_id, work_run_id, execution_subject_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, execution_subject_id
                ) REFERENCES insession_execution_subjects(
                    session_id, insession_task_id, execution_subject_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_v2_finish_gate_receipts (
                finish_gate_receipt_id TEXT PRIMARY KEY,
                apply_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                structure_sha256 TEXT NOT NULL CHECK(
                    length(structure_sha256)=64
                    AND structure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                base_task_graph_revision INTEGER CHECK(
                    base_task_graph_revision IS NULL
                    OR base_task_graph_revision >= 1
                ),
                target_task_graph_revision INTEGER NOT NULL CHECK(
                    target_task_graph_revision >= 1
                ),
                terminal_auxiliary_node_id TEXT NOT NULL,
                terminal_node_revision INTEGER NOT NULL CHECK(
                    terminal_node_revision >= 1
                ),
                terminal_completion_id TEXT NOT NULL UNIQUE,
                terminal_work_run_id TEXT NOT NULL UNIQUE,
                terminal_output_revision INTEGER NOT NULL CHECK(
                    terminal_output_revision >= 1
                ),
                required_completion_ids_json TEXT NOT NULL,
                required_completion_ids_sha256 TEXT NOT NULL CHECK(
                    length(required_completion_ids_sha256)=64
                    AND required_completion_ids_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                proposal_sha256 TEXT NOT NULL CHECK(
                    length(proposal_sha256)=64
                    AND proposal_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                validation_context_sha256 TEXT NOT NULL CHECK(
                    length(validation_context_sha256)=64
                    AND validation_context_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                production_evaluation_json TEXT NOT NULL,
                production_evaluation_sha256 TEXT NOT NULL CHECK(
                    length(production_evaluation_sha256)=64
                    AND production_evaluation_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_settlement_id TEXT NOT NULL UNIQUE,
                semantic_prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(semantic_prompt_payload_sha256)=64
                    AND semantic_prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_review_policy_sha256 TEXT NOT NULL CHECK(
                    length(semantic_review_policy_sha256)=64
                    AND semantic_review_policy_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_settlement_sha256 TEXT NOT NULL CHECK(
                    length(semantic_settlement_sha256)=64
                    AND semantic_settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(budget_snapshot_sha256)=64
                    AND budget_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                budget_disposition TEXT NOT NULL CHECK(
                    budget_disposition IN (
                        'within_limit', 'soft_limit_reached'
                    )
                ),
                readiness_status TEXT NOT NULL CHECK(
                    readiness_status IN ('proposal_ready', 'gapped_ready')
                ),
                non_blocking_gap_ids_json TEXT NOT NULL,
                non_blocking_gap_ids_sha256 TEXT NOT NULL CHECK(
                    length(non_blocking_gap_ids_sha256)=64
                    AND non_blocking_gap_ids_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    auxiliary_graph_id, goal_id, auxiliary_graph_revision,
                    finish_gate_receipt_id
                ),
                FOREIGN KEY(apply_id)
                    REFERENCES
                        insession_auxiliary_v2_terminal_seal_apply_receipts(
                            apply_id
                        ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    terminal_auxiliary_node_id, terminal_node_revision,
                    terminal_completion_id
                ) REFERENCES insession_auxiliary_node_completions_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision, completion_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(terminal_work_run_id, terminal_output_revision)
                    REFERENCES insession_work_run_output_windows(
                        work_run_id, output_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(semantic_settlement_id)
                    REFERENCES
                        insession_auxiliary_semantic_quorum_settlements(
                            settlement_id
                        ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (base_task_graph_revision IS NULL
                        AND target_task_graph_revision=1)
                    OR target_task_graph_revision=base_task_graph_revision+1
                )
            );

CREATE TABLE "insession_auxiliary_v2_node_execution_subject_bindings" (
                binding_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                executor_kind TEXT NOT NULL CHECK(executor_kind IN (
                    'model_work_run', 'user_gate', 'terminal_planner'
                )),
                definition_sha256 TEXT NOT NULL CHECK(
                    length(definition_sha256)=64
                    AND definition_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision
                ),
                UNIQUE(
                    auxiliary_graph_id, auxiliary_node_id, node_revision,
                    definition_sha256
                ),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    insession_task_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_node_id, node_revision,
                    definition_sha256
                ) REFERENCES insession_auxiliary_node_definitions_v2(
                    auxiliary_graph_id, auxiliary_node_id, node_revision,
                    definition_sha256
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_v2_task_graph_commit_receipts (
                apply_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL CHECK(
                    operation='commit_auxiliary_v2_task_graph_proposal'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                source_turn_id TEXT NOT NULL,
                terminal_proposal_receipt_id TEXT NOT NULL UNIQUE,
                finish_gate_receipt_id TEXT NOT NULL UNIQUE,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                base_task_graph_revision INTEGER CHECK(
                    base_task_graph_revision IS NULL
                    OR base_task_graph_revision >= 1
                ),
                committed_task_graph_revision INTEGER NOT NULL CHECK(
                    committed_task_graph_revision >= 1
                ),
                expected_task_state_version INTEGER NOT NULL CHECK(
                    expected_task_state_version >= 1
                ),
                committed_task_state_version INTEGER NOT NULL CHECK(
                    committed_task_state_version >= 1
                ),
                expected_window_revision INTEGER NOT NULL CHECK(
                    expected_window_revision >= 1
                ),
                committed_window_state_version INTEGER NOT NULL CHECK(
                    committed_window_state_version >= 1
                ),
                turn_task_link_revision INTEGER NOT NULL CHECK(
                    turn_task_link_revision >= 0
                ),
                command_sha256 TEXT NOT NULL CHECK(
                    length(command_sha256)=64
                    AND command_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                proposal_sha256 TEXT NOT NULL CHECK(
                    length(proposal_sha256)=64
                    AND proposal_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                validation_context_sha256 TEXT NOT NULL CHECK(
                    length(validation_context_sha256)=64
                    AND validation_context_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                production_evaluation_sha256 TEXT NOT NULL CHECK(
                    length(production_evaluation_sha256)=64
                    AND production_evaluation_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                semantic_settlement_id TEXT NOT NULL,
                semantic_settlement_sha256 TEXT NOT NULL CHECK(
                    length(semantic_settlement_sha256)=64
                    AND semantic_settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                transition_json TEXT,
                transition_sha256 TEXT CHECK(
                    transition_sha256 IS NULL OR (
                        length(transition_sha256)=64
                        AND transition_sha256 NOT GLOB '*[^0-9a-f]*'
                    )
                ),
                target_snapshot_json TEXT NOT NULL CHECK(
                    length(target_snapshot_json)>0
                ),
                target_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(target_snapshot_sha256)=64
                    AND target_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_json TEXT NOT NULL CHECK(length(result_json)>0),
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                CHECK(
                    (base_task_graph_revision IS NULL
                        AND committed_task_graph_revision=1
                        AND transition_json IS NULL
                        AND transition_sha256 IS NULL)
                    OR (base_task_graph_revision IS NOT NULL
                        AND committed_task_graph_revision=
                            base_task_graph_revision+1
                        AND transition_json IS NOT NULL
                        AND transition_sha256 IS NOT NULL)
                ),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, source_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(terminal_proposal_receipt_id)
                    REFERENCES
                        insession_auxiliary_v2_terminal_proposal_receipts(
                            terminal_proposal_receipt_id
                        ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(finish_gate_receipt_id)
                    REFERENCES insession_auxiliary_v2_finish_gate_receipts(
                        finish_gate_receipt_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(semantic_settlement_id)
                    REFERENCES insession_auxiliary_semantic_quorum_settlements(
                        settlement_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, committed_task_graph_revision
                ) REFERENCES insession_task_graph_revisions(
                    insession_task_id, graph_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, base_task_graph_revision
                ) REFERENCES insession_task_graph_revisions(
                    insession_task_id, graph_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) REFERENCES insession_auxiliary_graph_revision_snapshots(
                    auxiliary_graph_id, auxiliary_graph_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_v2_task_graph_node_carry_receipts (
                carry_receipt_id TEXT PRIMARY KEY,
                apply_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_task_graph_revision INTEGER NOT NULL CHECK(
                    base_task_graph_revision >= 1
                ),
                target_task_graph_revision INTEGER NOT NULL CHECK(
                    target_task_graph_revision=
                        base_task_graph_revision+1
                ),
                insession_task_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                source_delivery_id TEXT NOT NULL,
                definition_sha256 TEXT NOT NULL CHECK(
                    length(definition_sha256)=64
                    AND definition_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                dependency_delivery_ids_json TEXT NOT NULL,
                dependency_closure_sha256 TEXT NOT NULL CHECK(
                    length(dependency_closure_sha256)=64
                    AND dependency_closure_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                source_authority_sha256 TEXT NOT NULL CHECK(
                    length(source_authority_sha256)=64
                    AND source_authority_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                capability_catalog_sha256 TEXT NOT NULL CHECK(
                    length(capability_catalog_sha256)=64
                    AND capability_catalog_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                freshness_authority_sha256 TEXT NOT NULL CHECK(
                    length(freshness_authority_sha256)=64
                    AND freshness_authority_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(
                    apply_id, insession_task_node_id, node_revision
                ),
                UNIQUE(
                    insession_task_id, target_task_graph_revision,
                    insession_task_node_id, node_revision
                ),
                FOREIGN KEY(apply_id)
                    REFERENCES
                        insession_auxiliary_v2_task_graph_commit_receipts(
                            apply_id
                        ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(source_delivery_id)
                    REFERENCES insession_task_node_deliveries(delivery_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, base_task_graph_revision,
                    insession_task_node_id, node_revision
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, target_task_graph_revision,
                    insession_task_node_id, node_revision
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_v2_terminal_proposal_receipts (
                terminal_proposal_receipt_id TEXT PRIMARY KEY,
                finish_gate_receipt_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                terminal_completion_id TEXT NOT NULL UNIQUE,
                proposal_schema_version TEXT NOT NULL CHECK(
                    proposal_schema_version=
                        'insession-task-graph-revision-v2'
                ),
                proposal_json TEXT NOT NULL CHECK(length(proposal_json)>0),
                proposal_sha256 TEXT NOT NULL CHECK(
                    length(proposal_sha256)=64
                    AND proposal_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                output_window_json TEXT NOT NULL CHECK(
                    length(output_window_json)>0
                ),
                output_window_sha256 TEXT NOT NULL CHECK(
                    length(output_window_sha256)=64
                    AND output_window_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, auxiliary_graph_id,
                    goal_id, auxiliary_graph_revision,
                    terminal_proposal_receipt_id
                ),
                FOREIGN KEY(finish_gate_receipt_id)
                    REFERENCES insession_auxiliary_v2_finish_gate_receipts(
                        finish_gate_receipt_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(terminal_completion_id)
                    REFERENCES insession_auxiliary_node_completions_v2(
                        completion_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_auxiliary_v2_terminal_seal_apply_receipts (
                apply_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL CHECK(
                    operation='seal_terminal_proposal'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                terminal_auxiliary_node_id TEXT NOT NULL,
                terminal_node_revision INTEGER NOT NULL CHECK(
                    terminal_node_revision >= 1
                ),
                terminal_completion_id TEXT NOT NULL,
                semantic_settlement_id TEXT NOT NULL,
                invocation_turn_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL CHECK(
                    length(payload_sha256)=64
                    AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_json TEXT NOT NULL CHECK(length(result_json)>0),
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id, apply_id),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    terminal_auxiliary_node_id, terminal_node_revision,
                    terminal_completion_id
                ) REFERENCES insession_auxiliary_node_completions_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision, completion_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, semantic_settlement_id
                ) REFERENCES insession_auxiliary_semantic_quorum_settlements(
                    auxiliary_graph_id, goal_id,
                    auxiliary_graph_revision, settlement_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE "insession_auxiliary_v2_waiting_user_answer_bindings" (
                answer_binding_id TEXT PRIMARY KEY,
                apply_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                execution_subject_id TEXT NOT NULL,
                auxiliary_graph_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                auxiliary_graph_revision INTEGER NOT NULL CHECK(
                    auxiliary_graph_revision >= 1
                ),
                auxiliary_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                question_attempt_id TEXT NOT NULL,
                question_sha256 TEXT NOT NULL CHECK(
                    length(question_sha256)=64
                    AND question_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                answer_attempt_id TEXT NOT NULL,
                answer_source_turn_id TEXT NOT NULL,
                answer_source_message_id TEXT NOT NULL,
                answer_source_turn_idx INTEGER NOT NULL CHECK(
                    answer_source_turn_idx >= 0
                ),
                answer_source_content_sha256 TEXT NOT NULL CHECK(
                    length(answer_source_content_sha256)=64
                    AND answer_source_content_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                answer_source_utf8_bytes INTEGER NOT NULL CHECK(
                    answer_source_utf8_bytes >= 1
                ),
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                created_at TEXT NOT NULL,
                UNIQUE(work_run_id, question_attempt_id),
                UNIQUE(work_run_id, answer_attempt_id),
                UNIQUE(session_id, answer_source_turn_id),
                UNIQUE(session_id, answer_source_message_id),
                FOREIGN KEY(apply_id)
                    REFERENCES
                        "insession_auxiliary_v2_execution_apply_receipts"(
                            apply_id
                        ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, work_run_id, execution_subject_id
                ) REFERENCES insession_work_runs(
                    session_id, work_run_id, execution_subject_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, question_attempt_id)
                    REFERENCES insession_work_run_attempts(
                        work_run_id, attempt_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, answer_attempt_id)
                    REFERENCES insession_work_run_attempts(
                        work_run_id, attempt_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, answer_source_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(answer_source_message_id)
                    REFERENCES runtime_turn_inputs(message_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) REFERENCES insession_auxiliary_graph_revision_nodes_v2(
                    auxiliary_graph_id, auxiliary_graph_revision,
                    auxiliary_node_id, node_revision
                ) ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_execution_subjects (
                execution_subject_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                subject_kind TEXT NOT NULL CHECK(subject_kind IN (
                    'task_node', 'auxiliary_node'
                )),
                subject_contract_version TEXT NOT NULL CHECK(
                    subject_contract_version IN (
                        'task_node_v1', 'auxiliary_node_v2'
                    )
                ),
                task_node_binding_id TEXT UNIQUE,
                auxiliary_v2_binding_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, execution_subject_id),
                UNIQUE(
                    session_id, insession_task_id, execution_subject_id
                ),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(task_node_binding_id)
                    REFERENCES insession_task_node_execution_subject_bindings(
                        binding_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(auxiliary_v2_binding_id)
                    REFERENCES insession_auxiliary_v2_node_execution_subject_bindings(
                        binding_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                CHECK(
                    (subject_kind='task_node'
                        AND subject_contract_version='task_node_v1'
                        AND task_node_binding_id IS NOT NULL
                        AND auxiliary_v2_binding_id IS NULL)
                    OR
                    (subject_kind='auxiliary_node'
                        AND subject_contract_version='auxiliary_node_v2'
                        AND task_node_binding_id IS NULL
                        AND auxiliary_v2_binding_id IS NOT NULL)
                )
            );

CREATE TABLE insession_runtime_model_call_settlement_receipts (
                settle_apply_id TEXT PRIMARY KEY,
                settlement_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                logical_call_id TEXT NOT NULL,
                physical_attempt_id TEXT NOT NULL UNIQUE,
                physical_request_binding_sha256 TEXT NOT NULL CHECK(
                    length(physical_request_binding_sha256)=64
                    AND physical_request_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_ordinal INTEGER NOT NULL CHECK(
                    physical_ordinal BETWEEN 1 AND 32
                ),
                outcome TEXT NOT NULL CHECK(outcome IN (
                    'succeeded', 'retryable_failure',
                    'terminal_failure', 'uncertain'
                )),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                settlement_json TEXT NOT NULL CHECK(length(settlement_json)>0),
                settled_turn_id TEXT NOT NULL,
                settled_at TEXT NOT NULL,
                FOREIGN KEY(
                    session_id, logical_call_id, physical_attempt_id,
                    physical_request_binding_sha256, physical_ordinal
                ) REFERENCES insession_runtime_model_physical_attempts(
                    session_id, logical_call_id, physical_attempt_id,
                    binding_sha256, physical_ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, settled_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_model_logical_calls (
                logical_call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT,
                auxiliary_graph_id TEXT,
                goal_id TEXT,
                execution_subject_id TEXT,
                invocation_turn_id TEXT NOT NULL,
                call_kind TEXT NOT NULL CHECK(
                    length(call_kind) BETWEEN 1 AND 200
                ),
                purpose TEXT NOT NULL CHECK(
                    length(purpose) BETWEEN 1 AND 200
                ),
                provider TEXT NOT NULL CHECK(
                    length(provider) BETWEEN 1 AND 200
                ),
                model TEXT NOT NULL CHECK(length(model) BETWEEN 1 AND 300),
                endpoint_fingerprint TEXT NOT NULL CHECK(
                    length(endpoint_fingerprint)=64
                    AND endpoint_fingerprint NOT GLOB '*[^0-9a-f]*'
                ),
                request_contract TEXT NOT NULL CHECK(
                    length(request_contract) BETWEEN 1 AND 200
                ),
                request_sha256 TEXT NOT NULL CHECK(
                    length(request_sha256)=64
                    AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                typed_result_contract TEXT NOT NULL CHECK(
                    length(typed_result_contract) BETWEEN 1 AND 200
                ),
                max_physical_attempts INTEGER NOT NULL CHECK(
                    max_physical_attempts BETWEEN 1 AND 32
                ),
                state_guard_sha256 TEXT NOT NULL CHECK(
                    length(state_guard_sha256)=64
                    AND state_guard_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                request_json TEXT NOT NULL CHECK(length(request_json)>0),
                created_at TEXT NOT NULL,
                UNIQUE(session_id, logical_call_id),
                UNIQUE(
                    session_id, logical_call_id, binding_sha256, provider,
                    model, endpoint_fingerprint, request_sha256
                ),
                CHECK(
                    (auxiliary_graph_id IS NULL AND goal_id IS NULL)
                    OR
                    (auxiliary_graph_id IS NOT NULL AND goal_id IS NOT NULL
                        AND insession_task_id IS NOT NULL)
                ),
                CHECK(
                    execution_subject_id IS NULL
                    OR insession_task_id IS NOT NULL
                ),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) REFERENCES insession_auxiliary_graph_goals(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, execution_subject_id
                ) REFERENCES insession_execution_subjects(
                    session_id, insession_task_id, execution_subject_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_model_physical_attempts (
                physical_attempt_id TEXT PRIMARY KEY,
                physical_attempt_key TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                logical_call_id TEXT NOT NULL,
                logical_request_binding_sha256 TEXT NOT NULL CHECK(
                    length(logical_request_binding_sha256)=64
                    AND logical_request_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_ordinal INTEGER NOT NULL CHECK(
                    physical_ordinal BETWEEN 1 AND 32
                ),
                started_turn_id TEXT NOT NULL,
                provider TEXT NOT NULL CHECK(
                    length(provider) BETWEEN 1 AND 200
                ),
                model TEXT NOT NULL CHECK(length(model) BETWEEN 1 AND 300),
                endpoint_fingerprint TEXT NOT NULL CHECK(
                    length(endpoint_fingerprint)=64
                    AND endpoint_fingerprint NOT GLOB '*[^0-9a-f]*'
                ),
                request_sha256 TEXT NOT NULL CHECK(
                    length(request_sha256)=64
                    AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                provider_idempotency_key TEXT CHECK(
                    provider_idempotency_key IS NULL
                    OR length(provider_idempotency_key) BETWEEN 1 AND 300
                ),
                dispatch_authority_sha256 TEXT NOT NULL CHECK(
                    length(dispatch_authority_sha256)=64
                    AND dispatch_authority_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_request_json TEXT NOT NULL CHECK(
                    length(physical_request_json)>0
                ),
                status TEXT NOT NULL CHECK(status IN (
                    'pending', 'succeeded', 'retryable_failure',
                    'terminal_failure', 'uncertain'
                )),
                settlement_apply_id TEXT UNIQUE,
                started_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(logical_call_id, physical_ordinal),
                UNIQUE(
                    session_id, logical_call_id, physical_attempt_id,
                    binding_sha256, physical_ordinal
                ),
                CHECK(
                    (status='pending' AND settlement_apply_id IS NULL
                        AND settled_at IS NULL)
                    OR
                    (status<>'pending' AND settlement_apply_id IS NOT NULL
                        AND settled_at IS NOT NULL)
                ),
                FOREIGN KEY(
                    session_id, logical_call_id,
                    logical_request_binding_sha256, provider, model,
                    endpoint_fingerprint, request_sha256
                ) REFERENCES insession_runtime_model_logical_calls(
                    session_id, logical_call_id, binding_sha256, provider,
                    model, endpoint_fingerprint, request_sha256
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, started_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_model_rejected_outputs (
                session_id TEXT NOT NULL,
                logical_call_id TEXT NOT NULL,
                physical_attempt_id TEXT NOT NULL PRIMARY KEY,
                physical_ordinal INTEGER NOT NULL CHECK(
                    physical_ordinal BETWEEN 1 AND 32
                ),
                response_sha256 TEXT NOT NULL CHECK(
                    length(response_sha256)=64
                    AND response_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                byte_count INTEGER NOT NULL CHECK(
                    byte_count BETWEEN 0 AND 1500000
                ),
                response_text TEXT NOT NULL,
                record_sha256 TEXT NOT NULL CHECK(
                    length(record_sha256)=64
                    AND record_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                UNIQUE(session_id, logical_call_id, physical_ordinal),
                UNIQUE(
                    session_id, logical_call_id, physical_attempt_id,
                    physical_ordinal, response_sha256
                ),
                FOREIGN KEY(
                    session_id, logical_call_id, physical_attempt_id,
                    physical_ordinal
                ) REFERENCES insession_runtime_model_physical_attempts(
                    session_id, logical_call_id, physical_attempt_id,
                    physical_ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_tool_call_settlement_receipts (
                settle_apply_id TEXT PRIMARY KEY,
                settlement_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                logical_tool_call_id TEXT NOT NULL,
                physical_attempt_id TEXT NOT NULL UNIQUE,
                physical_request_binding_sha256 TEXT NOT NULL CHECK(
                    length(physical_request_binding_sha256)=64
                    AND physical_request_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_ordinal INTEGER NOT NULL CHECK(
                    physical_ordinal BETWEEN 1 AND 16
                ),
                outcome TEXT NOT NULL CHECK(outcome IN (
                    'succeeded', 'retryable_failure',
                    'terminal_failure', 'uncertain'
                )),
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                settlement_json TEXT NOT NULL CHECK(length(settlement_json)>0),
                settled_turn_id TEXT NOT NULL,
                settled_at TEXT NOT NULL,
                FOREIGN KEY(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    physical_attempt_id, physical_request_binding_sha256,
                    physical_ordinal
                ) REFERENCES insession_runtime_tool_physical_attempts(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    physical_attempt_id, binding_sha256, physical_ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, settled_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(settled_turn_id, work_run_id)
                    REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_tool_logical_calls (
                logical_tool_call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                call_ordinal INTEGER NOT NULL CHECK(call_ordinal BETWEEN 1 AND 4),
                invocation_turn_id TEXT NOT NULL,
                catalog_snapshot_sha256 TEXT NOT NULL CHECK(
                    length(catalog_snapshot_sha256)=64
                    AND catalog_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                tool_id TEXT NOT NULL CHECK(length(tool_id) BETWEEN 1 AND 200),
                contract_version TEXT NOT NULL CHECK(
                    length(contract_version) BETWEEN 1 AND 200
                ),
                implementation_version TEXT NOT NULL CHECK(
                    length(implementation_version) BETWEEN 1 AND 200
                ),
                provider_identity_sha256 TEXT NOT NULL CHECK(
                    length(provider_identity_sha256)=64
                    AND provider_identity_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                effect_profile_sha256 TEXT NOT NULL CHECK(
                    length(effect_profile_sha256)=64
                    AND effect_profile_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                effect_class TEXT NOT NULL CHECK(effect_class IN (
                    'read_only', 'protected_effect'
                )),
                retry_authority TEXT NOT NULL CHECK(retry_authority IN (
                    'read_only_replay', 'provider_idempotency',
                    'reconciliation_required'
                )),
                arguments_sha256 TEXT NOT NULL CHECK(
                    length(arguments_sha256)=64
                    AND arguments_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_contract TEXT NOT NULL CHECK(
                    length(result_contract) BETWEEN 1 AND 200
                ),
                max_physical_attempts INTEGER NOT NULL CHECK(
                    max_physical_attempts BETWEEN 1 AND 16
                ),
                state_guard_sha256 TEXT NOT NULL CHECK(
                    length(state_guard_sha256)=64
                    AND state_guard_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                request_json TEXT NOT NULL CHECK(length(request_json)>0),
                created_at TEXT NOT NULL,
                UNIQUE(session_id, logical_tool_call_id),
                UNIQUE(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    binding_sha256, provider_identity_sha256, effect_class,
                    retry_authority, result_contract
                ),
                CHECK(
                    (effect_class='read_only'
                        AND retry_authority='read_only_replay')
                    OR
                    (effect_class='protected_effect'
                        AND retry_authority IN (
                            'provider_idempotency', 'reconciliation_required'
                        ))
                ),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, work_run_id)
                    REFERENCES insession_work_runs(session_id, work_run_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    work_run_id, attempt_id, logical_tool_call_id, call_ordinal
                ) REFERENCES insession_work_run_tool_calls(
                    work_run_id, attempt_id, tool_call_id, ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(invocation_turn_id, work_run_id)
                    REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_runtime_tool_physical_attempts (
                physical_attempt_id TEXT PRIMARY KEY,
                physical_attempt_key TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                logical_tool_call_id TEXT NOT NULL,
                logical_request_binding_sha256 TEXT NOT NULL CHECK(
                    length(logical_request_binding_sha256)=64
                    AND logical_request_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_ordinal INTEGER NOT NULL CHECK(
                    physical_ordinal BETWEEN 1 AND 16
                ),
                started_turn_id TEXT NOT NULL,
                provider_identity_sha256 TEXT NOT NULL CHECK(
                    length(provider_identity_sha256)=64
                    AND provider_identity_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                effect_class TEXT NOT NULL CHECK(effect_class IN (
                    'read_only', 'protected_effect'
                )),
                retry_authority TEXT NOT NULL CHECK(retry_authority IN (
                    'read_only_replay', 'provider_idempotency',
                    'reconciliation_required'
                )),
                result_contract TEXT NOT NULL CHECK(
                    length(result_contract) BETWEEN 1 AND 200
                ),
                provider_idempotency_key TEXT CHECK(
                    provider_idempotency_key IS NULL
                    OR length(provider_idempotency_key) BETWEEN 1 AND 300
                ),
                dispatch_authority_sha256 TEXT NOT NULL CHECK(
                    length(dispatch_authority_sha256)=64
                    AND dispatch_authority_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                binding_sha256 TEXT NOT NULL CHECK(
                    length(binding_sha256)=64
                    AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                physical_request_json TEXT NOT NULL CHECK(
                    length(physical_request_json)>0
                ),
                status TEXT NOT NULL CHECK(status IN (
                    'pending', 'succeeded', 'retryable_failure',
                    'terminal_failure', 'uncertain'
                )),
                settlement_apply_id TEXT UNIQUE,
                started_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(logical_tool_call_id, physical_ordinal),
                UNIQUE(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    physical_attempt_id, binding_sha256, physical_ordinal
                ),
                CHECK(
                    (retry_authority='provider_idempotency'
                        AND provider_idempotency_key IS NOT NULL)
                    OR
                    (retry_authority<>'provider_idempotency'
                        AND provider_idempotency_key IS NULL)
                ),
                CHECK(
                    (status='pending' AND settlement_apply_id IS NULL
                        AND settled_at IS NULL)
                    OR
                    (status<>'pending' AND settlement_apply_id IS NOT NULL
                        AND settled_at IS NOT NULL)
                ),
                FOREIGN KEY(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    logical_request_binding_sha256, provider_identity_sha256,
                    effect_class, retry_authority, result_contract
                ) REFERENCES insession_runtime_tool_logical_calls(
                    session_id, work_run_id, attempt_id, logical_tool_call_id,
                    binding_sha256, provider_identity_sha256, effect_class,
                    retry_authority, result_contract
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, started_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(started_turn_id, work_run_id)
                    REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_branch_intents (
            branch_intent_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_turn_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            branch_key TEXT NOT NULL,
            branch_summary TEXT NOT NULL,
            source_start INTEGER NOT NULL CHECK(source_start >= 0),
            source_end INTEGER NOT NULL CHECK(source_end > source_start),
            source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(insession_task_id) REFERENCES insession_tasks(insession_task_id)
                ON DELETE CASCADE,
            UNIQUE(source_turn_id, insession_task_id, branch_key)
        );

CREATE TABLE insession_task_delivery_validation_requests (
                verification_request_id TEXT PRIMARY KEY,
                logical_call_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
                completed_task_state_version INTEGER NOT NULL CHECK(
                    completed_task_state_version >= 1
                ),
                root_delivery_id TEXT NOT NULL UNIQUE,
                invocation_turn_id TEXT NOT NULL,
                prompt_payload_sha256 TEXT NOT NULL CHECK(
                    length(prompt_payload_sha256)=64
                    AND prompt_payload_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                request_binding_sha256 TEXT NOT NULL CHECK(
                    length(request_binding_sha256)=64
                    AND request_binding_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                request_json TEXT NOT NULL CHECK(length(request_json)>0),
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, graph_revision,
                    verification_request_id
                ),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(root_delivery_id)
                    REFERENCES insession_task_node_deliveries(delivery_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_delivery_validation_results (
                verification_result_id TEXT PRIMARY KEY,
                verification_request_id TEXT NOT NULL UNIQUE,
                logical_call_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                disposition TEXT NOT NULL CHECK(
                    disposition IN ('pass','revise','blocked')
                ),
                result_sha256 TEXT NOT NULL CHECK(
                    length(result_sha256)=64
                    AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                result_json TEXT NOT NULL CHECK(length(result_json)>0),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(verification_request_id)
                    REFERENCES insession_task_delivery_validation_requests(
                        verification_request_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(logical_call_id)
                    REFERENCES insession_runtime_model_logical_calls(
                        logical_call_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_delivery_validation_settlements (
                settlement_id TEXT PRIMARY KEY,
                apply_id TEXT NOT NULL UNIQUE,
                verification_request_id TEXT NOT NULL UNIQUE,
                verification_result_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
                completed_task_state_version INTEGER NOT NULL CHECK(
                    completed_task_state_version >= 1
                ),
                root_delivery_id TEXT NOT NULL UNIQUE,
                disposition TEXT NOT NULL CHECK(
                    disposition IN ('pass','revise','blocked')
                ),
                command_sha256 TEXT NOT NULL CHECK(
                    length(command_sha256)=64
                    AND command_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                settlement_sha256 TEXT NOT NULL CHECK(
                    length(settlement_sha256)=64
                    AND settlement_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                settlement_json TEXT NOT NULL CHECK(length(settlement_json)>0),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(verification_request_id)
                    REFERENCES insession_task_delivery_validation_requests(
                        verification_request_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(verification_result_id)
                    REFERENCES insession_task_delivery_validation_results(
                        verification_result_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(root_delivery_id)
                    REFERENCES insession_task_node_deliveries(delivery_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_graph_apply_receipts (
            apply_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_turn_id TEXT NOT NULL,
            operation TEXT NOT NULL CHECK(operation IN ('create', 'revise')),
            proposal_hash TEXT NOT NULL,
            created_insession_task_ids_json TEXT NOT NULL,
            turn_task_link_revision INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

CREATE TABLE insession_task_graph_edges (
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
                child_insession_task_node_id TEXT NOT NULL,
                parent_insession_task_node_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                PRIMARY KEY(
                    insession_task_id, graph_revision,
                    child_insession_task_node_id
                ),
                FOREIGN KEY(
                    insession_task_id, graph_revision,
                    child_insession_task_node_id
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id
                ) ON DELETE CASCADE,
                FOREIGN KEY(
                    insession_task_id, graph_revision,
                    parent_insession_task_node_id
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id
                ) ON DELETE CASCADE
            );

CREATE TABLE insession_task_graph_execution_replan_applications (
                apply_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                request_sha256 TEXT NOT NULL CHECK(
                    length(request_sha256)=64
                    AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(base_graph_revision>=1),
                committed_graph_revision INTEGER NOT NULL CHECK(
                    committed_graph_revision=base_graph_revision+1
                ),
                task_graph_commit_apply_id TEXT NOT NULL UNIQUE,
                consumed_turn_id TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                created_at TEXT NOT NULL,
                FOREIGN KEY(request_id)
                    REFERENCES insession_task_graph_execution_replan_requests(
                        request_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(task_graph_commit_apply_id)
                    REFERENCES insession_auxiliary_v2_task_graph_commit_receipts(
                        apply_id
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, consumed_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_graph_execution_replan_requests (
                request_id TEXT PRIMARY KEY,
                create_apply_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(base_graph_revision>=1),
                target_graph_revision INTEGER NOT NULL CHECK(
                    target_graph_revision=base_graph_revision+1
                ),
                work_run_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                insession_task_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision>=1),
                source_node_alias TEXT NOT NULL CHECK(
                    source_node_alias GLOB 'base_node_[0-9][0-9][0-9]'
                ),
                reason TEXT NOT NULL CHECK(reason IN (
                    'task_decomposition_incomplete', 'node_contract_invalid',
                    'dependency_structure_invalid',
                    'capability_assignment_invalid'
                )),
                diagnosis TEXT NOT NULL CHECK(length(diagnosis)>0),
                revision_objective TEXT NOT NULL CHECK(
                    length(revision_objective)>0
                ),
                supporting_tool_result_ids_json TEXT NOT NULL,
                supporting_tool_result_ids_sha256 TEXT NOT NULL CHECK(
                    length(supporting_tool_result_ids_sha256)=64
                    AND supporting_tool_result_ids_sha256
                        NOT GLOB '*[^0-9a-f]*'
                ),
                task_state_version INTEGER NOT NULL CHECK(task_state_version>=1),
                created_turn_id TEXT NOT NULL,
                request_sha256 TEXT NOT NULL CHECK(
                    length(request_sha256)=64
                    AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                request_json TEXT NOT NULL CHECK(length(request_json)>0),
                created_at TEXT NOT NULL,
                UNIQUE(session_id, insession_task_id, base_graph_revision),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, base_graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT
            );

CREATE TABLE "insession_task_graph_nodes" (insession_task_id TEXT NOT NULL, graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1), insession_task_node_id TEXT NOT NULL, node_revision INTEGER NOT NULL CHECK(node_revision >= 1), node_kind TEXT NOT NULL CHECK(node_kind IN ('root', 'subtask')), ordinal INTEGER NOT NULL CHECK(ordinal >= 0), title TEXT NOT NULL, objective TEXT NOT NULL, source_anchor_ids_json TEXT NOT NULL, acceptance_criteria_json TEXT NOT NULL, constraints_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(insession_task_id, graph_revision, insession_task_node_id), FOREIGN KEY(insession_task_id, graph_revision) REFERENCES insession_task_graph_revisions(insession_task_id, graph_revision) ON DELETE CASCADE);

CREATE TABLE insession_task_graph_revision_apply_receipts (
            apply_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_turn_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            expected_graph_revision INTEGER
                CHECK(expected_graph_revision IS NULL OR expected_graph_revision >= 1),
            committed_graph_revision INTEGER NOT NULL
                CHECK(committed_graph_revision >= 1),
            expected_task_state_version INTEGER NOT NULL
                CHECK(expected_task_state_version >= 1),
            committed_task_state_version INTEGER NOT NULL
                CHECK(committed_task_state_version >= 1),
            expected_window_revision INTEGER NOT NULL
                CHECK(expected_window_revision >= 1),
            committed_window_state_version INTEGER NOT NULL
                CHECK(committed_window_state_version >= 1),
            proposal_hash TEXT NOT NULL CHECK(length(proposal_hash) = 64),
            turn_task_link_revision INTEGER NOT NULL DEFAULT 0
                CHECK(turn_task_link_revision >= 0),
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(insession_task_id) REFERENCES insession_tasks(insession_task_id)
                ON DELETE CASCADE,
            CHECK(
                (expected_graph_revision IS NULL AND committed_graph_revision = 1)
                OR committed_graph_revision = expected_graph_revision + 1
            ),
            UNIQUE(insession_task_id, committed_graph_revision)
        );

CREATE TABLE insession_task_graph_revision_trigger_applications (
                apply_id TEXT PRIMARY KEY,
                trigger_id TEXT NOT NULL UNIQUE,
                trigger_sha256 TEXT NOT NULL CHECK(
                    length(trigger_sha256)=64
                    AND trigger_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(
                    base_graph_revision >= 1
                ),
                committed_graph_revision INTEGER NOT NULL CHECK(
                    committed_graph_revision=base_graph_revision+1
                ),
                task_graph_commit_apply_id TEXT NOT NULL UNIQUE,
                consumed_turn_id TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL CHECK(
                    length(receipt_sha256)=64
                    AND receipt_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                receipt_json TEXT NOT NULL CHECK(length(receipt_json)>0),
                created_at TEXT NOT NULL,
                FOREIGN KEY(trigger_id)
                    REFERENCES insession_task_graph_revision_triggers(trigger_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(task_graph_commit_apply_id)
                    REFERENCES insession_auxiliary_v2_task_graph_commit_receipts(
                        apply_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, consumed_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_graph_revision_triggers (
                trigger_id TEXT PRIMARY KEY,
                create_apply_id TEXT NOT NULL UNIQUE,
                settlement_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                base_graph_revision INTEGER NOT NULL CHECK(
                    base_graph_revision >= 1
                ),
                target_graph_revision INTEGER NOT NULL CHECK(
                    target_graph_revision=base_graph_revision+1
                ),
                root_delivery_id TEXT NOT NULL UNIQUE,
                reopened_task_state_version INTEGER NOT NULL CHECK(
                    reopened_task_state_version >= 1
                ),
                trigger_sha256 TEXT NOT NULL CHECK(
                    length(trigger_sha256)=64
                    AND trigger_sha256 NOT GLOB '*[^0-9a-f]*'
                ),
                trigger_json TEXT NOT NULL CHECK(length(trigger_json)>0),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(settlement_id)
                    REFERENCES insession_task_delivery_validation_settlements(
                        settlement_id
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(insession_task_id, base_graph_revision)
                    REFERENCES insession_task_graph_revisions(
                        insession_task_id, graph_revision
                    ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(root_delivery_id)
                    REFERENCES insession_task_node_deliveries(delivery_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_graph_revisions (
            insession_task_id TEXT NOT NULL,
            graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
            source_turn_id TEXT NOT NULL,
            proposal_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, source_anchors_json TEXT NOT NULL DEFAULT '[]', authorization_anchor_ids_json TEXT NOT NULL DEFAULT '[]', required_anchor_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY(insession_task_id, graph_revision),
            FOREIGN KEY(insession_task_id) REFERENCES insession_tasks(insession_task_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_task_match_apply_receipts (
            apply_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_turn_id TEXT NOT NULL,
            proposal_hash TEXT NOT NULL,
            created_task_mapping_json TEXT NOT NULL,
            related_insession_task_ids_json TEXT NOT NULL,
            branch_intent_ids_json TEXT NOT NULL,
            turn_task_link_revision INTEGER NOT NULL DEFAULT 0,
            window_state_version INTEGER,
            created_at TEXT NOT NULL, execution_lane_manifest_json TEXT, execution_lane_manifest_hash TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

CREATE TABLE insession_task_node_deliveries (
                delivery_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL UNIQUE,
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
                insession_task_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                verification_request_id TEXT NOT NULL UNIQUE,
                submitted_attempt_id TEXT NOT NULL,
                output_revision INTEGER NOT NULL CHECK(output_revision >= 1),
                created_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(insession_task_id, insession_task_node_id, node_revision),
                FOREIGN KEY(session_id, work_run_id)
                    REFERENCES insession_work_runs(session_id, work_run_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY(work_run_id, output_revision)
                    REFERENCES insession_work_run_output_windows(
                        work_run_id, output_revision
                    ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(work_run_id, verification_request_id)
                    REFERENCES insession_work_run_verification_requests(
                        work_run_id, verification_request_id
                    ) ON DELETE RESTRICT,
                FOREIGN KEY(work_run_id, submitted_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) ON DELETE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id) ON DELETE RESTRICT
            );

CREATE TABLE insession_task_node_execution_subject_bindings (
                binding_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
                insession_task_node_id TEXT NOT NULL,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                created_at TEXT NOT NULL,
                UNIQUE(
                    session_id, insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ),
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) REFERENCES insession_task_graph_nodes(
                    insession_task_id, graph_revision,
                    insession_task_node_id, node_revision
                ) ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE insession_task_node_retrieval_capability_plans (
            session_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
            plan_sha256 TEXT NOT NULL CHECK(length(plan_sha256) = 64),
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, insession_task_id, graph_revision),
            FOREIGN KEY (insession_task_id, graph_revision)
                REFERENCES insession_task_graph_revisions(
                    insession_task_id, graph_revision
                ) ON DELETE CASCADE
        );

CREATE TABLE insession_task_node_retrieval_catalog_bindings (
            session_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            graph_revision INTEGER NOT NULL CHECK(graph_revision >= 1),
            insession_task_node_id TEXT NOT NULL,
            node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
            plan_sha256 TEXT NOT NULL CHECK(length(plan_sha256) = 64),
            catalog_snapshot_sha256 TEXT NOT NULL
                CHECK(length(catalog_snapshot_sha256) = 64),
            catalog_snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (
                session_id,
                insession_task_id,
                graph_revision,
                insession_task_node_id,
                node_revision
            ),
            FOREIGN KEY (session_id, insession_task_id, graph_revision)
                REFERENCES insession_task_node_retrieval_capability_plans(
                    session_id, insession_task_id, graph_revision
                ) ON DELETE CASCADE,
            FOREIGN KEY (
                insession_task_id,
                graph_revision,
                insession_task_node_id,
                node_revision
            ) REFERENCES insession_task_graph_nodes(
                insession_task_id,
                graph_revision,
                insession_task_node_id,
                node_revision
            ) ON DELETE CASCADE
        );

CREATE TABLE insession_task_node_states (
            insession_task_id TEXT NOT NULL,
            insession_task_node_id TEXT NOT NULL,
            node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
            status TEXT NOT NULL CHECK(status IN (
                'proposed', 'active', 'awaiting_user', 'waiting_external',
                'interrupted', 'blocked', 'cancelled', 'completed'
            )),
            state_version INTEGER NOT NULL DEFAULT 1 CHECK(state_version >= 1),
            updated_at TEXT NOT NULL,
            PRIMARY KEY(insession_task_id, insession_task_node_id, node_revision),
            FOREIGN KEY(insession_task_id) REFERENCES insession_tasks(insession_task_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_task_turn_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            insession_task_id TEXT NOT NULL,
            insession_task_node_id TEXT,
            relation TEXT NOT NULL CHECK(relation IN ('created', 'referenced', 'revised')),
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(insession_task_id) REFERENCES insession_tasks(insession_task_id)
                ON DELETE CASCADE
        );

CREATE TABLE "insession_tasks" (insession_task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, current_graph_revision INTEGER CHECK(current_graph_revision IS NULL OR current_graph_revision >= 1), current_status TEXT NOT NULL CHECK(current_status IN ('proposed', 'active', 'awaiting_user', 'waiting_external', 'interrupted', 'blocked', 'cancelled', 'completed')), state_version INTEGER NOT NULL DEFAULT 1 CHECK(state_version >= 1), root_title TEXT NOT NULL, root_objective TEXT NOT NULL, created_turn_id TEXT NOT NULL, creation_source_start INTEGER CHECK(creation_source_start IS NULL OR creation_source_start >= 0), creation_source_end INTEGER CHECK(creation_source_end IS NULL OR creation_source_end > creation_source_start), creation_source_sha256 TEXT CHECK(creation_source_sha256 IS NULL OR length(creation_source_sha256) = 64), created_at TEXT NOT NULL, updated_at TEXT NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE);

CREATE TABLE insession_work_run_acceptance_progress (
            work_run_id TEXT PRIMARY KEY,
            progress_revision INTEGER NOT NULL CHECK(progress_revision >= 1),
            snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash) = 64),
            snapshot_json TEXT NOT NULL,
            updated_attempt_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, evaluated_output_revision INTEGER NOT NULL DEFAULT 1 CHECK(evaluated_output_revision >= 1),
            FOREIGN KEY(work_run_id) REFERENCES insession_work_runs(work_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(work_run_id, updated_attempt_id)
                REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                DEFERRABLE INITIALLY DEFERRED
        );

CREATE TABLE insession_work_run_apply_receipts (
            apply_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL CHECK(operation IN (
                'create_work_run', 'start_attempt',
                'commit_attempt_decision', 'commit_output_action',
                'append_tool_result', 'close_attempt',
                'prepare_verification', 'commit_verification_result',
                'interrupt_verification', 'resume_verification',
                'charge_active_time', 'resume_active_attempt',
                'continue_waiting_user_and_start_attempt',
                'detach_safe_lane'
            )),
            session_id TEXT NOT NULL,
            work_run_id TEXT NOT NULL,
            payload_hash TEXT NOT NULL CHECK(length(payload_hash) = 64),
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(session_id, work_run_id)
                REFERENCES insession_work_runs(session_id, work_run_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_work_run_attempts (
            attempt_id TEXT PRIMARY KEY,
            work_run_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            input_turn_id TEXT NOT NULL,
            predecessor_question_attempt_id TEXT,
            ordinal INTEGER NOT NULL CHECK(ordinal >= 1 AND ordinal <= 32),
            status TEXT NOT NULL CHECK(status IN ('active', 'closed')),
            action TEXT CHECK(action IN (
                'call_tools', 'write_output_window', 'submit_output_window',
                'request_user_input', 'request_task_graph_revision'
            )),
            decision_json TEXT,
            progress_revision_before INTEGER NOT NULL CHECK(progress_revision_before>=1),
            progress_revision_after INTEGER CHECK(progress_revision_after>=1),
            input_output_revision INTEGER NOT NULL CHECK(input_output_revision>=1),
            input_output_hash TEXT NOT NULL CHECK(length(input_output_hash)=64),
            committed_output_revision INTEGER CHECK(committed_output_revision>=1),
            submitted_output_revision INTEGER CHECK(submitted_output_revision>=1),
            input_checkpoint_id TEXT,
            input_verification_request_id TEXT,
            catalog_snapshot_json TEXT NOT NULL,
            catalog_snapshot_hash TEXT NOT NULL CHECK(length(catalog_snapshot_hash)=64),
            budget_before_json TEXT NOT NULL,
            budget_after_json TEXT,
            close_reason TEXT,
            created_at TEXT NOT NULL,
            closed_at TEXT,
            budget_charge_id TEXT,
            UNIQUE(work_run_id, ordinal),
            UNIQUE(work_run_id, attempt_id),
            UNIQUE(predecessor_question_attempt_id),
            FOREIGN KEY(work_run_id) REFERENCES insession_work_runs(work_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE RESTRICT,
            FOREIGN KEY(turn_id, work_run_id)
                REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(input_turn_id, work_run_id)
                REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                ON DELETE RESTRICT,
            FOREIGN KEY(work_run_id, predecessor_question_attempt_id)
                REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
            FOREIGN KEY(work_run_id, input_verification_request_id)
                REFERENCES insession_work_run_verification_requests(
                    work_run_id, verification_request_id
                ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
            FOREIGN KEY(budget_charge_id)
                REFERENCES insession_work_run_budget_charges(budget_charge_id)
                ON DELETE RESTRICT,
            CHECK(predecessor_question_attempt_id IS NULL OR ordinal>1),
            CHECK(input_verification_request_id IS NULL OR (
                ordinal>1 AND input_checkpoint_id=input_verification_request_id
            )),
            CHECK(
                (status='active' AND closed_at IS NULL
                    AND budget_after_json IS NULL AND close_reason IS NULL
                    AND committed_output_revision IS NULL
                    AND submitted_output_revision IS NULL
                    AND ((action IS NULL AND decision_json IS NULL
                            AND progress_revision_after IS NULL)
                        OR (action='call_tools' AND decision_json IS NOT NULL
                            AND progress_revision_after IS NOT NULL)))
                OR
                (status='closed' AND action IS NOT NULL
                    AND decision_json IS NOT NULL
                    AND progress_revision_after IS NOT NULL
                    AND budget_after_json IS NOT NULL
                    AND close_reason IS NOT NULL AND closed_at IS NOT NULL
                    AND ((action='write_output_window'
                            AND committed_output_revision IS NOT NULL
                            AND submitted_output_revision IS NULL)
                        OR (action='submit_output_window'
                            AND committed_output_revision IS NOT NULL
                            AND submitted_output_revision=committed_output_revision)
                        OR (action IN ('call_tools','request_user_input',
                                'request_task_graph_revision')
                            AND committed_output_revision IS NULL
                            AND submitted_output_revision IS NULL)))
            )
        );

CREATE TABLE "insession_work_run_budget_charges" (
            budget_charge_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL CHECK(operation IN (
                'charge_active_time', 'commit_attempt_decision',
                'commit_output_action', 'close_attempt',
                'commit_verification_result', 'interrupt_verification'
            )),
            session_id TEXT NOT NULL,
            work_run_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            work_run_revision_before INTEGER NOT NULL CHECK(work_run_revision_before>=1),
            work_run_revision_after INTEGER NOT NULL CHECK(
                work_run_revision_after=work_run_revision_before+1
            ),
            window_state_version_before INTEGER NOT NULL CHECK(
                window_state_version_before>=1
            ),
            window_state_version_after INTEGER NOT NULL CHECK(
                window_state_version_after=window_state_version_before+1
            ),
            active_seconds_delta REAL NOT NULL CHECK(active_seconds_delta>0),
            active_seconds_before REAL NOT NULL CHECK(active_seconds_before>=0),
            active_seconds_after REAL NOT NULL CHECK(
                active_seconds_after=active_seconds_before+active_seconds_delta
            ),
            disposition TEXT NOT NULL CHECK(disposition IN (
                'within_limit','soft_limit_reached','hard_limit_reached'
            )),
            work_run_status_after TEXT NOT NULL CHECK(work_run_status_after IN (
                'active','waiting_user','waiting_external','turn_limit_reached',
                'interrupted','completed','failed','cancelled'
            )),
            work_run_reason_after TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(work_run_id, checkpoint_id),
            FOREIGN KEY(session_id, work_run_id)
                REFERENCES insession_work_runs(session_id, work_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(turn_id, work_run_id)
                REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                ON DELETE RESTRICT
        );

CREATE TABLE insession_work_run_output_windows (
                work_run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                output_revision INTEGER NOT NULL CHECK(output_revision >= 1),
                snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash) = 64),
                snapshot_json TEXT NOT NULL CHECK(length(snapshot_json) > 0),
                updated_turn_id TEXT NOT NULL,
                updated_attempt_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, frozen_at TEXT,
                UNIQUE(work_run_id, output_revision),
                FOREIGN KEY(session_id, work_run_id)
                    REFERENCES insession_work_runs(session_id, work_run_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(updated_turn_id, work_run_id)
                    REFERENCES insession_work_run_turn_links(turn_id, work_run_id)
                    ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(work_run_id, updated_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
            );

CREATE TABLE insession_work_run_tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            work_run_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
            provider_call_id TEXT,
            tool_id TEXT NOT NULL,
            tool_version TEXT NOT NULL,
            modifies_environment INTEGER NOT NULL CHECK(modifies_environment IN (0, 1)),
            arguments_hash TEXT NOT NULL CHECK(length(arguments_hash) = 64),
            arguments_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(attempt_id, ordinal),
            UNIQUE(work_run_id, attempt_id, tool_call_id),
            UNIQUE(work_run_id, attempt_id, tool_call_id, ordinal),
            FOREIGN KEY(work_run_id) REFERENCES insession_work_runs(work_run_id)
                ON DELETE CASCADE,
            FOREIGN KEY(work_run_id, attempt_id)
                REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_work_run_tool_results (
                    tool_result_id TEXT PRIMARY KEY,
                    work_run_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL UNIQUE,
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                    status TEXT NOT NULL CHECK(status IN (
                        'succeeded', 'rejected', 'failed', 'timed_out',
                        'cancelled', 'completion_unconfirmed'
                    )),
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(attempt_id, ordinal),
                    FOREIGN KEY(work_run_id) REFERENCES insession_work_runs(work_run_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(work_run_id, attempt_id)
                        REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(work_run_id, attempt_id, tool_call_id, ordinal)
                        REFERENCES insession_work_run_tool_calls(
                            work_run_id, attempt_id, tool_call_id, ordinal
                        )
                        ON DELETE CASCADE
                );

CREATE TABLE insession_work_run_turn_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            work_run_id TEXT NOT NULL,
            link_revision INTEGER NOT NULL CHECK(link_revision >= 1),
            relation TEXT NOT NULL CHECK(relation IN ('started', 'continued')),
            created_at TEXT NOT NULL,
            UNIQUE(turn_id, work_run_id),
            UNIQUE(turn_id, link_revision),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(session_id, turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE,
            FOREIGN KEY(session_id, work_run_id)
                REFERENCES insession_work_runs(session_id, work_run_id)
                ON DELETE CASCADE
        );

CREATE TABLE insession_work_run_verification_requests (
                verification_request_id TEXT PRIMARY KEY,
                execution_subject_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_turn_id TEXT NOT NULL,
                work_run_id TEXT NOT NULL,
                subject_kind TEXT NOT NULL CHECK(subject_kind IN (
                    'task_node', 'auxiliary_node'
                )),
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER CHECK(graph_revision >= 1),
                insession_task_node_id TEXT,
                auxiliary_graph_id TEXT,
                auxiliary_graph_revision INTEGER CHECK(auxiliary_graph_revision >= 1),
                auxiliary_node_id TEXT,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                submitted_attempt_id TEXT NOT NULL,
                output_revision INTEGER NOT NULL CHECK(output_revision >= 1),
                acceptance_progress_revision INTEGER NOT NULL
                    CHECK(acceptance_progress_revision >= 1),
                acceptance_ids_json TEXT NOT NULL CHECK(length(acceptance_ids_json) > 0),
                supporting_tool_result_ids_json TEXT NOT NULL,
                locked_work_run_revision INTEGER NOT NULL
                    CHECK(locked_work_run_revision >= 1),
                request_binding_hash TEXT NOT NULL CHECK(length(request_binding_hash) = 64),
                prepared_budget_json TEXT NOT NULL CHECK(length(prepared_budget_json) > 0),
                request_revision INTEGER NOT NULL CHECK(request_revision >= 1),
                status TEXT NOT NULL CHECK(status IN (
                    'pending', 'interrupted', 'completed'
                )),
                technical_error_code TEXT CHECK(
                    technical_error_code IS NULL OR (
                        length(technical_error_code) >= 1
                        AND length(technical_error_code) <= 160
                    )
                ),
                result_json TEXT,
                all_pass INTEGER CHECK(all_pass IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                dependency_delivery_ids_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(work_run_id, verification_request_id),
                UNIQUE(work_run_id, submitted_attempt_id),
                FOREIGN KEY(
                    session_id, work_run_id, execution_subject_id
                ) REFERENCES insession_work_runs(
                    session_id, work_run_id, execution_subject_id
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, request_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id) ON DELETE RESTRICT,
                FOREIGN KEY(work_run_id, submitted_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
                CHECK(
                    (subject_kind = 'task_node'
                        AND graph_revision IS NOT NULL
                        AND insession_task_node_id IS NOT NULL
                        AND auxiliary_graph_id IS NULL
                        AND auxiliary_graph_revision IS NULL
                        AND auxiliary_node_id IS NULL)
                    OR
                    (subject_kind = 'auxiliary_node'
                        AND graph_revision IS NULL
                        AND insession_task_node_id IS NULL
                        AND auxiliary_graph_id IS NOT NULL
                        AND auxiliary_graph_revision IS NOT NULL
                        AND auxiliary_node_id IS NOT NULL)
                ),
                CHECK(
                    (status='pending' AND technical_error_code IS NULL
                        AND result_json IS NULL AND all_pass IS NULL
                        AND completed_at IS NULL)
                    OR (status='interrupted' AND technical_error_code IS NOT NULL
                        AND result_json IS NULL AND all_pass IS NULL
                        AND completed_at IS NULL)
                    OR (status='completed' AND technical_error_code IS NULL
                        AND result_json IS NOT NULL AND all_pass IS NOT NULL
                        AND completed_at IS NOT NULL)
                )
            );

CREATE TABLE insession_work_runs (
                work_run_id TEXT PRIMARY KEY,
                execution_subject_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                subject_kind TEXT NOT NULL CHECK(subject_kind IN (
                    'task_node', 'auxiliary_node'
                )),
                insession_task_id TEXT NOT NULL,
                graph_revision INTEGER CHECK(graph_revision >= 1),
                insession_task_node_id TEXT,
                auxiliary_graph_id TEXT,
                auxiliary_graph_revision INTEGER CHECK(auxiliary_graph_revision >= 1),
                auxiliary_node_id TEXT,
                node_revision INTEGER NOT NULL CHECK(node_revision >= 1),
                status TEXT NOT NULL CHECK(status IN (
                    'active', 'paused', 'waiting_user', 'waiting_authorization',
                    'waiting_external', 'turn_limit_reached', 'interrupted',
                    'completed', 'failed', 'cancelled'
                )),
                reason TEXT,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                max_attempts INTEGER NOT NULL DEFAULT 32 CHECK(max_attempts = 32),
                soft_active_seconds REAL NOT NULL DEFAULT 720
                    CHECK(soft_active_seconds = 720),
                hard_active_seconds REAL NOT NULL DEFAULT 900
                    CHECK(hard_active_seconds = 900),
                attempts_started INTEGER NOT NULL DEFAULT 0
                    CHECK(attempts_started >= 0 AND attempts_started <= max_attempts),
                active_seconds_consumed REAL NOT NULL DEFAULT 0
                    CHECK(active_seconds_consumed >= 0),
                current_attempt_id TEXT,
                current_verification_request_id TEXT,
                created_turn_id TEXT NOT NULL,
                updated_turn_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, work_run_id),
                UNIQUE(session_id, work_run_id, execution_subject_id),
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(session_id, insession_task_id)
                    REFERENCES insession_tasks(session_id, insession_task_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY(
                    session_id, insession_task_id, execution_subject_id
                ) REFERENCES insession_execution_subjects(
                    session_id, insession_task_id, execution_subject_id
                ) ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, created_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id) ON DELETE RESTRICT,
                FOREIGN KEY(session_id, updated_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id) ON DELETE RESTRICT,
                FOREIGN KEY(work_run_id, current_attempt_id)
                    REFERENCES insession_work_run_attempts(work_run_id, attempt_id)
                    DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(work_run_id, current_verification_request_id)
                    REFERENCES insession_work_run_verification_requests(
                        work_run_id, verification_request_id
                    ) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
                CHECK(
                    (subject_kind = 'task_node'
                        AND graph_revision IS NOT NULL
                        AND insession_task_node_id IS NOT NULL
                        AND auxiliary_graph_id IS NULL
                        AND auxiliary_graph_revision IS NULL
                        AND auxiliary_node_id IS NULL)
                    OR
                    (subject_kind = 'auxiliary_node'
                        AND graph_revision IS NULL
                        AND insession_task_node_id IS NULL
                        AND auxiliary_graph_id IS NOT NULL
                        AND auxiliary_graph_revision IS NOT NULL
                        AND auxiliary_node_id IS NOT NULL)
                )
            );

CREATE TABLE l1_turn_plan_revisions (
            l1_turn_run_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK(revision BETWEEN 1 AND 64),
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            accepted_step_id TEXT,
            plan_json TEXT NOT NULL CHECK(length(plan_json) > 0),
            plan_hash TEXT NOT NULL CHECK(length(plan_hash) = 64),
            created_at TEXT NOT NULL,
            PRIMARY KEY(l1_turn_run_id, revision),
            UNIQUE(l1_turn_run_id, plan_hash),
            FOREIGN KEY(session_id, turn_id, l1_turn_run_id)
                REFERENCES l1_turn_runs(
                    session_id, turn_id, l1_turn_run_id
                ) ON DELETE CASCADE,
            FOREIGN KEY(accepted_step_id)
                REFERENCES l1_turn_steps(step_id) ON DELETE SET NULL
        );

CREATE TABLE l1_turn_run_states (
            l1_turn_run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL CHECK(stage IN (
                'bootstrap', 'model', 'tool', 'observation',
                'finalizing', 'completed', 'failed', 'cancelled'
            )),
            deadline_at TEXT NOT NULL,
            max_semantic_steps INTEGER NOT NULL
                CHECK(max_semantic_steps BETWEEN 1 AND 64),
            max_tool_calls_per_step INTEGER NOT NULL
                CHECK(max_tool_calls_per_step BETWEEN 1 AND 16),
            semantic_steps_used INTEGER NOT NULL DEFAULT 0
                CHECK(semantic_steps_used BETWEEN 0 AND 64),
            catalog_snapshot_json TEXT NOT NULL
                CHECK(length(catalog_snapshot_json) > 0),
            catalog_snapshot_hash TEXT NOT NULL
                CHECK(length(catalog_snapshot_hash) = 64),
            plan_json TEXT,
            plan_hash TEXT CHECK(plan_hash IS NULL OR length(plan_hash) = 64),
            latest_observation_json TEXT,
            latest_observation_hash TEXT CHECK(
                latest_observation_hash IS NULL
                OR length(latest_observation_hash) = 64
            ),
            final_reply TEXT,
            completion_report_json TEXT,
            completion_report_hash TEXT CHECK(
                completion_report_hash IS NULL
                OR length(completion_report_hash) = 64
            ),
            failure_code TEXT,
            updated_at TEXT NOT NULL, execution_config_json TEXT, execution_config_hash TEXT, verification_contract_version TEXT, verification_report_json TEXT, verification_report_hash TEXT, semantic_verification_contract_version TEXT, semantic_verification_report_json TEXT, semantic_verification_report_hash TEXT, corpus_manifest_contract_version TEXT, corpus_manifest_json TEXT, corpus_manifest_hash TEXT,
            UNIQUE(session_id, l1_turn_run_id),
            CHECK((plan_json IS NULL) = (plan_hash IS NULL)),
            CHECK(
                (latest_observation_json IS NULL)
                = (latest_observation_hash IS NULL)
            ),
            CHECK(
                (completion_report_json IS NULL)
                = (completion_report_hash IS NULL)
            ),
            CHECK(semantic_steps_used <= max_semantic_steps),
            FOREIGN KEY(session_id, turn_id, l1_turn_run_id)
                REFERENCES l1_turn_runs(
                    session_id, turn_id, l1_turn_run_id
                )
                ON DELETE CASCADE
        );

CREATE TABLE l1_turn_runs (
            l1_turn_run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(status IN (
                'created', 'running', 'completed', 'failed', 'cancelled'
            )),
            routing_policy_snapshot_hash TEXT NOT NULL
                CHECK(length(routing_policy_snapshot_hash) = 64),
            run_revision INTEGER NOT NULL DEFAULT 0 CHECK(run_revision >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(session_id, l1_turn_run_id),
            FOREIGN KEY(session_id, turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id)
                REFERENCES runtime_turn_routing_policy_snapshots(turn_id) ON DELETE CASCADE
        );

CREATE TABLE l1_turn_steps (
            step_id TEXT PRIMARY KEY,
            l1_turn_run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 64),
            status TEXT NOT NULL CHECK(status IN (
                'prepared', 'decided', 'observed', 'final_answer', 'failed'
            )),
            logical_model_call_id TEXT NOT NULL,
            request_json TEXT NOT NULL CHECK(length(request_json) > 0),
            request_hash TEXT NOT NULL CHECK(length(request_hash) = 64),
            state_guard_hash TEXT NOT NULL CHECK(length(state_guard_hash) = 64),
            decision_json TEXT,
            decision_hash TEXT CHECK(
                decision_hash IS NULL OR length(decision_hash) = 64
            ),
            action_kind TEXT CHECK(
                action_kind IS NULL OR action_kind IN ('call_tools', 'final_answer')
            ),
            tool_call_count INTEGER NOT NULL DEFAULT 0
                CHECK(tool_call_count BETWEEN 0 AND 16),
            observation_json TEXT,
            observation_hash TEXT CHECK(
                observation_hash IS NULL OR length(observation_hash) = 64
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(l1_turn_run_id, ordinal),
            UNIQUE(l1_turn_run_id, logical_model_call_id),
            UNIQUE(session_id, turn_id, step_id),
            UNIQUE(session_id, turn_id, l1_turn_run_id, step_id),
            CHECK((decision_json IS NULL) = (decision_hash IS NULL)),
            CHECK((observation_json IS NULL) = (observation_hash IS NULL)),
            FOREIGN KEY(session_id, turn_id, l1_turn_run_id)
                REFERENCES l1_turn_runs(
                    session_id, turn_id, l1_turn_run_id
                )
                ON DELETE CASCADE
        );

CREATE TABLE l1_turn_tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            l1_turn_run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            call_ordinal INTEGER NOT NULL CHECK(call_ordinal BETWEEN 1 AND 16),
            tool_id TEXT NOT NULL CHECK(length(tool_id) BETWEEN 1 AND 200),
            contract_version TEXT NOT NULL
                CHECK(length(contract_version) BETWEEN 1 AND 200),
            implementation_version TEXT NOT NULL
                CHECK(length(implementation_version) BETWEEN 1 AND 200),
            supports_obligation_keys_json TEXT NOT NULL,
            arguments_json TEXT NOT NULL CHECK(length(arguments_json) > 0),
            arguments_hash TEXT NOT NULL CHECK(length(arguments_hash) = 64),
            policy_json TEXT NOT NULL CHECK(length(policy_json) > 0),
            status TEXT NOT NULL CHECK(status IN (
                'pending', 'succeeded', 'rejected', 'failed',
                'timed_out', 'cancelled', 'completion_unconfirmed'
            )),
            outcome_json TEXT,
            outcome_hash TEXT CHECK(
                outcome_hash IS NULL OR length(outcome_hash) = 64
            ),
            created_at TEXT NOT NULL,
            settled_at TEXT, execution_class TEXT NOT NULL DEFAULT 'read_only', effect_profile_sha256 TEXT, provider_identity_sha256 TEXT, approval_receipt_ids_json TEXT NOT NULL DEFAULT '[]', approval_receipts_sha256 TEXT, protected_operation_binding_sha256 TEXT, protected_state_guard_sha256 TEXT, protected_phase TEXT NOT NULL DEFAULT 'not_required', physical_attempt_id TEXT, protected_receipt_json TEXT, protected_receipt_sha256 TEXT, protected_started_at TEXT, protected_settled_at TEXT,
            UNIQUE(l1_turn_run_id, step_id, call_ordinal),
            CHECK((outcome_json IS NULL) = (outcome_hash IS NULL)),
            CHECK(
                (status='pending' AND settled_at IS NULL AND outcome_json IS NULL)
                OR
                (status<>'pending' AND settled_at IS NOT NULL AND outcome_json IS NOT NULL)
            ),
            FOREIGN KEY(
                session_id, turn_id, l1_turn_run_id, step_id
            ) REFERENCES l1_turn_steps(
                session_id, turn_id, l1_turn_run_id, step_id
            )
                ON DELETE CASCADE
        );

CREATE TABLE runtime_turn_attachment_bindings (
            turn_id TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            binding_ordinal INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(turn_id, attachment_id),
            UNIQUE(turn_id, binding_ordinal),
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE CASCADE,
            FOREIGN KEY(attachment_id) REFERENCES session_attachments(attachment_id) ON DELETE RESTRICT
        );

CREATE TABLE runtime_turn_events_v1 (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            parent_event_id TEXT,
            insession_task_id TEXT,
            insession_task_node_id TEXT,
            work_run_id TEXT,
            attempt_id TEXT,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            duration_ms INTEGER,
            model_call_id TEXT,
            model_attempt INTEGER,
            operation_id TEXT,
            error_code TEXT,
            retryable INTEGER NOT NULL DEFAULT 0,
            diagnostic_ref TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE CASCADE
        );

CREATE TABLE runtime_turn_inputs (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL UNIQUE,
            turn_idx INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE CASCADE,
            FOREIGN KEY(session_id, turn_idx)
                REFERENCES session_turns(session_id, turn_idx) ON DELETE RESTRICT
        );

CREATE TABLE runtime_turn_input_file_refs (
            message_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
            project_id TEXT NOT NULL CHECK(length(project_id) > 0),
            file_id TEXT NOT NULL CHECK(length(file_id) > 0),
            file_version_id TEXT NOT NULL CHECK(length(file_version_id) > 0),
            created_at TEXT NOT NULL,
            PRIMARY KEY(message_id, ordinal),
            UNIQUE(message_id, project_id, file_id, file_version_id),
            FOREIGN KEY(message_id) REFERENCES runtime_turn_inputs(message_id)
                ON DELETE CASCADE
        );

CREATE TABLE runtime_turn_routing_policy_snapshots (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT NOT NULL CHECK(source IN (
                'default', 'session_default', 'request_override', 'legacy_compat'
            )),
            snapshot_json TEXT NOT NULL CHECK(length(snapshot_json) > 0),
            snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash) = 64),
            created_at TEXT NOT NULL,
            UNIQUE(session_id, turn_id),
            FOREIGN KEY(session_id, turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE
        );

CREATE TABLE "runtime_turns" (turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL, user_text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'incomplete', 'failed', 'rejected')), processing_level TEXT CHECK(processing_level IN ('L0', 'L1', 'L2')), error_code TEXT, received_at TEXT NOT NULL, completed_at TEXT, input_carried_into_turn_id TEXT, effective_user_text TEXT, client_request_id TEXT, input_message_id TEXT, end_reason TEXT, execution_snapshot_json TEXT, execution_snapshot_sha256 TEXT, FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE);

CREATE TABLE session_attachments (
            attachment_id       TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL,
            turn_id             TEXT,
            origin              TEXT NOT NULL
                CHECK(origin IN ('user_upload', 'agent_created', 'agent_modified')),
            original_name       TEXT NOT NULL,
            stored_rel_path     TEXT NOT NULL,
            media_type          TEXT NOT NULL,
            declared_media_type TEXT,
            size_bytes          INTEGER NOT NULL,
            content_hash        TEXT NOT NULL,
            kind                TEXT NOT NULL
                CHECK(kind IN ('image', 'text', 'document', 'audio', 'video', 'unknown')),
            project_id          TEXT,
            file_id             TEXT,
            file_version_id     TEXT,
            created_at          TEXT NOT NULL,
            bound_at            TEXT,
            CHECK(
                (project_id IS NULL AND file_id IS NULL AND file_version_id IS NULL)
                OR
                (project_id IS NOT NULL AND file_id IS NOT NULL AND file_version_id IS NOT NULL)
            ),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

CREATE TABLE session_context_repair_applies (
    preview_token TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    preview_revision INTEGER NOT NULL,
    applied_revision INTEGER NOT NULL,
    actor TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    result_json TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_context_resets (
    reset_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    cutoff_turn_idx INTEGER NOT NULL,
    cutoff_event_rowid INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    report_json TEXT NOT NULL,
    UNIQUE(session_id, request_id),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_context_revisions (
    session_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_evidence_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    content_excerpt TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(session_id, kind, source_ref, content_hash),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    previous_status TEXT
);

CREATE TABLE session_observation_candidates (
    candidate_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    state_type TEXT NOT NULL,
    state_key TEXT NOT NULL,
    proposed_value_json TEXT NOT NULL,
    operation TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    derived_from_json TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    confidence_hint REAL NOT NULL,
    valid_from TEXT,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_state_items (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    state_type TEXT NOT NULL,
    state_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    status TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    derived_from_json TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    reducer_version TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, domain, state_type, state_key),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_state_transitions (
    transition_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    old_state_ref TEXT,
    new_state_ref TEXT,
    decision TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    derived_from_json TEXT NOT NULL,
    reducer_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_turn_commit_references (
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                reference_kind TEXT NOT NULL CHECK(reference_kind IN (
                    'node_delivery', 'pending_question_attempt'
                )),
                node_delivery_id TEXT UNIQUE,
                question_attempt_id TEXT UNIQUE,
                PRIMARY KEY(run_id, ordinal),
                FOREIGN KEY(run_id) REFERENCES session_turn_commits(run_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(node_delivery_id)
                    REFERENCES insession_task_node_deliveries(delivery_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                FOREIGN KEY(question_attempt_id)
                    REFERENCES insession_work_run_attempts(attempt_id)
                    ON DELETE RESTRICT ON UPDATE RESTRICT,
                CHECK(
                    (
                        reference_kind='node_delivery'
                        AND node_delivery_id IS NOT NULL
                        AND question_attempt_id IS NULL
                    )
                    OR
                    (
                        reference_kind='pending_question_attempt'
                        AND node_delivery_id IS NULL
                        AND question_attempt_id IS NOT NULL
                    )
                )
            );

CREATE TABLE session_turn_commits (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_turn_idx INTEGER NOT NULL,
    assistant_turn_idx INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    turn_id TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_turn_routing_policies (
            session_id TEXT PRIMARY KEY,
            policy_json TEXT NOT NULL CHECK(length(policy_json) > 0),
            policy_hash TEXT NOT NULL CHECK(length(policy_hash) = 64),
            updated_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

CREATE TABLE session_turns (
    session_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (session_id, turn_idx),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE session_working_memory (
    session_id TEXT PRIMARY KEY,
    running_summary TEXT,
    summarized_upto INTEGER DEFAULT 0,
    updated_at TEXT
, summarized_through_turn_id TEXT, status TEXT NOT NULL DEFAULT 'ok' CHECK(status IN ('ok', 'stale', 'unavailable')), state_version INTEGER NOT NULL DEFAULT 0 CHECK(state_version >= 0), last_error_code TEXT);

CREATE TABLE session_workspace_read_grants (
    grant_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    canonical_root TEXT NOT NULL,
    root_device TEXT NOT NULL,
    root_inode TEXT NOT NULL,
    root_binding_sha256 TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE session_workspace_root_observations (
    session_id TEXT PRIMARY KEY,
    canonical_root TEXT NOT NULL,
    root_device TEXT NOT NULL,
    root_inode TEXT NOT NULL,
    root_binding_sha256 TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE TABLE session_workspace_write_grants (
    grant_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    canonical_root TEXT NOT NULL,
    root_device TEXT NOT NULL,
    root_inode TEXT NOT NULL,
    root_binding_sha256 TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL,
    title TEXT,
    folder_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT,
    last_active_at TEXT,
    archived_at TEXT,
    deleted_at TEXT,
    purge_after TEXT,
    previous_status TEXT,
    working_dir TEXT
);

CREATE TABLE turn_execution_windows (
            session_id TEXT PRIMARY KEY,
            turn_id TEXT,
            window_state TEXT NOT NULL CHECK(window_state IN (
                'empty', 'active', 'post_commit_pending', 'interrupted'
            )),
            stage TEXT,
            input_message_id TEXT,
            attachment_binding_revision INTEGER NOT NULL DEFAULT 0,
            turn_task_link_revision INTEGER NOT NULL DEFAULT 0,
            turn_workrun_link_revision INTEGER NOT NULL DEFAULT 0,
            current_work_run_id TEXT,
            current_attempt_id TEXT,
            latest_checkpoint_id TEXT,
            pending_operation_id TEXT,
            last_event_sequence INTEGER,
            state_version INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            heartbeat_at TEXT,
            interruption_reason TEXT,
            claimed_at TEXT,
            updated_at TEXT NOT NULL, current_l1_turn_run_id TEXT, current_l1_attempt_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE RESTRICT,
            FOREIGN KEY(input_message_id) REFERENCES runtime_turn_inputs(message_id) ON DELETE RESTRICT
        );

CREATE TABLE turn_post_commit_job_controls (
            control_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('retry', 'waive')),
            expected_window_revision INTEGER NOT NULL,
            failed_job_digest TEXT NOT NULL,
            job_ids_json TEXT NOT NULL,
            actor TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, request_id),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE CASCADE
        );

CREATE TABLE turn_post_commit_jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            job_kind TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'pending', 'processing', 'applied', 'retryable_failed', 'terminal_failed', 'waived'
            )),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
            next_retry_at TEXT,
            lease_owner TEXT,
            lease_until TEXT,
            reason_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(turn_id, job_kind),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(turn_id) REFERENCES runtime_turns(turn_id) ON DELETE CASCADE
        );

CREATE INDEX idx_aux_node_retrieval_catalog_plan
            ON insession_auxiliary_node_retrieval_catalog_bindings(
                session_id,
                insession_task_id,
                auxiliary_graph_id,
                auxiliary_graph_revision,
                plan_sha256
            );

CREATE INDEX idx_aux_replan_applications_task
                ON insession_auxiliary_replan_trigger_applications(
                    session_id, insession_task_id, created_at, apply_id
                );

CREATE INDEX idx_aux_replan_triggers_task
                ON insession_auxiliary_replan_triggers(
                    session_id, insession_task_id, created_at, trigger_id
                );

CREATE INDEX idx_aux_v2_answer_binding_source ON insession_auxiliary_v2_waiting_user_answer_bindings(session_id, answer_source_turn_id, answer_source_message_id);

CREATE INDEX idx_aux_v2_apply_goal_time
                ON insession_auxiliary_graph_revision_apply_receipts_v2(
                    goal_id, created_at, apply_id
                );

CREATE INDEX idx_aux_v2_containers_session
                ON insession_auxiliary_graph_v2_containers(
                    session_id, updated_at, auxiliary_graph_id
                );

CREATE INDEX idx_aux_v2_context_goal_node
                ON insession_auxiliary_planning_context_artifacts(
                    goal_id, producer_auxiliary_graph_revision,
                    producer_auxiliary_node_id, created_at, artifact_id
                );

CREATE INDEX idx_aux_v2_execution_receipts_run ON insession_auxiliary_v2_execution_apply_receipts(work_run_id, created_at, apply_id);

CREATE INDEX idx_aux_v2_finish_gate_goal
                ON insession_auxiliary_v2_finish_gate_receipts(
                    goal_id, auxiliary_graph_revision, created_at
                );

CREATE INDEX idx_aux_v2_goals_task_time
                ON insession_auxiliary_graph_goals(
                    session_id, insession_task_id, updated_at, goal_id
                );

CREATE INDEX idx_aux_v2_primitive_invocation_goal
                ON insession_auxiliary_planning_primitive_invocations(
                    goal_id, status, auxiliary_graph_revision,
                    auxiliary_node_id, reserved_at
                );

CREATE INDEX idx_aux_v2_revision_goal
                ON insession_auxiliary_graph_revision_snapshots(
                    goal_id, auxiliary_graph_revision, created_at
                );

CREATE INDEX idx_aux_v2_task_graph_carry_apply
                ON insession_auxiliary_v2_task_graph_node_carry_receipts(
                    apply_id, created_at, carry_receipt_id
                );

CREATE INDEX idx_aux_v2_task_graph_commit_task
                ON insession_auxiliary_v2_task_graph_commit_receipts(
                    session_id, insession_task_id,
                    committed_task_graph_revision, created_at
                );

CREATE INDEX idx_execution_finding_revisions_entry
            ON execution_finding_entry_revisions(
                ledger_id, entry_id, entry_revision
            );

CREATE INDEX idx_execution_finding_revisions_ledger_sequence
            ON execution_finding_entry_revisions(ledger_id, sequence);

CREATE INDEX idx_execution_findings_mutations_time
            ON execution_findings_mutations(ledger_id, created_at, mutation_id);

CREATE INDEX idx_execution_findings_session_time
            ON execution_findings_ledgers(
                session_id, updated_at DESC, ledger_id
            );

CREATE INDEX idx_execution_subjects_task
                ON insession_execution_subjects(
                    session_id, insession_task_id, subject_contract_version,
                    execution_subject_id
                );

CREATE INDEX idx_insession_task_branch_intents_turn
            ON insession_task_branch_intents(session_id, source_turn_id, branch_intent_id);

CREATE INDEX idx_insession_task_graph_apply_receipts_session_turn
            ON insession_task_graph_apply_receipts(session_id, source_turn_id, created_at DESC);

CREATE INDEX idx_insession_task_graph_edges_parent
                ON insession_task_graph_edges(
                    insession_task_id, graph_revision,
                    parent_insession_task_node_id, ordinal
                );

CREATE INDEX idx_insession_task_graph_nodes_lookup ON insession_task_graph_nodes(insession_task_id, graph_revision, ordinal);

CREATE INDEX idx_insession_task_graph_revision_receipts_turn
            ON insession_task_graph_revision_apply_receipts(
                session_id, source_turn_id, created_at DESC
            );

CREATE INDEX idx_insession_task_graph_revisions_turn
            ON insession_task_graph_revisions(source_turn_id, created_at DESC);

CREATE INDEX idx_insession_task_match_receipts_session_turn
            ON insession_task_match_apply_receipts(session_id, source_turn_id, created_at DESC);

CREATE INDEX idx_insession_task_node_deliveries_task ON insession_task_node_deliveries(insession_task_id, created_at, delivery_id);

CREATE INDEX idx_insession_task_node_states_status
            ON insession_task_node_states(insession_task_id, status, updated_at DESC);

CREATE INDEX idx_insession_task_turn_links_task
            ON insession_task_turn_links(insession_task_id, created_at, link_id);

CREATE INDEX idx_insession_task_turn_links_turn
            ON insession_task_turn_links(session_id, turn_id, link_id);

CREATE INDEX idx_insession_tasks_catalog ON insession_tasks(session_id, current_status, updated_at DESC, insession_task_id);

CREATE INDEX idx_insession_work_run_attempts_input_turn ON insession_work_run_attempts(input_turn_id, attempt_id);

CREATE INDEX idx_insession_work_run_budget_charges_run ON insession_work_run_budget_charges(work_run_id, work_run_revision_after, budget_charge_id);

CREATE INDEX idx_insession_work_run_outputs_session_time ON insession_work_run_output_windows(session_id, updated_at DESC, work_run_id);

CREATE INDEX idx_insession_work_run_receipts_run ON insession_work_run_apply_receipts(work_run_id, created_at, apply_id);

CREATE INDEX idx_insession_work_run_tool_calls_run
            ON insession_work_run_tool_calls(work_run_id, attempt_id, ordinal);

CREATE INDEX idx_insession_work_run_tool_results_attempt ON insession_work_run_tool_results(attempt_id, ordinal);

CREATE INDEX idx_insession_work_run_turn_links_run
            ON insession_work_run_turn_links(work_run_id, link_id);

CREATE INDEX idx_insession_work_run_verification_history
                ON insession_work_run_verification_requests(
                    work_run_id, output_revision, submitted_attempt_id, created_at
                );

CREATE INDEX idx_insession_work_runs_task_time
                ON insession_work_runs(
                    insession_task_id, updated_at DESC, work_run_id
                );

CREATE INDEX idx_l1_plan_revisions_turn
            ON l1_turn_plan_revisions(session_id, turn_id, revision);

CREATE INDEX idx_l1_turn_runs_session_time
            ON l1_turn_runs(session_id, created_at, l1_turn_run_id);

CREATE INDEX idx_l1_turn_steps_run_ordinal
            ON l1_turn_steps(l1_turn_run_id, ordinal);

CREATE INDEX idx_l1_turn_tool_calls_step
            ON l1_turn_tool_calls(step_id, call_ordinal);

CREATE INDEX idx_runtime_model_logical_calls_owner
                ON insession_runtime_model_logical_calls(
                    session_id, insession_task_id, auxiliary_graph_id, goal_id,
                    created_at, logical_call_id
                );

CREATE INDEX idx_runtime_model_logical_calls_subject
                ON insession_runtime_model_logical_calls(
                    execution_subject_id, created_at, logical_call_id
                ) WHERE execution_subject_id IS NOT NULL;

CREATE INDEX idx_runtime_model_physical_attempts_call
                ON insession_runtime_model_physical_attempts(
                    logical_call_id, physical_ordinal
                );

CREATE INDEX idx_runtime_model_rejected_output_lookup
                ON insession_runtime_model_rejected_outputs(
                    session_id, logical_call_id, physical_ordinal,
                    response_sha256
                );

CREATE INDEX idx_runtime_model_settlements_call
                ON insession_runtime_model_call_settlement_receipts(
                    logical_call_id, physical_ordinal, settled_at,
                    settle_apply_id
                );

CREATE INDEX idx_runtime_tool_logical_calls_owner
                ON insession_runtime_tool_logical_calls(
                    session_id, work_run_id, attempt_id, call_ordinal,
                    logical_tool_call_id
                );

CREATE INDEX idx_runtime_tool_physical_attempts_call
                ON insession_runtime_tool_physical_attempts(
                    logical_tool_call_id, physical_ordinal
                );

CREATE INDEX idx_runtime_tool_settlements_call
                ON insession_runtime_tool_call_settlement_receipts(
                    logical_tool_call_id, physical_ordinal, settled_at,
                    settle_apply_id
                );

CREATE INDEX idx_runtime_turn_attachment_bindings_attachment
            ON runtime_turn_attachment_bindings(attachment_id, turn_id);

CREATE INDEX idx_runtime_turn_events_v1_session_time
            ON runtime_turn_events_v1(session_id, occurred_at, event_id);

CREATE INDEX idx_runtime_turn_events_v1_turn
            ON runtime_turn_events_v1(turn_id, occurred_at, event_id);

CREATE INDEX idx_runtime_turn_inputs_session_time
            ON runtime_turn_inputs(session_id, created_at DESC, message_id DESC);

CREATE INDEX idx_runtime_turn_policy_session_time
            ON runtime_turn_routing_policy_snapshots(session_id, created_at, turn_id);

CREATE INDEX idx_runtime_turns_pending_carry_over ON runtime_turns(session_id, received_at, turn_id) WHERE status='failed' AND input_carried_into_turn_id IS NULL;

CREATE INDEX idx_runtime_turns_session_time ON runtime_turns(session_id, received_at DESC, turn_id DESC);

CREATE INDEX idx_session_attachments_turn
            ON session_attachments(session_id, turn_id, created_at);

CREATE INDEX idx_session_attachments_unbound
            ON session_attachments(session_id, created_at)
            WHERE turn_id IS NULL;

CREATE INDEX idx_session_candidates_version
    ON session_observation_candidates(session_id, extractor_version, valid_from, candidate_id);

CREATE INDEX idx_session_context_repair_applies_session
    ON session_context_repair_applies(session_id, applied_at, preview_token);

CREATE INDEX idx_session_context_resets_session_time
    ON session_context_resets(session_id, created_at, reset_id);

CREATE INDEX idx_session_evidence_events_session_time
    ON session_evidence_events(session_id, created_at, id);

CREATE INDEX idx_session_state_items_session_status
    ON session_state_items(session_id, status, domain, state_type, state_key);

CREATE INDEX idx_session_state_transitions_session_time
    ON session_state_transitions(session_id, created_at, transition_id);

CREATE INDEX idx_session_turn_commits_session
    ON session_turn_commits(session_id, user_turn_idx);

CREATE UNIQUE INDEX idx_session_workspace_active_grant
    ON session_workspace_read_grants(session_id)
    WHERE revoked_at IS NULL;

CREATE INDEX idx_session_workspace_grant_history
    ON session_workspace_read_grants(session_id, granted_at, grant_id);

CREATE UNIQUE INDEX idx_session_workspace_active_write_grant
    ON session_workspace_write_grants(session_id)
    WHERE revoked_at IS NULL;

CREATE INDEX idx_session_workspace_write_grant_history
    ON session_workspace_write_grants(session_id, granted_at, grant_id);

CREATE INDEX idx_task_delivery_validation_request
                ON insession_task_delivery_validation_requests(
                    session_id, insession_task_id, graph_revision, created_at
                );

CREATE INDEX idx_task_delivery_validation_settlement
                ON insession_task_delivery_validation_settlements(
                    session_id, insession_task_id, graph_revision, created_at
                );

CREATE INDEX idx_task_node_retrieval_catalog_plan
            ON insession_task_node_retrieval_catalog_bindings(
                session_id, insession_task_id, graph_revision, plan_sha256
            );

CREATE INDEX idx_turn_execution_windows_active
            ON turn_execution_windows(window_state, heartbeat_at, updated_at)
            WHERE turn_id IS NOT NULL;

CREATE INDEX idx_turn_post_commit_job_controls_turn_time
            ON turn_post_commit_job_controls(turn_id, created_at DESC, control_id DESC);

CREATE INDEX idx_turn_post_commit_jobs_due
            ON turn_post_commit_jobs(status, next_retry_at, lease_until, created_at, job_id);

CREATE INDEX idx_turn_post_commit_jobs_turn
            ON turn_post_commit_jobs(turn_id, job_kind);

CREATE UNIQUE INDEX uq_aux_v2_one_nonterminal_goal
                ON insession_auxiliary_graph_goals(auxiliary_graph_id)
                WHERE status IN (
                    'active', 'waiting_user', 'waiting_authorization',
                    'waiting_external', 'interrupted', 'proposal_ready',
                    'gapped_ready'
                );

CREATE UNIQUE INDEX uq_aux_v2_one_reserved_primitive_per_session
                ON insession_auxiliary_planning_primitive_invocations(session_id)
                WHERE status='reserved';

CREATE UNIQUE INDEX uq_execution_findings_l1_owner
            ON execution_findings_ledgers(l1_turn_run_id)
            WHERE owner_kind = 'l1_turn_run';

CREATE UNIQUE INDEX uq_execution_findings_work_owner
            ON execution_findings_ledgers(work_run_id)
            WHERE owner_kind = 'work_run';

CREATE UNIQUE INDEX uq_graph_revision_receipts_recipe_authority ON insession_task_graph_revision_apply_receipts(session_id, insession_task_id, committed_graph_revision, apply_id, source_turn_id);

CREATE UNIQUE INDEX uq_insession_task_graph_nodes_revision_binding
            ON insession_task_graph_nodes(
                insession_task_id, graph_revision,
                insession_task_node_id, node_revision
            );

CREATE UNIQUE INDEX uq_insession_task_turn_links_exact
            ON insession_task_turn_links(
                turn_id, insession_task_id, COALESCE(insession_task_node_id, '')
            );

CREATE UNIQUE INDEX uq_insession_tasks_session_identity ON insession_tasks(session_id, insession_task_id);

CREATE UNIQUE INDEX uq_insession_tasks_session_task
            ON insession_tasks(session_id, insession_task_id);

CREATE UNIQUE INDEX uq_insession_work_run_attempts_active ON insession_work_run_attempts(work_run_id) WHERE status = 'active';

CREATE UNIQUE INDEX uq_insession_work_run_attempts_budget_charge ON insession_work_run_attempts(budget_charge_id) WHERE budget_charge_id IS NOT NULL;

CREATE UNIQUE INDEX uq_insession_work_run_calls_one_environment_change
            ON insession_work_run_tool_calls(attempt_id)
            WHERE modifies_environment = 1;

CREATE UNIQUE INDEX uq_insession_work_run_verification_current
                ON insession_work_run_verification_requests(work_run_id)
                WHERE status IN ('pending', 'interrupted');

CREATE UNIQUE INDEX uq_insession_work_runs_active_session
                ON insession_work_runs(session_id) WHERE status = 'active';

CREATE UNIQUE INDEX uq_insession_work_runs_nonterminal_aux_subject
                ON insession_work_runs(
                    execution_subject_id
                ) WHERE subject_kind='auxiliary_node'
                    AND status NOT IN ('completed', 'failed', 'cancelled');

CREATE UNIQUE INDEX uq_insession_work_runs_nonterminal_task_subject
                ON insession_work_runs(
                    insession_task_id, insession_task_node_id, node_revision
                ) WHERE subject_kind='task_node'
                    AND status NOT IN ('completed', 'failed', 'cancelled');

CREATE UNIQUE INDEX uq_l1_turn_run_owner
            ON l1_turn_runs(session_id, turn_id, l1_turn_run_id);

CREATE UNIQUE INDEX uq_l1_turn_tool_calls_physical_attempt ON l1_turn_tool_calls(physical_attempt_id) WHERE physical_attempt_id IS NOT NULL;

CREATE UNIQUE INDEX uq_runtime_model_physical_attempt_owner_ordinal ON insession_runtime_model_physical_attempts(session_id, logical_call_id, physical_attempt_id, physical_ordinal);

CREATE UNIQUE INDEX uq_runtime_turns_session_client_request ON runtime_turns(session_id, client_request_id) WHERE client_request_id IS NOT NULL;

CREATE UNIQUE INDEX uq_runtime_turns_session_turn ON runtime_turns(session_id, turn_id);

CREATE UNIQUE INDEX uq_session_turn_commits_turn_id ON session_turn_commits(turn_id) WHERE turn_id IS NOT NULL;

CREATE UNIQUE INDEX uq_tool_results_dossier_authority ON insession_work_run_tool_results(work_run_id, tool_result_id);

CREATE UNIQUE INDEX uq_turn_execution_windows_turn
            ON turn_execution_windows(turn_id) WHERE turn_id IS NOT NULL;

CREATE UNIQUE INDEX uq_v59_delivery_dossier_subject ON insession_task_node_deliveries(session_id, insession_task_id, graph_revision, insession_task_node_id, node_revision, work_run_id, verification_request_id, delivery_id);

CREATE UNIQUE INDEX uq_v59_verification_dossier_subject
                ON insession_work_run_verification_requests(
                    session_id, insession_task_id, graph_revision,
                    insession_task_node_id, node_revision, work_run_id,
                    verification_request_id
                );

CREATE UNIQUE INDEX uq_v60_attempt_submitted_output_authority ON insession_work_run_attempts(work_run_id, attempt_id, submitted_output_revision, action);

CREATE UNIQUE INDEX uq_v60_output_window_authority ON insession_work_run_output_windows(work_run_id, output_revision, snapshot_hash);

CREATE UNIQUE INDEX uq_v60_tool_call_evidence_authority ON insession_work_run_tool_calls(work_run_id, attempt_id, tool_call_id, tool_id, tool_version, arguments_hash);

CREATE UNIQUE INDEX uq_v60_tool_result_evidence_authority ON insession_work_run_tool_results(work_run_id, attempt_id, tool_call_id, tool_result_id, status);

CREATE UNIQUE INDEX uq_v60_work_run_apply_receipt_authority ON insession_work_run_apply_receipts(session_id, work_run_id, apply_id, operation);

CREATE UNIQUE INDEX uq_work_run_aux_subject_binding_v63
                ON insession_work_runs(
                    session_id, insession_task_id, auxiliary_graph_id,
                    auxiliary_graph_revision, auxiliary_node_id, node_revision,
                    work_run_id
                );

CREATE UNIQUE INDEX uq_work_runs_dossier_subject_authority
                ON insession_work_runs(
                    session_id, insession_task_id, graph_revision,
                    insession_task_node_id, node_revision, work_run_id
                );

CREATE TRIGGER trg_aux_v2_completion_execution_subject_insert BEFORE INSERT ON insession_auxiliary_node_completions_v2 BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_work_runs AS run
        JOIN insession_execution_subjects AS subject
          ON subject.execution_subject_id=run.execution_subject_id
        JOIN insession_work_run_verification_requests AS request
          ON request.work_run_id=run.work_run_id
         AND request.verification_request_id=NEW.verification_request_id
         AND request.execution_subject_id=run.execution_subject_id
        WHERE run.session_id=NEW.session_id
          AND run.work_run_id=NEW.work_run_id
          AND run.insession_task_id=NEW.insession_task_id
          AND run.auxiliary_graph_id=NEW.auxiliary_graph_id
          AND run.auxiliary_graph_revision=NEW.auxiliary_graph_revision
          AND run.auxiliary_node_id=NEW.auxiliary_node_id
          AND run.node_revision=NEW.node_revision
          AND subject.subject_contract_version='auxiliary_node_v2'
    ) THEN RAISE(ABORT, 'V2 completion execution subject mismatch') END; END;

CREATE TRIGGER trg_aux_v2_completion_execution_subject_update BEFORE UPDATE OF session_id, insession_task_id, auxiliary_graph_id, auxiliary_graph_revision, auxiliary_node_id, node_revision, work_run_id, verification_request_id ON insession_auxiliary_node_completions_v2 BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_work_runs AS run
        JOIN insession_execution_subjects AS subject
          ON subject.execution_subject_id=run.execution_subject_id
        JOIN insession_work_run_verification_requests AS request
          ON request.work_run_id=run.work_run_id
         AND request.verification_request_id=NEW.verification_request_id
         AND request.execution_subject_id=run.execution_subject_id
        WHERE run.session_id=NEW.session_id
          AND run.work_run_id=NEW.work_run_id
          AND run.insession_task_id=NEW.insession_task_id
          AND run.auxiliary_graph_id=NEW.auxiliary_graph_id
          AND run.auxiliary_graph_revision=NEW.auxiliary_graph_revision
          AND run.auxiliary_node_id=NEW.auxiliary_node_id
          AND run.node_revision=NEW.node_revision
          AND subject.subject_contract_version='auxiliary_node_v2'
    ) THEN RAISE(ABORT, 'V2 completion execution subject mismatch') END; END;

CREATE TRIGGER trg_close_terminal_work_run_findings
        AFTER UPDATE OF status ON insession_work_runs
        WHEN NEW.status IN ('completed', 'failed', 'cancelled')
        BEGIN
            UPDATE execution_findings_ledgers
            SET status='closed',
                closed_at=COALESCE(closed_at, NEW.updated_at),
                updated_at=NEW.updated_at
            WHERE owner_kind='work_run'
              AND work_run_id=NEW.work_run_id
              AND status='open';
        END;

CREATE TRIGGER trg_insession_work_run_verification_requests_execution_subject_insert BEFORE INSERT ON insession_work_run_verification_requests BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_execution_subjects AS subject
        LEFT JOIN insession_task_node_execution_subject_bindings AS task_binding
          ON task_binding.binding_id=subject.task_node_binding_id
        LEFT JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2
          ON aux2.binding_id=subject.auxiliary_v2_binding_id
        WHERE subject.execution_subject_id=NEW.execution_subject_id
          AND subject.session_id=NEW.session_id
          AND subject.insession_task_id=NEW.insession_task_id
          AND subject.subject_kind=NEW.subject_kind
          AND (
            (subject.subject_contract_version='task_node_v1'
              AND task_binding.graph_revision IS NEW.graph_revision
              AND task_binding.insession_task_node_id
                    IS NEW.insession_task_node_id
              AND task_binding.node_revision=NEW.node_revision
              AND NEW.auxiliary_graph_id IS NULL
              AND NEW.auxiliary_graph_revision IS NULL
              AND NEW.auxiliary_node_id IS NULL)
            OR
            (subject.subject_contract_version='auxiliary_node_v2'
              AND NEW.graph_revision IS NULL
              AND NEW.insession_task_node_id IS NULL
              AND aux2.auxiliary_graph_id IS NEW.auxiliary_graph_id
              AND aux2.auxiliary_graph_revision
                    IS NEW.auxiliary_graph_revision
              AND aux2.auxiliary_node_id IS NEW.auxiliary_node_id
              AND aux2.node_revision=NEW.node_revision)
          )
    ) THEN RAISE(ABORT, 'execution subject projection mismatch') END; END;

CREATE TRIGGER trg_insession_work_run_verification_requests_execution_subject_update BEFORE UPDATE OF execution_subject_id, session_id, subject_kind, insession_task_id, graph_revision, insession_task_node_id, auxiliary_graph_id, auxiliary_graph_revision, auxiliary_node_id, node_revision ON insession_work_run_verification_requests BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_execution_subjects AS subject
        LEFT JOIN insession_task_node_execution_subject_bindings AS task_binding
          ON task_binding.binding_id=subject.task_node_binding_id
        LEFT JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2
          ON aux2.binding_id=subject.auxiliary_v2_binding_id
        WHERE subject.execution_subject_id=NEW.execution_subject_id
          AND subject.session_id=NEW.session_id
          AND subject.insession_task_id=NEW.insession_task_id
          AND subject.subject_kind=NEW.subject_kind
          AND (
            (subject.subject_contract_version='task_node_v1'
              AND task_binding.graph_revision IS NEW.graph_revision
              AND task_binding.insession_task_node_id
                    IS NEW.insession_task_node_id
              AND task_binding.node_revision=NEW.node_revision
              AND NEW.auxiliary_graph_id IS NULL
              AND NEW.auxiliary_graph_revision IS NULL
              AND NEW.auxiliary_node_id IS NULL)
            OR
            (subject.subject_contract_version='auxiliary_node_v2'
              AND NEW.graph_revision IS NULL
              AND NEW.insession_task_node_id IS NULL
              AND aux2.auxiliary_graph_id IS NEW.auxiliary_graph_id
              AND aux2.auxiliary_graph_revision
                    IS NEW.auxiliary_graph_revision
              AND aux2.auxiliary_node_id IS NEW.auxiliary_node_id
              AND aux2.node_revision=NEW.node_revision)
          )
    ) THEN RAISE(ABORT, 'execution subject projection mismatch') END; END;

CREATE TRIGGER trg_insession_work_runs_execution_subject_insert BEFORE INSERT ON insession_work_runs BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_execution_subjects AS subject
        LEFT JOIN insession_task_node_execution_subject_bindings AS task_binding
          ON task_binding.binding_id=subject.task_node_binding_id
        LEFT JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2
          ON aux2.binding_id=subject.auxiliary_v2_binding_id
        WHERE subject.execution_subject_id=NEW.execution_subject_id
          AND subject.session_id=NEW.session_id
          AND subject.insession_task_id=NEW.insession_task_id
          AND subject.subject_kind=NEW.subject_kind
          AND (
            (subject.subject_contract_version='task_node_v1'
              AND task_binding.graph_revision IS NEW.graph_revision
              AND task_binding.insession_task_node_id
                    IS NEW.insession_task_node_id
              AND task_binding.node_revision=NEW.node_revision
              AND NEW.auxiliary_graph_id IS NULL
              AND NEW.auxiliary_graph_revision IS NULL
              AND NEW.auxiliary_node_id IS NULL)
            OR
            (subject.subject_contract_version='auxiliary_node_v2'
              AND NEW.graph_revision IS NULL
              AND NEW.insession_task_node_id IS NULL
              AND aux2.auxiliary_graph_id IS NEW.auxiliary_graph_id
              AND aux2.auxiliary_graph_revision
                    IS NEW.auxiliary_graph_revision
              AND aux2.auxiliary_node_id IS NEW.auxiliary_node_id
              AND aux2.node_revision=NEW.node_revision)
          )
    ) THEN RAISE(ABORT, 'execution subject projection mismatch') END; END;

CREATE TRIGGER trg_insession_work_runs_execution_subject_update BEFORE UPDATE OF execution_subject_id, session_id, subject_kind, insession_task_id, graph_revision, insession_task_node_id, auxiliary_graph_id, auxiliary_graph_revision, auxiliary_node_id, node_revision ON insession_work_runs BEGIN SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM insession_execution_subjects AS subject
        LEFT JOIN insession_task_node_execution_subject_bindings AS task_binding
          ON task_binding.binding_id=subject.task_node_binding_id
        LEFT JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2
          ON aux2.binding_id=subject.auxiliary_v2_binding_id
        WHERE subject.execution_subject_id=NEW.execution_subject_id
          AND subject.session_id=NEW.session_id
          AND subject.insession_task_id=NEW.insession_task_id
          AND subject.subject_kind=NEW.subject_kind
          AND (
            (subject.subject_contract_version='task_node_v1'
              AND task_binding.graph_revision IS NEW.graph_revision
              AND task_binding.insession_task_node_id
                    IS NEW.insession_task_node_id
              AND task_binding.node_revision=NEW.node_revision
              AND NEW.auxiliary_graph_id IS NULL
              AND NEW.auxiliary_graph_revision IS NULL
              AND NEW.auxiliary_node_id IS NULL)
            OR
            (subject.subject_contract_version='auxiliary_node_v2'
              AND NEW.graph_revision IS NULL
              AND NEW.insession_task_node_id IS NULL
              AND aux2.auxiliary_graph_id IS NEW.auxiliary_graph_id
              AND aux2.auxiliary_graph_revision
                    IS NEW.auxiliary_graph_revision
              AND aux2.auxiliary_node_id IS NEW.auxiliary_node_id
              AND aux2.node_revision=NEW.node_revision)
          )
    ) THEN RAISE(ABORT, 'execution subject projection mismatch') END; END;

CREATE TRIGGER trg_runtime_turn_execution_snapshot_immutable
        BEFORE UPDATE OF execution_snapshot_json, execution_snapshot_sha256
        ON runtime_turns
        WHEN
            NEW.execution_snapshot_json IS NOT OLD.execution_snapshot_json
            OR NEW.execution_snapshot_sha256 IS NOT OLD.execution_snapshot_sha256
        BEGIN
            SELECT RAISE(ABORT, 'Runtime Turn execution snapshot is immutable');
        END;

CREATE TRIGGER trg_runtime_turn_execution_snapshot_insert
        BEFORE INSERT ON runtime_turns
        WHEN
            (NEW.execution_snapshot_json IS NULL)
                != (NEW.execution_snapshot_sha256 IS NULL)
            OR (
                NEW.execution_snapshot_json IS NOT NULL
                AND (
                    length(CAST(NEW.execution_snapshot_json AS BLOB)) > 65536
                    OR length(NEW.execution_snapshot_sha256) != 64
                    OR NEW.execution_snapshot_sha256 GLOB '*[^0-9a-f]*'
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'invalid Runtime Turn execution snapshot');
        END;

CREATE TRIGGER trg_session_context_resets_context_revision_delete AFTER DELETE ON session_context_resets BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_context_resets_context_revision_insert AFTER INSERT ON session_context_resets BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_context_resets_context_revision_update AFTER UPDATE ON session_context_resets BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_evidence_events_context_revision_delete AFTER DELETE ON session_evidence_events BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_evidence_events_context_revision_insert AFTER INSERT ON session_evidence_events BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_evidence_events_context_revision_update AFTER UPDATE ON session_evidence_events BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_observation_candidates_context_revision_delete AFTER DELETE ON session_observation_candidates BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_observation_candidates_context_revision_insert AFTER INSERT ON session_observation_candidates BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_observation_candidates_context_revision_update AFTER UPDATE ON session_observation_candidates BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_items_context_revision_delete AFTER DELETE ON session_state_items BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_items_context_revision_insert AFTER INSERT ON session_state_items BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_items_context_revision_update AFTER UPDATE ON session_state_items BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_transitions_context_revision_delete AFTER DELETE ON session_state_transitions BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_transitions_context_revision_insert AFTER INSERT ON session_state_transitions BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_state_transitions_context_revision_update AFTER UPDATE ON session_state_transitions BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_turns_context_revision_delete AFTER DELETE ON session_turns BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT OLD.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=OLD.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_turns_context_revision_insert AFTER INSERT ON session_turns BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_session_turns_context_revision_update AFTER UPDATE ON session_turns BEGIN INSERT INTO session_context_revisions(session_id, revision) SELECT NEW.session_id, 1 WHERE EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id) ON CONFLICT(session_id) DO UPDATE SET revision=revision+1; END;

CREATE TRIGGER trg_turn_window_execution_lane_exclusive_insert
            BEFORE INSERT ON turn_execution_windows
            WHEN (
                (NEW.current_l1_turn_run_id IS NOT NULL
                 OR NEW.current_l1_attempt_id IS NOT NULL)
                AND
                (NEW.current_work_run_id IS NOT NULL
                 OR NEW.current_attempt_id IS NOT NULL)
            ) OR (
                NEW.current_l1_attempt_id IS NOT NULL
                AND NEW.current_l1_turn_run_id IS NULL
            )
            BEGIN
                SELECT RAISE(ABORT, 'Turn Window execution lanes are mutually exclusive');
            END;

CREATE TRIGGER trg_turn_window_execution_lane_exclusive_update
            BEFORE UPDATE OF
                current_work_run_id,
                current_attempt_id,
                current_l1_turn_run_id,
                current_l1_attempt_id
            ON turn_execution_windows
            WHEN (
                (NEW.current_l1_turn_run_id IS NOT NULL
                 OR NEW.current_l1_attempt_id IS NOT NULL)
                AND
                (NEW.current_work_run_id IS NOT NULL
                 OR NEW.current_attempt_id IS NOT NULL)
            ) OR (
                NEW.current_l1_attempt_id IS NOT NULL
                AND NEW.current_l1_turn_run_id IS NULL
            )
            BEGIN
                SELECT RAISE(ABORT, 'Turn Window execution lanes are mutually exclusive');
            END;

CREATE TABLE trajectory_blobs (
    sha256        TEXT PRIMARY KEY,
    byte_count    INTEGER NOT NULL,
    text          TEXT NOT NULL,
    truncated     INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE trajectory_steps (
    step_id         TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    session_id      TEXT,
    turn_id         TEXT,
    model_call_id   TEXT,
    purpose         TEXT,
    duration_ms     INTEGER,
    outcome         TEXT NOT NULL,
    reason_code     TEXT,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE trajectory_parts (
    step_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL,
    PRIMARY KEY (step_id, seq),
    FOREIGN KEY (step_id) REFERENCES trajectory_steps(step_id) ON DELETE CASCADE,
    FOREIGN KEY (blob_sha256) REFERENCES trajectory_blobs(sha256)
);

CREATE INDEX idx_trajectory_step_turn
    ON trajectory_steps(turn_id, occurred_at, step_id);
CREATE INDEX idx_trajectory_step_session
    ON trajectory_steps(session_id, occurred_at, step_id);
CREATE INDEX idx_trajectory_step_call
    ON trajectory_steps(model_call_id);
CREATE INDEX idx_trajectory_part_blob
    ON trajectory_parts(blob_sha256);

CREATE TABLE doc_mounts (
    doc_id      TEXT NOT NULL,
    session_id TEXT NOT NULL,
    mounted_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_doc_mounts_session
    ON doc_mounts(session_id, mounted_at, doc_id);

CREATE TABLE doc_retrieval_snapshots (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_doc_retrieval_snapshots_session
    ON doc_retrieval_snapshots(session_id, created_at DESC, id);
"""
