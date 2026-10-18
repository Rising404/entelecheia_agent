"""当前 sessions.sqlite 的声明式 schema 基线。"""
from __future__ import annotations

CURRENT_SCHEMA_SQL = r"""
CREATE TABLE doc_mounts (
    doc_id      TEXT NOT NULL,
    session_id TEXT NOT NULL,
    mounted_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE doc_retrieval_snapshots (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

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
                'l1_turn_run'
            )),
    execution_owner_id TEXT NOT NULL,
    l1_turn_run_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    quota_json TEXT NOT NULL CHECK(length(quota_json) > 0),
    quota_hash TEXT NOT NULL CHECK(length(quota_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT,
    UNIQUE(session_id, ledger_id),
    CHECK(
                (status = 'open' AND closed_at IS NULL)
                OR (status = 'closed' AND closed_at IS NOT NULL)
            ),
    FOREIGN KEY(session_id, originating_turn_id)
                REFERENCES runtime_turns(session_id, turn_id) ON DELETE CASCADE,
    FOREIGN KEY(session_id, originating_turn_id, l1_turn_run_id)
                REFERENCES l1_turn_runs(session_id, turn_id, l1_turn_run_id)
                ON DELETE CASCADE,
    CHECK(l1_turn_run_id = execution_owner_id)
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

CREATE TABLE runtime_model_call_settlement_receipts (
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
                ) REFERENCES runtime_model_physical_attempts(
                    session_id, logical_call_id, physical_attempt_id,
                    binding_sha256, physical_ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, settled_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE runtime_model_logical_calls (
    logical_call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
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
    FOREIGN KEY(session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
    FOREIGN KEY(session_id, invocation_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
);

CREATE TABLE runtime_model_physical_attempts (
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
                ) REFERENCES runtime_model_logical_calls(
                    session_id, logical_call_id, binding_sha256, provider,
                    model, endpoint_fingerprint, request_sha256
                ) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY(session_id, started_turn_id)
                    REFERENCES runtime_turns(session_id, turn_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            );

CREATE TABLE runtime_model_rejected_outputs (
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
                ) REFERENCES runtime_model_physical_attempts(
                    session_id, logical_call_id, physical_attempt_id,
                    physical_ordinal
                ) ON DELETE CASCADE ON UPDATE RESTRICT
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

CREATE TABLE "runtime_turns" (turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL, user_text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'incomplete', 'failed', 'rejected')), processing_level TEXT CHECK(processing_level IN ('L0', 'L1')), error_code TEXT, received_at TEXT NOT NULL, completed_at TEXT, input_carried_into_turn_id TEXT, effective_user_text TEXT, client_request_id TEXT, input_message_id TEXT, end_reason TEXT, execution_snapshot_json TEXT, execution_snapshot_sha256 TEXT, FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE);

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

CREATE TABLE trajectory_blobs (
    sha256        TEXT PRIMARY KEY,
    byte_count    INTEGER NOT NULL,
    text          TEXT NOT NULL,
    truncated     INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL
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

CREATE TABLE turn_execution_windows (
    session_id TEXT PRIMARY KEY,
    turn_id TEXT,
    window_state TEXT NOT NULL CHECK(window_state IN (
                'empty', 'active', 'post_commit_pending', 'interrupted'
            )),
    stage TEXT,
    input_message_id TEXT,
    attachment_binding_revision INTEGER NOT NULL DEFAULT 0,
    latest_checkpoint_id TEXT,
    pending_operation_id TEXT,
    last_event_sequence INTEGER,
    state_version INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    heartbeat_at TEXT,
    interruption_reason TEXT,
    claimed_at TEXT,
    updated_at TEXT NOT NULL,
    current_l1_turn_run_id TEXT,
    current_l1_attempt_id TEXT,
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

CREATE INDEX idx_doc_mounts_session
    ON doc_mounts(session_id, mounted_at, doc_id);

CREATE INDEX idx_doc_retrieval_snapshots_session
    ON doc_retrieval_snapshots(session_id, created_at DESC, id);

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

CREATE INDEX idx_l1_plan_revisions_turn
            ON l1_turn_plan_revisions(session_id, turn_id, revision);

CREATE INDEX idx_l1_turn_runs_session_time
            ON l1_turn_runs(session_id, created_at, l1_turn_run_id);

CREATE INDEX idx_l1_turn_steps_run_ordinal
            ON l1_turn_steps(l1_turn_run_id, ordinal);

CREATE INDEX idx_l1_turn_tool_calls_step
            ON l1_turn_tool_calls(step_id, call_ordinal);

CREATE INDEX idx_runtime_model_physical_attempts_call
                ON runtime_model_physical_attempts(
                    logical_call_id, physical_ordinal
                );

CREATE INDEX idx_runtime_model_rejected_output_lookup
                ON runtime_model_rejected_outputs(
                    session_id, logical_call_id, physical_ordinal,
                    response_sha256
                );

CREATE INDEX idx_runtime_model_settlements_call
                ON runtime_model_call_settlement_receipts(
                    logical_call_id, physical_ordinal, settled_at,
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

CREATE UNIQUE INDEX idx_session_workspace_active_write_grant
    ON session_workspace_write_grants(session_id)
    WHERE revoked_at IS NULL;

CREATE INDEX idx_session_workspace_grant_history
    ON session_workspace_read_grants(session_id, granted_at, grant_id);

CREATE INDEX idx_session_workspace_write_grant_history
    ON session_workspace_write_grants(session_id, granted_at, grant_id);

CREATE INDEX idx_trajectory_part_blob
    ON trajectory_parts(blob_sha256);

CREATE INDEX idx_trajectory_step_call
    ON trajectory_steps(model_call_id);

CREATE INDEX idx_trajectory_step_session
    ON trajectory_steps(session_id, occurred_at, step_id);

CREATE INDEX idx_trajectory_step_turn
    ON trajectory_steps(turn_id, occurred_at, step_id);

CREATE INDEX idx_turn_execution_windows_active
            ON turn_execution_windows(window_state, heartbeat_at, updated_at)
            WHERE turn_id IS NOT NULL;

CREATE INDEX idx_turn_post_commit_job_controls_turn_time
            ON turn_post_commit_job_controls(turn_id, created_at DESC, control_id DESC);

CREATE INDEX idx_turn_post_commit_jobs_due
            ON turn_post_commit_jobs(status, next_retry_at, lease_until, created_at, job_id);

CREATE INDEX idx_turn_post_commit_jobs_turn
            ON turn_post_commit_jobs(turn_id, job_kind);

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

CREATE UNIQUE INDEX uq_execution_findings_l1_owner
            ON execution_findings_ledgers(l1_turn_run_id)
            WHERE owner_kind = 'l1_turn_run';

CREATE UNIQUE INDEX uq_l1_turn_run_owner
            ON l1_turn_runs(session_id, turn_id, l1_turn_run_id);

CREATE UNIQUE INDEX uq_l1_turn_tool_calls_physical_attempt ON l1_turn_tool_calls(physical_attempt_id) WHERE physical_attempt_id IS NOT NULL;

CREATE UNIQUE INDEX uq_runtime_model_physical_attempt_owner_ordinal ON runtime_model_physical_attempts(session_id, logical_call_id, physical_attempt_id, physical_ordinal);

CREATE UNIQUE INDEX uq_runtime_turns_session_client_request ON runtime_turns(session_id, client_request_id) WHERE client_request_id IS NOT NULL;

CREATE UNIQUE INDEX uq_runtime_turns_session_turn ON runtime_turns(session_id, turn_id);

CREATE UNIQUE INDEX uq_session_turn_commits_turn_id ON session_turn_commits(turn_id) WHERE turn_id IS NOT NULL;

CREATE UNIQUE INDEX uq_turn_execution_windows_turn
            ON turn_execution_windows(turn_id) WHERE turn_id IS NOT NULL;
"""
