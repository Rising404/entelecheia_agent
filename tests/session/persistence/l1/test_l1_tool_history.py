"""当前 L1 执行的只读历史结果：范围、分页与原始证据身份。"""

import hashlib
import json
import sqlite3

import pytest

from personagraph.persistent_turn_content.tool_results import ToolHistoryError
from personagraph.output_protocol.l1_persistence import legacy_l1_tool_result_id
from personagraph.session.persistence.l1.tool_history import L1ToolHistoryReader


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "session.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE l1_turn_runs (l1_turn_run_id TEXT PRIMARY KEY, session_id TEXT);
            CREATE TABLE l1_turn_steps (step_id TEXT PRIMARY KEY, l1_turn_run_id TEXT, ordinal INTEGER);
            CREATE TABLE l1_turn_tool_calls (
                tool_call_id TEXT PRIMARY KEY, l1_turn_run_id TEXT, session_id TEXT, step_id TEXT,
                call_ordinal INTEGER, tool_id TEXT, arguments_json TEXT, arguments_hash TEXT,
                status TEXT, outcome_json TEXT, outcome_hash TEXT);
            INSERT INTO l1_turn_runs VALUES ('run-a', 'session-a'), ('run-b', 'session-a');
            INSERT INTO l1_turn_steps VALUES ('step-a1', 'run-a', 1), ('step-a2', 'run-a', 2),
                                             ('step-b1', 'run-b', 1), ('step-b3', 'run-b', 3);
        """)
        rows = []
        for call_id, step, ordinal, run, status in (
            ("call-2", "step-a2", 1, "run-a", "failed"),
            ("call-1", "step-a1", 1, "run-a", "succeeded"),
            ("call-foreign", "step-b1", 1, "run-b", "succeeded"),
            ("call-foreign-only", "step-b3", 1, "run-b", "succeeded"),
        ):
            args = _canonical({"query": "检索词" * 300})
            outcome = _canonical(
                {
                    "status": status,
                    "result": {"text": "中文📄" * 100},
                    "error": None
                    if status == "succeeded"
                    else {"code": "file_unavailable"},
                }
            )
            rows.append(
                (
                    call_id,
                    run,
                    "session-a",
                    step,
                    ordinal,
                    "read_pdf_text",
                    args,
                    hashlib.sha256(args.encode()).hexdigest(),
                    status,
                    outcome,
                    hashlib.sha256(outcome.encode()).hexdigest(),
                )
            )
        conn.executemany(
            "INSERT INTO l1_turn_tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.execute(
            "INSERT INTO l1_turn_tool_calls VALUES ('pending','run-a','session-a','step-a2',2,"
            "'read_pdf_text','{}',?,'pending',NULL,NULL)",
            (hashlib.sha256(b"{}").hexdigest(),),
        )
    reader = L1ToolHistoryReader(
        database_path=path, session_id="session-a", l1_turn_run_id="run-a"
    )
    return path, reader, rows


def test_history_lists_only_bound_run_in_step_order_with_bounded_arguments(history):
    path, reader, _ = history
    before = path.read_bytes()
    first = reader.list_results(limit=1)
    assert first.total_results == 2
    assert first.next_offset == 1 and first.partial is True
    assert first.results[0].source.tool_call_id == "call-1"
    assert first.results[0].source.call_ref == "c1.1"
    assert first.results[0].arguments_truncated is True
    assert len(first.results[0].arguments_summary) == 512
    second = reader.list_results(offset=first.next_offset, limit=1)
    assert second.results[0].source.status == "failed"
    assert second.results[0].source.tool_call_id == "call-2"
    assert second.results[0].source.call_ref == "c2.1"
    assert second.next_offset is None
    assert path.read_bytes() == before


def test_history_reader_verifies_complete_outcome_without_rewriting_it(history):
    path, reader, rows = history
    before = path.read_bytes()
    original = rows[1][-2]
    item = reader.list_results().results[0]
    record = reader.read_result(call_ref=item.source.call_ref)
    assert record.source == item.source
    assert record.outcome == json.loads(original)
    assert hashlib.sha256(_canonical(record.outcome).encode()).hexdigest() == item.source.result_sha256
    assert path.read_bytes() == before


def test_history_errors_are_readable_but_keep_the_original_error_status(history):
    _, reader, _ = history
    item = reader.list_results().results[1]
    page = reader.read_result(call_ref=item.source.call_ref)
    assert page.source.status == "failed"
    assert page.outcome["error"]["code"] == "file_unavailable"


def test_history_rejects_unknown_foreign_and_wrong_session_scope(history):
    path, reader, _ = history
    for call_ref in ("c99.1", "c3.1", "c2.2"):
        with pytest.raises(ToolHistoryError, match="tool_result_unavailable"):
            reader.read_result(call_ref=call_ref)
    other = L1ToolHistoryReader(
        database_path=path, session_id="session-b", l1_turn_run_id="run-a"
    )
    with pytest.raises(ToolHistoryError, match="tool_history_scope_unavailable"):
        other.list_results()
    with pytest.raises(ToolHistoryError, match="tool_history_scope_unavailable"):
        other.read_result(call_ref="c1.1")


def test_same_short_reference_resolves_only_within_host_bound_run(history):
    path, reader, _ = history
    foreign = L1ToolHistoryReader(
        database_path=path, session_id="session-a", l1_turn_run_id="run-b"
    )
    assert reader.read_result(call_ref="c1.1").source.tool_call_id == "call-1"
    assert foreign.read_result(call_ref="c1.1").source.tool_call_id == "call-foreign"


@pytest.mark.parametrize("call_ref", [
    "l1result_" + "0" * 64, "c01.1", "c1.01", "c0.1", "c1.0",
    "c1.1\n", " c1.1", "c1.1 ", "c1.1.1", "c-1.1", "c１.1", "c1.١", None,
    "c9223372036854775808.1", "c999999999999999999999999.1", "c1.17",
])
def test_history_rejects_noncanonical_short_references_before_reading(history, call_ref):
    _, reader, _ = history
    with pytest.raises(ToolHistoryError, match="invalid_history_request"):
        reader.read_result(call_ref=call_ref)


def test_history_refuses_changed_raw_result_instead_of_returning_old_hash(history):
    path, reader, _ = history
    item = reader.list_results().results[0]
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE l1_turn_tool_calls SET outcome_json='{}' WHERE tool_call_id='call-1'"
        )
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.read_result(call_ref=item.source.call_ref)


def test_history_corrupt_identity_is_a_bounded_integrity_error(history):
    path, reader, _ = history
    item = reader.list_results().results[0]
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE l1_turn_tool_calls SET outcome_hash='broken' WHERE tool_call_id='call-1'"
        )
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.read_result(call_ref=item.source.call_ref)


def _append_finding_result(history, *, tool_id, source, claim="保留结论", status="succeeded", arguments_override=None):
    path, _, _ = history
    arguments = _canonical(arguments_override if arguments_override is not None else {
        "items": [{"claim": claim, "source_refs": [source]}],
    })
    outcome = _canonical({"status": status, "result": {
        "status": "applied", "active_projection": {"active_entries": [{"source_refs": [source]}]},
    } if status == "succeeded" else None,
        "error": None if status == "succeeded" else {"code": "invalid_tool_input"}})
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO l1_turn_tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            "call-finding", "run-a", "session-a", "step-a2", 3, tool_id,
            arguments, hashlib.sha256(arguments.encode()).hexdigest(), status,
            outcome, hashlib.sha256(outcome.encode()).hexdigest(),
        ))
    return arguments, outcome


@pytest.mark.parametrize("tool_id", ["record_execution_findings", "revise_execution_finding"])
@pytest.mark.parametrize("legacy", [False, True])
def test_history_finding_arguments_show_short_refs_without_rewriting_raw_records(history, tool_id, legacy):
    path, reader, rows = history
    source = {"tool_result_id": "call-1", "result_sha256": rows[1][-1], "chunk_id": "chunk-a"}
    if legacy:
        source = {"tool_result_id": legacy_l1_tool_result_id(
            tool_call_id="call-1", result_sha256=rows[1][-1]), "chunk_id": "chunk-a"}
    arguments, outcome = _append_finding_result(history, tool_id=tool_id, source=source)
    before = path.read_bytes()
    item = reader.list_results().results[-1]
    projected = json.loads(item.arguments_summary)
    assert projected == {"items": [{"claim": "保留结论", "source_refs": [
        {"call_ref": "c1.1", "chunk_id": "chunk-a"},
    ]}]}
    assert not item.arguments_truncated
    assert item.arguments_sha256 == hashlib.sha256(arguments.encode()).hexdigest()
    assert reader.read_result(call_ref="c2.3").outcome == json.loads(outcome)
    assert path.read_bytes() == before


@pytest.mark.parametrize("source_call_id", ["call-foreign", "call-1"])
def test_history_finding_argument_projection_rejects_foreign_or_changed_result_pins(history, source_call_id):
    _, reader, rows = history
    source = {"tool_result_id": source_call_id,
              "result_sha256": rows[2][-1] if source_call_id == "call-foreign" else "0" * 64}
    _append_finding_result(history, tool_id="record_execution_findings", source=source)
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.list_results()


def test_history_finding_argument_summary_still_obeys_character_budget(history):
    _, reader, rows = history
    _append_finding_result(history, tool_id="record_execution_findings",
                           source={"tool_result_id": "call-1", "result_sha256": rows[1][-1]},
                           claim="长结论" * 300)
    item = reader.list_results().results[-1]
    assert item.arguments_truncated and len(item.arguments_summary) == 512


@pytest.mark.parametrize("arguments", [
    {"items": "not-an-array"},
    {"items": [{"source_refs": "not-an-array"}]},
    {"items": [{"source_refs": [{"tool_result_id": "l1result_" + "f" * 64}]}]},
    {"items": [{"source_refs": [{"tool_result_id": "call-foreign", "result_sha256": "f" * 64}]}]},
])
def test_failed_finding_with_invalid_sources_stays_readable_without_exposing_arguments(history, arguments):
    path, reader, _ = history
    _append_finding_result(history, tool_id="record_execution_findings", source={},
                           status="failed", arguments_override=arguments)
    before = path.read_bytes()
    item = reader.list_results().results[-1]
    assert item.source.status == "failed"
    assert item.arguments_summary == "Arguments omitted for unsuccessful findings call; read its error."
    assert item.arguments_unavailable is True
    assert item.arguments_unavailable_reason == "unsuccessful_findings_call"
    assert item.arguments_truncated is False
    assert reader.read_result(call_ref="c2.3").outcome["error"]["code"] == "invalid_tool_input"
    assert path.read_bytes() == before


def test_failed_finding_argument_omission_does_not_bypass_raw_integrity_check(history):
    path, reader, _ = history
    _append_finding_result(history, tool_id="revise_execution_finding", source={},
                           status="failed", arguments_override={"items": "invalid"})
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE l1_turn_tool_calls SET arguments_json='{}' WHERE tool_call_id='call-finding'")
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.list_results()


def test_history_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    reader = L1ToolHistoryReader(
        database_path=path, session_id="session-a", l1_turn_run_id="run-a"
    )
    with pytest.raises(ToolHistoryError, match="tool_history_unavailable"):
        reader.list_results()
    assert not path.exists()


@pytest.mark.parametrize("operation", ["list", "read"])
def test_history_connection_denial_never_falls_back_or_leaks_database_path(
    history,
    monkeypatch,
    operation,
):
    path, reader, _ = history
    source = reader.list_results().results[0].source
    attempts = []

    def deny_connection(database, *, uri):
        attempts.append((database, uri))
        raise sqlite3.OperationalError(f"access denied: {path}")

    monkeypatch.setattr(sqlite3, "connect", deny_connection)
    with pytest.raises(ToolHistoryError) as error:
        if operation == "list":
            reader.list_results()
        else:
            reader.read_result(call_ref=source.call_ref)
    assert str(error.value) == "tool_history_unavailable"
    assert attempts == [(path.resolve().as_uri() + "?mode=ro", True)]


def test_history_list_count_and_rows_use_one_snapshot(history, monkeypatch):
    path, reader, rows = history
    connect = sqlite3.connect
    with connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    settled = False

    class ConcurrentSettlementConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal settled
            result = super().execute(sql, parameters)
            if sql.startswith("SELECT COUNT(*)") and not settled:
                settled = True
                with connect(path) as writer:
                    writer.execute(
                        "UPDATE l1_turn_tool_calls SET status='succeeded', outcome_json=?, outcome_hash=? "
                        "WHERE tool_call_id='pending'",
                        (rows[1][-2], rows[1][-1]),
                    )
            return result

    def connect_with_concurrent_settlement(database, *, uri):
        return connect(database, uri=uri, factory=ConcurrentSettlementConnection)

    monkeypatch.setattr(sqlite3, "connect", connect_with_concurrent_settlement)
    page = reader.list_results()
    assert settled is True
    assert page.total_results == len(page.results) == 2
    refreshed = reader.list_results()
    assert refreshed.total_results == 3
    assert [item.source.call_ref for item in page.results] == ["c1.1", "c2.1"]
    assert [item.source.call_ref for item in refreshed.results] == ["c1.1", "c2.1", "c2.2"]
    reopened = L1ToolHistoryReader(database_path=path, session_id="session-a", l1_turn_run_id="run-a")
    assert reopened.read_result(call_ref="c2.1").source.tool_call_id == "call-2"
