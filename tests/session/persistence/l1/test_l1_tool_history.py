"""当前 L1 执行的只读历史结果：范围、分页与原始证据身份。"""

import hashlib
import json
import sqlite3

import pytest

from personagraph.persistent_turn_content.evidence import l1_tool_result_id
from personagraph.persistent_turn_content.tool_results import ToolHistoryError
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
                                             ('step-b1', 'run-b', 1);
        """)
        rows = []
        for call_id, step, ordinal, run, status in (
            ("call-2", "step-a2", 1, "run-a", "failed"),
            ("call-1", "step-a1", 1, "run-a", "succeeded"),
            ("call-foreign", "step-b1", 1, "run-b", "succeeded"),
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
    assert first.results[0].arguments_truncated is True
    assert len(first.results[0].arguments_summary) == 512
    second = reader.list_results(offset=first.next_offset, limit=1)
    assert second.results[0].source.status == "failed"
    assert second.results[0].source.tool_call_id == "call-2"
    assert second.next_offset is None
    assert path.read_bytes() == before


def test_history_reader_verifies_complete_outcome_without_rewriting_it(history):
    path, reader, rows = history
    before = path.read_bytes()
    original = rows[1][-2]
    item = reader.list_results().results[0]
    record = reader.read_result(tool_result_id=item.source.tool_result_id)
    assert record.source == item.source
    assert record.outcome == json.loads(original)
    assert hashlib.sha256(_canonical(record.outcome).encode()).hexdigest() == item.source.result_sha256
    assert path.read_bytes() == before


def test_history_errors_are_readable_but_keep_the_original_error_status(history):
    _, reader, _ = history
    item = reader.list_results().results[1]
    page = reader.read_result(tool_result_id=item.source.tool_result_id)
    assert page.source.status == "failed"
    assert page.outcome["error"]["code"] == "file_unavailable"


def test_history_rejects_unknown_foreign_and_wrong_session_scope(history):
    path, reader, rows = history
    for result_id in (
        "l1result_" + "0" * 64,
        l1_tool_result_id(tool_call_id="call-foreign", result_sha256=rows[2][-1]),
    ):
        with pytest.raises(ToolHistoryError, match="tool_result_unavailable"):
            reader.read_result(tool_result_id=result_id)
    other = L1ToolHistoryReader(
        database_path=path, session_id="session-b", l1_turn_run_id="run-a"
    )
    with pytest.raises(ToolHistoryError, match="tool_history_scope_unavailable"):
        other.list_results()


def test_history_refuses_changed_raw_result_instead_of_returning_old_hash(history):
    path, reader, _ = history
    item = reader.list_results().results[0]
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE l1_turn_tool_calls SET outcome_json='{}' WHERE tool_call_id='call-1'"
        )
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.read_result(tool_result_id=item.source.tool_result_id)


def test_history_corrupt_identity_is_a_bounded_integrity_error(history):
    path, reader, _ = history
    item = reader.list_results().results[0]
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE l1_turn_tool_calls SET outcome_hash='broken' WHERE tool_call_id='call-1'"
        )
    with pytest.raises(ToolHistoryError, match="tool_result_integrity_invalid"):
        reader.read_result(tool_result_id=item.source.tool_result_id)


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
            reader.read_result(tool_result_id=source.tool_result_id)
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
    assert reader.list_results().total_results == 3
