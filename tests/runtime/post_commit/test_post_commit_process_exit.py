"""实际子进程 + SQLite + 本地 lexical 索引，验证回答后的显式收尾边界。"""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest


@pytest.mark.parametrize("wait_for_settlement", [False, True])
def test_process_exit_preserves_complete_index_only_after_explicit_wait(tmp_path, wait_for_settlement):
    project_root = Path(__file__).resolve().parents[3]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = r'''
import json,sys
from threading import Event
from personagraph.session import store
from personagraph.runtime.post_commit import scheduler
from tests.helpers.session_records import complete_test_turn_execution, completed_test_turn_id

sid=store.create_session("Entelecheia",working_dir=sys.argv[1])
with store.session_database_scope(sid):
    completed=complete_test_turn_execution(sid,1,post_commit_job_kinds=("session_summary","session_retrieval_index"))
tid=completed_test_turn_id(completed)
started=Event()
original=scheduler._process_due_turn_post_commit_jobs
def delayed(**kwargs):
    started.set()
    Event().wait(0.2)
    return original(**kwargs)
scheduler._process_due_turn_post_commit_jobs=delayed
scheduler.POST_COMMIT_POLL_SECONDS=0.01
scheduler.schedule_turn_post_commit_jobs(session_id=sid,store=store)
assert started.wait(2)
result={"session_id":sid,"turn_id":tid}
if sys.argv[2]=="wait":
    result["settlement"]=scheduler.wait_for_turn_post_commit_jobs(session_id=sid,turn_id=tid,store=store,timeout_seconds=10).to_dict()
    assert scheduler.stop_turn_post_commit_workers(store=store,timeout_seconds=2)
print(json.dumps(result),flush=True)
'''
    environment = dict(os.environ) | {
        "PERSONAGRAPH_STATE_DIR": str(tmp_path / "state"),
        "PERSONAGRAPH_LOCAL_CONFIG_DIR": str(tmp_path / "config"),
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(tmp_path / "default-projects"),
        "PERSONAGRAPH_MODEL_PROVIDER": "mock",
        "PERSONAGRAPH_RETRIEVAL_PROFILE": "lexical",
        "PERSONAGRAPH_RETRIEVAL_RERANKER": "off",
        "PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY": "true",
        "PYTHONPATH": str(project_root / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(workspace), "wait" if wait_for_settlement else "exit"],
        cwd=project_root, env=environment, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    database = tmp_path / "state" / "sessions" / payload["session_id"] / "session.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        statuses = connection.execute("SELECT job_kind,status FROM turn_post_commit_jobs").fetchall()
        window = connection.execute("SELECT window_state FROM turn_execution_windows").fetchone()[0]
    if not wait_for_settlement:
        assert any(status != "applied" for _, status in statuses)
        assert window == "post_commit_pending"
        return
    assert payload["settlement"]["status"] == "settled"
    assert payload["settlement"]["window_released"] is True
    assert statuses and all(status == "applied" for _, status in statuses)
    assert window == "empty"
    index = next((tmp_path / "state" / "projects").glob("*/session_retrieval.sqlite"))
    with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("SELECT count(*) FROM retrieval_units WHERE index_state='ready'").fetchone()[0] > 0
        assert connection.execute("SELECT count(*) FROM retrieval_data_versions WHERE role='active' AND state='ready'").fetchone()[0] == 1
