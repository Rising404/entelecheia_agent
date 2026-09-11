"""当前 Session 检索索引的持久 post-commit 应用服务。

本模块只处理一个已提交 Turn 对的派生索引、稳定失败投影，以及新 Turn 准入前的
effect-proof reconciliation。Runtime 仍持有 job 租约、kind 路由和 Turn Window
生命周期；Session Store 仍持有原子 job 状态转换。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import SessionRetrievalComposition, SessionTurnRetrievalBinding


SESSION_RETRIEVAL_INDEX_JOB_KIND = "session_retrieval_index"
_SESSION_RETRIEVAL_RETRY_SECONDS = 1

SessionRetrievalIndexer = Callable[[Mapping[str, object]], str]


class SessionRetrievalPostCommitStore(Protocol):
    """Session 检索 post-commit 服务所需的最小持久边界。"""

    def get_session(self, session_id: str) -> Mapping[str, object] | None: ...

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> Mapping[str, object] | None: ...

    def get_committed_turn_pair_for_post_commit(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> dict[str, object] | None: ...

    def mark_turn_post_commit_job_failed(
        self,
        *,
        job_id: str,
        worker_id: str,
        reason_code: str,
        retry_after_seconds: int | None,
    ) -> dict[str, object]: ...

    def mark_turn_post_commit_job_applied(
        self,
        *,
        job_id: str,
        worker_id: str,
    ) -> dict[str, object]: ...

    def reconcile_turn_post_commit_job_applied(
        self,
        *,
        session_id: str,
        turn_id: str,
        job_id: str,
        expected_job_kind: str,
    ) -> dict[str, object]: ...

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...
    def list_committed_turn_pairs(
        self,
        session_id: str,
        *,
        limit: int | None = ...,
    ) -> Sequence[Mapping[str, object]]: ...


def _build_default_composition(
    *,
    store: SessionRetrievalPostCommitStore,
) -> SessionRetrievalComposition:
    from .composition import build_session_retrieval_composition

    return build_session_retrieval_composition(store=store)


def _ensure_pair_ready(
    composition: SessionRetrievalComposition,
    *,
    pair: Mapping[str, object],
) -> str:
    from .lifecycle import ensure_committed_session_pair_retrieval_ready

    return ensure_committed_session_pair_retrieval_ready(
        composition,
        pair=pair,
    )


def _ensure_session_ready(
    composition: SessionRetrievalComposition,
    *,
    session_id: str,
    assistant_turn_cutoff: int,
) -> str:
    from .lifecycle import ensure_session_retrieval_ready

    return ensure_session_retrieval_ready(
        composition,
        session_id=session_id,
        assistant_turn_cutoff=assistant_turn_cutoff,
    )


def process_session_retrieval_index_job(
    *,
    session_id: str,
    job: Mapping[str, object],
    worker_id: str,
    store: SessionRetrievalPostCommitStore,
    index_pair: SessionRetrievalIndexer | None,
) -> bool:
    """索引一个精确已提交 Turn 对，并结算对应持久 job。"""

    job_id = _required_job_id(job)
    turn_id = _required_text(job, "turn_id")
    pair = store.get_committed_turn_pair_for_post_commit(
        session_id=session_id,
        turn_id=turn_id,
    )
    if not isinstance(pair, Mapping):
        store.mark_turn_post_commit_job_failed(
            job_id=job_id,
            worker_id=worker_id,
            reason_code="SESSION_RETRIEVAL_COMMITTED_PAIR_MISSING",
            retry_after_seconds=None,
        )
        return False
    try:
        if index_pair is None:
            composition = _build_default_composition(store=store)
            _ensure_pair_ready(
                composition,
                pair=pair,
            )
        else:
            index_pair(pair)
        store.mark_turn_post_commit_job_applied(
            job_id=job_id,
            worker_id=worker_id,
        )
        return True
    except Exception as exc:
        store.mark_turn_post_commit_job_failed(
            job_id=job_id,
            worker_id=worker_id,
            reason_code=_session_retrieval_failure_code(exc),
            retry_after_seconds=(
                None
                if _session_retrieval_failure_is_terminal(exc)
                else _SESSION_RETRIEVAL_RETRY_SECONDS
            ),
        )
        return False


def index_committed_session_pair(
    binding: 'SessionTurnRetrievalBinding',
    pair: Mapping[str, object],
) -> str:
    """使用已接受 Turn 冻结的 composition 索引一个精确 committed pair。"""

    return _ensure_pair_ready(
        binding.composition,
        pair=pair,
    )


def reconcile_session_retrieval_effects(
    *,
    session_id: str,
    store: SessionRetrievalPostCommitStore,
    binding: 'SessionTurnRetrievalBinding',
) -> None:
    """证明最新 Session 索引覆盖，并结算已完成但未落账的索引 job。"""

    composition = binding.composition
    pairs = store.list_committed_turn_pairs(session_id, limit=1)
    if pairs:
        latest = pairs[-1]
        cutoff = latest.get("assistant_turn_idx")
        if cutoff != binding.assistant_turn_cutoff:
            raise RuntimeError("session retrieval cutoff changed during pre-turn recovery")
        # 长回复可拆成多个单元，轮次数相等不代表每一块都存在。检查冻结前缀中的精确单元，
        # 既有健康单元复用 sync 回执，只有缺失/损坏单元重新编码。
        _ensure_session_ready(
            composition,
            session_id=session_id,
            assistant_turn_cutoff=int(cutoff),
        )

    inspection = store.inspect_turn_execution(session_id)
    jobs = inspection.get("post_commit_jobs")
    if not isinstance(jobs, list):
        return
    for job in jobs:
        if (
            not isinstance(job, dict)
            or job.get("job_kind") != SESSION_RETRIEVAL_INDEX_JOB_KIND
            or job.get("status") in {"applied", "waived"}
        ):
            continue
        turn_id = _required_text(job, "turn_id")
        pair = store.get_committed_turn_pair_for_post_commit(
            session_id=session_id,
            turn_id=turn_id,
        )
        if not isinstance(pair, Mapping):
            continue
        index_committed_session_pair(binding, pair)
        store.reconcile_turn_post_commit_job_applied(
            session_id=session_id,
            turn_id=turn_id,
            job_id=_required_job_id(job),
            expected_job_kind=SESSION_RETRIEVAL_INDEX_JOB_KIND,
        )


def _session_retrieval_failure_code(exc: Exception) -> str:
    """把内部 Retrieval 异常投影为稳定、无敏感细节的 job reason。"""

    message = str(exc)
    if "bound project documents context" in message:
        return "SESSION_RETRIEVAL_PROJECT_UNBOUND"
    if "identity_collision" in message or "rebuild_requires" in message:
        return "SESSION_RETRIEVAL_GENERATION_CONFLICT"
    if "projection" in message or "committed" in message:
        return "SESSION_RETRIEVAL_SOURCE_INVALID"
    if "failed" in message or "unavailable" in message:
        return "SESSION_RETRIEVAL_INDEX_UNAVAILABLE"
    return "SESSION_RETRIEVAL_INTERNAL_FAILURE"


def _session_retrieval_failure_is_terminal(exc: Exception) -> bool:
    return _session_retrieval_failure_code(exc) in {
        "SESSION_RETRIEVAL_PROJECT_UNBOUND",
        "SESSION_RETRIEVAL_GENERATION_CONFLICT",
        "SESSION_RETRIEVAL_SOURCE_INVALID",
    }


def _required_job_id(job: Mapping[str, object]) -> str:
    return _required_text(job, "job_id")


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"post-commit job is missing {key}")
    return value


__all__ = [
    "SESSION_RETRIEVAL_INDEX_JOB_KIND",
    "SessionRetrievalIndexer",
    "SessionRetrievalPostCommitStore",
    "index_committed_session_pair",
    "process_session_retrieval_index_job",
    "reconcile_session_retrieval_effects",
]
