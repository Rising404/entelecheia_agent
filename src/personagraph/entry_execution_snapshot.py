"""Accepted Turn 的规范执行快照契约。

该冷模块是快照结构、规范 JSON 与持久化认证的唯一所有者。Runtime Entry 负责创建和
消费快照；SQLite persistence 只调用这里的认证函数，不再复制协议解析。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal


EntryFeatureScalar = bool | str | int | float | None
_MAX_SNAPSHOT_UTF8_BYTES = 65_536
_EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "features",
        "file_retrieval_data_version",
        "session_retrieval_data_version",
        "session_retrieval_assistant_turn_cutoff",
        "post_commit_job_kinds",
    }
)


@dataclass(frozen=True, slots=True)
class EntryExecutionSnapshot:
    """一个 Turn 的不可变 Runtime 开关与检索 authority 快照。"""

    feature_items: tuple[tuple[str, EntryFeatureScalar], ...]
    post_commit_job_kinds: tuple[str, ...]
    file_retrieval_data_version: str | None = None
    session_retrieval_data_version: str | None = None
    session_retrieval_assistant_turn_cutoff: int | None = None
    schema_version: Literal[2] = 2

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("entry execution snapshot schema_version must be 2")
        keys = tuple(key for key, _value in self.feature_items)
        if (
            keys != tuple(sorted(keys))
            or len(keys) != len(set(keys))
            or len(keys) > 256
        ):
            raise ValueError("entry execution snapshot feature keys must be canonical")
        for key, value in self.feature_items:
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("entry execution snapshot feature key is invalid")
            if not isinstance(value, (bool, str, int, float, type(None))):
                raise ValueError("entry execution snapshot features must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("entry execution snapshot feature values must be finite")
        _validate_generation(
            self.file_retrieval_data_version,
            label="file retrieval generation",
        )
        _validate_generation(
            self.session_retrieval_data_version,
            label="Session retrieval generation",
        )
        cutoff = self.session_retrieval_assistant_turn_cutoff
        if cutoff is not None and (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < 0
            or cutoff >= 10**18
        ):
            raise ValueError("Session retrieval cutoff must be a supported integer")
        if cutoff is not None and self.session_retrieval_data_version is None:
            raise ValueError("Session retrieval cutoff requires a frozen generation")
        jobs = self.post_commit_job_kinds
        if (
            jobs != tuple(sorted(jobs))
            or len(jobs) != len(set(jobs))
            or any(
                not isinstance(item, str) or not item or len(item) > 96
                for item in jobs
            )
        ):
            raise ValueError("post-commit job kinds must be canonical")
        if len(self.to_json().encode("utf-8")) > _MAX_SNAPSHOT_UTF8_BYTES:
            raise ValueError("entry execution snapshot exceeds its byte limit")

    @classmethod
    def create(
        cls,
        *,
        features: Mapping[str, EntryFeatureScalar],
        post_commit_job_kinds: tuple[str, ...],
        file_retrieval_data_version: str | None = None,
        session_retrieval_data_version: str | None = None,
        session_retrieval_assistant_turn_cutoff: int | None = None,
    ) -> "EntryExecutionSnapshot":
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping")
        if any(not isinstance(key, str) for key in features):
            raise ValueError("entry execution snapshot feature keys must be strings")
        return cls(
            feature_items=tuple(sorted(features.items(), key=lambda item: item[0])),
            post_commit_job_kinds=tuple(sorted(post_commit_job_kinds)),
            file_retrieval_data_version=file_retrieval_data_version,
            session_retrieval_data_version=session_retrieval_data_version,
            session_retrieval_assistant_turn_cutoff=(
                session_retrieval_assistant_turn_cutoff
            ),
        )

    @property
    def features(self) -> dict[str, EntryFeatureScalar]:
        return dict(self.feature_items)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "features": dict(self.feature_items),
                "file_retrieval_data_version": self.file_retrieval_data_version,
                "session_retrieval_data_version": (
                    self.session_retrieval_data_version
                ),
                "session_retrieval_assistant_turn_cutoff": (
                    self.session_retrieval_assistant_turn_cutoff
                ),
                "post_commit_job_kinds": list(self.post_commit_job_kinds),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_sha256: str,
    ) -> "EntryExecutionSnapshot":
        if (
            not isinstance(payload, str)
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError("entry execution snapshot persistence is incomplete")
        if len(payload.encode("utf-8")) > _MAX_SNAPSHOT_UTF8_BYTES:
            raise ValueError("entry execution snapshot exceeds its byte limit")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != expected_sha256:
            raise ValueError("entry execution snapshot hash is invalid")
        try:
            raw = json.loads(payload)
            if (
                not isinstance(raw, dict)
                or raw.get("schema_version") != 2
                or set(raw) != _EXPECTED_KEYS
            ):
                raise ValueError
            features = raw["features"]
            jobs = raw["post_commit_job_kinds"]
            if not isinstance(features, dict) or not isinstance(jobs, list):
                raise ValueError
            snapshot = cls.create(
                features=features,
                post_commit_job_kinds=tuple(jobs),
                file_retrieval_data_version=raw["file_retrieval_data_version"],
                session_retrieval_data_version=raw[
                    "session_retrieval_data_version"
                ],
                session_retrieval_assistant_turn_cutoff=raw[
                    "session_retrieval_assistant_turn_cutoff"
                ],
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("entry execution snapshot is invalid") from exc
        if snapshot.to_json() != payload:
            raise ValueError("entry execution snapshot is not canonical")
        return snapshot


def authenticate_entry_execution_snapshot_payload(
    *,
    snapshot_json: str | None,
    snapshot_sha256: str | None,
) -> tuple[str, str] | None:
    """认证可选、原子提供的持久化快照正文与摘要。"""

    if snapshot_json is None and snapshot_sha256 is None:
        return None
    if not isinstance(snapshot_json, str) or not isinstance(snapshot_sha256, str):
        raise ValueError("Turn execution snapshot must be supplied atomically")
    EntryExecutionSnapshot.from_json(
        snapshot_json,
        expected_sha256=snapshot_sha256,
    )
    return snapshot_json, snapshot_sha256


def _validate_generation(value: str | None, *, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value or len(value) > 512
    ):
        raise ValueError(f"{label} must be a bounded identity")


__all__ = [
    "EntryExecutionSnapshot",
    "EntryFeatureScalar",
    "authenticate_entry_execution_snapshot_payload",
]
