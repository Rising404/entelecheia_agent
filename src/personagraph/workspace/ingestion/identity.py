"""共享处理身份与各 Session 请求幂等身份；不含模型资源句柄。"""

from __future__ import annotations

import hashlib
import json


def derive_file_preparation_operation_id(
    *,
    file_id: str,
    file_version_id: str,
    canonical_path: str,
    source_sha256: str,
    source_size: int,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
    target_generation_id: str,
    target_generation_fingerprint: str,
) -> str:
    material = dict(
        file_id=file_id,
        file_version_id=file_version_id,
        canonical_path=canonical_path,
        source_sha256=source_sha256,
        source_size=source_size,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        target_generation_id=target_generation_id,
        target_generation_fingerprint=target_generation_fingerprint,
    )
    return "file-processing:" + _digest(material)


def derive_file_preparation_request_id(
    *,
    session_id: str,
    job_id: str,
    source_mtime_ns: int,
    with_summary: bool,
) -> str:
    return "file-request:" + _digest(dict(
        session_id=session_id,
        job_id=job_id,
        source_mtime_ns=source_mtime_ns,
        with_summary=with_summary,
    ))


def _digest(material: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
