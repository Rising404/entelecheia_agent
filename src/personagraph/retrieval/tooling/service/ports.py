"""Retrieval Tool service 组合所需的窄 ports。"""

from __future__ import annotations

from typing import Protocol

from ..contracts import FileRetrievalReadinessResult
from ...contracts import CorpusKey
from ...ports import RetrievalDataVersionProvider
from ...service import RetrievalService
from ..contracts import FrozenFileRetrievalBinding, RetrievalToolRequest


class RetrievalFoundationReadPort(Protocol):
    """工具数据平面所需的 Retrieval foundation 子集。"""

    corpus_key: CorpusKey
    service: RetrievalService
    data_version_provider: RetrievalDataVersionProvider


class FileRetrievalReadinessPort(Protocol):
    """把冻结文件绑定解析成不泄露私有路径的 readiness receipt。

    实现必须把 ``authority_id`` 绑定到规范 Host 文件，并用冻结绑定中的修订或内容
    指纹执行 readiness 检查。``READY`` 因而是精确权威证明，而不只是同名文件存在。
    """

    def ensure_ready(
        self,
        request: RetrievalToolRequest,
        binding: FrozenFileRetrievalBinding,
    ) -> FileRetrievalReadinessResult: ...


__all__ = [
    "FileRetrievalReadinessPort",
    "RetrievalFoundationReadPort",
]
