"""检索本地 token 计数，与最终 Prompt 计数分离。

RAG 在 Prompt Builder 存在前就会打包 Source 内容，因此它使用所配置 generation 编码器
的稳定 token ID 计算自身条目预算。本模块并不声称能计算目标生成模型的 Prompt token：
外层 Context/Prompt 层必须始终执行最终物理检查。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading

from ..execution import checkpoint
from ..ports import BgeM3EncoderPort, RetrievalCancelled


@dataclass(frozen=True, slots=True)
class RetrievalTokenEstimatorSnapshot:
    """可纳入 Foundation 诊断的不含内容状态。"""

    kind: str
    fallback_count: int
    cache_entries: int


class BgeM3RetrievalTokenEstimator:
    """使用已配置编码器 token ID 计算 RAG 条目，并保守失败。

    为保持 API 兼容，保留历史类名。有界缓存只保存 SHA-256 摘要和计数，绝不保存 Source
    正文。若 tokenizer 访问暂时失败，UTF-8 字节长度会作为刻意保守的预算估算；它可能
    遗漏可用条目，但不会通过假装条目很小而让 RAG 打包更多内容。
    """

    def __init__(
        self,
        encoder: BgeM3EncoderPort,
        *,
        max_cache_entries: int = 1024,
    ) -> None:
        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be greater than zero")
        self._encoder = encoder
        self._max_cache_entries = max_cache_entries
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._fallback_count = 0
        self._lock = threading.Lock()

    def __call__(self, text: str) -> int:
        checkpoint()
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._cache.get(digest)
            if cached is not None:
                self._cache.move_to_end(digest)
                return cached
        try:
            count = max(1, len(tuple(self._encoder.token_ids(text))))
            checkpoint()
        except RetrievalCancelled:
            raise
        except Exception:
        # UTF-8 字节数会刻意高估常见中文和拉丁文本的分词数量。
        # 在最终提示词计量前，因高估而舍弃内容比悄然低估检索项更安全。
            count = max(1, len(text.encode("utf-8")))
            with self._lock:
                self._fallback_count += 1
        with self._lock:
            self._cache[digest] = count
            self._cache.move_to_end(digest)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        return count

    def diagnostic_snapshot(self) -> RetrievalTokenEstimatorSnapshot:
        """只返回计数器；分词后的文本和缓存键保持私有。"""

        with self._lock:
            return RetrievalTokenEstimatorSnapshot(
                kind="retrieval_encoder_token_ids",
                fallback_count=self._fallback_count,
                cache_entries=len(self._cache),
            )
