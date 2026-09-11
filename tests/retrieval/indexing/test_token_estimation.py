from __future__ import annotations

from dataclasses import dataclass

from personagraph.retrieval.indexing.token_estimation import (
    BgeM3RetrievalTokenEstimator,
)


@dataclass
class TokenIdEncoder:
    token_ids_by_text: dict[str, tuple[int, ...]]
    calls: int = 0
    fail: bool = False

    def token_ids(self, text: str):
        self.calls += 1
        if self.fail:
            raise RuntimeError("tokenizer_unavailable")
        return self.token_ids_by_text[text]


def test_bge_retrieval_token_estimator_uses_token_ids_and_keeps_only_content_free_cache_keys():
    text = "中文 retrieval token 计数"
    encoder = TokenIdEncoder({text: (1, 2, 3, 4)})
    estimator = BgeM3RetrievalTokenEstimator(encoder)

    assert estimator(text) == 4
    assert estimator(text) == 4
    assert encoder.calls == 1
    assert list(estimator._cache) != [text]
    assert estimator.diagnostic_snapshot().fallback_count == 0
    assert estimator.diagnostic_snapshot().cache_entries == 1


def test_bge_retrieval_token_estimator_falls_back_conservatively_when_tokenizer_fails():
    text = "中文"
    encoder = TokenIdEncoder({}, fail=True)
    estimator = BgeM3RetrievalTokenEstimator(encoder)

    assert estimator(text) == len(text.encode("utf-8"))
    assert estimator.diagnostic_snapshot().fallback_count == 1
    # 结果会被缓存，因此同一进程生命周期内重复打包时，
    # 不会反复调用已知不可用的分词器。
    assert estimator(text) == len(text.encode("utf-8"))
    assert encoder.calls == 1


def test_bge_retrieval_token_estimator_rejects_an_invalid_content_free_cache_limit():
    encoder = TokenIdEncoder({})

    try:
        BgeM3RetrievalTokenEstimator(encoder, max_cache_entries=0)
    except ValueError as exc:
        assert str(exc) == "max_cache_entries must be greater than zero"
    else:
        raise AssertionError("expected validation error")
