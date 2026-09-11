"""共享的轻量 token 估算与摘要文本上限工具。

默认估算按 CJK 约 1 token/字、其他字符约 4 字/token 计算。真正发送给
provider 的 envelope 由 :mod:`personagraph.context_budget` 统一测量和准入。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Protocol


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


BLOCK_DEFAULTS: dict[str, int] = {
    "summary": 600,
}


def task_budget() -> int:
    """任务工作集（PDF/代码等检索/摘要后片段）独立上限。env: PERSONAGRAPH_TASK_BUDGET。"""
    return _int_env("PERSONAGRAPH_TASK_BUDGET", 12000)


def block_cap(name: str) -> int:
    """某块的上限；env 覆盖 PERSONAGRAPH_BUDGET_<NAME>。"""
    return _int_env(f"PERSONAGRAPH_BUDGET_{name.upper()}", BLOCK_DEFAULTS.get(name, 600))


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


class TokenCounter(Protocol):
    name: str
    kind: str
    requested: str
    model: str | None
    fallback_reason: str | None

    def count_text(self, text: str) -> int:
        ...


@dataclass
class HeuristicTokenCounter:
    requested: str = "heuristic"
    model: str | None = None
    fallback_reason: str | None = None
    name: str = "heuristic"
    kind: str = "heuristic"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cjk = sum(1 for ch in text if _is_cjk(ch))
        other = len(text) - cjk
        return int(cjk + other / 4)


@dataclass
class TiktokenCounter:
    encoder: object
    requested: str = "tiktoken"
    model: str | None = None
    fallback_reason: str | None = None
    name: str = "tiktoken"
    kind: str = "tokenizer"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))  # type: ignore[attr-defined]


@dataclass
class HuggingFaceTokenCounter:
    tokenizer: object
    requested: str = "hf"
    model: str | None = None
    fallback_reason: str | None = None
    name: str = "hf"
    kind: str = "tokenizer"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]


def _fallback(requested: str, model: str | None, reason: str) -> HeuristicTokenCounter:
    return HeuristicTokenCounter(requested=requested, model=model, fallback_reason=reason)


@lru_cache(maxsize=16)
def _make_token_counter(requested: str, model: str | None) -> TokenCounter:
    mode = (requested or "heuristic").strip().lower()
    if mode in {"", "heuristic", "estimate"}:
        return HeuristicTokenCounter(requested=mode or "heuristic", model=model)
    if mode in {"tiktoken", "openai"}:
        try:
            # 可选 tokenizer 仅在显式选择时解析；核心包 cold import 不加载第三方栈。
            tiktoken = import_module("tiktoken")
        except Exception as exc:  # pragma: no cover - 依赖可选包
            return _fallback(mode, model, f"tiktoken_unavailable:{type(exc).__name__}")
        try:
            encoder = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover - 模型表随安装版本而异
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as inner_exc:
                return _fallback(mode, model, f"tiktoken_encoder_unavailable:{type(inner_exc).__name__}")
            return TiktokenCounter(
                encoder=encoder,
                requested=mode,
                model=model,
                fallback_reason=f"model_encoding_fallback:{type(exc).__name__}",
            )
        return TiktokenCounter(encoder=encoder, requested=mode, model=model)
    if mode in {"hf", "huggingface", "transformers"}:
        if not model:
            return _fallback(mode, model, "hf_model_required")
        try:
            # 与 tiktoken 相同：保持可选依赖和 cold-import 边界。
            AutoTokenizer = import_module("transformers").AutoTokenizer
        except Exception as exc:  # pragma: no cover - 依赖可选包
            return _fallback(mode, model, f"transformers_unavailable:{type(exc).__name__}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # pragma: no cover - 取决于本地环境、网络和模型
            return _fallback(mode, model, f"hf_tokenizer_unavailable:{type(exc).__name__}")
        return HuggingFaceTokenCounter(tokenizer=tokenizer, requested=mode, model=model)
    return _fallback(mode, model, "unknown_token_counter")


def get_token_counter() -> TokenCounter:
    requested = os.getenv("PERSONAGRAPH_TOKEN_COUNTER", "heuristic")
    model = os.getenv("PERSONAGRAPH_TOKENIZER_MODEL") or os.getenv("PERSONAGRAPH_MODEL")
    return _make_token_counter(requested, model)


def estimate_tokens(text: str) -> int:
    """调用前 token 预算估算；具体计数器由 PERSONAGRAPH_TOKEN_COUNTER 控制。"""
    return get_token_counter().count_text(text)


_TRUNC_MARK = " …[截断]"


def cap_text(text: str, max_tokens: int) -> str:
    """把文本按估算 token 截断到 max_tokens 以内（含截断标记，结果保证 ≤ max_tokens）。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # 预留截断标记的 token 额度，保证最终结果不超 max_tokens
    budget = max(0, max_tokens - estimate_tokens(_TRUNC_MARK))
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip()
        if estimate_tokens(candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best + _TRUNC_MARK
