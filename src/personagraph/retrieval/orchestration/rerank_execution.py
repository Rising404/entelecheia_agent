"""在供应商原有小批 forward 边界合作式停止，不重写其分词或评分算法。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from ..compute.inference import synchronize_device
from ..execution import RetrievalExecution, current_execution


class _CancellableTokenizer:
    """原 tokenizer 的透明调用桥；不复制、修改其分词与 padding 算法。"""

    def __init__(self, tokenizer: Any, execution: RetrievalExecution) -> None:
        self._tokenizer = tokenizer
        self._execution = execution

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def _run(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._execution.check_rerank_budget()
        with self._execution.measure("reranker_tokenize"):
            result = operation(*args, **kwargs)
        self._execution.checkpoint()
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(self._tokenizer, *args, **kwargs)

    def prepare_for_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(self._tokenizer.prepare_for_model, *args, **kwargs)

    def pad(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(self._tokenizer.pad, *args, **kwargs)


@contextmanager
def observe_reranker_batches(model: Any, *, device: str = "cpu") -> Iterator[None]:
    """只在持有共享推理资源时安装短期 Torch hooks，并保证异常时也移除。

    已开始的冷加载、单个 tokenizer 调用和 native forward 无法强杀。tokenizer 桥及
    pre/post hook 保证撤销后不进入下一次调用；原适配器仍负责排序、批量及分数还原。
    """

    execution = current_execution()
    if execution is None:
        yield
        return
    module = getattr(model, "model", None)
    if not callable(getattr(module, "register_forward_pre_hook", None)) or not callable(
        getattr(module, "register_forward_hook", None)
    ):
        raise ValueError("reranker_forward_hooks_unavailable")
    preparing_at = execution.clock()
    forward_at: float | None = None
    first_forward = True
    slowest_forward_s = 0.0

    def before(_module: Any, _args: tuple, kwargs: Mapping[str, Any]) -> None:
        nonlocal first_forward, forward_at
        # 首批耗时未知；后续批次按本次已观察到的最慢 forward 留出时间，不能把
        # “尚余 5 秒”误当成足够启动一个刚测得需要 8 秒的不可中断批次。
        execution.check_rerank_budget(expected_compute_s=slowest_forward_s)
        if first_forward:
            execution.record_duration(
                "reranker_tokenize_prepare", execution.clock() - preparing_at
            )
            first_forward = False
        input_ids = kwargs.get("input_ids")
        shape = getattr(input_ids, "shape", ())
        if len(shape) == 2:
            rows, width = int(shape[0]), int(shape[1])
            execution.increment("reranker_forward_pairs", rows)
            execution.increment("reranker_padded_tokens", rows * width)
            execution.set_metric("reranker_last_padded_sequence_length", width)
        execution.increment("reranker_forward_batches")
        forward_at = execution.clock()

    def after(
        _module: Any, _args: tuple, _kwargs: Mapping[str, Any], _output: Any
    ) -> None:
        nonlocal forward_at, slowest_forward_s
        if forward_at is not None:
            # GPU forward 返回时内核可能尚未完成；耗时和后续预算必须包含真实执行。
            synchronize_device(device)
            elapsed = execution.clock() - forward_at
            execution.record_duration("reranker_score", elapsed)
            slowest_forward_s = max(slowest_forward_s, elapsed)
            execution.set_metric(
                "reranker_max_observed_batch_ms",
                max(0, round(slowest_forward_s * 1000)),
            )
            forward_at = None
        execution.checkpoint()

    handles = []
    original_tokenizer = getattr(model, "tokenizer", None)
    try:
        if original_tokenizer is not None:
            model.tokenizer = _CancellableTokenizer(original_tokenizer, execution)
        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after, with_kwargs=True))
        yield
        execution.checkpoint()
    finally:
        if first_forward:
            execution.record_duration(
                "reranker_tokenize_prepare", execution.clock() - preparing_at
            )
            execution.set_metric("reranker_prepare_interrupted", True)
        # forward 异常时普通 post hook 不运行，仍保留实际耗时，不伪记为完成批次。
        if forward_at is not None:
            execution.record_duration(
                "reranker_score_interrupted", execution.clock() - forward_at
            )
        for handle in reversed(handles):
            handle.remove()
        if original_tokenizer is not None:
            model.tokenizer = original_tokenizer
