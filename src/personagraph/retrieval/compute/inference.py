"""本地模型的 GPU 错误边界、完成同步和释放；不决定调用是否允许回退。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import gc
from typing import Any


def classify_device_error(error: BaseException, device: str) -> str | None:
    """只分类可识别的设备故障；未知 RuntimeError 和数据错误不能借机换 CPU。"""
    if device == "cpu":
        return None
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (RuntimeError, NotImplementedError)):
            message = str(current).lower()
            name = type(current).__name__.lower()
            mentions_device = any(word in message for word in ("cuda", "mps", "gpu"))
            if "outofmemoryerror" in name or (
                mentions_device and "out of memory" in message
            ):
                return "device_out_of_memory"
            if mentions_device and any(
                token in message for token in (
                    "not implemented", "not supported", "unsupported", "no kernel image",
                    "not compiled with", "not linked with support",
                )
            ):
                return "device_unsupported"
            if mentions_device and any(
                token in message for token in (
                    "not available", "unavailable", "invalid device ordinal",
                    "no cuda gpus are available", "driver version is insufficient",
                    "found no nvidia driver",
                )
            ):
                return "device_unavailable"
        current = current.__cause__
    return None


class InferenceRuntimeError(Exception):
    """仅跨越供应商的 RuntimeError 缩批循环，离开边界后恢复原异常。"""

    def __init__(self, original: RuntimeError) -> None:
        super().__init__("local_inference_runtime_error")
        self.original = original


@contextmanager
def guard_device_errors(model: Any, device: str) -> Iterator[None]:
    """调用方须持有设备锁；临时方法替换在成功、取消和错误时都还原。

    FlagEmbedding 1.4 的首批试跑捕获任意 RuntimeError 后不断缩批，甚至缩到零。
    在最窄的模型边界把它转换为非 RuntimeError，供应商退出后恢复原错；不复制其算法。
    """
    module = getattr(model, "model", None)
    if device == "cpu" or module is None:
        yield
        return
    replacements: list[tuple[str, bool, Any]] = []
    tokenizer = getattr(model, "tokenizer", None)
    try:
        if tokenizer is not None:
            model.tokenizer = _DeviceTokenizer(tokenizer)
        # _call_impl 同时覆盖 Torch 的 post-forward hooks；GPU 完成同步错误也必须
        # 越过供应商缩批循环。forward 保留以支持不经过 Torch __call__ 的直接调用。
        for name in ("_call_impl", "forward", "to", "half"):
            original = getattr(module, name, None)
            if not callable(original):
                continue
            local = vars(module)
            replacements.append((name, name in local, local.get(name)))
            setattr(module, name, _escape_runtime_error(original))
        try:
            yield
        except InferenceRuntimeError as exc:
            raise exc.original from None
    finally:
        if tokenizer is not None:
            model.tokenizer = tokenizer
        for name, had_override, previous in reversed(replacements):
            if had_override:
                setattr(module, name, previous)
            else:
                delattr(module, name)


def _escape_runtime_error(method: Any) -> Any:
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except RuntimeError as exc:
            raise InferenceRuntimeError(exc) from exc
    return guarded


class _DeviceTokenizer:
    """仅保护 pad 后的输入搬运，分词算法和 BatchEncoding 本体不变。"""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def pad(self, *args: Any, **kwargs: Any) -> Any:
        batch = _escape_runtime_error(self._tokenizer.pad)(*args, **kwargs)
        return _DeviceBatch(batch) if callable(getattr(batch, "to", None)) else batch


class _DeviceBatch(Mapping):
    """.to 成功即返回原库对象，不把代理格式传入模型。"""

    def __init__(self, batch: Any) -> None:
        self._batch = batch

    def __getitem__(self, key: Any) -> Any:
        return self._batch[key]

    def __iter__(self) -> Iterator:
        return iter(self._batch)

    def __len__(self) -> int:
        return len(self._batch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._batch, name)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return _escape_runtime_error(self._batch.to)(*args, **kwargs)


def synchronize_device(device: str) -> None:
    """GPU forward 返回不代表计算完成；同步等待必须包含在计时和截止检查之间。"""
    if device == "cpu":
        return
    import torch

    if device in {"mps", "mps:0"}:
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize(device)


def release_device_cache(device: str) -> str | None:
    """只清空闲缓存。不可用设备不触碰 native；已知清理失败不能挡住 CPU 尝试。"""
    if device == "cpu":
        return None
    gc.collect()
    import torch

    if device in {"mps", "mps:0"}:
        capability = getattr(getattr(torch, "backends", None), "mps", None)
        if capability is None or not capability.is_available():
            return "device_unavailable"
        backend = torch.mps
    else:
        if not torch.cuda.is_available():
            return "device_unavailable"
        backend = torch.cuda
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except (RuntimeError, NotImplementedError) as exc:
            reason = classify_device_error(exc, device)
            if reason is None:
                raise
            return reason
    return None
