"""将设备偏好解析为一个具体后端；在冻结向量索引身份前完成选择。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inference import classify_device_error
from .resources import compute_resource_lock, normalize_device


def normalize_device_preference(device: str) -> str:
    value = str(device).strip().lower()
    return value if value == "auto" else normalize_device(value)


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    requested_device: str
    device: str | None
    fallback_reason: str | None = None
    probe_reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.device is not None

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "device": self.device,
            "ready": self.ready,
            "fallback_reason": self.fallback_reason,
            "probe_reasons": self.probe_reasons,
            # 一个小张量能运行，不代表 BGE 权重/全部算子已成功推理。
            "model_verified": False,
        }


def select_device(preference: str) -> DeviceSelection:
    """auto 优先单张 CUDA、其次 MPS；只有明确能力失败才自动回 CPU。"""
    requested = normalize_device_preference(preference)
    if requested == "cpu":
        return DeviceSelection(requested, "cpu")
    try:
        import torch
    except ImportError:
        return DeviceSelection(
            requested,
            "cpu" if requested == "auto" else None,
            "torch_unavailable" if requested == "auto" else None,
            ("torch_unavailable",),
        )
    candidates = ("cuda:0", "mps") if requested == "auto" else (requested,)
    reasons: list[str] = []
    last_reason = "device_unavailable"
    for device in candidates:
        if not _available(torch, device):
            reasons.append(f"{device}:device_unavailable")
            continue
        try:
            _probe(torch, device)
        except (RuntimeError, NotImplementedError) as exc:
            reason = classify_device_error(exc, device)
            if reason is None:
                raise
            last_reason = reason
            reasons.append(f"{device}:{reason}")
            continue
        return DeviceSelection(requested, device, probe_reasons=tuple(reasons))
    return DeviceSelection(
        requested,
        "cpu" if requested == "auto" else None,
        last_reason if requested == "auto" else None,
        tuple(reasons),
    )


def _available(torch: Any, device: str) -> bool:
    if device == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        return backend is not None and bool(backend.is_available())
    return bool(torch.cuda.is_available()) and int(device.split(":")[1]) < int(
        torch.cuda.device_count()
    )


def _probe(torch: Any, device: str) -> None:
    # 探测也是实际设备工作：MPS 的 item/synchronize 与另一个线程并发时可能触发
    # 原生 command-buffer 断言，Python 异常回退无法捕获。必须和模型推理共用锁，
    # 覆盖创建张量到最终同步的整个操作，而不只是最后一次 synchronize。
    with compute_resource_lock(device):
        tensor = torch.ones((1,), device=device)
        tensor.sum().item()
        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(device)
