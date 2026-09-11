"""具体计算设备的身份与进程内互斥入口；不依赖检索编排或设备探测。"""

from __future__ import annotations

import re
import threading


def normalize_device(device: str) -> str:
    """探测、编码和重排的设备别名必须落到同一把锁；auto 必须先解析。"""
    value = str(device).strip().lower()
    aliases = {"cuda": "cuda:0", "mps:0": "mps", "cpu:0": "cpu"}
    value = aliases.get(value, value)
    if value in {"cpu", "mps"}:
        return value
    if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value):
        return value
    raise ValueError("device must be cpu, mps, cuda or cuda:N (resolve auto first)")


_RESOURCE_LOCKS: dict[str, threading.Lock] = {}
_RESOURCE_LOCKS_GUARD = threading.Lock()


def compute_resource_lock(device: str) -> threading.Lock:
    """同进程同设备共用非重入锁；每次完整设备操作由最外层入口持锁一次。"""
    canonical = normalize_device(device)
    with _RESOURCE_LOCKS_GUARD:
        return _RESOURCE_LOCKS.setdefault(canonical, threading.Lock())
