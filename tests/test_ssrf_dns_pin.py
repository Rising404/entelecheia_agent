"""A2：DNS-pin 关闭 SSRF 的 TOCTOU/rebinding 窗口。"""
from __future__ import annotations

import socket
import threading

from personagraph.tools.web import web_security as security


def test_pin_host_overrides_resolution_thread_local():
    # pin 内：目标主机解析到指定 IP；pin 外：委托真实解析
    host = "pin-target.invalid"
    with security.pin_host(host, "203.0.113.7"):
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        assert infos[0][4][0] == "203.0.113.7"
    # 退出上下文后 pin 清除
    assert getattr(security._dns_pin_local, "map", None) in (None, {})


def test_pin_does_not_leak_across_threads():
    seen = {}

    def worker():
        # 子线程没有设 pin → 不应看到主线程的 pin（线程隔离）
        seen["map"] = getattr(security._dns_pin_local, "map", None)

    with security.pin_host("main-only.invalid", "203.0.113.9"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["map"] in (None, {})


def test_unpinned_host_delegates_to_real_resolver(monkeypatch):
    called = {"n": 0}
    real = security._real_getaddrinfo

    def spy(host, port, *a, **k):
        called["n"] += 1
        return real("127.0.0.1", port, *a, **k)

    monkeypatch.setattr(security, "_real_getaddrinfo", spy)
    socket.getaddrinfo("localhost", 80, type=socket.SOCK_STREAM)  # 无 pin
    assert called["n"] == 1  # 委托到真实解析


def test_resolve_pin_target_blocks_private(monkeypatch):
    # 解析到私网地址应被拒（不泄漏 pin 目标）
    def fake_addrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]
    monkeypatch.setattr(security, "_real_getaddrinfo", fake_addrinfo)
    ok, reason, url, host, ip = security.resolve_pin_target("http://evil.example/")
    assert ok is False and ip is None


def test_resolve_pin_target_returns_ip_for_public(monkeypatch):
    def fake_addrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]
    monkeypatch.setattr(security, "_real_getaddrinfo", fake_addrinfo)
    ok, reason, url, host, ip = security.resolve_pin_target("https://example.com/x")
    assert ok is True and ip == "93.184.216.34" and host == "example.com"
