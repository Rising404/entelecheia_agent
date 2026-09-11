"""Web 工具共享的网络安全 URL 解析。

本模块中的函数特意不了解工具注册表、研究 Session、缓存、Runtime 或提供商。
它们只建立调用方发出公共 HTTP 请求前所需的网络边界。
"""

from __future__ import annotations

from contextlib import contextmanager
import ipaddress
import socket
import threading
from collections.abc import Iterator
from urllib.parse import urlparse


_dns_pin_local = threading.local()
_real_getaddrinfo = socket.getaddrinfo


def is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """返回一个地址是否绝不能通过 Web 工具访问。"""

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _pinning_getaddrinfo(host, port, *args, **kwargs):
    pins = getattr(_dns_pin_local, "map", None)
    if pins and host in pins:
        ip = pins[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        sockaddr = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]
    return _real_getaddrinfo(host, port, *args, **kwargs)


if socket.getaddrinfo is not _pinning_getaddrinfo:
    socket.getaddrinfo = _pinning_getaddrinfo


@contextmanager
def pin_host(host: str, ip: str) -> Iterator[None]:
    """在当前线程中将一个已批准主机固定到一个已解析 IP。"""

    previous = getattr(_dns_pin_local, "map", None)
    _dns_pin_local.map = {**(previous or {}), host: ip}
    try:
        yield
    finally:
        _dns_pin_local.map = previous


def resolve_pin_target(
    url: str,
) -> tuple[bool, str, str | None, str | None, str | None]:
    """解析一个公共 URL，并选择经 DNS 固定的连接目标。"""

    ok, reason, safe_url, ips = resolve_public_http_url(url)
    if not ok or safe_url is None or not ips:
        return ok, reason, None, None, None
    host = urlparse(safe_url).hostname
    return True, "allowed", safe_url, host, sorted(ips)[0]


def resolve_public_http_url(url: str) -> tuple[bool, str, str | None, frozenset[str]]:
    """验证一个公共 HTTP(S) URL，并返回其当前公共地址。"""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "invalid_url", None, frozenset()
    if parsed.username or parsed.password:
        return False, "url_userinfo_blocked", None, frozenset()
    host = parsed.hostname
    if not host:
        return False, "invalid_host", None, frozenset()
    if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(
        ".local"
    ):
        return False, "private_host_blocked", None, frozenset()
    try:
        infos = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False, "dns_lookup_failed", None, frozenset()

    public_ips: set[str] = set()
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False, "invalid_address", None, frozenset()
        if is_blocked_ip(ip):
            return False, "private_address_blocked", None, frozenset()
        public_ips.add(str(ip))
    if not public_ips:
        return False, "dns_lookup_failed", None, frozenset()
    return True, "allowed", parsed.geturl(), frozenset(sorted(public_ips))


def validate_public_http_url(url: str) -> tuple[bool, str, str | None]:
    """兼容性良好的单遍公共 URL 验证。"""

    ok, reason, safe_url, _ips = resolve_public_http_url(url)
    return ok, reason, safe_url


def validate_stable_public_http_url(url: str) -> tuple[bool, str, str | None]:
    """拒绝 DNS 答案在初始验证期间发生变化的 URL。"""

    ok, reason, safe_url, first_ips = resolve_public_http_url(url)
    if not ok or safe_url is None:
        return ok, reason, None
    ok, reason, stable_url, second_ips = resolve_public_http_url(safe_url)
    if not ok or stable_url is None:
        if reason == "private_address_blocked":
            return False, "dns_rebinding_detected", None
        return ok, reason, None
    if first_ips.isdisjoint(second_ips):
        return False, "dns_rebinding_detected", None
    return True, "allowed", stable_url


__all__ = [
    "is_blocked_ip",
    "pin_host",
    "resolve_pin_target",
    "resolve_public_http_url",
    "validate_public_http_url",
    "validate_stable_public_http_url",
]
