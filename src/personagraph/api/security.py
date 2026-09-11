"""回环 HTTP 认证与精确 origin 策略。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import stat
from pathlib import Path

from ..configuration.paths import API_TOKEN_PATH


TOKEN_PATH = API_TOKEN_PATH
DEFAULT_ALLOWED_ORIGINS = {
    "null",  # Electron 文件渲染器；实际请求仍需 Bearer 认证。
    "file://",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
}
# 与 CORS allowlist 刻意分离：Electron 的 null/file origin 永远不能绕过 Bearer。
UNAUTHENTICATED_DEV_ORIGINS = frozenset(
    {
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    }
)
API_IDENTITY_SCHEME = "hmac-sha256-v1"
API_IDENTITY_CONTEXT = b"entelecheia-loopback-api-ownership-v1\0"
_API_IDENTITY_CHALLENGE = re.compile(r"[A-Za-z0-9_-]{43}")
_API_TOKEN: str | None = None


def initialize_api_token(explicit_token: str | None = None) -> str:
    global _API_TOKEN
    configured = explicit_token or os.getenv("PERSONAGRAPH_API_TOKEN")
    token = str(configured or secrets.token_urlsafe(32)).strip()
    if len(token) < 32:
        raise ValueError("PERSONAGRAPH_API_TOKEN must contain at least 32 characters")
    _API_TOKEN = token
    _write_token_file(token)
    return token


def token_matches(authorization: str | None) -> bool:
    token = _API_TOKEN
    if token is None:
        token = initialize_api_token()
    scheme, _, candidate = str(authorization or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(candidate.strip(), token)


def api_identity_proof(challenge: str) -> dict[str, str]:
    """证明当前监听者持有 API token，而不把 token 暴露给探测方。"""

    if (
        not isinstance(challenge, str)
        or _API_IDENTITY_CHALLENGE.fullmatch(challenge) is None
    ):
        raise ValueError("identity challenge must be 32-byte base64url without padding")
    token = _API_TOKEN
    if token is None:
        token = initialize_api_token()
    digest = hmac.new(
        token.encode("utf-8"),
        API_IDENTITY_CONTEXT + challenge.encode("ascii"),
        hashlib.sha256,
    ).digest()
    proof = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return {"scheme": API_IDENTITY_SCHEME, "proof": proof}


def origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True
    return origin in allowed_origins()


def allowed_origins() -> set[str]:
    configured = {
        value.strip()
        for value in os.getenv("PERSONAGRAPH_API_ALLOWED_ORIGINS", "").split(",")
        if value.strip()
    }
    return DEFAULT_ALLOWED_ORIGINS | configured


def unauthenticated_dev_origin_allowed(origin: str | None) -> bool:
    enabled = (
        os.getenv("PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS", "")
        .strip()
        .lower()
    )
    return (
        enabled in {"1", "true", "yes", "on"}
        and origin in UNAUTHENTICATED_DEV_ORIGINS
    )


def _write_token_file(token: str) -> None:
    path = Path(TOKEN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        temporary, descriptor = _open_private_token_temporary(path)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("API token temporary path is not a regular file")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = None
        with stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _open_private_token_temporary(path: Path) -> tuple[Path, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(128):
        temporary = path.with_name(
            f".{path.name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        return temporary, descriptor
    raise FileExistsError("could not allocate a private API token temporary file")


def _fsync_directory(directory: Path) -> None:
    """Persist the rename where directory fsync is supported by the platform."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows and a few filesystems reject directory fsync; the file itself
            # has already been flushed before the atomic replace.
            return
    finally:
        os.close(descriptor)
