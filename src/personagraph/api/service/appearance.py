"""一份由用户选择、在本地存储并提供回读的背景图像或视频。

玻璃表面会模糊其背后内容，而柔和渐变几乎没有可模糊结构——所以磨砂效果看起来很弱。照片
或视频能为模糊提供真实结构，这正是让磨砂玻璃看起来像玻璃的原因。

任一时刻只存在一个背景。它是单用户桌面应用中的个人外观选择，而非资料库，因此替换时
直接覆盖。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ...configuration.paths import STATE_DIR
from .errors import ApiError


DIR = STATE_DIR / "appearance"
META_PATH = DIR / "background.json"

# 视频比图片大得多，但这是本机文件、不过网络，真正的约束是解码和逐帧重糊的开销，
# 不是传输。48MB 已经足够放一段几秒的循环片段。
MAX_BYTES = 48 * 1024 * 1024

ALLOWED = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def _meta() -> dict[str, Any] | None:
    try:
        data = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("media_type") in ALLOWED else None


def _asset_path(meta: dict[str, Any]) -> Path:
    return Path(DIR) / f"background{ALLOWED[meta['media_type']]}"


def get_background() -> dict[str, Any]:
    """renderer 在 <img>、<video> 或无背景间做决定所需的信息。"""

    meta = _meta()
    if meta is None or not _asset_path(meta).exists():
        return {"background": None}
    return {
        "background": {
            "media_type": meta["media_type"],
            "kind": "video" if meta["media_type"].startswith("video/") else "image",
            "size_bytes": meta.get("size_bytes"),
            "updated_at": meta.get("updated_at"),
            # 带上版本号，换图之后浏览器不会拿旧的缓存
            "url": f"/api/appearance/background/asset?v={meta.get('updated_at', '')}",
        }
    }


def store_background(*, media_type: str, payload: bytes, now: str) -> dict[str, Any]:
    kind = str(media_type or "").split(";", 1)[0].strip().lower()
    if kind not in ALLOWED:
        raise ApiError(
            "UNSUPPORTED_BACKGROUND_TYPE",
            "背景只支持 JPG / PNG / WebP / GIF / MP4 / WebM",
            details={"media_type": kind},
        )
    if not payload:
        raise ApiError("EMPTY_BACKGROUND", "背景文件是空的")
    if len(payload) > MAX_BYTES:
        raise ApiError(
            "BACKGROUND_TOO_LARGE",
            f"背景文件不能超过 {MAX_BYTES // (1024 * 1024)}MB",
            details={"size_bytes": len(payload)},
        )

    directory = Path(DIR)
    directory.mkdir(parents=True, exist_ok=True)
    # 换格式时把旧的那份删掉，否则 jpg 换成 mp4 会留下一个再也没人读的文件
    for suffix in set(ALLOWED.values()):
        stale = directory / f"background{suffix}"
        if stale.suffix != ALLOWED[kind] and stale.exists():
            stale.unlink()

    target = directory / f"background{ALLOWED[kind]}"
    handle, temporary = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    META_PATH.write_text(
        json.dumps(
            {"media_type": kind, "size_bytes": len(payload), "updated_at": now},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return get_background()


def clear_background() -> dict[str, Any]:
    """恢复普通渐变。"""

    directory = Path(DIR)
    for suffix in set(ALLOWED.values()):
        asset = directory / f"background{suffix}"
        if asset.exists():
            asset.unlink()
    if Path(META_PATH).exists():
        Path(META_PATH).unlink()
    return {"background": None}


def read_background_asset() -> tuple[bytes, str]:
    """字节及内容类型，供唯一以二进制响应的路由使用。"""

    meta = _meta()
    if meta is None:
        raise ApiError("BACKGROUND_NOT_SET", "还没有设置背景", status=404)
    asset = _asset_path(meta)
    try:
        return asset.read_bytes(), meta["media_type"]
    except OSError as err:
        raise ApiError("BACKGROUND_NOT_SET", "背景文件读不到", status=404) from err
