"""Runtime 不得猜测的最小逐模型能力事实。

在此之前，entry 预算使用固定 24000 token guard，与任何真实模型窗口都无关；图像附件因文本
计数器无内容可数而被估算为零 token。二者都是针对具体模型的决定，因此应属于 profile，
而非常量。

这是为解除附件阻塞而刻意设计的最小 profile；采样参数、定价和工具调用方言在实际需要前
保持在外。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelProfile:
    """从输入准入角度看，一个目标模型接受的内容。"""

    model_id: str
    context_window: int
    accepts_images: bool = False
    max_image_bytes: int = 5 * 1024 * 1024
    max_images_per_call: int = 8
    # 视觉模型按像素面积计费。字节数并非面积的良好代理，但它是解码前唯一可用数值；高估可
    # 保持预算保守，避免图像突破 guard 误以为空闲的窗口。
    image_tokens_per_kib: int = 6
    min_image_tokens: int = 256

    def image_tokens(self, size_bytes: int) -> int:
        """估算一张图像在目标模型窗口中的成本。"""

        if not self.accepts_images:
            return 0
        kib = max(1, (max(0, size_bytes) + 1023) // 1024)
        return max(self.min_image_tokens, kib * self.image_tokens_per_kib)


# 模型族按子字符串匹配，因为部署可以自由重命名模型（日期后缀、供应商前缀、自托管别名）。
_FAMILY_PROFILES: tuple[tuple[str, ModelProfile], ...] = (
    ("claude", ModelProfile("claude", 200_000, accepts_images=True, max_images_per_call=20)),
    ("gpt-4o", ModelProfile("gpt-4o", 128_000, accepts_images=True)),
    ("gpt-4", ModelProfile("gpt-4", 128_000, accepts_images=True)),
    ("gpt-5", ModelProfile("gpt-5", 256_000, accepts_images=True)),
    ("gemini", ModelProfile("gemini", 1_000_000, accepts_images=True, max_images_per_call=16)),
    ("qwen-vl", ModelProfile("qwen-vl", 128_000, accepts_images=True)),
    # DeepSeek V4 当前文档声明 1M 上下文窗口。输出上限仍由调用点策略决定；此值只用于在
    # 扣除输出预留后判断安全输入准入。
    ("deepseek", ModelProfile("deepseek", 1_000_000, accepts_images=False)),
)

# 未识别模型采用保守形状：小窗口且无视觉能力。宣称供应商并不具备的图像支持，会让每次图像
# 上传变成硬模型错误；宣称无支持则只会在 manifest 中降级为如实的“此模型无法查看图像”。
UNKNOWN_MODEL_PROFILE = ModelProfile("unknown", 32_000, accepts_images=False)

MOCK_MODEL_PROFILE = ModelProfile("mock", 32_000, accepts_images=True)


def profile_for_model(model_id: str | None, *, provider: str | None = None) -> ModelProfile:
    """解析已配置模型的 profile，并采用保守默认值。"""

    if (provider or "").strip().lower() in {"", "mock"}:
        return MOCK_MODEL_PROFILE
    name = (model_id or "").strip().lower()
    if not name:
        return UNKNOWN_MODEL_PROFILE
    for token, profile in _FAMILY_PROFILES:
        if token in name:
            return replace(profile, model_id=model_id or profile.model_id)
    return replace(UNKNOWN_MODEL_PROFILE, model_id=model_id or "unknown")


def active_model_profile() -> ModelProfile:
    """本安装当前配置调用模型的 profile。"""

    from ..configuration.app_settings import get_setting

    return profile_for_model(get_setting("model", ""), provider=get_setting("provider", "mock"))
