from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR
from ..runtime.l1.semantic_contracts import (
    L1_SEMANTIC_VERIFICATION_FEATURE,
    parse_l1_semantic_verification_mode,
)
from .feature_contracts import FeatureFlags


DEFAULT_TURN_WALL_CLOCK_BUDGET_S = 1500.0


DEFAULT_FEATURES: FeatureFlags = {
# 交互式产品 Turn 可以因缺少用户输入而暂停；封闭世界运行则必须基于已获授权的证据面完成。
    "user_interaction_mode": "interactive",
    # Provider 控制通道：在原生工具调用能力经明确验证前，auto 一律走加固的 prompt-JSON 路径。
    "model_control_transport": "auto",
    # 普通产品运行保留公共 Web 能力；封闭世界评测必须在其 Runtime
    # 配置中显式关闭。
    "l1_external_web_tools_enabled": True,
    "l1_max_attempts": 24,
    "l1_max_tool_calls_per_attempt": 8,
# 执行局部 findings 仍是持久的 Host 伴随状态。此开关只控制模型可见工具、prompt 投影和
# 语义压缩，从而在不改变所有者恢复机制的前提下提供一条回滚路径。
    "execution_findings_enabled": True,
# 默认情况下，每个 L1 最终候选都要经过语义质量门。运维人员仍可为受控延迟实验显式选择
# conditional/off，但仅由 Host 执行的结构检查不能在语义上替代该 reviewer。
    "l1_semantic_verification_mode": "always",
# 整个 Turn 共用一个 deadline，而不是为每次模型调用分别分配时限。
    "turn_wall_clock_budget_s": DEFAULT_TURN_WALL_CLOCK_BUDGET_S,
# Project Document 与当前 Session 是 L1 的生产检索语料。两者使用独立派生索引，但共享
# retrieval profile、检索单元契约和方法组合；长期用户/Task 记忆仍严格关闭。
# 具体 encoder 能力由 retrieval profile/generation 再次失败关闭。
    "file_retrieval_write_enabled": True,
    "file_retrieval_read_enabled": True,
    # 兼容字段名暂保留 ``history``；当前实现只授权同一 Session、截止到 AcceptedTurn
    # 快照的已提交 Turn，不会启用长期记忆或 L2 History。
    "history_retrieval_write_enabled": True,
    "history_retrieval_read_enabled": True,
    "l1_retrieval_tools_enabled": True,
    "l2_aux_retrieval_tools_enabled": False,
    "l2_task_retrieval_tools_enabled": False,
    "session_context_repair_apply_enabled": False, # controlled replace 默认关闭，只开放 dry-run
    "tools_loop": True,
    "context_guard_limit": 24000,            # C2：provider envelope 的硬准入上限
}

def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_features(config_path: str | None) -> FeatureFlags:
    selected_path = str(config_path or os.getenv("PERSONAGRAPH_RUNTIME_CONFIG") or "").strip()
    if not selected_path:
        return DEFAULT_FEATURES.copy()
    path = Path(selected_path)
    if not path.is_absolute():
        path = CONFIG_DIR / selected_path
    data = _read_yaml_mapping(path)
    return resolve_features(data.get("features", {}))


def resolve_features(raw_features: Mapping[str, Any] | None) -> FeatureFlags:
    """将覆盖项解析为经过验证的完整功能快照。

    基线/评估代码必须记录此解析后映射，而不是部分配置文件；否则未来默认值变化会使旧运行
    无法复现。
    """
    raw_features = dict(raw_features or {})
    _reject_unknown_feature_keys(raw_features)
    features = DEFAULT_FEATURES.copy()
    features.update(raw_features)

    _validate_user_interaction_mode(features)
    _validate_model_control_transport(features)
    _validate_l1_external_web_tools_enabled(features)
    _validate_l1_execution_limits(features)
    _validate_execution_findings_enabled(features)
    _validate_turn_wall_clock_budget(features)
    _validate_l1_semantic_verification_mode(features)
    _validate_dual_corpus_retrieval_features(features)
    return features


def _validate_user_interaction_mode(features: dict) -> None:
    mode = features.get("user_interaction_mode", "interactive")
    if not isinstance(mode, str) or mode.strip().lower() not in {
        "interactive",
        "closed_world",
    }:
        raise ValueError(
            "user_interaction_mode must be 'interactive' or 'closed_world'"
        )
    features["user_interaction_mode"] = mode.strip().lower()


def _validate_model_control_transport(features: dict) -> None:
    mode = features.get("model_control_transport", "auto")
    if not isinstance(mode, str) or mode.strip().lower() not in {"auto", "native", "prompt_json"}:
        raise ValueError("model_control_transport must be 'auto', 'native', or 'prompt_json'")
    features["model_control_transport"] = mode.strip().lower()


def _validate_l1_external_web_tools_enabled(features: dict) -> None:
    enabled = features.get("l1_external_web_tools_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("l1_external_web_tools_enabled must be a boolean")


def _validate_l1_execution_limits(features: dict) -> None:
    for key, minimum, maximum in (
        ("l1_max_attempts", 2, 64),
        ("l1_max_tool_calls_per_attempt", 1, 16),
    ):
        value = features.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be within {minimum}..{maximum}")


def _validate_execution_findings_enabled(features: dict) -> None:
    enabled = features.get("execution_findings_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("execution_findings_enabled must be a boolean")


def _validate_turn_wall_clock_budget(features: dict) -> None:
    value = features.get("turn_wall_clock_budget_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("turn_wall_clock_budget_s must be a positive finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("turn_wall_clock_budget_s must be a positive finite number")
    features["turn_wall_clock_budget_s"] = normalized


def _validate_l1_semantic_verification_mode(features: dict) -> None:
    mode = parse_l1_semantic_verification_mode(
        features.get(L1_SEMANTIC_VERIFICATION_FEATURE)
    )
    features[L1_SEMANTIC_VERIFICATION_FEATURE] = mode.value


def _validate_dual_corpus_retrieval_features(features: dict) -> None:
    bool_keys = (
        "file_retrieval_write_enabled",
        "file_retrieval_read_enabled",
        "history_retrieval_write_enabled",
        "history_retrieval_read_enabled",
        "l1_retrieval_tools_enabled",
        "l2_aux_retrieval_tools_enabled",
        "l2_task_retrieval_tools_enabled",
    )
    for key in bool_keys:
        if not isinstance(features.get(key), bool):
            raise ValueError(f"{key} must be a boolean")
    if features["history_retrieval_read_enabled"] and not features[
        "history_retrieval_write_enabled"
    ]:
        raise ValueError(
            "history_retrieval_read_enabled requires "
            "history_retrieval_write_enabled"
        )
    if features["file_retrieval_read_enabled"] and not features[
        "file_retrieval_write_enabled"
    ]:
        raise ValueError(
            "file_retrieval_read_enabled requires file_retrieval_write_enabled"
        )
    if features["l1_retrieval_tools_enabled"] and not (
        features["file_retrieval_read_enabled"]
        or features["history_retrieval_read_enabled"]
    ):
        raise ValueError(
            "l1_retrieval_tools_enabled requires at least one retrieval read gate"
        )
    if (
        features["l2_aux_retrieval_tools_enabled"]
        or features["l2_task_retrieval_tools_enabled"]
    ) and not (
        features["file_retrieval_read_enabled"]
        or features["history_retrieval_read_enabled"]
    ):
        raise ValueError(
            "L2 retrieval exposure requires at least one retrieval read gate"
        )


def _reject_unknown_feature_keys(raw_features: Mapping[str, Any]) -> None:
    unknown = sorted(set(raw_features).difference(DEFAULT_FEATURES))
    if unknown:
        raise ValueError(f"unsupported feature key: {unknown[0]}")
