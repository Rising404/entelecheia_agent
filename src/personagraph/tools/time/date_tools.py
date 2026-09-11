"""默认工具目录暴露的当前本地日历工具。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration


DATE_TOOL_CONTRACT_VERSION = "2.0.0"
DATE_TOOL_IMPLEMENTATION_VERSION = "local-calendar-v1"
DATE_TOOL_IDS = ("get_today", "date_after")
DATE_TOOL_SOURCE_FINGERPRINT = "local-calendar-2026-09-03"
_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

DATE_TOOL_SOURCE = ToolSourceDescriptor(
    kind=ToolSourceKind.LOCAL,
    source_id="personagraph.local_calendar",
    fingerprint=DATE_TOOL_SOURCE_FINGERPRINT,
    display_name="Local Calendar",
)


def get_today(_payload: dict[str, Any]) -> dict[str, str]:
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    return {
        "date": today.isoformat(),
        "weekday": _WEEKDAYS[today.weekday()],
        "iso_week": f"{iso_year}-W{iso_week:02d}",
    }


def date_after(payload: dict[str, Any]) -> dict[str, str | int]:
    days = int(payload["days"])
    target = date.today() + timedelta(days=days)
    return {
        "date": target.isoformat(),
        "weekday": _WEEKDAYS[target.weekday()],
        "days_offset": days,
    }


def build_date_tool_registrations() -> tuple[ToolRegistration, ...]:
    effects = ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.RUNTIME_STATE,
                action=EffectAction.READ,
                scope_kind=EffectScopeKind.LOCAL,
                default_scope="local_calendar",
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )
    execution = ToolExecutionProfile(default_timeout_s=2, hard_timeout_s=5)
    common = {
        "contract_version": DATE_TOOL_CONTRACT_VERSION,
        "output_schema": {"type": "object"},
        "catalog_tags": ("read", "time"),
    }
    return (
        ToolRegistration(
            spec=ToolSpec(
                tool_id="get_today",
                name="获取今天日期",
                description="返回本机今天的 ISO 日期、星期和 ISO 周数。",
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                **common,
            ),
            implementation_version=DATE_TOOL_IMPLEMENTATION_VERSION,
            source=DATE_TOOL_SOURCE,
            handler=get_today,
            effect_profile=effects,
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=ToolSpec(
                tool_id="date_after",
                name="计算相对日期",
                description="返回距今天指定天数的 ISO 日期和星期；负数表示过去。",
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["days"],
                    "properties": {"days": {"type": "integer"}},
                },
                **common,
            ),
            implementation_version=DATE_TOOL_IMPLEMENTATION_VERSION,
            source=DATE_TOOL_SOURCE,
            handler=date_after,
            effect_profile=effects,
            execution_profile=execution,
        ),
    )


__all__ = [
    "DATE_TOOL_CONTRACT_VERSION",
    "DATE_TOOL_IDS",
    "DATE_TOOL_IMPLEMENTATION_VERSION",
    "DATE_TOOL_SOURCE",
    "DATE_TOOL_SOURCE_FINGERPRINT",
    "build_date_tool_registrations",
    "date_after",
    "get_today",
]
