"""自动起名读模型回复的边界。

这条断过一次，而且断得很隐蔽：解析失败会退回"把用户原话截断"，于是表面现象是
"自动起名根本没生效"，标题看着就是用户刚发的那句话——没人会想到去查 JSON 解析。
"""

import pytest

from personagraph.api.service.session_titles import _model_json_object


@pytest.mark.parametrize(
    "reply",
    [
        '{"title":"季度销售核查"}',
        '```json\n{"title":"季度销售核查"}\n```',
        '```\n{"title":"季度销售核查"}\n```',
        '好的，这是标题：{"title":"季度销售核查"}',
        '  \n{"title":"季度销售核查"}\n  ',
    ],
)
def test_reads_the_title_through_the_wrappers_models_actually_use(reply):
    assert (_model_json_object(reply) or {}).get("title") == "季度销售核查"


@pytest.mark.parametrize("reply", ["", "季度销售核查", "```json\n不是对象\n```", "[1, 2]"])
def test_returns_none_rather_than_raising_when_there_is_no_object(reply):
    """读不出来要返回 None，让调用方走兜底；抛异常会把整轮起名吞掉。"""

    assert _model_json_object(reply) is None


@pytest.mark.parametrize(
    "reply",
    ["季度销售核查", "`季度销售核查`", "  季度销售核查  "],
)
def test_a_bare_title_is_accepted_rather_than_thrown_away(reply):
    """提示词要求只输出 JSON，但模型有时直接把标题甩回来。

    那正是我们要的东西，只是没包起来。丢掉它意味着退回"用户原话截断"——比收下
    一个没包 JSON 的正确标题差得多。
    """

    from personagraph.api.service.session_titles import _bare_title

    assert _bare_title(reply) == "季度销售核查"


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "抱歉，我需要更多信息才能\n给这段对话起一个合适的标题",
        '{"headline":"字段名不对"}',
        "这是一个非常长的标题" * 6,
    ],
)
def test_bare_title_stays_narrow_so_explanations_do_not_become_titles(reply):
    """放宽的边界要窄：多行、过长、带 JSON 残留的更可能是模型在解释或出错。"""

    from personagraph.api.service.session_titles import _bare_title

    assert _bare_title(reply) == ""
