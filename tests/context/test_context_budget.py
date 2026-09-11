from personagraph.context_budget import token_counter as cb


def test_estimate_tokens_mixed(monkeypatch):
    monkeypatch.delenv("PERSONAGRAPH_TOKEN_COUNTER", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_TOKENIZER_MODEL", raising=False)
    assert cb.estimate_tokens("") == 0
    assert cb.estimate_tokens("中文五个字啊") == 6           # CJK ~1/字
    assert cb.estimate_tokens("abcdefgh") == 2               # 8 字符 /4
    assert cb.estimate_tokens("中文abcd") == 3               # 2 个中日韩字符 + 4/4


def test_token_counter_defaults_to_heuristic(monkeypatch):
    monkeypatch.delenv("PERSONAGRAPH_TOKEN_COUNTER", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_TOKENIZER_MODEL", raising=False)
    counter = cb.get_token_counter()
    assert counter.name == "heuristic"
    assert counter.kind == "heuristic"
    assert counter.fallback_reason is None


def test_unknown_token_counter_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_TOKEN_COUNTER", "mystery-tokenizer")
    counter = cb.get_token_counter()
    assert counter.name == "heuristic"
    assert counter.requested == "mystery-tokenizer"
    assert counter.fallback_reason == "unknown_token_counter"
    assert cb.estimate_tokens("中文abcd") == 3


def test_hf_token_counter_without_model_falls_back(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_TOKEN_COUNTER", "hf")
    monkeypatch.delenv("PERSONAGRAPH_TOKENIZER_MODEL", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)
    counter = cb.get_token_counter()
    assert counter.name == "heuristic"
    assert counter.requested == "hf"
    assert counter.fallback_reason == "hf_model_required"


def test_cap_text_truncates(monkeypatch):
    monkeypatch.delenv("PERSONAGRAPH_TOKEN_COUNTER", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_TOKENIZER_MODEL", raising=False)
    text = "目标" * 100                                       # 约 200 个令牌
    capped = cb.cap_text(text, 20)
    assert cb.estimate_tokens(capped) <= 25                  # 截到约 20（含标记余量）
    assert capped.endswith("…[截断]")
    # 不超限的原样返回
    assert cb.cap_text("短文本", 100) == "短文本"


def test_budget_env_adjustable(monkeypatch):
    assert cb.task_budget() == 12000
    monkeypatch.setenv("PERSONAGRAPH_TASK_BUDGET", "4000")
    assert cb.task_budget() == 4000
    monkeypatch.setenv("PERSONAGRAPH_BUDGET_SUMMARY", "999")
    assert cb.block_cap("summary") == 999
