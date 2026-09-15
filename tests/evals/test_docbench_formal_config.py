from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from evals.docbench.reproduce_or_run_script.config import (
    BENCH_EVAL_DIR_ENV,
    DocBenchConfigError,
    canonical_config_sha256,
    docbench_root,
    load_docbench_config,
    resolve_config_path,
)


PROMPT_SHA256 = "a" * 64
NATURAL_DOCUMENT_QA_PREAMBLE = "Answer the question using the attached document.\n"


def test_gate_off_experiment_changes_only_the_existing_semantic_review_switch(
    tmp_path: Path,
) -> None:
    from personagraph.configuration.features import DEFAULT_FEATURES, load_features

    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "evals/docbench/configs"
    environment = {BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")}
    baseline = load_docbench_config(
        config_dir / "l1_balanced_125.yaml", environment=environment,
    )
    experiment = load_docbench_config(
        config_dir / "l1_balanced_125_gate_off.yaml", environment=environment,
    )
    expected = deepcopy(baseline.raw)
    expected["run"]["runtime_features"] = (
        "evals/docbench/configs/runtime_closed_world_gate_off.yaml"
    )
    assert experiment.raw == expected
    baseline_features = load_features(str(baseline.resolved["run"]["runtime_features"]))
    experiment_features = load_features(str(experiment.resolved["run"]["runtime_features"]))
    assert baseline_features["l1_semantic_verification_mode"] == "always"
    assert experiment_features == baseline_features | {"l1_semantic_verification_mode": "off"}
    assert DEFAULT_FEATURES["l1_semantic_verification_mode"] == "always"


def test_checked_in_case_timeouts_leave_headroom_above_the_frozen_turn_budget(
    tmp_path: Path,
) -> None:
    from personagraph.configuration.features import load_features

    project_root = Path(__file__).resolve().parents[2]
    configs = sorted((project_root / "evals/docbench/configs").glob("l1_*.yaml"))
    assert configs
    for config_path in configs:
        loaded = load_docbench_config(
            config_path,
            project_root=project_root,
            environment={BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")},
        )
        features = load_features(str(loaded.resolved["run"]["runtime_features"]))
        assert features["turn_wall_clock_budget_s"] == 1500, config_path.name
        assert loaded.raw["run"]["per_case_timeout_s"] == 1800, config_path.name
        assert loaded.raw["run"]["per_case_timeout_s"] > features["turn_wall_clock_budget_s"]


def test_highland_235b_config_only_replaces_the_vision_endpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    configs = root / "evals/docbench/configs"
    environment = {BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")}
    baseline = load_docbench_config(
        configs / "l1_balanced_125_gate_off.yaml", environment=environment,
    )
    experiment = load_docbench_config(
        configs / "l1_balanced_125_gate_off_highland_235b.yaml", environment=environment,
    )
    expected = deepcopy(baseline.raw)
    expected["providers"]["vision"] = {
        "provider": "highland",
        "base_url": "https://www.highland-api.top",
        "model": "qwen3-vl-235b-a22b-instruct",
        "api_key_env": "HIGHLAND_API_KEY",
    }
    assert experiment.raw == expected
    assert experiment.sha256 != baseline.sha256


def test_highland_235b_endpoint_resolves_without_changing_installation_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals.docbench.reproduce_or_run_script import runner

    def reject_installation_read(*args, **kwargs):
        pytest.fail("explicit vision endpoint must not read or change the active profile")

    monkeypatch.setattr(runner, "_active_installation_profile", reject_installation_read)
    monkeypatch.setenv("HIGHLAND_API_KEY", "test-only-placeholder")
    resolved = runner._resolve_provider(
        {
            "provider": "highland", "base_url": "https://www.highland-api.top",
            "model": "qwen3-vl-235b-a22b-instruct", "api_key_env": "HIGHLAND_API_KEY",
        },
        kind="vision", require_secret=True,
    )
    assert resolved["model"] == "qwen3-vl-235b-a22b-instruct"
    assert resolved["api_key"] == "test-only-placeholder"
    assert resolved["credential_source"] == "env:HIGHLAND_API_KEY"
    monkeypatch.delenv("HIGHLAND_API_KEY")
    with pytest.raises(runner.DocBenchRunnerError, match="credential is unavailable"):
        runner._resolve_provider(
            {"provider": "highland", "base_url": "https://www.highland-api.top",
             "model": "qwen3-vl-235b-a22b-instruct", "api_key_env": "HIGHLAND_API_KEY"},
            kind="vision", require_secret=True,
        )


def _valid_payload() -> dict:
    return {
        "schema_version": "docbench-l1-eval-v1",
        "lane": "L1",
        "providers": {
            "main": {
                "provider": "openai-compatible",
                "request_dialect": "deepseek",
                "base_url": "https://models.example.test/api/v1",
                "model": "deepseek-chat",
                "api_key_env": "DOCBENCH_MAIN_API_KEY",
            },
            "vision": {
                "provider": "example-vision",
                "base_url": "https://vision.example.test",
                "model": "example-vl",
                "api_key_env": "DOCBENCH_VISION_API_KEY",
            },
        },
        "retrieval": {
            "profile": "bge_m3",
            "failure_policy": "strict",
            "required_methods": ["dense", "learned_sparse", "bm25"],
            "encoder": {
                "model_id": "BAAI/bge-m3",
                "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            },
            "reranker": {
                "mode": "bge_v2_m3",
                "model_id": "BAAI/bge-reranker-v2-m3",
                "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            },
            "local_files_only": True,
            "device": "cpu",
            "use_fp16": False,
        },
        "dataset": {
            "data_root": "fixtures/docbench/formal_data",
            "selection": "evals/docbench/selections/example_selection.json",
        },
        "run": {
            "runtime_features": "evals/docbench/configs/runtime_closed_world.yaml",
            "output_root": "artifacts/docbench/formal_runs",
            "per_case_timeout_s": 1800,
            "max_workers": 2,
        },
        "prompt": {
            "preamble": "Answer only from the attached PDF.\n\n",
        },
        "scoring": {
            "mode": "docbench_prompt_compatible",
            "prompt": "fixtures/docbench/evaluation_prompt.txt",
            "prompt_sha256": PROMPT_SHA256,
            "judge": {"source": "main"},
        },
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "docbench.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_loads_strict_l1_config_and_resolves_paths_from_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    config_path = _write_config(tmp_path, _valid_payload())

    loaded = load_docbench_config(config_path, project_root=project_root)

    assert loaded.source_path == config_path.resolve()
    assert loaded.canonical_snapshot["dataset"]["data_root"] == (
        "fixtures/docbench/formal_data"
    )
    assert loaded.resolved["dataset"]["data_root"] == (
        project_root / "fixtures/docbench/formal_data"
    ).resolve()
    assert loaded.resolved["dataset"]["selection"] == (
        project_root / "evals/docbench/selections/example_selection.json"
    ).resolve()
    assert loaded.resolved["run"]["runtime_features"] == (
        project_root / "evals/docbench/configs/runtime_closed_world.yaml"
    ).resolve()
    assert loaded.resolved["run"]["output_root"] == (
        project_root / "artifacts/docbench/formal_runs"
    ).resolve()
    assert loaded.resolved["scoring"]["prompt"] == (
        project_root / "fixtures/docbench/evaluation_prompt.txt"
    ).resolve()
    assert loaded.config_sha256 == canonical_config_sha256(
        loaded.canonical_snapshot
    )
    assert len(loaded.sha256) == 64


def test_checked_in_preambles_do_not_prescribe_an_agent_strategy(tmp_path: Path) -> None:
    configs_dir = Path(__file__).resolve().parents[2] / "evals/docbench/configs"

    for path in sorted(configs_dir.glob("l1_*.yaml")):
        loaded = load_docbench_config(
            path, environment={BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")},
        )
        preamble = loaded.raw["prompt"]["preamble"]

        assert preamble == NATURAL_DOCUMENT_QA_PREAMBLE
        normalized = preamble.casefold()
        assert "tool" not in normalized
        assert "visual" not in normalized
        assert "clarification" not in normalized


def test_bench_path_namespace_uses_external_docbench_root_without_changing_repo_paths(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["dataset"]["data_root"] = "bench://source/data"
    payload["run"]["output_root"] = "bench://runs"
    payload["scoring"]["prompt"] = "bench://source/evaluation_prompt.txt"
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    bench_eval_dir = tmp_path / "bench_eval"
    environment = {BENCH_EVAL_DIR_ENV: str(bench_eval_dir)}

    loaded = load_docbench_config(
        _write_config(tmp_path, payload),
        project_root=project_root,
        environment=environment,
    )

    assert loaded.canonical_snapshot["dataset"]["data_root"] == (
        "bench://source/data"
    )
    assert loaded.resolved["dataset"]["data_root"] == (
        bench_eval_dir / "docbench/source/data"
    ).resolve()
    assert loaded.resolved["run"]["output_root"] == (
        bench_eval_dir / "docbench/runs"
    ).resolve()
    assert loaded.resolved["scoring"]["prompt"] == (
        bench_eval_dir / "docbench/source/evaluation_prompt.txt"
    ).resolve()
    assert loaded.resolved["dataset"]["selection"] == (
        project_root / "evals/docbench/selections/example_selection.json"
    ).resolve()


def test_docbench_root_requires_explicit_external_directory() -> None:
    with pytest.raises(DocBenchConfigError, match=f"{BENCH_EVAL_DIR_ENV}.*required"):
        docbench_root(environment={})


@pytest.mark.parametrize(
    ("value", "message"),
    (("", "cannot be empty"), ("   ", "cannot be empty"), ("relative", "absolute path")),
)
def test_docbench_root_rejects_empty_or_relative_directories(value: str, message: str) -> None:
    with pytest.raises(DocBenchConfigError, match=message):
        docbench_root(environment={BENCH_EVAL_DIR_ENV: value})


def test_docbench_root_rejects_a_location_inside_source_checkout() -> None:
    checkout = Path(__file__).resolve().parents[2]
    with pytest.raises(DocBenchConfigError, match="outside the source checkout"):
        docbench_root(environment={BENCH_EVAL_DIR_ENV: str(checkout)})


@pytest.mark.parametrize(
    "value",
    ("bench://", "bench://../escape", r"bench://source\..\escape"),
)
def test_bench_path_namespace_rejects_invalid_or_escaping_paths(value: str) -> None:
    with pytest.raises(DocBenchConfigError, match="bench|escapes"):
        resolve_config_path(value)


def test_old_state_path_namespace_is_rejected() -> None:
    with pytest.raises(DocBenchConfigError, match="unsupported path namespace"):
        resolve_config_path("state://evals/docbench/upstream_data/data")


def test_installation_providers_and_explicit_deepseek_judge_are_supported(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["providers"] = {
        "main": {"source": "installation"},
        "vision": {"source": "installation"},
    }
    payload["scoring"]["judge"] = {
        "provider": "openai-compatible",
        "request_dialect": "deepseek",
        "base_url": "https://judge.example.test/v1",
        "model": "deepseek-v3",
        "api_key_env": "DOCBENCH_JUDGE_API_KEY",
    }

    loaded = load_docbench_config(_write_config(tmp_path, payload))

    assert loaded.raw["providers"]["main"] == {"source": "installation"}
    assert loaded.raw["scoring"]["judge"]["request_dialect"] == "deepseek"


def test_retrieval_profile_is_frozen_without_local_model_paths(tmp_path: Path) -> None:
    loaded = load_docbench_config(_write_config(tmp_path, _valid_payload()))

    assert loaded.raw["retrieval"] == {
        "profile": "bge_m3",
        "failure_policy": "strict",
        "required_methods": ["dense", "learned_sparse", "bm25"],
        "encoder": {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        },
        "reranker": {
            "mode": "bge_v2_m3",
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        },
        "local_files_only": True,
        "device": "cpu",
        "use_fp16": False,
    }


def test_retrieval_profile_can_freeze_auto_device_with_fp32(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["retrieval"]["device"] = "auto"

    loaded = load_docbench_config(_write_config(tmp_path, payload))

    assert loaded.raw["retrieval"]["device"] == "auto"
    assert loaded.raw["retrieval"]["use_fp16"] is False


def test_retrieval_profile_rejects_auto_device_with_fp16(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["retrieval"].update({"device": "auto", "use_fp16": True})

    with pytest.raises(DocBenchConfigError, match="False was expected"):
        load_docbench_config(_write_config(tmp_path, payload))


def test_dense_sparse_eval_method_set_is_an_explicit_supported_config(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["retrieval"]["required_methods"] = ["dense", "learned_sparse"]

    loaded = load_docbench_config(_write_config(tmp_path, payload))

    assert loaded.raw["retrieval"]["required_methods"] == [
        "dense",
        "learned_sparse",
    ]


def test_canonical_hash_ignores_mapping_order_and_yaml_formatting(
    tmp_path: Path,
) -> None:
    first = _valid_payload()
    second = {key: first[key] for key in reversed(first)}
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(yaml.safe_dump(first, sort_keys=False), encoding="utf-8")
    second_path.write_text(
        "\n---\n" + yaml.safe_dump(second, sort_keys=False),
        encoding="utf-8",
    )

    first_loaded = load_docbench_config(first_path)
    second_loaded = load_docbench_config(second_path)

    assert first_loaded.canonical_snapshot == second_loaded.canonical_snapshot
    assert first_loaded.sha256 == second_loaded.sha256

    changed = deepcopy(first)
    changed["run"]["max_workers"] = 3
    assert canonical_config_sha256(changed) != first_loaded.sha256


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda value: value.update({"unknown": True}),
            "Additional properties",
        ),
        (
            lambda value: value["providers"].update({"tiers": {}}),
            "Additional properties",
        ),
        (
            lambda value: value.update({"lane": "L2"}),
            "'L1' was expected",
        ),
        (
            lambda value: value["scoring"].update({"mode": "official"}),
            "docbench_prompt_compatible",
        ),
        (
            lambda value: value["retrieval"].update(
                {"required_methods": ["bm25"]}
            ),
            "dense",
        ),
        (
            lambda value: value["retrieval"]["encoder"].update(
                {"revision": "main"}
            ),
            "5617a9f61b028005a4858fdac845db406aefb181",
        ),
        (
            lambda value: value["retrieval"].update(
                {"local_files_only": False}
            ),
            "True was expected",
        ),
        (
            lambda value: value["scoring"].update(
                {
                    "judge": {
                        "provider": "openai-compatible",
                        "request_dialect": "openai",
                        "base_url": "https://judge.example.test/v1",
                        "model": "deepseek-v3",
                        "api_key_env": "DOCBENCH_JUDGE_API_KEY",
                    }
                }
            ),
            "not valid under any of the given schemas",
        ),
    ],
)
def test_rejects_unknown_l2_and_non_protocol_fields(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    payload = _valid_payload()
    mutate(payload)

    with pytest.raises(DocBenchConfigError, match=expected):
        load_docbench_config(_write_config(tmp_path, payload))


def test_rejects_inline_secret_even_before_schema_validation(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["providers"]["main"].pop("api_key_env")
    payload["providers"]["main"]["api_key"] = "sk-should-never-be-here"

    with pytest.raises(
        DocBenchConfigError,
        match=r"inline secret field is forbidden: providers\.main\.api_key",
    ):
        load_docbench_config(_write_config(tmp_path, payload))


def test_api_key_env_must_name_an_environment_variable(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["providers"]["main"]["api_key_env"] = "sk-inline-secret"

    with pytest.raises(DocBenchConfigError, match="schema validation failed"):
        load_docbench_config(_write_config(tmp_path, payload))
