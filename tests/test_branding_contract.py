from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from personagraph.api.router import ROUTES
from personagraph.api.server import ApiHandler
from personagraph.l2.task_execution.attempts import decision as attempt_decision
from personagraph.l2.auxiliary_execution.work_run import (
    controller as auxiliary_work_run_controller,
)
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.response import model as response_model
from personagraph.runtime.l1 import model as l1_model
from personagraph.tools.web import web_tools


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_public_distribution_and_entrypoint_names_use_entelecheia() -> None:
    metadata = tomllib.loads(_read("pyproject.toml"))["project"]

    assert metadata["name"] == "entelecheia-agent"
    assert "Entelecheia" in metadata["description"]
    assert metadata["scripts"] == {
        "entelecheia-api": "personagraph.api.server:main",
        "entelecheia-retrieval": "personagraph.retrieval.operations.generation_admin:main",
    }
    assert (ROOT / "src/personagraph").is_dir()


def test_frontend_package_and_visible_surfaces_use_entelecheia() -> None:
    package = json.loads(_read("frontend/package.json"))

    assert package["name"] == "entelecheia-frontend"
    assert "Entelecheia" in package["description"]
    readme = _read("README.md")
    assert "Entelecheia" in readme
    assert "`personagraph`" in readme
    assert "persona subsystem" in readme
    for relative_path in (
        "CONTRIBUTING.md",
        "ARCHITECTURE.md",
        "frontend/index.html",
        "frontend/src/components/AppNav.vue",
        "frontend/src/components/WelcomeGuide.vue",
    ):
        content = _read(relative_path)
        assert "Entelecheia" in content
        assert "PersonaGraph" not in content


def test_runtime_brand_changes_without_rotating_compatibility_ids() -> None:
    prompts = "\n".join(
        (
            ingress_model._CLASSIFIER_SYSTEM_PROMPT_BODY,
            response_model._L0_SYSTEM_PROMPT,
            response_model._L2_SYSTEM_PROMPT,
        )
    )

    assert ApiHandler.server_version == "EntelecheiaAPI/0.2"
    assert "Entelecheia" in prompts
    assert "PersonaGraph" not in prompts
    assert web_tools._USER_AGENT.startswith("Entelecheia-Agent/")

    health_route = next(route for route in ROUTES if route.path == ("api", "health"))
    electron_source = _read("frontend/electron/main.js")
    assert health_route.handler({}, {}, {}) == {
        "ok": True,
        "service": "personagraph-api",
    }
    assert '"personagraph:choose-document"' in electron_source


def test_public_documentation_links_only_to_available_checkout_surfaces() -> None:
    documents = (
        "README.md",
        "QUICKSTART.md",
        "examples/document_qa/README.md",
        "CONTRIBUTING.md",
        "ARCHITECTURE.md",
        "AGENTS.md",
        "scripts/DEPENDENCIES.md",
        "evals/README.md",
        "evals/docbench/reproduce_or_run_script/README.md",
        "evals/docbench/docs/formal_l1_eval_runbook.md",
        "evals/docbench/docs/state_isolation.md",
        "evals/docbench/configs/README.md",
        "evals/docbench/results/README.md",
        "evals/docbench/previous_results/README.md",
        "evals/docbench/previous_results/analysis/README.md",
        "evals/docbench/previous_results/analysis/EVALUATION_REVIEW.md",
        "evals/docbench/previous_results/analysis/FAILURE_ATTRIBUTION.md",
    )
    for relative_path in documents:
        document = ROOT / relative_path
        content = document.read_text(encoding="utf-8")
        assert "docs/reports/" not in content, relative_path
        assert "Desktop/bench_eval" not in content, relative_path
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            if "://" in target or target.startswith("#"):
                continue
            local_target = (document.parent / target.split("#", 1)[0]).resolve()
            assert local_target.is_relative_to(ROOT), (relative_path, target)
            assert local_target.exists(), (relative_path, target)


def test_durable_prompt_branding_matches_each_runtime_contract() -> None:
    legacy_prompts = (
        attempt_decision._ATTEMPT_DECISION_SYSTEM_PROMPT,
        auxiliary_work_run_controller._MODEL_WORK_RUN_SYSTEM_PROMPT,
    )

    assert all(prompt.startswith("你是 PersonaGraph") for prompt in legacy_prompts)
    assert l1_model._L1_SYSTEM_PROMPT.startswith(
        "你是负责完成当前用户请求的助手"
    )
    assert "PersonaGraph" not in l1_model._L1_SYSTEM_PROMPT


def test_docbench_config_schema_uses_current_branding() -> None:
    schema_path = ROOT / "evals/docbench/config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["title"] == "Entelecheia DocBench L1 evaluation configuration"
    assert schema["$id"].startswith("https://entelecheia.local/")


def test_public_eval_docs_identify_the_recorded_source_not_a_run() -> None:
    summary = json.loads(_read("evals/docbench/previous_results/showcase_125.summary.json"))
    source_hashes = {run["source_sha256"] for run in summary["runs"]}
    assert len(source_hashes) == 1

    for relative_path in (
        "evals/README.md",
        "evals/docbench/docs/formal_l1_eval_runbook.md",
    ):
        documented = set(re.findall(r"source_sha256=([0-9a-f]{64})", _read(relative_path)))
        assert documented == source_hashes, relative_path


def test_contributor_guide_explains_local_hook_activation() -> None:
    guide = _read("CONTRIBUTING.md")
    assert "git config --local core.hooksPath .githooks" in guide
    assert "git config --get core.hooksPath" in guide
    assert (ROOT / ".githooks/pre-commit").is_file()
    assert (ROOT / ".githooks/pre-push").is_file()
