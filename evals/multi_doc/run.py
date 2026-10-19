"""Run multi-document attachments through the existing Session/Entry/L1 product path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from .cases import CASES, Case, materialize
from .complex_cases import COMPLEX_CASES
from .pdf_cases import PDF_CASES
from .text_cases import TEXT_CASES


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = {
    "l1_external_web_tools_enabled": False,
    "l1_max_attempts": 12,
    "turn_wall_clock_budget_s": 600,
    "l1_semantic_verification_mode": "always",
}
ALL_CASES = CASES + COMPLEX_CASES + TEXT_CASES + PDF_CASES
SUITES = {
    "smoke": CASES,
    "complex": COMPLEX_CASES,
    "text": tuple(case for case in CASES if not case.needs_vision) + TEXT_CASES,
    "pdf": PDF_CASES,
}


def features_for(case) -> dict:
    features = dict(FEATURES)
    if case.suite in {"complex", "docbench"}:
        features.update(l1_max_attempts=32, turn_wall_clock_budget_s=1800)
    return features


def input_statistics(case) -> dict:
    if case.source_pdfs:
        return {
            "document_count": len(case.source_pdfs), "pdf_count": len(case.source_pdfs),
            "pdf_pages": sum(pdf.physical_pages for pdf in case.source_pdfs), "image_count": 0,
            "pdf_bytes": sum(pdf.path.stat().st_size for pdf in case.source_pdfs),
        }
    return {
        "document_count": len(case.documents) + len(case.images) + int(case.needs_vision and not case.images),
        "pdf_count": sum(name.endswith(".pdf") for name, _ in case.documents),
        "pdf_pages": sum(content.count("\f") + 1 for name, content in case.documents if name.endswith(".pdf")),
        "source_characters": sum(len(content) for _, content in case.documents),
        "source_words": sum(len(content.split()) for _, content in case.documents),
        "image_count": len(case.images) + int(case.needs_vision and not case.images),
    }


def chain_succeeded(response: dict, settlement: dict) -> bool:
    return (
        response.get("status") == "completed" and response.get("processing_level") == "L1"
        and not response.get("error_code")
        and response.get("end_reason") != "l1_terminal_notification"
        and settlement["status"] == "settled"
        and settlement["window_released"] and not settlement["timed_out"]
        and bool(settlement["jobs"])
        and all(job["status"] == "applied" for job in settlement["jobs"])
    )


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def isolated_environment(root: Path, quota_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "PERSONAGRAPH_STATE_DIR": str(root / "state"),
        "PERSONAGRAPH_API_TOKEN_FILE": str(root / "state" / "api_secret"),
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(root / "projects"),
        "PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH": str(quota_root / "quota.sqlite3"),
        "PERSONAGRAPH_DOCUMENT_READER": "native",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return environment


def require_worker_namespace(case_root: Path) -> None:
    if not case_root.is_absolute() or case_root.resolve().is_relative_to(ROOT):
        raise ValueError("worker requires an absolute directory outside the checkout")
    expected = isolated_environment(case_root, case_root.parent)
    for key in (
        "PERSONAGRAPH_STATE_DIR", "PERSONAGRAPH_API_TOKEN_FILE",
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR", "PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH",
    ):
        if os.environ.get(key) != expected[key]:
            raise ValueError(f"worker refuses a non-isolated {key}")


def worker(case: Case, case_root: Path, output: Path) -> int:
    # All state environment variables are supplied before this fresh process imports the app.
    require_worker_namespace(case_root)
    from personagraph.configuration.paths import load_dotenv

    load_dotenv()
    from personagraph.api.service import attachments, sessions
    from personagraph.configuration.features import load_features
    from personagraph.model_io.tier_bindings import ModelTier, resolve_tier
    from personagraph.retrieval.operations.document_maintenance import build_document_retrieval_composition
    from personagraph.runtime.post_commit.scheduler import wait_for_turn_post_commit_jobs
    from personagraph.session import store
    from personagraph.tools.visual.publication_recovery import build_visual_publication_recovery
    from personagraph.workspace.ingestion.composition import build_document_maintenance_lifecycle
    from personagraph.workspace.storage.context import current as current_project_database

    started = time.monotonic()
    row = {
        "case_id": case.id, "scenario": case.scenario, "question": case.question,
        "reference": case.reference, "status": "running", "reply": "",
        "chain_complete": False, "needs_vision": case.needs_vision,
        "suite": case.suite, "input_statistics": input_statistics(case),
        "gold_evidence": [{"file": file, "page": page, "fact": fact} for file, page, fact in case.evidence],
        "review_rubric": list(case.review_rubric),
    }
    try:
        if resolve_tier(ModelTier.L1).provider == "mock":
            raise ValueError("real provider required")
        inputs = materialize(case, case_root / "inputs")
        row["documents"] = [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in inputs
        ]
        project = case_root / "project"
        project.mkdir()
        config = case_root / "features.json"
        save(config, {"features": features_for(case)})  # JSON is accepted by the YAML loader.
        row["features"] = load_features(str(config))
        created = sessions.create_session({"title": f"multi-doc-{case.id}", "working_dir": str(project)})
        session_id = created["session"]["id"]
        row["session_id"] = session_id
        with store.session_database_scope(session_id):
            uploads = [attachments.upload_attachment(
                session_id, payload=path.read_bytes(), filename=path.name,
                declared_media_type=mimetypes.guess_type(path.name)[0] or "text/plain",
            )["attachment"] for path in inputs]
            uploaded = [item["attachment_id"] for item in uploads]
            row["attachments"] = uploads
            composition = build_document_retrieval_composition()
            row["retrieval"] = {
                "requested": composition.requested_profile.mode.value,
                "effective": composition.effective_profile.mode.value,
                "device": composition.effective_profile.device,
                "degraded_reason": composition.degraded_reason,
            }
            maintenance = build_document_maintenance_lifecycle(
                profile=composition.requested_profile,
                encoder=composition.encoder, reranker=composition.reranker,
                after_pass=build_visual_publication_recovery(current_project_database()),
            )
            maintenance.start()
            try:
                response = sessions.chat_turn({
                    "session_id": session_id,
                    "message": question_message(case),
                    "attachment_ids": uploaded,
                    "client_request_id": f"multi-doc-{uuid.uuid4().hex}",
                    "config": str(config),
                    "runtime_policy": {"schema_version": 2, "l1_enabled": True},
                })["result"]
                row.update({key: response.get(key) for key in (
                    "status", "processing_level", "reply", "error_code", "end_reason", "turn_id",
                )})
                row["answer_elapsed_s"] = round(time.monotonic() - started, 3)
                save(output, row)  # Preserve an answer even if background settlement fails.
                turn_id = response.get("turn_id")
                if turn_id:
                    settlement = wait_for_turn_post_commit_jobs(
                        session_id=session_id, turn_id=turn_id, store=store, timeout_seconds=120,
                    ).to_dict()
                    row["post_commit"] = settlement
                    execution = store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
                    if execution:
                        row["attempt_count"] = len(execution["attempts"])
                        row["tools"] = [{
                            "tool_id": item["tool_id"], "status": item["status"],
                            "tool_call_id": item["tool_call_id"],
                            "arguments": json.loads(item["arguments_json"]),
                            "error": json.loads(item["outcome_json"] or "{}").get("error"),
                        } for item in execution["tool_calls"]]
                        trace = {
                            "case_id": case.id,
                            "tools": [{**summary, "outcome": json.loads(item["outcome_json"] or "{}")}
                                      for summary, item in zip(row["tools"], execution["tool_calls"])],
                            "attempts": [{key: item.get(key) for key in (
                                "attempt_id", "attempt_index", "status", "action_kind", "decision_json",
                            )} for item in execution["attempts"]],
                        }
                        trace_path = output.with_name(case.id + ".trace.json")
                        save(trace_path, trace)
                        row["evidence_trace_file"] = trace_path.name
                    row["chain_complete"] = chain_succeeded(row, settlement)
            finally:
                maintenance.stop(timeout_seconds=30)
        # Literal coverage is a review aid, never an automatic accuracy score.
        row["literal_fact_coverage"] = {
            fact: fact in (row["reply"] or "") for fact in case.expected_facts
        }
    except Exception as exc:
        row.update({
            "status": "harness_error", "exception_type": type(exc).__name__,
            "error_code": getattr(exc, "code", "worker_failed"),
        })
    row["elapsed_s"] = round(time.monotonic() - started, 3)
    save(output, row)
    return 0 if row["chain_complete"] else 1


def write_report(directory: Path, report: dict) -> None:
    save(directory / "results.json", report)
    lines = ["# 多文档真实模型测试", "", f"运行：{report['run_id']}", "",
             "每题使用独立进程、Project 和 Session。参考答案不挂载给模型；来源与套件身份单独记录。",
             "工程完成状态与答案正确性分开；字面事实覆盖仅辅助人工检查，不作为正确率。", "",
             "## API 探测", "", "```json", json.dumps(report["probes"], ensure_ascii=False, indent=2), "```", ""]
    for row in report["cases"]:
        lines += [f"## {row['case_id']}：{row['scenario']}", "",
                  f"问题：{row['question']}", "",
                  f"状态：{row['status']}；完整链路：{row.get('chain_complete', False)}；耗时：{row.get('elapsed_s', 0)} 秒", "",
                  "### 模型回复", "", row.get("reply") or "（未生成回复）", "",
                  "### 人工复核参考", "", row["reference"], ""]
        if row.get("input_statistics"):
            lines += ["输入规模：`" + json.dumps(row["input_statistics"], ensure_ascii=False) + "`", ""]
        if row.get("evidence_trace_file"):
            lines += [f"[工具参数、返回证据与模型引用轨迹]({row['evidence_trace_file']})", ""]
        if row.get("error_code"):
            lines += [f"错误：`{row['error_code']}`", ""]
        if row.get("tools"):
            lines += ["工具：" + " → ".join(f"{t['tool_id']} ({t['status']})" for t in row["tools"]), ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def question_message(case: Case) -> str:
    if case.suite == "docbench":
        return ("Answer using only the four attached PDFs. Identify the relevant document and "
                "cite its filename and physical PDF page numbers. Do not use external knowledge.\n\n"
                + case.question)
    return "请仅依据本轮提供的附件回答，不使用网络或外部知识。\n" + case.question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--suite", choices=(*SUITES, "docbench"), default="pdf")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--data-root", type=Path, help="Original DocBench data directory (docbench suite only)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.list and not args.allow_live:
        parser.error("real API requests require --allow-live")
    if args.suite == "docbench":
        if args.data_root is None:
            parser.error("docbench requires --data-root")
        from .docbench import load_cases

        available = load_cases(args.data_root)
        defaults = available
    else:
        if args.data_root is not None:
            parser.error("--data-root is only valid for docbench")
        available, defaults = ALL_CASES, SUITES[args.suite]
    by_id = {case.id: case for case in available}
    requested = [args.worker] if args.worker else args.cases
    if requested and (len(requested) != len(set(requested)) or any(key not in by_id for key in requested)):
        parser.error("unknown or duplicate case ID for this suite")
    selected = tuple(by_id[key] for key in requested) if requested else defaults
    if args.list:
        for case in selected:
            print(f"{case.id}: {case.scenario} {input_statistics(case)}")
        return 0
    if args.worker:
        return worker(selected[0], args.case_root, args.output)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    directory = HERE / "runs" / run_id
    directory.mkdir(parents=True)
    private_root = Path(tempfile.mkdtemp(prefix=f"entelecheia-multi-doc-{run_id}-"))
    os.environ.update(isolated_environment(private_root / "probe", private_root))
    from personagraph.configuration.paths import load_dotenv
    from .probe import run_probes

    load_dotenv()
    probes = run_probes(private_root)
    source = hashlib.sha256()
    for path in sorted(list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))):
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    report = {"run_id": run_id, "probes": probes, "private_state_root": str(private_root),
              "source_sha256": source.hexdigest(), "suite": args.suite,
              "case_count": len(selected), "status": "running", "cases": []}
    if args.suite == "docbench":
        from .docbench import SELECTION

        report["selection_sha256"] = hashlib.sha256(SELECTION.read_bytes()).hexdigest()
        report["official_comparable"] = False
    write_report(directory, report)
    print(json.dumps({"probes": probes, "results": str(directory)}, ensure_ascii=False), flush=True)
    for case in selected:
        blocked = not probes["task"]["ok"] or (case.needs_vision and not probes["vision"]["ok"])
        if blocked:
            row = {
                "case_id": case.id, "scenario": case.scenario, "question": case.question,
                "reference": case.reference, "status": "blocked_preflight",
                "error_code": "vision_probe_failed" if probes["task"]["ok"] else "task_probe_failed",
                "chain_complete": False,
            }
        else:
            case_root = private_root / case.id
            case_root.mkdir()
            output = directory / f"{case.id}.json"
            print(f"START {case.id}", flush=True)
            command = [sys.executable, "-m", "evals.multi_doc.run", "--allow-live",
                       "--suite", args.suite, "--worker", case.id,
                       "--case-root", str(case_root), "--output", str(output)]
            if args.data_root is not None:
                command += ["--data-root", str(args.data_root.resolve())]
            with (directory / f"{case.id}.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.run(
                        command, cwd=ROOT, env=isolated_environment(case_root, private_root),
                        stdout=log, stderr=subprocess.STDOUT,
                        timeout=features_for(case)["turn_wall_clock_budget_s"] + 240, check=False,
                    )
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 124
            row = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
                "case_id": case.id, "scenario": case.scenario, "question": case.question,
                "reference": case.reference, "status": "worker_exited_without_result",
                "chain_complete": False,
            }
            row["exit_code"] = exit_code
            if exit_code == 124:
                row.update(status="worker_timeout", chain_complete=False)
            elif exit_code != 0:
                row["chain_complete"] = False
        report["cases"].append(row)
        write_report(directory, report)
        print(f"END {case.id}: {row['status']} chain_complete={row['chain_complete']}", flush=True)
    succeeded = all(row["chain_complete"] for row in report["cases"])
    report.update(status="completed" if succeeded else "completed_with_failures",
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_report(directory, report)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
