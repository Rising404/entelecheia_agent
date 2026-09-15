"""Export reviewed per-case evidence without modifying the private archive.

This is a publication projection, not a runnable state backup or a new score.
Only the standard library is needed; no model, database or network is opened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


SOURCE_TEXT_KEYS = frozenset({"text", "content", "snippet", "text_snippet", "text_preview"})
PRIVATE_INPUT_KEYS = frozenset({"current_user_text", "submitted_prompt", "user_text"})
CREDENTIAL_KEYS = frozenset({"api_key", "access_token", "authorization", "password", "secret"})
LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:/(?:Users|home|Volumes|private|tmp|var|mnt|workspace)/"
    r"[^\s\"'<>]*|[A-Za-z]:[\\/]Users[\\/][^\s\"'<>]*)"
)


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def omitted(value: object, reason: str) -> dict:
    raw = value.encode() if isinstance(value, str) else encode(value)
    return {"publication_omitted": reason, "source_sha256": digest(raw), "source_bytes": len(raw)}


def clean(value: object, *, source_text: bool = False) -> object:
    """Keep errors, identities and generated observations; mark excluded source bodies."""
    if isinstance(value, list):
        return [clean(item, source_text=source_text) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_INPUT_KEYS and item:
                result[key] = omitted(item, "benchmark_input")
            elif key.casefold() in CREDENTIAL_KEYS and item and not isinstance(item, (dict, list)):
                result[key] = omitted(item, "credential")
            elif source_text and key in SOURCE_TEXT_KEYS and isinstance(item, str) and item:
                result[key] = omitted(item, "document_text")
            elif source_text and key == "value" and isinstance(item, str) and item:
                # read_tool_result can expand a document text leaf without its key.
                result[key] = omitted(item, "expanded_text_leaf")
            else:
                result[key] = clean(item, source_text=source_text)
        return result
    if isinstance(value, str):
        return LOCAL_PATH.sub("<local-path>", value)
    return value


def publication(raw: bytes, *, source_relative_path: str) -> dict:
    return {
        "kind": "reviewed_public_projection",
        "source_relative_path": source_relative_path,
        "source_sha256": digest(raw),
        "source_bytes": len(raw),
        "original_metrics_preserved": True,
        "not_a_raw_or_replayable_copy": True,
    }


def export_trajectory(raw: bytes, source_relative_path: str) -> dict:
    source = json.loads(raw)
    result = deepcopy(source)
    blobs = {}
    for step in result["steps"]:
        for part in step.get("parts", []):
            original_hash = part["blob_sha256"]
            original = source["blobs"][original_hash]
            body = original["text"]
            assert digest(body.encode()) == original_hash, "invalid source trajectory blob"
            role = part["role"]
            if role in {"user", "tool_result", "tool_arguments"}:
                try:
                    parsed = json.loads(body)
                except ValueError:
                    projected = omitted(body, "unstructured_input_or_tool_body")
                else:
                    projected = clean(parsed, source_text=role in {"user", "tool_result"})
                body = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
            else:
                body = clean(body)
            new_hash = digest(body.encode())
            blob = deepcopy(original)
            blob["text"] = body
            # Export metadata must describe the new bytes, not the original text.
            for key in ("byte_count", "size_bytes", "utf8_bytes"):
                if key in blob:
                    blob[key] = len(body.encode())
            blobs[new_hash] = blob
            part["blob_sha256"] = new_hash
            if new_hash != original_hash:
                part["source_blob_sha256"] = original_hash
    result["blobs"] = blobs
    result["integrity"]["blob_count"] = len(blobs)
    result["publication"] = publication(raw, source_relative_path=source_relative_path)
    return clean(result)


def export_archive(archive: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    output = output.resolve()
    if output == archive or output.is_relative_to(archive):
        raise ValueError("public output must be outside the private archive")
    if output.exists():
        raise ValueError("public output already exists; originals and exports are never overwritten")
    payloads: dict[str, bytes] = {}
    runs = []
    for run in sorted((archive / "runs").iterdir()):
        if not run.is_dir():
            continue
        run_prefix = f"runs/{run.name}"
        scores = {}
        for score_path in sorted((run / "scoring/cases").glob("*.json")):
            raw = score_path.read_bytes()
            score = json.loads(raw)
            case_id = score["case_id"]
            assert case_id not in scores
            public_score = {k: v for k, v in score.items() if k not in {
                "question", "reference_answer", "reference_text",
            }}
            public_score["publication"] = publication(raw, source_relative_path=str(score_path.relative_to(archive)))
            scores[case_id] = public_score
            payloads[f"{run_prefix}/scoring/cases/{case_id.replace(':', '-')}.json"] = encode(clean(public_score))
        count = 0
        for result_path in sorted((run / "cases").glob("*/result.json")):
            raw = result_path.read_bytes()
            result = json.loads(raw)
            case_id = result["case_id"]
            assert re.fullmatch(r"docbench:\d+:\d+", case_id)
            assert case_id in scores, f"missing scoring for {case_id}"
            case_prefix = f"cases/{case_id.replace(':', '-')}"
            trajectory_path = run / result["trajectory"]["path"]
            assert trajectory_path.resolve().is_relative_to(run.resolve())
            trajectory_raw = trajectory_path.read_bytes()
            assert digest(trajectory_raw) == result["trajectory"]["sha256"]
            trajectory = export_trajectory(trajectory_raw, str(trajectory_path.relative_to(archive)))
            trajectory_bytes = encode(trajectory)
            payloads[f"{run_prefix}/{case_prefix}/trajectory.json"] = trajectory_bytes
            projected = {k: v for k, v in result.items() if k not in {
                "question", "reference_answer", "evidence", "submitted_prompt",
                "session_working_dir_locator",
            }}
            projected = clean(projected)
            projected["publication"] = publication(raw, source_relative_path=str(result_path.relative_to(archive)))
            projected["trajectory"] = {
                **projected["trajectory"],
                "path": f"{case_prefix}/trajectory.json",
                "sha256": digest(trajectory_bytes),
                "byte_count": len(trajectory_bytes),
                "blob_count": len(trajectory["blobs"]),
            }
            payloads[f"{run_prefix}/{case_prefix}/result.json"] = encode(projected)
            count += 1
        assert count == len(scores), "score/result case sets differ"
        summary_raw = (run / "scoring/summary.json").read_bytes()
        summary = json.loads(summary_raw)
        summary["cases"] = [{
            "case_id": cid, "score": score["score"], "status": score["status"],
            "path": f"cases/{cid.replace(':', '-')}.json",
        } for cid, score in sorted(scores.items())]
        assert summary["case_count"] == count
        assert summary["correct_count"] == sum(s["score"] or 0 for s in scores.values())
        summary["publication"] = publication(summary_raw, source_relative_path=str((run / "scoring/summary.json").relative_to(archive)))
        payloads[f"{run_prefix}/scoring/summary.json"] = encode(clean(summary))
        manifest_raw = (run / "run_manifest.json").read_bytes()
        original = json.loads(manifest_raw)
        manifest = {k: original[k] for k in (
            "schema_version", "benchmark_id", "lane", "run_id", "created_at",
            "config_sha256", "selection_sha256", "providers", "provenance", "frozen_cases_sha256",
        ) if k in original}
        manifest["publication"] = publication(manifest_raw, source_relative_path=str((run / "run_manifest.json").relative_to(archive)))
        payloads[f"{run_prefix}/run_manifest.json"] = encode(clean(manifest))
        runs.append({"directory": run.name, "run_id": original["run_id"], "executions": count,
                     "judge_correct": summary["correct_count"]})
    if not runs:
        raise ValueError("archive has no runs")
    payloads["README.md"] = (
        "# 逐题公开证据 / Public per-case evidence\n\n"
        "这是原始档案的公开投影，不是原样副本、模型重跑或重新评分。\n\n"
        "- `runs/<batch>/cases/<case>/result.json`：回答、状态、统计及轨迹引用。\n"
        "- 同题 `trajectory.json`：全部已记录步骤、调用参数、返回及单次指标；正文仍通过 blobs 关联。\n"
        "- `runs/<batch>/scoring/`：原裁判逐题记录与汇总，裁判 token 与生成阶段分开。\n"
        "- `run_manifest.json`：原运行身份、模型及配置/源码哈希，不含完整本机配置。\n\n"
        "仅省略原题/参考答案字段、原始用户请求、文档 text/content/snippet 和无法判别来源的展开文本叶；\n"
        "这些位置保留 publication_omitted、原字节数与 SHA-256。模型生成的回答、视觉观察、\n"
        "文件 ID、工具参数、错误和性能指标保留；私有绝对路径替换为 <local-path>，凭据不公开。\n"
        "模型上下文不是原样请求，不能按公开文本重算原 token 或直接重放；已有 assistant 摘要不补写。\n"
        "模型回答/观察可能包含引文，未另行将源 PDF 或 QA 数据集打包。来源身份用于追溯，不授予第三方版权。\n\n"
        "`publication.source_sha256` 指向私有原件；result.trajectory.sha256 指向这里的公开轨迹，\n"
        "parts 的 source_blob_sha256 保留被修改前的正文身份。原始 incomplete、失败和零分全部保留。\n"
        "路径中的冒号改为连字符便于跨平台 checkout；JSON 内的 case_id 不变。\n\n"
        "## 批次 / Runs\n\n" + "\n".join(
            f"- `{r['run_id']}`：{r['executions']} 次执行，原裁判 {r['judge_correct']}/{r['executions']}。"
            for r in runs
        ) + "\n\nDocBench prompt-compatible、非官方可比成绩；补跑不覆盖首跑，不能直接相加作为独立题目准确率。\n"
    ).encode()
    manifest = {
        "kind": "reviewed_public_evidence", "benchmark_id": "docbench", "runs": runs,
        "files": {p: {"sha256": digest(b), "bytes": len(b)} for p, b in sorted(payloads.items())},
    }
    output.mkdir(parents=True)
    for path, raw in payloads.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (output / "publication_manifest.json").write_bytes(encode(manifest))
    return {"executions": sum(r["executions"] for r in runs), "files": len(payloads) + 1,
            "bytes": sum(map(len, payloads.values())), "manifest_sha256": digest(encode(manifest))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_archive(args.archive, args.output)))


if __name__ == "__main__":
    main()
