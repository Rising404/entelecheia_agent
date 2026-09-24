"""Gold-conditioned CPU dense smoke; not a retrieval benchmark score."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

DEFAULT_CASE_IDS = ['docbench:51:3', 'docbench:53:3', 'docbench:54:4', 'docbench:55:2',
                    'docbench:61:6', 'docbench:66:3', 'docbench:80:5', 'docbench:85:0', 'docbench:88:9']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--case-id', action='append')
    ap.add_argument('--max-length', default=4096, type=int)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--max-units', default=80, type=int)
    ap.add_argument('--distractors-per-doc', default=4, type=int)
    ap.add_argument('--batch-size', default=2, type=int)
    ap.add_argument('--threads', default=4, type=int)
    ap.add_argument('--time-budget-seconds', default=600, type=int)
    args = ap.parse_args()
    if not (1 <= args.max_units <= 80 and 1 <= args.batch_size <= 4 and 1 <= args.threads <= 4
            and 1 <= args.max_length <= 4096 and args.distractors_per_doc >= 0):
        raise ValueError('Smoke bounds: <=80 units, <=4096 tokens, <=4 batch size/threads')
    from evals.docbench_hybrid_retrieval_optimize.paths import artifact_path
    dataset, output = artifact_path(args.dataset), artifact_path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest_path = dataset.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dataset_hash = sha256(dataset.read_bytes()).hexdigest()
    if manifest['dataset_sha256'] != dataset_hash:
        raise ValueError('dataset manifest hash mismatch')
    wal = Path(str(dataset) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Dataset has an uncheckpointed WAL')
    from evals.docbench_hybrid_retrieval_optimize.retrieval_eval import QUESTION_KINDS
    ids = args.case_id or DEFAULT_CASE_IDS
    if len(ids) != len(set(ids)) or len(ids) != 9:
        raise ValueError('This smoke expects exactly nine unique repaired cases')
    with sqlite3.connect(f'{dataset.as_uri()}?mode=ro&immutable=1', uri=True) as c:
        c.row_factory = sqlite3.Row
        queries = []
        primary = {}
        pools = {}
        units = {}
        for case_id in ids:
            query = c.execute('SELECT case_id,doc_id,question_type,query,annotation_status FROM queries WHERE case_id=?', (case_id,)).fetchone()
            if query is None or query['annotation_status'] != 'reviewed':
                raise ValueError(f'Case is not reviewed: {case_id}')
            query = dict(query)
            gold = {r[0] for r in c.execute("SELECT unit_id FROM qrels WHERE case_id=? AND role='primary'", (case_id,))}
            if not gold:
                raise ValueError(f'No primary labels: {case_id}')
            primary[case_id] = gold
            kinds = QUESTION_KINDS[query['question_type']]
            candidates = [dict(r) for r in c.execute('SELECT unit_id,doc_id,kind,content,source_revision,content_sha256,asset_path,metadata_json FROM units WHERE doc_id=? ORDER BY unit_id', (query['doc_id'],)) if r['kind'] in kinds]
            candidate_ids = {r['unit_id'] for r in candidates}
            if not gold <= candidate_ids:
                raise ValueError(f'Gold outside candidate scope: {case_id}')
            units.update({r['unit_id']: r for r in candidates})
            pools[case_id] = [r['unit_id'] for r in candidates if r['unit_id'] not in gold]
            query['full_document_candidate_count'] = len(candidates)
            queries.append(query)
    selected = set().union(*primary.values())
    if len(selected) > args.max_units:
        raise ValueError('All primary units alone exceed the smoke budget')
    for offset in range(args.distractors_per_doc):
        for case_id in ids:
            if offset < len(pools[case_id]) and len(selected) < args.max_units:
                selected.add(pools[case_id][offset])
    selected_ids = sorted(selected)
    selected_units = [units[unit_id] for unit_id in selected_ids]
    for unit in selected_units:
        if sha256(unit['content'].encode()).hexdigest() != unit['content_sha256']:
            raise ValueError(f'Content hash mismatch: {unit["unit_id"]}')
        if unit['asset_path']:
            asset = (dataset.parent / unit['asset_path']).resolve()
            if not asset.is_relative_to(dataset.parent) or not asset.is_file():
                raise ValueError('Invalid selected asset')
            if sha256(asset.read_bytes()).hexdigest() != json.loads(unit['metadata_json'])['asset_sha256']:
                raise ValueError('Selected asset hash mismatch')
    print(json.dumps({'stage': 'subset_selected', 'queries': len(queries), 'units': len(selected_units),
                      'primary_units': len(set().union(*primary.values())), 'device': 'cpu',
                      'max_length': args.max_length, 'formal_recall': False}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181', local_files_only=True, trust_remote_code=False)
    texts = [u['content'] for u in selected_units] + [q['query'] for q in queries]
    tokens_with_specials = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False, return_length=True)['length']
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_counts = [count-special_tokens for count in tokens_with_specials]
    true_max = max(tokens_with_specials)
    if true_max > args.max_length:
        raise ValueError(f'Selected text needs {true_max} tokens, exceeds smoke cap {args.max_length}; no silent truncation')
    args.max_length = true_max
    print(json.dumps({'stage': 'token_preflight', 'dataset_sha256': dataset_hash, 'annotation_counts': manifest.get('annotation_counts'), 'units': len(selected_units), 'queries': len(queries), 'unit_max_tokens_with_specials': max(tokens_with_specials[:len(selected_units)]), 'query_max_tokens_with_specials': max(tokens_with_specials[len(selected_units):]), 'max_length': args.max_length, 'truncated_inputs': 0}), flush=True)
    if args.preflight_only:
        return 0
    import numpy as np
    import torch
    from personagraph.retrieval.indexing.encoder import BgeM3Encoder
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    encoder = BgeM3Encoder(device='cpu', local_files_only=True, use_fp16=False,
                          query_max_length=args.max_length, passage_max_length=args.max_length)
    # Probe the longest inputs first, making the throughput estimate conservative.
    order = sorted(range(len(texts)), key=lambda i: (-min(token_counts[i], args.max_length), i))
    encoded = {}
    batch_durations = []
    stop_reason = None
    for start in range(0, len(order), args.batch_size):
        positions = order[start:start + args.batch_size]
        batch_started = time.perf_counter()
        values = encoder.encode([texts[i] for i in positions])
        elapsed = time.perf_counter() - batch_started
        batch_durations.append(elapsed)
        for position, value in zip(positions, values, strict=True):
            vector = np.asarray(value.dense_vector, dtype=np.float32)
            if vector.shape != (1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
                raise ValueError('Invalid dense vector')
            encoded[position] = vector / np.linalg.norm(vector)
        remaining = len(texts) - len(encoded)
        projected = ((time.perf_counter() - started) + remaining * elapsed / len(positions)) if start else None
        print(json.dumps({'stage': 'encoded_batch', 'completed': len(encoded), 'total': len(texts),
                          'batch_seconds': round(elapsed, 3), 'projected_total_seconds': round(projected, 2) if projected is not None else None}), flush=True)
        if remaining and ((projected is not None and projected > args.time_budget_seconds) or time.perf_counter() - started > args.time_budget_seconds):
            stop_reason = 'projected_cpu_time_exceeds_smoke_budget'
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temp:
        stage = Path(temp)
        write_jsonl(stage / 'units.jsonl', [{'unit_id': u['unit_id'], 'doc_id': u['doc_id'], 'kind': u['kind'],
                    'text': u['content'], 'source_revision': u['source_revision'], 'content_sha256': u['content_sha256'],
                    'asset_path': u['asset_path'], 'tokens_without_specials': token_counts[i],
                    'tokens_with_specials': tokens_with_specials[i],
                    'may_be_truncated': tokens_with_specials[i] > args.max_length} for i, u in enumerate(selected_units)])
        write_jsonl(stage / 'queries.jsonl', [dict(q, tokens_with_specials=tokens_with_specials[len(selected_units)+i], may_be_truncated=False) for i,q in enumerate(queries)])
        write_jsonl(stage / 'subset_qrels.jsonl', [{'case_id': case_id, 'unit_id': uid, 'grade': 1}
                    for case_id in ids for uid in sorted(primary[case_id])])
        unit_positions = [i for i in range(len(selected_units)) if i in encoded]
        query_positions = [i for i in range(len(selected_units), len(texts)) if i in encoded]
        np.savez_compressed(stage / 'unit_vectors.npz', unit_ids=np.asarray([selected_ids[i] for i in unit_positions]),
                            vectors=np.stack([encoded[i] for i in unit_positions]) if unit_positions else np.empty((0, 1024), dtype=np.float32))
        np.savez_compressed(stage / 'query_vectors.npz', case_ids=np.asarray([queries[i-len(selected_units)]['case_id'] for i in query_positions]),
                            vectors=np.stack([encoded[i] for i in query_positions]) if query_positions else np.empty((0, 1024), dtype=np.float32))
        rankings = []
        if stop_reason is None:
            for qi, query in enumerate(queries):
                scored = [(u['unit_id'], float(np.dot(encoded[len(selected_units)+qi], encoded[ui])))
                          for ui, u in enumerate(selected_units) if u['doc_id'] == query['doc_id'] and u['kind'] in QUESTION_KINDS[query['question_type']]]
                scored.sort(key=lambda r: (-r[1], r[0]))
                rankings.append({'case_id': query['case_id'], 'subset_candidate_count': len(scored),
                                 'full_document_candidate_count': query['full_document_candidate_count'],
                                 'first_primary_rank': next((rank for rank, (uid, _) in enumerate(scored, 1) if uid in primary[query['case_id']]), None),
                                 'hits': [{'unit_id': uid, 'cosine': score, 'rank': rank} for rank, (uid, score) in enumerate(scored, 1)]})
        write_jsonl(stage / 'rankings.jsonl', rankings)
        report = {'status': 'complete' if stop_reason is None else 'stopped_budget', 'stop_reason': stop_reason,
                  'purpose': 'gold-conditioned CPU dense semantic smoke, NOT formal Recall or production hybrid evaluation',
                  'route': 'canonical_bge_m3_cpu_dense_smoke', 'dataset_sha256': dataset_hash,
                  'manifest_sha256': sha256(manifest_path.read_bytes()).hexdigest(),
                  'model': encoder.diagnostic_snapshot(), 'device': 'cpu', 'fp16': False, 'threads': args.threads,
                  'query_max_length': args.max_length, 'passage_max_length': args.max_length,
                  'truncated_inputs': 0, 'max_input_tokens_with_specials': true_max, 'special_tokens_per_sequence': special_tokens,
                  'batch_size': args.batch_size, 'elapsed_seconds': time.perf_counter()-started,
                  'batch_seconds': batch_durations, 'units': len(selected_units), 'queries': len(queries),
                  'encoded_units': len(unit_positions), 'encoded_queries': len(query_positions),
                  'selection': f'all reviewed primary units, plus <={args.distractors_per_doc} lexicographically first same-document same-modality distractors per case; max {args.max_units} units',
                  'case_ids': ids, 'gold_conditioned_candidate_pool': True,
                  'model_input_fields': ['units.content', 'queries.query'],
                  'unused_sparse_computed': encoder.learned_sparse_available,
                  'scoring': 'L2-normalized dense cosine only; no sparse/BM25/reranker/LLM; no aggregate benchmark metrics',
                  'network': 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; local_files_only=True',
                  'files_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}}
        if sha256(dataset.read_bytes()).hexdigest() != dataset_hash:
            raise ValueError('Dataset changed during smoke')
        write_json(stage / 'report.json', report)
        (stage / 'README.md').write_text('# CPU semantic smoke\n\nThis small pool deliberately includes all selected gold evidence and a few deterministic distractors. It checks whether repaired text representations produce usable dense vectors and query rankings. It is not a full-corpus Recall score or production hybrid result. No answer, evidence, annotation rationale, or old question-conditioned embedding is passed to the encoder.\n\nThe unchanged canonical encoder may compute unused learned-sparse outputs; only normalized 1024-dimensional dense vectors are stored and scored. The max_length was set from exact local tokenization including special tokens to retain every selected input without truncation. Per-input token counts and zero-truncation flags are recorded.\n')
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(json.dumps({'stage': 'finished', 'status': report['status'], 'output': str(output),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return 0 if stop_reason is None else 3


if __name__ == '__main__':
    raise SystemExit(main())
