"""Freeze a retrieval dataset from explicitly ordered, completed DocBench runs.

This boundary reads historical storage without migrating it. Historical question-driven
vision observations are deliberately not corpus inputs. Gold annotations remain a
separate, reviewable operation; automatic evidence matches are proposals only.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Iterator, Sequence

from .paths import PROJECT_ROOT as PROJECT_ROOT, artifact_path


class RetrievalDatasetError(ValueError):
    """A source or annotation cannot safely participate in this dataset."""


TEXT_UNIT_KINDS = frozenset({'chunk'})
VISUAL_UNIT_KINDS = frozenset({'table', 'figure', 'vector_graphics'})
RETRIEVAL_UNIT_KINDS = TEXT_UNIT_KINDS | VISUAL_UNIT_KINDS


SCHEMA = """
CREATE TABLE documents (
 doc_id TEXT PRIMARY KEY, pdf_path TEXT NOT NULL, pdf_sha256 TEXT NOT NULL,
 source_db TEXT NOT NULL, source_db_sha256 TEXT NOT NULL, source_run_id TEXT NOT NULL,
 source_document_id TEXT NOT NULL, source_version_id TEXT NOT NULL,
 processor_fingerprint TEXT NOT NULL, chunker_fingerprint TEXT NOT NULL,
 page_count INTEGER NOT NULL, processing_status TEXT NOT NULL, manifest_json TEXT NOT NULL
);
CREATE TABLE units (
 unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
 kind TEXT NOT NULL CHECK(kind IN ('chunk','table','figure','vector_graphics')), content TEXT NOT NULL,
 source_unit_id TEXT NOT NULL, source_revision TEXT NOT NULL, pages_json TEXT NOT NULL,
 locator_json TEXT NOT NULL, asset_path TEXT, content_sha256 TEXT NOT NULL,
 description_method TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE INDEX units_document_kind ON units(doc_id,kind);
CREATE TABLE queries (
 case_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
 question_type TEXT NOT NULL, query TEXT NOT NULL, answer TEXT NOT NULL, evidence TEXT NOT NULL,
 annotation_status TEXT NOT NULL DEFAULT 'pending' CHECK(annotation_status IN ('pending','reviewed')),
 annotation_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE qrels (
 case_id TEXT NOT NULL REFERENCES queries(case_id), unit_id TEXT NOT NULL REFERENCES units(unit_id),
 role TEXT NOT NULL CHECK(role IN ('primary','support')), provenance TEXT NOT NULL,
 reviewer TEXT NOT NULL, PRIMARY KEY(case_id,unit_id)
);
CREATE TABLE curation_events (
 event_id INTEGER PRIMARY KEY, target_type TEXT NOT NULL, target_id TEXT NOT NULL,
 field TEXT NOT NULL, original_value TEXT, revised_value TEXT NOT NULL,
 reviewer TEXT NOT NULL, note TEXT NOT NULL, source_json TEXT NOT NULL
);
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _external(path: Path) -> Path:
    try:
        return artifact_path(path)
    except ValueError as error:
        raise RetrievalDatasetError(str(error)) from error


@contextmanager
def _historical_database(path: Path) -> Iterator[sqlite3.Connection]:
    # Copy WAL as well, if present. Never ask SQLite to create SHM beside a source.
    # A changing source fails closed instead of freezing an incoherent DB/WAL pair.
    with tempfile.TemporaryDirectory(prefix='docbench-retrieval-read-') as temp:
        destination = Path(temp) / 'source.sqlite'
        identities = {}
        for suffix in ('', '-wal'):
            source = Path(str(path) + suffix)
            if source.exists():
                identities[suffix] = _sha(source)
                shutil.copyfile(source, Path(str(destination) + suffix))
        if any(_sha(Path(str(path) + suffix)) != digest for suffix, digest in identities.items()):
            raise RetrievalDatasetError(f'Historical database changed during export: {path}')
        with sqlite3.connect(destination) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA query_only=ON')
            yield connection


def _find_document(runs: Sequence[Path], case: dict) -> tuple[Path, Path, dict, dict, list, list]:
    for run in runs:
        candidates = sorted((run / 'cases' / case['case_id'] / 'state' / 'projects').glob('*/documents.sqlite'))
        for database in candidates:
            with _historical_database(database) as connection:
                records = connection.execute(
                    'SELECT d.id AS document_id,d.path,v.* FROM documents d '
                    'JOIN document_versions v ON v.id=d.current_version_id '
                    'WHERE v.source_sha256=?', (case['pdf_sha256'],)
                ).fetchall()
                if len(records) > 1:
                    raise RetrievalDatasetError(f'Ambiguous source versions for {case["case_id"]}')
                if not records:
                    continue
                version = dict(records[0])
                chunks = [dict(row) for row in connection.execute(
                    'SELECT * FROM doc_chunks WHERE source_version_id=? ORDER BY seq,id',
                    (version['id'],),
                )]
                if not chunks:
                    continue
                manifest = json.loads(version['page_manifest_json'])
                elements = json.loads(version['source_elements_json'])
                return run, database, version, manifest, chunks, elements
    raise RetrievalDatasetError(f'No matching populated source version for {case["case_id"]}')


def _visual_description(unit: dict, by_id: dict) -> str:
    """Describe only text spatially inside the exported image or its own element."""
    own_ids = [unit.get('element_id'), *unit.get('text_element_ids', [])]
    own = [by_id[element_id]['content'] for element_id in own_ids if element_id in by_id]
    # Historical source_elements often retain only a string locator, so ordinal
    # adjacency cannot establish that text actually belongs inside a visual crop.
    text = unit['region_text'].strip()
    if not text:
        text = '\n'.join(dict.fromkeys(own))
    return f"{unit['kind']} on page {unit['locator']['page']}\n{text}".strip()


def _canonical_visual_units(units: list[dict], by_id: dict) -> list[dict]:
    """Collapse only exact same-kind, same-page pixels and text representations."""
    canonical = {}
    for unit in sorted(units, key=lambda value: value['unit_id']):
        content = _visual_description(unit, by_id)
        key = (unit['kind'], unit['locator']['page'], tuple(unit['export_locator']['render_pixel_box']), content)
        if key in canonical:
            canonical[key]['source_unit_aliases'].append(unit['unit_id'])
        else:
            canonical[key] = {**unit, 'description': content, 'source_unit_aliases': []}
    return list(canonical.values())


def _visual_bbox(unit: dict) -> tuple[float, float, float, float] | None:
    raw = unit['locator'].get('bbox')
    if raw is None:
        return None
    if len(raw) != 4:
        raise RetrievalDatasetError(f'Invalid visual geometry: {unit["unit_id"]}')
    bbox = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise RetrievalDatasetError(f'Invalid visual geometry: {unit["unit_id"]}')
    if unit['kind'] != 'vector_graphics' and (bbox[2] == bbox[0] or bbox[3] == bbox[1]):
        raise RetrievalDatasetError(f'Invalid visual geometry: {unit["unit_id"]}')
    return bbox


def _visual_context_bbox(unit: dict, units: list[dict], width: float, height: float) -> tuple[tuple, list[str], str]:
    """Use document geometry, never the benchmark query, to preserve visual context."""
    bbox = _visual_bbox(unit)
    associated_ids = [unit['unit_id']]
    if bbox is None:
        return (0.0, 0.0, width, height), associated_ids, 'page_no_bbox'
    left, top, right, bottom = bbox
    area = max(0.0, min(width, right) - max(0.0, left)) * max(0.0, min(height, bottom) - max(0.0, top))
    if unit['kind'] == 'vector_graphics' and area >= width * height * 0.6:
        return (0.0, 0.0, width, height), associated_ids, 'page_vector_context'
    if unit['kind'] == 'table':
        # Native PDF detectors frequently emit each shaded row as a separate table.
        # Adjacent aligned strips share a context image, including headers above it.
        remaining = [other for other in units if other['kind'] == 'table' and other['unit_id'] != unit['unit_id']]
        changed = True
        while changed:
            changed = False
            for other in remaining[:]:
                other_bbox = _visual_bbox(other)
                if other_bbox is None:
                    continue
                x0, y0, x1, y1 = other_bbox
                overlap = max(0.0, min(right, x1) - max(left, x0))
                gap = max(0.0, y0 - bottom, top - y1)
                if overlap >= 0.7 * min(right - left, x1 - x0) and gap <= 48.0:
                    left, top, right, bottom = min(left, x0), min(top, y0), max(right, x1), max(bottom, y1)
                    associated_ids.append(other['unit_id'])
                    remaining.remove(other)
                    changed = True
        horizontal_margin = max(16.0, width * 0.12) if right - left < width * 0.4 else 16.0
        margins = (horizontal_margin, 64.0, horizontal_margin, 32.0)
        extent = 'region_table_context'
    else:
        margins = (32.0, 32.0, 32.0, 48.0)
        # Native PDF rules have legitimate zero-width/height bounding boxes.
        # Preserve the source geometry and render their surrounding pixels.
        extent = ('region_vector_line_context' if right == left or bottom == top
                  else 'region_visual_context')
    bounds = (max(0.0, left - margins[0]), max(0.0, top - margins[1]),
              min(width, right + margins[2]), min(height, bottom + margins[3]))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise RetrievalDatasetError(f'Invalid visual geometry: {unit["unit_id"]}')
    return bounds, sorted(associated_ids), extent


def _render_visual_page(pdf, page_number: int, units: list[dict], output: Path, doc_id: str) -> list[dict]:
    for unit in units:
        unit_id = unit.get('unit_id')
        if not isinstance(unit_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', unit_id):
            raise RetrievalDatasetError(f'Unsafe visual unit id: {unit_id!r}')
    page = pdf[page_number - 1]
    try:
        width, height = page.get_size()
        cropbox = page.get_cropbox()
        needs_page_context = page.get_rotation() != 0 or abs(cropbox[0]) > 0.01 or abs(cropbox[1]) > 0.01
        scale = min(2.0, 1800.0 / max(width, height))
        bitmap = page.render(scale=scale)
        try:
            image = bitmap.to_pil().convert('RGB')
        finally:
            bitmap.close()
        text_page = page.get_textpage()
        try:
            results = []
            cached_regions = {}
            for unit in units:
                locator = dict(unit['locator'])
                bounds, associated_ids, extent = _visual_context_bbox(unit, units, width, height)
                if needs_page_context:
                    bounds, extent = (0.0, 0.0, width, height), 'page_transformed_geometry'
                box = (max(0, math.floor(bounds[0] * scale)), max(0, math.floor(bounds[1] * scale)),
                       min(image.width, math.ceil(bounds[2] * scale)), min(image.height, math.ceil(bounds[3] * scale)))
                if box not in cached_regions:
                    region_key = _text_sha(_json([page_number, box]))[:24]
                    relative = Path('assets') / doc_id / f'page-{page_number}-{region_key}.jpg'
                    path = (output / relative).resolve()
                    if not path.is_relative_to(output.resolve()):
                        raise RetrievalDatasetError('Visual asset resolves outside output directory')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(box).save(path, format='JPEG', quality=88)
                    # Text and pixels use exactly the same region. PDFium text uses
                    # bottom-left coordinates; image/PDF locator boxes use top-left.
                    text = (text_page.get_text_bounded() if needs_page_context else
                            text_page.get_text_bounded(left=box[0] / scale, right=box[2] / scale,
                                                      bottom=height - box[3] / scale, top=height - box[1] / scale))
                    cached_regions[box] = (relative.as_posix(), _sha(path), text)
                asset_path, asset_sha256, text = cached_regions[box]
                locator.update({'asset_extent': extent, 'asset_granularity': 'page' if extent.startswith('page_') else 'region',
                                'context_source_unit_ids': associated_ids, 'render_scale': scale,
                                'render_pixel_box': box, 'context_bbox': bounds})
                results.append({**unit, 'export_locator': locator, 'asset_path': asset_path,
                                'asset_sha256': asset_sha256, 'region_text': text})
            return results
        finally:
            text_page.close()
    finally:
        page.close()


def build_retrieval_dataset(*, selection_path: Path, data_root: Path, run_dirs: Sequence[Path], output_dir: Path,
                            visual_doc_ids: Sequence[int | str] | None = None, curation_path: Path | None = None) -> dict:
    """Reuse only text/table/figure cases from the specified immutable selection."""
    import pypdfium2 as pdfium

    output_dir = _external(output_dir)
    if output_dir.exists():
        raise RetrievalDatasetError(f'Output already exists: {output_dir}')
    if not run_dirs:
        raise RetrievalDatasetError('At least one historical run is required')
    selection = json.loads(selection_path.read_text())
    cases = [case for case in selection['cases'] if case['question_type'] in {'text-only', 'multimodal-t', 'multimodal-f'}]
    data_root = data_root.expanduser().resolve()
    for case in cases:
        doc_id, question_index = case.get('doc_id'), case.get('question_index')
        if (type(doc_id) is not int or doc_id < 0
                or type(question_index) is not int or question_index < 0):
            raise RetrievalDatasetError('Invalid selection: document id and question index must be nonnegative integers')
        if case.get('case_id') != f'docbench:{doc_id}:{question_index}':
            raise RetrievalDatasetError('Invalid selection: case identity does not match document and question index')
        filename = case.get('pdf_filename')
        if (not isinstance(filename, str) or not filename or filename in {'.', '..'}
                or '/' in filename or '\\' in filename or Path(filename).name != filename):
            raise RetrievalDatasetError('Invalid selection: PDF filename must be a basename')
    if not cases or len({case['case_id'] for case in cases}) != len(cases):
        raise RetrievalDatasetError('Selection is empty or contains duplicate cases')
    if len({str(case['doc_id']) for case in cases}) != len(cases):
        raise RetrievalDatasetError('Historical reuse currently requires one selected question per document')
    document_ids = {str(case['doc_id']) for case in cases}
    visual_documents = document_ids if visual_doc_ids is None else {str(value) for value in visual_doc_ids}
    if not visual_documents <= document_ids:
        raise RetrievalDatasetError('Visual document scope contains an unselected document')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after the complete dataset has been built successfully.
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}-', dir=output_dir.parent) as temp:
        stage = Path(temp)
        with sqlite3.connect(stage / 'dataset.sqlite') as target:
            target.executescript(SCHEMA)
            target.execute('PRAGMA foreign_keys=ON')
            sources = []
            for case in cases:
                doc_id = str(case['doc_id'])
                qa = (data_root / doc_id / f'{doc_id}_qa.jsonl').resolve()
                pdf_path = (data_root / doc_id / case['pdf_filename']).resolve()
                if not qa.is_relative_to(data_root) or not pdf_path.is_relative_to(data_root):
                    raise RetrievalDatasetError('Selected source resolves outside data root')
                if _sha(qa) != case['qa_sha256'] or _sha(pdf_path) != case['pdf_sha256']:
                    raise RetrievalDatasetError(f'Source hash mismatch: {case["case_id"]}')
                record = json.loads(qa.read_text().splitlines()[case['question_index']])
                if record['type'] != case['question_type']:
                    raise RetrievalDatasetError('Question type mismatch')
                run, database, version, manifest, chunks, elements = _find_document(run_dirs, case)
                source_hashes = {suffix or 'db': _sha(Path(str(database) + suffix)) for suffix in ('', '-wal') if Path(str(database) + suffix).exists()}
                target.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                    doc_id, str(pdf_path.resolve()), case['pdf_sha256'], str(database.resolve()),
                    source_hashes['db'], run.name, version['document_id'], version['id'],
                    version['processor_fingerprint'], version['chunker_fingerprint'],
                    version['physical_page_count'], version['processing_status'], _json(manifest),
                ))
                target.execute('INSERT INTO queries(case_id,doc_id,question_type,query,answer,evidence) VALUES (?,?,?,?,?,?)', (
                    case['case_id'], doc_id, record['type'], record['question'], record['answer'], record['evidence'],
                ))
                for chunk in chunks:
                    unit_id = f'docbench:{doc_id}:chunk:{chunk["producer_chunk_id"]}'
                    content = chunk['content']
                    if _text_sha(content) != chunk['content_sha256']:
                        raise RetrievalDatasetError(f'Chunk content hash mismatch: {unit_id}')
                    target.execute('INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (
                        unit_id, doc_id, 'chunk', content, chunk['id'], version['id'],
                        chunk['source_pages_json'], chunk['span_json'], None, chunk['content_sha256'],
                        'historical_parser', chunk['metadata_json'],
                    ))
                visual_count = 0
                if doc_id in visual_documents:
                    by_id = {element['element_id']: element for element in elements}
                    pdf = pdfium.PdfDocument(str(pdf_path))
                    try:
                        for page in manifest['pages']:
                            units = [unit for unit in page['nontext_units'] if unit['kind'] in VISUAL_UNIT_KINDS]
                            if not units:
                                continue
                            rendered = _render_visual_page(pdf, page['page_number'], units, stage, doc_id)
                            for unit in _canonical_visual_units(rendered, by_id):
                                unit_id = f'docbench:{doc_id}:visual:{unit["unit_id"]}'
                                content = unit['description']
                                target.execute('INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (
                                    unit_id, doc_id, unit['kind'], content, unit['unit_id'], version['id'],
                                    _json(unit.get('source_pages') or [page['page_number']]),
                                    _json(unit['export_locator']), unit['asset_path'], _text_sha(content),
                                    'pdf_text_layer_region' if unit['region_text'].strip() else 'source_element_only',
                                    _json({'element_id': unit.get('element_id'), 'asset_sha256': unit['asset_sha256'],
                                           'source_unit_aliases': unit['source_unit_aliases']}),
                                ))
                                visual_count += 1
                    finally:
                        pdf.close()
                sources.append({'case_id': case['case_id'], 'doc_id': doc_id, 'source_run_id': run.name,
                                'source_db_hashes': source_hashes, 'chunk_count': len(chunks), 'visual_count': visual_count})
                print(f'exported {case["case_id"]}: {len(chunks)} chunks, {visual_count} visual units', flush=True)
            curation_report = _curate_corpus(target, curation_path) if curation_path is not None else None
            target.commit()
            counts = dict(target.execute('SELECT kind,count(*) FROM units GROUP BY kind'))
        manifest = {
            'schema_version': 'docbench-retrieval-dataset-v1', 'status': 'awaiting_annotation',
            'selection_sha256': _sha(selection_path), 'case_count': len(cases),
            'question_types': dict(Counter(case['question_type'] for case in cases)),
            'unit_counts': counts, 'scope': 'per_document',
            'visual_description_method': 'pdf_text_layer_region_or_own_element', 'query_conditioned_observations_included': False,
            'visual_document_ids': sorted(visual_documents, key=int),
            'visual_document_scope': 'all_selected_documents' if visual_doc_ids is None else 'explicit_document_ids',
            'curation': curation_report,
            'label_policy': 'independent_review_required', 'historical_source_order': [str(path.resolve()) for path in run_dirs],
            'sources': sources, 'dataset_sha256': _sha(stage / 'dataset.sqlite'),
            'limitations': ['Historical parser/chunker versions are heterogeneous and frozen per document.',
                            'Detector visual units may be fragments or decorations, not semantic whole figures.',
                            'Visual descriptions use the text layer inside the exported region, with own-element text fallback; no fresh VLM interpretation.',
                            'No original question, reference answer or evidence is used to construct corpus descriptions.'],
        }
        (stage / 'manifest.json').write_text(_json(manifest) + '\n')
        write_review_packets(stage / 'dataset.sqlite', stage / 'review')
        (stage / 'README.md').write_text(
            '# DocBench retrieval dataset\n\nPrivate local artifact; do not publish original QA, corpus or images.\n\n'
            'The SQLite database is the dataset authority. manifest.json pins source identities. '
            'review/ contains automatic evidence-location proposals, not accepted gold labels. '
            'assets/ contains detector crops from original PDFs. Historical question-conditioned '
            'vision responses were excluded. Descriptions come from the original text layer inside '
            'the exported image region, or its own source element; they are not vision model interpretations.\n'
        )
        shutil.move(str(stage), str(output_dir))
    return manifest


def _tokens(text: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def write_review_packets(dataset_path: Path, output_dir: Path) -> None:
    """Create annotation aids using gold evidence; these never become retrieval input."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    with sqlite3.connect(dataset_path) as connection:
        connection.row_factory = sqlite3.Row
        for query in connection.execute('SELECT * FROM queries ORDER BY case_id'):
            kinds = TEXT_UNIT_KINDS if query['question_type'] == 'text-only' else VISUAL_UNIT_KINDS
            units = [dict(row) for row in connection.execute(
                'SELECT * FROM units WHERE doc_id=? ORDER BY unit_id', (query['doc_id'],)) if row['kind'] in kinds]
            evidence_tokens = _tokens(query['evidence'])
            probe_tokens = evidence_tokens or _tokens(query['query'])
            def score(unit):
                return len(probe_tokens & _tokens(unit['content'])) / max(1, len(probe_tokens))
            ranked = sorted(units, key=lambda unit: (-score(unit), unit['unit_id']))[:8]
            payload = {'query': dict(query), 'candidate_count': len(units), 'proposals': [
                {**unit, 'proposal_overlap': score(unit)} for unit in ranked]}
            primary_ids = {row[0] for row in connection.execute(
                "SELECT unit_id FROM qrels WHERE case_id=? AND role='primary'", (query['case_id'],))}
            payload['reviewed_primary_ids'] = sorted(primary_ids)
            stem = query['case_id'].replace(':', '-')
            (output_dir / f'{stem}.json').write_text(_json(payload) + '\n')
            items = []
            reviewed_units = [unit for unit in units if unit['unit_id'] in primary_ids]
            displayed = reviewed_units + [unit for unit in ranked if unit['unit_id'] not in primary_ids]
            for unit in displayed:
                picture = f'<img style="max-width:850px;max-height:650px" src="../{html.escape(unit["asset_path"])}">' if unit['asset_path'] else ''
                label = 'Reviewed primary' if unit['unit_id'] in primary_ids else 'Automatic proposal'
                items.append(f'<section><h3>{label}: {html.escape(unit["unit_id"])}</h3><p>Pages {html.escape(unit["pages_json"])}</p>{picture}<pre>{html.escape(unit["content"])}</pre></section>')
            (output_dir / f'{stem}.html').write_text(
                '<meta charset="utf-8"><style>body{font:16px system-ui;margin:32px}pre{white-space:pre-wrap}section{border-top:1px solid #ddd;margin-top:24px}</style>'
                f'<h1>{html.escape(query["case_id"])}</h1><p>{html.escape(query["query"])}</p>'
                f'<p>Reference: {html.escape(query["answer"])}</p><p>Evidence: {html.escape(query["evidence"])}</p>'
                f'<p>Status: {html.escape(query["annotation_status"])}</p><p>Review: {html.escape(query["annotation_note"])}</p>'
                '<p>Automatic proposals for annotation only. Inspect alternatives in dataset.sqlite; rankings here are not an evaluation result.</p>'
                + ''.join(items)
            )
            index_rows.append(
                f'<tr><td><a href="review/{stem}.html">{html.escape(query["case_id"])}</a></td>'
                f'<td>{html.escape(query["question_type"])}</td><td>{html.escape(query["annotation_status"])}</td>'
                f'<td>{len(primary_ids)}</td><td>{html.escape(query["query"])}</td></tr>'
            )
    (output_dir.parent / 'INDEX.html').write_text(
        '<meta charset="utf-8"><title>DocBench retrieval review</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#24578b}</style>'
        '<h1>DocBench retrieval dataset</h1><p>Private review browser. Reviewed labels are model-assisted, not an independent human gold standard. '
        'Pending cases remain visible and are excluded from scored denominators. Raw detector regions include fragments and decorations.</p>'
        '<table><thead><tr><th>Case</th><th>Type</th><th>Status</th><th>Primary units</th><th>Query</th></tr></thead><tbody>'
        + ''.join(index_rows) + '</tbody></table>\n'
    )


def apply_annotations(*, dataset_path: Path, annotation_path: Path) -> dict:
    """Review a frozen snapshot offline, retaining recovery copies until publication."""
    dataset_path = _external(dataset_path)
    stage = dataset_path.parent / f'.{dataset_path.name}-annotation-staging'
    if stage.exists():
        raise RetrievalDatasetError(f'Annotation already active or recovery required: {stage}')
    manifest_path = dataset_path.parent / 'manifest.json'
    try:
        original_manifest = manifest_path.read_bytes()
        manifest = json.loads(original_manifest)
    except (OSError, ValueError) as error:
        raise RetrievalDatasetError('A readable valid dataset manifest is required') from error
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 'docbench-retrieval-dataset-v1':
        raise RetrievalDatasetError('Unsupported dataset manifest schema')
    wal_path = Path(str(dataset_path) + '-wal')
    if wal_path.exists() and wal_path.stat().st_size:
        raise RetrievalDatasetError('Dataset has an unmerged WAL; close writers and checkpoint before annotation')
    original_hash = _sha(dataset_path)
    if manifest.get('dataset_sha256') != original_hash:
        raise RetrievalDatasetError('Dataset hash does not match manifest')
    annotation_bytes = annotation_path.read_bytes()
    annotations = json.loads(annotation_bytes)
    if not isinstance(annotations, list):
        raise RetrievalDatasetError('Annotations must be a list')
    # A fixed exclusive directory serializes this command and preserves backups if
    # the process is interrupted between the two file replacements. Never silently
    # discard an existing recovery directory from an earlier interrupted publish.
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RetrievalDatasetError(f'Annotation already active or recovery required: {stage}') from error
    cleanup = True
    try:
        backup = stage / 'original.sqlite'
        staged_database = stage / 'updated.sqlite'
        staged_manifest = stage / 'updated-manifest.json'
        shutil.copyfile(dataset_path, backup)
        shutil.copyfile(backup, staged_database)
        (stage / 'original-manifest.json').write_bytes(original_manifest)
        seen = set()
        with sqlite3.connect(staged_database) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('PRAGMA foreign_keys=ON')
            for item in annotations:
                case_id = item['case_id']
                if case_id in seen:
                    raise RetrievalDatasetError('Duplicate annotation case')
                seen.add(case_id)
                query = connection.execute('SELECT * FROM queries WHERE case_id=?', (case_id,)).fetchone()
                if query is None:
                    raise RetrievalDatasetError(f'Unknown case: {case_id}')
                if query['annotation_status'] == 'reviewed':
                    raise RetrievalDatasetError(f'Already reviewed: {case_id}')
                primary = item.get('primary_unit_ids', [])
                if not item.get('reviewer') or not item.get('note'):
                    raise RetrievalDatasetError('Reviewed labels require primary refs, reviewer and rationale')
                status = item.get('status', 'reviewed')
                if status not in {'pending', 'reviewed'}:
                    raise RetrievalDatasetError(f'Unsupported annotation status: {status}')
                if status == 'pending':
                    if primary or item.get('support_unit_ids'):
                        raise RetrievalDatasetError('Pending annotations cannot include gold references')
                    connection.execute('UPDATE queries SET annotation_note=? WHERE case_id=?', (item['note'], case_id))
                    continue
                if not primary:
                    raise RetrievalDatasetError('Reviewed labels require primary refs, reviewer and rationale')
                rows = [(unit_id, 'primary') for unit_id in primary] + [(unit_id, 'support') for unit_id in item.get('support_unit_ids', [])]
                for unit_id, role in rows:
                    unit = connection.execute('SELECT * FROM units WHERE unit_id=?', (unit_id,)).fetchone()
                    if unit is None or unit['doc_id'] != query['doc_id']:
                        raise RetrievalDatasetError(f'Gold reference outside query document: {unit_id}')
                    if role == 'primary' and (unit['kind'] == 'chunk') != (query['question_type'] == 'text-only'):
                        raise RetrievalDatasetError(f'Gold reference has wrong modality: {unit_id}')
                    connection.execute('INSERT INTO qrels VALUES (?,?,?,?,?)', (
                        case_id, unit_id, role, item['note'], item['reviewer'],
                    ))
                connection.execute("UPDATE queries SET annotation_status='reviewed',annotation_note=? WHERE case_id=?", (item['note'], case_id))
            manifest['annotation_counts'] = dict(connection.execute('SELECT annotation_status,count(*) FROM queries GROUP BY annotation_status'))
        connection.close()
        manifest.setdefault('annotation_batches', []).append({
            'sha256': hashlib.sha256(annotation_bytes).hexdigest(), 'case_ids': sorted(seen),
        })
        manifest['dataset_sha256'] = _sha(staged_database)
        manifest['status'] = ('reviewed' if manifest['annotation_counts'].get('pending', 0) == 0
                              else 'partially_reviewed' if manifest['annotation_counts'].get('reviewed', 0)
                              else 'awaiting_annotation')
        staged_manifest.write_text(_json(manifest) + '\n')
        if (_sha(dataset_path) != original_hash or _sha(backup) != original_hash
                or manifest_path.read_bytes() != original_manifest
                or (wal_path.exists() and wal_path.stat().st_size)):
            raise RetrievalDatasetError('Dataset changed during annotation; no labels published')
        (stage / 'recovery.json').write_text(_json({
            'dataset_path': str(dataset_path), 'manifest_path': str(manifest_path),
            'original_dataset_sha256': original_hash,
            'updated_dataset_sha256': manifest['dataset_sha256'],
            'recovery': 'Restore original.sqlite and original-manifest.json together to the paths above before removing this directory.',
        }) + '\n')
        cleanup = False
        staged_database.replace(dataset_path)
        try:
            staged_manifest.replace(manifest_path)
        except OSError as error:
            try:
                rollback = stage / 'rollback.sqlite'
                shutil.copyfile(backup, rollback)
                rollback.replace(dataset_path)
                if manifest_path.read_bytes() != original_manifest:
                    rollback_manifest = stage / 'rollback-manifest.json'
                    rollback_manifest.write_bytes(original_manifest)
                    rollback_manifest.replace(manifest_path)
            except OSError as recovery_error:
                cleanup = False
                raise RetrievalDatasetError(f'Annotation publication failed; recovery copies retained at {stage}') from recovery_error
            cleanup = True
            raise RetrievalDatasetError('Annotation manifest publication failed; original dataset restored') from error
        cleanup = True
        return manifest
    finally:
        if cleanup:
            shutil.rmtree(stage)


def _curate_corpus(connection: sqlite3.Connection, curation_path: Path) -> dict:
    """Apply reviewed source transcriptions inside the unpublished build transaction."""
    raw = curation_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get('schema_version') != 'docbench-retrieval-curation-v1':
        raise RetrievalDatasetError('Unsupported curation schema')
    if set(payload) - {'schema_version', 'units', 'corrections'}:
        raise RetrievalDatasetError('Unknown curation fields')
    if not isinstance(payload.get('units', []), list) or not isinstance(payload.get('corrections', []), list):
        raise RetrievalDatasetError('Curation units and corrections must be lists')
    batch_hash = hashlib.sha256(raw).hexdigest()

    def row(sql, arguments):
        cursor = connection.execute(sql, arguments)
        result = cursor.fetchone()
        return dict(zip((column[0] for column in cursor.description), result, strict=True)) if result else None

    def reviewed(item):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip() for key in ('reviewer', 'note')
        ):
            raise RetrievalDatasetError('Curation requires reviewer and source rationale')

    seen = set()
    for item in payload.get('units', []):
        reviewed(item)
        unit_id = item.get('unit_id')
        if not isinstance(unit_id, str) or unit_id in seen:
            raise RetrievalDatasetError('Invalid or duplicate curated unit id')
        seen.add(unit_id)
        source_id = item.get('derived_from_unit_id', unit_id)
        source = row('SELECT u.*,d.pdf_sha256 FROM units u JOIN documents d USING(doc_id) WHERE unit_id=?', (source_id,))
        if source is None:
            raise RetrievalDatasetError(f'Unknown curation source unit: {source_id}')
        if item.get('source_pdf_sha256') != source['pdf_sha256']:
            raise RetrievalDatasetError('Curation PDF hash differs from source')
        pages = item.get('source_pages')
        if (not isinstance(pages, list) or not pages or any(type(page) is not int or page < 1 for page in pages)
                or not set(pages).issubset(json.loads(source['pages_json']))):
            raise RetrievalDatasetError('Curation page is outside source unit')
        content = item.get('content')
        if not isinstance(content, str) or not content.strip():
            raise RetrievalDatasetError('Curated source text must be nonempty')
        metadata = json.loads(source['metadata_json'])
        metadata['curation'] = {'batch_sha256': batch_hash, 'reviewer': item['reviewer'], 'source_unit_id': source_id}
        original = source['content']
        if source_id != unit_id:
            if (item.get('kind') != 'chunk' or not unit_id.startswith(f'docbench:{source["doc_id"]}:chunk:')
                    or connection.execute('SELECT 1 FROM units WHERE unit_id=?', (unit_id,)).fetchone()):
                raise RetrievalDatasetError('New curated unit must be a unique same-document text transcription')
            original = None
            connection.execute('INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (
                unit_id, source['doc_id'], 'chunk', content, source['source_unit_id'], source['source_revision'],
                _json(pages), source['locator_json'], None, _text_sha(content),
                'reviewed_source_transcription', _json(metadata),
            ))
        else:
            connection.execute('UPDATE units SET content=?,content_sha256=?,description_method=?,metadata_json=? WHERE unit_id=?', (
                content, _text_sha(content), 'reviewed_source_transcription', _json(metadata), unit_id,
            ))
        connection.execute('INSERT INTO curation_events(target_type,target_id,field,original_value,revised_value,reviewer,note,source_json) VALUES (?,?,?,?,?,?,?,?)', (
            'unit', unit_id, 'content', original, content, item['reviewer'], item['note'],
            _json({'pdf_sha256': source['pdf_sha256'], 'pages': pages, 'source_unit_id': source_id, 'batch_sha256': batch_hash}),
        ))
    corrected = set()
    for item in payload.get('corrections', []):
        reviewed(item)
        case_id, field = item.get('case_id'), item.get('field')
        if field not in {'answer', 'evidence'} or not isinstance(case_id, str) or (case_id, field) in corrected:
            raise RetrievalDatasetError('Only unique answer/evidence corrections are allowed; queries stay unchanged')
        corrected.add((case_id, field))
        query = row('SELECT * FROM queries WHERE case_id=?', (case_id,))
        source = row('SELECT * FROM units WHERE unit_id=?', (item.get('source_unit_id'),))
        if query is None or source is None or query['doc_id'] != source['doc_id']:
            raise RetrievalDatasetError('Correction source is outside question document')
        if query['annotation_status'] != 'pending' or query[field] != item.get('old_value'):
            raise RetrievalDatasetError('Correction original value/state differs from the source')
        revised = item.get('new_value')
        if not isinstance(revised, str) or not revised.strip():
            raise RetrievalDatasetError('Corrected reference must be nonempty')
        connection.execute(f'UPDATE queries SET {field}=? WHERE case_id=?', (revised, case_id))
        connection.execute('INSERT INTO curation_events(target_type,target_id,field,original_value,revised_value,reviewer,note,source_json) VALUES (?,?,?,?,?,?,?,?)', (
            'query', case_id, field, item['old_value'], revised, item['reviewer'], item['note'],
            _json({'source_unit_id': source['unit_id'], 'pages': json.loads(source['pages_json']), 'batch_sha256': batch_hash}),
        ))
    return {'sha256': batch_hash, 'unit_count': len(seen), 'correction_count': len(corrected),
            'method': 'reviewed_source_transcription; original references retained in curation_events'}


def write_retrieval_exports(dataset_path: Path, output_dir: Path) -> None:
    """Export indexable text and scoped queries separately from references and judgments."""
    dataset_path, output_dir = _external(dataset_path), _external(output_dir)
    before = _sha(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'{dataset_path.as_uri()}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        corpus = ({
            'id': unit['unit_id'], 'text': unit['content'], 'doc_id': unit['doc_id'], 'kind': unit['kind'],
            'source_revision': unit['source_revision'], 'content_sha256': unit['content_sha256'],
            'source_pages': json.loads(unit['pages_json']), 'locator': json.loads(unit['locator_json']),
            'asset_path': unit['asset_path'], 'description_method': unit['description_method'],
        } for unit in connection.execute('SELECT * FROM units ORDER BY unit_id'))
        queries = ({
            'id': query['case_id'], 'query': query['query'], 'doc_id': query['doc_id'],
            'question_type': query['question_type'], 'annotation_status': query['annotation_status'],
            'allowed_kinds': sorted(TEXT_UNIT_KINDS if query['question_type'] == 'text-only' else VISUAL_UNIT_KINDS),
        } for query in connection.execute('SELECT * FROM queries ORDER BY case_id'))
        qrels = ({'query_id': item['case_id'], 'corpus_id': item['unit_id'], 'score': 1, 'role': 'primary'}
                 for item in connection.execute("SELECT r.* FROM qrels r JOIN queries q USING(case_id) WHERE q.annotation_status='reviewed' AND r.role='primary' ORDER BY r.case_id,r.unit_id"))
        files = {}
        for filename, records in [('corpus.jsonl', corpus), ('queries.jsonl', queries), ('qrels.jsonl', qrels)]:
            count = 0
            temporary = output_dir / f'.{filename}.tmp'
            with temporary.open('w', encoding='utf-8') as stream:
                for record in records:
                    stream.write(_json(record) + '\n')
                    count += 1
            temporary.replace(output_dir / filename)
            files[filename] = {'rows': count, 'sha256': _sha(output_dir / filename)}
    if _sha(dataset_path) != before:
        raise RetrievalDatasetError('Dataset changed while exporting semantic corpus; regenerate exports')
    (output_dir / 'manifest.json').write_text(_json({
        'dataset_sha256': before, 'files': files, 'indexed_field': 'corpus.text',
        'scope': 'filter by queries.doc_id and allowed_kinds before top-k',
        'asset_path_base': 'dataset root (parent of exports directory)',
        'qrels_policy': 'reviewed primary only; unjudged candidates are not proven negatives',
    }) + '\n')
