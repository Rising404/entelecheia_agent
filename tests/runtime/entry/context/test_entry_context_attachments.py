from __future__ import annotations

import pytest

from personagraph.runtime.entry.context.attachments import (
    AttachmentAccess,
    AttachmentProjectionItem,
)
from personagraph.runtime.entry.context.attachments import (
    attachment_projection_token_cost,
    build_on_demand_attachment_projection,
    render_attachment_manifest,
)


def _record(attachment_id: str, kind: str, *, name: str) -> dict[str, object]:
    return {
        "attachment_id": attachment_id,
        "original_name": name,
        "media_type": "application/octet-stream",
        "size_bytes": 1_024,
        "kind": kind,
        "stored_rel_path": f"input/{attachment_id}/{name}",
        "content_hash": "0" * 64,
    }


def test_on_demand_projection_keeps_metadata_without_touching_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.runtime.entry.context import attachments as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("on-demand projection must perform no source I/O")

    monkeypatch.setattr(module, "open", forbidden, raising=False)
    projection = build_on_demand_attachment_projection(
        [
            _record("text-1", "text", name="notes.txt"),
            _record("document-1", "document", name="report.pdf"),
            _record("image-1", "image", name="diagram.png"),
        ]
    )

    assert [item.access for item in projection.items] == [
        AttachmentAccess.ON_DEMAND,
        AttachmentAccess.ON_DEMAND,
        AttachmentAccess.ON_DEMAND,
    ]
    assert set(projection.items[0].model_dump()) == {
        "attachment_id",
        "name",
        "media_type",
        "size_bytes",
        "kind",
        "access",
        "stored_rel_path",
        "content_hash",
    }
    rendered = render_attachment_manifest(projection)
    assert "内容按需读取" in rendered
    assert "input/text-1/notes.txt" not in rendered
    assert "0" * 64 not in rendered
    assert attachment_projection_token_cost(projection) <= 2_000


@pytest.mark.parametrize(
    "invalid_update",
    (
        {"text": "not loaded"},
        {"reason": "too_large"},
        {"source_coverage": "complete"},
        {"source_diagnostics": ()},
        {"estimated_tokens": 1},
        {"access": "image"},
        {"access": "full_text"},
        {"access": "excerpt"},
        {"access": "unavailable"},
    ),
)
def test_on_demand_contract_cannot_claim_content_or_processing(invalid_update) -> None:
    payload = {
        "attachment_id": "a1",
        "name": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": 128,
        "kind": "document",
        "access": AttachmentAccess.ON_DEMAND,
        "stored_rel_path": "input/a1/report.pdf",
        "content_hash": "0" * 64,
    }
    payload.update(invalid_update)
    with pytest.raises(ValueError):
        AttachmentProjectionItem(**payload)
