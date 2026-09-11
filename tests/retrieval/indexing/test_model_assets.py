from __future__ import annotations

import importlib.metadata

import pytest

from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    document_paper_generation_spec,
)
from personagraph.retrieval.indexing.methods import BgeM3Encoder
from personagraph.retrieval.indexing.model_assets import (
    BGE_M3_HYBRID_MANIFEST,
    LocalModelAssetError,
    LocalModelAssetRef,
    preflight_local_model_asset,
)


def test_bge_encoder_fails_closed_for_learned_sparse_when_projection_assets_are_incomplete(tmp_path):
    incomplete_model = tmp_path / "incomplete-bge-m3"
    incomplete_model.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (incomplete_model / name).write_bytes(b"placeholder")

    asset = LocalModelAssetRef.path(
        incomplete_model,
        canonical_identity="test:incomplete-bge-m3",
    )
    incomplete = BgeM3Encoder(asset=asset, use_fp16=False)
    assert incomplete.learned_sparse_available is False
    assert "sparse_projection_assets=unavailable" in incomplete.fingerprint()

    (incomplete_model / "sparse_linear.pt").write_bytes(b"placeholder")
    (incomplete_model / "colbert_linear.pt").write_bytes(b"placeholder")
    complete = BgeM3Encoder(asset=asset, use_fp16=False)

    assert complete.learned_sparse_available is True
    assert "sparse_projection_assets=available" in complete.fingerprint()
    assert str(tmp_path) not in complete.fingerprint()


@pytest.mark.parametrize("revision", ["main", "v1.2.3", "5617a9f"])
def test_hub_model_asset_requires_a_full_immutable_commit_sha(revision):
    with pytest.raises(ValueError, match="commit SHA"):
        LocalModelAssetRef.hub("BAAI/bge-m3", revision=revision)


def test_bge_generation_identity_binds_runtime_implementation_versions(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ):
        (model_path / name).write_bytes(b"fixture")
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-m3@weights-v1",
    )
    versions = {
        "FlagEmbedding": "1.4.0",
        "transformers": "4.57.6",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    first_encoder = BgeM3Encoder(asset=asset, use_fp16=False)
    first_fingerprint = first_encoder.fingerprint()
    first_generation = document_paper_generation_spec(
        encoder_fingerprint=first_fingerprint,
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=3,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )

    versions["FlagEmbedding"] = "1.4.1"
    second_encoder = BgeM3Encoder(asset=asset, use_fp16=False)
    second_fingerprint = second_encoder.fingerprint()
    second_generation = document_paper_generation_spec(
        encoder_fingerprint=second_fingerprint,
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=3,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )

    assert "flagembedding=1.4.0" in first_fingerprint
    assert "transformers=4.57.6" in first_fingerprint
    assert first_encoder.fingerprint() == first_fingerprint
    assert second_fingerprint != first_fingerprint
    assert second_generation.version_id != first_generation.version_id
    assert str(tmp_path) not in first_fingerprint


def test_model_preflight_rejects_an_index_whose_weight_shard_is_missing(tmp_path):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ):
        (model_path / name).write_bytes(b"fixture")
    (model_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"encoder.layer":"model-00001-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-m3@missing-shard",
    )

    capability = preflight_local_model_asset(asset, BGE_M3_HYBRID_MANIFEST)

    assert not capability.ready
    assert capability.reason_code == "local_model_asset_incomplete"
    assert capability.missing_files == (
        "missing_weight_shard(model.safetensors.index.json)",
    )


def test_model_preflight_accepts_a_complete_indexed_weight_set(tmp_path):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "sparse_linear.pt",
        "colbert_linear.pt",
        "model-00001-of-00002.safetensors",
    ):
        (model_path / name).write_bytes(b"fixture")
    (model_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"encoder.layer":"model-00001-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-m3@complete-shards",
    )

    capability = preflight_local_model_asset(asset, BGE_M3_HYBRID_MANIFEST)

    assert capability.ready
    assert capability.missing_files == ()
    assert capability.content_sha256 is not None
    assert len(capability.content_sha256) == 64


def test_managed_local_weight_change_rotates_encoder_and_generation_identity(tmp_path):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ):
        (model_path / name).write_bytes(f"first:{name}".encode("utf-8"))
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-m3@managed",
    )
    first_encoder = BgeM3Encoder(asset=asset, use_fp16=False)
    first_generation = document_paper_generation_spec(
        encoder_fingerprint=first_encoder.fingerprint(),
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=3,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )

    (model_path / "model.safetensors").write_bytes(b"second:changed-model-weights")
    second_encoder = BgeM3Encoder(asset=asset, use_fp16=False)
    second_generation = document_paper_generation_spec(
        encoder_fingerprint=second_encoder.fingerprint(),
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=3,
        index_recipe=DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    )

    assert "content_sha256=" in first_encoder.fingerprint()
    assert first_encoder.fingerprint() != second_encoder.fingerprint()
    assert first_generation.version_id != second_generation.version_id
    assert str(tmp_path) not in first_encoder.fingerprint()


def test_managed_local_asset_change_after_preflight_fails_closed(tmp_path):
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ):
        (model_path / name).write_bytes(f"first:{name}".encode("utf-8"))
    asset = LocalModelAssetRef.path(
        model_path,
        canonical_identity="test:bge-m3@managed",
    )
    capability = preflight_local_model_asset(asset, BGE_M3_HYBRID_MANIFEST)
    assert capability.ready

    (model_path / "model.safetensors").write_bytes(b"changed-after-preflight")

    with pytest.raises(LocalModelAssetError, match="changed_after_preflight"):
        capability.require_unchanged()
