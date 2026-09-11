from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.retrieval.contracts import RetrievalMethod
from personagraph.retrieval.lifecycle.generation import (
    DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT,
    DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT,
)
from personagraph.retrieval.indexing.model_assets import LocalModelAssetRef
from personagraph.retrieval.indexing import model_assets as asset_module
from personagraph.retrieval import profile as profile_module
from personagraph.retrieval.profile import (
    DocumentRetrievalProfile,
    RetrievalFailurePolicy,
    RetrievalProfileMode,
    RetrievalProfileUnavailable,
    RetrievalRerankerMode,
    build_document_retrieval_runtime,
    preflight_document_retrieval_profile,
)


def _complete_model(path: Path, *, hybrid: bool) -> LocalModelAssetRef:
    path.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (path / name).write_bytes(b"fixture")
    if hybrid:
        (path / "sparse_linear.pt").write_bytes(b"fixture")
        (path / "colbert_linear.pt").write_bytes(b"fixture")
    return LocalModelAssetRef.path(
        path,
        canonical_identity=f"test:{path.name}@1",
    )


@pytest.fixture
def available_optional_runtime(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        asset_module.importlib.util,
        "find_spec",
        lambda name: marker,
    )
    monkeypatch.setattr(
        profile_module.importlib.util,
        "find_spec",
        lambda name: marker,
    )


def test_production_profile_is_strict_pinned_local_hybrid():
    profile = DocumentRetrievalProfile.production()

    assert profile.mode is RetrievalProfileMode.BGE_M3
    assert profile.failure_policy is RetrievalFailurePolicy.STRICT
    assert profile.reranker_mode is RetrievalRerankerMode.BGE_V2_M3
    assert profile.local_files_only is True
    assert profile.retrieval_methods == (
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    )
    assert profile.index_recipe == DOCUMENT_HYBRID_INDEX_RECIPE_CONTRACT
    assert profile.device == "auto"
    assert "BAAI/bge-m3@5617a9f" in profile.fingerprint()
    assert "/Users/" not in profile.fingerprint()


def test_lexical_profile_is_explicit_bm25_rollback():
    profile = DocumentRetrievalProfile.lexical()

    assert profile.retrieval_methods == (RetrievalMethod.BM25,)
    assert profile.index_recipe == DOCUMENT_LEXICAL_INDEX_RECIPE_CONTRACT
    assert profile.reranker_mode is RetrievalRerankerMode.OFF


def test_complete_local_assets_build_lazy_bge_runtime_without_path_in_identity(
    tmp_path,
    available_optional_runtime,
):
    encoder = _complete_model(tmp_path / "encoder", hybrid=True)
    reranker = _complete_model(tmp_path / "reranker", hybrid=False)
    profile = DocumentRetrievalProfile(
        encoder_asset=encoder,
        reranker_asset=reranker,
        device="cpu",
    )

    runtime = build_document_retrieval_runtime(profile)

    assert runtime.capability.ready is True
    assert runtime.effective_profile is profile
    assert runtime.degraded_reason is None
    assert runtime.encoder.diagnostic_snapshot()["loaded"] is False
    assert runtime.reranker is not None
    assert runtime.reranker.diagnostic_snapshot()["loaded"] is False
    snapshot = runtime.capability.diagnostic_snapshot()
    assert str(tmp_path) not in repr(snapshot)
    assert str(tmp_path) not in runtime.encoder.fingerprint()
    assert runtime.chunking_profile.tokenizer_id.startswith("bge_m3_tokenizer:")
    assert runtime.chunking_profile.target_tokens == 450
    assert runtime.chunking_profile.max_tokens == 600
    assert runtime.chunking_profile.min_tokens == 80
    assert runtime.chunking_profile.split_overlap_tokens == 64


def test_auto_device_is_resolved_once_before_components_and_identity_are_built(
    tmp_path,
    available_optional_runtime,
    monkeypatch,
):
    encoder = _complete_model(tmp_path / "encoder", hybrid=True)
    reranker = _complete_model(tmp_path / "reranker", hybrid=False)
    selection = SimpleNamespace(
        requested_device="auto",
        device="mps",
        fallback_reason=None,
        probe_reasons=(),
        ready=True,
    )
    requested: list[str] = []

    def select_device(value: str):
        requested.append(value)
        return selection

    monkeypatch.setattr(profile_module, "select_device", select_device, raising=False)
    profile = DocumentRetrievalProfile(
        encoder_asset=encoder,
        reranker_asset=reranker,
        device="auto",
    )

    runtime = build_document_retrieval_runtime(profile)

    assert requested == ["auto"]
    assert runtime.requested_profile is profile
    assert runtime.requested_profile.device == "auto"
    assert runtime.effective_profile.device == "mps"
    assert runtime.capability.device_selection is selection
    assert runtime.encoder.diagnostic_snapshot()["device"] == "mps"
    assert runtime.reranker is not None
    assert runtime.reranker.diagnostic_snapshot()["device"] == "mps"
    assert "device=mps" in runtime.encoder.fingerprint()
    assert "device=auto" not in runtime.encoder.fingerprint()
    assert runtime.reranker.fingerprint().endswith(";cpu_fallback=true")


def test_auto_device_can_select_cpu_before_generation_identity_is_built(
    tmp_path,
    available_optional_runtime,
    monkeypatch,
):
    encoder = _complete_model(tmp_path / "encoder", hybrid=True)
    reranker = _complete_model(tmp_path / "reranker", hybrid=False)
    selection = SimpleNamespace(
        requested_device="auto",
        device="cpu",
        fallback_reason="accelerator_unavailable",
        probe_reasons=("mps_unavailable", "cuda_unavailable"),
        ready=True,
    )
    monkeypatch.setattr(
        profile_module,
        "select_device",
        lambda value: selection,
        raising=False,
    )
    profile = DocumentRetrievalProfile(
        encoder_asset=encoder,
        reranker_asset=reranker,
        device="auto",
    )

    runtime = build_document_retrieval_runtime(profile)

    assert runtime.requested_profile.device == "auto"
    assert runtime.effective_profile.device == "cpu"
    assert runtime.effective_profile.mode is RetrievalProfileMode.BGE_M3
    assert runtime.capability.ready is True
    assert runtime.capability.device_selection is selection
    assert "device=cpu" in runtime.encoder.fingerprint()
    assert runtime.reranker is not None
    assert ";cpu_fallback=true" not in runtime.reranker.fingerprint()


def test_unavailable_explicit_device_is_not_silently_rewritten_to_cpu(
    tmp_path,
    available_optional_runtime,
    monkeypatch,
):
    encoder = _complete_model(tmp_path / "encoder", hybrid=True)
    reranker = _complete_model(tmp_path / "reranker", hybrid=False)
    selection = SimpleNamespace(
        requested_device="cuda:7",
        device=None,
        fallback_reason=None,
        probe_reasons=("cuda_device_index_unavailable",),
        ready=False,
    )
    requested: list[str] = []

    def select_device(value: str):
        requested.append(value)
        return selection

    monkeypatch.setattr(profile_module, "select_device", select_device, raising=False)
    profile = DocumentRetrievalProfile(
        encoder_asset=encoder,
        reranker_asset=reranker,
        device="cuda:7",
    )

    capability = preflight_document_retrieval_profile(profile)

    assert requested == ["cuda:7"]
    assert capability.ready is False
    assert capability.device_selection is selection
    assert "retrieval_device_unavailable" in capability.reason_codes
    with pytest.raises(RetrievalProfileUnavailable, match="retrieval_device_unavailable"):
        build_document_retrieval_runtime(profile)


def test_auto_device_rejects_fp16_instead_of_changing_precision_silently():
    with pytest.raises(ValueError, match="auto.*fp16|fp16.*auto"):
        DocumentRetrievalProfile(device="auto", use_fp16=True)


def test_strict_profile_fails_during_preflight_before_model_load(
    tmp_path,
    available_optional_runtime,
):
    missing = LocalModelAssetRef.path(
        tmp_path / "missing",
        canonical_identity="test:missing@1",
    )
    profile = DocumentRetrievalProfile(
        encoder_asset=missing,
        reranker_mode=RetrievalRerankerMode.OFF,
    )

    capability = preflight_document_retrieval_profile(profile)
    assert capability.ready is False
    assert capability.reason_codes == ("encoder:local_model_path_unavailable",)
    with pytest.raises(RetrievalProfileUnavailable, match="local_model_path_unavailable"):
        build_document_retrieval_runtime(profile)


def test_lexical_fallback_requires_explicit_failure_policy(
    tmp_path,
    available_optional_runtime,
):
    missing = LocalModelAssetRef.path(
        tmp_path / "missing",
        canonical_identity="test:missing@1",
    )
    profile = DocumentRetrievalProfile(
        failure_policy=RetrievalFailurePolicy.LEXICAL_FALLBACK,
        reranker_mode=RetrievalRerankerMode.OFF,
        encoder_asset=missing,
    )

    runtime = build_document_retrieval_runtime(profile)

    assert runtime.requested_profile is profile
    assert runtime.effective_profile.mode is RetrievalProfileMode.LEXICAL
    assert runtime.degraded_reason == "encoder:local_model_path_unavailable"
    assert runtime.effective_profile.retrieval_methods == (RetrievalMethod.BM25,)


def test_environment_rejects_network_enabled_or_unpinned_custom_hub():
    with pytest.raises(ValueError, match="must remain true"):
        DocumentRetrievalProfile.from_environment(
            {"PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY": "false"}
        )
    with pytest.raises(ValueError, match="REVISION"):
        DocumentRetrievalProfile.from_environment(
            {
                "PERSONAGRAPH_RETRIEVAL_BGE_MODEL": "example/custom-model",
                "PERSONAGRAPH_RETRIEVAL_BGE_REVISION": "",
            }
        )


def test_environment_can_select_dense_sparse_first_stage_without_changing_default():
    default_profile = DocumentRetrievalProfile.from_environment({})
    experimental_profile = DocumentRetrievalProfile.from_environment(
        {
            "PERSONAGRAPH_RETRIEVAL_METHODS": "dense,learned_sparse",
        }
    )

    assert default_profile.retrieval_methods == (
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
        RetrievalMethod.BM25,
    )
    assert experimental_profile.retrieval_methods == (
        RetrievalMethod.DENSE,
        RetrievalMethod.LEARNED_SPARSE,
    )
    assert "methods=dense,learned_sparse;" in experimental_profile.fingerprint()


@pytest.mark.parametrize(
    "value",
    ("", "bm25", "dense,bm25", "dense,dense", "literal_boolean"),
)
def test_environment_rejects_unapproved_bge_first_stage_method_sets(value: str):
    with pytest.raises(ValueError, match="PERSONAGRAPH_RETRIEVAL_METHODS"):
        DocumentRetrievalProfile.from_environment(
            {"PERSONAGRAPH_RETRIEVAL_METHODS": value}
        )


def test_chunking_identity_binds_the_encoder_tokenizer_and_shared_limits():
    class FakeEncoder:
        def __init__(self, name: str, multiplier: int) -> None:
            self.name = name
            self.multiplier = multiplier

        def token_ids(self, text: str):
            return tuple(range(len(text) * self.multiplier))

        def tokenizer_fingerprint(self):
            return self.name

    lexical = profile_module.production_document_chunking_profile(
        DocumentRetrievalProfile.lexical(),
        encoder=FakeEncoder("lexical-tokenizer@1", 1),
    )
    bge = profile_module.production_document_chunking_profile(
        DocumentRetrievalProfile.production(),
        encoder=FakeEncoder("bge-tokenizer@1", 2),
    )

    assert lexical.fingerprint() != bge.fingerprint()
    assert lexical.tokens_of("abc论文") == 5
    assert bge.tokens_of("abc论文") == 10
    assert (bge.target_tokens, bge.max_tokens, bge.min_tokens) == (450, 600, 80)


def test_chunking_profile_only_requires_the_encoder_tokenizer(
    tmp_path,
    monkeypatch,
):
    encoder = _complete_model(tmp_path / "encoder", hybrid=True)
    missing_reranker = LocalModelAssetRef.path(
        tmp_path / "missing-reranker",
        canonical_identity="test:missing-reranker@1",
    )
    profile = DocumentRetrievalProfile(
        encoder_asset=encoder,
        reranker_asset=missing_reranker,
        device="auto",
    )

    def fail_if_device_is_probed(_value: str):
        raise AssertionError("tokenizer-only chunking must not probe a compute device")

    monkeypatch.setattr(profile_module, "select_device", fail_if_device_is_probed)

    chunking = profile_module.production_document_chunking_profile(profile)

    assert chunking.tokenizer_id.startswith("bge_m3_tokenizer:")
    assert (chunking.target_tokens, chunking.max_tokens) == (450, 600)
