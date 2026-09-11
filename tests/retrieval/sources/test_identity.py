from __future__ import annotations

import pytest

from personagraph.retrieval.sources.identity import (
    current_session_source_unit_id,
    parse_current_session_source_unit_id,
    parse_picture_observation_source_unit_id,
    picture_observation_ref_and_content,
    picture_observation_source_unit_id,
)
from personagraph.retrieval.contracts import SourceType


def test_current_session_source_unit_identity_round_trips() -> None:
    source_unit_id = current_session_source_unit_id(
        session_id="private-session-1",
        run_id="private-run-2",
        role="assistant",
        ordinal=3,
    )

    identity = parse_current_session_source_unit_id(source_unit_id)

    assert identity is not None
    assert identity.session_id == "private-session-1"
    assert identity.run_id == "private-run-2"
    assert identity.role == "assistant"
    assert identity.ordinal == 3


@pytest.mark.parametrize(
    "source_unit_id",
    (
        "cs2:cHJpdmF0ZS1zZXNzaW9uLTE=:cHJpdmF0ZS1ydW4tMg:pair:0",
        "cs2:cHJpdmF0ZS1zZXNzaW9uLTE:cHJpdmF0ZS1ydW4tMg:pair:0:extra",
        "cs2:cHJpdmF0ZS1zZXNzaW9uLTE:cHJpdmF0ZS1ydW4tMg:system:0",
        "cs2:cHJpdmF0ZS1zZXNzaW9uLTE:cHJpdmF0ZS1ydW4tMg:pair:01",
        "cs2:cHJpdmF0ZS1zZXNzaW9uLTE:cHJpdmF0ZS1ydW4tMg:pair:1000000000000000000",
        "cs2:*:cHJpdmF0ZS1ydW4tMg:pair:0",
        "cs2:_w:cHJpdmF0ZS1ydW4tMg:pair:0",
    ),
)
def test_current_session_source_unit_parser_rejects_noncanonical_values(
    source_unit_id: str,
) -> None:
    assert parse_current_session_source_unit_id(source_unit_id) is None


def test_picture_observation_identity_and_normalized_content_are_canonical() -> None:
    source_unit_id = picture_observation_source_unit_id("pobs:private/1")
    identity = parse_picture_observation_source_unit_id(source_unit_id)

    assert source_unit_id.startswith("picture-observation-v1:")
    assert identity is not None
    assert identity.observation_id == "pobs:private/1"

    ref, content = picture_observation_ref_and_content(
        observation_id="pobs:private/1",
        payload_sha256="a" * 64,
        text="  visible semantics\n",
    )
    assert ref.source_type is SourceType.PICTURE
    assert ref.source_unit_id == source_unit_id
    assert ref.source_revision == "a" * 64
    assert content == "visible semantics"


@pytest.mark.parametrize(
    "source_unit_id",
    (
        "picture-observation-v1:",
        "picture-observation-v1:***",
        "picture-observation-v1:cG9icw==",
        "picture-observation-v1:cG9icw:extra",
        "picture-observation-v2:cG9icw",
    ),
)
def test_picture_observation_identity_parser_rejects_noncanonical_values(
    source_unit_id: str,
) -> None:
    assert parse_picture_observation_source_unit_id(source_unit_id) is None


def test_picture_observation_ref_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        picture_observation_ref_and_content(
            observation_id="pobs-1",
            payload_sha256="a" * 64,
            text="  \n",
        )
