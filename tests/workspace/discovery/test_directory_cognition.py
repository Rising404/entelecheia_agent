"""两个目录认知工具的行为。

这里的每项测试都对应 `24/001 D5` 冻结的三条可靠性规则之一——确定性、显式
截断、可区分失败；被替换的工具正是未能提供这些性质。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.workspace.discovery import (
    FindResult,
    MatchKind,
    SkipReason,
    build_overview,
    find,
    list_files,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "alpha.md").write_text("shared marker here\n", encoding="utf-8")
    (tmp_path / "docs" / "beta.md").write_text("shared marker twice\nshared marker\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('shared marker')\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("nothing of interest\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("top level\n", encoding="utf-8")
    return tmp_path


def test_repeating_a_search_returns_an_identical_page(tree: Path):
    first = find(tree, content="shared marker", limit=2)
    second = find(tree, content="shared marker", limit=2)
    assert first.hits == second.hits
    assert first.total_matched == second.total_matched


def test_a_capped_page_says_so_and_reports_the_true_total(tree: Path):
    full = find(tree, content="shared marker", limit=50)
    assert full.truncated is False

    page = find(tree, content="shared marker", limit=2)
    assert page.returned == 2
    assert page.truncated is True
    assert page.total_matched == full.total_matched > 2


def test_paging_walks_the_same_ordering_without_repeats(tree: Path):
    first = find(tree, content="shared marker", limit=2, offset=0)
    second = find(tree, content="shared marker", limit=2, offset=2)
    assert not set(first.hits) & set(second.hits)
    assert second.offset == 2


def test_a_denied_file_is_counted_not_silently_dropped(tmp_path: Path, monkeypatch):
    """被策略拒绝的路径必须以排除项的形式显式呈现。

    返回该路径会泄露秘密；静默丢弃则会让模型误以为目录中完全没有配置。
    """

    (tmp_path / "notes.md").write_text("api marker\n", encoding="utf-8")
    (tmp_path / ".env").write_text("api marker=secret\n", encoding="utf-8")
    from personagraph.configuration import paths
    monkeypatch.setattr(paths, "API_TOKEN_PATH", tmp_path / ".env")

    result = find(tmp_path, content="api marker", hidden=True, respect_ignore=False)
    assert [hit.rel_path for hit in result.hits] == ["notes.md"]
    assert dict(result.skipped)[SkipReason.DENIED_BY_POLICY] == 1


def test_no_matches_is_not_the_same_shape_as_an_exclusion(tree: Path):
    empty = find(tree, content="no such string anywhere")
    assert empty.returned == 0
    assert empty.total_matched == 0
    assert empty.skipped == ()
    assert empty.scanned_files > 0


def test_name_and_content_matches_stay_distinguishable(tree: Path):
    by_name = find(tree, name="*.md")
    assert by_name.returned == 3
    assert {hit.match for hit in by_name.hits} == {MatchKind.NAME}
    assert all(hit.line is None for hit in by_name.hits)

    by_content = find(tree, content="shared marker")
    assert {hit.match for hit in by_content.hits} == {MatchKind.CONTENT}
    assert all(hit.line and hit.snippet for hit in by_content.hits)


def test_sorting_by_mtime_puts_the_newest_first(tree: Path):
    import os
    import time

    newest = tree / "docs" / "alpha.md"
    os.utime(newest, (time.time() + 60, time.time() + 60))
    result = find(tree, name="*.md", sort="mtime")
    assert result.hits[0].rel_path == "docs/alpha.md"


def test_a_subpath_may_not_escape_the_root(tree: Path):
    with pytest.raises(ValueError):
        find(tree, name="*.md", subpath="..")


def test_a_request_without_a_pattern_is_rejected(tree: Path):
    with pytest.raises(ValueError):
        find(tree)


def test_ignore_rules_are_applied_but_reported(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.txt").write_text("marker\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("marker\n", encoding="utf-8")

    honoured = find(tmp_path, content="marker")
    assert [hit.rel_path for hit in honoured.hits] == ["keep.txt"]
    assert honoured.ignore_rules_applied is True

    everything = find(tmp_path, content="marker", respect_ignore=False)
    assert len(everything.hits) == 2
    assert everything.ignore_rules_applied is False


def test_overview_aggregates_by_directory_rather_than_listing_files(tree: Path):
    overview = build_overview(tree, depth=1)
    by_path = {entry.rel_path: entry for entry in overview.entries}
    assert by_path["docs"].files == 2
    assert by_path["src"].files == 2
    assert by_path["."].files == 1
    assert dict(by_path["src"].kinds)[".py"] == 2
    assert overview.total_files == 5
    assert overview.truncated is False


def test_a_capped_overview_reports_what_it_left_out(tree: Path):
    overview = build_overview(tree, depth=1, max_entries=1)
    assert overview.truncated is True
    assert overview.omitted_dirs == 2
    assert len(overview.entries) == 1
    assert overview.total_dirs == 3


def test_the_result_type_rejects_a_dishonest_truncation_claim():
    with pytest.raises(ValueError):
        FindResult(
            hits=(), total_matched=0, offset=0, truncated=True,
            scanned_files=1, elapsed_ms=0,
        )


def test_file_listing_can_stop_at_a_deterministic_explicit_bound(
    tmp_path: Path,
):
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    paths, report = list_files(tmp_path, max_paths=2)

    assert paths == ["a.txt", "b.txt"]
    assert report.scanned_files == 2
    assert report.truncated is True


def test_combined_content_search_uses_one_wall_clock_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import personagraph.workspace.discovery.search as module
    from personagraph.workspace.discovery import ScanReport

    observed_timeouts: list[float] = []
    ticks = iter((10.0, 10.1, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    def fake_list_files(_root, **kwargs):
        observed_timeouts.append(kwargs["timeout_s"])
        return [], ScanReport()

    def fake_search_content(_root, _content, **kwargs):
        observed_timeouts.append(kwargs["timeout_s"])
        return [], 0, ScanReport()

    monkeypatch.setattr(module, "list_files", fake_list_files)
    monkeypatch.setattr(module, "search_content", fake_search_content)

    result = module.find(tmp_path, content="needle", timeout_s=1.0)

    assert result.total_matched == 0
    assert observed_timeouts == pytest.approx([0.9, 0.4])
