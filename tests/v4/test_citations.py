"""Tests for engine_v4.citations."""
import json
from pathlib import Path
from unittest.mock import patch

from engine_v4 import citations
from engine_v4.types import Chapter, Source, SourceSpec


def test_find_references_extracts_etal_and_paren():
    text = (
        "As shown by Smith et al. 2020, the result holds. "
        "Earlier work (Jones and Brown, 2018) confirmed it. "
        '"Some Cool Paper Title" (2019) is also relevant.'
    )
    refs = citations._find_references(text)
    assert any("Smith et al" in r for r in refs)
    assert any("Jones and Brown" in r for r in refs)
    assert any("Some Cool Paper Title" in r for r in refs)


def test_inject_footnotes_round_trip():
    text = "The Smith et al. 2020 paper proves it."
    cits = [{
        "ref_str": "Smith et al. 2020",
        "title": "On Things",
        "authors": "J. Smith",
        "year": "2020",
        "url": "http://arxiv.org/abs/1234.5678",
    }]
    out = citations._inject_footnotes(text, cits)
    assert "Smith et al. 2020[^1]" in out
    assert "[^1]: J. Smith On Things (2020). <http://arxiv.org/abs/1234.5678>" in out


def test_inject_footnotes_empty_is_noop():
    text = "no refs here"
    assert citations._inject_footnotes(text, []) == text


def _mk_source(tmp_path: Path) -> Source:
    ws = tmp_path / "src"
    (ws / "citations").mkdir(parents=True)
    return Source(
        source_id="src",
        spec=SourceSpec(location="x"),
        workspace=ws,
        media_path=None,
        duration_sec=0.0,
    )


def test_enrich_uses_cache_when_present(tmp_path):
    source = _mk_source(tmp_path)
    cached = [{
        "ref_str": "Smith et al. 2020",
        "title": "Cached Title",
        "authors": "S",
        "year": "2020",
        "url": "http://example.com",
    }]
    (source.workspace / "citations" / "ch_001.json").write_text(json.dumps(cached))

    chapter = Chapter(id="ch_001", title="T", body_md="Smith et al. 2020 said so.")
    with patch.object(citations, "_resolve_citation") as resolve:
        out = citations.enrich(chapter, source)
        resolve.assert_not_called()
    assert out.citations and out.citations[0].title == "Cached Title"
    assert "[^1]" in out.body_md


def test_enrich_resolves_and_writes_cache(tmp_path):
    source = _mk_source(tmp_path)
    chapter = Chapter(id="ch_002", title="T", body_md="Smith et al. 2020 said so.")
    fake = {"title": "Found", "authors": "A", "year": "2020", "url": "u"}
    with patch.object(citations, "_resolve_citation", return_value=fake) as resolve:
        out = citations.enrich(chapter, source)
        assert resolve.call_count == 1
    cache = source.workspace / "citations" / "ch_002.json"
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert data and data[0]["title"] == "Found"
    assert out.citations[0].title == "Found"


def test_enrich_swallows_resolve_failures(tmp_path):
    source = _mk_source(tmp_path)
    chapter = Chapter(id="ch_003", title="T", body_md="Smith et al. 2020 said so.")
    with patch.object(citations, "_resolve_citation", side_effect=RuntimeError("net down")):
        out = citations.enrich(chapter, source)
    assert out.citations == []
    assert "[^1]" not in out.body_md
