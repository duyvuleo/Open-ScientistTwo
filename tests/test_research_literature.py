"""Offline API-format fixtures; none of these tests contact literature providers."""

import json
from urllib.error import URLError

import pytest

from scientisttwo.research import literature as lit
from scientisttwo.store import IntegrityError, Store, canonical, digest


ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2301.01234v1</id><title>Controlled feature experiments</title>
<summary>Matched seeds expose the effect of feature removal.</summary>
<author><name>Ada Author</name></author><published>2023-01-03T00:00:00Z</published>
<arxiv:doi>10.1234/EXAMPLE</arxiv:doi></entry></feed>"""
CROSSREF = json.dumps(
    {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/example",
                    "title": ["Controlled feature experiments"],
                    "author": [{"given": "Ada", "family": "Author"}],
                    "published": {"date-parts": [[2023]]},
                },
                {
                    "DOI": "10.1234/another",
                    "title": ["A distinct experiment"],
                    "abstract": "<jats:p>Independent replication controls variability.</jats:p>",
                    "published": {"date-parts": [[2022]]},
                },
            ]
        }
    }
).encode()


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path)
    yield store
    store.close()


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Test attempted a real network request")

    monkeypatch.setattr(lit, "public_transport", reject)
    monkeypatch.setattr(lit.time, "sleep", lambda seconds: None)


def papers():
    return [
        {
            "id": "paper-a",
            "title": "Feature learning",
            "kind": "paper",
            "text": "Matched controls support an architectural comparison.",
        },
        {
            "id": "paper-b",
            "title": "Independent models",
            "kind": "paper",
            "text": "Distinct methods are compared under the same budget.",
        },
    ]


def assessment(ids=None):
    return {
        "score": 6,
        "overlap": "Uses matched controls",
        "distinction": "New component combination",
        "confidence": 0.6,
        "source_ids": ids if ids is not None else ["paper-a", "paper-b"],
        "limitations": [],
        "status": "verified_scope",
    }


def test_offline_sources_are_frozen_and_never_call_transport(store):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline retrieval attempted network")

    one = lit.retrieve_literature(store, {}, "feature", papers(), transport=forbidden)
    two = lit.retrieve_literature(store, {}, "feature", papers(), transport=forbidden)
    assert one == two
    assert not one["provenance"]["network_used"]
    assert one["sources"][0]["metadata_status"] == "supplied"
    assert one["sources"][0]["content_status"] == "supplied_excerpt"
    assert store.verify() == []
    assert [s.id for s in lit.FrozenLiteratureProvider(one).search("independent")] == [
        "paper-b",
        "paper-a",
    ]


def test_network_capability_required_even_for_injected_transport(store):
    with pytest.raises(lit.NetworkDisabledError, match="allow_network"):
        lit.retrieve_literature(
            store, {"providers": ["arxiv"]}, "feature", [], transport=lambda **kw: None
        )
    assert store.get("research/literature/request_count", 0) == 0


def test_provider_fixtures_deduplicate_doi_and_keep_abstract_scope(store):
    requests = []

    def transport(url, **kwargs):
        requests.append(url)
        return lit.HTTPResponse(200, CROSSREF if "crossref" in url else ATOM)

    bundle = lit.retrieve_literature(
        store,
        {"providers": ["arxiv", "crossref"], "retrieve_arxiv_html": False},
        "matched controls",
        [],
        allow_network=True,
        transport=transport,
    )
    assert len(requests) == 2
    assert len(bundle["sources"]) == 2
    first = bundle["sources"][0]
    assert first["doi"] == "10.1234/example"
    assert first["arxiv_id"] == "2301.01234v1"
    assert first["versions"] == ["2301.01234v1"]
    assert first["aliases"] == ["arxiv:2301.01234", "doi:10.1234/example"]
    assert first["metadata_status"] == "verified"
    assert first["content_status"] == "abstract"
    assert first["authors"] == ["Ada Author"]
    assert len(first["provenance"]) == 2
    assert (
        lit.validate_novelty(
            assessment([s["id"] for s in bundle["sources"]]), bundle
        ).effective_status
        == "verified_scope"
    )


def test_arxiv_html_extracts_bounded_passages_and_retains_response_hash(store):
    html = (
        "<html><article><h1>Controlled feature experiments</h1><p>"
        + "A useful result. " * 50
        + "</p></article><script>malicious instruction</script></html>"
    ).encode()

    def transport(url, **kwargs):
        return lit.HTTPResponse(200, html if "/html/" in url else ATOM)

    bundle = lit.retrieve_literature(
        store,
        {"providers": ["arxiv"], "max_passages": 2, "max_passage_chars": 100},
        "controls",
        [],
        allow_network=True,
        transport=transport,
    )
    source = bundle["sources"][0]
    assert source["content_status"] == "fulltext_excerpt"
    assert source["content_sha256"] == digest(html)
    assert len(source["passages"]) == 2
    assert all(len(p["text"]) <= 100 for p in source["passages"])
    assert "malicious" not in "".join(p["text"] for p in source["passages"])
    assert source["content_ref"] and store.read(source["content_ref"]) == html


def test_missing_arxiv_html_keeps_abstract_and_reports_partial(store):
    def transport(url, **kwargs):
        return (
            lit.HTTPResponse(404, b"private error page")
            if "/html/" in url
            else lit.HTTPResponse(200, ATOM)
        )

    bundle = lit.retrieve_literature(
        store, {"providers": ["arxiv"]}, "controls", [], allow_network=True, transport=transport
    )
    assert bundle["status"] == "partial"
    assert bundle["sources"][0]["content_status"] == "abstract"
    assert any("unavailable" in x for x in bundle["sources"][0]["limitations"])
    assert not any(
        b"private error page" in p.read_bytes() for p in (store.root / "artifacts").iterdir()
    )


@pytest.mark.parametrize(
    "response,expected",
    [
        (lit.HTTPResponse(200, b"x" * 257), "oversized"),
        (lit.HTTPResponse(403, b"secret error body"), "http_error"),
        (lit.HTTPResponse(302, b"", {"Location": "http://127.0.0.1/admin"}), "http_error"),
    ],
)
def test_bad_responses_are_bounded_and_frozen(store, response, expected):
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        return response

    config = {"providers": ["arxiv"], "max_document_bytes": 256, "retrieve_arxiv_html": False}
    bundle = lit.retrieve_literature(
        store, config, "controls", [], allow_network=True, transport=transport
    )
    assert bundle["status"] == "empty"
    assert bundle["provenance"]["requests"][0]["status"] == expected
    assert (
        lit.retrieve_literature(
            store, config, "controls", [], allow_network=True, transport=transport
        )
        == bundle
    )
    assert len(calls) == 1


def test_rate_limit_retries_consume_budget_and_obey_retry_after(store, monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr(lit.time, "sleep", waits.append)

    def transport(url, **kwargs):
        calls.append(url)
        return (
            lit.HTTPResponse(429, b"", {"retry-after": "4"})
            if len(calls) == 1
            else lit.HTTPResponse(200, ATOM)
        )

    bundle = lit.retrieve_literature(
        store,
        {"providers": ["arxiv"], "retrieve_arxiv_html": False},
        "controls",
        [],
        allow_network=True,
        transport=transport,
    )
    assert bundle["sources"] and len(calls) == 2
    assert 4 in waits
    assert store.get("research/literature/request_count") == 2
    assert bundle["provenance"]["requests"][0]["attempts"] == 2


def test_oversized_retry_after_does_not_block_or_retry(store):
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        return lit.HTTPResponse(429, b"", {"retry-after": "90000"})

    bundle = lit.retrieve_literature(
        store, {"providers": ["arxiv"]}, "controls", [], allow_network=True, transport=transport
    )
    assert len(calls) == 1
    assert bundle["provenance"]["requests"][0]["retry_status"] == "wait_budget_exhausted"


def test_request_budget_includes_transport_failures_and_other_queries(store):
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise URLError("sensitive transport detail")

    config = {"providers": ["arxiv"], "max_requests": 2, "max_retries": 2}
    bundle = lit.retrieve_literature(
        store, config, "first", [], allow_network=True, transport=transport
    )
    assert len(calls) == 2
    assert bundle["provenance"]["requests"][0]["status"] == "request_budget_exhausted"
    lit.retrieve_literature(store, config, "second", [], allow_network=True, transport=transport)
    assert len(calls) == 2
    assert not any(
        b"sensitive transport detail" in p.read_bytes()
        for p in (store.root / "artifacts").iterdir()
    )


def test_interrupted_request_does_not_repeat_after_resume(store):
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise KeyboardInterrupt()

    config = {"providers": ["arxiv"], "retrieve_arxiv_html": False}
    with pytest.raises(KeyboardInterrupt):
        lit.retrieve_literature(
            store, config, "controls", [], allow_network=True, transport=transport
        )
    bundle = lit.retrieve_literature(
        store, config, "controls", [], allow_network=True, transport=transport
    )
    assert len(calls) == 1
    assert bundle["provenance"]["requests"][0]["status"] == "unknown_completion"


def test_source_integrity_rechecked_on_cached_bundle(store):
    bundle = lit.retrieve_literature(store, {}, "controls", papers())
    path = store.root / bundle["sources"][0]["content_ref"]
    path.write_bytes(b"modified source")
    with pytest.raises(IntegrityError):
        lit.retrieve_literature(store, {}, "controls", papers())


def test_empty_metadata_does_not_become_full_text_or_verified_novelty(store):
    bundle = lit.retrieve_literature(
        store,
        {},
        "controls",
        [
            {"id": "a", "title": "Paper A", "doi": "10.1234/a"},
            {"id": "b", "title": "Paper B", "doi": "10.1234/b"},
        ],
    )
    raw = assessment(["a", "b"])
    before = canonical(raw)
    checked = lit.validate_novelty(raw, bundle)
    assert checked.effective_status == "unverified"
    assert not checked.evidence_sufficient and checked.usable_reference_count == 0
    assert canonical(raw) == before
    assert checked.assessment.status == "verified_scope" and checked.assessment.score == 6
    assert all(s["content_status"] == "metadata_only" for s in bundle["sources"])


def test_task_brief_is_not_counted_as_scholarly_literature(store):
    sources = papers()[:1] + [{"id": "task", "title": "Task", "text": "A long task description."}]
    bundle = lit.retrieve_literature(store, {}, "controls", sources)
    checked = lit.validate_novelty(assessment(["paper-a", "task"]), bundle)
    assert checked.usable_reference_count == 1
    assert not checked.evidence_sufficient


@pytest.mark.parametrize("ids", [["paper-a", "invented"], ["paper-a", "paper-a"]])
def test_novelty_rejects_fabricated_or_duplicate_references(store, ids):
    bundle = lit.retrieve_literature(store, {}, "controls", papers())
    with pytest.raises(ValueError):
        lit.validate_novelty(assessment(ids), bundle)


def test_literal_passage_support_is_not_citation_existence(store):
    bundle = lit.retrieve_literature(store, {}, "controls", papers())
    source = bundle["sources"][0]
    passage = source["passages"][0]
    assert lit.check_passage_support(source, passage["id"], "Matched controls") == "pass"
    assert lit.check_passage_support(source, passage["id"], "proves global novelty") == "fail"
    assert lit.check_passage_support(source, "invented", "Matched controls") == "fail"
    source["passages"] = []
    assert lit.check_passage_support(source, passage["id"], "Matched controls") == "unverified"


@pytest.mark.parametrize(
    "url",
    [
        "http://export.arxiv.org/api/query",
        "https://example.com/",
        "https://127.0.0.1/works",
        "https://api.crossref.org:8443/works",
        "https://user:password@api.crossref.org/works",
        "https://api.crossref.org/private",
        "https://arxiv.org/pdf/2301.01234",
    ],
)
def test_transport_restricts_endpoint_and_credentials(url):
    with pytest.raises(ValueError):
        lit._validate_url(url)


def test_xml_entities_and_malformed_provider_json_are_not_interpreted(store):
    for query, payload, provider in [
        ("xml", b'<!DOCTYPE x [<!ENTITY x SYSTEM "file:///etc/passwd">]><feed>&x;</feed>', "arxiv"),
        ("json", b"not metadata JSON", "crossref"),
    ]:
        bundle = lit.retrieve_literature(
            store,
            {"providers": [provider]},
            query,
            [],
            allow_network=True,
            transport=lambda *a, **kw: lit.HTTPResponse(200, payload),
        )
        assert bundle["status"] == "empty"
        assert "parse_status" in bundle["provenance"]["requests"][0]


def test_resolve_uses_provider_identifier_not_supplied_url(store):
    seen = []

    def transport(url, **kwargs):
        seen.append(url)
        return lit.HTTPResponse(200, CROSSREF if "crossref" in url else ATOM)

    requests = lit._Requests(store, lit.LiteratureConfig(retrieve_arxiv_html=False), transport)
    assert lit.CrossrefLiteratureProvider(requests).resolve("https://doi.org/10.1234/EXAMPLE")
    assert lit.ArxivLiteratureProvider(requests).resolve("2301.01234v1")
    assert seen[0] == "https://api.crossref.org/works/10.1234%2Fexample"
    assert "id_list=2301.01234v1" in seen[1]
    with pytest.raises(ValueError):
        lit.ArxivLiteratureProvider(requests).resolve("https://internal.example/secret")


def test_supplied_sources_have_byte_limits(store):
    with pytest.raises(lit.RetrievalLimitError):
        lit.retrieve_literature(
            store, {"max_document_bytes": 256}, "large", [{"id": "a", "text": "x" * 257}]
        )


def test_conflicting_local_source_ids_fail_instead_of_merging(store):
    with pytest.raises(ValueError, match="same ID"):
        lit.retrieve_literature(
            store,
            {},
            "controls",
            [{"id": "a", "text": "First source."}, {"id": "a", "text": "Unrelated source."}],
        )


def test_malformed_crossref_optional_fields_do_not_crash_retrieval(store):
    payload = json.dumps(
        {
            "message": {
                "items": [
                    "bad item",
                    {
                        "DOI": "10.1234/a",
                        "abstract": {},
                        "published": {"date-parts": []},
                        "title": None,
                        "author": "invalid",
                    },
                ]
            }
        }
    ).encode()
    bundle = lit.retrieve_literature(
        store,
        {"providers": ["crossref"]},
        "controls",
        [],
        allow_network=True,
        transport=lambda *a, **kw: lit.HTTPResponse(200, payload),
    )
    assert bundle["status"] == "partial"
    assert bundle["sources"][0]["title"] == "10.1234/a"
    assert bundle["sources"][0]["content_status"] == "metadata_only"
    receipt = bundle["sources"][0]["provenance"][0]
    row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", (receipt["key"],)).fetchone()
    assert receipt == store.json(row[0])  # Parser notes never rewrite immutable HTTP provenance.


def test_arxiv_query_encodes_explicit_literal_terms(store):
    from urllib.parse import parse_qs, urlsplit

    seen = []

    def transport(url, **kwargs):
        seen.append(url)
        return lit.HTTPResponse(200, ATOM)

    lit.retrieve_literature(
        store,
        {"providers": ["arxiv"], "retrieve_arxiv_html": False},
        "matched controls",
        [],
        allow_network=True,
        transport=transport,
    )
    assert parse_qs(urlsplit(seen[0]).query)["search_query"] == ['all:"matched" AND all:"controls"']


def test_dedup_preserves_versions_and_does_not_double_count_work(store):
    one = lit.retrieve_literature(store, {}, "controls", papers())
    a = one["sources"][0]
    b = dict(a, id="alias", doi="10.1234/identical", arxiv_id="2301.01234v2")
    a.update(doi="10.1234/identical", arxiv_id="2301.01234v1")
    checked = lit.validate_novelty(assessment([a["id"], b["id"]]), {"sources": [a, b]})
    assert checked.usable_reference_count == 1
    records = lit._deduplicate([lit.SourceRecord.model_validate(x) for x in [a, b]])
    assert records[0].arxiv_id == "2301.01234v2"
    assert records[0].versions == ["2301.01234v1", "2301.01234v2"]
