"""Controller-owned, bounded literature retrieval and scoped novelty evidence.

Network requests are opt-in, use fixed public provider endpoints, and never run in
candidate containers. A frozen bundle distinguishes metadata, accessible excerpts
and scientific interpretation. Neither retrieval nor literal passage checks prove
novelty or entailment. Pending requests survive interruption as unknown completion;
resuming does not silently repeat them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
import time
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import xml.etree.ElementTree as ET

from pydantic import Field, model_validator

from ..models import Record
from ..store import Store, canonical, digest


SCHEMA_VERSION = "research-literature-v1"
PROVIDER_VERSION = "public-api-v1"
_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_DOI = re.compile(r"10\.\d{4,9}/[^\s<>]+", re.IGNORECASE)
_ARXIV = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


class LiteratureConfig(Record):
    schema_version: Literal["research-literature-v1"] = SCHEMA_VERSION
    providers: list[Literal["local", "arxiv", "crossref"]] = ["local"]
    max_results_per_provider: int = Field(default=4, ge=1, le=20)
    max_sources: int = Field(default=8, ge=1, le=40)
    max_requests: int = Field(default=16, ge=1, le=200)
    max_document_bytes: int = Field(default=2_000_000, ge=256, le=10_000_000)
    max_passages: int = Field(default=4, ge=1, le=10)
    max_passage_chars: int = Field(default=2000, ge=100, le=10_000)
    request_timeout_seconds: float = Field(default=20, gt=0, le=60)
    max_retries: int = Field(default=1, ge=0, le=2)
    retry_backoff_seconds: float = Field(default=1, ge=0, le=10)
    max_wait_seconds: float = Field(default=30, ge=3, le=60)
    retrieve_arxiv_html: bool = True
    minimum_references: int = Field(default=2, ge=2, le=10)

    @model_validator(mode="after")
    def unique_providers(self):
        if len(self.providers) != len(set(self.providers)):
            raise ValueError("Literature providers must be unique")
        return self


class SourcePassage(Record):
    id: str
    text: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope: Literal["abstract", "fulltext_excerpt", "supplied_excerpt"]

    @model_validator(mode="after")
    def hash_matches(self):
        if digest(self.text.encode()) != self.sha256:
            raise ValueError("Source passage hash mismatch")
        return self


class SourceRecord(Record):
    schema_version: Literal["research-source-v1"] = "research-source-v1"
    id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    provider: Literal["local", "arxiv", "crossref"]
    scholarly: bool = False
    retrieved_at: str
    version: str | None = None
    metadata_status: Literal["verified", "supplied", "unverified"]
    content_status: Literal["metadata_only", "abstract", "fulltext_excerpt", "supplied_excerpt"] = (
        "metadata_only"
    )
    passages: list[SourcePassage] = []
    content_ref: str | None = None
    content_sha256: str | None = None
    provenance: list[dict] = []
    limitations: list[str] = []
    aliases: list[str] = []
    versions: list[str] = []


class NoveltyAssessment(Record):
    schema_version: Literal["research-novelty-v1"] = "research-novelty-v1"
    score: float = Field(ge=0, le=10)
    overlap: str = Field(min_length=1)
    distinction: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source_ids: list[str] = []
    limitations: list[str] = []
    status: Literal["verified_scope", "unverified"] = "unverified"


class NoveltyValidation(Record):
    assessment: NoveltyAssessment
    effective_status: Literal["verified_scope", "unverified"]
    evidence_sufficient: bool
    usable_reference_count: int
    limitations: list[str] = []


class NetworkDisabledError(ValueError):
    """The campaign did not explicitly authorize literature networking."""


class RetrievalLimitError(ValueError):
    pass


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def __call__(self, url: str, *, timeout: float, max_bytes: int) -> HTTPResponse: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_doi(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip())
    return value.lower() if _DOI.fullmatch(value) else None


def _normalize_arxiv(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    value = re.sub(r"^(?:https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/|arxiv:)", "", value)
    value = value.removesuffix(".pdf").strip()
    return value if _ARXIV.fullmatch(value) else None


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("Only public HTTPS provider endpoints are permitted")
    permitted = (
        parsed.hostname == "export.arxiv.org"
        and parsed.path == "/api/query"
        or parsed.hostname == "arxiv.org"
        and parsed.path.startswith("/html/")
        or parsed.hostname == "api.crossref.org"
        and (parsed.path == "/works" or parsed.path.startswith("/works/"))
    )
    if not permitted:
        raise ValueError("Unapproved literature endpoint")


class _NoRedirect(HTTPRedirectHandler):
    # Redirects are reported as unavailable, never followed to arbitrary sites.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_transport(url: str, *, timeout: float, max_bytes: int) -> HTTPResponse:
    _validate_url(url)
    request = Request(
        url, headers={"User-Agent": "ScientistTwo-research/1.0", "Accept-Encoding": "identity"}
    )
    # Disable ambient proxies and authentication handlers; credentials are never forwarded.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and (not length.isdigit() or int(length) > max_bytes):
                raise RetrievalLimitError("Document exceeds byte limit")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise RetrievalLimitError("Document exceeds byte limit")
            return HTTPResponse(
                response.status, data, {"content-type": response.headers.get("Content-Type", "")}
            )
    except HTTPError as exc:
        # Do not retain error pages, transport credentials, cookies, or exception strings.
        try:
            return HTTPResponse(exc.code, b"", {"retry-after": exc.headers.get("Retry-After", "")})
        finally:
            exc.close()


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0
        self.has_article = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav"}:
            self.suppressed += 1
        attr = dict(attrs)
        if tag in {"article", "main"} or "ltx_document" in attr.get("class", ""):
            self.has_article = True
        if tag in {"p", "div", "section", "h1", "h2", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav"} and self.suppressed:
            self.suppressed -= 1

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)

    @property
    def text(self):
        return "\n".join(
            " ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip()
        )


def _text(value: str) -> str:
    parser = _Text()
    parser.feed(value)
    return parser.text


def _passages(text: str, scope: str, config: LiteratureConfig) -> list[SourcePassage]:
    # Excerpts are explicit: do not claim a complete document was read.
    text = text.strip()
    passages = []
    for index in range(config.max_passages):
        fragment = text[index * config.max_passage_chars : (index + 1) * config.max_passage_chars]
        if not fragment:
            break
        sha = digest(fragment.encode())
        passages.append(
            SourcePassage(id=f"p{index + 1}-{sha[:12]}", text=fragment, sha256=sha, scope=scope)
        )
    return passages


class _Requests:
    def __init__(self, store: Store, config: LiteratureConfig, transport: Transport):
        self.store, self.config, self.transport = store, config, transport
        self.waited = 0.0
        self.receipts: list[dict] = []

    def get(self, url: str) -> dict:
        _validate_url(url)
        inputs = {
            "url": url,
            "version": PROVIDER_VERSION,
            "max_bytes": self.config.max_document_bytes,
            "timeout": self.config.request_timeout_seconds,
            "retries": self.config.max_retries,
        }
        key = "research/literature/request/" + digest(canonical(inputs))
        cached = self.store.cached(key, inputs)
        if cached is not None:
            if cached.get("body_ref"):
                self.store.read(cached["body_ref"])
            self.receipts.append(cached)
            return cached
        pending = self.store.get(key)
        if pending is not None:
            result = {
                "key": key,
                "url": url,
                "status": "unknown_completion",
                "retrieved_at": pending["started"],
            }
            self.receipts.append(result)
            return result
        provider = urlsplit(url).hostname
        result: dict = {"key": key, "url": url}
        for attempt in range(self.config.max_retries + 1):
            used = self.store.get("research/literature/request_count", 0)
            if used >= self.config.max_requests:
                result.update(status="request_budget_exhausted", retrieved_at=_now())
                break
            previous = self.store.get("research/literature/last_request/" + provider, 0)
            interval = 3.0 if "arxiv" in provider else 1.0
            wait = max(0.0, interval - (time.time() - previous))
            if self.waited + wait > self.config.max_wait_seconds:
                result.update(status="wait_budget_exhausted", retrieved_at=_now())
                break
            if wait:
                time.sleep(wait)
                self.waited += wait
            # Commit admission before external I/O. An interrupted admission is not retried.
            started = _now()
            with self.store.db:
                for name, value in (
                    (key, {"started": started, "attempt": attempt}),
                    ("research/literature/request_count", used + 1),
                    ("research/literature/last_request/" + provider, time.time()),
                ):
                    self.store.db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                        (name, canonical(value).decode()),
                    )
            result.update(retrieved_at=started, attempts=attempt + 1)
            try:
                response = self.transport(
                    url,
                    timeout=self.config.request_timeout_seconds,
                    max_bytes=self.config.max_document_bytes,
                )
                if len(response.body) > self.config.max_document_bytes:
                    raise RetrievalLimitError("Document exceeds byte limit")
                result["http_status"] = response.status
                if 200 <= response.status < 300:
                    ref = self.store.put(response.body, suffix="bin", kind="literature-response")
                    result.update(status="retrieved", body_ref=ref, sha256=digest(response.body))
                    break
                retryable = response.status in {429, 500, 502, 503, 504}
                result["status"] = "rate_limited" if response.status == 429 else "http_error"
                retry_after = response.headers.get(
                    "retry-after", response.headers.get("Retry-After", "")
                )
                try:
                    delay = max(
                        self.config.retry_backoff_seconds * (2**attempt), float(retry_after or 0)
                    )
                except ValueError:
                    # An HTTP-date Retry-After is not guessed; wait until a later explicit run.
                    retryable, delay = False, 0
            except RetrievalLimitError:
                result.update(status="oversized")
                break
            except (URLError, OSError, TimeoutError):
                result.update(status="transport_error")
                retryable, delay = True, self.config.retry_backoff_seconds * (2**attempt)
            if not retryable or attempt >= self.config.max_retries:
                break
            if self.waited + delay > self.config.max_wait_seconds:
                result["retry_status"] = "wait_budget_exhausted"
                break
            time.sleep(delay)
            self.waited += delay
        result = self.store.cache(key, inputs, result)
        self.receipts.append(result)
        self.store.event(key, "literature", "Public literature retrieval: " + result["status"])
        return result


class LiteratureProvider(Protocol):
    def search(self, query: str) -> list[SourceRecord]: ...


class LocalLiteratureProvider:
    def __init__(self, store: Store, config: LiteratureConfig, sources: list[dict]):
        self.store, self.config, self.sources = store, config, sources

    def search(self, query: str) -> list[SourceRecord]:
        terms = set(query.lower().split())
        ordered = sorted(
            self.sources, key=lambda s: -len(terms & set(str(s.get("title", "")).lower().split()))
        )
        results = []
        seen_ids = {}
        for item in ordered[: self.config.max_sources]:
            if item.get("id") in seen_ids and seen_ids[item["id"]] != digest(canonical(item)):
                raise ValueError("Conflicting supplied source records use the same ID")
            if item.get("id"):
                seen_ids[item["id"]] = digest(canonical(item))
            if "schema_version" in item and item["schema_version"] == "research-source-v1":
                record = SourceRecord.model_validate(item)
                if record.content_ref:
                    self.store.read(record.content_ref)
                results.append(record)
                continue
            content = str(item.get("text", item.get("abstract", "")))
            if len(content.encode()) > self.config.max_document_bytes:
                raise RetrievalLimitError("Supplied source exceeds document byte limit")
            doi, arxiv = _normalize_doi(item.get("doi")), _normalize_arxiv(item.get("arxiv_id"))
            title = str(item.get("title", "Untitled supplied source"))
            source_id = str(
                item.get("id")
                or (
                    "doi:" + doi
                    if doi
                    else "arxiv:" + arxiv
                    if arxiv
                    else "local:" + digest(canonical(item))[:20]
                )
            )
            scope = "abstract" if "abstract" in item and "text" not in item else "supplied_excerpt"
            ref = (
                self.store.put(content.encode(), suffix="txt", kind="literature-source")
                if content
                else None
            )
            results.append(
                SourceRecord(
                    id=source_id,
                    title=title,
                    authors=item.get("authors", []),
                    year=item.get("year"),
                    doi=doi,
                    arxiv_id=arxiv,
                    url=item.get("url"),
                    provider="local",
                    scholarly=bool(doi or arxiv or item.get("kind") == "paper"),
                    retrieved_at=_now(),
                    version=item.get("version"),
                    metadata_status="supplied",
                    content_status=scope if content else "metadata_only",
                    passages=_passages(content, scope, self.config),
                    content_ref=ref,
                    content_sha256=digest(content.encode()) if content else None,
                    provenance=[
                        {
                            "provider": "local",
                            "status": "supplied",
                            "input_sha256": digest(canonical(item)),
                        }
                    ],
                    limitations=["Supplied source metadata has not been independently resolved."],
                )
            )
        return results


class ArxivLiteratureProvider:
    def __init__(self, requests: _Requests):
        self.requests = requests

    def search(self, query: str) -> list[SourceRecord]:
        config = self.requests.config
        # A plain query becomes escaped terms, not executable provider search syntax.
        terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)
        if not terms:
            return []
        params = {
            "search_query": " AND ".join('all:"' + term + '"' for term in terms),
            "start": 0,
            "max_results": config.max_results_per_provider,
        }
        return self._query("https://export.arxiv.org/api/query?" + urlencode(params))

    def resolve(self, identifier: str) -> list[SourceRecord]:
        identifier = _normalize_arxiv(identifier)
        if not identifier:
            raise ValueError("Invalid arXiv identifier")
        return self._query(
            "https://export.arxiv.org/api/query?" + urlencode({"id_list": identifier})
        )

    def _query(self, url: str) -> list[SourceRecord]:
        receipt = self.requests.get(url)
        if receipt["status"] != "retrieved":
            return []
        payload = self.requests.store.read(receipt["body_ref"])
        if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
            receipt["parse_status"] = "unsafe_xml"
            return []
        try:
            feed = ET.fromstring(payload)
        except ET.ParseError:
            receipt["parse_status"] = "invalid_xml"
            return []
        sources = []
        config = self.requests.config
        for entry in feed.findall("atom:entry", _ATOM)[: config.max_results_per_provider]:
            identifier = _normalize_arxiv(entry.findtext("atom:id", namespaces=_ATOM))
            if not identifier:
                continue
            title = " ".join((entry.findtext("atom:title", "", _ATOM)).split())
            abstract = " ".join(entry.findtext("atom:summary", "", _ATOM).split())
            doi = _normalize_doi(entry.findtext("arxiv:doi", namespaces=_ATOM))
            published = entry.findtext("atom:published", "", _ATOM)
            version = re.search(r"v\d+$", identifier)
            source = SourceRecord(
                id="arxiv:" + re.sub(r"v\d+$", "", identifier),
                title=title,
                authors=[
                    author.findtext("atom:name", "", _ATOM)
                    for author in entry.findall("atom:author", _ATOM)
                ],
                year=int(published[:4]) if published[:4].isdigit() else None,
                doi=doi,
                arxiv_id=identifier,
                url="https://arxiv.org/abs/" + identifier,
                provider="arxiv",
                scholarly=True,
                retrieved_at=receipt["retrieved_at"],
                version=version.group() if version else None,
                metadata_status="verified",
                content_status="abstract" if abstract else "metadata_only",
                passages=_passages(abstract, "abstract", config),
                content_ref=receipt["body_ref"],
                content_sha256=receipt["sha256"],
                provenance=[{k: v for k, v in receipt.items() if k != "parse_status"}],
                limitations=[
                    "Only bounded accessible passages are supplied; retrieval is not a full-paper review."
                ],
            )
            if config.retrieve_arxiv_html:
                html = self.requests.get("https://arxiv.org/html/" + identifier)
                source.provenance.append(dict(html))
                if html["status"] == "retrieved":
                    parser = _Text()
                    parser.feed(
                        self.requests.store.read(html["body_ref"]).decode("utf-8", errors="replace")
                    )
                    if parser.has_article and len(parser.text) >= 200:
                        source.passages = _passages(parser.text, "fulltext_excerpt", config)
                        source.content_status = "fulltext_excerpt"
                        source.content_ref, source.content_sha256 = html["body_ref"], html["sha256"]
                    else:
                        source.limitations.append(
                            "Retrieved HTML did not expose a usable article body."
                        )
                else:
                    source.limitations.append(
                        "Article HTML unavailable; accessible abstract retained."
                    )
            sources.append(source)
        return sources


class CrossrefLiteratureProvider:
    def __init__(self, requests: _Requests):
        self.requests = requests

    def search(self, query: str) -> list[SourceRecord]:
        params = {
            "query.bibliographic": query,
            "rows": self.requests.config.max_results_per_provider,
        }
        return self._query("https://api.crossref.org/works?" + urlencode(params))

    def resolve(self, doi: str) -> list[SourceRecord]:
        normalized = _normalize_doi(doi)
        if not normalized:
            raise ValueError("Invalid DOI")
        return self._query("https://api.crossref.org/works/" + quote(normalized, safe=""))

    def _query(self, url: str) -> list[SourceRecord]:
        receipt = self.requests.get(url)
        if receipt["status"] != "retrieved":
            return []
        try:
            data = json.loads(self.requests.store.read(receipt["body_ref"]))
            message = data["message"]
            items = message.get("items", [message])
            if not isinstance(items, list):
                raise ValueError("Unexpected metadata response")
        except (ValueError, KeyError, TypeError, AttributeError):
            receipt["parse_status"] = "invalid_json"
            return []
        results = []
        for item in items[: self.requests.config.max_results_per_provider]:
            if not isinstance(item, dict):
                receipt["parse_status"] = "invalid_metadata_record"
                continue
            doi = _normalize_doi(item.get("DOI"))
            if not doi:
                continue
            abstract = (
                _text(item.get("abstract", "")) if isinstance(item.get("abstract", ""), str) else ""
            )
            published = item.get("published", {})
            parts = published.get("date-parts", []) if isinstance(published, dict) else []
            year = (
                parts[0][0]
                if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]
                else None
            )
            titles = item.get("title", [])
            title = (
                titles[0]
                if isinstance(titles, list) and titles and isinstance(titles[0], str)
                else doi
            )
            authors = item.get("author", [])
            authors = authors if isinstance(authors, list) else []
            names = [
                " ".join(str(a[key]) for key in ("given", "family") if isinstance(a.get(key), str))
                for a in authors
                if isinstance(a, dict)
            ]
            results.append(
                SourceRecord(
                    id="doi:" + doi,
                    title=title,
                    authors=names,
                    year=year if isinstance(year, int) else None,
                    doi=doi,
                    url="https://doi.org/" + doi,
                    provider="crossref",
                    scholarly=True,
                    retrieved_at=receipt["retrieved_at"],
                    metadata_status="verified",
                    content_status="abstract" if abstract else "metadata_only",
                    passages=_passages(abstract, "abstract", self.requests.config),
                    content_ref=receipt["body_ref"],
                    content_sha256=receipt["sha256"],
                    provenance=[{k: v for k, v in receipt.items() if k != "parse_status"}],
                    limitations=[
                        "Crossref metadata resolution does not establish full-text access or claim support."
                    ],
                )
            )
        return results


def _deduplicate(sources: list[SourceRecord]) -> list[SourceRecord]:
    groups: list[list[SourceRecord]] = []
    keys: list[set[str]] = []
    for source in sources:
        identifiers = {"id:" + source.id}
        if source.doi:
            identifiers.add("doi:" + source.doi)
        if source.arxiv_id:
            identifiers.add("arxiv:" + re.sub(r"v\d+$", "", source.arxiv_id))
        matches = [i for i, known in enumerate(keys) if identifiers & known]
        group = [source]
        for i in reversed(matches):
            group.extend(groups.pop(i))
            identifiers |= keys.pop(i)
        groups.append(group)
        keys.append(identifiers)
    result = []
    quality = {"metadata_only": 0, "abstract": 1, "supplied_excerpt": 2, "fulltext_excerpt": 3}
    for group in groups:

        def rank(s):
            version = re.search(r"v(\d+)$", s.arxiv_id or "")
            return quality[s.content_status], int(version[1]) if version else 0

        chosen = max(group, key=rank).model_copy(deep=True)
        chosen.aliases = sorted({s.id for s in group} | {a for s in group for a in s.aliases})
        chosen.versions = sorted(
            {s.arxiv_id for s in group if s.arxiv_id} | {v for s in group for v in s.versions}
        )
        chosen.doi = next((s.doi for s in group if s.doi), None)
        chosen.arxiv_id = chosen.arxiv_id or next((s.arxiv_id for s in group if s.arxiv_id), None)
        chosen.provenance = [p for s in group for p in s.provenance]
        # Verification is about the same resolved identity, not whether a local excerpt is true.
        if any(s.metadata_status == "verified" for s in group):
            chosen.metadata_status = "verified"
        result.append(chosen)
    return result


def retrieve_literature(
    store: Store,
    config: LiteratureConfig | dict,
    query: str,
    local_sources: list[dict],
    *,
    allow_network: bool = False,
    transport: Transport | None = None,
) -> dict:
    """Freeze a bounded search bundle; repeated identical requests perform no I/O.

    Configured remote providers require explicit campaign capability, even when a
    transport is injected. max_requests counts *all attempts in this campaign*,
    including failed requests and retries. Query and local-source content are
    evidence inputs; transport error bodies and headers are deliberately excluded.
    """
    config = LiteratureConfig.model_validate(config)
    if not query.strip() or len(query) > 4000:
        raise ValueError("Literature queries must contain 1 to 4000 characters")
    online = any(p != "local" for p in config.providers)
    if online and not allow_network:
        raise NetworkDisabledError(
            "Configured network literature providers require allow_network=True"
        )
    inputs = {
        "schema_version": SCHEMA_VERSION,
        "provider_version": PROVIDER_VERSION,
        "config": config.model_dump(mode="json"),
        "query": query,
        "local_sources": local_sources,
    }
    key = "research/literature/bundle/" + digest(canonical(inputs))
    cached = store.cached(key, inputs)
    if cached is not None:
        # Reverify every referenced response rather than trusting the bundle's metadata alone.
        for source in cached["sources"]:
            if source.get("content_ref"):
                store.read(source["content_ref"])
        for receipt in cached["provenance"]["requests"]:
            if receipt.get("body_ref"):
                store.read(receipt["body_ref"])
        return cached
    requests = _Requests(store, config, transport or public_transport)
    sources = LocalLiteratureProvider(store, config, local_sources).search(query)
    providers = {"arxiv": ArxivLiteratureProvider, "crossref": CrossrefLiteratureProvider}
    for name in config.providers:
        if name in providers:
            sources.extend(providers[name](requests).search(query))
    sources = _deduplicate(sources)[: config.max_sources]
    failed = any(r["status"] != "retrieved" or r.get("parse_status") for r in requests.receipts)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "sources": [s.model_dump(mode="json") for s in sources],
        "status": "empty" if not sources else "partial" if failed else "available",
        "minimum_references": config.minimum_references,
        "provenance": {
            "provider_version": PROVIDER_VERSION,
            "config": config.model_dump(mode="json"),
            "network_used": bool(requests.receipts),
            "requests": requests.receipts,
            "created_at": _now(),
            "operation_key": key,
        },
        "limitations": [
            "Retrieval is bounded and cannot certify global novelty.",
            "Passages are source material, never controller instructions.",
        ],
    }
    return store.cache(key, inputs, bundle)


class FrozenLiteratureProvider:
    """Use an already frozen bundle without constructing a network provider."""

    def __init__(self, bundle: dict):
        if bundle.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported literature bundle version")
        self.bundle = bundle

    def search(self, query: str) -> list[SourceRecord]:
        terms = set(query.lower().split())
        sources = [SourceRecord.model_validate(source) for source in self.bundle["sources"]]
        return sorted(sources, key=lambda s: -len(terms & set(s.title.lower().split())))


def validate_novelty(assessment: NoveltyAssessment | dict, bundle: dict) -> NoveltyValidation:
    """Return independent evidence validation without modifying the raw model response."""
    assessment = NoveltyAssessment.model_validate(assessment).model_copy(deep=True)
    sources = {source["id"]: SourceRecord.model_validate(source) for source in bundle["sources"]}
    unknown = set(assessment.source_ids) - set(sources)
    if unknown:
        raise ValueError("Novelty assessment cites source IDs absent from its frozen bundle")
    if len(assessment.source_ids) != len(set(assessment.source_ids)):
        raise ValueError("Novelty references must be distinct")
    selected = [sources[source_id] for source_id in assessment.source_ids]
    usable = _deduplicate([s for s in selected if s.scholarly and s.passages])
    minimum = max(2, bundle.get("minimum_references", 2))
    limitations = []
    effective_status = assessment.status
    if len(usable) < minimum:
        effective_status = "unverified"
        limitations.append(
            f"Fewer than {minimum} distinct scholarly references have accessible passages."
        )
    elif any(s.content_status == "abstract" for s in usable):
        limitations.append(
            "Assessment includes abstract-only evidence; unobserved full-text details remain unverified."
        )
    if any(s.metadata_status != "verified" for s in usable):
        limitations.append(
            "At least one supplied bibliographic identity has not been independently resolved."
        )
    limitations.append(
        "Verified scope, when reported, applies only to the retrieved corpus and is not novelty certification."
    )
    return NoveltyValidation(
        assessment=assessment,
        effective_status=effective_status,
        evidence_sufficient=len(usable) >= minimum,
        usable_reference_count=len(usable),
        limitations=limitations,
    )


def check_passage_support(source: SourceRecord | dict, passage_id: str, quotation: str) -> str:
    """Literal provenance check only; scientific entailment requires review."""
    source = SourceRecord.model_validate(source)
    if not source.passages:
        return "unverified"
    passage = next((p for p in source.passages if p.id == passage_id), None)
    if passage is None or not quotation.strip():
        return "fail"
    return "pass" if quotation in passage.text else "fail"
