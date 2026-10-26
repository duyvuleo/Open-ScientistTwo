"""Replaceable source retrieval; the shipped provider never makes network requests."""

from typing import Protocol


class LiteratureProvider(Protocol):
    def search(self, query: str, limit: int = 2) -> list[dict]: ...


class LocalLiterature:
    def __init__(self, sources: list[dict]):
        self.sources = sources

    def search(self, query, limit=2):
        terms = set(query.lower().split())
        return sorted(
            self.sources,
            key=lambda source: -len(terms.intersection(source.get("title", "").lower().split())),
        )[:limit]
