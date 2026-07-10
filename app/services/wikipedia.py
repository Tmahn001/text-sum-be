"""Contextual retrieval from Wikipedia.

Design notes (for the defense):
* We use the official MediaWiki APIs directly over HTTPS (no third-party
  scraping library), which keeps dependencies light and behaviour predictable.
* Two calls per entity:
    1. the search/opensearch endpoint resolves a free-text entity to the best
       matching article title (handles redirects, disambiguation, casing);
    2. the REST v1 "page/summary" endpoint returns a clean, plain-text extract
       plus a canonical URL and thumbnail.
* Failures are swallowed per-entity: if one lookup times out or 404s, the others
  still return, so the endpoint degrades gracefully instead of failing whole.
* Disambiguation pages are skipped (they aren't real background info).
"""
from __future__ import annotations

import logging

import requests

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SEARCH_URL = f"https://{settings.wikipedia_lang}.wikipedia.org/w/api.php"
_SUMMARY_URL = (
    f"https://{settings.wikipedia_lang}.wikipedia.org/api/rest_v1/page/summary/"
)
_HEADERS = {"User-Agent": settings.wikipedia_user_agent}


def _search_title(query: str) -> str | None:
    """Resolve a free-text query to the best-matching article title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    try:
        resp = requests.get(
            _SEARCH_URL,
            params=params,
            headers=_HEADERS,
            timeout=settings.wikipedia_timeout,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        logger.warning("Wikipedia search failed for %r: %s", query, exc)
        return None


def _fetch_summary(title: str) -> dict | None:
    """Fetch a clean summary/extract for a resolved article title."""
    try:
        resp = requests.get(
            _SUMMARY_URL + requests.utils.quote(title, safe=""),
            headers=_HEADERS,
            timeout=settings.wikipedia_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Wikipedia summary failed for %r: %s", title, exc)
        return None

    if data.get("type", "").endswith("disambiguation"):
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None

    return {
        "title": data.get("title", title),
        "extract": extract,
        "url": data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://{settings.wikipedia_lang}.wikipedia.org/wiki/"
        + requests.utils.quote(title.replace(" ", "_")),
        "thumbnail": (data.get("thumbnail") or {}).get("source"),
    }


def fetch_context(entities: list[dict]) -> list[dict]:
    """For each extracted entity, retrieve one Wikipedia context block.

    Returns a list of WikiContext-shaped dicts. Entities that resolve to the same
    article (common with aliases) are collapsed so context isn't duplicated.
    """
    context: list[dict] = []
    used_titles: set[str] = set()

    for ent in entities:
        query = ent["text"]
        title = _search_title(query)
        if not title or title in used_titles:
            continue
        summary = _fetch_summary(title)
        if not summary:
            continue
        used_titles.add(summary["title"])
        context.append({"entity": query, **summary})

    logger.info("Retrieved %d Wikipedia context block(s)", len(context))
    return context
