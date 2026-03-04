"""
Web search integration for ambiguity risk scoring.

This module retrieves structured search evidence and formats it into
prompt-ready context for downstream ambiguity analysis.
"""

import json
import re
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from config import (
    DEFAULT_WEB_SEARCH_DEPTH,
    DEFAULT_WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_TOPIC,
    TAVILY_API_KEY,
    TAVILY_SEARCH_URL,
)
from models import SearchContext, SearchDebugInfo, SearchEvidenceItem

COMMUNITY_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "manifold.markets",
    "twitter.com",
    "x.com",
    "www.x.com",
    "youtube.com",
    "www.youtube.com",
    "news.ycombinator.com",
}

MEDIA_DOMAINS = {
    "techcrunch.com",
    "www.techcrunch.com",
    "theverge.com",
    "www.theverge.com",
    "mashable.com",
    "www.mashable.com",
    "reuters.com",
    "www.reuters.com",
    "bloomberg.com",
    "www.bloomberg.com",
}


def build_search_query(question: str) -> str:
    """
    Build a search query focused on authoritative resolution evidence.

    Args:
        question: Market question being analyzed

    Returns:
        Search query string
    """
    return f"{question} official source definition resolution criteria"


def build_official_site_queries(question: str) -> list[str]:
    """
    Build follow-up queries biased toward likely official domains.

    Args:
        question: Market question being analyzed

    Returns:
        List of site-biased search queries
    """
    site_queries = []
    seen_domains = set()

    for domain in _build_candidate_official_domains(question):
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        site_queries.append(f"site:{domain} {question} official announcement release")

    return site_queries


def format_search_context(search_context: SearchContext) -> str:
    """
    Format structured search evidence into prompt-ready text.

    Args:
        search_context: Structured search context

    Returns:
        Formatted context string
    """
    sections = [
        "Web Search Evidence:",
        f"Provider: {search_context.provider}",
        f"Search Query: {search_context.query}",
    ]

    if search_context.summary:
        sections.extend(["Search Summary:", search_context.summary])

    if not search_context.evidence:
        sections.append("Evidence: No relevant search results were returned.")
    else:
        sections.append("Top Sources:")
        for index, item in enumerate(search_context.evidence, 1):
            sections.append(f"{index}. {item.title}")

            metadata = []
            if item.source:
                metadata.append(f"Source: {item.source}")
            if item.published_date:
                metadata.append(f"Published: {item.published_date}")
            if item.score is not None:
                metadata.append(f"Relevance: {item.score:.2f}")

            if metadata:
                sections.append("   " + " | ".join(metadata))

            sections.append(f"   URL: {item.url}")
            sections.append(f"   Snippet: {item.snippet}")

    sections.append(
        "Use this evidence to judge whether the market question has a unique subject, clear terminology, authoritative sources, and objective resolution criteria."
    )

    return "\n".join(sections)


class WebSearchClient:
    """
    Tavily-backed web search client for ambiguity analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_url: Optional[str] = None,
        max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS,
        search_depth: str = DEFAULT_WEB_SEARCH_DEPTH,
        topic: str = DEFAULT_WEB_SEARCH_TOPIC,
    ):
        self.api_key = api_key or TAVILY_API_KEY
        self.search_url = search_url or TAVILY_SEARCH_URL
        self.max_results = max_results
        self.search_depth = search_depth
        self.topic = topic

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required when use_web_search=True")

    def search(self, question: str) -> SearchContext:
        """
        Search the web for evidence relevant to resolving a market question.

        Args:
            question: Market question to search around

        Returns:
            Structured search context
        """
        return self.search_with_debug(question).simplified_context

    def search_with_debug(self, question: str) -> SearchDebugInfo:
        """
        Search the web and return both prompt-ready context and debug information.

        Args:
            question: Market question to search around

        Returns:
            SearchDebugInfo containing raw and simplified evidence chains
        """
        initial_query = build_search_query(question)
        follow_up_queries = build_official_site_queries(question)
        primary_response = self._run_search_request(initial_query)

        raw_results = list(primary_response.get("results", []))
        merged_results = list(raw_results)
        for official_query in follow_up_queries:
            official_response = self._run_search_request(official_query)
            merged_results.extend(official_response.get("results", []))

        deduplicated_results = self._deduplicate_results(merged_results)
        raw_evidence = self._normalize_results(deduplicated_results)
        search_context = self._parse_response(
            query=initial_query,
            response_data={
                "answer": primary_response.get("answer"),
                "results": deduplicated_results,
            },
        )
        formatted_context = format_search_context(search_context)

        return SearchDebugInfo(
            provider="tavily",
            initial_query=initial_query,
            follow_up_queries=follow_up_queries,
            raw_answer=self._clean_text(primary_response.get("answer")),
            raw_results=raw_evidence,
            simplified_context=search_context,
            formatted_context=formatted_context,
        )

    def build_context(self, question: str) -> str:
        """
        Search and format prompt-ready context for a market question.

        Args:
            question: Market question to search around

        Returns:
            Formatted web-search context string
        """
        return self.search_with_debug(question).formatted_context

    def _parse_response(self, query: str, response_data: Dict[str, Any]) -> SearchContext:
        """
        Parse Tavily search response into structured models.

        Args:
            query: Search query used
            response_data: Raw Tavily response body

        Returns:
            Structured search context
        """
        evidence = self._normalize_results(response_data.get("results", []))
        evidence = self._prioritize_authoritative_sources(query=query, evidence=evidence)
        summary = self._clean_text(response_data.get("answer")) or self._build_summary(evidence)

        return SearchContext(
            query=query,
            provider="tavily",
            summary=summary,
            evidence=evidence,
        )

    def _run_search_request(self, query: str) -> Dict[str, Any]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        request = Request(
            self.search_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Tavily search failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Tavily search request failed: {exc.reason}") from exc

    @staticmethod
    def _deduplicate_results(results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        deduped = []
        seen_urls = set()

        for result in results:
            url = result.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(result)

        return deduped

    def _normalize_results(self, results: list[Dict[str, Any]]) -> list[SearchEvidenceItem]:
        evidence = []

        for result in results:
            title = self._clean_text(result.get("title")) or result.get("url") or "Untitled result"
            url = result.get("url", "")
            snippet = self._truncate_text(
                self._clean_text(result.get("content") or result.get("snippet")) or "No snippet provided.",
                limit=400,
            )
            evidence.append(
                SearchEvidenceItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self._extract_source(url),
                    score=self._safe_float(result.get("score")),
                    published_date=result.get("published_date"),
                )
            )

        return evidence

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return " ".join(value.split())

    @staticmethod
    def _truncate_text(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_source(url: str) -> Optional[str]:
        if not url:
            return None
        return urlparse(url).netloc or None

    def _prioritize_authoritative_sources(
        self,
        query: str,
        evidence: list[SearchEvidenceItem],
    ) -> list[SearchEvidenceItem]:
        subject_terms = self._extract_subject_terms(query)

        return sorted(
            evidence,
            key=lambda item: (
                self._authority_rank(item, subject_terms),
                item.score if item.score is not None else float("-inf"),
            ),
            reverse=True,
        )

    def _authority_rank(self, item: SearchEvidenceItem, subject_terms: set[str]) -> int:
        domain = (item.source or "").lower()
        domain_labels = set(part for part in re.split(r"[^a-z0-9]+", domain) if part)
        domain_matches_subject = bool(subject_terms & domain_labels)

        rank = 0

        if domain.endswith(".gov") or ".gov." in domain:
            rank += 7
        if domain.endswith(".edu") or ".edu." in domain:
            rank += 5

        if domain_matches_subject and domain not in COMMUNITY_DOMAINS and domain not in MEDIA_DOMAINS:
            rank += 6

        if any(label in {"docs", "blog", "news", "press", "support", "help"} for label in domain_labels):
            rank += 1

        if domain in MEDIA_DOMAINS:
            rank += 1

        if domain in COMMUNITY_DOMAINS:
            rank -= 4

        return rank

    @staticmethod
    def _extract_subject_terms(query: str) -> set[str]:
        stop_terms = {
            "official",
            "source",
            "definition",
            "resolution",
            "criteria",
            "will",
            "this",
            "that",
            "with",
            "from",
            "have",
            "what",
            "when",
            "where",
            "which",
            "release",
            "model",
            "march",
            "year",
            "before",
            "after",
            "market",
            "question",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) >= 3 and token not in stop_terms and not token.isdigit()
        }
        return tokens

    @staticmethod
    def _build_summary(evidence: list[SearchEvidenceItem]) -> Optional[str]:
        if not evidence:
            return None

        top_titles = [item.title for item in evidence[:3] if item.title]
        top_sources = [item.source for item in evidence[:3] if item.source]

        summary_parts = []
        if top_titles:
            summary_parts.append("Top evidence focuses on: " + "; ".join(top_titles))
        if top_sources:
            summary_parts.append("Primary sources include: " + ", ".join(top_sources))

        return " ".join(summary_parts) if summary_parts else None


def _build_candidate_official_domains(question: str) -> list[str]:
    excluded_terms = {
        "will",
        "would",
        "could",
        "should",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "january",
        "february",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "this",
        "that",
    }
    capitalized_terms = re.findall(r"\b[A-Z][A-Za-z0-9]+\b", question)
    candidate_terms = []

    for term in capitalized_terms:
        normalized = re.sub(r"[^a-z0-9]", "", term.lower())
        if len(normalized) >= 3 and normalized not in excluded_terms:
            candidate_terms.append(normalized)

    return [f"{term}.com" for term in dict.fromkeys(candidate_terms)]
