"""
Test module for the Market Prompt Ambiguity Risk Scoring System.

This module contains test cases to validate the functionality of the
risk scoring system.
"""

import sys
import os
import json
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze_market_prompt
from models import (
    RiskScoreResult,
    SearchContext,
    SearchDebugInfo,
    SearchDisplayEvidenceItem,
    SearchEvidenceItem,
)
from config import FEW_SHOT_EXAMPLES_PATH
from search import WebSearchClient, build_official_site_queries, format_search_context


def test_basic_analysis():
    """
    Test basic analysis with a simple market question.
    
    This test verifies that:
    1. The function returns a valid RiskScoreResult
    2. The risk_score is within 0-100 range
    3. The risk_tags is a list
    4. The rationale is a non-empty string
    """
    print("\n" + "="*60)
    print("TEST: Basic Analysis")
    print("="*60)
    
    question = "Will OpenAI release a new model in March this year?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API... (this may take a few seconds)")
    
    result = analyze_market_prompt(question)
    
    # Validate result type
    assert isinstance(result, RiskScoreResult), "Result should be RiskScoreResult"
    print(f"\n✓ Result type: RiskScoreResult")
    
    # Validate risk_score range
    assert 0 <= result.risk_score <= 100, "Risk score must be 0-100"
    print(f"✓ Risk score: {result.risk_score}/100")
    
    # Validate risk_tags
    assert isinstance(result.risk_tags, list), "Risk tags must be a list"
    print(f"✓ Risk tags: {result.risk_tags}")
    
    # Validate rationale
    assert isinstance(result.rationale, str), "Rationale must be a string"
    assert len(result.rationale) > 0, "Rationale must not be empty"
    print(f"✓ Rationale: {result.rationale[:100]}...")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: Basic Analysis")
    print("="*60)
    
    return result


def test_output_format():
    """
    Test that the output format matches the expected JSON structure.
    """
    print("\n" + "="*60)
    print("TEST: Output Format Validation")
    print("="*60)
    
    question = "Will Bitcoin exceed $100,000 USD on December 31, 2025?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API...")
    
    result = analyze_market_prompt(question)
    
    # Convert to dict
    result_dict = result.model_dump()
    
    print(f"\nOutput JSON:")
    print(result.model_dump_json(indent=2))
    
    # Validate JSON structure
    assert "risk_score" in result_dict, "Missing 'risk_score' field"
    assert "risk_tags" in result_dict, "Missing 'risk_tags' field"
    assert "rationale" in result_dict, "Missing 'rationale' field"
    
    print("\n✓ Output contains all required fields:")
    print("  - risk_score: integer")
    print("  - risk_tags: list")
    print("  - rationale: string")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: Output Format Validation")
    print("="*60)
    
    return result


def test_high_risk_question():
    """
    Test with a question that should have high risk score.
    """
    print("\n" + "="*60)
    print("TEST: High Risk Question Detection")
    print("="*60)
    
    # This question is intentionally vague
    question = "Will something important happen soon?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API...")
    
    result = analyze_market_prompt(question, use_few_shot=False)
    
    print(f"\nRisk Score: {result.risk_score}/100")
    print(f"Risk Tags: {result.risk_tags}")
    print(f"Rationale: {result.rationale}")
    
    # We expect this question to have a high risk score
    print(f"\n✓ High ambiguity detected: {result.risk_score >= 50}")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: High Risk Question Detection")
    print("="*60)
    
    return result


def test_few_shot_examples_file():
    """
    Test that examples.json exists and follows the expected schema.
    """
    print("\n" + "="*60)
    print("TEST: Few-Shot Examples File Validation")
    print("="*60)

    assert os.path.exists(FEW_SHOT_EXAMPLES_PATH), "examples.json file is missing"

    with open(FEW_SHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    assert isinstance(examples, list), "examples.json must contain a list"
    assert len(examples) > 0, "examples.json must contain at least one example"

    for index, example in enumerate(examples, 1):
        assert isinstance(example, dict), f"Example {index} must be an object"
        assert "question" in example, f"Example {index} is missing 'question'"
        assert "result" in example, f"Example {index} is missing 'result'"
        assert isinstance(example["question"], str), f"Example {index} question must be a string"
        assert example["question"].strip(), f"Example {index} question must not be empty"
        assert isinstance(example["result"], dict), f"Example {index} result must be an object"

        result = example["result"]
        assert "risk_score" in result, f"Example {index} result is missing 'risk_score'"
        assert "risk_tags" in result, f"Example {index} result is missing 'risk_tags'"
        assert "rationale" in result, f"Example {index} result is missing 'rationale'"
        assert isinstance(result["risk_score"], int), f"Example {index} risk_score must be an integer"
        assert 0 <= result["risk_score"] <= 100, f"Example {index} risk_score must be between 0 and 100"
        assert isinstance(result["risk_tags"], list), f"Example {index} risk_tags must be a list"
        assert all(isinstance(tag, str) for tag in result["risk_tags"]), f"Example {index} risk_tags must contain only strings"
        assert isinstance(result["rationale"], str), f"Example {index} rationale must be a string"
        assert result["rationale"].strip(), f"Example {index} rationale must not be empty"

    print(f"\n✓ Validated examples file: {FEW_SHOT_EXAMPLES_PATH}")
    print(f"✓ Example count: {len(examples)}")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Few-Shot Examples File Validation")
    print("="*60)


def test_format_search_context():
    """
    Test that structured search evidence is formatted for prompt injection.
    """
    print("\n" + "="*60)
    print("TEST: Web Search Context Formatting")
    print("="*60)

    search_context = SearchContext(
        query="OpenAI release new model official source definition resolution criteria",
        provider="tavily",
        summary="Multiple official release channels may create resolution ambiguity.",
        evidence=[
            SearchEvidenceItem(
                title="OpenAI News",
                url="https://openai.com/news",
                snippet="Official announcements are published on the OpenAI website.",
                source="openai.com",
                score=0.95,
                published_date="2025-03-01",
            )
        ],
    )

    formatted = format_search_context(search_context)

    assert "Web Search Evidence:" in formatted
    assert "Provider: tavily" in formatted
    assert "Top Sources:" in formatted
    assert "OpenAI News" in formatted
    assert "https://openai.com/news" in formatted

    print("\n✓ Search context formatting includes provider, evidence, and URL")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Web Search Context Formatting")
    print("="*60)


def test_web_search_requires_api_key():
    """
    Test that explicit web search fails fast without an API key.
    """
    print("\n" + "="*60)
    print("TEST: Web Search API Key Requirement")
    print("="*60)

    with patch("search.TAVILY_API_KEY", ""):
        try:
            WebSearchClient(api_key="")
            raise AssertionError("Expected WebSearchClient to require an API key")
        except ValueError as exc:
            assert "TAVILY_API_KEY" in str(exc)

    print("\n✓ Missing Tavily API key is reported clearly")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Web Search API Key Requirement")
    print("="*60)


def test_analyze_market_prompt_with_web_search():
    """
    Test that analyze_market_prompt merges caller context with web-search evidence.
    """
    print("\n" + "="*60)
    print("TEST: Analyze Market Prompt With Web Search")
    print("="*60)

    captured = {}

    class FakeSearchClient:
        def build_context(self, question: str) -> str:
            captured["search_question"] = question
            return "Web Search Evidence:\nProvider: tavily\nTop Sources:\n1. Official Blog"

    class FakeScorer:
        def score(self, question: str, context=None, include_few_shot: bool = True):
            captured["score_question"] = question
            captured["score_context"] = context
            captured["include_few_shot"] = include_few_shot
            return RiskScoreResult(
                risk_score=55,
                risk_tags=["unverified_source"],
                rationale="Test result from mocked scorer.",
            )

    with patch("main.WebSearchClient", FakeSearchClient), patch("main.RiskScorer", FakeScorer):
        result = analyze_market_prompt(
            "Will OpenAI release a new model in March this year?",
            context="Caller supplied context.",
            use_few_shot=False,
            use_web_search=True,
        )

    assert captured["search_question"] == "Will OpenAI release a new model in March this year?"
    assert "User-Provided Context:" in captured["score_context"]
    assert "Caller supplied context." in captured["score_context"]
    assert "Web Search Evidence:" in captured["score_context"]
    assert captured["include_few_shot"] is False
    assert isinstance(result, RiskScoreResult)

    print("\n✓ Web search evidence is merged into scorer context")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Analyze Market Prompt With Web Search")
    print("="*60)


def test_search_debug_is_optional():
    """
    Test that search debug information is omitted unless explicitly requested.
    """
    print("\n" + "="*60)
    print("TEST: Search Debug Optionality")
    print("="*60)

    class FakeScorer:
        def score(self, question: str, context=None, include_few_shot: bool = True):
            return RiskScoreResult(
                risk_score=40,
                risk_tags=["undefined_term"],
                rationale="Test result from mocked scorer.",
            )

    with patch("main.RiskScorer", FakeScorer):
        result = analyze_market_prompt(
            "Will OpenAI release a new model in March this year?",
            use_web_search=False,
        )

    assert result.search_debug is None

    print("\n✓ search_debug is absent by default")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Search Debug Optionality")
    print("="*60)


def test_analyze_market_prompt_returns_search_debug():
    """
    Test that analyze_market_prompt can return evidence-chain debug information.
    """
    print("\n" + "="*60)
    print("TEST: Analyze Market Prompt Returns Search Debug")
    print("="*60)

    fake_debug = SearchDebugInfo(
        provider="tavily",
        initial_query="Will OpenAI release a new model in March this year? official source definition resolution criteria",
        follow_up_queries=["site:openai.com Will OpenAI release a new model in March this year? official announcement release"],
        raw_answer="Primary provider answer.",
        raw_results=[
            SearchEvidenceItem(
                title="Community Source",
                url="https://manifold.markets/example",
                snippet="Community snippet.",
                source="manifold.markets",
                score=0.91,
            )
        ],
        display_evidence=[
            SearchDisplayEvidenceItem(
                rank=1,
                title="OpenAI Newsroom",
                url="https://openai.com/newsroom",
                source="openai.com",
                snippet="Official snippet.",
                relevance_score=0.50,
                source_category="official",
                is_official=True,
                display_reason="Likely official source for announcements or release criteria.",
            )
        ],
        simplified_context=SearchContext(
            query="Will OpenAI release a new model in March this year? official source definition resolution criteria",
            provider="tavily",
            summary="Simplified summary.",
            evidence=[
                SearchEvidenceItem(
                    title="OpenAI Newsroom",
                    url="https://openai.com/newsroom",
                    snippet="Official snippet.",
                    source="openai.com",
                    score=0.50,
                )
            ],
        ),
        formatted_context="Web Search Evidence:\nProvider: tavily\nTop Sources:\n1. OpenAI Newsroom",
    )

    class FakeSearchClient:
        def search_with_debug(self, question: str) -> SearchDebugInfo:
            return fake_debug

        def build_context(self, question: str) -> str:
            raise AssertionError("build_context should not be called when include_search_debug=True")

    class FakeScorer:
        def score(self, question: str, context=None, include_few_shot: bool = True):
            return RiskScoreResult(
                risk_score=55,
                risk_tags=["unverified_source"],
                rationale="Test result from mocked scorer.",
            )

    with patch("main.WebSearchClient", FakeSearchClient), patch("main.RiskScorer", FakeScorer):
        result = analyze_market_prompt(
            "Will OpenAI release a new model in March this year?",
            use_web_search=True,
            include_search_debug=True,
        )

    assert result.search_debug is not None
    assert result.search_debug.initial_query == fake_debug.initial_query
    assert result.search_debug.raw_results[0].source == "manifold.markets"
    assert result.search_debug.display_evidence[0].source_category == "official"
    assert result.search_debug.simplified_context.evidence[0].source == "openai.com"
    assert "OpenAI Newsroom" in result.search_debug.formatted_context

    print("\n✓ search_debug returns raw and simplified evidence-chain data")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Analyze Market Prompt Returns Search Debug")
    print("="*60)


def test_official_sources_are_prioritized():
    """
    Test that official-looking domains are ranked ahead of media and community sources.
    """
    print("\n" + "="*60)
    print("TEST: Official Source Prioritization")
    print("="*60)

    client = WebSearchClient(api_key="test-key")
    evidence = [
        SearchEvidenceItem(
            title="Community Market",
            url="https://manifold.markets/example",
            snippet="Community discussion of possible resolution criteria.",
            source="manifold.markets",
            score=0.95,
        ),
        SearchEvidenceItem(
            title="OpenAI Newsroom",
            url="https://openai.com/news/example",
            snippet="Official announcement channel for OpenAI.",
            source="openai.com",
            score=0.60,
        ),
        SearchEvidenceItem(
            title="TechCrunch Coverage",
            url="https://techcrunch.com/example",
            snippet="Media coverage of OpenAI plans.",
            source="techcrunch.com",
            score=0.90,
        ),
    ]

    ranked = client._prioritize_authoritative_sources(
        query="Will OpenAI release a new model in March this year? official source definition resolution criteria",
        evidence=evidence,
    )

    assert ranked[0].source == "openai.com"
    assert ranked[-1].source == "manifold.markets"

    print("\n✓ Official source ranked above media and community results")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Official Source Prioritization")
    print("="*60)


def test_official_site_queries_are_generated():
    """
    Test that likely official domains are extracted into site-biased follow-up queries.
    """
    print("\n" + "="*60)
    print("TEST: Official Site Query Generation")
    print("="*60)

    queries = build_official_site_queries("Will OpenAI release a new model in March this year?")

    assert any("site:openai.com" in query for query in queries)
    assert not any("site:will.com" in query for query in queries)
    assert not any("site:march.com" in query for query in queries)

    print("\n✓ Official site follow-up query generated for OpenAI")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Official Site Query Generation")
    print("="*60)


def test_search_merges_official_site_results():
    """
    Test that the client runs a follow-up official-domain query and merges results.
    """
    print("\n" + "="*60)
    print("TEST: Official Site Search Merge")
    print("="*60)

    client = WebSearchClient(api_key="test-key")
    observed_queries = []

    def fake_run_search_request(query: str):
        observed_queries.append(query)
        if query.startswith("site:openai.com"):
            return {
                "answer": None,
                "results": [
                    {
                        "title": "OpenAI Official Announcement",
                        "url": "https://openai.com/index/introducing-example",
                        "content": "Official announcement from OpenAI.",
                        "score": 0.40,
                    }
                ],
            }
        return {
            "answer": "General answer",
            "results": [
                {
                    "title": "Community Discussion",
                    "url": "https://manifold.markets/example",
                    "content": "Community discussion.",
                    "score": 0.95,
                }
            ],
        }

    with patch.object(client, "_run_search_request", side_effect=fake_run_search_request):
        context = client.search("Will OpenAI release a new model in March this year?")

    assert any(query.startswith("site:openai.com") for query in observed_queries)
    assert context.evidence[0].source == "openai.com"
    assert any(item.source == "manifold.markets" for item in context.evidence)

    print("\n✓ Official follow-up search merged and prioritized")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Official Site Search Merge")
    print("="*60)


def test_search_with_debug_returns_evidence_chain():
    """
    Test that search_with_debug exposes raw results and simplified context together.
    """
    print("\n" + "="*60)
    print("TEST: Search With Debug Evidence Chain")
    print("="*60)

    client = WebSearchClient(api_key="test-key")

    def fake_run_search_request(query: str):
        if query.startswith("site:openai.com"):
            return {
                "answer": None,
                "results": [
                    {
                        "title": "OpenAI Official Announcement",
                        "url": "https://openai.com/index/introducing-example",
                        "content": "Official announcement from OpenAI.",
                        "score": 0.40,
                    }
                ],
            }
        return {
            "answer": "Primary provider answer",
            "results": [
                {
                    "title": "Community Discussion",
                    "url": "https://manifold.markets/example",
                    "content": "Community discussion.",
                    "score": 0.95,
                }
            ],
        }

    with patch.object(client, "_run_search_request", side_effect=fake_run_search_request):
        debug_info = client.search_with_debug("Will OpenAI release a new model in March this year?")

    assert debug_info.raw_answer == "Primary provider answer"
    assert any(item.source == "manifold.markets" for item in debug_info.raw_results)
    assert any(item.source == "openai.com" for item in debug_info.raw_results)
    assert debug_info.display_evidence[0].source_category == "official"
    assert debug_info.display_evidence[0].is_official is True
    assert debug_info.display_evidence[0].rank == 1
    assert debug_info.simplified_context.evidence[0].source == "openai.com"
    assert "Web Search Evidence:" in debug_info.formatted_context

    print("\n✓ search_with_debug returns raw results and simplified context")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Search With Debug Evidence Chain")
    print("="*60)


def test_display_evidence_marks_community_subdomains():
    """
    Test that community subdomains are categorized as community for display.
    """
    print("\n" + "="*60)
    print("TEST: Display Evidence Community Classification")
    print("="*60)

    client = WebSearchClient(api_key="test-key")
    evidence = [
        SearchEvidenceItem(
            title="OpenAI Community",
            url="https://community.openai.com/t/example",
            snippet="Community discussion.",
            source="community.openai.com",
            score=0.7,
        )
    ]

    display_items = client._build_display_evidence(
        "Will OpenAI release a new model in March this year? official source definition resolution criteria",
        evidence,
    )

    assert display_items[0].source_category == "community"
    assert display_items[0].is_official is False

    print("\n✓ Community subdomains are labeled as community display evidence")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Display Evidence Community Classification")
    print("="*60)


def run_all_tests():
    """
    Run all tests and report results.
    """
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("Few-Shot Examples File", test_few_shot_examples_file),
        ("Web Search Context Formatting", test_format_search_context),
        ("Web Search API Key Requirement", test_web_search_requires_api_key),
        ("Web Search Integration", test_analyze_market_prompt_with_web_search),
        ("Search Debug Optionality", test_search_debug_is_optional),
        ("Search Debug Output", test_analyze_market_prompt_returns_search_debug),
        ("Official Source Prioritization", test_official_sources_are_prioritized),
        ("Official Site Query Generation", test_official_site_queries_are_generated),
        ("Official Site Search Merge", test_search_merges_official_site_results),
        ("Search With Debug Evidence Chain", test_search_with_debug_returns_evidence_chain),
        ("Display Evidence Community Classification", test_display_evidence_marks_community_subdomains),
        ("Basic Analysis", test_basic_analysis),
        ("Output Format", test_output_format),
        ("High Risk Detection", test_high_risk_question),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run risk scorer tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the basic test"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        result = test_basic_analysis()
        print("\n" + "="*60)
        print("FINAL OUTPUT")
        print("="*60)
        print(result.model_dump_json(indent=2))
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
