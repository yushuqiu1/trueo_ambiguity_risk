"""
Main entry point for the Market Prompt Ambiguity Risk Scoring System.

This module provides the primary interface for analyzing market prompts
and detecting ambiguity risks.
"""

from typing import Optional

from scorer import RiskScorer
from models import RiskScoreResult, MarketProposal
from search import WebSearchClient


def merge_analysis_context(
    context: Optional[str] = None,
    web_search_context: Optional[str] = None
) -> Optional[str]:
    """
    Merge user-provided context with web-search evidence.

    Args:
        context: User-provided context
        web_search_context: Formatted web-search evidence

    Returns:
        Combined context string or None if neither exists
    """
    sections = []

    if context:
        sections.append("User-Provided Context:")
        sections.append(context)

    if web_search_context:
        sections.append(web_search_context)

    return "\n\n".join(sections) if sections else None


def analyze_market_prompt(
    question: str,
    context: Optional[str] = None,
    use_few_shot: bool = True,
    use_web_search: bool = False,
    include_search_debug: bool = False,
) -> RiskScoreResult:
    """
    Analyze a market prompt for ambiguity risks.
    
    This is the main entry point for the risk scoring system. It takes a
    market question and returns a comprehensive risk assessment.
    
    Args:
        question: The market question to analyze
        context: Optional additional context supplied by the caller
        use_few_shot: Whether to use few-shot examples for better prompting
        use_web_search: Whether to augment the analysis with web-search evidence
        include_search_debug: Whether to include web-search evidence-chain details
        
    Returns:
        RiskScoreResult containing:
            - risk_score: Integer 0-100 (higher = more ambiguous)
            - risk_tags: List of identified risk categories
            - rationale: Detailed explanation of the assessment
            - search_debug: Optional evidence-chain details when enabled
            
    Example:
        >>> result = analyze_market_prompt(
        ...     "Will OpenAI release a new model in March this year?"
        ... )
        >>> print(result.risk_score)
        65
        >>> print(result.risk_tags)
        ['ambiguous_time', 'undefined_term']
    """
    merged_context = context
    search_debug = None

    if use_web_search:
        search_client = WebSearchClient()
        if include_search_debug:
            search_debug = search_client.search_with_debug(question)
            web_search_context = search_debug.formatted_context
        else:
            web_search_context = search_client.build_context(question)
        merged_context = merge_analysis_context(
            context=context,
            web_search_context=web_search_context
        )
    
    # Create scorer and analyze
    scorer = RiskScorer()
    return scorer.score(
        question=question,
        context=merged_context,
        include_few_shot=use_few_shot
    ).model_copy(update={"search_debug": search_debug})


def analyze_proposal(
    proposal: MarketProposal,
    use_few_shot: bool = True,
    use_web_search: bool = False,
    include_search_debug: bool = False,
) -> RiskScoreResult:
    """
    Analyze a MarketProposal object for ambiguity risks.
    
    Args:
        proposal: MarketProposal containing the question and optional context
        use_few_shot: Whether to use few-shot examples for better prompting
        use_web_search: Whether to augment the analysis with web-search evidence
        include_search_debug: Whether to include web-search evidence-chain details
        
    Returns:
        RiskScoreResult containing the risk assessment
    """
    return analyze_market_prompt(
        question=proposal.question,
        context=proposal.context,
        use_few_shot=use_few_shot,
        use_web_search=use_web_search,
        include_search_debug=include_search_debug,
    )


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Analyze market prompts for ambiguity risks"
    )
    parser.add_argument(
        "question",
        type=str,
        help="The market question to analyze"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context for the analysis"
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples in prompting"
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Augment the analysis with Tavily web-search evidence"
    )
    parser.add_argument(
        "--include-search-debug",
        action="store_true",
        help="Include web-search evidence-chain details in the result"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    result = analyze_market_prompt(
        question=args.question,
        context=args.context,
        use_few_shot=not args.no_few_shot,
        use_web_search=args.use_web_search,
        include_search_debug=args.include_search_debug,
    )
    
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"\n{'='*60}")
        print("MARKET PROMPT RISK ANALYSIS")
        print(f"{'='*60}")
        print(f"\nQuestion: {args.question}")
        print(f"\nRisk Score: {result.risk_score}/100")
        print(f"\nRisk Tags: {', '.join(result.risk_tags) if result.risk_tags else 'None'}")
        print(f"\nRationale:\n{result.rationale}")
        if args.include_search_debug and result.search_debug is not None:
            print(f"\nSearch Debug:\n{result.search_debug.formatted_context}")
        print(f"\n{'='*60}\n")
