#!/usr/bin/env python3
"""
Main entry point for Contrastive Cognitive Routing Agent
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.proxy_agent import EpistemicProxyAgent


def _generate_demo_queries(agent: EpistemicProxyAgent) -> list:
    """Dynamically generate demo queries using the LLM."""
    prompt = (
        "You are a Chief of Staff assistant. Generate exactly 2 realistic, diverse "
        "business decision queries that a Chief of Staff might face. "
        "Cover different domains: one financial/budget decision and one operational/ethical decision. "
        "Return ONLY the 2 queries, one per line, no numbering, no extra text."
    )
    response = agent.model_client.generate(prompt, temperature=0.9, max_tokens=100)
    queries = [line.strip() for line in response.strip().split("\n") if line.strip()]
    # Guarantee exactly 2 fallback queries if generation fails
    fallbacks = [
        "Should we approve a $38,000 product development budget increase?",
        "An engineer wants to share internal performance data with a new analytics vendor.",
    ]
    if len(queries) < 2:
        queries = fallbacks
    return queries[:2]


def run_demo():
    """Demonstrate Contrastive Cognitive Routing with dynamically generated queries."""
    agent = EpistemicProxyAgent()

    print("\nGenerating demo queries dynamically...")
    queries = _generate_demo_queries(agent)

    print("\n" + "=" * 70)
    print("CONTRASTIVE COGNITIVE ROUTING DEMONSTRATION")
    print("a* = arg max_a min_{C' ∈ E(C)} P(a | x, C')")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n\n📌 QUERY {i}: {query}")
        print("-" * 70)

        result = agent.process_query(query)

        print(f"\n🎯 SELECTED ACTION:")
        print(f"  {result['routing_result'].selected_action}")

        print(f"\n📊 CCR METRICS:")
        metrics = result["metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")

        print(f"\n📝 EXPLANATION:")
        print(f"{result['response']}")


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Cognitive Routing for Epistemic-Aware Proxy Agents"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "single", "eval"],
        default="demo",
        help="Mode to run",
    )
    parser.add_argument("--query", type=str, help="Query to process (single mode)")

    args = parser.parse_args()

    if args.query:
        agent = EpistemicProxyAgent()
        result = agent.process_query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"\nSelected Action: {result['routing_result'].selected_action}")
        print(f"\nResponse: {result['response']}")
        print(f"\nMetrics: {result['metrics']}")

    elif args.mode == "demo":
        run_demo()

    elif args.mode == "eval":
        from evaluation.run_eval import run_evaluation
        run_evaluation()


if __name__ == "__main__":
    main()
