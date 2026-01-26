#!/usr/bin/env python3
"""
Main entry point for Contrastive Cognitive Routing Agent
"""

import argparse
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.proxy_agent import EpistemicProxyAgent

def run_demo():
    """Demonstrate Contrastive Cognitive Routing"""
    agent = EpistemicProxyAgent()
    
    queries = [
        "Should we approve a $45,000 marketing campaign?",
        "How should we handle sensitive customer data sharing?",
    ]
    
    print("\n" + "="*70)
    print("CONTRASTIVE COGNITIVE ROUTING DEMONSTRATION")
    print("a* = arg max_a min_{C' ∈ E(C)} P(a | x, C')")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n📌 QUERY {i}: {query}")
        print("-"*70)
        
        result = agent.process_query(query)
        
        print(f"\n🎯 SELECTED ACTION:")
        print(f"  {result['routing_result'].selected_action}")
        
        print(f"\n📊 CCR METRICS:")
        metrics = result['metrics']
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\n📝 EXPLANATION:")
        print(f"{result['response']}")

def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Cognitive Routing for Epistemic-Aware Proxy Agents"
    )
    parser.add_argument("--mode", choices=["demo", "single"],
                       default="demo", help="Mode to run")
    parser.add_argument("--query", type=str, help="Query to process")
    
    args = parser.parse_args()
    
    if args.query:
        # Process single query
        agent = EpistemicProxyAgent()
        result = agent.process_query(args.query)
        
        print(f"\nQuery: {args.query}")
        print(f"\nSelected Action: {result['routing_result'].selected_action}")
        print(f"\nResponse: {result['response']}")
        print(f"\nMetrics: {result['metrics']}")
        
    elif args.mode == "demo":
        run_demo()

if __name__ == "__main__":
    main()