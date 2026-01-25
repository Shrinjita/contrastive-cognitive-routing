import os
from agent.identity_agent import IdentityAgent
from baseline.rag_baseline import RAGBaseline
import config

def main():
    print("=" * 60)
    print("Conscious Proxy Agent - Demo")
    print("=" * 60)
    
    # Check API key
    if not config.config.OPENAI_API_KEY:
        print("ERROR: OpenAI API key not found in .env file")
        return
    
    # Initialize systems
    print("\nInitializing systems...")
    identity_agent = IdentityAgent()
    rag_baseline = RAGBaseline()
    
    # Demo queries
    demo_queries = [
        "Should we approve a $45,000 marketing campaign?",
        "What's our policy on hiring senior developers?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'─' * 50}")
        print(f"QUERY {i}: {query}")
        print(f"{'─' * 50}")
        
        # Identity Agent
        print("\nIDENTITY AGENT:")
        identity_result = identity_agent.query(query)
        print(f"Response: {identity_result['response'][:200]}...")
        
        # RAG Baseline
        print("\nRAG BASELINE:")
        rag_result = rag_baseline.query(query)
        print(f"Response: {rag_result['response'][:200]}...")
    
    print(f"\n{'=' * 60}")
    print("Demo Complete!")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()