# compare_runner.py
from core.llm import OpenAILLM
from core.rag_agent import RAGAgent
from core.rlm_memory import RecursiveMemory
from core.ccr_agent import CCRDecisionAgent
from core.epistemic import EpistemicState
from core.scorer import aggregate_score
from core.utils import log_result
import config

def run_baseline_llm(query, llm):
    prompt = f"Query: {query}\n\nProvide a decision."
    return llm.generate(prompt)

def run_rag(query, rag_agent):
    return rag_agent.decide(query)

def run_ccr(query, ccr_agent):
    return ccr_agent.decide(query)

def compare_all(query):
    llm = OpenAILLM(config.OPENAI_API_KEY, config.OPENAI_MODEL)
    
    rag_agent = RAGAgent(llm)
    if os.path.exists('data/policy_docs'):
        rag_agent.ingest_documents('data/policy_docs')
    
    memory = RecursiveMemory(config.MEMORY_PATH)
    epistemic = EpistemicState()
    ccr_agent = CCRDecisionAgent(llm, memory, epistemic)
    
    llm_result = run_baseline_llm(query, llm)
    rag_result = run_rag(query, rag_agent)
    ccr_result = run_ccr(query, ccr_agent)
    
    results = {
        'query': query,
        'llm': llm_result,
        'rag': rag_result,
        'ccr': ccr_result,
        'scores': aggregate_score(llm_result, rag_result, str(ccr_result))
    }
    
    return results

def print_results(results):
    print(f"\nQuery: {results['query']}\n")
    print(f"LLM:\n{results['llm']}\n")
    print(f"RAG:\n{results['rag']}\n")
    print(f"CCR:\n{results['ccr']}\n")
    print(f"Scores:\n{results['scores']}")

import os