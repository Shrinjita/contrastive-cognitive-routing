# core/scorer.py
from difflib import SequenceMatcher

def decision_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def hallucination_penalty(response):
    hallucination_markers = ['definitely', 'certainly', 'absolutely', 'guaranteed']
    count = sum(1 for marker in hallucination_markers if marker in response.lower())
    return min(count * 0.1, 0.5)

def confidence_calibration(response):
    if 'uncertain' in response.lower() or 'unsure' in response.lower():
        return 0.3
    return 0.7

def aggregate_score(llm, rag, ccr):
    scores = {
        'llm_rag_similarity': decision_similarity(llm, rag),
        'llm_ccr_similarity': decision_similarity(llm, ccr),
        'rag_ccr_similarity': decision_similarity(rag, ccr)
    }
    return scores