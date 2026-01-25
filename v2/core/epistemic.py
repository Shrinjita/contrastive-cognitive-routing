# core/epistemic.py
import re

class EpistemicState:
    def __init__(self):
        self.beliefs = []
        self.confidence = 0.0
        self.unknowns = []

def extract_claims(text):
    sentences = text.split('.')
    return [s.strip() for s in sentences if s.strip()]

def detect_uncertainty(text):
    uncertainty_markers = ['uncertain', 'unclear', 'unknown', 'possibly', 'maybe', 'might', 'could', 'unsure']
    text_lower = text.lower()
    return any(marker in text_lower for marker in uncertainty_markers)

def score_epistemic_alignment(memory, response):
    memory_claims = set(extract_claims(memory))
    response_claims = set(extract_claims(response))
    
    if not memory_claims:
        return 0.5
    
    intersection = memory_claims.intersection(response_claims)
    return len(intersection) / len(memory_claims)