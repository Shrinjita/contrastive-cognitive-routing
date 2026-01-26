import numpy as np
from typing import List, Dict, Callable
import random

class EpistemicVariantGenerator:
    """
    Generate epistemic variants E(C) of context C
    Implements different degradation strategies
    """
    
    def __init__(self):
        self.variant_strategies = [
            self._partial_information,
            self._contradictory_information,
            self._temporal_shift,
            self._perspective_shift,
            self._noisy_information
        ]
    
    def generate_variants(self, context: str, query: str, n_variants: int = 3) -> List[Dict]:
        """
        Generate n epistemic variants of the context
        Each variant represents a different 'possible world'
        """
        variants = []
        
        for i in range(n_variants):
            strategy = self.variant_strategies[i % len(self.variant_strategies)]
            variant = strategy(context, query)
            
            variants.append({
                'id': f'V{i+1}',
                'strategy': strategy.__name__,
                'context': variant,
                'degradation_level': self._calculate_degradation(context, variant),
                'epistemic_distance': self._calculate_epistemic_distance(context, variant)
            })
        
        return variants
    
    def _partial_information(self, context: str, query: str) -> str:
        """Remove 30-50% of key information"""
        sentences = context.split('. ')
        if len(sentences) <= 2:
            return context
        
        # Remove random sentences
        n_to_remove = max(1, len(sentences) // 3)
        indices_to_remove = random.sample(range(len(sentences)), n_to_remove)
        
        degraded = [sentences[i] for i in range(len(sentences)) 
                   if i not in indices_to_remove]
        
        return '. '.join(degraded) + '.'
    
    def _contradictory_information(self, context: str, query: str) -> str:
        """Add subtle contradictions"""
        contradictions = [
            " Note: Some reports suggest the opposite.",
            " However, alternative data conflicts with this.",
            " There are conflicting opinions about this.",
            " This information may not be fully reliable."
        ]
        
        if len(context.split()) > 50:
            # Insert contradiction at random point
            words = context.split()
            insert_point = random.randint(len(words)//3, 2*len(words)//3)
            words.insert(insert_point, random.choice(contradictions))
            return ' '.join(words)
        
        return context + random.choice(contradictions)
    
    def _temporal_shift(self, context: str, query: str) -> str:
        """Shift temporal perspective"""
        temporal_shifts = [
            " This situation occurred 6 months ago under different market conditions.",
            " Looking forward 6 months, assumptions may change significantly.",
            " Historical context from last year suggests different outcomes."
        ]
        
        return context + random.choice(temporal_shifts)
    
    def _perspective_shift(self, context: str, query: str) -> str:
        """Shift stakeholder perspective"""
        perspectives = [
            " From a financial perspective, the priorities differ.",
            " The engineering team would analyze this differently.",
            " Customer feedback suggests alternative interpretations."
        ]
        
        return context + random.choice(perspectives)
    
    def _noisy_information(self, context: str, query: str) -> str:
        """Add irrelevant or misleading information"""
        noise_phrases = [
            " Unrelated data suggests other factors may be at play.",
            " There are unspecified variables that could affect outcomes.",
            " External market conditions introduce additional uncertainty."
        ]
        
        return context + random.choice(noise_phrases)
    
    def _calculate_degradation(self, original: str, variant: str) -> float:
        """Calculate information degradation level (0-1)"""
        # Simple heuristic based on length difference
        orig_len = len(original.split())
        var_len = len(variant.split())
        
        if orig_len == 0:
            return 0.0
        
        degradation = abs(orig_len - var_len) / orig_len
        return min(1.0, degradation)
    
    def _calculate_epistemic_distance(self, original: str, variant: str) -> float:
        """Calculate epistemic distance between contexts"""
        # Simple word overlap distance
        orig_words = set(original.lower().split()[:50])
        var_words = set(variant.lower().split()[:50])
        
        if not orig_words:
            return 0.0
        
        intersection = len(orig_words.intersection(var_words))
        union = len(orig_words.union(var_words))
        
        if union == 0:
            return 1.0
        
        distance = 1.0 - (intersection / union)
        return distance