import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.epistemic_variants import EpistemicVariantGenerator

@dataclass
class RoutingResult:
    selected_action: str
    action_scores: Dict[str, Dict]
    epistemic_variants: List[Dict]
    robustness_score: float
    epistemic_variance: float
    worst_case_score: float

class ContrastiveCognitiveRouter:
    """
    Implements Contrastive Cognitive Routing (CCR)
    a* = arg max_a min_{C' ∈ E(C)} P(a | x, C')
    """
    
    def __init__(self, llm_scorer):
        self.llm_scorer = llm_scorer
        self.variant_generator = EpistemicVariantGenerator()
    
    def route(self, query: str, context: str, 
              candidate_actions: List[str]) -> RoutingResult:
        """
        Perform contrastive cognitive routing
        
        Args:
            query (x): User query
            context (C): Original context
            candidate_actions: Possible actions [a1, a2, ..., an]
        
        Returns:
            RoutingResult with selected action and analysis
        """
        
        # Step 1: Generate epistemic variants E(C)
        epistemic_variants = self.variant_generator.generate_variants(
            context, query, n_variants=3
        )
        
        # Step 2: Score each action across all variants
        action_scores = self._score_actions_across_variants(
            query, candidate_actions, epistemic_variants
        )
        
        # Step 3: Apply Distributionally Robust Optimization (DRO)
        # a* = arg max_a min_{C' ∈ E(C)} P(a | x, C')
        dro_scores = {}
        for action, scores in action_scores.items():
            min_score = np.min(scores['variant_scores'])
            variance = np.var(scores['variant_scores'])
            
            # DRO objective: maximize worst-case, penalize variance
            dro_score = min_score - (0.3 * variance)  # Penalize instability
            
            dro_scores[action] = {
                'dro_score': dro_score,
                'min_score': min_score,
                'mean_score': np.mean(scores['variant_scores']),
                'variance': variance,
                'scores': scores['variant_scores']
            }
        
        # Step 4: Select action with highest DRO score
        selected_action = max(dro_scores.items(), 
                            key=lambda x: x[1]['dro_score'])[0]
        
        # Step 5: Calculate robustness metrics
        robustness_score = self._calculate_robustness(
            dro_scores[selected_action]
        )
        
        return RoutingResult(
            selected_action=selected_action,
            action_scores=dro_scores,
            epistemic_variants=epistemic_variants,
            robustness_score=robustness_score,
            epistemic_variance=dro_scores[selected_action]['variance'],
            worst_case_score=dro_scores[selected_action]['min_score']
        )
    
    def _score_actions_across_variants(self, query: str, 
                                      actions: List[str],
                                      variants: List[Dict]) -> Dict:
        """
        Score each action across all epistemic variants
        Returns P(a | x, C') for each action and variant
        """
        action_scores = {action: {'variant_scores': []} 
                        for action in actions}
        
        for variant in variants:
            variant_context = variant['context']
            
            # Score all actions for this variant
            variant_scores = self.llm_scorer.score_actions(
                query, variant_context, actions
            )
            
            # Store scores
            for action, score in zip(actions, variant_scores):
                action_scores[action]['variant_scores'].append(score)
        
        return action_scores
    
    def _calculate_robustness(self, action_scores: Dict) -> float:
        """
        Calculate robustness score for an action
        Higher = more robust across epistemic variants
        """
        min_score = action_scores['min_score']
        variance = action_scores['variance']
        
        # Robustness = high worst-case score + low variance
        robustness = min_score * (1.0 - variance)
        return min(1.0, max(0.0, robustness))
    
    def analyze_epistemic_sensitivity(self, routing_result: RoutingResult) -> Dict:
        """
        Analyze how sensitive the decision is to epistemic variants
        """
        action_scores = routing_result.action_scores
        
        # Calculate sensitivity metrics
        sensitivities = {}
        for action, scores in action_scores.items():
            score_range = np.max(scores['scores']) - np.min(scores['scores'])
            sensitivities[action] = {
                'sensitivity': score_range,
                'is_robust': score_range < 0.3,
                'stability': 1.0 - min(1.0, score_range)
            }
        
        return {
            'sensitivities': sensitivities,
            'most_robust_action': min(sensitivities.items(), 
                                    key=lambda x: x[1]['sensitivity'])[0],
            'epistemic_stability': 1.0 - routing_result.epistemic_variance
        }