import numpy as np
from typing import Dict, List

class CCRMetrics:
    """Metrics specific to Contrastive Cognitive Routing"""
    
    @staticmethod
    def calculate_epistemic_robustness(action_scores: Dict) -> float:
        """Calculate robustness across epistemic variants"""
        scores = action_scores.get('scores', [])
        if not scores:
            return 0.0
        
        # Robustness = 1 - coefficient of variation
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        robustness = 1.0 - min(1.0, cv)
        return robustness
    
    @staticmethod
    def calculate_worst_case_improvement(baseline_scores: List[float],
                                        ccr_scores: List[float]) -> float:
        """Calculate improvement in worst-case performance"""
        if not baseline_scores or not ccr_scores:
            return 0.0
        
        baseline_worst = np.min(baseline_scores)
        ccr_worst = np.min(ccr_scores)
        
        if baseline_worst == 0:
            return 0.0
        
        improvement = (ccr_worst - baseline_worst) / baseline_worst
        return improvement
    
    @staticmethod
    def calculate_epistemic_gap(action_scores: Dict) -> float:
        """Calculate gap between best and worst epistemic variant"""
        scores = action_scores.get('scores', [])
        if not scores:
            return 1.0  # Maximum uncertainty
        
        return np.max(scores) - np.min(scores)
    
    @staticmethod
    def calculate_variance_reduction(baseline_var: float, ccr_var: float) -> float:
        """Calculate reduction in decision variance"""
        if baseline_var == 0:
            return 0.0
        
        reduction = (baseline_var - ccr_var) / baseline_var
        return max(-1.0, min(1.0, reduction))  # Clip to [-1, 1]