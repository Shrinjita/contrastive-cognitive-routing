import numpy as np
from typing import Dict, List, Optional


class CCRMetrics:
    """Metrics specific to Contrastive Cognitive Routing"""

    @staticmethod
    def calculate_epistemic_robustness(action_scores: Dict) -> float:
        scores = action_scores.get("scores", [])
        if not scores:
            return 0.0
        mean = np.mean(scores)
        std = np.std(scores)
        if mean == 0:
            return 0.0
        cv = std / mean
        return float(1.0 - min(1.0, cv))

    @staticmethod
    def calculate_worst_case_improvement(
        baseline_scores: List[float], ccr_scores: List[float]
    ) -> float:
        if not baseline_scores or not ccr_scores:
            return 0.0
        baseline_worst = np.min(baseline_scores)
        ccr_worst = np.min(ccr_scores)
        if baseline_worst == 0:
            return 0.0
        return float((ccr_worst - baseline_worst) / baseline_worst)

    @staticmethod
    def calculate_epistemic_gap(action_scores: Dict) -> float:
        scores = action_scores.get("scores", [])
        if not scores:
            return 1.0
        return float(np.max(scores) - np.min(scores))

    @staticmethod
    def calculate_variance_reduction(baseline_var: float, ccr_var: float) -> float:
        if baseline_var == 0:
            return 0.0
        reduction = (baseline_var - ccr_var) / baseline_var
        return float(max(-1.0, min(1.0, reduction)))

    @staticmethod
    def calculate_decision_consistency(runs: List[str]) -> float:
        """Fraction of runs that agree with the majority decision."""
        if not runs:
            return 0.0
        from collections import Counter
        most_common_count = Counter(runs).most_common(1)[0][1]
        return most_common_count / len(runs)

    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95
    ) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        arr = np.array(scores)
        boot_means = [
            np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(n_bootstrap)
        ]
        alpha = 1 - ci
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return {"mean": float(np.mean(arr)), "lower": lower, "upper": upper}
