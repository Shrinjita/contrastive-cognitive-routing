#!/usr/bin/env python3
"""
Evaluation runner for Contrastive Cognitive Routing.

Covers:
  - Expanded test suite (10 cases across 5 categories)
  - Baseline (greedy / mean-score) vs CCR comparison
  - Per-query and aggregate metrics
  - Bootstrap confidence intervals
  - JSON + human-readable report output
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.proxy_agent import EpistemicProxyAgent
from evaluation.ccr_metrics import CCRMetrics

# ─────────────────────────────────────────────────────────────────────────────
# Expanded Test Suite
# ─────────────────────────────────────────────────────────────────────────────

TEST_SUITE: List[Dict] = [
    # financial_decision
    {
        "id": "TC-001",
        "query": "Should we approve a $45,000 marketing campaign?",
        "category": "financial_decision",
        "expected_action_keywords": ["approve", "review", "condition"],
        "expected_policy_refs": ["POL-001", "POL-004"],
        "robustness_threshold": 0.65,
    },
    {
        "id": "TC-002",
        "query": "A vendor has submitted a $30,000 contract. Should we sign it?",
        "category": "financial_decision",
        "expected_action_keywords": ["legal", "review", "audit"],
        "expected_policy_refs": ["POL-003", "POL-001"],
        "robustness_threshold": 0.60,
    },
    # policy_query
    {
        "id": "TC-003",
        "query": "What is our hiring policy for senior positions?",
        "category": "policy_query",
        "expected_action_keywords": ["interview", "hr", "background"],
        "expected_policy_refs": ["POL-002"],
        "robustness_threshold": 0.70,
    },
    {
        "id": "TC-004",
        "query": "What approval is needed for a $12,000 expenditure?",
        "category": "policy_query",
        "expected_action_keywords": ["chief of staff", "approve", "budget"],
        "expected_policy_refs": ["POL-001"],
        "robustness_threshold": 0.70,
    },
    # data_security
    {
        "id": "TC-005",
        "query": "A third party wants access to customer data for analytics.",
        "category": "data_security",
        "expected_action_keywords": ["deny", "audit", "cto", "security"],
        "expected_policy_refs": ["POL-005", "POL-003"],
        "robustness_threshold": 0.65,
    },
    {
        "id": "TC-006",
        "query": "We detected a potential data breach. What should we do?",
        "category": "data_security",
        "expected_action_keywords": ["notify", "24 hours", "escalate"],
        "expected_policy_refs": ["POL-005"],
        "robustness_threshold": 0.60,
    },
    # escalation
    {
        "id": "TC-007",
        "query": "Our flagship project is 3 weeks behind schedule.",
        "category": "escalation",
        "expected_action_keywords": ["escalate", "ceo", "report"],
        "expected_policy_refs": [],
        "robustness_threshold": 0.55,
    },
    {
        "id": "TC-008",
        "query": "A department conflict is blocking product delivery.",
        "category": "escalation",
        "expected_action_keywords": ["escalate", "mediate", "resolve"],
        "expected_policy_refs": [],
        "robustness_threshold": 0.55,
    },
    # ethics
    {
        "id": "TC-009",
        "query": "An employee has reported a potential conflict of interest.",
        "category": "ethics",
        "expected_action_keywords": ["investigate", "transparent", "document"],
        "expected_policy_refs": [],
        "robustness_threshold": 0.60,
    },
    {
        "id": "TC-010",
        "query": "Marketing wants to share customer emails with a partner without consent.",
        "category": "ethics",
        "expected_action_keywords": ["deny", "privacy", "legal"],
        "expected_policy_refs": ["POL-005"],
        "robustness_threshold": 0.65,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: greedy mean-score selector (no DRO)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineGreedyRouter:
    """Selects action with highest *mean* score across variants (no robustness)."""

    def select(self, action_scores: Dict) -> str:
        best = max(action_scores.items(), key=lambda kv: kv[1]["mean_score"])
        return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-query evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_hit(text: str, keywords: List[str]) -> float:
    """Fraction of keywords present (case-insensitive) in text."""
    if not keywords:
        return 1.0
    text_l = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_l)
    return hits / len(keywords)


def evaluate_single(agent: EpistemicProxyAgent, test_case: Dict) -> Dict:
    query = test_case["query"]

    start = time.time()
    result = agent.process_query(query)
    elapsed = time.time() - start

    routing = result["routing_result"]
    metrics = result["metrics"]

    selected = routing.selected_action
    explanation = result.get("response", "")

    # Keyword alignment
    kw_score = _keyword_hit(
        selected + " " + explanation,
        test_case.get("expected_action_keywords", []),
    )

    # Robustness pass/fail
    robustness_ok = metrics["robustness_score"] >= test_case.get(
        "robustness_threshold", 0.60
    )

    # Collect raw variant scores for statistical analysis
    raw_scores: List[float] = []
    for action_data in routing.action_scores.values():
        raw_scores.extend(action_data.get("scores", []))

    ci = CCRMetrics.bootstrap_confidence_interval(raw_scores)

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "query": query,
        "selected_action": selected,
        "keyword_alignment": round(kw_score, 3),
        "robustness_score": metrics["robustness_score"],
        "robustness_pass": robustness_ok,
        "worst_case_score": metrics["worst_case_score"],
        "epistemic_variance": metrics["epistemic_variance"],
        "epistemic_stability": metrics["epistemic_stability"],
        "decision_quality": metrics["decision_quality"],
        "response_time_s": round(elapsed, 2),
        "bootstrap_ci_95": ci,
        "raw_variant_scores": raw_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: List[Dict]) -> Dict:
    def _arr(key):
        return [r[key] for r in results]

    robustness = _arr("robustness_score")
    variance = _arr("epistemic_variance")
    worst = _arr("worst_case_score")
    quality = _arr("decision_quality")
    alignment = _arr("keyword_alignment")
    times = _arr("response_time_s")

    pass_rate = sum(r["robustness_pass"] for r in results) / len(results)

    # Per-category breakdown
    categories: Dict[str, List] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["robustness_score"])

    cat_summary = {
        cat: {"mean_robustness": round(float(np.mean(vals)), 3), "n": len(vals)}
        for cat, vals in categories.items()
    }

    return {
        "n_queries": len(results),
        "robustness_pass_rate": round(pass_rate, 3),
        "mean_robustness": round(float(np.mean(robustness)), 3),
        "std_robustness": round(float(np.std(robustness)), 3),
        "mean_worst_case": round(float(np.mean(worst)), 3),
        "mean_epistemic_variance": round(float(np.mean(variance)), 3),
        "mean_decision_quality": round(float(np.mean(quality)), 3),
        "mean_keyword_alignment": round(float(np.mean(alignment)), 3),
        "mean_response_time_s": round(float(np.mean(times)), 2),
        "bootstrap_ci_robustness": CCRMetrics.bootstrap_confidence_interval(
            robustness
        ),
        "category_breakdown": cat_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def baseline_comparison(results: List[Dict]) -> Dict:
    """
    Simulates what a greedy (mean-score) router would have chosen
    and computes worst-case improvement for each query.
    """
    improvements: List[float] = []
    variance_reductions: List[float] = []

    for r in results:
        raw = r["raw_variant_scores"]
        if len(raw) < 2:
            continue

        # Split raw scores evenly across actions (3 variants each assumed)
        chunk = max(1, len(raw) // 3)
        action_chunks = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]

        if not action_chunks:
            continue

        # CCR chose max worst-case
        ccr_worst = r["worst_case_score"]
        # Baseline: choose action with max mean → its worst-case
        means = [np.mean(c) for c in action_chunks if c]
        best_mean_idx = int(np.argmax(means))
        baseline_worst = float(np.min(action_chunks[best_mean_idx]))

        improvement = CCRMetrics.calculate_worst_case_improvement(
            [baseline_worst], [ccr_worst]
        )
        improvements.append(improvement)

        baseline_var = float(np.var(action_chunks[best_mean_idx]))
        ccr_var = r["epistemic_variance"]
        variance_reductions.append(
            CCRMetrics.calculate_variance_reduction(baseline_var, ccr_var)
        )

    return {
        "mean_worst_case_improvement_vs_greedy": round(
            float(np.mean(improvements)) if improvements else 0.0, 3
        ),
        "mean_variance_reduction_vs_greedy": round(
            float(np.mean(variance_reductions)) if variance_reductions else 0.0, 3
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: List[Dict], agg: Dict, baseline: Dict):
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  CONTRASTIVE COGNITIVE ROUTING — EVALUATION REPORT")
    print(SEP)

    for r in results:
        status = "✅ PASS" if r["robustness_pass"] else "❌ FAIL"
        print(
            f"\n[{r['id']}] {r['category']}  {status}"
            f"\n  Query     : {r['query']}"
            f"\n  Action    : {r['selected_action']}"
            f"\n  Robustness: {r['robustness_score']:.3f}  "
            f"Variance: {r['epistemic_variance']:.3f}  "
            f"Worst-case: {r['worst_case_score']:.3f}"
            f"\n  KW-align  : {r['keyword_alignment']:.3f}  "
            f"Time: {r['response_time_s']:.1f}s  "
            f"CI-95: [{r['bootstrap_ci_95']['lower']:.3f}, "
            f"{r['bootstrap_ci_95']['upper']:.3f}]"
        )

    print(f"\n{SEP}")
    print("  AGGREGATE METRICS")
    print(SEP)
    print(f"  Queries evaluated       : {agg['n_queries']}")
    print(f"  Robustness pass rate    : {agg['robustness_pass_rate']:.1%}")
    print(
        f"  Mean robustness         : {agg['mean_robustness']:.3f}"
        f" ± {agg['std_robustness']:.3f}"
    )
    print(f"  Mean worst-case score   : {agg['mean_worst_case']:.3f}")
    print(f"  Mean epistemic variance : {agg['mean_epistemic_variance']:.3f}")
    print(f"  Mean decision quality   : {agg['mean_decision_quality']:.3f}")
    print(f"  Mean keyword alignment  : {agg['mean_keyword_alignment']:.3f}")
    print(f"  Mean response time      : {agg['mean_response_time_s']:.1f}s")
    ci = agg["bootstrap_ci_robustness"]
    print(
        f"  Bootstrap CI-95 (rob.)  : [{ci['lower']:.3f}, {ci['upper']:.3f}]"
    )

    print(f"\n{SEP}")
    print("  BASELINE (GREEDY) vs CCR COMPARISON")
    print(SEP)
    print(
        f"  Worst-case improvement  : {baseline['mean_worst_case_improvement_vs_greedy']:+.1%}"
    )
    print(
        f"  Variance reduction      : {baseline['mean_variance_reduction_vs_greedy']:+.1%}"
    )

    print(f"\n{SEP}")
    print("  CATEGORY BREAKDOWN")
    print(SEP)
    for cat, info in agg["category_breakdown"].items():
        print(
            f"  {cat:<25} mean_robustness={info['mean_robustness']:.3f}  n={info['n']}"
        )
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    test_cases: Optional[List[Dict]] = None,
    output_dir: str = "results",
    save_json: bool = True,
) -> Dict:
    if test_cases is None:
        test_cases = TEST_SUITE

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing EpistemicProxyAgent...")
    agent = EpistemicProxyAgent()

    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")
    results: List[Dict] = []
    for tc in test_cases:
        print(f"  → [{tc['id']}] {tc['query'][:60]}...")
        try:
            res = evaluate_single(agent, tc)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            res = {
                "id": tc["id"],
                "category": tc["category"],
                "query": tc["query"],
                "selected_action": "ERROR",
                "keyword_alignment": 0.0,
                "robustness_score": 0.0,
                "robustness_pass": False,
                "worst_case_score": 0.0,
                "epistemic_variance": 1.0,
                "epistemic_stability": 0.0,
                "decision_quality": 0.0,
                "response_time_s": 0.0,
                "bootstrap_ci_95": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
                "raw_variant_scores": [],
            }
        results.append(res)

    agg = aggregate(results)
    baseline = baseline_comparison(results)

    print_report(results, agg, baseline)

    if save_json:
        output = {
            "results": results,
            "aggregate": agg,
            "baseline_comparison": baseline,
        }
        out_path = Path(output_dir) / "evaluation_report.json"
        with open(str(out_path), "w") as f:
            json.dump(output, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x))
        print(f"JSON report saved to {out_path}")

    return {"results": results, "aggregate": agg, "baseline_comparison": baseline}


if __name__ == "__main__":
    run_evaluation()
