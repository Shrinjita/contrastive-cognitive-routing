from core.hypothesis import generate_hypotheses
from core.scorer import score
from core.collapse import collapse
from metrics.entropy import entropy

def route(mcp, question):
    hyps = generate_hypotheses(question)
    scores = [score(mcp, h) for h in hyps]
    H = entropy(scores)
    C = collapse(scores)

    best = hyps[scores.index(max(scores))]

    if H > 1.0 or C:
        return {
            "decision": "DEFER",
            "entropy": H,
            "beliefs": list(zip(hyps, scores))
        }

    return {
        "decision": best,
        "entropy": H,
        "beliefs": list(zip(hyps, scores))
    }
