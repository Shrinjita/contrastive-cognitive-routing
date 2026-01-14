import math

def score(mcp, hypothesis):
    prompt = f"""
You are an epistemic evaluator.
Rate how plausible this belief is on a 0 to 1 scale.

Belief:
{hypothesis}

Return only a number.
"""
    r = mcp.complete("Epistemic Judge", prompt)
    try:
        return float(r.strip())
    except:
        return 0.0
