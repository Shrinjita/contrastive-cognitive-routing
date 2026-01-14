import math

def entropy(probs):
    total = sum(probs)
    if total == 0:
        return 0
    p = [x/total for x in probs]
    return -sum(pi * math.log(pi + 1e-9) for pi in p)