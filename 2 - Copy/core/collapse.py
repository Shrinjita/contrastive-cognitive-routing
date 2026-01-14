def collapse(scores):
    scores = sorted(scores, reverse=True)
    if len(scores) < 2:
        return False
    return (scores[0] - scores[1]) > 0.7
