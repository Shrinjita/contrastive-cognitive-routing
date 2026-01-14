def generate_hypotheses(question, k=5):
    return [
        f"H{i}: {question} under assumption set {i}"
        for i in range(k)
    ]
