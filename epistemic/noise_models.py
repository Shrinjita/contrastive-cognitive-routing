import random
import re


def identity_noise(text: str) -> str:
    """
    No noise. Represents full epistemic access.
    """
    return text


def drop_tokens_noise(text: str, drop_prob: float = 0.15) -> str:
    """
    Simulates missing information by dropping tokens.
    """
    tokens = text.split()
    if not tokens:
        return text

    kept_tokens = [
        t for t in tokens if random.random() > drop_prob
    ]

    # Avoid empty input
    return " ".join(kept_tokens) if kept_tokens else text


def perturb_numbers_noise(text: str, delta: int = 2) -> str:
    """
    Simulates noisy numerical beliefs (measurement error).
    """

    def _perturb(match):
        number = int(match.group())
        return str(number + random.randint(-delta, delta))

    return re.sub(r"\d+", _perturb, text)
