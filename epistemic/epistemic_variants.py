from typing import Dict, Callable
from agent.state import AgentState
from epistemic.noise_models import (
    identity_noise,
    drop_tokens_noise,
    perturb_numbers_noise
)


class EpistemicVariantGenerator:
    """
    Generates epistemic variants of the same AgentState.
    Each variant represents a different belief condition
    (clean, noisy, degraded).
    """

    def __init__(self):
        self.noise_functions: Dict[str, Callable[[str], str]] = {
            "clean": identity_noise,
            "token_drop": drop_tokens_noise,
            "numeric_perturb": perturb_numbers_noise,
        }

    def generate(self, base_state: AgentState) -> Dict[str, AgentState]:
        """
        Returns multiple epistemic versions of the same state.
        """
        variants = {}

        for name, noise_fn in self.noise_functions.items():
            noisy_input = noise_fn(base_state.input)

            variant_state = AgentState(
                input=noisy_input,
                context=dict(base_state.context),  # shallow copy
            )

            variant_state.log(f"Epistemic variant generated: {name}")
            variants[name] = variant_state

        return variants
