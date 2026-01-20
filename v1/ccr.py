# ccr.py
from typing import Dict, List, Callable
import math


# ----------- Core epistemic structures -----------

class World:
    def __init__(self, name: str, facts: Dict[str, bool]):
        self.name = name
        self.facts = facts

    def holds(self, proposition: str) -> bool:
        return self.facts.get(proposition, False)


class EpistemicAgent:
    """
    Represents Ki accessibility relation
    """
    def __init__(self, name: str):
        self.name = name
        self.accessible: Dict[str, List[str]] = {}

    def set_accessible(self, world: str, reachable: List[str]):
        self.accessible[world] = reachable

    def alternatives(self, world: str) -> List[str]:
        return self.accessible.get(world, [])


class KripkeModel:
    def __init__(self, worlds: Dict[str, World], agents: Dict[str, EpistemicAgent]):
        self.worlds = worlds
        self.agents = agents


# ----------- Epistemic operators -----------

def K(model: KripkeModel, agent: str, world: str, proposition: str) -> bool:
    """
    Ki φ  — true if φ holds in all accessible worlds
    """
    for w in model.agents[agent].alternatives(world):
        if not model.worlds[w].holds(proposition):
            return False
    return True


def not_K(model: KripkeModel, agent: str, world: str, proposition: str) -> bool:
    """
    ¬Ki φ
    """
    return not K(model, agent, world, proposition)


# ----------- CCR = Contrastive Epistemic Router -----------

class CCR:
    """
    Routes a query to a sub-agent based on epistemic confidence gradients
    """

    def __init__(self, model: KripkeModel, agent: str):
        self.model = model
        self.agent = agent

    def epistemic_entropy(self, world: str, proposition: str) -> float:
        """
        Measures epistemic uncertainty.
        0 = fully known
        1 = maximally unknown
        """
        accessible = self.model.agents[self.agent].alternatives(world)
        if not accessible:
            return 1.0

        truth = 0
        for w in accessible:
            if self.model.worlds[w].holds(proposition):
                truth += 1

        p = truth / len(accessible)
        if p == 0 or p == 1:
            return 0.0
        return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))


    def route(self, world: str, proposition: str) -> str:
        """
        This is the core CCR decision:
        Which cognitive module should answer?
        """
        H = self.epistemic_entropy(world, proposition)

        if H == 0:
            return "FACTUAL_AGENT"
        elif H < 0.5:
            return "REASONING_AGENT"
        elif H < 0.9:
            return "RETRIEVAL_AGENT"
        else:
            return "EXPLORATORY_AGENT"
