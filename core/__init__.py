# core/__init__.py
from .proxy_agent import EpistemicProxyAgent
from .contrastive_router import ContrastiveCognitiveRouter
from .epistemic_variants import EpistemicVariantGenerator

__all__ = ['EpistemicProxyAgent', 'ContrastiveCognitiveRouter', 'EpistemicVariantGenerator']