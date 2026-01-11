# Contrastive Cognitive Routing
Real-world AI agent deployments prioritize reliability and control over full autonomy, often relying on fixed task-decomposition workflows. We propose a Contrastive Cognitive Router (CCR), a lightweight adaptive component that selects among bounded task-decomposition protocols—direct, sequential, and parallel—based on query characteristics.

The system dynamically selects among simple task-decomposition protocols based on lightweight epistemic proxies, without fine-tuning large language models.

## Status
Research prototype (work in progress).

## Structure
- `router/` — cognitive routing logic
- `epistemic/` — epistemic proxy signals
- `protocols/` — task-decomposition strategies
- `main.py` — experiment runner
