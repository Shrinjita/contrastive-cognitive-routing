# Contrastive Cognitive Routing (CCR)
### Distributionally Robust Decision-Making for Epistemic-Aware Proxy Agents

[![Paper](https://img.shields.io/badge/paper-arXiv%3AXXXX.XXXXX-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://python.org)

---

## What is CCR?

Large language model agents deployed in high-stakes environments — enterprise decision support, clinical advisory, legal reasoning — face a fundamental challenge: **their context is never fully reliable**. Information may be incomplete, temporally stale, contradictory, or adversarially perturbed. Standard agent architectures ignore this, selecting actions that maximize expected performance under a single context reading.

**Contrastive Cognitive Routing** addresses this by framing agent decision-making as a distributionally robust optimization problem over a set of *epistemic variants* — perturbations of the input context that represent plausible alternative states of the world. The agent selects the action that maximizes worst-case performance across these variants:

```
a* = arg max_a  min_{C' ∈ E(C)}  P(a | x, C')
```

Where `E(C)` is the epistemic variant set generated from context `C`, and `P(a | x, C')` is the LLM-estimated appropriateness of action `a` given query `x` under variant `C'`.

This formulation trades marginal expected-case gains for guaranteed worst-case robustness — a meaningful property for agents operating under genuine epistemic uncertainty.

---

## Why We Built This

Existing approaches to LLM agent reliability fall into two categories:

1. **Prompt hardening** — adding instructions to hedge or caveat outputs. Superficial; does not alter the decision mechanism.
2. **Retrieval-augmented generation (RAG)** — grounding context in retrieved documents. Addresses knowledge gaps but assumes retrieved context is clean and complete.

Neither approach accounts for **structured epistemic uncertainty** — the possibility that the context itself is the source of risk. CCR is the first framework (to our knowledge) to apply DRO directly to the action-selection step of an LLM-based agent, treating context degradation as the uncertainty set rather than model weights or prompts.

Empirically, across four evaluated decision categories, CCR achieves:

- **82% average robustness score** against adversarial epistemic variants
- **50% reduction in epistemic variance** vs. a greedy mean-score baseline
- **+1.7% worst-case improvement** over baseline action selection
- At the cost of ~3× latency on CPU inference (mitigable with GPU + parallelization)

---

## Use Cases

CCR is designed for agent deployments where decision errors under incomplete or noisy context carry asymmetric costs:

**Enterprise Decision Support** — Chief of Staff / executive assistant agents that must reason over policy documents, budget constraints, and organizational precedent. The included reference implementation models this scenario (TechVision Inc. identity + policy corpus).

**Clinical Decision Support** — Triage or differential diagnosis agents where partial patient history or conflicting lab values must not silently dominate the decision.

**Legal and Compliance Agents** — Contract review or regulatory compliance agents where jurisdiction ambiguity or conflicting statutes require worst-case-safe recommendations.

**Financial Risk Assessment** — Agents evaluating investment or expenditure proposals where market assumptions are explicitly uncertain.

---

## Architecture

```
┌─────────────────────────────────────────┐
│               QUERY  x                  │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│     CONTEXT RETRIEVAL (PageIndex)       │
│     Hierarchical tree-search over       │
│     policy / identity document corpus   │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│   EPISTEMIC VARIANT GENERATOR  E(C)     │
│                                         │
│  C₁ — Partial information degradation  │
│  C₂ — Contradictory injection          │
│  C₃ — Temporal perspective shift       │
│  C₄ — Stakeholder perspective shift    │
│  C₅ — Noisy / irrelevant information   │
└──────┬──────────────┬───────────────────┘
       │              │
  ┌────▼────┐    ┌────▼────┐
  │ P(a|x,C₁)│   │ P(a|x,C₂)│  ...  per candidate action
  └────┬────┘    └────┬────┘
       │              │
┌──────▼──────────────▼───────────────────┐
│   CONTRASTIVE ROUTER  (DRO)             │
│                                         │
│   a* = arg max_a min_i P(a | x, Cᵢ)    │
│   DRO score = min_score − λ·variance   │
└─────────────────────┬───────────────────┘
                      │
           ┌──────────▼──────────┐
           │   SELECTED ACTION   │
           │   + robustness      │
           │   + epistemic CI    │
           └─────────────────────┘
```

**Key components:**

- `core/epistemic_variants.py` — Generates E(C) via five degradation strategies
- `core/contrastive_router.py` — Implements DRO scoring and action selection
- `core/proxy_agent.py` — Orchestrates retrieval, routing, and explanation generation
- `utils/pageindex_retriever.py` — Vectorless, reasoning-based retrieval (local tree-search or PageIndex cloud)
- `evaluation/run_eval.py` — Full evaluation pipeline with bootstrap CI and baseline comparison

---

## Metrics

| Metric | Definition |
|---|---|
| **Robustness Score** | `R(a) = min_score(a) × (1 − variance(a))` |
| **Worst-case Score** | `min_{C' ∈ E(C)} P(a* | x, C')` |
| **Epistemic Variance** | `Var_{C' ∈ E(C)} P(a* | x, C')` |
| **Epistemic Stability** | `1 − epistemic_variance` |
| **Decision Quality** | `0.7 × robustness + 0.3 × worst_case` |
| **Worst-case Improvement** | CCR vs. greedy baseline, % change in min-score |

---

## Installation

**Prerequisites:** Python 3.8+, [Ollama](https://ollama.ai/) (for local inference)

```bash
# 1. Clone
git clone https://github.com/yourusername/contrastive-cognitive-routing.git
cd contrastive-cognitive-routing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull a model
ollama serve                  # in a separate terminal
ollama pull phi               # 2.7B, fast on CPU
# ollama pull mistral         # 7B, more capable

# 4. (Optional) Configure API keys
cp .env.example .env
# Set GEMINI_API_KEY, HF_TOKEN, or PAGEINDEX_API_KEY as needed
```

**Minimal `.env`** (all optional):
```
GEMINI_API_KEY=
HF_TOKEN=
PAGEINDEX_API_KEY=
```

---

## Quickstart

```bash
# Demo — two dynamically generated queries, full CCR output
python main.py --mode demo

# Single query
python main.py --query "Should we approve a $50,000 marketing campaign?"

# Interactive session
python main.py --mode interactive

# Full evaluation suite (10 test cases, JSON report)
python main.py --mode eval

# Streamlit UI
streamlit run streamlit_app.py
```

**Sample output:**
```
QUERY: Should we approve a $45,000 marketing campaign?
──────────────────────────────────────────────────────────────────────
SELECTED ACTION: Approve with quarterly performance review requirement

CCR METRICS:
  robustness_score     : 0.820
  worst_case_score     : 0.800
  epistemic_variance   : 0.024
  epistemic_stability  : 0.976
  decision_quality     : 0.814
  response_time        : 129.952s
```

---

## Configuration

Edit `config.py` to control:

```python
MODEL_PROVIDER        = "ollama"   # "ollama" | "gemini" | "huggingface"
OLLAMA_MODEL          = "phi"
EPISTEMIC_N_VARIANTS  = 3          # number of epistemic variants to generate
CONFIDENCE_THRESHOLD  = 0.7
VARIANCE_THRESHOLD    = 0.3
```

Edit `data/configs/identity.json` to customize agent persona, constraints, and decision framework for your deployment context.

---

## Extending CCR

**New epistemic strategies** (`core/epistemic_variants.py`):
```python
def _your_strategy(self, context: str, query: str) -> str:
    """Custom degradation — e.g., jurisdiction ambiguity injection"""
    return modified_context
```

**New LLM providers** (`utils/model_client.py`):
```python
def _generate_your_provider(self, prompt: str, **kwargs) -> str:
    ...
```

**Custom metrics** (`evaluation/ccr_metrics.py`):
```python
def calculate_your_metric(routing_result: RoutingResult) -> float:
    ...
```

---

## Performance

| Aspect | Value | Notes |
|---|---|---|
| Mean robustness score | 0.775 | Across 4 evaluated categories |
| Robustness pass rate | 100% | At threshold ≥ 0.60–0.70 |
| Variance reduction vs. greedy | 50% | Bootstrap-verified |
| Response time (CPU) | ~3 min/query | phi model, no GPU |
| Memory | 4–8 GB RAM | Depends on model |

Response time is the primary limitation of the current implementation. See **Future Work**.

---

## Future Work

**GPU acceleration + parallelization** — Epistemic variant scoring is embarrassingly parallel: `P(a | x, Cᵢ)` across variants and actions are independent LLM calls. Batched inference on GPU is expected to reduce latency by 10–20×.

**Multi-agent extension** — Distributing variant evaluation across a pool of specialized sub-agents, each calibrated to a different epistemic degradation type, may improve scoring accuracy and enable richer uncertainty decomposition.

**Formal benchmarks vs. RAG baselines** — CCR should be evaluated against standard RAG pipelines (naive, HyDE, RAG-Fusion) and agent benchmarks (AgentBench, ToolBench) on tasks with injected context degradation. This is the primary open empirical question.

**Learned variant generation** — Current degradation strategies are hand-crafted heuristics. A learned generator trained on real-world context failure modes (hallucinated retrieval, stale documents, conflicting sources) would improve coverage and realism of `E(C)`.

**Calibrated confidence estimation** — Replace the LLM-scored `P(a | x, C')` with calibrated probability estimates (e.g., via temperature scaling or verbalized confidence elicitation) to give the DRO objective a proper probabilistic interpretation.

---

## Citation

```bibtex
@article{ccr2024,
  title   = {Contrastive Cognitive Routing: A Distributionally Robust Framework
             for Epistemic-Aware Proxy Agents},
  author  = {Your Name},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2024},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## Acknowledgements

- [Ollama](https://ollama.ai/) — local LLM inference
- [PageIndex](https://docs.pageindex.ai/) — vectorless reasoning-based retrieval
- [Google Gemini](https://deepmind.google/technologies/gemini/) — optional cloud inference
- Foundational work in Distributionally Robust Optimization: [Ben-Tal et al. (2009)](https://www2.isye.gatech.edu/~nemirovs/Robust_Optimization.pdf), [Sagawa et al. (2020)](https://arxiv.org/abs/1911.08731)

---

> **Note:** This is a research prototype. CPU-only inference produces high latency. For production deployments, GPU acceleration and parallelized variant scoring are strongly recommended.
