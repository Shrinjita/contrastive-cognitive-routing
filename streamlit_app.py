"""
Streamlit UI for Contrastive Cognitive Routing Agent
Run: streamlit run streamlit_app.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CCR Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        margin: 4px 0;
        border-left: 4px solid #7c3aed;
    }
    .cot-step {
        background: #0f172a;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #22d3ee;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .action-badge {
        background: #065f46;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .variant-box {
        background: #1c1917;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        border: 1px solid #44403c;
        font-size: 0.82rem;
    }
    .robustness-high { border-left-color: #22c55e !important; }
    .robustness-mid  { border-left-color: #f59e0b !important; }
    .robustness-low  { border-left-color: #ef4444 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent loader (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Initializing CCR Agent...")
def load_agent():
    from core.proxy_agent import EpistemicProxyAgent
    return EpistemicProxyAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — document upload + config
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("📂 Upload Documents")
    uploaded_identity = st.file_uploader(
        "Identity JSON",
        type=["json"],
        help="Upload a custom identity.json to override the default agent persona.",
    )
    uploaded_policies = st.file_uploader(
        "Policies JSON",
        type=["json"],
        help="Upload a custom company_policies.json.",
    )

    if uploaded_identity is not None:
        dest = Path("data/configs/identity.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded_identity.read())
        st.success("✅ identity.json saved")
        st.cache_resource.clear()

    if uploaded_policies is not None:
        dest = Path("data/policies/company_policies.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded_policies.read())
        st.success("✅ company_policies.json saved")
        st.cache_resource.clear()

    st.markdown("---")
    st.subheader("🔧 Model Settings")

    model_provider = st.selectbox(
        "Model Provider",
        ["ollama", "gemini", "huggingface"],
        index=0,
    )
    if model_provider == "ollama":
        model_name = st.text_input("Ollama Model", value="phi")
    elif model_provider == "gemini":
        model_name = st.text_input("Gemini Model", value="models/gemini-flash-latest")
        gemini_key = st.text_input("Gemini API Key", type="password")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
    else:
        model_name = st.text_input("HuggingFace Model", value="gpt2")

    n_variants = st.slider("Epistemic Variants (n)", min_value=2, max_value=6, value=3)

    st.markdown("---")
    st.caption("Contrastive Cognitive Routing\n`a* = argmax_a min_C' P(a|x,C')`")

# ─────────────────────────────────────────────────────────────────────────────
# Update config from sidebar
# ─────────────────────────────────────────────────────────────────────────────

try:
    from config import config as cfg

    cfg.MODEL_PROVIDER = model_provider
    cfg.EPISTEMIC_N_VARIANTS = n_variants
    if model_provider == "ollama":
        cfg.OLLAMA_MODEL = model_name
    elif model_provider == "gemini":
        cfg.GEMINI_MODEL = model_name
    elif model_provider == "huggingface":
        cfg.HF_MODEL = model_name
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧠 Contrastive Cognitive Routing Agent")
st.markdown(
    "Epistemic-aware decision engine using **Distributionally Robust Optimization** "
    "over epistemic variants."
)

tab_query, tab_eval = st.tabs(["💬 Query & Reasoning", "📊 Batch Evaluation"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Query + CoT + Metrics
# ─────────────────────────────────────────────────────────────────────────────

with tab_query:
    st.subheader("Ask a Decision Question")

    query_input = st.text_area(
        "Your Query",
        placeholder="e.g. Should we approve a $45,000 marketing campaign?",
        height=100,
    )

    col_run, col_clear = st.columns([1, 5])
    run_btn = col_run.button("▶ Run CCR", type="primary", use_container_width=True)
    if col_clear.button("🗑 Clear", use_container_width=False):
        st.session_state.pop("last_result", None)
        st.rerun()

    if run_btn and query_input.strip():
        with st.spinner("Running Contrastive Cognitive Routing..."):
            try:
                agent = load_agent()
                result = agent.process_query(query_input.strip())
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Agent error: {e}")

    # ── Display result ────────────────────────────────────────────────────────

    if "last_result" in st.session_state:
        result: Dict = st.session_state["last_result"]
        routing = result["routing_result"]
        metrics = result["metrics"]

        st.markdown("---")

        # Selected action
        st.subheader("🎯 Selected Action")
        st.markdown(
            f'<span class="action-badge">{routing.selected_action}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Two-column layout: CoT | Metrics
        col_cot, col_metrics = st.columns([3, 2])

        with col_cot:
            st.subheader("🔗 Chain of Thought")

            # Step 1: Context retrieved
            with st.expander("Step 1 — Context Retrieval", expanded=True):
                st.markdown(
                    f'<div class="cot-step">Context mode: <b>{result.get("context_mode", "local")}</b><br>'
                    f"PageIndex tree-search identified relevant policy nodes for:<br>"
                    f"<i>{result['query']}</i></div>",
                    unsafe_allow_html=True,
                )

            # Step 2: Candidate actions
            with st.expander("Step 2 — Candidate Action Generation", expanded=True):
                for i, (action, scores) in enumerate(
                    routing.action_scores.items(), 1
                ):
                    rob_class = (
                        "robustness-high"
                        if scores["dro_score"] >= 0.7
                        else "robustness-mid"
                        if scores["dro_score"] >= 0.4
                        else "robustness-low"
                    )
                    st.markdown(
                        f'<div class="cot-step {rob_class}">'
                        f"<b>Option {i}:</b> {action}<br>"
                        f"DRO score: <b>{scores['dro_score']:.3f}</b>  "
                        f"Min: {scores['min_score']:.3f}  "
                        f"Mean: {scores['mean_score']:.3f}  "
                        f"Var: {scores['variance']:.3f}</div>",
                        unsafe_allow_html=True,
                    )

            # Step 3: Epistemic variants
            with st.expander("Step 3 — Epistemic Variants E(C)", expanded=False):
                for v in routing.epistemic_variants:
                    st.markdown(
                        f'<div class="variant-box">'
                        f"<b>[{v['id']}]</b> Strategy: <code>{v['strategy']}</code><br>"
                        f"Degradation: {v['degradation_level']:.3f}  "
                        f"Epistemic distance: {v['epistemic_distance']:.3f}<br>"
                        f"<small>{v['context'][:200]}...</small></div>",
                        unsafe_allow_html=True,
                    )

            # Step 4: DRO selection
            with st.expander("Step 4 — DRO Selection", expanded=True):
                st.markdown(
                    f'<div class="cot-step">'
                    f"Applied: <code>a* = argmax_a min_C' P(a|x,C')</code><br>"
                    f"Selected action with highest worst-case score across all variants.<br>"
                    f"Robustness: <b>{routing.robustness_score:.3f}</b>  "
                    f"Worst-case: <b>{routing.worst_case_score:.3f}</b>  "
                    f"Variance: <b>{routing.epistemic_variance:.3f}</b></div>",
                    unsafe_allow_html=True,
                )

            # Step 5: Final explanation
            with st.expander("Step 5 — Final Explanation", expanded=True):
                st.write(result.get("response", "No explanation generated."))

        with col_metrics:
            st.subheader("📊 CCR Metrics")

            # Robustness gauge
            rob = metrics["robustness_score"]
            rob_color = "#22c55e" if rob >= 0.7 else "#f59e0b" if rob >= 0.5 else "#ef4444"
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{rob_color}">'
                f"<b>Robustness Score</b><br>"
                f"<span style='font-size:2rem;color:{rob_color}'>{rob:.3f}</span></div>",
                unsafe_allow_html=True,
            )

            for label, key in [
                ("Worst-case Score", "worst_case_score"),
                ("Epistemic Variance", "epistemic_variance"),
                ("Epistemic Stability", "epistemic_stability"),
                ("Decision Quality", "decision_quality"),
                ("Response Time (s)", "response_time"),
            ]:
                val = metrics.get(key, 0.0)
                st.markdown(
                    f'<div class="metric-card">'
                    f"<b>{label}</b><br>"
                    f"<span style='font-size:1.4rem'>{val:.3f}</span></div>",
                    unsafe_allow_html=True,
                )

            # Raw JSON toggle
            with st.expander("Raw JSON Output"):
                safe = {
                    "query": result["query"],
                    "selected_action": routing.selected_action,
                    "metrics": metrics,
                    "method": result.get("method"),
                    "context_mode": result.get("context_mode"),
                }
                st.json(safe)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Evaluation
# ─────────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Batch Evaluation — CCR vs Baseline")
    st.markdown(
        "Runs the full 10-case test suite and compares CCR against a greedy "
        "(mean-score) baseline."
    )

    n_cases = st.slider("Number of test cases to run", 1, 10, 3)

    if st.button("▶ Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                from evaluation.run_eval import TEST_SUITE, run_evaluation

                selected_cases = TEST_SUITE[:n_cases]
                eval_output = run_evaluation(
                    test_cases=selected_cases,
                    output_dir="results",
                    save_json=True,
                )
                st.session_state["eval_output"] = eval_output
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    if "eval_output" in st.session_state:
        eo = st.session_state["eval_output"]
        agg = eo["aggregate"]
        baseline = eo["baseline_comparison"]
        results_list = eo["results"]

        st.markdown("---")
        st.subheader("Aggregate Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pass Rate", f"{agg['robustness_pass_rate']:.1%}")
        c2.metric("Mean Robustness", f"{agg['mean_robustness']:.3f}")
        c3.metric("Mean Worst-case", f"{agg['mean_worst_case']:.3f}")
        c4.metric("Mean Variance", f"{agg['mean_epistemic_variance']:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Decision Quality", f"{agg['mean_decision_quality']:.3f}")
        c6.metric("KW Alignment", f"{agg['mean_keyword_alignment']:.3f}")
        c7.metric(
            "Worst-case vs Greedy",
            f"{baseline['mean_worst_case_improvement_vs_greedy']:+.1%}",
        )
        c8.metric(
            "Variance Reduction",
            f"{baseline['mean_variance_reduction_vs_greedy']:+.1%}",
        )

        st.markdown("---")
        st.subheader("Per-Query Results")

        for r in results_list:
            status_icon = "✅" if r["robustness_pass"] else "❌"
            with st.expander(
                f"{status_icon} [{r['id']}] {r['query'][:70]}", expanded=False
            ):
                col_a, col_b = st.columns(2)
                col_a.markdown(f"**Action:** {r['selected_action']}")
                col_b.markdown(f"**Category:** {r['category']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Robustness", f"{r['robustness_score']:.3f}")
                m2.metric("Variance", f"{r['epistemic_variance']:.3f}")
                m3.metric("Worst-case", f"{r['worst_case_score']:.3f}")
                m4.metric("KW Align", f"{r['keyword_alignment']:.3f}")
                ci = r["bootstrap_ci_95"]
                st.caption(
                    f"Bootstrap CI-95: [{ci['lower']:.3f}, {ci['upper']:.3f}]  |  "
                    f"Time: {r['response_time_s']:.1f}s"
                )

        st.markdown("---")
        st.subheader("Category Breakdown")
        for cat, info in agg["category_breakdown"].items():
            st.markdown(
                f"**{cat}** — mean robustness: `{info['mean_robustness']:.3f}`  "
                f"n={info['n']}"
            )

        with st.expander("Download Full JSON Report"):
            report_path = Path("results/evaluation_report.json")
            if report_path.exists():
                st.download_button(
                    "⬇ Download evaluation_report.json",
                    data=report_path.read_text(),
                    file_name="evaluation_report.json",
                    mime="application/json",
                )
