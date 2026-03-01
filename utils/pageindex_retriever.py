"""
PageIndex Retriever — Vectorless, reasoning-based RAG over policy/identity documents.

Replaces the flat string concatenation in proxy_agent._build_context() with
hierarchical tree-search retrieval. Two modes:

  1. LOCAL  — builds a tree from the repo's JSON files on first run,
              then uses an LLM-guided tree-search to answer queries.
              Requires: CHATGPT_API_KEY (or OPENAI_API_KEY) in .env
              Does NOT require a PageIndex account.

  2. CLOUD  — uses the PageIndex SDK (pip install -U pageindex) for
              richer tree construction on real PDFs.
              Requires: PAGEINDEX_API_KEY in .env

The agent always falls back to local mode if the SDK / key is absent.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path) -> Any:
    with open(str(path), "r") as f:
        return json.load(f)


def _flatten_tree(node: Dict, depth: int = 0) -> str:
    """Recursively flatten a PageIndex-style tree node to readable text."""
    indent = "  " * depth
    parts = [f"{indent}[{node.get('node_id', '?')}] {node.get('title', '')}"]
    if node.get("summary"):
        parts.append(f"{indent}    {node['summary']}")
    for child in node.get("nodes", []):
        parts.append(_flatten_tree(child, depth + 1))
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Local Tree Builder  (JSON → PageIndex-style tree without the SDK)
# ─────────────────────────────────────────────────────────────────────────────

class LocalDocumentTree:
    """
    Converts the repo's JSON configs into a PageIndex-compatible hierarchical
    tree so that tree-search retrieval works without a network call.

    Tree schema mirrors PageIndex:
      {title, node_id, summary, nodes: [...]}
    """

    def __init__(self):
        self.trees: Dict[str, Dict] = {}  # doc_name → tree root
        self._build_from_repo()

    def _build_from_repo(self):
        """Build trees from identity.json and company_policies.json."""

        # ── Identity tree ────────────────────────────────────────────────────
        try:
            identity = _load_json(config.IDENTITY_PATH)
            self.trees["identity"] = {
                "title": f"Agent Identity — {identity.get('role', 'Agent')}",
                "node_id": "ID-ROOT",
                "summary": (
                    f"Role: {identity.get('role')}. "
                    f"Company: {identity.get('company_name', 'N/A')}. "
                    f"Values: {', '.join(identity.get('company_values', []))}."
                ),
                "nodes": [
                    {
                        "title": "Core Responsibilities",
                        "node_id": "ID-01",
                        "summary": "; ".join(identity.get("core_responsibilities", [])),
                        "nodes": [],
                    },
                    {
                        "title": "Constraints",
                        "node_id": "ID-02",
                        "summary": json.dumps(identity.get("constraints", {})),
                        "nodes": [
                            {
                                "title": f"Constraint: {k.capitalize()}",
                                "node_id": f"ID-02-{i}",
                                "summary": "; ".join(v) if isinstance(v, list) else str(v),
                                "nodes": [],
                            }
                            for i, (k, v) in enumerate(
                                identity.get("constraints", {}).items()
                            )
                        ],
                    },
                    {
                        "title": "Decision Framework",
                        "node_id": "ID-03",
                        "summary": (
                            "Steps: "
                            + "; ".join(
                                identity.get("decision_framework", {}).get("steps", [])
                            )
                            + " | Escalation triggers: "
                            + "; ".join(
                                identity.get("decision_framework", {}).get(
                                    "escalation_triggers", []
                                )
                            )
                        ),
                        "nodes": [],
                    },
                ],
            }
        except Exception as e:
            print(f"  ⚠️  LocalDocumentTree: identity load failed — {e}")

        # ── Policies tree ────────────────────────────────────────────────────
        try:
            policies_doc = _load_json(config.POLICIES_PATH)
            policy_nodes = [
                {
                    "title": p["title"],
                    "node_id": p["id"],
                    "summary": p["content"],
                    "nodes": [],
                }
                for p in policies_doc.get("policies", [])
            ]
            decision_nodes = [
                {
                    "title": f"Past Decision: {d['situation'][:60]}",
                    "node_id": d["id"],
                    "summary": (
                        f"Decision: {d['decision']}. "
                        f"Reasoning: {d['reasoning']}. "
                        f"Policies referenced: {', '.join(d.get('constraints_referenced', []))}."
                    ),
                    "nodes": [],
                }
                for d in policies_doc.get("past_decisions", [])
            ]
            self.trees["policies"] = {
                "title": "Company Policies & Past Decisions",
                "node_id": "POL-ROOT",
                "summary": (
                    f"{len(policy_nodes)} active policies, "
                    f"{len(decision_nodes)} recorded past decisions."
                ),
                "nodes": [
                    {
                        "title": "Active Policies",
                        "node_id": "POL-SEC-1",
                        "summary": "Binding operational and financial policies.",
                        "nodes": policy_nodes,
                    },
                    {
                        "title": "Past Decisions",
                        "node_id": "POL-SEC-2",
                        "summary": "Historical decisions for precedent lookup.",
                        "nodes": decision_nodes,
                    },
                ],
            }
        except Exception as e:
            print(f"  ⚠️  LocalDocumentTree: policies load failed — {e}")

    def get_tree_text(self, doc_name: str) -> str:
        tree = self.trees.get(doc_name)
        if not tree:
            return ""
        return _flatten_tree(tree)

    def get_all_summaries(self) -> str:
        parts = []
        for name, tree in self.trees.items():
            parts.append(f"=== {name.upper()} ===")
            parts.append(_flatten_tree(tree))
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Tree Search  (LLM-guided, mirrors PageIndex tree-search logic)
# ─────────────────────────────────────────────────────────────────────────────

class LocalTreeSearcher:
    """
    Implements PageIndex-style tree search over LocalDocumentTree nodes.

    At each tree level the LLM selects which child node(s) to descend into,
    then extracts the relevant summary. This is the same two-step process
    PageIndex uses: (1) reason over the index, (2) retrieve content.
    """

    def __init__(self, model_client, max_depth: int = 3):
        self.model = model_client
        self.max_depth = max_depth

    def search(self, query: str, tree: Dict, depth: int = 0) -> List[str]:
        """
        Recursively descend the tree to collect relevant summaries.
        Returns a list of relevant text chunks.
        """
        if depth >= self.max_depth or not tree:
            return []

        children = tree.get("nodes", [])
        if not children:
            return [tree.get("summary", "")]

        # Build a compact index of child titles + summaries for the LLM
        index_text = "\n".join(
            f"[{c['node_id']}] {c['title']}: {c.get('summary', '')[:150]}"
            for c in children
        )

        prompt = (
            f"You are navigating a document index to answer this query:\n"
            f"QUERY: {query}\n\n"
            f"Current section: {tree['title']}\n"
            f"Child sections available:\n{index_text}\n\n"
            f"List the node_ids (comma-separated) of the sections most relevant "
            f"to the query. Return ONLY the node_ids, nothing else."
        )

        response = self.model.generate(prompt, temperature=0.0, max_tokens=50)
        selected_ids = {
            nid.strip()
            for nid in re.split(r"[,\s]+", response)
            if nid.strip()
        }

        results = []
        for child in children:
            if child["node_id"] in selected_ids:
                results.append(
                    f"[{child['node_id']}] {child['title']}:\n{child.get('summary', '')}"
                )
                # Recurse
                results.extend(self.search(query, child, depth + 1))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# PageIndex Retriever  (public interface used by proxy_agent)
# ─────────────────────────────────────────────────────────────────────────────

class PageIndexRetriever:
    """
    Drop-in replacement for the flat _build_context() in proxy_agent.py.

    Usage:
        retriever = PageIndexRetriever(model_client)
        context   = retriever.retrieve(query)

    Automatically selects cloud vs. local mode.
    """

    def __init__(self, model_client=None):
        self.model_client = model_client
        self._cloud_client = None
        self._local_tree = None
        self._searcher = None
        self._mode = "uninitialized"
        self._setup()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup(self):
        pageindex_key = os.getenv("PAGEINDEX_API_KEY", "")
        if pageindex_key:
            try:
                from pageindex import PageIndexClient  # noqa: F401
                self._cloud_client = PageIndexClient(api_key=pageindex_key)
                self._mode = "cloud"
                print("  ✓ PageIndexRetriever: cloud mode (SDK)")
                return
            except ImportError:
                print("  ⚠️  pageindex SDK not installed — falling back to local mode")
            except Exception as e:
                print(f"  ⚠️  PageIndex cloud init failed: {e} — falling back to local mode")

        # Local mode
        self._local_tree = LocalDocumentTree()
        if self.model_client:
            self._searcher = LocalTreeSearcher(self.model_client)
        self._mode = "local"
        print("  ✓ PageIndexRetriever: local tree-search mode")

    # ── Public API ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> str:
        """
        Returns a context string constructed by tree-search over the document
        corpus, ranked by relevance to `query`.
        """
        if self._mode == "cloud":
            return self._retrieve_cloud(query)
        return self._retrieve_local(query)

    # ── Cloud mode ───────────────────────────────────────────────────────────

    def _retrieve_cloud(self, query: str) -> str:
        """Query the PageIndex Chat API with the agent's document corpus."""
        try:
            # Retrieve all indexed doc IDs stored on first index
            doc_ids = self._get_or_index_cloud_docs()
            if not doc_ids:
                return self._retrieve_local(query)

            response = self._cloud_client.chat_completions(
                messages=[{"role": "user", "content": query}],
                doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
            )
            content = response["choices"][0]["message"]["content"]
            return f"[PageIndex Cloud Retrieval]\n{content}"
        except Exception as e:
            print(f"  ⚠️  PageIndex cloud retrieval failed: {e} — using local")
            return self._retrieve_local(query)

    def _get_or_index_cloud_docs(self) -> List[str]:
        """Submit agent docs to PageIndex cloud if not already indexed."""
        cache_path = config.DATA_DIR / ".pageindex_doc_ids.json"
        if cache_path.exists():
            return _load_json(cache_path)

        doc_ids = []
        for path in [config.POLICIES_PATH, config.IDENTITY_PATH]:
            try:
                result = self._cloud_client.submit_document(str(path))
                doc_ids.append(result["doc_id"])
            except Exception as e:
                print(f"  ⚠️  Cloud indexing failed for {path}: {e}")

        if doc_ids:
            cache_path.write_text(json.dumps(doc_ids))
        return doc_ids

    # ── Local mode ───────────────────────────────────────────────────────────

    def _retrieve_local(self, query: str) -> str:
        """
        Tree-search over local JSON documents.
        Falls back to flat dump when no model_client is available.
        """
        if not self._local_tree:
            return self._fallback_context()

        # Fast path: no LLM available → return full tree summaries
        if not self._searcher:
            return self._local_tree.get_all_summaries()

        all_chunks: List[str] = []
        for doc_name, tree in self._local_tree.trees.items():
            chunks = self._searcher.search(query, tree)
            if chunks:
                all_chunks.append(f"=== {doc_name.upper()} ===")
                all_chunks.extend(chunks)

        if not all_chunks:
            # Searcher returned nothing — return full summaries
            return self._local_tree.get_all_summaries()

        return "\n".join(all_chunks)

    def _fallback_context(self) -> str:
        """Last-resort: read raw JSON files and return as text."""
        parts = []
        for path in [config.IDENTITY_PATH, config.POLICIES_PATH]:
            try:
                parts.append(f"=== {path.stem.upper()} ===")
                parts.append(json.dumps(_load_json(path), indent=2))
            except Exception:
                pass
        return "\n".join(parts)
