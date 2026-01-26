import json
import time
import re
from typing import Dict, List
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.epistemic_variants import EpistemicVariantGenerator
from core.contrastive_router import ContrastiveCognitiveRouter, RoutingResult

try:
    from utils.model_client import ModelClient
    from config import config
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.model_client import ModelClient
    from config import config

class EpistemicProxyAgent:
    """
    Epistemic-aware proxy agent using Contrastive Cognitive Routing
    """
    
    def __init__(self):
        self.model_client = ModelClient()
        self.variant_generator = EpistemicVariantGenerator()
        self.router = ContrastiveCognitiveRouter(self.LLMScorer(self.model_client))
        self.load_identity()
        
        print(f"✅ Epistemic Proxy Agent initialized")
        print(f"   Method: Contrastive Cognitive Routing (CCR)")
        print(f"   Core Equation: a* = arg max_a min_C' P(a|x,C')")
    
    class LLMScorer:
        """Wrapper to score actions using LLM"""
        def __init__(self, model_client):
            self.model = model_client
        
        def score_actions(self, query: str, context: str, 
                         actions: List[str]) -> List[float]:
            """Score actions P(a|x,C') using LLM"""
            scores = []
            
            for action in actions:
                prompt = self._create_scoring_prompt(query, context, action)
                response = self.model.generate(prompt, temperature=0.1, max_tokens=10)
                score = self._extract_score(response)
                scores.append(score)
            
            return scores
        
        def _create_scoring_prompt(self, query: str, context: str, 
                                 action: str) -> str:
            return f"""Given this context and query, rate how appropriate the action is.

Context: {context[:300]}

Query: {query}

Proposed Action: {action}

Rate appropriateness on scale 0.0 to 1.0 where:
0.0 = Completely inappropriate
0.5 = Neutral/Uncertain  
1.0 = Perfectly appropriate

Consider alignment with context and practical feasibility.
Return ONLY a number:"""
        
        def _extract_score(self, response: str) -> float:
            """Extract numeric score from response"""
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            if numbers:
                score = float(numbers[0])
                if score > 10:  # Might be percentage
                    score = score / 100.0
                elif score > 1:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            return 0.5  # Default
    
    def load_identity(self):
        """Load agent identity"""
        try:
            with open(str(config.IDENTITY_PATH), 'r') as f:
                self.identity = json.load(f)
            print(f"  ✓ Loaded identity from {config.IDENTITY_PATH}")
        except Exception as e:
            print(f"  ⚠️  Error loading identity: {e}")
            self.identity = {
                "role": "Chief of Staff",
                "company_values": ["Integrity", "Innovation", "Customer Focus"]
            }
    
    def process_query(self, query: str) -> Dict:
        """
        Process query using Contrastive Cognitive Routing
        """
        start_time = time.time()
        
        # Step 1: Build context
        context = self._build_context(query)
        
        # Step 2: Generate candidate actions
        candidate_actions = self._generate_candidate_actions(query, context)
        
        # Step 3: Perform Contrastive Cognitive Routing
        print("  🔀 Performing Contrastive Cognitive Routing...")
        routing_result = self.router.route(query, context, candidate_actions)
        
        # Step 4: Generate explanation
        explanation = self._generate_explanation(query, routing_result)
        
        # Step 5: Calculate metrics
        response_time = time.time() - start_time
        metrics = self._calculate_ccr_metrics(routing_result, response_time)
        
        return {
            'query': query,
            'response': explanation,
            'routing_result': routing_result,
            'metrics': metrics,
            'method': 'contrastive_cognitive_routing'
        }
    
    def _build_context(self, query: str) -> str:
        """Build decision context"""
        context_parts = []
        
        # Add identity
        context_parts.append(f"ROLE: {self.identity['role']}")
        context_parts.append(f"VALUES: {', '.join(self.identity['company_values'])}")
        
        # Add relevant constraints
        if 'financial' in query.lower() or '$' in query:
            context_parts.append("CONSTRAINT: Expenditures over $10,000 require CEO approval")
        
        return '\n'.join(context_parts)
    
    def _generate_candidate_actions(self, query: str, context: str) -> List[str]:
        """Generate diverse candidate actions"""
        prompt = f"""Based on this situation, generate 4-5 possible decisions:

Situation: {query}
Context: {context}

Generate diverse options including:
1. Conservative/risk-averse approach
2. Ambitious/optimistic approach
3. Compromise/middle-ground approach
4. Deferral/more-info approach

Format each as a concise action starting with a verb.
One per line:"""
        
        response = self.model_client.generate(prompt, temperature=0.8, max_tokens=150)
        
        # Parse actions
        actions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                # Remove numbering
                if line[0].isdigit() and '. ' in line[:5]:
                    line = line.split('. ', 1)[1]
                actions.append(line)
        
        # Ensure we have actions
        if not actions:
            actions = [
                "Approve with standard conditions",
                "Request additional information",
                "Deny based on constraints",
                "Approve with modified scope"
            ]
        
        return actions[:5]
    
    def _generate_explanation(self, query: str, 
                            routing_result: RoutingResult) -> str:
        """Generate explanation of CCR decision"""
        
        prompt = f"""As {self.identity['role']}, explain this decision with epistemic reasoning:

Query: {query}

Selected Action: {routing_result.selected_action}

Epistemic Analysis:
- Selected through contrastive cognitive routing
- Robustness score: {routing_result.robustness_score:.3f}
- Worst-case performance: {routing_result.worst_case_score:.3f}
- Epistemic variance: {routing_result.epistemic_variance:.3f}

Explain:
1. Why this action was selected
2. How it performs across different information scenarios
3. Confidence level given epistemic uncertainties
4. Any conditions or monitoring needed

Decision Memo:"""
        
        return self.model_client.generate(prompt, temperature=0.3)
    
    def _calculate_ccr_metrics(self, routing_result: RoutingResult, 
                              response_time: float) -> Dict:
        """Calculate CCR-specific metrics"""
        return {
            'robustness_score': round(routing_result.robustness_score, 3),
            'worst_case_score': round(routing_result.worst_case_score, 3),
            'epistemic_variance': round(routing_result.epistemic_variance, 3),
            'response_time': round(response_time, 3),
            'epistemic_stability': round(1.0 - routing_result.epistemic_variance, 3),
            'decision_quality': round(routing_result.robustness_score * 0.7 + 
                                     routing_result.worst_case_score * 0.3, 3)
        }