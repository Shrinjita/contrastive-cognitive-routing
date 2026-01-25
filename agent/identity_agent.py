import json
import re
from typing import Dict, List, Optional, Tuple
from .memory_system import MemorySystem
from utils.gemini_client import GeminiClient
import config

class IdentityAgent:
    """Identity-aware agent with persistent memory and reasoning using Gemini"""
    
    def __init__(self, identity_path: str = config.config.IDENTITY_PATH):
        self.gemini = GeminiClient()
        self.memory = MemorySystem()
        self.load_identity(identity_path)
        self.load_policies()
        
    def load_identity(self, identity_path: str):
        """Load agent identity from JSON file"""
        with open(identity_path, 'r') as f:
            self.identity = json.load(f)
        
        # Create identity context for system instruction
        self.system_instruction = f"""
        You are the {self.identity['role']} at {self.identity.get('company_name', 'the company')}.
        
        Core Values: {', '.join(self.identity['company_values'])}
        
        Responsibilities:
        {chr(10).join(f'- {resp}' for resp in self.identity['core_responsibilities'])}
        
        Constraints:
        Financial: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['financial'])}
        Operational: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['operational'])}
        Ethical: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['ethical'])}
        
        Decision Framework (FOLLOW THESE STEPS for every decision):
        {chr(10).join(f'{i+1}. {step}' for i, step in enumerate(self.identity['decision_framework']['steps']))}
        
        Communication Style: {self.identity['communication_style']}
        
        IMPORTANT: Always reference your role, follow the decision framework, and cite relevant policies.
        """
    
    def load_policies(self):
        """Load company policies and past decisions"""
        try:
            with open(config.config.POLICIES_PATH, 'r') as f:
                self.policies_data = json.load(f)
        except FileNotFoundError:
            self.policies_data = {"policies": [], "past_decisions": []}
    
    def _extract_policy_references(self, text: str) -> List[str]:
        """Extract policy references from text"""
        policy_pattern = r'POL-\d{3}'
        return re.findall(policy_pattern, text)
    
    def _extract_decision_references(self, text: str) -> List[str]:
        """Extract decision references from text"""
        decision_pattern = r'DEC-\d{4}'
        return re.findall(decision_pattern, text)
    
    def _get_relevant_policies(self, query: str) -> List[Dict]:
        """Find relevant policies for a query"""
        relevant = []
        query_lower = query.lower()
        
        for policy in self.policies_data['policies']:
            policy_text = f"{policy['title']} {policy['content']}".lower()
            if any(word in policy_text for word in query_lower.split()[:10]):
                relevant.append(policy)
        
        return relevant[:3]
    
    def _get_relevant_past_decisions(self, query: str) -> List[Dict]:
        """Find relevant past decisions"""
        relevant = []
        query_lower = query.lower()
        
        for decision in self.policies_data['past_decisions']:
            decision_text = f"{decision['situation']} {decision['decision']}".lower()
            if any(word in decision_text for word in query_lower.split()[:10]):
                relevant.append(decision)
        
        memory_results = self.memory.search_memory(query, k=2)
        for result in memory_results:
            if result['type'] == 'decision':
                relevant.append(result['content'])
        
        return relevant[:3]
    
    def _build_prompt(self, query: str) -> str:
        """Build comprehensive prompt with context"""
        
        prompt_parts = []
        
        # 1. Current query
        prompt_parts.append(f"USER QUERY: {query}")
        prompt_parts.append("")
        
        # 2. Recent conversations
        recent_convos = self.memory.get_recent_context(3)
        if recent_convos:
            prompt_parts.append("RECENT CONVERSATIONS:")
            for conv in recent_convos:
                prompt_parts.append(f"- User: {conv['query'][:100]}...")
                prompt_parts.append(f"- You: {conv['response'][:100]}...")
            prompt_parts.append("")
        
        # 3. Relevant policies
        relevant_policies = self._get_relevant_policies(query)
        if relevant_policies:
            prompt_parts.append("RELEVANT COMPANY POLICIES:")
            for policy in relevant_policies:
                prompt_parts.append(f"Policy {policy['id']}: {policy['title']}")
                prompt_parts.append(f"Content: {policy['content'][:300]}")
            prompt_parts.append("")
        
        # 4. Relevant past decisions
        relevant_decisions = self._get_relevant_past_decisions(query)
        if relevant_decisions:
            prompt_parts.append("RELEVANT PAST DECISIONS:")
            for decision in relevant_decisions:
                if 'id' in decision:
                    prompt_parts.append(f"Decision {decision['id']}:")
                prompt_parts.append(f"Situation: {decision['situation'][:200]}")
                prompt_parts.append(f"Decision: {decision['decision'][:150]}")
                if 'reasoning' in decision:
                    prompt_parts.append(f"Reasoning: {decision['reasoning'][:150]}")
            prompt_parts.append("")
        
        # 5. Memory search results
        memory_results = self.memory.search_memory(query, k=2)
        if memory_results:
            prompt_parts.append("FROM YOUR MEMORY:")
            for result in memory_results:
                if result['type'] == 'conversation':
                    prompt_parts.append(f"- Previous query: {result['content']['query'][:100]}")
                    prompt_parts.append(f"  Your response: {result['content']['response'][:100]}")
            prompt_parts.append("")
        
        # 6. Response instructions
        prompt_parts.append("RESPONSE REQUIREMENTS:")
        prompt_parts.append("1. Start by acknowledging your role as Chief of Staff")
        prompt_parts.append("2. Reference relevant company values")
        prompt_parts.append("3. Cite applicable policies (use format 'POL-XXX')")
        prompt_parts.append("4. Follow the decision framework step-by-step")
        prompt_parts.append("5. Provide clear recommendation with reasoning")
        prompt_parts.append("6. State confidence level if uncertain")
        prompt_parts.append("7. Mention if escalation is needed")
        
        return "\n".join(prompt_parts)
    
    def query(self, user_query: str) -> Dict:
        """Process a user query with Gemini API"""
        
        # Build context-aware prompt
        prompt = self._build_prompt(user_query)
        
        # Generate response using Gemini
        response_text = self.gemini.generate_content(
            prompt=prompt,
            system_instruction=self.system_instruction,
            temperature=config.config.TEMPERATURE
        )
        
        # Extract structured information
        policy_refs = self._extract_policy_references(response_text)
        decision_refs = self._extract_decision_references(response_text)
        
        # Create decision record
        decision_record = {
            'query': user_query,
            'response': response_text,
            'policy_references': policy_refs,
            'decision_references': decision_refs,
            'timestamp': self.memory.past_decisions[-1]['timestamp'] if self.memory.past_decisions else None
        }
        
        # Add to memory
        self.memory.add_conversation(user_query, response_text)
        
        # Check if this is a decision that should be recorded
        decision_keywords = ['approve', 'deny', 'recommend', 'decide', 'suggest', 'propose']
        if any(keyword in user_query.lower() for keyword in decision_keywords):
            self.memory.add_decision({
                'situation': user_query,
                'decision': response_text[:200],
                'reasoning': response_text,
                'constraints_referenced': policy_refs
            })
        
        return {
            'response': response_text,
            'policy_references': policy_refs,
            'decision_references': decision_refs,
            'identity_consistency': self._check_identity_consistency(response_text),
            'memory_used': len(policy_refs) + len(decision_refs)
        }
    
    def _check_identity_consistency(self, response: str) -> float:
        """Check how consistent response is with identity"""
        score = 0
        total_checks = 4
        
        # Check 1: Role mention
        if "chief of staff" in response.lower() or self.identity['role'].split()[0].lower() in response.lower():
            score += 1
        
        # Check 2: Company values mentioned
        values_mentioned = sum(1 for value in self.identity['company_values'] 
                             if any(word.lower() in response.lower() for word in value.split()))
        if values_mentioned >= 1:
            score += 1
        
        # Check 3: Decision framework followed
        framework_steps = len(self.identity['decision_framework']['steps'])
        steps_mentioned = sum(1 for step in self.identity['decision_framework']['steps']
                            if any(word.lower() in response.lower() for word in step.split()[:3]))
        if steps_mentioned >= 2:
            score += 1
        
        # Check 4: Professional tone
        professional_indicators = ['according to', 'policy', 'recommend', 'consider', 'suggest', 'based on']
        if any(indicator in response.lower() for indicator in professional_indicators):
            score += 1
        
        return score / total_checks
    
    def streaming_query(self, user_query: str):
        """Process query with streaming response"""
        prompt = self._build_prompt(user_query)
        
        print(f"\n{'─' * 60}")
        print(f"Query: {user_query}")
        print(f"{'─' * 60}")
        print("\nResponse (streaming):\n")
        
        response_text = self.gemini.generate_content_stream(
            prompt=prompt,
            system_instruction=self.system_instruction
        )
        
        # Store in memory
        self.memory.add_conversation(user_query, response_text)
        
        return response_text