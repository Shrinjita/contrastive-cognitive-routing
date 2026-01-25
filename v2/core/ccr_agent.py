# core/ccr_agent.py
from core.epistemic import detect_uncertainty, extract_claims

class CCRDecisionAgent:
    def __init__(self, llm, memory, epistemic):
        self.llm = llm
        self.memory = memory
        self.epistemic = epistemic
    
    def _route(self, query):
        memory_text = self.memory.recall()
        
        if not memory_text:
            return 'direct'
        
        if detect_uncertainty(memory_text):
            return 'refuse'
        
        return 'contrast'
    
    def _contrast(self, memory, query):
        prompt = f"""Memory state:
{memory}

Query: {query}

Compare memory against query. Identify conflicts, gaps, and consistencies."""
        
        return self.llm.generate(prompt)
    
    def _decide_with_constraints(self, query, memory):
        contrast = self._contrast(memory, query)
        
        prompt = f"""Memory: {memory}

Contrastive Analysis: {contrast}

Query: {query}

Decision: Provide a decision. If uncertain or conflicting, output REFUSE. Include confidence (0-1) and justification."""
        
        return self.llm.generate(prompt)
    
    def decide(self, query):
        route = self._route(query)
        
        if route == 'refuse':
            return {
                'decision': 'REFUSE',
                'confidence': 0.0,
                'justification': 'Epistemic uncertainty in memory',
                'refusal_flag': True
            }
        
        memory = self.memory.recall()
        response = self._decide_with_constraints(query, memory)
        
        refusal = 'REFUSE' in response.upper()
        confidence = 0.5
        
        if 'confidence' in response.lower():
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                confidence = float(match.group(1))
        
        return {
            'decision': response,
            'confidence': confidence,
            'justification': response,
            'refusal_flag': refusal
        }