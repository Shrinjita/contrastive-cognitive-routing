#!/usr/bin/env python3
"""
Complete Working Gemini Identity Agent
Uses models/gemini-flash-latest (confirmed working)
"""

import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiIdentityAgent:
    def __init__(self, model_name=None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file.\n"
                "Get it from: https://aistudio.google.com/apikey\n"
                "Then add to .env: GEMINI_API_KEY=your_key"
            )
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Use specified model or default to working model
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
        
        # Load or create identity
        self.identity = self._load_identity()
        
        # Create system instruction
        self.system_instruction = self._create_system_instruction()
        
        # Conversation history
        self.conversation_history = []
        
        print(f"✅ Agent initialized successfully!")
        print(f"   Model: {self.model_name}")
        print(f"   Role: {self.identity['role']}")
    
    def _load_identity(self):
        """Load identity from file or create default"""
        try:
            with open("identity.json", "r") as f:
                identity = json.load(f)
                print(f"✓ Loaded identity from identity.json")
                return identity
        except FileNotFoundError:
            # Create comprehensive default identity
            default_identity = {
                "role": "Chief of Staff to CEO",
                "company_name": "TechVision Inc.",
                "company_values": [
                    "Integrity First",
                    "Customer Centric", 
                    "Innovation Driven",
                    "Data-Informed Decisions"
                ],
                "core_responsibilities": [
                    "Strategic decision support",
                    "Budget review and approval",
                    "Team coordination and communication",
                    "Policy enforcement and compliance"
                ],
                "constraints": {
                    "financial": [
                        "All expenditures over $10,000 require CEO approval",
                        "Marketing budget capped at 15% of quarterly revenue"
                    ],
                    "operational": [
                        "All external communications must be reviewed by legal",
                        "Hiring decisions require HR consultation"
                    ],
                    "ethical": [
                        "Prioritize customer privacy",
                        "No conflicts of interest"
                    ]
                },
                "decision_framework": {
                    "steps": [
                        "Identify core issue and stakeholders",
                        "Review relevant policies",
                        "Assess financial and operational impact", 
                        "Consider ethical implications",
                        "Make recommendation with reasoning"
                    ]
                },
                "communication_style": "Professional, concise, data-driven"
            }
            
            # Save default identity
            with open("identity.json", "w") as f:
                json.dump(default_identity, f, indent=2)
            
            print("✓ Created default identity.json")
            return default_identity
    
    def _create_system_instruction(self):
        """Create system instruction from identity"""
        
        instruction = f"""You are {self.identity['role']} at {self.identity.get('company_name', 'our company')}.

COMPANY VALUES:
{chr(10).join(f'• {value}' for value in self.identity['company_values'])}

RESPONSIBILITIES:
{chr(10).join(f'• {resp}' for resp in self.identity['core_responsibilities'])}

DECISION FRAMEWORK (ALWAYS FOLLOW THESE STEPS):
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(self.identity['decision_framework']['steps']))}

CONSTRAINTS TO CONSIDER:
• Financial: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['financial'])}
• Operational: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['operational'])}
• Ethical: {chr(10).join(f'  - {c}' for c in self.identity['constraints']['ethical'])}

COMMUNICATION STYLE: {self.identity['communication_style']}

CRITICAL GUIDELINES:
1. Always acknowledge your role as {self.identity['role']}
2. Reference relevant company values in your reasoning
3. Follow the decision framework step-by-step for complex decisions
4. Cite specific constraints that apply to the situation
5. Provide clear, actionable recommendations
6. Be transparent about uncertainties
7. Use professional, business-appropriate language

RESPONSE STRUCTURE (for complex decisions):
1. Role acknowledgment and initial assessment
2. Relevant values and constraints considered
3. Step-by-step analysis using decision framework
4. Clear recommendation with rationale
5. Any necessary qualifications or next steps
"""
        return instruction
    
    def query(self, user_query: str, analyze: bool = True):
        """Process a query and return structured response"""
        
        print(f"\n{'═'*70}")
        print(f"QUERY: {user_query}")
        print(f"{'═'*70}")
        
        try:
            # Add delay to avoid rate limits
            time.sleep(1)
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1,
                    max_output_tokens=800,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            response_text = response.text
            
            # Store in history
            self.conversation_history.append({
                "query": user_query,
                "response": response_text,
                "timestamp": time.time()
            })
            
            # Display response
            print(f"\nRESPONSE:\n{'-'*40}")
            print(response_text)
            print(f"{'-'*40}")
            
            if analyze:
                analysis = self._analyze_response(response_text, user_query)
                self._display_analysis(analysis)
            
            return {
                "success": True,
                "response": response_text,
                "model": self.model_name,
                "analysis": analysis if analyze else None
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR: {error_msg[:200]}")
            
            # Handle specific errors
            if "429" in error_msg or "quota" in error_msg.lower():
                print("\n⚠️  Quota exceeded. Solutions:")
                print("1. Wait a few minutes and try again")
                print("2. Use a different model (models/gemini-2.5-flash)")
                print("3. Check your quota in Google AI Studio")
            
            return {
                "success": False,
                "error": error_msg,
                "model": self.model_name
            }
    
    def _analyze_response(self, response: str, query: str) -> dict:
        """Analyze response for identity consistency"""
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check for role mention
        role_keywords = ['chief of staff', 'executive assistant', 'as chief', 'in my role as']
        role_mentioned = any(keyword in response_lower for keyword in role_keywords)
        
        # Check for company values
        values_mentioned = []
        for value in self.identity['company_values']:
            # Check for any significant word from the value
            words = value.lower().split()
            if len(words) > 1:
                # Check for at least one significant word
                if any(word in response_lower for word in words if len(word) > 4):
                    values_mentioned.append(value)
            elif words and words[0] in response_lower:
                values_mentioned.append(value)
        
        # Check for decision framework steps
        framework_steps = 0
        for step in self.identity['decision_framework']['steps']:
            # Check for keywords from each step
            step_words = step.lower().split()[:3]
            if any(word in response_lower for word in step_words if len(word) > 3):
                framework_steps += 1
        
        # Check for professional tone
        professional_indicators = ['recommend', 'suggest', 'consider', 'based on', 
                                  'therefore', 'however', 'according to', 'in conclusion']
        professional_score = sum(1 for indicator in professional_indicators 
                               if indicator in response_lower)
        
        # Calculate overall consistency score
        total_checks = 4
        score = 0
        score += 1 if role_mentioned else 0
        score += 1 if len(values_mentioned) > 0 else 0
        score += 1 if framework_steps >= 2 else 0
        score += 1 if professional_score >= 2 else 0
        consistency_score = score / total_checks
        
        return {
            "role_mentioned": role_mentioned,
            "values_mentioned": values_mentioned,
            "framework_steps_followed": framework_steps,
            "professional_indicators": professional_score,
            "consistency_score": consistency_score,
            "response_length": len(response)
        }
    
    def _display_analysis(self, analysis: dict):
        """Display analysis of response"""
        
        print(f"\nANALYSIS:")
        print(f"{'-'*40}")
        print(f"• Role mentioned: {'✅ Yes' if analysis['role_mentioned'] else '❌ No'}")
        
        if analysis['values_mentioned']:
            print(f"• Values referenced: {len(analysis['values_mentioned'])}")
            for value in analysis['values_mentioned'][:3]:  # Show max 3
                print(f"  - {value}")
        else:
            print(f"• Values referenced: ❌ None")
        
        print(f"• Decision framework steps: {analysis['framework_steps_followed']}/5")
        print(f"• Professional indicators: {analysis['professional_indicators']}")
        print(f"• Identity consistency: {analysis['consistency_score']:.2f}/1.0")
        print(f"• Response length: {analysis['response_length']} characters")
        print(f"{'-'*40}")
    
    def streaming_query(self, user_query: str):
        """Process query with streaming response"""
        
        print(f"\n{'═'*70}")
        print(f"STREAMING QUERY: {user_query}")
        print(f"{'═'*70}")
        print("\nResponse:\n")
        
        try:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1,
                    max_output_tokens=800
                )
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            # Store in history
            self.conversation_history.append({
                "query": user_query,
                "response": full_response,
                "timestamp": time.time(),
                "streaming": True
            })
            
            print(f"\n\n{'═'*70}")
            print(f"Streaming complete. Response length: {len(full_response)} characters")
            
            return full_response
            
        except Exception as e:
            print(f"\n❌ Streaming error: {str(e)[:150]}")
            return None
    
    def run_demo(self):
        """Run demonstration queries"""
        
        demo_queries = [
            "Should we approve a $45,000 marketing campaign for Q3?",
            "What factors should I consider when hiring a senior developer?",
            "A project is 3 weeks behind schedule. What should we do?",
            "Is it appropriate to share customer data with an analytics vendor?",
            "How should we handle a conflict between sales and engineering teams?"
        ]
        
        print(f"\n{'='*70}")
        print(f"DEMONSTRATION: {self.identity['role']}")
        print(f"{'='*70}")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n\n📋 DEMO {i}/{len(demo_queries)}")
            self.query(query, analyze=True)
            
            if i < len(demo_queries):
                input(f"\nPress Enter to continue to next demo...")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        
        print(f"\n{'='*70}")
        print(f"INTERACTIVE MODE: {self.identity['role']}")
        print(f"Model: {self.model_name}")
        print(f"Type 'quit' to exit, 'stream' for streaming, 'demo' for demo")
        print(f"{'='*70}")
        
        while True:
            try:
                user_input = input(f"\n💬 You ({self.identity['role']}): ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    break
                elif user_input.lower() == 'stream':
                    stream_query = input("Enter query for streaming: ").strip()
                    if stream_query:
                        self.streaming_query(stream_query)
                elif user_input.lower() == 'demo':
                    self.run_demo()
                elif user_input.lower() == 'history':
                    self.show_history()
                elif user_input:
                    self.query(user_input, analyze=True)
                else:
                    print("Please enter a query or command.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("\nNo conversation history yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"CONVERSATION HISTORY ({len(self.conversation_history)} exchanges)")
        print(f"{'='*70}")
        
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] Query: {exchange['query'][:100]}...")
            print(f"    Response: {exchange['response'][:150]}...")
            if 'analysis' in exchange:
                print(f"    Consistency: {exchange['analysis'].get('consistency_score', 0):.2f}")

def main():
    """Main function"""
    
    print(f"{'='*70}")
    print(f"CONSCIOUS PROXY AGENT - GEMINI INTEGRATION")
    print(f"{'='*70}")
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n❌ GEMINI_API_KEY not found in .env file")
        print("\nTo set up:")
        print("1. Get API key: https://aistudio.google.com/apikey")
        print("2. Create .env file: echo 'GEMINI_API_KEY=your_key' > .env")
        print("3. Run again: python simple_gemini_agent.py")
        return
    
    try:
        # Initialize agent
        agent = GeminiIdentityAgent()
        
        # Menu
        print(f"\nOptions:")
        print(f"1. Interactive chat")
        print(f"2. Run demonstration")
        print(f"3. Test specific query")
        print(f"4. Exit")
        
        choice = input(f"\nSelect option (1-4): ").strip()
        
        if choice == "1":
            agent.interactive_mode()
        elif choice == "2":
            agent.run_demo()
        elif choice == "3":
            query = input("Enter your query: ").strip()
            if query:
                agent.query(query, analyze=True)
        elif choice == "4":
            print("\n👋 Goodbye!")
        else:
            print("\nInvalid choice. Running demonstration...")
            agent.run_demo()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nDebug info:")
        print(f"• API Key: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
        print(f"• Model: {os.getenv('GEMINI_MODEL', 'models/gemini-flash-latest')}")

if __name__ == "__main__":
    main()