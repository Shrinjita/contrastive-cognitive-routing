import time
import requests
from typing import Optional
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import config
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config

class ModelClient:
    """Unified client for different model providers"""
    
    def __init__(self):
        self.provider = config.MODEL_PROVIDER
        self.model_name = ""
        self._setup_client()
    
    def _setup_client(self):
        """Setup client based on provider"""
        if self.provider == "ollama":
            self.model_name = config.OLLAMA_MODEL
            self.base_url = config.OLLAMA_BASE_URL
            print(f"  Using Ollama model: {self.model_name}")
            
        elif self.provider == "gemini" and config.GEMINI_API_KEY:
            try:
                from google import genai
                self.client = genai.Client(api_key=config.GEMINI_API_KEY)
                self.model_name = config.GEMINI_MODEL
                print(f"  Using Gemini model: {self.model_name}")
            except ImportError:
                print("  ⚠️  Google Generative AI not installed, falling back to Ollama")
                self.provider = "ollama"
                self.model_name = config.OLLAMA_MODEL
                self.base_url = config.OLLAMA_BASE_URL
                
        elif self.provider == "huggingface" and config.HF_TOKEN:
            try:
                from huggingface_hub import InferenceApi
                self.client = InferenceApi(
                    repo_id=config.HF_MODEL,
                    token=config.HF_TOKEN
                )
                self.model_name = config.HF_MODEL
                print(f"  Using Hugging Face model: {self.model_name}")
            except ImportError:
                print("  ⚠️  Hugging Face Hub not installed, falling back to Ollama")
                self.provider = "ollama"
                self.model_name = config.OLLAMA_MODEL
                self.base_url = config.OLLAMA_BASE_URL
                
        else:
            # Fallback to Ollama
            self.provider = "ollama"
            self.model_name = config.OLLAMA_MODEL
            self.base_url = config.OLLAMA_BASE_URL
            print(f"  Falling back to Ollama: {self.model_name}")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 500) -> str:
        """Generate text from model"""
        
        if self.provider == "ollama":
            return self._generate_ollama(prompt, temperature, max_tokens)
        
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, temperature, max_tokens)
        
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, temperature, max_tokens)
        
        else:
            return f"Error: No model provider configured"
    
    def _generate_ollama(self, prompt: str, temperature: float, 
                        max_tokens: int) -> str:
        """Generate using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response from Ollama")
            else:
                return f"Ollama error: {response.status_code}"
                
        except Exception as e:
            return f"Ollama connection error: {str(e)[:100]}"
    
    def _generate_gemini(self, prompt: str, temperature: float,
                        max_tokens: int) -> str:
        """Generate using Gemini"""
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            return response.text if hasattr(response, 'text') else "No text response"
            
        except Exception as e:
            return f"Gemini error: {str(e)[:100]}"
    
    def _generate_huggingface(self, prompt: str, temperature: float,
                             max_tokens: int) -> str:
        """Generate using Hugging Face"""
        try:
            params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
            
            result = self.client(inputs=prompt, params=params)
            
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict) and 'generated_text' in item:
                    return item['generated_text']
            
            return str(result)[:200] if result else "No response"
            
        except Exception as e:
            return f"Hugging Face error: {str(e)[:100]}"
    
    def check_connection(self) -> bool:
        """Check if model connection works"""
        try:
            test_response = self.generate("Test", max_tokens=10)
            return "error" not in test_response.lower()
        except:
            return False