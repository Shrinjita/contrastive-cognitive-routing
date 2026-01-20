# core/llm.py
import openai
import time

class OpenAILLM:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def generate(self, prompt):
        return self._retry_guard(prompt)
    
    def _retry_guard(self, prompt):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500
                )
                text = response.choices[0].message.content
                return self._normalize_output(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
    
    def _normalize_output(self, text):
        return text.strip()