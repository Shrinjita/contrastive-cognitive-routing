from google import genai
from google.genai import types
import config

class GeminiClient:
    """Singleton Gemini client for consistent API usage"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiClient, cls).__new__(cls)
            cls._instance.client = genai.Client(api_key=config.config.GEMINI_API_KEY)
        return cls._instance
    
    def generate_content(self, prompt, system_instruction=None, temperature=None):
        """Generate content using Gemini API"""
        
        # Prepare config
        gen_config = types.GenerateContentConfig(
            temperature=temperature or config.config.TEMPERATURE,
            max_output_tokens=config.config.MAX_TOKENS
        )
        
        # Add system instruction if provided
        if system_instruction:
            gen_config.system_instruction = system_instruction
        
        try:
            response = self.client.models.generate_content(
                model=config.config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config
            )
            return response.text
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_chat(self, messages, system_instruction=None):
        """Generate content with chat history using Gemini"""
        client = self.client
        chat = client.chats.create(
            model=config.config.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or "",
                temperature=config.config.TEMPERATURE
            )
        )
        
        # Send all messages
        response = chat.send_message(messages[-1]["content"])
        return response.text
    
    def generate_content_stream(self, prompt, system_instruction=None):
        """Stream response from Gemini"""
        gen_config = types.GenerateContentConfig(
            temperature=config.config.TEMPERATURE,
            max_output_tokens=config.config.MAX_TOKENS
        )
        
        if system_instruction:
            gen_config.system_instruction = system_instruction
        
        response = self.client.models.generate_content_stream(
            model=config.config.GEMINI_MODEL,
            contents=prompt,
            config=gen_config
        )
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                print(chunk.text, end="", flush=True)
        
        return full_response