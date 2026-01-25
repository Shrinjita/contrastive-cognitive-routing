# config.py
import os

OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEMPERATURE = 0.0
MAX_TOKENS = 500
RAG_TOP_K = 3
RLM_SUMMARY_TRIGGER = 10
MEMORY_PATH = "data/memory.json"
VECTOR_STORE_PATH = "data/vectors.pkl"