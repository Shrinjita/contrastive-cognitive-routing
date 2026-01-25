import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
    
    # Alternative working models based on your tests
    ALTERNATIVE_MODELS = [
        "models/gemini-flash-latest",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-pro-latest"
    ]
    
    # Paths
    IDENTITY_PATH = "identity.json"
    POLICIES_PATH = "company_policies.json"
    TEST_SUITE_PATH = "test_suite.json"
    
    # Agent Configuration
    MAX_MEMORY_ITEMS = 100
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Evaluation
    N_TEST_CASES = 5  # Start small
    METRICS = ["identity_consistency", "memory_accuracy", "decision_quality"]

config = Config()