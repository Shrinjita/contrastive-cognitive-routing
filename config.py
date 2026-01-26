import os
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory
PROJECT_ROOT = Path(__file__).parent

load_dotenv()

class Config:
    # Model Configuration
    MODEL_PROVIDER = "ollama"  # "ollama", "gemini", or "huggingface"
    OLLAMA_MODEL = "phi"  # phi, mistral, llama2, etc.
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # Gemini Configuration (optional fallback)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "models/gemini-flash-latest"
    
    # Hugging Face (optional fallback)
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_MODEL = "gpt2"
    
    # Epistemic Layer Configuration
    EPISTEMIC_N_VARIANTS = 3
    EPISTEMIC_TEMPERATURE = 0.7
    CONFIDENCE_THRESHOLD = 0.7
    VARIANCE_THRESHOLD = 0.3
    
    # Data Paths (relative to project root)
    DATA_DIR = PROJECT_ROOT / "data"
    CONFIGS_DIR = DATA_DIR / "configs"
    POLICIES_DIR = DATA_DIR / "policies"
    TEST_DIR = DATA_DIR / "test_cases"
    TRAINING_DIR = DATA_DIR / "training"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # File paths
    IDENTITY_PATH = CONFIGS_DIR / "identity.json"
    POLICIES_PATH = POLICIES_DIR / "company_policies.json"
    TEST_SUITE_PATH = TEST_DIR / "test_suite.json"
    MEMORY_PATH = DATA_DIR / "agent_memory.pkl"
    TRAINING_DATA_PATH = TRAINING_DIR / "epistemic_training.jsonl"
    
    # Evaluation Metrics
    METRICS = [
        "epistemic_consistency",
        "decision_stability", 
        "hallucination_score",
        "response_time",
        "token_efficiency",
        "value_alignment"
    ]
    
    # Agent Configuration
    MAX_MEMORY_ITEMS = 100
    TEMPERATURE = 0.3
    MAX_TOKENS = 500
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = LOGS_DIR / "agent.log"

config = Config()