# core/utils.py
import os
import hashlib
import json
from datetime import datetime

def load_documents(path):
    documents = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.md'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
    return documents

def clean_text(text):
    return ' '.join(text.split())

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def log_result(data):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().isoformat()
    filename = f"logs/result_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)