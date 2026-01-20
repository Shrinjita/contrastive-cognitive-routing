# core/rlm_memory.py
import json
import os

class RecursiveMemory:
    def __init__(self, memory_path):
        self.memory_path = memory_path
        self.entries = []
        self._load_disk()
    
    def store(self, entry):
        self.entries.append(entry)
        self._write_disk()
    
    def recall(self):
        return '\n'.join(self.entries)
    
    def compress(self, llm):
        if not self._should_compress():
            return
        
        current_memory = self.recall()
        prompt = f"""Compress the following memory into essential epistemic facts:

{current_memory}

Preserve only: beliefs, uncertainties, key decisions, contradictions."""
        
        compressed = llm.generate(prompt)
        self.entries = [compressed]
        self._write_disk()
    
    def _should_compress(self):
        return len(self.entries) >= 10
    
    def _write_disk(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, 'w') as f:
            json.dump(self.entries, f)
    
    def _load_disk(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                self.entries = json.load(f)