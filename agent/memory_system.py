import pickle
from collections import deque

class MemorySystem:
    def __init__(self):
        self.memory_file = "agent_memory.pkl"
        self.conversation_history = deque(maxlen=50)
        self.load_memory()
    
    def load_memory(self):
        try:
            with open(self.memory_file, 'rb') as f:
                data = pickle.load(f)
                self.conversation_history = data.get('conversation_history', deque(maxlen=50))
        except FileNotFoundError:
            self.save_memory()
    
    def save_memory(self):
        data = {'conversation_history': list(self.conversation_history)}
        with open(self.memory_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_conversation(self, query, response):
        self.conversation_history.append({
            'query': query,
            'response': response
        })
        self.save_memory()