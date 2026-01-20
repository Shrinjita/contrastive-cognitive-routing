# core/rag_agent.py
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGAgent:
    def __init__(self, llm):
        self.llm = llm
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
    
    def ingest_documents(self, path):
        documents = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        documents.append(f.read())
        
        for doc in documents:
            self.chunks.extend(self._chunk(doc))
        
        self._vectorize(self.chunks)
    
    def _chunk(self, text):
        lines = text.split('\n')
        chunks = []
        current = []
        
        for line in lines:
            current.append(line)
            if len(current) >= 5:
                chunks.append('\n'.join(current))
                current = []
        
        if current:
            chunks.append('\n'.join(current))
        
        return chunks
    
    def _vectorize(self, chunks):
        self.vectors = self.vectorizer.fit_transform(chunks)
    
    def _retrieve(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def decide(self, query):
        context_chunks = self._retrieve(query)
        context = '\n\n'.join(context_chunks)
        
        prompt = f"""Context:
{context}

Query: {query}

Provide a decision based on the context above."""
        
        return self.llm.generate(prompt)