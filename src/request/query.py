import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Internal imports
# from src.utils.chat_history import query_cache, get_cache  # Commented out as these functions don't exist

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import json
import time
from sentence_transformers import CrossEncoder

class Query:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("truro7/vn-law-embedding")
        self.model = AutoModel.from_pretrained("truro7/vn-law-embedding")
        self.model.to(self.device)
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def get_vietnamese_embedding(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
        
    def retrieve(self, query, top_k=5 , index_file='src/database/faiss.index', output_file='data/retrieval.json'):
        start_time = time.time()
        
        index = faiss.read_index(index_file)
        with open("data/Chunk.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        query_embedding = self.get_vietnamese_embedding(query).reshape(1, -1)
        similarities, indices = index.search(query_embedding, top_k)        ### lay top k 
    
        retrieved_chunks = [metadata[idx] for idx in indices[0]]
    
        # [query, chunk]
        query_chunk = [[query, f"{chunk.get('muc', '')} {chunk.get('dieu', '')} {chunk['noidung']}"] for chunk in retrieved_chunks]    # muc + dieu + muc + noi dung + querr -> re-rerank 
    
        # score  --------------------------------------- sum = weight * score 
        scores = self.rerank_model.predict(query_chunk)
        scores = 1/ ( 1+ np.exp(-scores))
        # sort
        sorted_indices = np.argsort(scores)[::-1]  # Giảm dần
        sorted_chunks = [retrieved_chunks[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        results = []
        total_tokens = 0
        
        for chunk, score in zip(sorted_chunks, sorted_scores):
            answer = f"Theo {chunk.get('chuong', '')} {chunk.get('muc', '')} {chunk.get('dieu', '')}, {chunk.get('noidung', '')}"
            results.append({"answer": answer, "score": float(score)})
            total_tokens += len(self.tokenizer.encode(answer))
    
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        retrieval_time = time.time() - start_time
        return [result["answer"] for result in results], [result["score"] for result in results], retrieval_time, total_tokens

