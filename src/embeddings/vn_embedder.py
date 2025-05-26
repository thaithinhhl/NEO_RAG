import json 
import os
import numpy as np 
import torch 
import faiss 
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

'''Vietnamese Embedding '''
tokenizer = AutoTokenizer.from_pretrained('truro7/vn-law-embedding')
model = AutoModel.from_pretrained('truro7/vn-law-embedding')

'''concat'''
def get_embedding(chunk):
    parts = []
    if chunk.get('chuong'):
        parts.append(chunk.get('chuong'))
    if chunk.get('muc'):
        parts.append(chunk.get('muc'))  
    if chunk.get('dieu'):
        parts.append(chunk.get('dieu'))
    if chunk.get('noidung'):
        parts.append(chunk.get('noidung'))
    return ' '.join([p for p in parts if p]).strip()

'''embedding'''
def vietnamese_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=500)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def save_embedding(file_path, file_index):
    os.makedirs(os.path.dirname(file_index), exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    all_embeddings = []
    for chunk in tqdm(chunks, desc="Creating embeddings"):
        text = get_embedding(chunk)
        if text:
            embedding = vietnamese_embedding(text)
            all_embeddings.append(embedding)
    
    embeddings_array = np.array(all_embeddings)
    print(f"Tạo {len(all_embeddings)} embeddings")
    
    dimension = embeddings_array.shape[1]
    # print(f"Dimension của embedding: {dimension}")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, file_index)

if __name__ == '__main__':
    input_file = 'data/Chunk.json'
    file_index = 'src/database/faiss.index'
    save_embedding(input_file, file_index)  
