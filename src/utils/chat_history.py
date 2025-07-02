import redis 
from redis.sentinel import Sentinel
import json 
import uuid 
from datetime import datetime 
import tiktoken

class ChatHistory:
    def __init__(self, sentinel_host, sentinel_port, service_name, password, db = 0):
        self.sentinel_host = '10.252.10.248'
        self.sentinel_port = 26379
        self.service_name = 'mymaster' 
        self.password = 'mymasanpass#12x'
        self.db = 0
        
        # Khởi tạo Sentinel
        self.sentinel = Sentinel(
            [(self.sentinel_host, self.sentinel_port)],
            password=self.password,
        )
        
        self.redis_client = self.sentinel.master_for(
            self.service_name,
            password=self.password,
            db=self.db,
        )
        
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')
    
    def count_tokens(self, text): 
        return len(self.tokenizer.encode(text))
    
    def truncate_tokens(self, text, max_tokens):
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens)
        return text 
        
    def save_chat(self, user_id, query, response, context = None, max_tokens = 500): 
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        chat_data = {
            "query": query,
            "response": response,
            "context": context if context else [], 
        }
        
        json_data = json.dumps(chat_data, ensure_ascii=False)
        num_tokens = self.count_tokens(json_data)
        if num_tokens > max_tokens: 
            if isinstance(chat_data['context'], list) and chat_data['context']:
                chat_data['context'] = chat_data['context'][:-1]
            elif len(chat_data['response']) > 100:
                chat_data['response'] = self.truncate_tokens(chat_data['response'], len(self.tokenizer.encode(chat_data['response'])) - 50 )
            elif len(chat_data['query']) > 100:
                chat_data['query'] = self.truncate_tokens(chat_data['query'], len(self.tokenizer.encode(chat_data['query'])) - 50 )
        
        encoded_data = json.dumps(chat_data, ensure_ascii=False).encode('utf-8')
        self.redis_client.hset(f"user_id:{user_id}", message_id, encoded_data)
        self.redis_client.lpush(f'chat_list:{user_id}', message_id)
        
        return message_id
    
    def get_chat_history(self, user_id, limit = 10):
        self.redis_client.hset(f"user_id:{user_id}", message_id, encoded_data)
        self.redis_client.lpush(f'chat_list:{user_id}', message_id)
        
        return message_id
    
    def get_chat_history(self, user_id, limit = 10):
        message_ids = self.redis_client.lrange(f'chat_list:{user_id}', 0, limit - 1)
        chat_history = []
        
        for message_id in message_ids:
            if isinstance(message_id, bytes):
                message_id = message_id.decode('utf-8')
            chat_data = self.redis_client.hget(f"user_id:{user_id}", message_id)
            if chat_data:
                chat_history.append(json.loads(chat_data))
        
        chat_history.sort(key=lambda x: x['timestamp'], reverse=True)
                
        return chat_history
    
