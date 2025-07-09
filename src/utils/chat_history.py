import redis 
from redis.sentinel import Sentinel
import json 
import uuid 
from datetime import datetime 

DEFAULT_USER_ID = "default_user_neo"

class ChatHistory:
    def __init__(self, sentinel_host, sentinel_port, service_name, password, db=0):
        self.sentinel = Sentinel(
            [(sentinel_host, sentinel_port)],
            password=password,
        )
        
        self.redis_client = self.sentinel.master_for(
            service_name,
            password=password,
            db=db
        )

    def save_chat(self, query: str, response: str, context: dict = None, user_id: str = None):
        try:
            user_id = user_id or DEFAULT_USER_ID
            
            # Tạo message_id
            message_id = str(uuid.uuid4())
            
            # Chuẩn bị dữ liệu
            chat_data = {
                "message_id": message_id,
                "query": query,
                "response": response,
                "context": context if context else {},
            }
            
            encoded_data = json.dumps(chat_data, ensure_ascii=False).encode('utf-8')            
            self.redis_client.hset(f"chat:user:{user_id}", message_id, encoded_data)            
            self.redis_client.lpush(f"chat:history:{user_id}", message_id)
            return message_id
            
        except Exception as e:
            print(f"Lỗi khi lưu chat: {str(e)}")
            return None

    def get_chat_history(self, user_id: str = None, limit: int = 5) -> list:
        try:
            user_id = user_id or DEFAULT_USER_ID
            
            msg_ids = self.redis_client.lrange(f'chat:history:{user_id}', 0, limit - 1)
            sessions = []
            for msg_id in msg_ids:
                if isinstance(msg_id, bytes):
                    msg_id = msg_id.decode('utf-8')
                    
                session_raw = self.redis_client.hget(f'chat:user:{user_id}', msg_id)
                if session_raw:
                    session = json.loads(session_raw.decode('utf-8'))
                    sessions.append(session)
            return sessions
        except Exception as e:
            print(f"Lỗi khi lấy lịch sử chat: {str(e)}")
            return []
            

    
