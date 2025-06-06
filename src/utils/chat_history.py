import redis 
import json 
import os
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, db=0)

# query cache 
def query_cache(query, result, seconds = 3600):
    key = f'query:{query}'
    value = json.dumps(result)
    r.set(key, value, ex = seconds)

def get_cache(query):
    key = f'query:{query}'
    cached = r.get(key)
    if cached: 
        return json.loads(cached)
    return None

## chat history 
def message_history(session_id, role, content):
    key = f'chat_history:{session_id}'
    message = json.dumps({
        'role': role,
        'content': content,
    })
    r.rpush(key, message)
    r.expire(key, 3600)

def get_history(session_id): 
    key = f'chat_history:{session_id}'
    messages = r.lrange(key, 0, -1)
    return [json.loads(m) for m in messages]

def delete_history(session_id):
    key = f'chat_history:{session_id}'
    r.delete(key)

### session management 
def create_session(session_id, data, expire_seconds=1800):
    key = f"session:{session_id}"
    r.hmset(key, data)
    r.expire(key, expire_seconds)

def get_session(session_id):
    key = f"session:{session_id}"
    session_data = r.hgetall(key)
    return session_data if session_data else None

def delete_session(session_id):
    key = f"session:{session_id}"
    r.delete(key)












































# import redis
# import json
# from datetime import datetime

# r = redis.Redis(host='localhost', port=6379, db=0)

# def save_chat_history(session_id, question, answer):
#     key = f"chat_history:{session_id}"
#     entry = {
#         "timestamp": datetime.now().isoformat(),
#         "question": question,
#         "answer": answer
#     }
#     r.rpush(key, json.dumps(entry))
#     # Lưu session_id vào danh sách các session
#     r.sadd("all_sessions", session_id)

# def get_chat_history(session_id, limit=20):
#     key = f"chat_history:{session_id}"
#     history = r.lrange(key, -limit, -1)
#     return [json.loads(item) for item in history]

# def delete_chat_history(session_id):
#     key = f"chat_history:{session_id}"
#     r.delete(key)
#     r.srem("all_sessions", session_id)

# def get_all_sessions():
#     """Trả về list các session đã sắp xếp theo thời gian tạo"""
#     try:
#         sessions = list(r.smembers("all_sessions"))
#         # Sắp xếp sessions theo thời gian tạo
#         session_info = []
#         for s in sessions:
#             try:
#                 sid = s.decode() if isinstance(s, bytes) else s
#                 info = get_session_info(sid)
#                 if info:
#                     session_info.append((sid, info))
#             except Exception:
#                 continue
#         # Sắp xếp theo thời gian tạo mới nhất
#         session_info.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
#         return [s[0] for s in session_info]
#     except Exception as e:
#         print(f"Error getting all sessions: {str(e)}")
#         return []

# def save_session_info(session_id, info):
#     """Lưu thông tin của session (title, created_at, etc)"""
#     key = f"session_info:{session_id}"
#     info["created_at"] = datetime.now().isoformat()
#     r.hmset(key, info)

# def get_session_info(session_id):
#     """Lấy thông tin của session"""
#     try:
#         key = f"session_info:{session_id}"
#         info = r.hgetall(key)
#         if not info:
#             return None
#         return {k.decode() if isinstance(k, bytes) else k: 
#                 v.decode() if isinstance(v, bytes) else v 
#                 for k, v in info.items()}
#     except Exception as e:
#         print(f"Error getting session info for {session_id}: {str(e)}")
#         return None