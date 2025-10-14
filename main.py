from typing import Optional, Dict, Any, List
import json
import re
from langchain_ollama import OllamaLLM
import os
import sys
import time
import threading
from tools import CompanyTools
from config_api import ConfigAPI
from history_chat import LockHistoryAPI
import requests
from datetime import datetime, timedelta
import uuid
from dotenv import load_dotenv
import logging
import asyncio
from embedding_json import EmbeddingJSON

# Load environment variables
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FunctionCalling:
    def __init__(self, tools_config_path: str = None):
        # Load tools configuration
        if tools_config_path is None:
            tools_config_path = os.getenv("CONFIG_FALLBACK_FILE")
        self.tools = json.load(open(tools_config_path, "r", encoding="utf-8"))
        # Lịch sử chat
        self.history_chat = LockHistoryAPI()
        self.toolbox = CompanyTools()
        self.embedding_matcher = EmbeddingJSON()
        # Weaviate configuration
        self.weaviate_url = os.getenv("URL_RAG")
        self.weaviate_collection_id = "78cb6809-a75f-43e6-9a7e-63cfbc24090d"
        #Graphrag configuration
        self.graphrag_url = os.getenv("URL_GRAPH_RAG")

        # Wait state timeout (30 seconds)
        self.wait_timeout_seconds = 30

        self.threshold = float(os.getenv("EMBEDDING_THRESHOLD"))
        # Load env variables
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            format="json", 
            temperature=float(os.getenv("OLLAMA_TEMPERATURE")), 
            base_url = os.getenv("BASE_URL"),
            top_k=int(os.getenv("OLLAMA_TOP_K")), 
            top_p=float(os.getenv("OLLAMA_TOP_P")), 
            repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY")), 
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX")), 
            stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:', '```'],
            seed=int(os.getenv("OLLAMA_SEED")),
        ) 
    
    def _events_to_message_for_llm(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuyển đổi danh sách sự kiện thành tin nhắn cho LLM"""
        role_mapping = {'user': 'user', 'bot': 'assistant'}
        messages = []
        for event in events:
            if not isinstance(event, dict):
                continue
            role = role_mapping.get(event.get('event'))
            text = event.get('text')
            if not role or not text:
                continue
            if role == 'assistant' and len(text) > 800:
                text = text[:800] + '...'
            messages.append({'role': role, 'content': text})
        return messages

    def _get_history_message(self, user_id: str) -> List[Dict[str, Any]]:
        """Lấy tin nhắn từ lịch sử"""
        events = self.history_chat.get_events(user_id)
        return self._events_to_message_for_llm(events)
    
    def _get_history_for_prompt(self, user_id: str) -> str:
        """Định dạng lịch sử thành một chuỗi hội thoại tự nhiên."""
        messages = self._get_history_message(user_id) # Dùng lại hàm cũ để lấy messages
        if not messages:
            return ""
        # Chuyển đổi list of dicts thành chuỗi chat
        history_str = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            history_str += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n"
        return history_str

    def execute_function(self, func_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        func_to_call = getattr(self.toolbox, func_name, None)
        if not func_to_call or not callable(func_to_call):
            return f'Không tìm thấy hàm {func_name} trong toolbox'
        try:
            result = func_to_call(**arguments)
            return result
        except Exception as e:
            return f'Lỗi khi thực thi hàm {func_name}: {str(e)}'

    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            try:
                return json.loads(response)
            except:
                pass
            start = response.rfind('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and start < end:
                try:
                    json_str = response[start:end+1]
                    return json.loads(json_str)
                except:
                    pass
            logger.warning("Không tìm thấy JSON hợp lệ trong response: %s", response)
            return None
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất JSON: {str(e)}")
            return None

    def get_user_friendly_name(self, func_name: str, param_name: str) -> str:
        """Chuyển đổi tên parameter sang user-friendly"""
        for tool in self.tools:
            if tool.get("name") == func_name:
                properties = tool.get("parameters", {}).get("properties", {})
                if param_name in properties:
                    description = properties[param_name].get("description")
                    if description:
                        return description 
                break
        return param_name

    def query_weaviate_database(self, query: str) -> Optional[str]:
        """
        Query Weaviate database để lấy context chunks
        Args:
            query: Câu hỏi từ user
        Returns:
            Context chunks từ database hoặc None nếu có lỗi
        """
        try:
            payload = {
                "query": query,
                "collection_id": self.weaviate_collection_id
            }
            logger.info(f"WEAVIATE REQUEST: Calling {self.weaviate_url} with payload: {payload}")
            response = requests.post(
                self.weaviate_url,
                json=payload,
                timeout=90,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"WEAVIATE RESPONSE: Status {response.status_code}")
            # Lấy context chunks từ response
            context = result.get('context', result.get('chunks', result.get('documents', '')))
            return context if context else result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout khi gọi Weaviate API: {self.weaviate_url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gọi Weaviate API: {str(e)}")
            return None
        except json.JSONDecodeError:
            logger.error("Lỗi decode JSON từ Weaviate API")
            return None
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi Weaviate: {str(e)}")
            return None

    def extract_chunk_rag(self, weaviate_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Trích xuất danh sách {chunk_index, document_id} từ JSON trả về, sau đó thực hiện truy vấn trong graphRAG
        """
        try:
            # Lấy danh sách context từ Weaviate
            contexts = []
            if isinstance(weaviate_result, dict):
                contexts = weaviate_result.get("contexts") or []
            else:
                logger.warning("Không tìm thấy contexts trong Weaviate result")
                return []

            # Tạo danh sách chunks từ contexts để gửi sang GraphRAG
            chunks_payload = []
            for context_item in contexts:
                chunks_payload.append({
                    "chunk_index": context_item.get("chunk_index"),
                    "document_id": context_item.get("document_id")
                })

            if not chunks_payload:
                logger.warning("Không tìm thấy chunks trong Weaviate result")
                return []

            graphrag_payload = {"chunks": chunks_payload}

            # Log thông tin gửi đến GraphRAG
            logger.info(f"GRAPHRAG REQUEST: url={self.graphrag_url}, payload={graphrag_payload}")
            graphrag_response = requests.post(
                self.graphrag_url,
                json=graphrag_payload,
                timeout=90, 
                headers={"Content-Type": "application/json"}
            )

            graphrag_response.raise_for_status()

            graphrag_result = graphrag_response.json()
            logger.info(f"GRAPHRAG RESPONSE STATUS: {graphrag_response.status_code}")

            return graphrag_result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout khi gọi GraphRAG API: {self.graphrag_url}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gọi GraphRAG API: {str(e)}")
            return []
        except json.JSONDecodeError:
            logger.error("Lỗi decode JSON từ GraphRAG API")
            return []
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi GraphRAG: {str(e)}")
            return []

    def process_rag_query(self, query: str, combined_context: str) -> str:
        """
        Xử lý query với context từ RAG database sử dụng prompt template chuyên về điện lực
        Args:
            query: Câu hỏi từ user
            context: Context chunks từ database
        Returns:
            Response từ LLM
        """
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in the Vietnamese electricity sector. You are only allowed to answer questions related to the following topics:
- Electricity systems and networks
- Electrical devices and electrical safety
- Electricity policies and regulations
- Electricity bills and pricing
- Customer services in the electricity industry
- Technical issues related to electricity

INFORMATION SOURCES:
You will receive information from TWO different knowledge sources:

1. **WEAVIATE (Vector Search/RAG)**: 
   - Performs semantic search based on vector embeddings
   - Provides specific text chunks from documents directly relevant to the question
   - Contains detailed content, specific regulations, and factual information
   - Format: contexts with chunk_index, document_id, text, metadata

2. **GRAPHRAG (Graph Relationships)**:
   - Searches for relationships and connections through knowledge graphs
   - Provides related chunks that may not be directly relevant but have contextual relationships
   - Helps understand broader context and connections between concepts
   - Format: related_chunks with chunk_index, document_id, text

STRICT RULES:
1. For questions NOT related to the electricity field, you MUST respond with:
"Tôi là trợ lý chuyên về điện lực. Rất tiếc, tôi không thể trả lời câu hỏi này vì nó không liên quan đến lĩnh vực điện. Xin vui lòng hỏi về các vấn đề điện lực."

2. For electricity-related questions:
- **PRIMARY SOURCE: Use Weaviate (RAG) as the main source** - it provides the most accurate and directly relevant information
- **SECONDARY SOURCE: Use GraphRAG only as supplementary context** when Weaviate information needs additional background or relationships
- **In case of conflicts: Always prioritize Weaviate information** over GraphRAG due to higher accuracy
- Base your answer primarily on Weaviate content, use GraphRAG only to enhance understanding
- Clearly cite information from the sources, emphasizing Weaviate findings
- If Weaviate provides sufficient information, GraphRAG content can be omitted

RESPONSE FORMAT:
- Start with a direct answer to the question
- Support with evidence from the sources
- Use clear and understandable Vietnamese language
- Structure information logically

IMPORTANT NOTES:
- Always respond in Vietnamese
- Do not answer questions outside the electricity field
- **Focus primarily on Weaviate (RAG) information** as it has higher accuracy and relevance
- Use GraphRAG only when additional context or relationships are needed to enhance the Weaviate-based answer
- When Weaviate provides complete information, you may rely solely on it without needing GraphRAG content
<|eot|>
<|start_header_id|>user<|end_header_id|>

Reference Information from Knowledge Sources:
{combined_context}

Question: {query}
<|eot|><|start_header_id|>assistant<|end_header_id|>
Answer: """

        try:
            # Tạo LLM instance cho RAG (không dùng format="json")
            rag_llm = OllamaLLM(
                model=os.getenv("OLLAMA_MODEL"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE")), 
                base_url=os.getenv("BASE_URL"),
                top_k=int(os.getenv("OLLAMA_TOP_K")), 
                top_p=float(os.getenv("OLLAMA_TOP_P")), 
                repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY")), 
                num_ctx=int(os.getenv("OLLAMA_NUM_CTX")), 
                stop=['<|eot|>', 'Question:', 'Câu hỏi:', 'Human:', 'User:'],
            )
            response = rag_llm.invoke(rag_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Lỗi khi xử lý RAG query: {str(e)}")
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."

    def handle_rag_workflow(self, query: str) -> str:
        """
        Xử lý workflow RAG hoàn chỉnh: Query DB → Get Context → Generate Response
        Args:
            query: Câu hỏi từ user
        Returns:
            Final response từ RAG system
        """
        logger.info(f"RAG WORKFLOW: Starting for query: {query}")

        # Step 1: Query Weaviate database để lấy RAG context chunks
        try:
            # Gọi trực tiếp Weaviate API từ function.py
            chunk_context = self.query_weaviate_database(query)
            logger.info(f"RAG context chunks: {chunk_context}")
            print(f"RAG context chunks: {chunk_context}")
            if not chunk_context:
                logger.info("RAG WORKFLOW: No context found from database")
                return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            # Step 2: Process RAG chunks với Ollama sử dụng prompt template điện lực
            response_rag = self.process_rag_query(query, str(chunk_context))
            return response_rag
        except Exception as e:
            logger.error(f"RAG WORKFLOW ERROR: {str(e)}")
            return "Đã có lỗi xảy ra khi xử lý câu hỏi của bạn."

    def create_prompt(self, query: str, user_id: str, matched_functions: List[str] = None) -> str:
        # Lọc tools chỉ giữ lại functions được match có 
        if matched_functions and matched_functions != ["none_function"]:
            filtered_tools = [tool for tool in self.tools if tool.get("name") in matched_functions]
            tools_json_str = json.dumps(filtered_tools, ensure_ascii=False)
        else:
            tools_json_str = json.dumps(self.tools, ensure_ascii=False)
        history_json = self._get_history_for_prompt(user_id)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that analyzes user questions to determine which function to call. CRITICAL: ONLY return properly formatted JSON.

STRICT RULES:
1. ONLY return JSON with format: {{"function": "function_name", "arguments": {{...}}, "missing_info": [...]}}
2. ONLY USE functions that are in the AVAILABLE FUNCTIONS list below
3. ABSOLUTELY DO NOT invent new function names or parameter names
4. USE ONLY the exact parameter names defined in each function
5. If sufficient information: fill arguments completely, missing_info = []
6. If insufficient information: arguments have null for missing parameters, missing_info list parameter names
7. If not related or no suitable function: {{"function": "Not_call_function_calling", "arguments": {{}}, "missing_info": []}}
8. ABSOLUTELY return only valid JSON format, nothing else

VÍ DỤ CHÍNH XÁC:
User: "địa chỉ công ty"
→ {{"function": "dia_chi_cong_ty_NEO", "arguments": {{}}, "missing_info": []}}

User: "nhân sự công ty"
→ {{"function": "nhan_su_cong_ty_NEO", "arguments": {{"phong_ban": null}}, "missing_info": ["phong_ban"]}}

User: "tính lương"
→ {{"function": "uoc_tinh_luong", "arguments": {{"so_nam_kinh_nghiem": null, "vi_tri_cong_viec": null}}, "missing_info": ["so_nam_kinh_nghiem", "vi_tri_cong_viec"]}}

User: "electricity bill information"
→ {{"function": "Not_call_function_calling", "arguments": {{}}, "missing_info": []}}

NOTE: If the question doesn't match any function in the list, return "Not_call_function_calling"
<|end_of_text|><|start_header_id|>user<|end_header_id|>
CHAT HISTORY: {history_json}

AVAILABLE FUNCTIONS (ONLY USE THESE):
{tools_json_str}

USER QUESTION: {query}
<|end_of_text|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    def _save_response(self, user_id: str, query: str, intent_name: str, utter_action: str, response_text: str, session_status: str) -> str:
        """Tạo và lưu user_events, bot_events"""
        user_events = {
            "event": 'user',
            "metadata": {"channel": None},
            "parsed_data": {'intent': {'name': intent_name, 'confidence': 1.0}},
            "text": query, 
            "timestamp": datetime.now().isoformat()
        }

        bot_events = {
            "event": 'bot', 
            "metadata": {"utter_action": utter_action}, 
            "text": response_text, 
            "timestamp": datetime.now().isoformat()
        }
        current_turn = [user_events, bot_events]
        self.history_chat.add(user_id, current_turn, session_status)
        return response_text

    def _combine_rag_graphrag(self, query: str) -> str:
        """
        Kết hợp context từ rag và graphRAG
        """
        weaviate_result = self.query_weaviate_database(query)
        if weaviate_result:
            try:
                # Parse và query GraphRAG để lấy thêm context
                weaviate_data = json.loads(weaviate_result) if isinstance(weaviate_result, str) else weaviate_result
                graphrag_result = self.extract_chunk_rag(weaviate_data)
                # Kết hợp context từ cả hai nguồn
                combined_context = f"Weaviate: {weaviate_result}\nGraphRAG: {graphrag_result}"
                logger.info(f"COMBINED CONTEXT: {combined_context}")
                return self.process_rag_query(query, combined_context)
            except Exception as e:
                logger.warning(f"GraphRAG error, using Weaviate only: {e}")
                return self.process_rag_query(query, str(weaviate_result))
        else:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    def process_query(self, query: str, user_id: str) -> Optional[str]:
        lock = None
        try: 
            lock = self.history_chat.create_lock(user_id)
            if not lock:
                return {"state": "error", "message": "Hệ thống đang bận, vui lòng thử lại sau"}
            # Nếu đang bổ sung tham số, bỏ qua embedding
            ctx = json.loads(self.history_chat.redis.get(f"CONTEXT_{user_id}") or '{}')
            skip_embed = ctx.get("session_status") == "function_calling"
            if skip_embed:
                # Đang trong phiên function calling - recover function từ lịch sử để đảm bảo có đúng function
                events = self.history_chat.get_events(user_id)
                last_function = None
                for event in reversed(events[-5:]):
                    func_name = event.get('parsed_data', {}).get('intent', {}).get('name')
                    if func_name and func_name != "Not_call_function_calling":
                        last_function = func_name
                        break
                # Nếu không tìm được function từ lịch sử, revert về embedding cho query hiện tại
                if last_function:
                    matched_functions = [last_function]
                else:
                    # Fallback: dùng embedding để tìm function phù hợp với query bổ sung
                    matched_functions = self.embedding_matcher._find_best_function(query, self.threshold)
            else:
                # Thực hiện embedding để tìm function phù hợp
                matched_functions = self.embedding_matcher._find_best_function(query, self.threshold)
            logger.info(f"EMBEDDING MATCH: {matched_functions}")
            # Nếu không có function nào phù hợp VÀ KHÔNG đang trong function calling, chuyển sang RAG
            if (matched_functions is None or matched_functions == ["none_function"]) and not skip_embed:
                final_response = self._combine_rag_graphrag(query)
                return self._save_response(user_id, query, "Not_call_function_calling", "Retrieval_DB", final_response, "completed")
            # Có functions phù hợp, sử dụng prompt với các function liên quan để function calling
            prompt = self.create_prompt(query, user_id, matched_functions)
            response = self.llm.invoke(prompt)
            parsed = self.extract_json_from_response(response)
            if not parsed:
                return {"state": "error", "message": "Lỗi: Không thể phân tích phản hồi từ LLM. Vui lòng thử lại."}
            func_name = parsed.get("function")
            arguments = parsed.get("arguments", {})
            missing_params = parsed.get("missing_info", [])

            if func_name == "Not_call_function_calling":
                final_response = self._combine_rag_graphrag(query)
                return self._save_response(user_id, query, func_name, "Retrieval_DB", final_response, "completed")
            else: 
                for param_name, param_value in arguments.items():
                    if param_value is None and param_name not in missing_params:
                        missing_params.append(param_name)
                if not missing_params:
                    bot_event_text = self.execute_function(func_name, arguments)
                    session_status_value = "completed"
                else:
                    context_friendly = [self.get_user_friendly_name(func_name, param) for param in missing_params]
                    if len(context_friendly) == 1:
                        bot_event_text = f"Tôi cần cung cấp thông tin thêm về {context_friendly[0]}. Bạn có thể cho tôi biết được không."
                    else:
                        missing_str = ", ".join(context_friendly[:-1]) + f" và {context_friendly[-1]}"
                        bot_event_text = f"Bạn có thể cho tôi biết thêm {missing_str} được không?"
                    session_status_value = "function_calling"
                return self._save_response(user_id, query, func_name, "Function_Calling", bot_event_text, session_status_value) 
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return {"state": "error", "message": "Lỗi khi xử lý câu hỏi. Vui lòng thử lại."}
        finally:
            if lock:
                lock.release()

def main():
    function_calling = FunctionCalling()
    session_id = str(uuid.uuid4())
    print(f"Bắt đầu phiên chat với Session ID: {session_id}")
    while True:
        query = input("Nhập câu hỏi: ")
        if query.lower() in ['exit', 'quit', 'thoát']:
            print("Kết thúc phiên chat!")
            break

        result = function_calling.process_query(query, session_id)
        print(result)

if __name__ == "__main__":
    main()
