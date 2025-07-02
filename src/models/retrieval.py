from langchain_community.llms.ollama import Ollama
import sys
import os
import time
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.request.query import Query

class LLMManager:
    def __init__(self):
        """Khởi tạo LLM Manager"""
        self.llm = None
        self.query_processor = Query()
        
    def create_llm(self):
        """Tạo kết nối LLM"""
        print('Đang kết nối Mistral...')
        try:
            self.llm = Ollama(
                model="mistral:7b",  
                temperature=0.1,     
                top_k=10,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=4096,  
                stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:']
            )
            print('Kết nối Ollama thành công')
            return self.llm
        except Exception as e:
            print(f'Lỗi kết nối Ollama: {str(e)}')
            raise e

    def get_llm(self):
        """Lấy LLM instance (tạo nếu chưa có)"""
        if self.llm is None:
            self.llm = self.create_llm()
        return self.llm

    def query_response(self, query):
        """Xử lý câu hỏi và trả về response"""
        try:
            current_llm = self.get_llm()
            
            # Sử dụng Query class để retrieve context
            context, scores, retrieval_time, total_tokens = self.query_processor.retrieve(query)

            MIN_TOKENS_THRESHOLD = 150
            if total_tokens <= MIN_TOKENS_THRESHOLD:
                # Không đủ context 
                return "Tôi chỉ có thể hỗ trợ trả lời các thông tin liên quan đến pháp luật, vui lòng cung cấp câu hỏi cho tôi. "
            
            else: 
                # In context và scores
                print(f"\n=== RETRIEVED CONTEXTS (Total tokens: {total_tokens}) ===")
                for i, (c, score) in enumerate(zip(context, scores), 1):
                    if isinstance(c, dict):
                        c_text = c['answer']
                    else:
                        c_text = c
                    print(f"{i}. Score: {score:.4f}")
                    print(f"   Context: {c_text.strip()}")
                    # print("-" * 80)
                
                # Sử dụng context được retrieve
                prompt = f'''Bạn là một luật sư chuyên nghiệp người Việt Nam. 
Hãy trả lời câu hỏi dựa trên các nội dung pháp luật được cung cấp.
YÊU CẦU:
1. LUÔN trả lời bằng tiếng Việt
2. Trả lời đầy đủ, chi tiết, cụ thể, dễ hiểu 
3. Chỉ sử dụng thông tin từ các nội dung được cung cấp trong tài liệu truy xuất được 
4. Nếu không có đủ thông tin để trả lời, hãy nói "Tôi không có đủ thông tin để trả lời câu hỏi này"

Nội dung pháp luật được trích xuất:
'''
                
                for i, c in enumerate(context, 1):
                    if isinstance(c, dict):
                        c = c['answer']
                    prompt += f'{i}. {c.strip()}\n'  #prompt + context 
                    
                prompt += '\n------------\n'
                prompt += f'Câu hỏi: {query.strip()}\n'
                prompt += 'Trả lời bằng tiếng Việt: '
                
            response = current_llm.invoke(prompt)
            return response

        except Exception as e:
            print(f"Lỗi trong query_response: {str(e)}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

            