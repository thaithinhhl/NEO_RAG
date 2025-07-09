import os
import sys
import uuid 
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.function_calling import FunctionCalling
from src.models.retrieval import LLMManager       

class RAG: 
    def __init__(self):
        self.function_calling = FunctionCalling() 
        self.retrieval = LLMManager() 

    def request_user(self, query: str, user_id: str = "default") -> Tuple[str, str]:
        if not query or not isinstance(query, str):
            return "Câu hỏi không hợp lệ", "error"
            
        try:
            # Xử lý function calling
            function_result = self.function_calling.process_query(query)
            if function_result:
                return function_result, "function_calling"

            # Xử lý retrieval
            try:
                print("Đang xử lý câu hỏi qua retrieval...")
                retrieval_result = self.retrieval.query_response(query)
                
                if not retrieval_result or not isinstance(retrieval_result, str):
                    print(f"Kết quả retrieval không hợp lệ: {retrieval_result}")
                    return "Xin lỗi, tôi không thể xử lý câu hỏi này lúc này", "error"

                print(f"Đã nhận kết quả retrieval: {retrieval_result[:100]}...")
                return retrieval_result, "retrieval_llm"
                
            except Exception as e:
                print(f"Lỗi trong quá trình retrieval: {str(e)}")
                return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi", "error"
            
        except Exception as e:
            error_msg = f"Lỗi hệ thống: {str(e)}"
            print(error_msg)
            return error_msg, "error"


_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance

def initialize_pipeline():
    rag = get_rag()
    
    def process_chain(query: str) -> Tuple[str, str]:
        response, source = rag.request_user(query)
        return response, source
    
    rag.process_chain = process_chain
    return rag

if __name__ == "__main__":    
    try:
        print("Đang khởi tạo hệ thống...")
        from interface.api import app
        app.run(host="0.0.0.0", port=8000, debug=False)
        
    except KeyboardInterrupt:
        print("\nĐã dừng server!")
    except Exception as e:
        print(f"Lỗi khởi động: {str(e)}")
        sys.exit(1)
