from langchain_community.llms.ollama import Ollama
import sys
import os
import time
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.chat_history import message_history, get_history
from src.models.function_calling import process_query
from src.retrieval.query import retrieve

print('Đang kết nối Mistral-7B...')
try:
    llm = Ollama(
        model="mistral:7b",  
        temperature=0.1,     # Giảm temperature để output nhất quán hơn
        top_k=10,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=4096,  
        stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:']
    )
    print('Kết nối Ollama thành công')
except Exception as e:
    print(f'Lỗi kết nối Ollama: {str(e)}')
    sys.exit(1)

### -------------------prompt template -------------------
def prompt_template(query, context, n_context=10):
    prompt = """Bạn là một luật sư chuyên nghiệp người Việt Nam. Hãy trả lời câu hỏi dựa trên các nội dung pháp luật được cung cấp.
    YÊU CẦU:
    1. LUÔN trả lời bằng tiếng Việt
    2. Trả lời đầy đủ, chi tiết, cụ thể, dễ hiểu 
    3. Chỉ sử dụng thông tin từ các nội dung được cung cấp
    4. Nếu không có đủ thông tin để trả lời, hãy nói "Tôi không có đủ thông tin để trả lời câu hỏi này"
    """
    
    prompt += "\nNội dung pháp luật được trích xuất:\n"
    
    for i, c in enumerate(context[:n_context], 1): 
        if isinstance(c, dict):
            c = c['answer']
        prompt += f'{i}. {c.strip()}\n'
    prompt += '\n------------\n'
    prompt += f'Câu hỏi: {query.strip()}\n'
    prompt += 'Trả lời bằng tiếng Việt: '
    return prompt

### -------------------llm -------------------
def main():
    query = input('Nhập câu hỏi: ')
    try:
        print("\nĐang phân tích câu hỏi...")
        # Tạo một session_id tạm thời cho CLI
        temp_session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        function_result = process_query(query, temp_session_id)
        
        if function_result:
            print('\nAnswer:')
            print(function_result)
            return
                
    except Exception as e:
        print(f"Lỗi khi tra cứu thông tin: {str(e)}")
    
    print("\nĐang tìm kiếm thông tin liên quan...")
    retrieval_start = time.time()
    context, scores, retrieval_time, total_tokens = retrieve(query)
    retrieval_end = time.time()

    print("\nĐiểm số các đoạn văn bản:")
    for i, (c, s) in enumerate(zip(context[:10], scores[:10]), 1):
        if isinstance(c, dict):
            c = c['answer']
        print(f"Đoạn {i}: {s:.4f}")

    prompt = prompt_template(query, context)
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Đang tạo câu trả lời từ Mistral-7B...")
        
        # time response
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_end = time.time()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
        print(f"Time response: {(llm_end - llm_start):.2f} giây")
        print('\nAnswer:')
        print(response)
        
        total_time = llm_end - retrieval_start
        print(f"\nTổng thời gian xử lý: {total_time:.2f} giây")
        
    except Exception as e:
        print(f'Lỗi khi tạo câu trả lời: {e}')

if __name__ == '__main__':
    main()