import uuid
import gradio as gr
from datetime import datetime
import logging
import redis
import os
import sys
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Internal imports
from src.models.llm import prompt_template
from src.retrieval.query import retrieve
from src.models.function_calling import process_query
from langchain_community.llms.ollama import Ollama
from src.utils.chat_history import (
    message_history,
    get_history,
    delete_history,
    create_session,
    get_session,
    delete_session
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# LLM
print('Đang kết nối Mistral-7B...')
try:
    llm = Ollama(
        model="mistral:7b",  
        temperature=0.7,     
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

def qa_pipeline(query, session_id):
    try:
        logger.info(f"Processing query for session {session_id}: {query}")
        
        # Lưu câu hỏi của user trước
        message_history(session_id, "user", query)
        current_history = get_history(session_id)
        chat_history = [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
                       for msg in current_history]
        
        try:
            print("\nĐang phân tích câu hỏi...")
            function_result = process_query(query, session_id)
            
            if function_result is not None:
                # Lưu câu trả lời
                message_history(session_id, "assistant", function_result)
                
                if not get_session(session_id):
                    create_session(session_id, {
                        "title": query[:50] + "..." if len(query) > 50 else query,
                        "created_at": datetime.now().isoformat()
                    })
                
                # Lấy lịch sử mới nhất
                updated_history = get_history(session_id)
                return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
                        for msg in updated_history]
                    
        except Exception as e:
            logger.error(f"Lỗi khi xử lý function calling: {str(e)}")
        
        print("\nĐang tìm kiếm thông tin liên quan...")
        context, scores, retrieval_time, total_tokens = retrieve(query)
        
        # In ra context và scores để debug
        print("\nCác đoạn văn bản liên quan:")
        for i, (c, s) in enumerate(zip(context[:10], scores[:10]), 1):
            if isinstance(c, dict):
                content = c['answer']
            else:
                content = c
            print(f"{i}. {content.strip()}\n")

        # Quyết định dựa trên tổng số token từ retrieval
        MIN_TOKENS_THRESHOLD = 150
        if total_tokens >= MIN_TOKENS_THRESHOLD:
            print(f"\nSử dụng thông tin từ các đoạn văn bản trên để trả lời (tổng số tokens: {total_tokens})")
            prompt = prompt_template(query, context)
        else:
            print(f"\nSố token quá ít ({total_tokens}), sử dụng kiến thức có sẵn để trả lời")
            prompt = f'''Bạn là một luật sư chuyên nghiệp người Việt Nam. 

YÊU CẦU:
1. LUÔN trả lời bằng tiếng Việt
2. Trả lời đầy đủ, chi tiết, cụ thể, dễ hiểu
3. Nếu không chắc chắn về thông tin, hãy nói rõ điều đó
4. trả lời dựa trên tri thức của bạn

Câu hỏi: {query}

Trả lời bằng tiếng Việt:'''

        # Gọi LLM để trả lời
        answer = llm.invoke(prompt)
        
        # Lưu câu trả lời
        message_history(session_id, "assistant", answer)

        if not get_session(session_id):
            create_session(session_id, {
                "title": query[:50] + "..." if len(query) > 50 else query,
                "created_at": datetime.now().isoformat()
            })

        # Lấy lịch sử mới nhất
        final_history = get_history(session_id)
        return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
                for msg in final_history]
                
    except Exception as e:
        logger.error(f"Error in qa_pipeline: {str(e)}")
        error_message = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
        
        # Lưu thông báo lỗi
        message_history(session_id, "assistant", error_message)
        error_history = get_history(session_id)
        return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
                for msg in error_history]

def load_session(session_id):
    history = get_history(session_id)
    return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
            for msg in history], session_id

def get_session_titles():
    try:
        sessions = r.keys("session:*")
        titles = []
        for session_key in sessions:
            try:
                session_id = session_key.decode().split(":")[1]
                session_data = get_session(session_id)
                if session_data and b"title" in session_data and b"created_at" in session_data:
                    title = session_data[b"title"].decode()
                    created_at = datetime.fromisoformat(session_data[b"created_at"].decode())
                    formatted_time = created_at.strftime("%d/%m/%Y %H:%M")
                    display_title = f"{title} ({formatted_time})"
                    titles.append((session_id, display_title))
            except Exception as e:
                logger.error(f"Error processing session {session_id}: {str(e)}")
        return {title: sid for sid, title in titles} if titles else {}
    except Exception as e:
        logger.error(f"Error in get_session_titles: {str(e)}")
        return {}

# CSS styles remain unchanged
css = """
body {
    background-color: #fdfdfd;
    color: #222;
}
.message.user {
    text-align: right;
    background-color: #1976d2;
    color: white;
    border-radius: 15px 15px 0 15px;
    padding: 10px 15px;
    margin: 5px;
    max-width: 80%;
    margin-left: auto;
}
.message.bot {
    text-align: left;
    background-color: #f1f3f4;
    color: #000;
    border-radius: 15px 15px 15px 0;
    padding: 10px 15px;
    margin: 5px;
    max-width: 80%;
}
.chat-container {
    height: 600px;
    overflow-y: auto;
    padding: 20px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 0px 5px #ccc;
}
.legalchat-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #1976d2;
    margin-top: 10px;
    margin-bottom: 30px;
    font-family: 'Segoe UI', sans-serif;
}
"""

def update_session_list():
    try:
        titles = list(get_session_titles().keys())
        return gr.Radio(choices=titles, value=None)
    except Exception as e:
        logger.error(f"Error updating session list: {str(e)}")
        return gr.Radio(choices=[], value=None)

def on_submit(query, chat_history, session_id):
    try:
        if not query.strip():
            return chat_history, "", update_session_list()
        
        # Show user message immediately
        new_history = chat_history + [(query, None)]
        
        # Get bot response
        bot_history = qa_pipeline(query, session_id)
        
        return bot_history, "", update_session_list()
    except Exception as e:
        logger.error(f"Error in on_submit: {str(e)}")
        error_message = [(query, f"Xin lỗi, đã có lỗi xảy ra: {str(e)}")]
        history = error_message if not chat_history else chat_history + error_message
        return history, "", update_session_list()

# Gradio UI
with gr.Blocks(css=css) as demo:
    with gr.Row():
        gr.HTML("<div class='legalchat-title'>⚖️ LegalChat</div>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Lịch sử câu hỏi")
            session_list = gr.Radio(
                choices=[],
                label="",
                interactive=True,
                value=None,
                elem_classes="history-container"
            )
            new_chat_btn = gr.Button("💬 Tạo câu hỏi mới", variant="primary")
            avatar_urls = [
                "https://cdn-icons-png.flaticon.com/512/1253/1253756.png",  # Avatar người dùng
                "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"  # Avatar chatbot
            ]
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chatbot",
                height=500,
                bubble_full_width=False,
                avatar_images=avatar_urls,
                show_label=False,
                elem_classes="chat-container"
            )
            with gr.Row():
                input_box = gr.Textbox(
                    label="",
                    placeholder="Nhập câu hỏi pháp lý của bạn...",
                    lines=2,
                    scale=8
                )
                submit_btn = gr.Button("Gửi ➤", variant="primary", scale=1)
            current_session = gr.State(str(uuid.uuid4()))

    # CALLBACKS
    def on_select_session(selected_title):
        try:
            if selected_title:
                titles = get_session_titles()
                session_id = titles[selected_title]
                history, _ = load_session(session_id)
                return history, session_id
            return [], str(uuid.uuid4())
        except Exception as e:
            logger.error(f"Error in on_select_session: {str(e)}")
            return [], str(uuid.uuid4())

    def on_new_chat():
        try:
            new_id = str(uuid.uuid4())
            return [], new_id, update_session_list()
        except Exception as e:
            logger.error(f"Error in on_new_chat: {str(e)}")
            return [], str(uuid.uuid4()), gr.Radio(choices=[], value=None)

    # Event bindings
    new_chat_btn.click(on_new_chat, outputs=[chatbot, current_session, session_list])
    session_list.change(on_select_session, inputs=session_list, outputs=[chatbot, current_session])
    submit_btn.click(on_submit, inputs=[input_box, chatbot, current_session], outputs=[chatbot, input_box, session_list])
    input_box.submit(on_submit, inputs=[input_box, chatbot, current_session], outputs=[chatbot, input_box, session_list])

# Run app
if __name__ == "__main__":
    demo.launch(
        show_error=True,
        share=True,  # Enable temporary public URL
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7861,  # Sử dụng port khác
        max_threads=5,  # Increase max concurrent users
        auth=None,  # No authentication required
    )
