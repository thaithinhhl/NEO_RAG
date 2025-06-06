import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.llm import prompt_template
from src.retrieval.query import retrieve
from src.models.function_calling import process_query
from langchain_community.llms.ollama import Ollama
from src.utils.chat_history import message_history, get_history

app = FastAPI()
llm = Ollama(model="mistral:7b")

class QueryRequest(BaseModel):
    query: str
    session_id: str  

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        query = request.query
        session_id = request.session_id
        
        # Process the query
        function_result = process_query(query)
        context, scores, retrieval_time, total_tokens = retrieve(query)
        prompt = prompt_template(query, context, function_result)
        answer = llm.invoke(prompt)
        
        # Save to chat history
        message_history(session_id, "user", query)
        message_history(session_id, "assistant", answer)
        
        # Get full chat history
        history = get_history(session_id)
        
        return {
            "status": "success",
            "answer": answer,
            "context": context,
            "history": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )