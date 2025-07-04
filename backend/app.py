# backend/app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector import QdrantManager
from llm import MistralLLM
from collections import defaultdict
app = FastAPI()


chat_histories = defaultdict(list)  # session hist handling (context = 6 change in llm.py)


manager = QdrantManager(collection_name="docs") # Initialize vector database manager
llm = MistralLLM()

# Enable CORS to allow frontend (on different port) to access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
class ChatRequest(BaseModel):
    query: str
    username: str  

@app.post("/chat")
def chat(request: ChatRequest):
    user = request.username
    query = request.query

    # Apend user message
    chat_histories[user].append({"role": "user", "content": query})

    # Rag part context
    search_results = manager.search(query)

    # Generate assistant reply with full chat context
    full_prompt = llm.create_prompt_with_history(
        history=chat_histories[user],
        context=llm.format_context(search_results)
    )


    response = llm.call_api(full_prompt)


    chat_histories[user].append({"role": "assistant", "content": response})

    return {"response": response}

