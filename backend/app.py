# fast api code ( main backend )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from vector import QdrantManager
from llm import MistralLLM
from collections import defaultdict
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

# session hist handling (context = 6 change in llm.py)
chat_histories = defaultdict(list) 

# Initialize vector database manager
manager = QdrantManager(collection_name="docs") 
llm = MistralLLM()

# Defining input model
class ChatRequest(BaseModel):
    query: str
    username: str

# Defining response model
class ChatResponse(BaseModel):
    response: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
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
        
        # response from mistral 
        response = llm.call_api(full_prompt)

        # Append assistant response to history
        chat_histories[user].append({"role": "assistant", "content": response})

        # Keep only last 20 messages to prevent memory issues
        if len(chat_histories[user]) > 20:
            chat_histories[user] = chat_histories[user][-20:]

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


