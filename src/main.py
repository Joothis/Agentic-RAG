from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os

from agent import AgenticRAG

# Initialize FastAPI app
app = FastAPI()

# Define the request body model
class ChatMessage(BaseModel):
    message: str

# Mount the static directory to serve index.html
static_path = os.path.join(os.path.dirname(__file__), "..")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize the Agentic RAG system
try:
    agent = AgenticRAG(
        model_name="llama2",
        temperature=0.7,
        documents_path="./data/documents",
        db_path="./data/database.db"
    )
except Exception as e:
    print(f"Error initializing AgenticRAG: {str(e)}")
    agent = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open(os.path.join(static_path, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    if not agent:
        return {"response": "Error: Agent not initialized."}
    
    try:
        response = await agent.chat(chat_message.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)