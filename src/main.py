from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os

from src.agent import AgenticRAG, AdvancedAgenticRAG, check_ollama_connection, DEFAULT_MODEL

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Agentic RAG Chatbot",
    description="A multi-agent RAG system with Chain-of-Thought reasoning",
    version="2.0.0"
)

# Define request/response models
class ChatMessage(BaseModel):
    message: str
    include_reasoning: Optional[bool] = False


class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    confidence: float


class ChatResponse(BaseModel):
    response: str
    reasoning_steps: Optional[List[ReasoningStep]] = None
    total_confidence: Optional[float] = None
    status: str = "success"


# Mount the static directory to serve index.html
static_path = os.path.join(os.path.dirname(__file__), "..")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize the Agentic RAG system
agent: Optional[AdvancedAgenticRAG] = None

# Check Ollama connection first
ollama_status = check_ollama_connection()
if ollama_status["connected"]:
    print(f"✓ Ollama connected. Available models: {', '.join(ollama_status['models'])}")
else:
    print(f"✗ Ollama not connected: {ollama_status.get('error', 'Unknown error')}")

try:
    agent = AgenticRAG(
        model_name=DEFAULT_MODEL,  # Use configured default (mistral)
        temperature=0.7,
        documents_path="./data/documents",
        db_path="./data/database.db",
        max_refinements=3,
        min_confidence_threshold=0.6
    )
    print("✓ AdvancedAgenticRAG initialized with Chain-of-Thought reasoning")
except Exception as e:
    print(f"✗ Error initializing AgenticRAG: {str(e)}")
    agent = None


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface."""
    with open(os.path.join(static_path, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Process a chat message with Chain-of-Thought reasoning.

    - **message**: The user's message
    - **include_reasoning**: If true, includes reasoning steps in response
    """
    if not agent:
        return ChatResponse(
            response="Error: Agent not initialized. Please check if Ollama is running.",
            status="error"
        )

    try:
        response = await agent.chat(chat_message.message)

        # Optionally include reasoning trace
        reasoning_steps = None
        total_confidence = None

        if chat_message.include_reasoning and agent.reasoning_history:
            latest_trace = agent.reasoning_history[-1]
            reasoning_steps = [
                ReasoningStep(
                    step_number=step.step_number,
                    thought=step.thought,
                    confidence=step.confidence
                )
                for step in latest_trace.steps
            ]
            total_confidence = latest_trace.total_confidence

        return ChatResponse(
            response=response,
            reasoning_steps=reasoning_steps,
            total_confidence=total_confidence,
            status="success"
        )
    except Exception as e:
        return ChatResponse(
            response=f"Error processing your message: {str(e)}",
            status="error"
        )


@app.get("/reasoning-history")
async def get_reasoning_history():
    """Get the reasoning history for the current session."""
    if not agent:
        return {"error": "Agent not initialized", "history": []}

    history = []
    for trace in agent.get_reasoning_history():
        history.append({
            "query": trace.query,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "confidence": s.confidence
                }
                for s in trace.steps
            ],
            "final_answer": trace.final_answer[:200] if trace.final_answer else None,
            "total_confidence": trace.total_confidence
        })

    return {"history": history}


@app.post("/clear-history")
async def clear_history():
    """Clear the reasoning history."""
    if agent:
        agent.clear_reasoning_history()
        return {"status": "success", "message": "Reasoning history cleared"}
    return {"status": "error", "message": "Agent not initialized"}


@app.get("/health")
async def health_check():
    """Health check endpoint with Ollama status."""
    ollama_status = check_ollama_connection()
    return {
        "status": "healthy" if agent and ollama_status["connected"] else "degraded",
        "agent_initialized": agent is not None,
        "ollama": {
            "connected": ollama_status["connected"],
            "models": ollama_status.get("models", []),
            "error": ollama_status.get("error")
        },
        "version": "2.0.0",
        "features": ["chain-of-thought", "multi-agent", "iterative-refinement"]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)