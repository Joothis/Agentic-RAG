"""
Agentic RAG API Server with MCP Integration
Provides REST API for intelligent document analysis and insights
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
<<<<<<< HEAD
from typing import Optional, Dict, Any
=======
from typing import Optional, List, Dict, Any
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc
import uvicorn
import os
import asyncio

<<<<<<< HEAD
from agent import AgenticRAG
from mcp_server import create_mcp_server, MCPTool

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG System",
    description="AI-powered document analysis with meaningful insights",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatMessage(BaseModel):
    message: str
    with_critique: Optional[bool] = True
=======
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

>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc


class MCPMessage(BaseModel):
    method: str
    id: Optional[int] = 1
    params: Optional[Dict[str, Any]] = {}


class InsightsRequest(BaseModel):
    detailed: Optional[bool] = False


# Mount static directory
static_path = os.path.join(os.path.dirname(__file__), "..")
app.mount("/static", StaticFiles(directory=static_path), name="static")

<<<<<<< HEAD
# Global instances
agent: Optional[AgenticRAG] = None
mcp_server = None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent and MCP server on startup"""
    global agent, mcp_server
    
    # Initialize MCP Server
    mcp_server = create_mcp_server()
    
    # Initialize Agent
    try:
        # Create data directories if they don't exist
        os.makedirs("./data/documents", exist_ok=True)
        os.makedirs("./data/vector_store", exist_ok=True)
        
        agent = AgenticRAG(
            documents_path="./data/documents",
            db_path="./data/database.db"
        )
        
        # Register agent tools with MCP server
        for tool_info in agent.get_available_tools():
            mcp_server.register_tool(MCPTool(
                name=tool_info["name"],
                description=tool_info["description"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query or input for the tool"}
                    },
                    "required": ["query"]
                }
            ))
        
        print("✅ Agentic RAG System initialized successfully!")
        print(f"📊 Available tools: {[t['name'] for t in agent.get_available_tools()]}")
        
    except Exception as e:
        print(f"⚠️ Warning: Agent initialization issue: {str(e)}")
        print("The system will start but some features may be limited.")

=======
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
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
<<<<<<< HEAD
    """Serve the main web interface"""
    try:
        with open(os.path.join(static_path, "index.html")) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Agentic RAG System</h1><p>Web interface not found. API is running.</p>",
            status_code=200
        )

=======
    """Serve the main chat interface."""
    with open(os.path.join(static_path, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
<<<<<<< HEAD
    """Process chat messages and return AI responses"""
    if not agent:
        return JSONResponse(
            status_code=503,
            content={"response": "Agent not initialized. Please check server logs."}
        )
    
    try:
        response = await agent.chat(
            chat_message.message,
            with_critique=chat_message.with_critique
        )
        
        # Include evaluation info if available
        evaluation = agent.get_last_evaluation()
        
        return {
            "response": response,
            "evaluation": {
                "quality_score": evaluation.get("quality_score"),
                "key_insights": evaluation.get("key_insights", [])
            } if evaluation else None
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"response": f"Error processing request: {str(e)}"}
        )


@app.get("/insights")
async def get_insights():
    """Get meaningful insights from usage patterns"""
    if not agent:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent not initialized"}
        )
    
    try:
        insights = await agent.get_insights()
        return JSONResponse(content=insights)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/tools")
async def list_tools():
    """List all available tools (MCP-style)"""
    if not agent:
        return {"tools": []}
    
    return {"tools": agent.get_available_tools()}


@app.post("/mcp")
async def mcp_endpoint(message: MCPMessage):
    """MCP (Model Context Protocol) endpoint for tool orchestration"""
    if not mcp_server:
        return JSONResponse(
            status_code=503,
            content={"error": "MCP server not initialized"}
        )
    
    try:
        response = await mcp_server.handle_message({
            "method": message.method,
            "id": message.id,
            "params": message.params
        })
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
=======
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
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc


@app.get("/health")
async def health_check():
<<<<<<< HEAD
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "mcp_initialized": mcp_server is not None,
        "tools_available": len(agent.get_available_tools()) if agent else 0
    }


@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "name": "Agentic RAG System",
        "version": "2.0.0",
        "features": [
            "OpenRouter LLM Integration (Free Models)",
            "MCP (Model Context Protocol) Support",
            "Multi-Agent Architecture with Critic",
            "RAG Document Search",
            "SQL Database Queries",
            "Mathematical Calculations",
            "External API Calls",
            "Usage Analytics & Insights"
        ],
        "endpoints": {
            "chat": "POST /chat - Send messages to the AI",
            "insights": "GET /insights - Get usage analytics",
            "tools": "GET /tools - List available tools",
            "mcp": "POST /mcp - MCP protocol endpoint",
            "health": "GET /health - Health check"
        }
=======
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
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc
    }


if __name__ == "__main__":
    print("🚀 Starting Agentic RAG System...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )