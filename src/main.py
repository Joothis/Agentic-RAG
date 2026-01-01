"""
Agentic RAG API Server with MCP Integration
Provides REST API for intelligent document analysis and insights
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
import asyncio

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


class MCPMessage(BaseModel):
    method: str
    id: Optional[int] = 1
    params: Optional[Dict[str, Any]] = {}


class InsightsRequest(BaseModel):
    detailed: Optional[bool] = False


# Mount static directory
static_path = os.path.join(os.path.dirname(__file__), "..")
app.mount("/static", StaticFiles(directory=static_path), name="static")

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


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    try:
        with open(os.path.join(static_path, "index.html")) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Agentic RAG System</h1><p>Web interface not found. API is running.</p>",
            status_code=200
        )


@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
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


@app.get("/health")
async def health_check():
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