"""
MCP (Model Context Protocol) Server Implementation
Provides standardized tool interface for AI agents
"""

import json
import asyncio
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types"""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOL_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCE_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPT_GET = "prompts/get"
    ERROR = "error"
    RESPONSE = "response"


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


@dataclass
class MCPPrompt:
    """MCP Prompt template"""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    template: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


class MCPServer:
    """
    Model Context Protocol Server
    Provides standardized interface for AI tool orchestration
    """
    
    def __init__(self, name: str = "agentic-rag-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.initialized = False
        
    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the MCP server"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource with the MCP server"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.name}")
    
    def register_prompt(self, prompt: MCPPrompt) -> None:
        """Register a prompt template with the MCP server"""
        self.prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message"""
        msg_type = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})
        
        try:
            if msg_type == MCPMessageType.INITIALIZE.value:
                return self._handle_initialize(msg_id)
            
            elif msg_type == MCPMessageType.TOOLS_LIST.value:
                return self._handle_tools_list(msg_id)
            
            elif msg_type == MCPMessageType.TOOL_CALL.value:
                return await self._handle_tool_call(msg_id, params)
            
            elif msg_type == MCPMessageType.RESOURCES_LIST.value:
                return self._handle_resources_list(msg_id)
            
            elif msg_type == MCPMessageType.RESOURCE_READ.value:
                return await self._handle_resource_read(msg_id, params)
            
            elif msg_type == MCPMessageType.PROMPTS_LIST.value:
                return self._handle_prompts_list(msg_id)
            
            elif msg_type == MCPMessageType.PROMPT_GET.value:
                return self._handle_prompt_get(msg_id, params)
            
            else:
                return self._error_response(msg_id, f"Unknown method: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self._error_response(msg_id, str(e))
    
    def _handle_initialize(self, msg_id: Any) -> Dict[str, Any]:
        """Handle initialization request"""
        self.initialized = True
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                },
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                    "prompts": {"listChanged": True}
                }
            }
        }
    
    def _handle_tools_list(self, msg_id: Any) -> Dict[str, Any]:
        """Return list of available tools"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [tool.to_dict() for tool in self.tools.values()]
            }
        }
    
    async def _handle_tool_call(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self._error_response(msg_id, f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        if tool.handler is None:
            return self._error_response(msg_id, f"Tool {tool_name} has no handler")
        
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result) if not isinstance(result, str) else result
                        }
                    ]
                }
            }
        except Exception as e:
            return self._error_response(msg_id, f"Tool execution error: {str(e)}")
    
    def _handle_resources_list(self, msg_id: Any) -> Dict[str, Any]:
        """Return list of available resources"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "resources": [res.to_dict() for res in self.resources.values()]
            }
        }
    
    async def _handle_resource_read(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a resource"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            return self._error_response(msg_id, f"Unknown resource: {uri}")
        
        resource = self.resources[uri]
        
        # For now, return placeholder - can be extended
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": f"Resource content for {resource.name}"
                    }
                ]
            }
        }
    
    def _handle_prompts_list(self, msg_id: Any) -> Dict[str, Any]:
        """Return list of available prompts"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "prompts": [p.to_dict() for p in self.prompts.values()]
            }
        }
    
    def _handle_prompt_get(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific prompt with arguments"""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return self._error_response(msg_id, f"Unknown prompt: {prompt_name}")
        
        prompt = self.prompts[prompt_name]
        
        # Render template with arguments
        rendered = prompt.template
        for key, value in arguments.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "description": prompt.description,
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": rendered
                        }
                    }
                ]
            }
        }
    
    def _error_response(self, msg_id: Any, message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }


def create_mcp_server() -> MCPServer:
    """Create and configure the MCP server with default tools"""
    server = MCPServer()
    
    # Register default prompts
    server.register_prompt(MCPPrompt(
        name="research_assistant",
        description="Research assistant prompt for comprehensive analysis",
        arguments=[
            {"name": "topic", "description": "The topic to research", "required": True},
            {"name": "depth", "description": "Analysis depth (basic/detailed/comprehensive)", "required": False}
        ],
        template="""You are an expert research assistant. Analyze the following topic thoroughly:

Topic: {topic}
Depth: {depth}

Provide:
1. Key facts and background
2. Current state and trends
3. Analysis and insights
4. Recommendations

Be thorough, cite sources when available, and provide actionable insights."""
    ))
    
    server.register_prompt(MCPPrompt(
        name="data_analyst",
        description="Data analysis prompt for extracting insights from data",
        arguments=[
            {"name": "data_description", "description": "Description of the data to analyze", "required": True},
            {"name": "questions", "description": "Specific questions to answer", "required": False}
        ],
        template="""You are a data analyst. Analyze the following data:

Data: {data_description}
Questions: {questions}

Provide:
1. Data summary and statistics
2. Key patterns and trends
3. Anomalies or outliers
4. Actionable insights
5. Visualizations suggestions"""
    ))
    
    # Register resources
    server.register_resource(MCPResource(
        uri="rag://documents",
        name="Document Store",
        description="Access to indexed documents for RAG search",
        mime_type="application/json"
    ))
    
    server.register_resource(MCPResource(
        uri="rag://insights",
        name="Usage Insights",
        description="Analytics and insights from system usage",
        mime_type="application/json"
    ))
    
    return server
