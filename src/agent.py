"""
Agentic RAG System with OpenRouter LLM and MCP Integration
Provides intelligent document analysis with meaningful insights
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from dotenv import load_dotenv
import os
from datetime import datetime

from tools import RAGTool, SQLTool, CalculatorTool, APITool
from tools.insights_tool import InsightsTool

# Load environment variables
load_dotenv()


class OpenRouterLLM:
    """Wrapper for OpenRouter API using LangChain's ChatOpenAI"""
    
    @staticmethod
    def create(
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> ChatOpenAI:
        """Create an OpenRouter-compatible LLM instance"""
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = model_name or os.getenv("MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Agentic RAG System"
            }
        )


class InsightGenerator:
    """Generates meaningful insights from RAG queries and responses"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.query_history: List[Dict[str, Any]] = []
        self.insights_cache: Dict[str, Any] = {}
    
    def log_interaction(self, query: str, response: str, sources: List[str] = None):
        """Log an interaction for insight generation"""
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources": sources or [],
            "response_length": len(response)
        })
    
    async def generate_insights(self) -> Dict[str, Any]:
        """Generate meaningful insights from query history"""
        if not self.query_history:
            return {"message": "No interactions logged yet"}
        
        # Analyze patterns
        total_queries = len(self.query_history)
        avg_response_length = sum(q["response_length"] for q in self.query_history) / total_queries
        
        # Get topic analysis from LLM
        recent_queries = [q["query"] for q in self.query_history[-10:]]
        
        prompt = f"""Analyze these recent user queries and provide meaningful insights:

Queries:
{json.dumps(recent_queries, indent=2)}

Provide insights in JSON format with these keys:
- main_topics: List of main topics users are asking about
- knowledge_gaps: Areas where users seem to need more information
- trends: Emerging patterns in user questions
- recommendations: Suggestions for improving the knowledge base
- sentiment: Overall sentiment of user queries (curious, frustrated, satisfied, etc.)

Return ONLY valid JSON."""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            llm_insights = json.loads(content)
        except Exception as e:
            llm_insights = {
                "main_topics": ["Unable to analyze"],
                "knowledge_gaps": [],
                "trends": [],
                "recommendations": [],
                "sentiment": "unknown",
                "error": str(e)
            }
        
        return {
            "statistics": {
                "total_queries": total_queries,
                "average_response_length": round(avg_response_length, 2),
                "unique_sources_used": len(set(
                    src for q in self.query_history for src in q.get("sources", [])
                ))
            },
            "analysis": llm_insights,
            "recent_activity": self.query_history[-5:] if len(self.query_history) >= 5 else self.query_history
        }


class CriticAgent:
    """Evaluates and provides feedback on generated responses"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def evaluate(self, query: str, response: str, context: str = "") -> Dict[str, Any]:
        """Evaluate response quality and provide structured feedback"""
        
        prompt = f"""You are an expert evaluator. Analyze this response for quality.

Original Query: {query}

Context Used: {context[:1000] if context else "No context provided"}

Generated Response: {response}

Evaluate and respond in JSON format:
{{
    "verdict": "ACCEPT" or "IMPROVE",
    "quality_score": 1-10,
    "accuracy": "high/medium/low",
    "relevance": "high/medium/low", 
    "completeness": "high/medium/low",
    "feedback": "specific improvement suggestions if needed",
    "key_insights": ["list of key insights from the response"]
}}

Return ONLY valid JSON."""

        try:
            result = await self.llm.ainvoke(prompt)
            content = result.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return json.loads(content)
        except Exception as e:
            return {
                "verdict": "ACCEPT",
                "quality_score": 7,
                "accuracy": "medium",
                "relevance": "medium",
                "completeness": "medium",
                "feedback": f"Evaluation error: {str(e)}",
                "key_insights": []
            }


class AgenticRAG:
    """
    Advanced Agentic RAG System with:
    - OpenRouter LLM integration (free models)
    - MCP-style tool orchestration
    - Meaningful insights generation
    - Multi-agent architecture with critic feedback
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        documents_path: str = "./data/documents",
        db_path: str = "./data/database.db"
    ):
        load_dotenv()
        
        # Initialize LLM via OpenRouter
        self.llm = OpenRouterLLM.create(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize memory with window to prevent context overflow
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 interactions
        )
        
        # Initialize tools (MCP-style)
        self.tools = self._initialize_tools(documents_path, db_path)
        
        # Initialize agents
        self.agent = self._create_agent()
        self.critic = CriticAgent(self.llm)
        self.insight_generator = InsightGenerator(self.llm)
        
        # Tracking
        self.last_sources: List[str] = []
        self.last_evaluation: Dict[str, Any] = {}
    
    def _initialize_tools(self, documents_path: str, db_path: str) -> List[BaseTool]:
        """Initialize all available tools with MCP-style configuration"""
        tools = []
        
        # RAG Tool for document search
        try:
            if os.path.exists(documents_path):
                tools.append(RAGTool(documents_path=documents_path))
        except Exception as e:
            print(f"Warning: RAG tool initialization failed: {e}")
        
        # SQL Tool for database queries
        try:
            if os.path.exists(db_path):
                tools.append(SQLTool(db_path=db_path))
        except Exception as e:
            print(f"Warning: SQL tool initialization failed: {e}")
        
        # Calculator for math operations
        tools.append(CalculatorTool())
        
        # API Tool for external calls
        tools.append(APITool())
        
        # Insights Tool for analytics
        tools.append(InsightsTool())
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the main ReAct agent with tools"""
        
        # ReAct prompt template
        template = """You are an intelligent research assistant with access to various tools.
Your goal is to provide accurate, insightful, and well-reasoned answers.

You have access to these tools:
{tools}

Tool names: {tool_names}

Use this format:

Question: the input question you must answer
Thought: think step by step about what you need to do
Action: the tool to use (one of [{tool_names}])
Action Input: the input to the tool
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: provide a comprehensive, insightful answer

Remember to:
1. Break down complex questions into smaller parts
2. Use multiple tools when needed for comprehensive answers
3. Synthesize information from different sources
4. Provide actionable insights, not just raw data
5. Cite sources when available

Chat History:
{chat_history}

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def chat(self, message: str, with_critique: bool = True) -> str:
        """Process a chat message and return response with insights"""
        try:
            # Get agent response
            result = await self.agent.ainvoke({"input": message})
            response = result.get("output", "I couldn't process your request.")
            
            # Optionally critique and refine
            if with_critique:
                evaluation = await self.critic.evaluate(message, response)
                self.last_evaluation = evaluation
                
                # If response needs improvement, try once more
                if evaluation.get("verdict") == "IMPROVE" and evaluation.get("quality_score", 10) < 6:
                    feedback = evaluation.get("feedback", "Please provide a more complete answer.")
                    refined_result = await self.agent.ainvoke({
                        "input": f"Previous response was insufficient. Feedback: {feedback}. Original question: {message}"
                    })
                    response = refined_result.get("output", response)
            
            # Log for insights
            self.insight_generator.log_interaction(
                query=message,
                response=response,
                sources=self.last_sources
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def get_insights(self) -> Dict[str, Any]:
        """Get meaningful insights from usage patterns"""
        return await self.insight_generator.generate_insights()
    
    def get_last_evaluation(self) -> Dict[str, Any]:
        """Get the last response evaluation"""
        return self.last_evaluation
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools (MCP-style)"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]


# Alias for backward compatibility
ManagerAgent = AgenticRAG