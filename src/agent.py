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
<<<<<<< HEAD
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
=======
import re
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.tools import RAGTool, SQLTool, CalculatorTool, APITool

# Default Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # Changed from llama2 to mistral


def check_ollama_connection(base_url: str = OLLAMA_BASE_URL) -> dict:
    """Check if Ollama is running and return available models."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"connected": True, "models": models}
    except Exception as e:
        return {"connected": False, "error": str(e), "models": []}
    return {"connected": False, "models": []}


def get_available_model(preferred_model: str = DEFAULT_MODEL) -> str:
    """Get the best available model, falling back if preferred isn't available."""
    status = check_ollama_connection()
    if not status["connected"]:
        raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is Ollama running?")

    available = status["models"]
    if not available:
        raise ValueError("No models available in Ollama. Please pull a model first (e.g., 'ollama pull mistral')")

    # Check if preferred model is available (handle tags like 'mistral:latest')
    for model in available:
        if model.startswith(preferred_model) or preferred_model in model:
            return model.split(":")[0]  # Return base model name

    # Fall back to first available model
    print(f"⚠ Model '{preferred_model}' not found. Using '{available[0]}' instead.")
    return available[0].split(":")[0]


@dataclass
class ThoughtStep:
    """Represents a single step in the chain of thought."""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query."""
    query: str
    steps: List[ThoughtStep]
    final_answer: str
    total_confidence: float


class ChainOfThoughtReasoner:
    """Dedicated Chain-of-Thought reasoning engine."""

    def __init__(self, llm):
        self.llm = llm

    async def reason(self, query: str, context: str = "") -> ReasoningTrace:
        """Perform step-by-step reasoning on a query."""

        cot_prompt = f"""
You are an advanced reasoning agent. Think through this problem step by step.

Query: {query}
{f"Context: {context}" if context else ""}

Follow this EXACT format for your reasoning:

STEP 1: [Understand the question]
Thought: What is being asked? What are the key components?
Confidence: [0.0-1.0]

STEP 2: [Break down the problem]
Thought: What sub-problems need to be solved?
Confidence: [0.0-1.0]

STEP 3: [Identify information needed]
Thought: What information do I need to answer this?
Confidence: [0.0-1.0]

STEP 4: [Analyze and reason]
Thought: Based on available information, what conclusions can I draw?
Confidence: [0.0-1.0]

STEP 5: [Synthesize answer]
Thought: How do I combine my reasoning into a coherent answer?
Confidence: [0.0-1.0]

FINAL_ANSWER: [Your complete answer based on the reasoning above]
TOTAL_CONFIDENCE: [Average confidence 0.0-1.0]
"""

        response = await self.llm.ainvoke(cot_prompt)
        return self._parse_reasoning_trace(query, response)

    def _parse_reasoning_trace(self, query: str, response: str) -> ReasoningTrace:
        """Parse the LLM response into a structured ReasoningTrace."""
        steps = []

        # Parse each step
        step_pattern = r"STEP (\d+):.*?Thought: (.*?)(?:Confidence: ([\d.]+))?"
        matches = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_num = int(match[0]) if match[0] else len(steps) + 1
            thought = match[1].strip() if match[1] else ""
            confidence = float(match[2]) if match[2] else 0.5
            steps.append(ThoughtStep(
                step_number=step_num,
                thought=thought,
                confidence=min(max(confidence, 0.0), 1.0)
            ))

        # Parse final answer
        final_match = re.search(r"FINAL_ANSWER:\s*(.*?)(?:TOTAL_CONFIDENCE|$)", response, re.DOTALL | re.IGNORECASE)
        final_answer = final_match.group(1).strip() if final_match else response

        # Parse total confidence
        conf_match = re.search(r"TOTAL_CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        total_confidence = float(conf_match.group(1)) if conf_match else 0.5

        return ReasoningTrace(
            query=query,
            steps=steps if steps else [ThoughtStep(1, response, confidence=0.5)],
            final_answer=final_answer,
            total_confidence=min(max(total_confidence, 0.0), 1.0)
        )


class SpecializedAgent:
    """Agent specialized for specific tasks with CoT-enhanced prompts."""

    def __init__(self, llm, tools, memory, agent_type: str = "general"):
        self.agent_type = agent_type
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

    async def run(self, message: str, reasoning_context: str = "") -> str:
        """Run the agent with optional reasoning context."""
        enhanced_message = message
        if reasoning_context:
            enhanced_message = f"""
Based on the following reasoning:
{reasoning_context}

Please execute: {message}
"""
        return await self.agent.arun(input=enhanced_message)


class CriticAgent:
    """Enhanced critic that evaluates both response quality and reasoning."""

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, message: str, response: str, reasoning_trace: Optional[ReasoningTrace] = None) -> Dict:
        """Evaluate response with reasoning quality assessment."""

        reasoning_section = ""
        if reasoning_trace:
            steps_text = "\n".join([f"Step {s.step_number}: {s.thought} (confidence: {s.confidence})"
                                   for s in reasoning_trace.steps])
            reasoning_section = f"""
Reasoning Steps Used:
{steps_text}
Overall Reasoning Confidence: {reasoning_trace.total_confidence}
"""

        prompt = f"""
You are a critical evaluator. Assess the following response thoroughly.

Original Query: {message}
Generated Response: {response}
{reasoning_section}

Evaluate on these criteria (score 1-10 for each):

1. ACCURACY: Is the information factually correct?
2. RELEVANCE: Does it directly address the query?
3. COMPLETENESS: Are all aspects of the query covered?
4. REASONING_QUALITY: Is the logic sound and well-structured?
5. CLARITY: Is the response clear and understandable?

Provide your evaluation in this EXACT format:
ACCURACY: [score]
RELEVANCE: [score]
COMPLETENESS: [score]
REASONING_QUALITY: [score]
CLARITY: [score]
OVERALL_SCORE: [average score]
VERDICT: [ACCEPT if overall >= 7, otherwise IMPROVE]
FEEDBACK: [Specific improvements needed if VERDICT is IMPROVE]
"""

        result = self.llm.invoke(prompt)
        return self._parse_evaluation(result)

    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation response into structured format."""
        evaluation = {
            "accuracy": 5,
            "relevance": 5,
            "completeness": 5,
            "reasoning_quality": 5,
            "clarity": 5,
            "overall_score": 5,
            "verdict": "IMPROVE",
            "feedback": "",
            "raw_response": response
        }

        patterns = {
            "accuracy": r"ACCURACY:\s*(\d+)",
            "relevance": r"RELEVANCE:\s*(\d+)",
            "completeness": r"COMPLETENESS:\s*(\d+)",
            "reasoning_quality": r"REASONING_QUALITY:\s*(\d+)",
            "clarity": r"CLARITY:\s*(\d+)",
            "overall_score": r"OVERALL_SCORE:\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                evaluation[key] = float(match.group(1))

        verdict_match = re.search(r"VERDICT:\s*(ACCEPT|IMPROVE)", response, re.IGNORECASE)
        if verdict_match:
            evaluation["verdict"] = verdict_match.group(1).upper()

        feedback_match = re.search(r"FEEDBACK:\s*(.*?)(?:$)", response, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            evaluation["feedback"] = feedback_match.group(1).strip()

        return evaluation

class ManagerAgent:
    """Basic manager agent for backward compatibility."""

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 temperature: float = 0.7,
                 documents_path: str = "./data/documents",
                 db_path: str = "./data/database.db"):

        load_dotenv()

        # Verify Ollama connection and get available model
        actual_model = get_available_model(model_name)
        print(f"✓ Initializing with Ollama model: {actual_model}")

        self.llm = OllamaLLM(
            model=actual_model,
            temperature=temperature,
            base_url=OLLAMA_BASE_URL
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.retrieval_agent = SpecializedAgent(
            llm=self.llm,
            tools=[RAGTool(documents_path=documents_path)],
            memory=self.memory,
            agent_type="retrieval"
        )
        self.generative_agent = SpecializedAgent(
            llm=self.llm,
            tools=[SQLTool(db_path=db_path), CalculatorTool(), APITool()],
            memory=self.memory,
            agent_type="generative"
        )
        self.critic_agent = CriticAgent(llm=self.llm)
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc

        try:
<<<<<<< HEAD
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
        
=======
            decomposed_query = await self._decompose_query(message)

            retrieval_results = await self.retrieval_agent.run(decomposed_query["retrieval"])

            response = await self.generative_agent.run(
                decomposed_query["generative"].format(retrieved_info=retrieval_results)
            )

            for _ in range(3):
                evaluation = self.critic_agent.evaluate(message, response)
                if evaluation.get("verdict") == "ACCEPT" or "ACCEPT" in str(evaluation):
                    break

                response = await self.generative_agent.run(
                    f"Based on the feedback, improve your response: {evaluation.get('feedback', evaluation)}"
                )

            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"

    async def _decompose_query(self, message: str) -> dict:
        prompt = f"""
        Decompose the following query into two parts:
        1. A "retrieval" part for finding relevant information.
        2. A "generative" part for answering the question based on the retrieved information.

        Query: {message}
        """

        decomposed_query_str = await self.llm.ainvoke(prompt)

        try:
            retrieval_part = decomposed_query_str.split("generative:")[0].replace("retrieval:", "").strip()
            generative_part = decomposed_query_str.split("generative:")[1].strip()
        except IndexError:
            retrieval_part = message
            generative_part = "Answer based on: {retrieved_info}"

>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc
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


<<<<<<< HEAD
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
=======
class AdvancedAgenticRAG(ManagerAgent):
    """
    Advanced Agentic RAG with Chain-of-Thought reasoning.

    Features:
    - Step-by-step reasoning before actions
    - Confidence scoring for each reasoning step
    - Enhanced evaluation with reasoning quality metrics
    - Iterative refinement based on structured feedback
    """

    def __init__(self,
                 model_name: str = "llama2",
                 temperature: float = 0.7,
                 documents_path: str = "./data/documents",
                 db_path: str = "./data/database.db",
                 max_refinements: int = 3,
                 min_confidence_threshold: float = 0.6):

        super().__init__(model_name, temperature, documents_path, db_path)

        # Advanced components
        self.reasoner = ChainOfThoughtReasoner(self.llm)
        self.max_refinements = max_refinements
        self.min_confidence_threshold = min_confidence_threshold

        # Track reasoning history for the session
        self.reasoning_history: List[ReasoningTrace] = []

    async def chat(self, message: str) -> str:
        """
        Process a message with full Chain-of-Thought reasoning pipeline.

        Flow:
        1. Initial CoT reasoning on the query
        2. Decompose query based on reasoning
        3. Retrieve information with reasoning context
        4. Generate response with reasoning context
        5. Evaluate response and reasoning quality
        6. Refine if needed (up to max_refinements)
        """
        try:
            # Step 1: Initial Chain-of-Thought reasoning
            reasoning_trace = await self.reasoner.reason(message)
            self.reasoning_history.append(reasoning_trace)

            # Check if confidence is too low - might need clarification
            if reasoning_trace.total_confidence < self.min_confidence_threshold:
                clarification = await self._request_clarification(message, reasoning_trace)
                if clarification:
                    return clarification

            # Step 2: Decompose query with reasoning context
            decomposed_query = await self._decompose_query_with_reasoning(message, reasoning_trace)

            # Step 3: Retrieve information with reasoning context
            reasoning_context = self._format_reasoning_for_context(reasoning_trace)
            retrieval_results = await self.retrieval_agent.run(
                decomposed_query["retrieval"],
                reasoning_context=reasoning_context
            )

            # Step 4: Generate initial response
            generation_prompt = decomposed_query["generative"].format(
                retrieved_info=retrieval_results
            )
            response = await self.generative_agent.run(
                generation_prompt,
                reasoning_context=reasoning_context
            )

            # Step 5 & 6: Evaluate and refine
            response = await self._evaluate_and_refine(
                message, response, reasoning_trace, retrieval_results
            )

            return response

        except Exception as e:
            return f"Error in advanced reasoning pipeline: {str(e)}"

    async def _decompose_query_with_reasoning(self, message: str, reasoning_trace: ReasoningTrace) -> dict:
        """Decompose query using insights from CoT reasoning."""

        steps_summary = "\n".join([f"- {s.thought}" for s in reasoning_trace.steps[:3]])

        prompt = f"""
Based on the following step-by-step reasoning about the query:

{steps_summary}

Now decompose the original query into specific tasks:

Query: {message}

Think step by step:
1. What specific information needs to be RETRIEVED from documents/databases?
2. What GENERATIVE task needs to be performed with that information?

Respond in this EXACT format:
RETRIEVAL_TASK: [Specific search/query to retrieve needed information]
GENERATIVE_TASK: [Task to perform using retrieved info, use {{retrieved_info}} as placeholder]
"""

        response = await self.llm.ainvoke(prompt)

        # Parse the response
        retrieval_match = re.search(r"RETRIEVAL_TASK:\s*(.*?)(?:GENERATIVE_TASK|$)", response, re.DOTALL | re.IGNORECASE)
        generative_match = re.search(r"GENERATIVE_TASK:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)

        retrieval_part = retrieval_match.group(1).strip() if retrieval_match else message
        generative_part = generative_match.group(1).strip() if generative_match else "Answer based on: {retrieved_info}"

        return {
            "retrieval": retrieval_part,
            "generative": generative_part
        }

    async def _evaluate_and_refine(self, message: str, response: str,
                                   reasoning_trace: ReasoningTrace,
                                   retrieval_results: str) -> str:
        """Evaluate response and refine using structured feedback."""

        for iteration in range(self.max_refinements):
            # Evaluate with reasoning context
            evaluation = self.critic_agent.evaluate(message, response, reasoning_trace)

            # Check if response is acceptable
            if evaluation["verdict"] == "ACCEPT" or evaluation["overall_score"] >= 7:
                break

            # Generate refined response based on structured feedback
            refinement_prompt = f"""
Your previous response needs improvement.

Original Query: {message}
Your Previous Response: {response}

Evaluation Scores:
- Accuracy: {evaluation['accuracy']}/10
- Relevance: {evaluation['relevance']}/10
- Completeness: {evaluation['completeness']}/10
- Reasoning Quality: {evaluation['reasoning_quality']}/10
- Clarity: {evaluation['clarity']}/10

Specific Feedback: {evaluation['feedback']}

Retrieved Information Available: {retrieval_results[:500]}...

Please provide an improved response that addresses the feedback.
Think step by step about how to improve each weak area.
"""

            response = await self.generative_agent.run(refinement_prompt)

        return response

    async def _request_clarification(self, message: str, reasoning_trace: ReasoningTrace) -> Optional[str]:
        """Request clarification if confidence is too low."""

        low_confidence_steps = [s for s in reasoning_trace.steps if s.confidence < 0.5]

        if not low_confidence_steps:
            return None

        unclear_aspects = "\n".join([f"- {s.thought}" for s in low_confidence_steps])

        clarification_prompt = f"""
I want to give you the best possible answer, but I'm uncertain about some aspects of your question.

Your question: {message}

I'm uncertain about:
{unclear_aspects}

Could you please clarify or provide more context about what you're looking for?
"""
        return clarification_prompt

    def _format_reasoning_for_context(self, reasoning_trace: ReasoningTrace) -> str:
        """Format reasoning trace as context for other agents."""

        steps_text = "\n".join([
            f"Step {s.step_number}: {s.thought}"
            for s in reasoning_trace.steps
        ])

        return f"""
Reasoning Analysis:
{steps_text}

Key Insight: {reasoning_trace.final_answer[:200] if reasoning_trace.final_answer else 'N/A'}
Confidence: {reasoning_trace.total_confidence:.2f}
"""

    def get_reasoning_history(self) -> List[ReasoningTrace]:
        """Get the full reasoning history for the session."""
        return self.reasoning_history

    def clear_reasoning_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []


# Backward compatibility alias
class AgenticRAG(AdvancedAgenticRAG):
    """Alias for AdvancedAgenticRAG for backward compatibility."""
    pass
>>>>>>> fd6723c4cf0be401d45adfc5dda51fc42e927cfc
