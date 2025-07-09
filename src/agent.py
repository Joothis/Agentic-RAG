from typing import List
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

from tools import RAGTool, SQLTool, CalculatorTool, APITool

class AgenticRAG:
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 documents_path: str = "./data/documents",
                 db_path: str = "./data/database.db"):
        
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature
        )
        
        # Initialize tools
        self.tools = [
            RAGTool(documents_path=documents_path),
            SQLTool(db_path=db_path),
            CalculatorTool(),
            APITool()
        ]
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, message: str) -> str:
        """Process a chat message and return the agent's response"""
        try:
            response = self.agent.run(input=message)
            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"
