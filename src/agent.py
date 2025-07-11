import asyncio
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

from tools import RAGTool, SQLTool, CalculatorTool, APITool

class SpecializedAgent:
    def __init__(self, llm, tools, memory):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )

    async def run(self, message: str) -> str:
        return await self.agent.arun(input=message)

class CriticAgent:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, message: str, response: str) -> str:
        prompt = f"""
        Original query: {message}
        Generated response: {response}

        Please evaluate the response for accuracy, relevance, and completeness.
        If the response is good, say "ACCEPT".
        Otherwise, provide feedback on how to improve it.
        """
        return self.llm.invoke(prompt)

class ManagerAgent:
    def __init__(self, 
                 model_name: str = "llama2",
                 temperature: float = 0.7,
                 documents_path: str = "./data/documents",
                 db_path: str = "./data/database.db"):
        
        load_dotenv()
        
        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.retrieval_agent = SpecializedAgent(
            llm=self.llm,
            tools=[RAGTool(documents_path=documents_path)],
            memory=self.memory
        )
        self.generative_agent = SpecializedAgent(
            llm=self.llm,
            tools=[SQLTool(db_path=db_path), CalculatorTool(), APITool()],
            memory=self.memory
        )
        self.critic_agent = CriticAgent(llm=self.llm)

    async def chat(self, message: str) -> str:
        try:
            decomposed_query = await self._decompose_query(message)
            
            retrieval_results = await self.retrieval_agent.run(decomposed_query["retrieval"])
            
            response = await self.generative_agent.run(decomposed_query["generative"].format(retrieved_info=retrieval_results))
            
            for _ in range(3):  # Allow for up to 3 rounds of refinement
                evaluation = self.critic_agent.evaluate(message, response)
                if "ACCEPT" in evaluation:
                    break
                
                response = await self.generative_agent.run(
                    f"Based on the retrieved information and the following feedback, please provide a new response: {evaluation}"
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
        
        # This is a simple parsing method. A more robust solution would use structured output.
        retrieval_part = decomposed_query_str.split("generative:")[0].replace("retrieval:", "").strip()
        generative_part = decomposed_query_str.split("generative:")[1].strip()

        return {
            "retrieval": retrieval_part,
            "generative": generative_part
        }

class AgenticRAG(ManagerAgent):
    pass