from typing import Any, List
from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

class RAGTool(BaseTool):
    name: str = "rag_search"
    description: str = "Search through documents using vector similarity search"
    text_splitter: RecursiveCharacterTextSplitter = None
    embeddings: HuggingFaceEmbeddings = None
    vector_store: Chroma = None
    k: int = 3

    def __init__(self, documents_path: str, k: int = 3):
        super().__init__()
        self.k = k
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            self.initialize_vector_store(documents_path)
        except Exception as e:
            print(f"Error during vector store initialization: {e}")
            raise
    
    def _load_documents_from_directory(self, directory_path: str) -> List[Document]:
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(Document(page_content=content, metadata={"source": filepath}))
        return documents

    def initialize_vector_store(self, documents_path: str):
        """Initialize the vector store with documents from the given path"""
        # Load documents from the specified directory
        documents = self._load_documents_from_directory(documents_path)

        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)

        # Create and persist the vector store
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./data/vector_store"
        )

    def _run(self, query: str) -> str:
        """Run vector similarity search on the query"""
        if not self.vector_store:
            return "Vector store not initialized"
        
        try:
            results = self.vector_store.similarity_search(query, k=self.k)
            return "\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"Error during similarity search: {e}"

    async def _arun(self, query: str) -> Any:
        """Async implementation of run"""
        raise NotImplementedError("RAGTool does not support async")
