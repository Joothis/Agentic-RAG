from typing import Any, Optional
import sqlite3
from langchain.tools import BaseTool

class SQLTool(BaseTool):
    name: str = "sql_query"
    description: str = "Execute SQL queries on a SQLite database"
    db_path: Optional[str] = None
    connection: Optional[sqlite3.Connection] = None

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize the database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
    
    def _run(self, query: str) -> str:
        """Execute an SQL query and return the results"""
        if not self.connection:
            return "Database connection not initialized"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            
            # Format results as a readable string
            formatted_results = []
            formatted_results.append(" | ".join(column_names))
            formatted_results.append("-" * len(formatted_results[0]))
            
            for row in results:
                formatted_results.append(" | ".join(str(item) for item in row))
            
            return "\n".join(formatted_results)
            
        except sqlite3.Error as e:
            return f"Error executing query: {e}"

    async def _arun(self, query: str) -> Any:
        """Async implementation of run"""
        raise NotImplementedError("SQLTool does not support async")
