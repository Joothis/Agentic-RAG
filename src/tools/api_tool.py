from typing import Dict, Any, ClassVar, Optional
from langchain.tools import BaseTool
import requests

class APITool(BaseTool):
    def __init__(self, base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        super().__init__()
        self.name = "api_call"
        self.description = "Make HTTP requests to external APIs"
        self.base_url = base_url
        self.headers = headers or {}
        self.session = requests.Session()
        
        if headers:
            self.session.headers.update(headers)
    
    def _run(self, params: str) -> str:
        """Make an API call with the given parameters
        
        params should be a string in the format:
        "method|endpoint|body"
        where body is optional JSON string
        """
        try:
            # Parse parameters
            parts = params.split('|')
            if len(parts) < 2:
                return "Invalid parameters. Format: method|endpoint|body"
            
            method = parts[0].upper()
            endpoint = parts[1]
            body = parts[2] if len(parts) > 2 else None
            
            # Construct URL
            url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint
            
            # Make request
            response = self.session.request(
                method=method,
                url=url,
                json=body if body else None
            )
            
            # Return response
            return response.text
            
        except Exception as e:
            return f"Error making API call: {str(e)}"

    async def _arun(self, params: str) -> str:
        """Async implementation of run"""
        raise NotImplementedError("APITool does not support async")
